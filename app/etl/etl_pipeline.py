"""
ETL Pipeline module that integrates with DANGlocal_etl_run.py
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import io
from datetime import datetime

# Add the parent directory to the path to import DANGlocal_etl_run
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'DORC_PROCESS_OPTIMIZER'))

try:
    from DANGlocal_etl_run import run_etl
except ImportError:
    print("Warning: Could not import DANGlocal_etl_run. ETL functionality may be limited.")
    run_etl = None

from config import TARGET_COLS, TIME_INDEX_COL, STATIC_PREFIX, FUTURE_PREFIX


def process_uploaded_files(uploaded_files: List[Tuple[str, bytes]], temp_dir: str = "temp_uploads") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Process uploaded files using the existing ETL pipeline.
    
    Args:
        uploaded_files: List of tuples containing (filename, file_bytes)
        temp_dir: Temporary directory to save files for processing
        
    Returns:
        Tuple of (targets_df, statics_df, futures_df) or (None, None, None) if failed
    """
    if not uploaded_files:
        return None, None, None
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # For now, we'll process the first Excel file found
        # In a production system, you might want to handle multiple files differently
        excel_file = None
        for filename, file_bytes in uploaded_files:
            if filename.lower().endswith(('.xlsx', '.xls')):
                excel_file = (filename, file_bytes)
                break
        
        if not excel_file:
            print("No Excel file found in uploaded files")
            return None, None, None
        
        filename, file_bytes = excel_file
        
        # Save the file temporarily
        temp_file_path = os.path.join(temp_dir, filename)
        with open(temp_file_path, 'wb') as f:
            f.write(file_bytes)
        
        # Run the ETL pipeline
        if run_etl is None:
            print("ETL function not available")
            return None, None, None
        
        targets, statics, futures = run_etl(temp_file_path, temp_dir)
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return targets, statics, futures
        
    except Exception as e:
        print(f"Error processing files: {e}")
        return None, None, None


def unify_csvs(uploaded_files: List[Tuple[str, bytes]]) -> pd.DataFrame:
    """
    Fallback ETL function for CSV files (from original autoformer_ebm_web_app_aws_ready.py)
    """
    dfs = []
    for name, fb in uploaded_files:
        try:
            df = pd.read_csv(io.BytesIO(fb))
            # ensure date column
            if TIME_INDEX_COL in df.columns:
                df[TIME_INDEX_COL] = pd.to_datetime(df[TIME_INDEX_COL], errors="coerce")
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {name}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    # Outer join on date if present
    df = None
    for d in dfs:
        if df is None:
            df = d
        else:
            join_cols = [TIME_INDEX_COL] if (TIME_INDEX_COL in df.columns and TIME_INDEX_COL in d.columns) else None
            df = df.merge(d, on=join_cols, how="outer") if join_cols else pd.concat([df, d], axis=1)
    
    # sort and forward-fill
    if TIME_INDEX_COL in df.columns:
        df = df.sort_values(TIME_INDEX_COL).reset_index(drop=True)
        df = df.ffill().bfill()
    
    # drop duplicate columns created by merges
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def build_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build static features from the time series data.
    This is a simplified version - in production, you'd use the statics from the ETL pipeline.
    """
    static = {}
    
    # Example: compute diversity and dominant crude for columns ending with '_pct'
    BLEND_PCT_PREFIXES = ["blend_", "crude_"]
    pct_cols = [c for c in df.columns if c.endswith("_pct") and any(c.startswith(p) for p in BLEND_PCT_PREFIXES)]
    
    if pct_cols:
        pct = df[pct_cols].clip(lower=0).fillna(0)
        # normalize each row to sum 1.0
        row_sum = pct.sum(axis=1).replace(0, np.nan)
        norm = pct.div(row_sum, axis=0).fillna(0)
        
        # shannon diversity per row, then average over time to form *static* summary
        eps = 1e-12
        shannon = -np.sum(norm * np.log(norm + eps), axis=1)
        static["static_blend_diversity_index"] = float(np.nanmean(shannon))
        
        # dominant crude share (avg max share)
        static["static_dominant_crude_pct"] = float(np.nanmean(norm.max(axis=1)))
        static["static_num_significant_crudes"] = int(np.nanmean((norm > 0.05).sum(axis=1)))
        
        # per-crude mean/std across time
        for c in pct_cols:
            base = c.replace("_pct", "")
            static[f"static_{base}_mean"] = float(np.nanmean(norm[c]))
            static[f"static_{base}_std"] = float(np.nanstd(norm[c]))
    
    # Return single-row DataFrame
    return pd.DataFrame([static])


def split_columns(df: pd.DataFrame):
    """
    Split DataFrame into targets and features.
    """
    targets = [c for c in TARGET_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in targets and c != TIME_INDEX_COL]
    return df, targets, feature_cols


def prepare_model_inputs(targets_df: pd.DataFrame, statics_df: pd.DataFrame, futures_df: pd.DataFrame) -> dict:
    """
    Prepare inputs for the SageMaker model.
    
    Args:
        targets_df: DataFrame containing target variables
        statics_df: DataFrame containing static features
        futures_df: DataFrame containing future covariates
        
    Returns:
        Dictionary containing model input payload
    """
    # Get the latest timestamp
    last_ts = targets_df[TIME_INDEX_COL].max() if TIME_INDEX_COL in targets_df.columns else None
    
    # Prepare targets
    target_cols = [c for c in TARGET_COLS if c in targets_df.columns]
    
    # Prepare data sample (last 256 rows)
    data_sample = targets_df.tail(256).to_dict(orient="records")
    
    # Prepare static features
    static_features = statics_df.to_dict(orient="records")[0] if not statics_df.empty else {}
    
    payload = {
        "horizon": 7,
        "schema": {
            "time": TIME_INDEX_COL,
            "targets": target_cols,
        },
        "data_sample": data_sample,
        "static": static_features,
    }
    
    return payload

