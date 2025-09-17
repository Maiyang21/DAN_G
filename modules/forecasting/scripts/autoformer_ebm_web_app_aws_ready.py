# Project: refinery-forecast-app
# Web app to upload files, run ETL, push transformed data to S3 (via Access Point),
# and call a SageMaker endpoint hosting Autoformer + EBM for 7‑day forecasts + explanations.
#
# ├── streamlit_app.py            # UI: upload → ETL → upload to S3 → predict → explain → visualize
# ├── etl_pipeline.py             # Minimal ETL that builds targets/covariates from uploaded CSVs
# ├── aws_io.py                   # S3 Access Point upload + SageMaker Runtime invoke helpers
# ├── config.py                   # Reads env vars (bucket/access point, region, endpoint name)
# ├── requirements.txt            # App deps
# └── README.md                   # How to run locally & deploy

# ==============================
# File: requirements.txt
# ==============================
streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
boto3==1.34.150
botocore==1.34.150
pyarrow==16.1.0
plotly==5.22.0
scikit-learn==1.4.2
python-dateutil==2.9.0.post0

# ==============================
# File: config.py
# ==============================
import os

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Use either S3_BUCKET *or* S3_ACCESS_POINT_ARN (preferred for shared VPC access)
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ACCESS_POINT_ARN = os.getenv("S3_ACCESS_POINT_ARN", "")  # e.g., arn:aws:s3:us-east-1:123456789012:accesspoint/my-ap
S3_PREFIX = os.getenv("S3_PREFIX", "refinery-forecast/input/")

SM_ENDPOINT = os.getenv("SM_ENDPOINT", "autoformer-ebm-endpoint")

# Which columns are considered targets (flows & yields)
TARGET_COLS = [
    "target_RCO_flow","target_Heavy_Diesel_flow","target_Light_Diesel_flow","target_Kero_flow","target_Naphtha_flow",
    "target_RCO_Yield","target_Heavy_Diesel_Yield","target_Light_Diesel_Yield","target_Kero_Yield"
]

# If your ETL computes future/static features with these names:
STATIC_PREFIX = "static_"
FUTURE_PREFIX = "future_"
TIME_INDEX_COL = "date"

# ==============================
# File: aws_io.py
# ==============================
from typing import Dict, Tuple
import io
import json
import pandas as pd
import boto3
from botocore.config import Config as BotoConfig
from config import AWS_REGION, S3_BUCKET, S3_ACCESS_POINT_ARN, S3_PREFIX, SM_ENDPOINT

_s3_cfg = BotoConfig(s3={"addressing_style": "virtual"}, s3_use_arn_region=True, retries={"max_attempts": 10})
s3 = boto3.client("s3", region_name=AWS_REGION, config=_s3_cfg)
smr = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

def upload_df_csv(df: pd.DataFrame, key: str) -> str:
    """Upload a DataFrame as CSV to S3 bucket or Access Point. Returns the S3 URI used."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    if S3_ACCESS_POINT_ARN:
        s3.put_object(Bucket=S3_ACCESS_POINT_ARN, Key=key, Body=csv_bytes)
        return f"s3://{S3_ACCESS_POINT_ARN}/{key}"
    elif S3_BUCKET:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=csv_bytes)
        return f"s3://{S3_BUCKET}/{key}"
    else:
        raise RuntimeError("Neither S3_ACCESS_POINT_ARN nor S3_BUCKET configured.")


def invoke_endpoint(payload: Dict) -> Dict:
    """Invoke SageMaker endpoint with JSON payload. Returns JSON dict."""
    resp = smr.invoke_endpoint(
        EndpointName=SM_ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = resp["Body"].read()
    try:
        return json.loads(body)
    except Exception:
        return {"raw": body.decode("utf-8", errors="ignore")}

# ==============================
# File: etl_pipeline.py
# ==============================
"""
Minimal ETL to unify uploaded refinery CSVs into a single model input frame.
Expects: at least one uploaded CSV that contains a 'date' column and target columns.
If multiple CSVs are uploaded, they are merged on 'date' (outer join), with columns deduped.
Computes simple static summaries for blend composition columns (e.g., blend_XXX_pct) and passes
future controls through unchanged. This is a *template* — plug in your full ETL when ready.
"""
from typing import List, Tuple
import pandas as pd
import numpy as np
from config import TARGET_COLS, TIME_INDEX_COL

BLEND_PCT_PREFIXES = ["blend_", "crude_"]  # columns like blend_AGB_pct, crude_AGB_pct


def _read_any_csv(file_bytes, filename: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_bytes)
    except Exception:
        file_bytes.seek(0)
        return pd.read_csv(file_bytes, engine="python")


def unify_csvs(uploaded_files: List[Tuple[str, bytes]]) -> pd.DataFrame:
    dfs = []
    for name, fb in uploaded_files:
        df = _read_any_csv(io.BytesIO(fb), name)
        # ensure date
        if TIME_INDEX_COL in df.columns:
            df[TIME_INDEX_COL] = pd.to_datetime(df[TIME_INDEX_COL], errors="coerce")
        dfs.append(df)
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
    static = {}
    # Example: compute diversity and dominant crude for columns ending with '_pct'
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
    # Future controls can be averaged as static priors if desired
    # Return single-row DataFrame
    return pd.DataFrame([static])


def split_columns(df: pd.DataFrame):
    targets = [c for c in TARGET_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in targets]
    return df, targets, feature_cols

# ==============================
# File: streamlit_app.py
# ==============================
import io
import json
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from aws_io import upload_df_csv, invoke_endpoint
from etl_pipeline import unify_csvs, build_static_features, split_columns
from config import S3_PREFIX, TIME_INDEX_COL, TARGET_COLS

st.set_page_config(page_title="Refinery Forecast App", layout="wide")
st.title("Refinery Forecast App — Autoformer + EBM (AWS)")

with st.sidebar:
    st.header("1) Upload input CSVs")
    files = st.file_uploader(
        "Upload CSV files (Lab.csv, TS Monitoring Tags.csv, blend.csv, etc.)",
        type=["csv"], accept_multiple_files=True
    )
    st.markdown("---")
    st.header("2) Actions")
    do_etl = st.button("Run ETL & Upload to S3")
    do_predict = st.button("Request 7‑day Forecast")

# Store in session state
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "etl_df" not in st.session_state:
    st.session_state.etl_df = None
if "static_df" not in st.session_state:
    st.session_state.static_df = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ======================
# ETL Phase
# ======================
if do_etl:
    if not files:
        st.error("Please upload at least one CSV file.")
    else:
        up = [(f.name, f.getvalue()) for f in files]
        df = unify_csvs(up)
        st.session_state.raw_df = df.copy()
        st.success(f"Unified dataset shape: {df.shape}")
        st.dataframe(df.head(20))

        # Build static variates
        static_df = build_static_features(df)
        st.session_state.static_df = static_df
        st.info("Computed static features (single row):")
        st.dataframe(static_df)

        # Optionally keep only columns needed by the model (targets + covariates)
        etl_df, targets, feats = split_columns(df)
        st.session_state.etl_df = etl_df
        st.write(f"Targets found: {len(targets)} → {targets}")
        # Upload both to S3
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        data_key = f"{S3_PREFIX}batch_{timestamp}/timeseries.csv"
        static_key = f"{S3_PREFIX}batch_{timestamp}/static.csv"
        data_uri = upload_df_csv(etl_df, data_key)
        static_uri = upload_df_csv(static_df, static_key)
        st.success("Uploaded to S3:")
        st.code(data_uri)
        st.code(static_uri)

# ======================
# Predict Phase
# ======================
if do_predict:
    if st.session_state.etl_df is None or st.session_state.static_df is None:
        st.error("Run ETL & upload first.")
    else:
        # Construct payload expected by the SageMaker endpoint
        # You can adapt the schema here to match your endpoint handler.
        last_ts = st.session_state.etl_df[TIME_INDEX_COL].max() if TIME_INDEX_COL in st.session_state.etl_df.columns else None
        payload = {
            "horizon": 7,
            "schema": {
                "time": TIME_INDEX_COL,
                "targets": [c for c in TARGET_COLS if c in st.session_state.etl_df.columns],
            },
            "data_sample": st.session_state.etl_df.tail(256).to_dict(orient="records"),
            "static": st.session_state.static_df.to_dict(orient="records")[0],
        }
        with st.spinner("Invoking SageMaker endpoint..."):
            resp = invoke_endpoint(payload)
        st.session_state.prediction = resp
        st.success("Prediction received.")
        st.json(resp if len(json.dumps(resp)) < 5000 else {"keys": list(resp.keys())})

        # Optional: visualize forecasts if endpoint returns a standard structure
        # Expecting: {"forecast": [{"date":..., "target":..., "value":...}, ...], "explanations": {...}}
        if isinstance(resp, dict) and "forecast" in resp:
            fdf = pd.DataFrame(resp["forecast"])  # columns: date, target, p50 (or value)
            # try to parse date
            if "date" in fdf.columns:
                fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce")
            st.subheader("Forecast Visualizations")
            # plot one figure per target
            for tgt in fdf["target"].unique():
                sub = fdf[fdf["target"] == tgt]
                y_col = "p50" if "p50" in sub.columns else ("value" if "value" in sub.columns else None)
                if y_col is None:
                    continue
                fig = px.line(sub, x="date", y=y_col, title=f"7‑day Forecast — {tgt}")
                st.plotly_chart(fig, use_container_width=True)

        # Show EBM explanations if present
        if isinstance(resp, dict) and "explanations" in resp:
            st.subheader("Static Feature Explanations (EBM)")
            try:
                # Example schema: explanations = {feature_name: importance_float, ...}
                imp = pd.DataFrame([
                    {"feature": k, "importance": v} for k, v in resp["explanations"].items()
                ]).sort_values("importance", ascending=False)
                st.dataframe(imp)
                fig = px.bar(imp.head(20), x="feature", y="importance", title="Top Static Drivers (EBM)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write("Raw explanations:")
                st.json(resp["explanations"]) 

# ==============================
# File: README.md
# ==============================
# Refinery Forecast App — Autoformer + EBM (AWS)

## 1) Configure
Set the following environment variables locally or in your deployment:

```
AWS_REGION=us-east-1
S3_ACCESS_POINT_ARN=arn:aws:s3:us-east-1:123456789012:accesspoint/my-ap   # or set S3_BUCKET
S3_PREFIX=refinery-forecast/input/
SM_ENDPOINT=autoformer-ebm-endpoint
```

Grant the app IAM permissions:
- s3:PutObject (on your Access Point or bucket prefix)
- sagemaker:InvokeEndpoint (on your endpoint)

## 2) Install & Run
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 3) Expected Endpoint Contract
The app sends a JSON payload:
```json
{
  "horizon": 7,
  "schema": {"time": "date", "targets": ["target_RCO_flow", "target_RCO_Yield", ...]},
  "data_sample": [{"date": "2025-06-01", "target_RCO_flow": 700, ...}, ...],
  "static": {"static_blend_diversity_index": 1.2, "static_blend_AGB_mean": 0.2, ...}
}
```
Your SageMaker inference handler should parse this, construct model inputs (Autoformer for time‑series; EBM for static), and return:
```json
{
  "forecast": [
    {"date": "2025-08-30", "target": "target_RCO_flow", "p50": 680.2, "p10": 650.0, "p90": 710.5},
    ...
  ],
  "explanations": {"static_blend_diversity_index": 0.34, "static_blend_AGB_mean": 0.22, ...}
}
```

## 4) Notes
- Replace the minimal ETL in `etl_pipeline.py` with your production pipeline.
- If using an S3 **Access Point**, ensure your VPC endpoint & policies allow PutObject via the access point ARN.
- The UI visualizes forecasts per target and a bar chart of EBM feature importances when provided by the endpoint.
