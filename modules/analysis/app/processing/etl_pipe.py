import numpy as np
import pandas as pd
import json
import re
import os
from datetime import datetime

# --- Utility Functions (Keep safe_float as needed, remove_nans can be imported) ---
def safe_float(value):
    # ... (implementation from your script) ...
    if pd.isna(value) or value in [None, '', 'nan', 'NaN', '-']: return np.nan
    try:
        if isinstance(value, str):
            value = value.strip().replace(',', '').replace('%', '')
            if not value: return np.nan
        return float(value)
    except (ValueError, TypeError): return np.nan

# --- Refactored Data Preparation Functions ---
# Keep generate_synthetic_target as is

def generate_synthetic_target(base_value, std_dev_factor, days, clamp_buffer=0.01):
    # ... (implementation from your script) ...
    try: base_value = float(base_value)
    except (ValueError, TypeError): return np.zeros(days)
    if base_value <= 0: return np.zeros(days)
    mean = base_value; std_daily = abs(mean) * std_dev_factor if mean != 0 else 0.01
    noise_sample = np.random.normal(loc=0.0, scale=std_daily, size=days)
    synthetic_data = noise_sample + mean
    min_val = np.min(synthetic_data)
    offset = (-min_val + clamp_buffer) if min_val < 0 else 0
    return synthetic_data + offset


def get_dates_and_tags_darts(ts_monitoring_data):
    # ... (Keep implementation, ensure it returns dates, df_wide_tags) ...
    if not ts_monitoring_data or not isinstance(ts_monitoring_data, list):
        raise ValueError("Invalid 'ts_monitoring' data for ETL.")
    example_entry = ts_monitoring_data[0]; date_cols = []
    for k in example_entry.keys():
        if k != 'Tag':
            try: pd.to_datetime(k, errors='raise'); date_cols.append(k)
            except: continue
    if not date_cols: raise ValueError("No date columns found in 'ts_monitoring'.")
    try: dates = pd.to_datetime(sorted(date_cols, key=lambda d: datetime.strptime(d, '%m/%d/%Y')), format='%m/%d/%Y')
    except ValueError: dates = pd.to_datetime(sorted(date_cols), infer_datetime_format=True)

    df_ts = pd.DataFrame(ts_monitoring_data)
    if 'Tag' not in df_ts.columns: raise ValueError("'Tag' column missing.")
    df_ts['Tag'] = df_ts['Tag'].astype(str)
    unique_tags = df_ts['Tag'].unique()
    selected_tags = unique_tags[:499] if len(unique_tags) > 499 else unique_tags
    df_ts_filtered = df_ts[df_ts['Tag'].isin(selected_tags)].copy()

    df_long = pd.melt(df_ts_filtered, id_vars=['Tag'], value_vars=date_cols, var_name='date_str', value_name='value')
    df_long['date'] = pd.to_datetime(df_long['date_str'], errors='coerce')
    df_long = df_long.dropna(subset=['date']).drop(columns=['date_str'])
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce').dropna()

    try: df_wide_tags = df_long.pivot_table(index='date', columns='Tag', values='value')
    except Exception as e: print(f"Pivoting error: {e}. Aggregating."); df_wide_tags = df_long.pivot_table(index='date', columns='Tag', values='value', aggfunc='mean')
    df_wide_tags = df_wide_tags.reindex(dates).ffill().bfill()
    return dates, df_wide_tags


def prepare_targets_darts(extracted_data, dates, df_wide_tags): # Now needs tags for flows
    # ... (Keep implementation, ensure it returns df_targets_wide) ...
    print("\nPreparing target variables (Wide Format)...")
    df_targets_wide = pd.DataFrame(index=dates); num_days = len(dates)
    # 1. Synthetic Quality
    quality_metrics = {}; product_lab_data = extracted_data.get("product_lab_analysis", [])
    guide_quality_products = ["Naphtha", "Kero", "Combined Diesel", "RCO"]
    if isinstance(product_lab_data, list) and product_lab_data:
        for entry in product_lab_data:
            metric_name = entry.get("Parameter"); product_col_name = ""
            if isinstance(metric_name, str):
                metric_name_clean = re.sub(r'\s*\(.*\)\s*', '', metric_name).strip()
                for prod_key in entry.keys():
                    if prod_key in guide_quality_products:
                        value = entry.get(prod_key); float_value = safe_float(value)
                        if not np.isnan(float_value):
                            quality_metrics.setdefault(prod_key, {})[metric_name_clean] = float_value
    quality_target_count = 0
    for product, metrics in quality_metrics.items():
        for metric, mean_value in metrics.items():
            target_series = generate_synthetic_target(mean_value, 0.01, num_days)
            df_targets_wide[f"target_{product}_{metric}"] = target_series; quality_target_count += 1
    print(f"Generated {quality_target_count} synthetic quality targets.")
    # 2. Synthetic Yield
    product_yield_avg = {}; product_yield_data = extracted_data.get("product_yield", [])
    if isinstance(product_yield_data, list):
        cb_pattern = re.compile(r'^CB\d+( Rev)?$')
        guide_yield_products = ["LPG", "Naphtha", "Kero", "Light_Diesel", "Heavy_Diesel", "RCO"]
        for entry in product_yield_data:
            parameter = entry.get("Parameter"); product_name = ""
            if isinstance(parameter, str) and "Yield %" not in parameter and "Total" not in parameter: product_name = parameter.strip()
            if isinstance(entry, dict):
                cb_values = [v for k, v in entry.items() if cb_pattern.match(k) and isinstance(v, (int, float)) and not np.isnan(v)]
                if cb_values and product_name:
                    standard_name = product_name.replace(" ", "_").replace("Kerosene", "Kero") # Basic standardizing
                    if standard_name in guide_yield_products: product_yield_avg[f"target_{standard_name}_Yield"] = np.mean(cb_values)
    yield_target_count = 0
    for target_col_name, mean_value in product_yield_avg.items():
        target_series = generate_synthetic_target(mean_value, 0.02, num_days)
        df_targets_wide[target_col_name] = target_series; yield_target_count += 1
    print(f"Generated {yield_target_count} synthetic yield targets.")
    # 3. Flowrate Targets
    flow_tags_map = {"target_RCO_flow": "101FIC3101", "target_Heavy_Diesel_flow": "101FIC4201", "target_Light_Diesel_flow": "101FIC5303", "target_Kero_flow": "101FIC3001", "target_Naphtha_flow": "102FIC3201"}
    flow_targets_found = 0
    for target_name, tag_name in flow_tags_map.items():
        if tag_name in df_wide_tags.columns:
            flow_series = df_wide_tags[tag_name].copy(); min_val = flow_series.min()
            offset = (-min_val + 0.01) if min_val < 0 else 0
            df_targets_wide[target_name] = flow_series + offset; flow_targets_found += 1
        else: print(f"Warning: Flow tag '{tag_name}' for '{target_name}' not found.")
    print(f"Added {flow_targets_found} actual flowrate targets.")
    return df_targets_wide


def prepare_past_covariates_darts(df_wide_tags):
    # ... (Keep implementation, returns df_wide_tags itself) ...
    print("\nPreparing past covariates (Wide Format)...")
    df_past_cov = df_wide_tags.copy()
    print(f"Using {len(df_past_cov.columns)} tags as past covariates.")
    return df_past_cov


def prepare_future_covariates_darts(extracted_data, dates):
    # ... (Keep implementation, returns df_future_cov_wide) ...
    print("\nPreparing future covariates (Wide Format)...")
    num_days = len(dates); df_future_cov = pd.DataFrame(index=dates)
    # 1. Blend Info
    blend_comp_data = extracted_data.get("blend_composition", [])
    if isinstance(blend_comp_data, list) and blend_comp_data:
        cb_pattern = re.compile(r'^CB\d+( Rev)?$'); blend_props = {'API': {}, 'Sulphur': {}}; cb_cols_in_data = set()
        api_row = next((item for item in blend_comp_data if item.get("Parameter") == "API"), None)
        sulphur_row = next((item for item in blend_comp_data if item.get("Parameter") == "Sulphur"), None)
        if api_row:
            for k, v in api_row.items():
                if cb_pattern.match(k): blend_props['API'][k] = safe_float(v); cb_cols_in_data.add(k)
        if sulphur_row:
            for k, v in sulphur_row.items():
                if cb_pattern.match(k): blend_props['Sulphur'][k] = safe_float(v); cb_cols_in_data.add(k)
        cb_cols_ordered = sorted(list(cb_cols_in_data), key=lambda x: int(re.search(r'\d+', x).group()))
        if blend_props['API'] and blend_props['Sulphur'] and cb_cols_ordered:
            num_blends = len(cb_cols_ordered)
            blend_indices = [i % num_blends for i in range(num_days)]
            api_values = [blend_props['API'].get(cb_cols_ordered[i], np.nan) for i in blend_indices]
            sulphur_values = [blend_props['Sulphur'].get(cb_cols_ordered[i], np.nan) for i in blend_indices]
            blend_num_values = [int(re.search(r'\d+', cb_cols_ordered[i]).group()) for i in blend_indices]
            df_future_cov['future_API'] = api_values; df_future_cov['future_Sulphur'] = sulphur_values; df_future_cov['future_Blend_Num'] = blend_num_values
            df_future_cov = df_future_cov.ffill().bfill() # Fill potential NaNs from missing CB values
        else: print("Warning: Could not find sufficient blend API/Sulphur data."); df_future_cov[['future_API', 'future_Sulphur', 'future_Blend_Num']] = np.nan
    else: print("Warning: 'blend_composition' data missing."); df_future_cov[['future_API', 'future_Sulphur', 'future_Blend_Num']] = np.nan
    # 2. Dosage Placeholders
    df_future_cov['future_Neutralizer_dosage'] = 10 + np.random.normal(0, 1, num_days)
    df_future_cov['future_AntiCorrosion_dosage'] = 5 + np.random.normal(0, 0.5, num_days)
    return df_future_cov

# --- Main Orchestration Function for ETL ---
def run_etl_pipeline(extracted_json_data):
    """Runs the full ETL process on extracted JSON data."""
    print("\n--- Running ETL Pipeline ---")
    if not extracted_json_data:
        raise ValueError("ETL pipeline received empty or invalid extracted data.")

    # 1. Extract dates and tags
    dates, df_wide_tags = get_dates_and_tags_darts(extracted_json_data.get('ts_monitoring'))

    # 2. Prepare Targets (needs tags for flows)
    df_targets_wide = prepare_targets_darts(extracted_json_data, dates, df_wide_tags)

    # 3. Prepare Past Covariates (uses tags)
    df_past_cov_wide = prepare_past_covariates_darts(df_wide_tags)

    # 4. Prepare Future Covariates
    df_future_cov_wide = prepare_future_covariates_darts(extracted_json_data, dates)

    # 5. Combine
    print("\nCombining into final Darts DataFrame...")
    df_final_darts = df_targets_wide.copy()
    past_cov_cols_to_add = [col for col in df_past_cov_wide.columns if col not in df_final_darts.columns]
    df_final_darts = pd.concat([df_final_darts, df_past_cov_wide[past_cov_cols_to_add]], axis=1)
    df_final_darts = pd.concat([df_final_darts, df_future_cov_wide], axis=1)
    df_final_darts.index = dates
    df_final_darts['time_idx'] = (df_final_darts.index - df_final_darts.index.min()).days
    df_final_darts = df_final_darts.ffill().bfill()

    nan_check = df_final_darts.isnull().sum().sum()
    if nan_check > 0: print(f"Warning: {nan_check} NaNs remain in final DataFrame!")
    else: print("No NaNs found in final DataFrame.")
    print(f"Final Darts-ready DataFrame shape: {df_final_darts.shape}")
    print("--- ETL Pipeline Finished ---")
    return df_final_darts