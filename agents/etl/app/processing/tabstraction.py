import pandas as pd
import json
import math
import numpy as np
import os
import re
import traceback # For detailed error logging

# --- Utility Functions (Keep safe_float, remove_nans as they are) ---
def safe_float(value):
    # ... (implementation from your script) ...
    if pd.isna(value) or value in [None, '', 'nan', 'NaN', '-']: return np.nan
    try:
        if isinstance(value, str):
            value = value.strip().replace(',', '').replace('%', '')
            if not value: return np.nan
        return float(value)
    except (ValueError, TypeError): return np.nan

def remove_nans(obj):
    # ... (implementation from your script) ...
    if isinstance(obj, dict):
        processed_dict = {k: remove_nans(v) for k, v in obj.items()}
        return {k: v for k, v in processed_dict.items() if v is not None}
    elif isinstance(obj, list):
        processed_list = [remove_nans(item) for item in obj]
        return [item for item in processed_list if item is not None]
    return obj


# --- Refactored extract_table ---
def extract_table(csv_path, table_name, skiprows=None, header_row=None, columns=None, usecols=None, nrows=None):
    """ Extracts a table from a CSV file path."""
    print(f"\nAttempting to extract table '{table_name}' from: {csv_path}")
    # ... (Keep the core logic of your extract_table function) ...
    # Important: Ensure it handles file not found gracefully or raises exceptions
    # Make sure it RETURNS the extracted list of dictionaries or None on error
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return None
    try:
        # ... (Rest of your pandas reading and processing logic) ...
        read_header = header_row if columns is None else None
        df = pd.read_csv(
            csv_path, skiprows=skiprows, header=read_header, usecols=usecols,
            nrows=nrows, encoding='utf-8', keep_default_na=True
        )
        initial_cols = df.columns.tolist()
        # ... (Column renaming logic) ...
        if columns:
            if len(initial_cols) >= len(columns):
                df = df.iloc[:, :len(columns)]; df.columns = columns
            else: print(f"Warning: Read {len(initial_cols)} cols, but {len(columns)} names provided for '{table_name}'.")
        df.columns = [str(col).strip() if pd.notna(col) else f'Unnamed: {i}' for i, col in enumerate(df.columns)]
        df = df.dropna(axis=0, how='all')
        # ... (safe_float application) ...
        parameter_col_name = df.columns[0]
        for col in df.columns:
            if col != parameter_col_name:
                df[col] = [safe_float(v) if not pd.isna(v) else None for v in df[col]]
            else: df[col] = df[col].astype(str).str.strip()
        df = df.replace({np.nan: None})
        result = df.to_dict(orient='records')
        print(f"Successfully extracted {len(result)} records for '{table_name}'.")
        return result
    except Exception as e:
        print(f"Error extracting table '{table_name}' from {csv_path}: {e}")
        traceback.print_exc()
        return None

# --- Refactored Table Extraction Logic ---
def extract_lab_data(lab_csv_path):
    """Extracts tables from the Lab CSV."""
    if not os.path.exists(lab_csv_path): return None
    tables = {'raw_crude_lab_analysis': None, 'product_lab_analysis': None, 'overhead_water_analysis': None}
    # ... (Keep the specific extract_table calls using lab_csv_path) ...
    crude_cols = ['Parameter', 'Limit', 'Raw Crude', 'Train A', 'Train B']; crude_idx = [0, 2, 4, 5, 6]
    tables['raw_crude_lab_analysis'] = extract_table(lab_csv_path, 'raw_crude_lab_analysis', skiprows=3, header_row=None, columns=crude_cols, usecols=crude_idx, nrows=7)
    prod_cols = ['Parameter', 'Naphtha', 'Kero', 'ATF', 'Light Diesel', 'Heavy Diesel', 'RCO', 'Combined Diesel']; prod_idx = [0, 2, 3, 4, 5, 6, 7, 8]
    tables['product_lab_analysis'] = extract_table(lab_csv_path, 'product_lab_analysis', skiprows=11, header_row=None, columns=prod_cols, usecols=prod_idx, nrows=8)
    water_cols = ['Parameter', 'Limit', 'Value']; water_idx = [0, 1, 2]
    tables['overhead_water_analysis'] = extract_table(lab_csv_path, 'overhead_water_analysis', skiprows=20, header_row=None, columns=water_cols, usecols=water_idx, nrows=4)
    return tables

def extract_ts_monitoring_data(ts_csv_path):
    """Extracts time series monitoring data."""
    if not os.path.exists(ts_csv_path): return None
    try:
        header_df = pd.read_csv(ts_csv_path, skiprows=1, nrows=1, encoding='utf-8')
        all_cols = [str(col).strip() for col in header_df.columns]
        fixed_cols = ['S/N', 'DCS Tag', 'Tag', 'Average']
        date_cols = [col for col in all_cols if col not in fixed_cols and col and not col.startswith('Unnamed:')]
        usecols_ts = ['Tag'] + date_cols
        return extract_table(ts_csv_path, 'ts_monitoring', skiprows=1, header_row=0, usecols=usecols_ts, columns=None)
    except Exception as e: print(f"Error extracting 'ts_monitoring': {e}"); return None

def extract_blend_data(blend_csv_path):
    """Extracts blend composition and product yield data."""
    if not os.path.exists(blend_csv_path): return {'blend_composition': None, 'product_yield': None}
    # ... (Keep the logic from your modified extract_blend_table) ...
    # Make sure it RETURNS the dictionary {'blend_composition': [...], 'product_yield': [...]}
    print(f"\nExtracting full blend table from: {blend_csv_path}")
    try:
        header_df = pd.read_csv(blend_csv_path, nrows=1, encoding='utf-8')
        all_columns = [str(col).strip() for col in header_df.columns]
        parameter_col_name = all_columns[0]
        cb_pattern = re.compile(r'^CB\d+( Rev)?$')
        cb_columns = [col for col in all_columns if cb_pattern.match(col)]
        df_blend = pd.read_csv(blend_csv_path, header=0, encoding='utf-8', keep_default_na=True)
        df_blend.columns = [str(col).strip() for col in df_blend.columns]

        blend_composition = []
        product_yield = []
        in_yield_section = False
        for index, row in df_blend.iterrows():
            parameter = row[parameter_col_name]
            if pd.isna(parameter): continue
            parameter_str = str(parameter).strip()
            if "Yield %" in parameter_str: in_yield_section = True; continue
            elif parameter_str == '': continue
            record = {'Parameter': parameter_str}; has_valid_cb_value = False
            for cb_col in cb_columns:
                if cb_col in row:
                    cb_value_float = safe_float(row[cb_col])
                    if not np.isnan(cb_value_float): record[cb_col] = cb_value_float; has_valid_cb_value = True
            if has_valid_cb_value:
                if in_yield_section:
                    if 'Total' not in parameter_str: product_yield.append(record)
                else:
                     # Crude name check simplified for brevity
                     if parameter_str in ['API','Sulphur'] or parameter_str.isupper(): # Basic check
                         blend_composition.append(record)
            else: print(f"Skipping row '{parameter_str}' - no valid CB values.")
        return {'blend_composition': blend_composition, 'product_yield': product_yield}
    except Exception as e: print(f"Error processing blend table: {e}"); traceback.print_exc(); return {'blend_composition': None, 'product_yield': None}


# --- Main Orchestration Function for Extraction ---
def run_extraction(lab_csv_path, ts_csv_path, blend_csv_path):
    """Runs all extraction steps given file paths."""
    data = {}
    print("\n--- Running Data Extraction ---")
    lab_tables = extract_lab_data(lab_csv_path)
    if lab_tables: data.update(lab_tables)
    data['ts_monitoring'] = extract_ts_monitoring_data(ts_csv_path)
    blend_yield_tables = extract_blend_data(blend_csv_path)
    data.update(blend_yield_tables)

    final_data = {k: v for k, v in data.items() if not (v is None or (isinstance(v, list) and not v))}
    if not final_data: print("Warning: No data extracted or all tables were empty.")
    cleaned_data = remove_nans(final_data)
    print("--- Data Extraction Finished ---")
    return cleaned_data