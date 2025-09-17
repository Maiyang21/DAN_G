import pandas as pd
import argparse
import sys
import os


def run_etl(excel_path, output_dir="processed"):
    # Check if input file exists
    if not os.path.exists(excel_path):
        print(f"❌ Error: Input file not found: {excel_path}")
        return None, None, None
    
    print(f"📁 Processing Excel file: {excel_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # === 1. Load sheets ===
    try:
        ts_df = pd.read_excel(excel_path, sheet_name="TS Monitoring Tags")
        blend_df = pd.read_excel(excel_path, sheet_name="Blends")
        lab_df = pd.read_excel(excel_path, sheet_name="Lab")
        print("✅ Successfully loaded all required sheets")
    except Exception as e:
        print(f"❌ Error loading Excel sheets: {e}")
        return None, None, None

    # === 2. Transform TS Monitoring Tags ===
    # The structure varies between files:
    # June 2024: Row 0: [S/N, Tag, Date1, Date2, ...], Row 1+: data
    # February 2025: Row 0: [S/N, DCS Tag, Tag, Date1, Date2, ...], Row 1+: data
    
    # Extract the header row and data rows
    header_row = ts_df.iloc[0]  # Row 0: contains column headers
    data_rows = ts_df.iloc[1:]  # Row 1+: actual data
    
    print(f"Header row: {header_row.tolist()[:10]}...")
    print(f"Data rows shape: {data_rows.shape}")
    
    # Create a proper DataFrame with correct column names
    ts_df = data_rows.copy()
    ts_df.columns = header_row.values
    
    print(f"After setting column names:")
    print(f"Columns: {list(ts_df.columns)[:10]}...")
    print(f"First few rows:")
    print(ts_df.head())
    
    # Drop rows where all values are NaN (empty rows)
    ts_df = ts_df.dropna(how="all", axis=0)
    
    # Detect if this file has DCS Tag column or not
    has_dcs_tag = 'DCS Tag' in ts_df.columns
    
    if has_dcs_tag:
        print("📋 File structure: Contains DCS Tag column (February 2025 format)")
        id_cols_count = 3  # S/N, DCS Tag, Tag
        tag_col_index = 2   # Tag is 3rd column
        dcs_tag_col_index = 1  # DCS Tag is 2nd column
    else:
        print("📋 File structure: No DCS Tag column (June 2024 format)")
        id_cols_count = 2  # S/N, Tag
        tag_col_index = 1   # Tag is 2nd column
        dcs_tag_col_index = None  # No DCS Tag column
    
    # The structure is now:
    # June 2024: Columns: S/N, Tag, Date1, Date2, Date3, ...
    # February 2025: Columns: S/N, DCS Tag, Tag, Date1, Date2, Date3, ...
    
    # Identify the structure
    id_cols = ts_df.iloc[:, :id_cols_count]  # First N columns: S/N, (DCS Tag), Tag
    data_cols = ts_df.iloc[:, id_cols_count:]  # Remaining columns: dates and values
    
    print(f"ID columns shape: {id_cols.shape}")
    print(f"Data columns shape: {data_cols.shape}")
    
    # Get the tag names from the Tag column
    tag_names = ts_df.iloc[:, tag_col_index].dropna().tolist()  # Tag column
    print(f"Available tags: {tag_names[:10]}...")  # Show first 10
    
    # Get the dates from the column names (these should be the actual date headers)
    dates = data_cols.columns.tolist()
    print(f"Available dates: {dates[:10]}...")  # Show first 10
    
    # Create a proper time series structure
    records = []
    
    for idx, row in ts_df.iterrows():
        if pd.isna(row.iloc[tag_col_index]):  # Skip rows without tag names
            continue
            
        tag_name = row.iloc[tag_col_index]  # Tag name
        tag_id = row.iloc[dcs_tag_col_index] if dcs_tag_col_index is not None else tag_name  # DCS Tag ID or use tag name if no DCS Tag
        
        # Get values for this tag across all dates
        for i, date_col in enumerate(data_cols.columns):
            # Get the value using iloc to avoid Series issues
            value = row.iloc[i + id_cols_count]  # +id_cols_count because first N columns are ID columns
            if pd.notna(value):  # Only check if the value is not NaN
                try:
                    # Convert date column name to datetime
                    date_val = pd.to_datetime(date_col, errors='coerce')
                    if pd.notna(date_val):
                        records.append({
                            'date': date_val,
                            'tag': tag_name,
                            'tag_id': tag_id,
                            'value': value
                        })
                except:
                    continue
    
    # Create DataFrame from records
    ts_long = pd.DataFrame(records)
    print(f"Long format data shape: {ts_long.shape}")
    
    if len(ts_long) == 0:
        print("❌ No valid time series data found!")
        print("Debugging: Let's check what's in the data columns...")
        print(f"Sample data column values: {data_cols.iloc[0, :5].tolist()}")
        return None, None, None

    # Pivot: one column per Tag
    ts_wide = ts_long.pivot(index="date", columns="tag", values="value").reset_index()

    # Show available columns for debugging
    print("Available tags:", ts_wide.columns.tolist())

    # === 3. Transform Blends to get YIELDS ===
    print(f"\n🔍 Processing blend yields...")
    
    # Extract blend yields (crude oil fractions)
    # Row 0: API, Row 1: Sulphur, Row 2: Crude Oil%, Row 3+: Crude types
    crude_compositions = blend_df.iloc[3:, :]  # Skip API, Sulphur, Crude Oil% rows
    
    # Get crude type names (first column)
    crude_types = crude_compositions.iloc[:, 0].dropna().tolist()
    print(f"Crude types found: {crude_types}")
    
    # Get blend columns (CB01, CB02, etc.)
    blend_cols = [col for col in blend_df.columns if col != 'Blend No']
    
    # Create blend yield records
    blend_yield_records = []
    for blend in blend_cols:
        blend_data = {"blend_id": blend}
        
        # Get API and Sulphur
        blend_data["API"] = blend_df.iloc[0, blend_df.columns.get_loc(blend)]
        blend_data["Sulphur"] = blend_df.iloc[1, blend_df.columns.get_loc(blend)]
        
        # Get crude compositions (yields)
        for crude_type in crude_types:
            crude_row = crude_compositions[crude_compositions.iloc[:, 0] == crude_type]
            if not crude_row.empty:
                blend_idx = blend_df.columns.get_loc(blend)
                yield_value = crude_row.iloc[0, blend_idx]
                blend_data[f"yield_{crude_type}"] = yield_value if pd.notna(yield_value) else 0.0
        
        blend_yield_records.append(blend_data)
    
    blend_yields_df = pd.DataFrame(blend_yield_records)
    print(f"Blend yields shape: {blend_yields_df.shape}")
    print(f"Blend yield columns: {list(blend_yields_df.columns)}")

    # === DATA ORGANIZATION FOR AUTOFORMER MODEL ===
    
    # 1. TARGETS (What we want to predict - Product Streams + Key Measurements + Blend Yields)
    target_tags = [
        # Product Streams (Main outputs) - with descriptive names
        '102FIC3201',  # Total Stabilized Naphtha Product Flowrate
        '101FIC3001',  # Total Kerosene Product Flowrate
        '104FIC2902',  # Jet Fuel Product from Train 1
        '101FIC5303',  # Total Light Diesel Product Flowrate
        '101FIC4201',  # Total Heavy Diesel Product Flowrate
        '101FIC3101',  # Total Atmospheric Residue Flowrate
        
        # Additional Key Measurements (Critical flows)
        '101FIC5802',  # Crude Column Naphtha to SGCU
        '101FIC3002',  # Kerosene to Light Diesel to DHT
        '101FIC2001',  # Light Diesel to DHT Unit
        '101FIC6101',  # Heavy Diesel to MHC
        '101FIC3103',  # Atm. Residue to RFCC Unit
        '101FIC3104',  # Atm. Residue to Storage
        '101FIC6302',  # Atm. Residue to Storage from 101-EE-1027
    ]
    
    # 2. INPUT FEATURES (What Autoformer uses to predict - Feed Streams + Other Monitoring Tags)
    input_feature_tags = [
        # Feed Streams (Primary input features)
        '101FIC1401',  # Raw Crude Pump Discharge Flow
        '101FI1402',   # Train A Raw Crude Flow
        '101FI1403',   # Train B Raw Crude Flow
        '102FIC2701',  # Rich Gas Knockout Drum Liquid to CDU
        '102FIC2602',  # Rich Sponge Oil Flash Drum Bottoms to CDU
    ]
    
    # Filter for target columns
    target_cols = [c for c in ts_wide.columns if c in target_tags]
    
    if not target_cols:
        print("⚠️ No target tags found. Falling back to all columns.")
        target_cols = [c for c in ts_wide.columns if c != "date"]
    else:
        print(f"✅ Found {len(target_cols)} target tags")
        print(f"Target tags: {target_cols}")
    
    # Filter for input feature columns (only primary feed streams)
    input_feature_cols = [c for c in ts_wide.columns if c in input_feature_tags]
    
    print(f"✅ Found {len(input_feature_cols)} primary input features (feed streams)")
    
    # Create the three main datasets
    targets = ts_wide[["date"] + target_cols]
    input_features = ts_wide[["date"] + input_feature_cols]
    
    # Rename target columns for better interpretation
    target_column_mapping = {
        '102FIC3201': 'Total_Stabilized_Naphtha_Product_Flowrate',
        '101FIC3001': 'Total_Kerosene_Product_Flowrate', 
        '104FIC2902': 'Jet_Fuel_Product_Train1_Flowrate',
        '101FIC5303': 'Total_Light_Diesel_Product_Flowrate',
        '101FIC4201': 'Total_Heavy_Diesel_Product_Flowrate',
        '101FIC3101': 'Total_Atmospheric_Residue_Flowrate',
        '101FIC5802': 'Crude_Column_Naphtha_to_SGCU_Flowrate',
        '101FIC3002': 'Kerosene_to_Light_Diesel_DHT_Flowrate',
        '101FIC2001': 'Light_Diesel_to_DHT_Unit_Flowrate',
        '101FIC6101': 'Heavy_Diesel_to_MHC_Flowrate',
        '101FIC3103': 'Atm_Residue_to_RFCC_Unit_Flowrate',
        '101FIC3104': 'Atm_Residue_to_Storage_Flowrate',
        '101FIC6302': 'Atm_Residue_to_Storage_EE1027_Flowrate'
    }
    
    # Apply the mapping to targets
    targets = targets.rename(columns=target_column_mapping)
    
    # Also rename input feature columns for clarity
    input_feature_column_mapping = {
        '101FIC1401': 'Raw_Crude_Pump_Discharge_Flow',
        '101FI1402': 'Train_A_Raw_Crude_Flow',
        '101FI1403': 'Train_B_Raw_Crude_Flow',
        '102FIC2701': 'Rich_Gas_Knockout_Drum_Liquid_to_CDU',
        '102FIC2602': 'Rich_Sponge_Oil_Flash_Drum_Bottoms_to_CDU'
    }
    
    # Apply the mapping to input features
    input_features = input_features.rename(columns=input_feature_column_mapping)
    futures = input_features.copy()
    
    # Add blend yields to targets (these are important targets for optimization)
    print(f"\n🎯 Adding blend yields as targets...")
    yield_cols = [col for col in blend_yields_df.columns if col.startswith('yield_')]
    print(f"Yield columns to add: {yield_cols}")
    
    # Filter yields to only include those that have corresponding flow rate measurements
    # Map yield names to target flow rate tags
    yield_to_flow_mapping = {
        'Gas & LPG': '101FIC5802',      # Gas & LPG yield -> Crude Column Naphtha to SGCU (closest gas flow)
        'Naphtha ': '102FIC3201',       # Naphtha yield -> Total Stabilized Naphtha Product Flowrate
        'Kerosene': '101FIC3001',       # Kerosene yield -> Total Kerosene Product Flowrate
        'Light Diesel': '101FIC5303',   # Light Diesel yield -> Total Light Diesel Product Flowrate
        'Heavy Diesel': '101FIC4201',   # Heavy Diesel yield -> Total Heavy Diesel Product Flowrate
        'RCO': '101FIC3101',            # RCO yield -> Total Atmospheric Residue Flowrate
    }
    
    # Only keep yields that have corresponding flow rate measurements
    filtered_yield_cols = []
    for yield_col in yield_cols:
        crude_type = yield_col.replace('yield_', '')
        if crude_type in yield_to_flow_mapping:
            filtered_yield_cols.append(yield_col)
            print(f"✅ Including yield: {crude_type} -> {yield_to_flow_mapping[crude_type]}")
        else:
            print(f"⚠️ Excluding yield: {crude_type} (no corresponding flow rate measurement)")
    
    print(f"\nFiltered yield columns: {filtered_yield_cols}")
    
    # For each date, we'll add the filtered blend yields
    # Since we don't have date-specific blend data, we'll use the average yields
    avg_yields = blend_yields_df[filtered_yield_cols].mean()
    
    # Add yield columns to targets with descriptive names
    for yield_col in filtered_yield_cols:
        crude_type = yield_col.replace('yield_', '')
        descriptive_name = f"Blend_Yield_{crude_type}"
        targets[descriptive_name] = avg_yields[yield_col]
    
    print(f"Final targets shape: {targets.shape}")
    print(f"Final input features shape: {input_features.shape}")
    print(f"Target columns: {target_cols}")
    print(f"Primary input features: {input_feature_cols}")
    print(f"Blend yield targets added: {len(filtered_yield_cols)}")

    # === 4. Transform Blends for Static Features ===
    api_row = blend_df.iloc[0, 1:]  # API values
    sulphur_row = blend_df.iloc[1, 1:]
    blend_ids = blend_df.columns[1:]  # CB01, CB02, ...

    static_records = []
    for blend in blend_ids:
        blend_data = {"blend_id": blend}
        blend_data["API"] = api_row[blend]
        blend_data["Sulphur"] = sulphur_row[blend]

        # crude compositions start from row index 3 onward
        crude_rows = blend_df.iloc[3:, [0, blend_df.columns.get_loc(blend)]]
        crude_rows = crude_rows.dropna()
        crude_dict = dict(zip(crude_rows.iloc[:, 0], crude_rows.iloc[:, 1]))

        # Convert values to numeric and filter out non-numeric values
        numeric_crude_dict = {}
        for k, v in crude_dict.items():
            try:
                numeric_value = float(v)
                if not pd.isna(numeric_value):
                    numeric_crude_dict[k] = numeric_value
            except (ValueError, TypeError):
                continue  # Skip non-numeric values

        total = sum(numeric_crude_dict.values())
        for k, v in numeric_crude_dict.items():
            blend_data[f"crude_{k}"] = v / total if total > 0 else 0

        static_records.append(blend_data)

    statics = pd.DataFrame(static_records)

    # === 5. Input Features (Futures for Autoformer) ===
    # These are all the input features that the Autoformer will use to predict targets
    futures = input_features.copy()

    # === 6. Save outputs ===
    os.makedirs(output_dir, exist_ok=True)
    targets.to_csv(f"{output_dir}/targets.csv", index=False)
    statics.to_csv(f"{output_dir}/statics.csv", index=False)
    futures.to_csv(f"{output_dir}/futures.csv", index=False)

    print(f"Saved processed data to {output_dir}")
    print(f"Targets shape: {targets.shape}")
    print(f"Statics shape: {statics.shape}")
    print(f"Futures shape: {futures.shape}")
    
    print(f"\n🎯 AUTOFORMER MODEL DATA SUMMARY:")
    print(f"   Targets: {len(target_cols)} columns (Product Streams + Key Measurements)")
    print(f"   Blend Yields: {len(filtered_yield_cols)} columns (Product Yields with Flow Measurements)")
    print(f"   Total Targets: {targets.shape[1]-1} columns (including date)")
    print(f"   Primary Input Features: {len(input_feature_cols)} columns (Feed Streams)")
    print(f"   Static Features: {len(statics.columns)} columns (Blend Information)")
    
    return targets, statics, futures





def main():
    """Main CLI function for the ETL process"""
    parser = argparse.ArgumentParser(
        description="ETL Pipeline for CDU Process Optimization Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default output directory
  python test_ETL.py "path/to/your/excel/file.xlsx"
  
  # Process with custom output directory
  python test_ETL.py "path/to/your/excel/file.xlsx" --output "custom/output/dir"
  
  # Show help
  python test_ETL.py --help
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the Excel file containing CDU data (must have 'TS Monitoring Tags' and 'Blends' sheets)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="processed",
        help="Output directory for processed CSV files (default: 'processed')"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting ETL process for CDU Process Optimization...")
        print("=" * 60)
        
        targets, statics, futures = run_etl(args.input_file, args.output)
        
        if targets is not None:
            print("\n" + "=" * 60)
            print("🎉 ETL process completed successfully!")
            print(f"📊 Targets shape: {targets.shape}")
            print(f"📊 Statics shape: {statics.shape}")
            print(f"📊 Futures shape: {futures.shape}")
            print(f"💾 Output saved to: {args.output}/")
            print("=" * 60)
        else:
            print("\n❌ ETL process failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running ETL: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()