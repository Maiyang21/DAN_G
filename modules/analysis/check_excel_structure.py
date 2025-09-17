import pandas as pd
import os

def check_excel_file():
    """
    Check the Excel file path and examine the Tag Monitoring sheet structure
    """
    # The hardcoded path from your code
    excel_path = "C:/Users/PC/Documents/abu/Documents/DATABASE/PROC_OPTIM/Monthly Reports/TS Average, Blend, and Lab Data February 2025.xlsx"
    
    print(f"🔍 Checking Excel file path: {excel_path}")
    print("=" * 80)
    
    # Check if file exists
    if not os.path.exists(excel_path):
        print(f"❌ ERROR: File does not exist at this path!")
        print(f"   Path: {excel_path}")
        print("\n💡 Possible solutions:")
        print("1. Check if the file path is correct")
        print("2. Check if the file exists in a different location")
        print("3. Update the path in your code")
        return False
    
    print(f"✅ File exists!")
    
    try:
        # List all sheets
        excel_file = pd.ExcelFile(excel_path)
        print(f"\n📊 Available sheets: {excel_file.sheet_names}")
        
        # Check if required sheets exist
        required_sheets = ["TS Monitoring Tags", "Blends", "Lab"]
        missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_file.sheet_names]
        
        if missing_sheets:
            print(f"⚠️  Missing required sheets: {missing_sheets}")
        else:
            print(f"✅ All required sheets found!")
        
        # Examine the TS Monitoring Tags sheet in detail
        if "TS Monitoring Tags" in excel_file.sheet_names:
            print(f"\n🔍 Examining 'TS Monitoring Tags' sheet:")
            print("-" * 50)
            
            ts_df = pd.read_excel(excel_path, sheet_name="TS Monitoring Tags")
            print(f"Shape: {ts_df.shape}")
            print(f"Columns: {list(ts_df.columns)}")
            
            print(f"\nFirst 5 rows:")
            print(ts_df.head())
            
            print(f"\nData types:")
            print(ts_df.dtypes)
            
            # Check for empty columns
            empty_cols = ts_df.columns[ts_df.isna().all()].tolist()
            if empty_cols:
                print(f"\n⚠️  Empty columns: {empty_cols}")
            
            # Check the structure - what are the first few column names?
            print(f"\n🔍 Column structure analysis:")
            print(f"Column 0 (first): '{ts_df.columns[0]}'")
            print(f"Column 1 (second): '{ts_df.columns[1]}'")
            print(f"Column 2 (third): '{ts_df.columns[2]}'")
            
            # Check if there are date columns
            print(f"\n📅 Checking for date-like columns:")
            for i, col in enumerate(ts_df.columns[:10]):  # Check first 10 columns
                try:
                    sample_values = ts_df[col].dropna().iloc[:5]
                    if len(sample_values) > 0:
                        pd.to_datetime(sample_values, errors='coerce')
                        print(f"  Column {i} '{col}': Potential date column")
                except:
                    pass
            
            # Check the Tag column (column 2) for what values it contains
            if len(ts_df.columns) > 2:
                tag_col = ts_df.columns[2]
                print(f"\n🏷️  Tag column analysis (column 2: '{tag_col}'):")
                unique_tags = ts_df[tag_col].dropna().unique()
                print(f"Unique tags found: {unique_tags[:20]}")  # Show first 20
                
                # Check for yield/flow related tags
                yield_flow_tags = [tag for tag in unique_tags if any(keyword in str(tag).lower() 
                                                                   for keyword in ['yield', 'flow', 'rate', 'production', 'output'])]
                if yield_flow_tags:
                    print(f"✅ Found yield/flow related tags: {yield_flow_tags}")
                else:
                    print(f"⚠️  No yield/flow related tags found in first 20 tags")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Excel File Structure Checker")
    print("=" * 80)
    
    success = check_excel_file()
    
    if not success:
        print(f"\n💡 To fix the empty target issue:")
        print("1. Make sure the Excel file exists at the specified path")
        print("2. Check that the 'TS Monitoring Tags' sheet has the right structure")
        print("3. Verify that the Tag column contains values with 'yield', 'flow', etc.")
        print("4. Update the file path in your code to point to the correct location")

