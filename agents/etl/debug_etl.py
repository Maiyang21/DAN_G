import pandas as pd
import os

def debug_excel_file(file_path):
    """
    Debug function to examine an Excel file and understand its structure
    """
    print(f"=== DEBUGGING EXCEL FILE: {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File does not exist at {file_path}")
        return
    
    try:
        # List all sheets
        excel_file = pd.ExcelFile(file_path)
        print(f"📊 Available sheets: {excel_file.sheet_names}")
        
        # Examine each sheet
        for sheet_name in excel_file.sheet_names:
            print(f"\n--- Sheet: {sheet_name} ---")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"First few rows:")
            print(df.head())
            print(f"Data types: {df.dtypes}")
            
            # Check for empty columns
            empty_cols = df.columns[df.isna().all()].tolist()
            if empty_cols:
                print(f"⚠️  Empty columns: {empty_cols}")
            
            # Check for date-like columns
            date_cols = []
            for col in df.columns:
                try:
                    pd.to_datetime(df[col].iloc[:5], errors='coerce')
                    date_cols.append(col)
                except:
                    pass
            
            if date_cols:
                print(f"📅 Potential date columns: {date_cols}")
                
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def test_target_filtering():
    """
    Test the target column filtering logic
    """
    print("\n=== TESTING TARGET COLUMN FILTERING ===")
    
    # Sample column names to test
    test_columns = [
        "date", "Yield_Product_A", "flow_rate_1", "PRODUCTION_1", 
        "Output_Value", "Temperature", "Pressure", "yield_product_b"
    ]
    
    print(f"Test columns: {test_columns}")
    
    # Test the filtering logic
    target_cols = [c for c in test_columns if c != "date" and (
        "yield" in c.lower() or 
        "flow" in c.lower() or 
        "rate" in c.lower() or
        "production" in c.lower() or
        "output" in c.lower()
    )]
    
    print(f"Found target columns: {target_cols}")

if __name__ == "__main__":
    print("🔍 ETL Debug Script")
    print("=" * 50)
    
    # Test target filtering logic
    test_target_filtering()
    
    # Debug a specific file (update this path)
    test_file = "./sample_data.xlsx"  # Update this to your actual Excel file path
    
    print(f"\n🔍 Debugging file: {test_file}")
    debug_excel_file(test_file)
    
    print("\n💡 To fix the empty target issue:")
    print("1. Make sure the Excel file exists and has the expected sheets")
    print("2. Check that the 'TS Monitoring Tags' sheet has the right structure")
    print("3. Verify that the Tag column contains values with 'yield', 'flow', etc.")
    print("4. Update the file path in the script to point to your actual Excel file")

