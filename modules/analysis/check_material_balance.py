import pandas as pd
import os

def check_material_balance_sheet():
    """
    Check the Material Balance sheets from the new Excel file to understand targets
    """
    # The new Excel file path
    excel_path = r"C:\Users\PC\Documents\abu\Documents\DATABASE\PROC_OPTIM\Monthly Reports\CDU Monthly Report - February 2025.xlsx"
    
    print(f"🔍 Checking Excel file path: {excel_path}")
    print("=" * 80)
    
    # Check if file exists
    if not os.path.exists(excel_path):
        print(f"❌ ERROR: File does not exist at this path!")
        print(f"   Path: {excel_path}")
        return False
    
    print(f"✅ File exists!")
    
    try:
        # List all sheets
        excel_file = pd.ExcelFile(excel_path)
        print(f"\n📊 Available sheets: {excel_file.sheet_names}")
        
        # Check if Material Balance sheets exist
        material_balance_sheets = ['Material Balance Full', 'Material Balance Overview']
        found_sheets = [sheet for sheet in material_balance_sheets if sheet in excel_file.sheet_names]
        
        if not found_sheets:
            print(f"❌ ERROR: No material balance sheets found!")
            print(f"Available sheets: {excel_file.sheet_names}")
            return False
        
        print(f"✅ Found material balance sheets: {found_sheets}")
        
        # Read each material balance sheet
        for sheet_name in found_sheets:
            print(f"\n🔍 Examining '{sheet_name}' sheet:")
            print("-" * 50)
            
            mb_df = pd.read_excel(excel_path, sheet_name=sheet_name)
            print(f"Shape: {mb_df.shape}")
            print(f"Columns: {list(mb_df.columns)}")
            
            print(f"\nFirst 10 rows:")
            print(mb_df.head(10))
            
            print(f"\nData types:")
            print(mb_df.dtypes)
            
            # Check for empty columns
            empty_cols = mb_df.columns[mb_df.isna().all()].tolist()
            if empty_cols:
                print(f"\n⚠️  Empty columns: {empty_cols}")
            
            # Look for target-related information
            print(f"\n🔍 Looking for target-related information:")
            
            # Check if there are any rows with "Target" or similar keywords
            target_keywords = ['target', 'yield', 'flow', 'production', 'output', 'rate']
            
            for idx, row in mb_df.iterrows():
                row_str = ' '.join(str(val).lower() for val in row if pd.notna(val))
                for keyword in target_keywords:
                    if keyword in row_str:
                        print(f"Row {idx}: Found '{keyword}' - {row.tolist()}")
            
            # Check for specific columns that might contain targets
            print(f"\n🔍 Checking for specific target columns:")
            for col in mb_df.columns:
                col_str = str(col).lower()
                if any(keyword in col_str for keyword in target_keywords):
                    print(f"Column '{col}': Potential target column")
                    # Show unique values in this column
                    unique_vals = mb_df[col].dropna().unique()
                    print(f"  Unique values: {unique_vals[:10]}...")
            
            # Look for material balance specific terms
            balance_keywords = ['feed', 'product', 'yield', 'loss', 'balance', 'inlet', 'outlet']
            print(f"\n🔍 Looking for material balance terms:")
            
            for idx, row in mb_df.iterrows():
                row_str = ' '.join(str(val).lower() for val in row if pd.notna(val))
                for keyword in balance_keywords:
                    if keyword in row_str:
                        print(f"Row {idx}: Found '{keyword}' - {row.tolist()}")
            
            # Check if there are any tag references (like 101FI1402, etc.)
            print(f"\n🔍 Looking for tag references:")
            tag_patterns = ['101FI', '101TI', '102FI', '102TI', '103FI', '103TI', '104FI', '104TI']
            
            for idx, row in mb_df.iterrows():
                row_str = ' '.join(str(val) for val in row if pd.notna(val))
                for pattern in tag_patterns:
                    if pattern in row_str:
                        print(f"Row {idx}: Found tag pattern '{pattern}' - {row.tolist()}")
            
            print(f"\n" + "="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Material Balance Sheet Checker")
    print("=" * 80)
    
    success = check_material_balance_sheet()
    
    if not success:
        print(f"\n💡 To fix the target identification issue:")
        print("1. Make sure the Excel file exists at the specified path")
        print("2. Check that the material balance sheets have the right structure")
        print("3. Look for target-related information in the sheets")
        print("4. Identify which tags constitute the targets")
