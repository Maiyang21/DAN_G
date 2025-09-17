import pandas as pd
import os

def check_blend_yields():
    excel_path = r"C:\Users\PC\Documents\abu\Documents\DATABASE\PROC_OPTIM\Monthly Reports\TS Average, Blend, and Lab Data February 2025.xlsx"
    
    if not os.path.exists(excel_path):
        print(f"❌ File not found: {excel_path}")
        return
    
    print(f"📊 Examining blend data from: {excel_path}")
    
    try:
        # Read the Blends sheet
        blend_df = pd.read_excel(excel_path, sheet_name="Blends")
        
        print(f"\n📋 Blends sheet shape: {blend_df.shape}")
        print(f"📋 Blends sheet columns: {list(blend_df.columns)}")
        
        # Show the first few rows more clearly
        print(f"\n📊 First 10 rows of blend data:")
        for i in range(min(10, len(blend_df))):
            print(f"Row {i}: {blend_df.iloc[i, :5].tolist()}")
        
        # Check the structure more carefully
        print(f"\n🔍 Detailed structure analysis:")
        print(f"Row 0 (API): {blend_df.iloc[0, :5].tolist()}")
        print(f"Row 1 (Sulphur): {blend_df.iloc[1, :5].tolist()}")
        print(f"Row 2: {blend_df.iloc[2, :5].tolist()}")
        print(f"Row 3: {blend_df.iloc[3, :5].tolist()}")
        print(f"Row 4: {blend_df.iloc[4, :5].tolist()}")
        
        # Look for any percentage or yield-like data
        print(f"\n🔍 Checking for percentage/yield data in blend compositions...")
        crude_compositions = blend_df.iloc[3:, :]
        print(f"Crude composition rows: {crude_compositions.shape}")
        
        # Show crude types (first column)
        crude_types = crude_compositions.iloc[:, 0].dropna().tolist()
        print(f"Crude types found: {crude_types[:10]}...")
        
        # Check if any values look like percentages (0-100 range)
        sample_values = crude_compositions.iloc[:, 1:6].values.flatten()
        sample_values = [v for v in sample_values if pd.notna(v)]
        if sample_values:
            print(f"Sample composition values: {sample_values[:10]}")
            percent_like = [v for v in sample_values if isinstance(v, (int, float)) and 0 <= v <= 1]
            print(f"Values in 0-1 range (potential yields as decimals): {len(percent_like)} out of {len(sample_values)}")
            
            # Check if these sum to 1.0 (indicating they are yields)
            blend_sums = []
            for col in blend_df.columns[1:]:  # Skip 'Blend No' column
                col_values = crude_compositions[col].dropna()
                if len(col_values) > 0:
                    col_sum = col_values.sum()
                    blend_sums.append((col, col_sum))
            
            print(f"\n🔍 Blend composition sums (should be close to 1.0 if yields):")
            for blend, total in blend_sums[:10]:  # Show first 10
                print(f"  {blend}: {total:.3f}")
        
        # Check if there are any explicit yield columns
        print(f"\n🔍 Looking for explicit yield columns...")
        all_columns = [col.lower() for col in blend_df.columns]
        yield_keywords = ['yield', 'yld', 'product', 'output', 'fraction']
        yield_cols = []
        for col in blend_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in yield_keywords):
                yield_cols.append(col)
        
        if yield_cols:
            print(f"✅ Found potential yield columns: {yield_cols}")
        else:
            print("⚠️ No explicit yield columns found")
            
    except Exception as e:
        print(f"❌ Error reading blend data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_blend_yields()

