import pandas as pd
import numpy as np

# Load data and check Total_Stabilized_Naphtha_Product_Flowrate
print("Loading data...")
df = pd.read_csv('final_concatenated_data_mice_imputed.csv')

target_col = 'Total_Stabilized_Naphtha_Product_Flowrate'
print(f"Target column exists: {target_col in df.columns}")

if target_col in df.columns:
    data = df[target_col]
    print(f"Data type: {data.dtype}")
    print(f"Non-null values: {data.count()}")
    print(f"Null values: {data.isnull().sum()}")
    print(f"Zero values: {(data == 0).sum()}")
    print(f"Negative values: {(data < 0).sum()}")
    print(f"Mean: {data.mean():.3f}")
    print(f"Std: {data.std():.3f}")
    print(f"Min: {data.min():.3f}")
    print(f"Max: {data.max():.3f}")
    print(f"Skewness: {data.skew():.3f}")
    
    # Check for outliers
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
    print(f"Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
    
    # Check static columns
    static_cols = [col for col in df.columns if col.startswith('crude_') or col in ['API', 'Sulphur', 'blend_id']]
    print(f"Static columns available: {len(static_cols)}")
    
    # Check correlations
    correlations = {}
    for col in static_cols[:10]:  # Check first 10
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            corr = df[col].corr(data)
            if not np.isnan(corr):
                correlations[col] = corr
    
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top correlations:")
    for feature, corr in sorted_corrs[:5]:
        print(f"  {feature}: {corr:.3f}")
else:
    print("Target column not found!")
    print("Available columns:", df.columns.tolist()[:10])

