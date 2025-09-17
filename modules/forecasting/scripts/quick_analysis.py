# Quick EBM Analysis - Lightweight Version
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from interpret.glassbox import ExplainableBoostingRegressor
import matplotlib.pyplot as plt

print("🚀 Starting Quick Analysis...")

# Load data
df = pd.read_csv('final_concatenated_data_mice_imputed.csv')
print(f"✅ Loaded {len(df)} rows")

# Target columns
targets = [
    'Total_Stabilized_Naphtha_Product_Flowrate',
    'Total_Kerosene_Product_Flowrate', 
    'Jet_Fuel_Product_Train1_Flowrate',
    'Total_Light_Diesel_Product_Flowrate',
    'Total_Heavy_Diesel_Product_Flowrate',
    'Total_Atmospheric_Residue_Flowrate'
]

# Static features (blend characteristics)
static_cols = [col for col in df.columns if col.startswith('crude_') or col in ['API', 'Sulphur']]

# Clean data
df = df.fillna(0)
print(f"✅ Data cleaned")

# Train simple models
print("🔍 Training EBM models...")
for target in targets[:3]:  # Just first 3 targets
    if target in df.columns:
        print(f"  {target}...")
        
        # Simple features
        X = df[static_cols].fillna(0)
        y = df[target].fillna(0)
        
        # Train EBM
        ebm = ExplainableBoostingRegressor(random_state=42)
        ebm.fit(X, y)
        
        # Predict
        y_pred = ebm.predict(X)
        r2 = r2_score(y, y_pred)
        
        print(f"    R² = {r2:.3f}")

print("✅ Done! Check the results above.")

