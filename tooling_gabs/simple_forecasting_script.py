# -*- coding: utf-8 -*-
"""
Simplified Forecasting Script - EBM Focus
A lightweight version that focuses on EBM explanations with simple forecasting
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from interpret.glassbox import ExplainableBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data"""
    print("🔍 Loading data...")
    
    file_path = 'final_concatenated_data_mice_imputed.csv'
    
    # Target columns
    target_cols = [
        'Total_Stabilized_Naphtha_Product_Flowrate',
        'Total_Kerosene_Product_Flowrate',
        'Jet_Fuel_Product_Train1_Flowrate',
        'Total_Light_Diesel_Product_Flowrate',
        'Total_Heavy_Diesel_Product_Flowrate',
        'Total_Atmospheric_Residue_Flowrate',
        'Blend_Yield_Gas & LPG',
        'Blend_Yield_Kerosene',
        'Blend_Yield_Light Diesel',
        'Blend_Yield_Heavy Diesel',
        'Blend_Yield_RCO'
    ]
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Handle dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    # Convert to numeric
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    df = df.fillna(0)
    
    # Define features
    feature_cols = [col for col in df.columns if col != 'date' and col not in target_cols]
    static_cols = [col for col in feature_cols if col.startswith('crude_') or col in ['API', 'Sulphur', 'blend_id']]
    ts_cols = [col for col in feature_cols if col not in static_cols]
    
    print(f"✅ Data prepared: {len(df)} rows")
    print(f"   - Target columns: {len(target_cols)}")
    print(f"   - Static columns: {len(static_cols)}")
    print(f"   - Time series columns: {len(ts_cols)}")
    
    return df, target_cols, static_cols, ts_cols

def create_time_series_features(df, target_cols, ts_cols, static_cols, lookback=7):
    """Create time series features for forecasting"""
    print(f"🔧 Creating time series features (lookback={lookback})...")
    
    # Create lagged features
    for col in ts_cols + target_cols:
        for lag in range(1, lookback + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Create rolling statistics
    for col in target_cols:
        df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7).mean()
        df[f'{col}_rolling_std_7'] = df[col].rolling(window=7).std()
    
    # Drop rows with NaN values from lagged features
    df = df.dropna()
    
    print(f"✅ Created features: {len(df)} rows after feature engineering")
    return df

def train_forecasting_models(df, target_cols, static_cols, ts_cols):
    """Train simple forecasting models"""
    print("🚀 Training forecasting models...")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in target_cols + ['date']]
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Train models for each target
    models = {}
    results = {}
    
    for i, target_name in enumerate(target_cols):
        print(f"  Training model for {target_name}...")
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train_scaled, y_train_scaled[:, i])
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train_scaled[:, i])
        
        # Predictions
        rf_pred = rf.predict(X_test_scaled)
        lr_pred = lr.predict(X_test_scaled)
        
        # Ensemble (average)
        ensemble_pred = (rf_pred + lr_pred) / 2
        
        # Evaluate
        rf_r2 = r2_score(y_test_scaled[:, i], rf_pred)
        lr_r2 = r2_score(y_test_scaled[:, i], lr_pred)
        ensemble_r2 = r2_score(y_test_scaled[:, i], ensemble_pred)
        
        models[target_name] = {
            'rf': rf,
            'lr': lr,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }
        
        results[target_name] = {
            'rf_r2': rf_r2,
            'lr_r2': lr_r2,
            'ensemble_r2': ensemble_r2,
            'best_model': 'rf' if rf_r2 > lr_r2 else 'lr'
        }
        
        print(f"    RF R²: {rf_r2:.3f}, LR R²: {lr_r2:.3f}, Ensemble R²: {ensemble_r2:.3f}")
    
    print("✅ Forecasting models trained!")
    return models, results

def train_ebm_models(df, target_cols, static_cols):
    """Train EBM models for explanations"""
    print("🔍 Training EBM models...")
    
    # Prepare data
    X_static = df[static_cols].values
    y_targets = df[target_cols].values
    
    ebm_models = {}
    ebm_results = {}
    
    for i, target_name in enumerate(target_cols):
        print(f"  Training EBM for {target_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_static, y_targets[:, i], test_size=0.2, random_state=42
        )
        
        # Train EBM
        ebm = ExplainableBoostingRegressor(
            interactions=3,  # Reduced for speed
            max_bins=64,
            max_interaction_bins=8,
            random_state=42
        )
        
        ebm.fit(X_train, y_train)
        
        # Evaluate
        y_pred = ebm.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        ebm_models[target_name] = ebm
        ebm_results[target_name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"    R² = {r2:.3f}, MAE = {mae:.3f}")
    
    print("✅ EBM models trained!")
    return ebm_models, ebm_results

def create_visualizations(ebm_models, static_cols, target_cols, ebm_results):
    """Create visualizations"""
    print("📊 Creating visualizations...")
    
    # Plot EBM feature importance
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, target_name in enumerate(target_cols[:6]):
        ax = axes[i]
        ebm = ebm_models[target_name]
        
        # Get feature importance
        importance = ebm.feature_importances_
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1][:10]
        
        # Plot
        ax.barh(range(len(sorted_idx)), importance[sorted_idx])
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([static_cols[idx] for idx in sorted_idx])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{target_name}\nR² = {ebm_results[target_name]["r2"]:.3f}')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('./ebm_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved!")

def main():
    """Main execution"""
    print("="*60)
    print("SIMPLIFIED FORECASTING + EBM ANALYSIS")
    print("="*60)
    
    try:
        # Step 1: Load data
        df, target_cols, static_cols, ts_cols = load_and_prepare_data()
        
        # Step 2: Create time series features
        df = create_time_series_features(df, target_cols, ts_cols, static_cols)
        
        # Step 3: Train forecasting models
        models, results = train_forecasting_models(df, target_cols, static_cols, ts_cols)
        
        # Step 4: Train EBM models
        ebm_models, ebm_results = train_ebm_models(df, target_cols, static_cols)
        
        # Step 5: Create visualizations
        create_visualizations(ebm_models, static_cols, target_cols, ebm_results)
        
        # Step 6: Print results
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        print("\n📊 Forecasting Performance:")
        for target_name, result in results.items():
            print(f"  {target_name}:")
            print(f"    Best R² = {result['ensemble_r2']:.3f}")
        
        print("\n🔍 EBM Model Performance:")
        for target_name, result in ebm_results.items():
            print(f"  {target_name}: R² = {result['r2']:.3f}")
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
