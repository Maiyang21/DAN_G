# -*- coding: utf-8 -*-
"""
Quick XGBoost Test - Fast Version to Avoid Training Trap
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
import warnings

# XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data (fast version)"""
    print("🔍 Loading data...")
    
    file_path = 'final_concatenated_data_mice_imputed.csv'
    
    # Target columns (reduced for testing)
    target_cols = [
        'Total_Stabilized_Naphtha_Product_Flowrate',
        'Total_Kerosene_Product_Flowrate',
        'Jet_Fuel_Product_Train1_Flowrate'
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
    
    # Handle missing values - fill target features with mean first
    print("🧹 Handling missing values...")
    original_cols = len(df.columns)
    
    # Fill target features NaN values with mean
    print("  Filling target features NaN values with mean...")
    for col in target_cols:
        if col in df.columns:
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                mean_value = df[col].mean()
                df[col] = df[col].fillna(mean_value)
                print(f"    - {col}: {nan_count} NaN values filled with mean {mean_value:.3f}")
    
    # Calculate NaN percentage for each column (after target filling)
    nan_percentages = (df.isnull().sum() / len(df)) * 100
    
    # Identify columns with >30% NaN values
    high_nan_cols = nan_percentages[nan_percentages > 30].index.tolist()
    
    if high_nan_cols:
        print(f"  Dropping {len(high_nan_cols)} columns with >30% NaN values")
        df = df.drop(columns=high_nan_cols)
    
    # Fill remaining NaN values with 0
    df = df.fillna(0)
    
    print(f"  Columns: {original_cols} → {len(df.columns)} (dropped {original_cols - len(df.columns)})")
    
    # Define features
    feature_cols = [col for col in df.columns if col != 'date' and col not in target_cols]
    static_cols = [col for col in feature_cols if col.startswith('crude_') or col in ['API', 'Sulphur', 'blend_id']]
    ts_cols = [col for col in feature_cols if col not in static_cols]
    
    print(f"✅ Data prepared: {len(df)} rows")
    print(f"   - Target columns: {len(target_cols)}")
    print(f"   - Static columns: {len(static_cols)}")
    print(f"   - Time series columns: {len(ts_cols)}")
    
    return df, target_cols, static_cols, ts_cols

def create_time_series_features(df, target_cols, ts_cols, static_cols, lookback=3):
    """Create time series features (reduced lookback for speed)"""
    print(f"🔧 Creating time series features (lookback={lookback})...")
    
    # Create lagged features (reduced)
    for col in ts_cols + target_cols:
        for lag in range(1, lookback + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Create rolling statistics (reduced)
    for col in target_cols:
        df[f'{col}_rolling_mean_3'] = df[col].rolling(window=3).mean()
        df[f'{col}_rolling_std_3'] = df[col].rolling(window=3).std()
    
    # Drop rows with NaN values from lagged features
    df = df.dropna()
    
    print(f"✅ Created features: {len(df)} rows after feature engineering")
    return df

def calculate_metrics(y_true, y_pred):
    """Calculate basic evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def train_xgboost_fast(X_train, y_train, X_test, y_test, target_name):
    """Train XGBoost model with minimal hyperparameter tuning"""
    if not XGBOOST_AVAILABLE:
        return None, None
    
    print(f"    Training XGBoost for {target_name}...")
    
    # Minimal parameter grid for speed
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6],
        'learning_rate': [0.1]
    }
    
    # Create XGBoost regressor
    xgb_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=1,
        eval_metric='rmse'
    )
    
    # Use GridSearchCV with minimal CV
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid, 
        cv=2,  # Reduced CV folds
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=0
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_xgb = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_xgb.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    print(f"      XGBoost - R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.3f}")
    
    return best_xgb, metrics

def train_l1_l2_fast(X_train, y_train, X_test, y_test, target_name):
    """Train L1/L2 models with minimal hyperparameter tuning"""
    print(f"    Training L1/L2 models for {target_name}...")
    
    models = {}
    metrics = {}
    
    # L1 Regularization (Lasso) - reduced parameters
    lasso_params = {'alpha': [0.1, 1.0]}
    lasso = Lasso(random_state=42, max_iter=1000)
    from sklearn.model_selection import GridSearchCV
    lasso_grid = GridSearchCV(lasso, lasso_params, cv=2, scoring='neg_mean_squared_error', n_jobs=1)
    lasso_grid.fit(X_train, y_train)
    lasso_best = lasso_grid.best_estimator_
    lasso_pred = lasso_best.predict(X_test)
    models['lasso'] = lasso_best
    metrics['lasso'] = calculate_metrics(y_test, lasso_pred)
    
    # L2 Regularization (Ridge) - reduced parameters
    ridge_params = {'alpha': [0.1, 1.0]}
    ridge = Ridge(random_state=42, max_iter=1000)
    ridge_grid = GridSearchCV(ridge, ridge_params, cv=2, scoring='neg_mean_squared_error', n_jobs=1)
    ridge_grid.fit(X_train, y_train)
    ridge_best = ridge_grid.best_estimator_
    ridge_pred = ridge_best.predict(X_test)
    models['ridge'] = ridge_best
    metrics['ridge'] = calculate_metrics(y_test, ridge_pred)
    
    # ElasticNet - reduced parameters
    elastic_params = {'alpha': [0.1, 1.0], 'l1_ratio': [0.5]}
    elastic = ElasticNet(random_state=42, max_iter=1000)
    elastic_grid = GridSearchCV(elastic, elastic_params, cv=2, scoring='neg_mean_squared_error', n_jobs=1)
    elastic_grid.fit(X_train, y_train)
    elastic_best = elastic_grid.best_estimator_
    elastic_pred = elastic_best.predict(X_test)
    models['elasticnet'] = elastic_best
    metrics['elasticnet'] = calculate_metrics(y_test, elastic_pred)
    
    print(f"      Lasso - R²: {metrics['lasso']['r2']:.3f}, RMSE: {metrics['lasso']['rmse']:.3f}")
    print(f"      Ridge - R²: {metrics['ridge']['r2']:.3f}, RMSE: {metrics['ridge']['rmse']:.3f}")
    print(f"      ElasticNet - R²: {metrics['elasticnet']['r2']:.3f}, RMSE: {metrics['elasticnet']['rmse']:.3f}")
    
    return models, metrics

def train_baseline_fast(X_train, y_train, X_test, y_test, target_name):
    """Train baseline models quickly"""
    print(f"    Training baseline models for {target_name}...")
    
    models = {}
    metrics = {}
    
    # Random Forest (reduced trees)
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    models['random_forest'] = rf
    metrics['random_forest'] = calculate_metrics(y_test, rf_pred)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    models['linear_regression'] = lr
    metrics['linear_regression'] = calculate_metrics(y_test, lr_pred)
    
    print(f"      Random Forest - R²: {metrics['random_forest']['r2']:.3f}, RMSE: {metrics['random_forest']['rmse']:.3f}")
    print(f"      Linear Regression - R²: {metrics['linear_regression']['r2']:.3f}, RMSE: {metrics['linear_regression']['rmse']:.3f}")
    
    return models, metrics

def main():
    """Main execution - fast version"""
    print("="*60)
    print("XGBOOST QUICK TEST - AVOIDING TRAINING TRAP")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Step 1: Load and prepare data (reduced targets)
        df, target_cols, static_cols, ts_cols = load_and_prepare_data()
        
        # Step 2: Create time series features (reduced lookback)
        df = create_time_series_features(df, target_cols, ts_cols, static_cols, lookback=3)
        
        # Step 3: Prepare features
        feature_cols = [col for col in df.columns if col not in target_cols + ['date']]
        X = df[feature_cols].values
        y = df[target_cols].values
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Step 5: Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Step 6: Train models for each target (sequential for speed)
        all_results = {}
        
        for i, target_name in enumerate(target_cols):
            print(f"\n🎯 Training models for {target_name}...")
            
            # Scale target
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train[:, i].reshape(-1, 1)).flatten()
            y_test_scaled = scaler_y.transform(y_test[:, i].reshape(-1, 1)).flatten()
            
            target_results = {}
            
            # Train XGBoost
            if XGBOOST_AVAILABLE:
                xgb_model, xgb_metrics = train_xgboost_fast(
                    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, target_name
                )
                if xgb_model is not None:
                    target_results['xgboost'] = xgb_metrics
            
            # Train L1/L2 models
            l1_l2_models, l1_l2_metrics = train_l1_l2_fast(
                X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, target_name
            )
            target_results.update(l1_l2_metrics)
            
            # Train baseline models
            baseline_models, baseline_metrics = train_baseline_fast(
                X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, target_name
            )
            target_results.update(baseline_metrics)
            
            all_results[target_name] = target_results
        
        # Step 7: Print results
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("QUICK TEST RESULTS")
        print("="*60)
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        
        print("\n📊 Model Performance Summary:")
        print("-" * 60)
        
        for target_name, results in all_results.items():
            print(f"\n{target_name}:")
            for model_name, metrics in results.items():
                print(f"  {model_name:15s}: R² = {metrics['r2']:6.3f}, RMSE = {metrics['rmse']:6.3f}")
        
        print("\n✅ Quick test completed successfully!")
        print("   This version avoids the training trap by using:")
        print("   • Reduced targets (3 instead of 11)")
        print("   • Reduced lookback (3 instead of 7)")
        print("   • Minimal hyperparameter grids")
        print("   • Reduced cross-validation folds")
        print("   • Sequential training instead of parallel")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

