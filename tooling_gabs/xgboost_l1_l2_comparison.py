# -*- coding: utf-8 -*-
"""
XGBoost and L1/L2 Regression Comparison Script
Comprehensive performance comparison with existing Random Forest and Linear Regression models
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from scipy import stats

# XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data (same as original script)"""
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
        print(f"  Dropping {len(high_nan_cols)} columns with >30% NaN values:")
        for col in high_nan_cols:
            print(f"    - {col}: {nan_percentages[col]:.1f}% NaN")
        df = df.drop(columns=high_nan_cols)
    
    # Fill remaining NaN values with 0
    remaining_nan = df.isnull().sum().sum()
    if remaining_nan > 0:
        print(f"  Filling {remaining_nan} remaining NaN values with 0...")
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

def calculate_comprehensive_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    rmae = math.sqrt(mae)  # Root Mean Absolute Error
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'rmae': rmae,
        'r2': r2,
        'mape': mape,
        'max_error': max_error
    }

def train_xgboost_model(X_train, y_train, X_test, y_test, target_name):
    """Train XGBoost model with hyperparameter tuning"""
    if not XGBOOST_AVAILABLE:
        return None, None
    
    print(f"    Training XGBoost for {target_name}...")
    
    # Define parameter grid for hyperparameter tuning (optimized for speed)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.9, 1.0]
    }
    
    # Create XGBoost regressor
    xgb_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=1,  # Use 1 job for parallel processing compatibility
        eval_metric='rmse'
    )
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid, 
        cv=3, 
        scoring='neg_mean_squared_error',
        n_jobs=1,  # Use 1 job for parallel processing compatibility
        verbose=0
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_xgb = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_xgb.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y_test, y_pred)
    
    print(f"      XGBoost - R²: {metrics['r2']:.3f}, RMAE: {metrics['rmae']:.3f}")
    
    return best_xgb, metrics

def train_l1_l2_models(X_train, y_train, X_test, y_test, target_name):
    """Train L1 (Lasso), L2 (Ridge), and ElasticNet models"""
    print(f"    Training L1/L2 models for {target_name}...")
    
    models = {}
    metrics = {}
    
    # L1 Regularization (Lasso)
    lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
    lasso = Lasso(random_state=42, max_iter=2000)
    lasso_grid = GridSearchCV(lasso, lasso_params, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
    lasso_grid.fit(X_train, y_train)
    lasso_best = lasso_grid.best_estimator_
    lasso_pred = lasso_best.predict(X_test)
    models['lasso'] = lasso_best
    metrics['lasso'] = calculate_comprehensive_metrics(y_test, lasso_pred)
    
    # L2 Regularization (Ridge)
    ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge(random_state=42, max_iter=2000)
    ridge_grid = GridSearchCV(ridge, ridge_params, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
    ridge_grid.fit(X_train, y_train)
    ridge_best = ridge_grid.best_estimator_
    ridge_pred = ridge_best.predict(X_test)
    models['ridge'] = ridge_best
    metrics['ridge'] = calculate_comprehensive_metrics(y_test, ridge_pred)
    
    # ElasticNet (L1 + L2)
    elastic_params = {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    elastic = ElasticNet(random_state=42, max_iter=2000)
    elastic_grid = GridSearchCV(elastic, elastic_params, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
    elastic_grid.fit(X_train, y_train)
    elastic_best = elastic_grid.best_estimator_
    elastic_pred = elastic_best.predict(X_test)
    models['elasticnet'] = elastic_best
    metrics['elasticnet'] = calculate_comprehensive_metrics(y_test, elastic_pred)
    
    print(f"      Lasso - R²: {metrics['lasso']['r2']:.3f}, RMAE: {metrics['lasso']['rmae']:.3f}")
    print(f"      Ridge - R²: {metrics['ridge']['r2']:.3f}, RMAE: {metrics['ridge']['rmae']:.3f}")
    print(f"      ElasticNet - R²: {metrics['elasticnet']['r2']:.3f}, RMAE: {metrics['elasticnet']['rmae']:.3f}")
    
    return models, metrics

def train_baseline_models(X_train, y_train, X_test, y_test, target_name):
    """Train baseline models (Random Forest and Linear Regression) for comparison"""
    print(f"    Training baseline models for {target_name}...")
    
    models = {}
    metrics = {}
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    models['random_forest'] = rf
    metrics['random_forest'] = calculate_comprehensive_metrics(y_test, rf_pred)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    models['linear_regression'] = lr
    metrics['linear_regression'] = calculate_comprehensive_metrics(y_test, lr_pred)
    
    print(f"      Random Forest - R²: {metrics['random_forest']['r2']:.3f}, RMAE: {metrics['random_forest']['rmae']:.3f}")
    print(f"      Linear Regression - R²: {metrics['linear_regression']['r2']:.3f}, RMAE: {metrics['linear_regression']['rmae']:.3f}")
    
    return models, metrics

def train_all_models_parallel(args):
    """Train all models for a single target in parallel"""
    target_name, target_idx, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = args
    
    print(f"  Training all models for {target_name} on core {os.getpid()}...")
    
    all_models = {}
    all_metrics = {}
    
    # Train baseline models
    baseline_models, baseline_metrics = train_baseline_models(
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, target_name
    )
    all_models.update(baseline_models)
    all_metrics.update(baseline_metrics)
    
    # Train L1/L2 models
    l1_l2_models, l1_l2_metrics = train_l1_l2_models(
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, target_name
    )
    all_models.update(l1_l2_models)
    all_metrics.update(l1_l2_metrics)
    
    # Train XGBoost model
    if XGBOOST_AVAILABLE:
        xgb_model, xgb_metrics = train_xgboost_model(
            X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, target_name
        )
        if xgb_model is not None:
            all_models['xgboost'] = xgb_model
            all_metrics['xgboost'] = xgb_metrics
    
    # Create ensemble model (average of all models)
    ensemble_pred = np.mean([
        all_models[model].predict(X_test_scaled) 
        for model in all_models.keys()
    ], axis=0)
    all_metrics['ensemble'] = calculate_comprehensive_metrics(y_test_scaled, ensemble_pred)
    
    # Store scalers for forecasting
    all_models['scaler_X'] = scaler_X
    all_models['scaler_y'] = scaler_y
    
    print(f"      {target_name} - Best R²: {max([m['r2'] for m in all_metrics.values()]):.3f}")
    
    return target_name, all_models, all_metrics

def train_all_models(df, target_cols, n_jobs=None):
    """Train all models in parallel across multiple targets"""
    print("🚀 Training all models in parallel...")
    
    if n_jobs is None:
        n_jobs = cpu_count()
    
    print(f"   Using {n_jobs} CPU cores for parallel processing")
    print(f"   Training models for {len(target_cols)} targets")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in target_cols + ['date']]
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale targets - each target gets its own scaler
    scaler_y_dict = {}
    y_train_scaled = np.zeros_like(y_train)
    y_test_scaled = np.zeros_like(y_test)
    
    for i, target_name in enumerate(target_cols):
        scaler_y = StandardScaler()
        scaler_y.fit(y_train[:, i].reshape(-1, 1))
        scaler_y_dict[target_name] = scaler_y
        
        y_train_scaled[:, i] = scaler_y.transform(y_train[:, i].reshape(-1, 1)).flatten()
        y_test_scaled[:, i] = scaler_y.transform(y_test[:, i].reshape(-1, 1)).flatten()
    
    # Create chunks of targets for parallel processing
    chunk_size = max(1, len(target_cols) // n_jobs)
    target_chunks = []
    target_indices_chunks = []
    
    for i in range(0, len(target_cols), chunk_size):
        chunk = target_cols[i:i + chunk_size]
        indices = list(range(i, min(i + chunk_size, len(target_cols))))
        target_chunks.append(chunk)
        target_indices_chunks.append(indices)
    
    # Ensure we don't exceed available cores
    target_chunks = target_chunks[:n_jobs]
    target_indices_chunks = target_indices_chunks[:n_jobs]
    
    print(f"   Created {len(target_chunks)} chunks:")
    for i, chunk in enumerate(target_chunks):
        print(f"     Core {i+1}: {len(chunk)} targets - {chunk}")
    
    # Prepare arguments for parallel processing
    args_list = []
    for chunk, indices in zip(target_chunks, target_indices_chunks):
        for target_name, target_idx in zip(chunk, indices):
            args_list.append((
                target_name, target_idx, X_train_scaled, X_test_scaled,
                y_train_scaled, y_test_scaled, scaler_X, scaler_y_dict[target_name]
            ))
    
    # Train models in parallel
    start_time = time.time()
    with Pool(processes=len(target_chunks)) as pool:
        results = pool.map(train_all_models_parallel, args_list)
    
    training_time = time.time() - start_time
    print(f"✅ All models trained in {training_time:.2f} seconds!")
    
    # Organize results
    all_models = {}
    all_metrics = {}
    
    for target_name, models, metrics in results:
        all_models[target_name] = models
        all_metrics[target_name] = metrics
    
    return all_models, all_metrics

def create_performance_comparison_plot(all_metrics, target_cols):
    """Create comprehensive performance comparison visualization"""
    print("📊 Creating performance comparison plots...")
    
    # Prepare data for plotting
    model_names = ['random_forest', 'linear_regression', 'lasso', 'ridge', 'elasticnet']
    if XGBOOST_AVAILABLE:
        model_names.append('xgboost')
    model_names.append('ensemble')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Model Performance Comparison: XGBoost vs L1/L2 vs Baseline Models', fontsize=16)
    
    # Metrics to compare
    metrics_to_plot = ['r2', 'mae', 'rmse', 'rmae', 'mape', 'max_error']
    metric_names = ['R² Score', 'MAE', 'RMSE', 'RMAE', 'MAPE (%)', 'Max Error']
    
    for i, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Collect data for this metric
        data_for_metric = {model: [] for model in model_names}
        
        for target_name in target_cols:
            if target_name in all_metrics:
                for model_name in model_names:
                    if model_name in all_metrics[target_name]:
                        data_for_metric[model_name].append(all_metrics[target_name][model_name][metric])
        
        # Create box plot
        box_data = [data_for_metric[model] for model in model_names if data_for_metric[model]]
        box_labels = [model for model in model_names if data_for_metric[model]]
        
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
                     'lightpink', 'lightgray', 'lightcyan']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            ax.set_title(f'{metric_name} Distribution')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_performance_table(all_metrics, target_cols):
    """Create detailed performance comparison table"""
    print("📋 Creating detailed performance table...")
    
    model_names = ['random_forest', 'linear_regression', 'lasso', 'ridge', 'elasticnet']
    if XGBOOST_AVAILABLE:
        model_names.append('xgboost')
    model_names.append('ensemble')
    
    # Create comprehensive results table
    results_data = []
    
    for target_name in target_cols:
        if target_name in all_metrics:
            for model_name in model_names:
                if model_name in all_metrics[target_name]:
                    metrics = all_metrics[target_name][model_name]
                    results_data.append({
                        'Target': target_name,
                        'Model': model_name,
                        'R²': metrics['r2'],
                        'MAE': metrics['mae'],
                        'RMSE': metrics['rmse'],
                        'RMAE': metrics['rmae'],
                        'MAPE': metrics['mape'],
                        'Max Error': metrics['max_error']
                    })
    
    results_df = pd.DataFrame(results_data)
    
    # Save to CSV
    results_df.to_csv('./model_performance_comparison.csv', index=False)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE COMPARISON")
    print("="*80)
    
    # Group by model and calculate average performance
    model_summary = results_df.groupby('Model').agg({
        'R²': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'RMAE': ['mean', 'std'],
        'MAPE': ['mean', 'std']
    }).round(3)
    
    print("\nAverage Performance by Model:")
    print("-" * 50)
    for model in model_names:
        if model in model_summary.index:
            r2_mean = model_summary.loc[model, ('R²', 'mean')]
            r2_std = model_summary.loc[model, ('R²', 'std')]
            mae_mean = model_summary.loc[model, ('MAE', 'mean')]
            mae_std = model_summary.loc[model, ('MAE', 'std')]
            print(f"{model:15s}: R² = {r2_mean:.3f} ± {r2_std:.3f}, MAE = {mae_mean:.3f} ± {mae_std:.3f}")
    
    # Find best performing model for each target
    print("\nBest Model for Each Target:")
    print("-" * 50)
    for target_name in target_cols:
        if target_name in all_metrics:
            target_results = results_df[results_df['Target'] == target_name]
            best_r2_idx = target_results['R²'].idxmax()
            best_model = target_results.loc[best_r2_idx, 'Model']
            best_r2 = target_results.loc[best_r2_idx, 'R²']
            print(f"{target_name:30s}: {best_model:15s} (R² = {best_r2:.3f})")
    
    return results_df

def create_feature_importance_comparison(all_models, target_cols, feature_cols):
    """Create feature importance comparison across different models"""
    print("🔍 Creating feature importance comparison...")
    
    # Analyze top 3 targets
    for i, target_name in enumerate(target_cols[:3]):
        if target_name not in all_models:
            continue
            
        print(f"  Analyzing feature importance for {target_name}...")
        
        models = all_models[target_name]
        
        # Get feature importance from different models
        importance_data = {}
        
        # Random Forest feature importance
        if 'random_forest' in models:
            rf_importance = models['random_forest'].feature_importances_
            importance_data['Random Forest'] = rf_importance
        
        # XGBoost feature importance
        if 'xgboost' in models and XGBOOST_AVAILABLE:
            xgb_importance = models['xgboost'].feature_importances_
            importance_data['XGBoost'] = xgb_importance
        
        # L1/L2 feature importance (coefficients)
        if 'lasso' in models:
            lasso_importance = np.abs(models['lasso'].coef_)
            importance_data['Lasso (L1)'] = lasso_importance
        
        if 'ridge' in models:
            ridge_importance = np.abs(models['ridge'].coef_)
            importance_data['Ridge (L2)'] = ridge_importance
        
        if 'elasticnet' in models:
            elastic_importance = np.abs(models['elasticnet'].coef_)
            importance_data['ElasticNet'] = elastic_importance
        
        # Create comparison plot
        if importance_data:
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f'Feature Importance Comparison: {target_name}', fontsize=16)
            axes = axes.flatten()
            
            # Get top 15 features for each model
            for j, (model_name, importance) in enumerate(importance_data.items()):
                if j < 4:  # Limit to 4 subplots
                    ax = axes[j]
                    
                    # Get top 15 features
                    top_indices = np.argsort(importance)[-15:]
                    top_importance = importance[top_indices]
                    top_features = [feature_cols[idx] for idx in top_indices]
                    
                    # Create horizontal bar plot
                    ax.barh(range(len(top_importance)), top_importance)
                    ax.set_yticks(range(len(top_importance)))
                    ax.set_yticklabels(top_features, fontsize=8)
                    ax.set_xlabel('Feature Importance')
                    ax.set_title(f'{model_name} - Top 15 Features')
                    ax.invert_yaxis()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'./feature_importance_comparison_{target_name.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

def generate_forecasts(all_models, df, target_cols):
    """Generate forecasts using all trained models"""
    print("🔮 Generating forecasts using all models...")
    
    # Prepare features for forecasting
    feature_cols = [col for col in df.columns if col not in target_cols + ['date']]
    X_latest = df[feature_cols].iloc[-1:].values
    
    all_forecasts = {}
    
    for target_name in target_cols:
        if target_name not in all_models:
            continue
            
        models = all_models[target_name]
        scaler_X = models['scaler_X']
        scaler_y = models['scaler_y']
        
        # Scale features
        X_forecast_scaled = scaler_X.transform(X_latest)
        
        # Generate predictions from all models
        forecasts = {}
        for model_name, model in models.items():
            if model_name not in ['scaler_X', 'scaler_y']:
                try:
                    pred_scaled = model.predict(X_forecast_scaled)
                    # Inverse transform
                    pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                    forecasts[model_name] = pred_original[0]
                except Exception as e:
                    print(f"    Error forecasting with {model_name} for {target_name}: {e}")
                    forecasts[model_name] = 0.0
        
        all_forecasts[target_name] = forecasts
    
    return all_forecasts

def main():
    """Main execution function"""
    print("="*80)
    print("XGBOOST AND L1/L2 REGRESSION COMPARISON ANALYSIS")
    print("="*80)
    
    # Get system info
    n_cores = cpu_count()
    print(f"🖥️  System has {n_cores} CPU cores available")
    print(f"📦 XGBoost available: {XGBOOST_AVAILABLE}")
    
    try:
        start_time = time.time()
        
        # Step 1: Load and prepare data
        df, target_cols, static_cols, ts_cols = load_and_prepare_data()
        
        # Step 2: Create time series features
        df = create_time_series_features(df, target_cols, ts_cols, static_cols)
        
        # Step 3: Train all models in parallel
        all_models, all_metrics = train_all_models(df, target_cols)
        
        # Step 4: Create performance comparison plots
        create_performance_comparison_plot(all_metrics, target_cols)
        
        # Step 5: Create detailed performance table
        results_df = create_detailed_performance_table(all_metrics, target_cols)
        
        # Step 6: Create feature importance comparison
        feature_cols = [col for col in df.columns if col not in target_cols + ['date']]
        create_feature_importance_comparison(all_models, target_cols, feature_cols)
        
        # Step 7: Generate forecasts
        forecasts = generate_forecasts(all_models, df, target_cols)
        
        total_time = time.time() - start_time
        
        # Step 8: Print comprehensive results
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"🖥️  CPU cores utilized: {min(n_cores, len(target_cols))}")
        
        print("\n📊 Model Performance Summary:")
        print("-" * 60)
        
        # Calculate overall best model
        model_performance = {}
        for target_name in target_cols:
            if target_name in all_metrics:
                for model_name, metrics in all_metrics[target_name].items():
                    if model_name not in model_performance:
                        model_performance[model_name] = []
                    model_performance[model_name].append(metrics['r2'])
        
        # Find best overall model
        avg_r2 = {model: np.mean(scores) for model, scores in model_performance.items()}
        best_model = max(avg_r2, key=avg_r2.get)
        
        print(f"Best Overall Model: {best_model} (Average R² = {avg_r2[best_model]:.3f})")
        
        print("\nAverage R² by Model:")
        for model, r2 in sorted(avg_r2.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model:15s}: {r2:.3f}")
        
        print("\n🔮 Forecast Results (Latest Values):")
        print("-" * 60)
        for target_name, target_forecasts in forecasts.items():
            print(f"\n  {target_name}:")
            for model_name, forecast in target_forecasts.items():
                print(f"    {model_name:15s}: {forecast:.2f}")
        
        print("\n📈 Analysis Features:")
        print("-" * 60)
        print("  • XGBoost with hyperparameter tuning")
        print("  • L1 (Lasso) and L2 (Ridge) regularization")
        print("  • ElasticNet (L1 + L2) regularization")
        print("  • Baseline Random Forest and Linear Regression")
        print("  • Ensemble model combining all approaches")
        print("  • Comprehensive performance comparison")
        print("  • Feature importance analysis across models")
        print("  • Parallel processing for efficiency")
        
        print("\n✅ Comprehensive model comparison completed successfully!")
        print("   All results and visualizations saved to current directory.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

