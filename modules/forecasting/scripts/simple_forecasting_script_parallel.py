# -*- coding: utf-8 -*-
"""
Optimized Parallel Forecasting Script - EBM Focus
A multiprocessing-optimized version that leverages multiple CPU cores

SCALER FIX IMPLEMENTATION:
- Fixed StandardScaler dimension mismatch in explainability functions
- Created separate scalers for explainability analysis to prevent data leakage
- Target-derived features (lagged features, rolling statistics) are excluded from explainability
- Each explainability function now creates its own scaler fitted on safe features only
- This prevents the "missing 99 features" error when using SHAP, LIME, and PDP analysis

Key Functions Added:
- create_safe_scaler_and_transform(): Creates new scaler for filtered features
- get_target_derived_features(): Identifies features to exclude from explainability
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
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
# Enhanced explainable imports
from sklearn.inspection import partial_dependence, permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import time
from scipy import stats

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("[OK] XGBoost available for gradient boosting")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not available. Install with: pip install xgboost")

# Try to import explainable libraries
try:
    import shap
    SHAP_AVAILABLE = True
    print("[OK] SHAP available for advanced explanations")
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
    print("[OK] LIME available for local explanations")
except ImportError:
    LIME_AVAILABLE = False
    print("[WARNING] LIME not available. Install with: pip install lime")

warnings.filterwarnings('ignore')

def clean_data_for_scaling(X):
    """Clean data for StandardScaler by handling NaN, infinite values, and constant columns"""
    # Check for NaN and infinite values
    X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check for constant columns (zero variance)
    feature_vars = np.var(X_clean, axis=0)
    constant_features = np.where(feature_vars == 0)[0]
    if len(constant_features) > 0:
        print(f"    Warning: Found {len(constant_features)} constant features, adding small noise...")
        # Add small random noise to constant features to prevent StandardScaler error
        for idx in constant_features:
            X_clean[:, idx] += np.random.normal(0, 1e-8, X_clean.shape[0])
    
    return X_clean

def create_safe_scaler_and_transform(X_train, X_test, feature_names=None):
    """Create a new scaler for safe features and transform data"""
    print(f"  Creating new scaler for {X_train.shape[1]} safe features...")
    
    # Validate input dimensions
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Feature dimension mismatch: X_train has {X_train.shape[1]} features, X_test has {X_test.shape[1]} features")
    
    if X_train.shape[1] == 0:
        raise ValueError("No features available for scaling")
    
    # Clean data before scaling
    X_train_clean = clean_data_for_scaling(X_train)
    X_test_clean = clean_data_for_scaling(X_test)
    
    # Create new scaler fitted on safe features only
    safe_scaler_X = StandardScaler()
    X_train_scaled = safe_scaler_X.fit_transform(X_train_clean)
    X_test_scaled = safe_scaler_X.transform(X_test_clean)
    
    print(f"  Successfully created scaler for {X_train.shape[1]} features")
    return X_train_scaled, X_test_scaled, safe_scaler_X

def get_target_derived_features(target_cols, lookback=7):
    """Get list of target-derived features to exclude from explainability analysis"""
    target_derived_features = []
    for target in target_cols:
        # Add lagged features of targets
        for lag in range(1, lookback + 1):
            target_derived_features.append(f'{target}_lag_{lag}')
        # Add rolling statistics of targets
        target_derived_features.extend([
            f'{target}_rolling_mean_7',
            f'{target}_rolling_std_7'
        ])
    return target_derived_features

def load_and_prepare_data():
    """Load and prepare data"""
    print("[INFO] Loading data...")
    
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
    print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
    
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
    print("[CLEAN] Handling missing values...")
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
    
    print(f"[OK] Data prepared: {len(df)} rows")
    print(f"   - Target columns: {len(target_cols)}")
    print(f"   - Static columns: {len(static_cols)}")
    print(f"   - Time series columns: {len(ts_cols)}")
    
    return df, target_cols, static_cols, ts_cols

def clean_data_for_explainability(df, target_cols):
    """Enhanced data cleaning specifically for explainable models"""
    print("[CLEAN] Applying enhanced data cleaning for explainability...")
    
    original_len = len(df)
    
    for col in target_cols:
        if col in df.columns:
            # Remove negative values (physically impossible for flow rates)
            negative_mask = df[col] < 0
            if negative_mask.sum() > 0:
                print(f"  Removing {negative_mask.sum()} negative values from {col}")
                df = df[~negative_mask]
            
            # Remove extreme outliers using percentile method (more robust than IQR)
            lower_percentile = np.percentile(df[col], 2.5)
            upper_percentile = np.percentile(df[col], 97.5)
            outlier_mask = (df[col] < lower_percentile) | (df[col] > upper_percentile)
            if outlier_mask.sum() > 0:
                print(f"  Removing {outlier_mask.sum()} extreme outliers from {col}")
                df = df[~outlier_mask]
    
    cleaned_len = len(df)
    print(f"[OK] Data cleaning completed: {original_len} → {cleaned_len} rows ({cleaned_len/original_len*100:.1f}% retained)")
    
    return df.reset_index(drop=True)

def create_time_series_features(df, target_cols, ts_cols, static_cols, lookback=7):
    """Create time series features for forecasting"""
    print(f"[PROCESS] Creating time series features (lookback={lookback})...")
    
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
    
    print(f"[OK] Created features: {len(df)} rows after feature engineering")
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

def train_multiple_forecasting_models(args):
    """Train forecasting models for multiple targets (chunked parallel processing)"""
    target_chunk, target_indices, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y_dict = args
    
    print(f"  Training models for {len(target_chunk)} targets on core {os.getpid()}...")
    
    models = {}
    results = {}
    
    for target_name, target_idx in zip(target_chunk, target_indices):
        print(f"    Training {target_name}...")
        
        # XGBoost (replacing Random Forest)
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1,
                eval_metric='rmse'
            )
            xgb_model.fit(X_train_scaled, y_train_scaled[:, target_idx])
            xgb_pred = xgb_model.predict(X_test_scaled)
        else:
            # Fallback to Random Forest if XGBoost not available
            xgb_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
            xgb_model.fit(X_train_scaled, y_train_scaled[:, target_idx])
            xgb_pred = xgb_model.predict(X_test_scaled)
        
        # Ridge Regression (replacing Linear Regression)
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train_scaled, y_train_scaled[:, target_idx])
        ridge_pred = ridge.predict(X_test_scaled)
        
        # Ensemble (average)
        ensemble_pred = (xgb_pred + ridge_pred) / 2
        
        # Comprehensive evaluation
        xgb_metrics = calculate_comprehensive_metrics(y_test_scaled[:, target_idx], xgb_pred)
        ridge_metrics = calculate_comprehensive_metrics(y_test_scaled[:, target_idx], ridge_pred)
        ensemble_metrics = calculate_comprehensive_metrics(y_test_scaled[:, target_idx], ensemble_pred)
        
        # Get the target-specific scaler_y
        target_scaler_y = scaler_y_dict[target_name]
        
        # Store model data - CRITICAL FIX
        model_data = {
            'xgb': xgb_model,
            'ridge': ridge,
            'scaler_X': scaler_X,
            'scaler_y': target_scaler_y
        }
        models[target_name] = model_data
        
        results[target_name] = {
            'xgb_metrics': xgb_metrics,
            'ridge_metrics': ridge_metrics,
            'ensemble_metrics': ensemble_metrics,
            'best_model': 'xgb' if xgb_metrics['r2'] > ridge_metrics['r2'] else 'ridge'
        }
        
        print(f"      {target_name} - XGB R²: {xgb_metrics['r2']:.3f}, Ridge R²: {ridge_metrics['r2']:.3f}, Ensemble R²: {ensemble_metrics['r2']:.3f}")
        print(f"      {target_name} - XGB RMAE: {xgb_metrics['rmae']:.3f}, Ridge RMAE: {ridge_metrics['rmae']:.3f}, Ensemble RMAE: {ensemble_metrics['rmae']:.3f}")
    
    return models, results

def train_forecasting_models_parallel(df, target_cols, static_cols, ts_cols, n_jobs=None):
    """Train forecasting models in parallel with chunked multitarget processing"""
    print("[TRAIN] Training forecasting models in parallel (multitarget chunks)...")
    
    if n_jobs is None:
        n_jobs = cpu_count()
    
    print(f"   Using {n_jobs} CPU cores for parallel processing")
    print(f"   Distributing {len(target_cols)} targets across {n_jobs} cores")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in target_cols + ['date']]
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Validate and clean data before scaling
    print("  Validating data for StandardScaler...")
    
    # Clean data using helper function
    X_train_clean = clean_data_for_scaling(X_train)
    X_test_clean = clean_data_for_scaling(X_test)
    
    # Scale features - global scaler_X for all targets (same features)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_clean)
    X_test_scaled = scaler_X.transform(X_test_clean)
    
    # Scale targets - each target gets its own scaler_y
    scaler_y_dict = {}
    y_train_scaled = np.zeros_like(y_train) # Corrected to match the shape of y_train
    y_test_scaled = np.zeros_like(y_test)
    
    for i, target_name in enumerate(target_cols):
        print(f"    Scaling target {target_name}...")
        
        # Clean target data
        y_train_clean = np.nan_to_num(y_train[:, i], nan=0.0, posinf=1e6, neginf=-1e6)
        y_test_clean = np.nan_to_num(y_test[:, i], nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Check for constant target
        if np.var(y_train_clean) == 0:
            print(f"      Warning: {target_name} has zero variance, adding small noise...")
            y_train_clean += np.random.normal(0, 1e-8, len(y_train_clean))
            y_test_clean += np.random.normal(0, 1e-8, len(y_test_clean))
        
        scaler_y = StandardScaler()
        scaler_y.fit(y_train_clean.reshape(-1, 1))
        scaler_y_dict[target_name] = scaler_y
        
        y_train_scaled[:, i] = scaler_y.transform(y_train_clean.reshape(-1, 1)).flatten()
        y_test_scaled[:, i] = scaler_y.transform(y_test_clean.reshape(-1, 1)).flatten()
    
    # Create chunks of targets for each core
    chunk_size = max(1, len(target_cols) // n_jobs)
    target_chunks = []
    target_indices_chunks = []
    
    for i in range(0, len(target_cols), chunk_size):
        chunk = target_cols[i:i + chunk_size]
        indices = list(range(i, min(i + chunk_size, len(target_cols)))) # for a core with multiple targets, we need to create a list of indices for each target
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
        args_list.append((
            chunk, indices, X_train_scaled, X_test_scaled, 
            y_train_scaled, y_test_scaled, scaler_X, scaler_y_dict
        ))
    
    # Train models in parallel
    start_time = time.time()
    with Pool(processes=len(args_list)) as pool:
        chunk_results = pool.map(train_multiple_forecasting_models, args_list)
    
    training_time = time.time() - start_time
    print(f"[OK] Forecasting models trained in {training_time:.2f} seconds!")
    
    # Organize results from all chunks - CRITICAL FIX
    models = {}
    model_results = {}
    
    print("[MERGE] Merging results from parallel processes...")
    for i, (chunk_models, chunk_results_dict) in enumerate(chunk_results):
        print(f"   Chunk {i+1}: Found {len(chunk_models)} models, {len(chunk_results_dict)} results")
        models.update(chunk_models)
        model_results.update(chunk_results_dict)
    
    print(f"[OK] Final merged results: {len(models)} models, {len(model_results)} results")
    print(f"   Model keys: {list(models.keys())}")
    
    return models, model_results

def parallel_forecast(args):
    """Generate forecasts for multiple targets in parallel"""
    target_chunk, target_indices, models, X_forecast, scaler_X = args
    
    print(f"  Generating forecasts for {len(target_chunk)} targets on core {os.getpid()}...")
    
    forecasts = {}
    
    for target_name, target_idx in zip(target_chunk, target_indices):
        print(f"    Forecasting {target_name}...")
        
        try:
            model_data = models[target_name]
            xgb_model = model_data['xgb']
            ridge = model_data['ridge']
            scaler_y = model_data['scaler_y']
            
            # Generate predictions
            xgb_pred = xgb_model.predict(X_forecast)
            ridge_pred = ridge.predict(X_forecast)
            
            # Ensemble forecast
            ensemble_pred = (xgb_pred + ridge_pred) / 2
            
            # Inverse transform using the target's own scaler_y
            forecast_original = scaler_y.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
            
            forecasts[target_name] = forecast_original[0]
            
            print(f"      {target_name} - Forecast: {forecast_original[0]:.2f}")
            
        except Exception as e:
            print(f"      [ERROR] Error forecasting {target_name}: {e}")
            forecasts[target_name] = 0.0  # Default value
    
    return forecasts

def generate_parallel_forecasts(models, df, target_cols, n_jobs=None, forecast_steps=7):
    """Generate forecasts in parallel across multiple cores"""
    print("[FORECAST] Generating parallel forecasts...")
    
    if n_jobs is None:
        n_jobs = cpu_count()
    
    print(f"   Using {n_jobs} CPU cores for parallel forecasting")
    print(f"   Forecasting {forecast_steps} steps ahead")
    
    # Debug: Check if models dictionary is populated
    print(f"   Available models: {len(models)}")
    print(f"   Model keys: {list(models.keys())}")
    
    if not models:
        raise ValueError("Models dictionary is empty! Check the training process.")
    
    # Prepare features for forecasting
    feature_cols = [col for col in df.columns if col not in target_cols + ['date']]
    X_latest = df[feature_cols].iloc[-1:].values
    
    # Create chunks of targets for each core
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
    
    print(f"   Created {len(target_chunks)} forecast chunks:")
    for i, chunk in enumerate(target_chunks):
        print(f"     Core {i+1}: {len(chunk)} targets - {chunk}")
    
    # Get scaler_X from first available model
    first_target = list(models.keys())[0]
    scaler_X = models[first_target]['scaler_X']
    
    # Clean and scale features
    X_latest_clean = clean_data_for_scaling(X_latest)
    X_forecast_scaled = scaler_X.transform(X_latest_clean)
    
    # Prepare arguments for parallel processing
    args_list = []
    for chunk, indices in zip(target_chunks, target_indices_chunks):
        args_list.append((chunk, indices, models, X_forecast_scaled, scaler_X))
    
    # Generate forecasts in parallel
    start_time = time.time()
    with Pool(processes=len(args_list)) as pool:
        chunk_forecasts = pool.map(parallel_forecast, args_list)
    
    forecast_time = time.time() - start_time
    print(f"[OK] Parallel forecasts generated in {forecast_time:.2f} seconds!")
    
    # Organize results from all chunks
    all_forecasts = {}
    for chunk_forecasts_dict in chunk_forecasts:
        all_forecasts.update(chunk_forecasts_dict)
    
    return all_forecasts
    
def create_shap_explanations(models, df, target_cols, static_cols, feature_cols):
    """Create SHAP explanations for model interpretability - excludes target-derived features"""
    if not SHAP_AVAILABLE:
        print("[WARNING] SHAP not available. Skipping SHAP explanations.")
        return None
    
    print("[INFO] Creating SHAP explanations for static feature analysis...")
    
    # Get target-derived features to exclude
    target_derived_features = get_target_derived_features(target_cols)
    
    # Filter feature_cols to exclude target-derived features
    safe_feature_cols = [col for col in feature_cols if col not in target_derived_features]
    print(f"  Excluding {len(target_derived_features)} target-derived features from SHAP analysis")
    print(f"  Using {len(safe_feature_cols)} safe features for SHAP analysis")
    
    # Prepare data with safe features only
    X = df[safe_feature_cols].values
    y = df[target_cols].values
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create new scaler for safe features
    X_train_scaled, X_test_scaled, safe_scaler_X = create_safe_scaler_and_transform(X_train, X_test, safe_feature_cols)
    
    # Get static feature indices from safe features
    static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
    static_feature_names = [safe_feature_cols[i] for i in static_indices]
    
    shap_results = {}
    
    # Analyze top 3 targets
    for i, target_name in enumerate(target_cols[:3]):
        print(f"  Creating SHAP explanations for {target_name}...")
        
        try:
            # Get the best model for this target
            model_data = models[target_name]
            xgb_model = model_data['xgb']
            ridge = model_data['ridge']
            
            # Create SHAP explainers
            if hasattr(xgb_model, 'feature_importances_'):
                # XGBoost or Random Forest
                xgb_explainer = shap.TreeExplainer(xgb_model)
                xgb_shap_values = xgb_explainer.shap_values(X_test_scaled)
            else:
                # Fallback for other models
                xgb_explainer = shap.LinearExplainer(xgb_model, X_train_scaled)
                xgb_shap_values = xgb_explainer.shap_values(X_test_scaled)
            
            # Ridge explainer for comparison
            ridge_explainer = shap.LinearExplainer(ridge, X_train_scaled)
            ridge_shap_values = ridge_explainer.shap_values(X_test_scaled)
            
            # Focus on static features
            xgb_static_shap = xgb_shap_values[:, static_indices]
            ridge_static_shap = ridge_shap_values[:, static_indices]
            X_static_test = X_test_scaled[:, static_indices]
            
            # Create comprehensive SHAP plots
            fig, axes = plt.subplots(2, 3, figsize=(24, 16))
            fig.suptitle(f'SHAP Analysis: {target_name} - Static Features Impact', fontsize=16)
            
            # Plot 1: XGB SHAP summary (fixed API)
            try:
                # Create a separate figure for summary plot
                plt.figure(figsize=(8, 6))
                shap.summary_plot(xgb_static_shap, X_static_test, 
                                feature_names=static_feature_names, 
                                show=False, max_display=10)
                plt.title('XGBoost - SHAP Summary (Static Features)')
                plt.tight_layout()
                plt.savefig(f'./shap_xgb_summary_{target_name.replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create bar plot for XGB importance
                xgb_importance = np.abs(xgb_static_shap).mean(0)
                top_indices = np.argsort(xgb_importance)[-10:]
                axes[0, 0].barh(range(len(top_indices)), xgb_importance[top_indices])
                axes[0, 0].set_yticks(range(len(top_indices)))
                axes[0, 0].set_yticklabels([static_feature_names[i] for i in top_indices])
                axes[0, 0].set_title('XGB - Top 10 Static Feature Importance')
                axes[0, 0].invert_yaxis()
            except Exception as e:
                axes[0, 0].text(0.5, 0.5, f'XGB SHAP Error: {str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('XGB - SHAP Summary (Error)')
            
            # Plot 2: Ridge SHAP summary (fixed API)
            try:
                # Create a separate figure for Ridge summary plot
                plt.figure(figsize=(8, 6))
                shap.summary_plot(ridge_static_shap, X_static_test,
                                feature_names=static_feature_names,
                                show=False, max_display=10)
                plt.title('Ridge Regression - SHAP Summary (Static Features)')
                plt.tight_layout()
                plt.savefig(f'./shap_ridge_summary_{target_name.replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create bar plot for Ridge importance
                ridge_importance = np.abs(ridge_static_shap).mean(0)
                top_indices_ridge = np.argsort(ridge_importance)[-10:]
                axes[0, 1].barh(range(len(top_indices_ridge)), ridge_importance[top_indices_ridge])
                axes[0, 1].set_yticks(range(len(top_indices_ridge)))
                axes[0, 1].set_yticklabels([static_feature_names[i] for i in top_indices_ridge])
                axes[0, 1].set_title('Ridge - Top 10 Static Feature Importance')
                axes[0, 1].invert_yaxis()
            except Exception as e:
                axes[0, 1].text(0.5, 0.5, f'Ridge SHAP Error: {str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Ridge - SHAP Summary (Error)')
            
            # Plot 3: Feature importance comparison
            try:
                xgb_importance = np.abs(xgb_static_shap).mean(0)
                ridge_importance = np.abs(ridge_static_shap).mean(0)
                
                # Get top 10 features
                top_indices = np.argsort(xgb_importance)[-10:]
                
                axes[0, 2].barh(range(len(top_indices)), xgb_importance[top_indices])
                axes[0, 2].set_yticks(range(len(top_indices)))
                axes[0, 2].set_yticklabels([static_feature_names[i] for i in top_indices])
                axes[0, 2].set_title('XGB - Top 10 Static Feature Importance')
                axes[0, 2].invert_yaxis()
            except Exception as e:
                axes[0, 2].text(0.5, 0.5, f'Importance Error: {str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Feature Importance (Error)')
            
            # Plot 4: SHAP waterfall for a sample prediction (fixed API)
            try:
                if len(xgb_static_shap) > 0:
                    # Create waterfall plot in separate figure
                    plt.figure(figsize=(10, 6))
                    shap.waterfall_plot(xgb_explainer.expected_value, xgb_static_shap[0], 
                                      X_static_test[0], feature_names=static_feature_names,
                                      show=False)
                    plt.title('XGB - Sample Prediction Explanation')
                    plt.tight_layout()
                    plt.savefig(f'./shap_waterfall_{target_name.replace(" ", "_")}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Create simple bar plot in subplot
                    sample_shap = xgb_static_shap[0]
                    sample_values = X_static_test[0]
                    axes[1, 0].bar(range(len(sample_shap)), sample_shap)
                    axes[1, 0].set_xticks(range(len(sample_shap)))
                    axes[1, 0].set_xticklabels(static_feature_names, rotation=45, ha='right')
                    axes[1, 0].set_title('XGB - Sample Prediction SHAP Values')
                    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, f'Waterfall Error: {str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Waterfall Plot (Error)')
            
            # Plot 5: SHAP dependence plot for top feature (fixed API)
            try:
                if len(top_indices) > 0:
                    top_feature_idx = top_indices[-1]
                    # Create dependence plot in separate figure
                    plt.figure(figsize=(8, 6))
                    shap.dependence_plot(top_feature_idx, xgb_static_shap, X_static_test,
                                       feature_names=static_feature_names,
                                       show=False)
                    plt.title(f'XGB - Dependence: {static_feature_names[top_feature_idx]}')
                    plt.tight_layout()
                    plt.savefig(f'./shap_dependence_{target_name.replace(" ", "_")}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Create simple scatter plot in subplot
                    feature_values = X_static_test[:, top_feature_idx]
                    shap_values = xgb_static_shap[:, top_feature_idx]
                    axes[1, 1].scatter(feature_values, shap_values, alpha=0.6)
                    axes[1, 1].set_xlabel(static_feature_names[top_feature_idx])
                    axes[1, 1].set_ylabel('SHAP Value')
                    axes[1, 1].set_title(f'XGB - Dependence: {static_feature_names[top_feature_idx]}')
                    axes[1, 1].grid(True, alpha=0.3)
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'Dependence Error: {str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Dependence Plot (Error)')
            
            # Plot 6: Model comparison
            try:
                xgb_importance = np.abs(xgb_static_shap).mean(0)
                ridge_importance = np.abs(ridge_static_shap).mean(0)
                axes[1, 2].scatter(xgb_importance, ridge_importance, alpha=0.7)
                axes[1, 2].plot([0, max(xgb_importance)], [0, max(xgb_importance)], 'r--')
                axes[1, 2].set_xlabel('XGBoost Importance')
                axes[1, 2].set_ylabel('Ridge Regression Importance')
                axes[1, 2].set_title('Model Importance Comparison')
                axes[1, 2].grid(True, alpha=0.3)
            except Exception as e:
                axes[1, 2].text(0.5, 0.5, f'Comparison Error: {str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Model Comparison (Error)')
            
            plt.tight_layout()
            plt.savefig(f'./shap_static_analysis_{target_name.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Store results
            shap_results[target_name] = {
                'xgb_shap_values': xgb_static_shap,
                'ridge_shap_values': ridge_static_shap,
                'static_feature_names': static_feature_names,
                'xgb_importance': xgb_importance,
                'ridge_importance': ridge_importance
            }
            
            print(f"    [OK] SHAP analysis completed for {target_name}")
            
        except Exception as e:
            print(f"    [ERROR] Error in SHAP analysis for {target_name}: {e}")
            import traceback
            traceback.print_exc()
            shap_results[target_name] = {'error': str(e)}
    
    return shap_results

def create_lime_explanations(models, df, target_cols, static_cols, feature_cols, results):
    """Create LIME explanations for local interpretability - excludes target-derived features"""
    if not LIME_AVAILABLE:
        print("[WARNING] LIME not available. Skipping LIME explanations.")
        return None
    
    print("[INFO] Creating LIME explanations for individual predictions...")
    
    # Get target-derived features to exclude
    target_derived_features = get_target_derived_features(target_cols)
    
    # Filter feature_cols to exclude target-derived features
    safe_feature_cols = [col for col in feature_cols if col not in target_derived_features]
    print(f"  Excluding {len(target_derived_features)} target-derived features from LIME analysis")
    print(f"  Using {len(safe_feature_cols)} safe features for LIME analysis")
    
    # Prepare data with safe features only
    X = df[safe_feature_cols].values
    y = df[target_cols].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create new scaler for safe features
    X_train_scaled, X_test_scaled, safe_scaler_X = create_safe_scaler_and_transform(X_train, X_test, safe_feature_cols)
    
    # Get static feature indices from safe features
    static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
    static_feature_names = [safe_feature_cols[i] for i in static_indices]
    
    lime_results = {}
    
    # Analyze top 3 targets
    for i, target_name in enumerate(target_cols[:3]):
        print(f"  Creating LIME explanations for {target_name}...")
        
        try:
            # Get the target-specific scaler_y
            model_data = models[target_name]
            scaler_y = model_data['scaler_y']
            
            # Scale target data using the same scaler as training
            y_train_clean = np.nan_to_num(y_train[:, i], nan=0.0, posinf=1e6, neginf=-1e6)
            y_test_clean = np.nan_to_num(y_test[:, i], nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Check for constant target
            if np.var(y_train_clean) == 0:
                print(f"      Warning: {target_name} has zero variance, adding small noise...")
                y_train_clean += np.random.normal(0, 1e-8, len(y_train_clean))
                y_test_clean += np.random.normal(0, 1e-8, len(y_test_clean))
            
            y_train_scaled = scaler_y.transform(y_train_clean.reshape(-1, 1)).flatten()
            y_test_scaled = scaler_y.transform(y_test_clean.reshape(-1, 1)).flatten()
            
            # Train surrogate model on safe features for LIME
            best_model_name = results[target_name]['best_model']
            
            if best_model_name == 'xgb':
                if XGBOOST_AVAILABLE:
                    surrogate_model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        n_jobs=1,
                        eval_metric='rmse'
                    )
                else:
                    surrogate_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
            else:
                surrogate_model = Ridge(alpha=1.0, random_state=42)
            
            surrogate_model.fit(X_train_scaled, y_train_scaled)
            
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                X_train_scaled,
                feature_names=safe_feature_cols,
                mode='regression',
                discretize_continuous=True
            )
            
            # Explain a few sample predictions
            sample_indices = np.random.choice(len(X_test_scaled), size=min(3, len(X_test_scaled)), replace=False)
            
            explanations = []
            for idx in sample_indices:
                explanation = explainer.explain_instance(
                    X_test_scaled[idx], 
                    surrogate_model.predict,
                    num_features=10
                )
                explanations.append(explanation)
            
            # Create visualization
            fig, axes = plt.subplots(1, len(explanations), figsize=(6*len(explanations), 8))
            if len(explanations) == 1:
                axes = [axes]
            
            fig.suptitle(f'LIME Explanations: {target_name} (Surrogate {best_model_name.upper()} Model)', fontsize=16)
            
            for j, (explanation, idx) in enumerate(zip(explanations, sample_indices)):
                # Get explanation data
                exp_data = explanation.as_list()
                features = [item[0] for item in exp_data]
                weights = [item[1] for item in exp_data]
                
                # Focus on static features
                static_exp = [(feat, weight) for feat, weight in exp_data 
                             if any(static_feat in feat for static_feat in static_cols)]
                
                if static_exp:
                    static_features = [item[0] for item in static_exp[:8]]  # Top 8
                    static_weights = [item[1] for item in static_exp[:8]]
                    
                    colors = ['green' if w > 0 else 'red' for w in static_weights]
                    axes[j].barh(range(len(static_features)), static_weights, color=colors, alpha=0.7)
                    axes[j].set_yticks(range(len(static_features)))
                    axes[j].set_yticklabels(static_features, fontsize=8)
                    axes[j].set_title(f'Sample {j+1}\nActual: {y_test[idx]:.2f}')
                    axes[j].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    axes[j].grid(True, alpha=0.3)
                else:
                    axes[j].text(0.5, 0.5, 'No static features\nin explanation', 
                               ha='center', va='center', transform=axes[j].transAxes)
                    axes[j].set_title(f'Sample {j+1}')
            
            plt.tight_layout()
            plt.savefig(f'./lime_explanations_{target_name.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            lime_results[target_name] = {
                'explanations': explanations,
                'sample_indices': sample_indices,
                'best_model': best_model_name,
                'surrogate_model': True  # Flag to indicate surrogate model was used
            }
            
            print(f"    [OK] LIME explanations created for {target_name} using surrogate model")
            
        except Exception as e:
            print(f"    [ERROR] Error creating LIME explanations for {target_name}: {e}")
            lime_results[target_name] = {'error': str(e)}
    
    return lime_results

def create_2d_partial_dependence_plots(models, df, target_cols, static_cols, ts_cols, feature_cols):
    """Create 2D Partial Dependence Plots for best combined static and non-static feature effects - excludes target-derived features"""
    print("[PLOT] Creating 2D Partial Dependence Plots for combined feature effects...")
    
    # Filter out target-derived features to prevent data leakage
    target_derived_features = []
    for target in target_cols:
        # Add lagged features of targets
        for lag in range(1, 8):  # Assuming lookback of 7
            target_derived_features.append(f'{target}_lag_{lag}')
        # Add rolling statistics of targets
        target_derived_features.extend([
            f'{target}_rolling_mean_7',
            f'{target}_rolling_std_7'
        ])
    
    # Filter feature_cols to exclude target-derived features
    safe_feature_cols = [col for col in feature_cols if col not in target_derived_features]
    print(f"  Excluding {len(target_derived_features)} target-derived features from 2D PDP analysis")
    
    # Prepare data with safe features only
    X = df[safe_feature_cols].values
    y = df[target_cols].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use the same scaler_X as training (global for all targets)
    first_target = target_cols[0]
    scaler_X = models[first_target]['scaler_X']
    
    # Clean data before scaling
    X_train_clean = clean_data_for_scaling(X_train)
    X_train_scaled = scaler_X.transform(X_train_clean)
    
    # Get feature indices from safe features
    static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
    ts_indices = [i for i, col in enumerate(safe_feature_cols) if col in ts_cols]
    static_feature_names = [safe_feature_cols[i] for i in static_indices]
    ts_feature_names = [safe_feature_cols[i] for i in ts_indices]
    
    # Analyze top 3 targets
    for i, target_name in enumerate(target_cols[:3]):
        print(f"  Creating 2D PDP plots for {target_name}...")
        
        try:
            # Get models
            model_data = models[target_name]
            xgb_model = model_data['xgb']
            ridge = model_data['ridge']
            
            # Get feature importance for non-static features only (from safe features)
            if hasattr(xgb_model, 'feature_importances_'):
                xgb_importance = xgb_model.feature_importances_
                # Map to safe feature indices
                safe_ts_indices = [i for i, col in enumerate(safe_feature_cols) if col in ts_cols]
                safe_static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
                ts_importance = xgb_importance[safe_ts_indices] if safe_ts_indices else np.array([])
                static_importance = xgb_importance[safe_static_indices] if safe_static_indices else np.array([])
            else:
                # Fallback for non-tree models
                safe_ts_indices = [i for i, col in enumerate(safe_feature_cols) if col in ts_cols]
                safe_static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
                ts_importance = np.ones(len(safe_ts_indices)) / max(1, len(safe_ts_indices)) if safe_ts_indices else np.array([])
                static_importance = np.ones(len(safe_static_indices)) / max(1, len(safe_static_indices)) if safe_static_indices else np.array([])
            
            # Find best static and non-static features from safe features
            if len(static_importance) > 0 and len(ts_importance) > 0:
                best_static_idx = safe_static_indices[np.argmax(static_importance)]
                best_ts_idx = safe_ts_indices[np.argmax(ts_importance)]
                
                best_static_name = safe_feature_cols[best_static_idx]
                best_ts_name = safe_feature_cols[best_ts_idx]
            else:
                print(f"    [WARNING] No safe features available for 2D PDP analysis of {target_name}")
                continue
            
            print(f"    Best static feature: {best_static_name} (importance: {static_importance[np.argmax(static_importance)]:.4f})")
            print(f"    Best non-static feature: {best_ts_name} (importance: {ts_importance[np.argmax(ts_importance)]:.4f})")
            
            # Create 2D PDP plots
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(f'2D Partial Dependence: {target_name} - {best_static_name} vs {best_ts_name}', fontsize=16)
            
            # XGBoost 2D PDP
            try:
                pdp_xgb_2d, axes_xgb = partial_dependence(
                    xgb_model, X_train_scaled, [best_static_idx, best_ts_idx], 
                    kind='average', grid_resolution=20
                )
                
                # Create contour plot for XGB
                X_grid, Y_grid = np.meshgrid(axes_xgb[0], axes_xgb[1])
                contour = axes[0].contourf(X_grid, Y_grid, pdp_xgb_2d, levels=20, cmap='viridis')
                axes[0].set_xlabel(best_static_name, fontsize=12)
                axes[0].set_ylabel(best_ts_name, fontsize=12)
                axes[0].set_title('XGBoost - 2D Partial Dependence')
                plt.colorbar(contour, ax=axes[0])
                
            except Exception as e:
                axes[0].text(0.5, 0.5, f'XGB Error: {str(e)[:30]}...', 
                           ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title('Random Forest - Error')
            
            # Ridge Regression 2D PDP
            try:
                pdp_ridge_2d, axes_ridge = partial_dependence(
                    ridge, X_train_scaled, [best_static_idx, best_ts_idx], 
                    kind='average', grid_resolution=20
                )
                
                # Create contour plot for Ridge
                X_grid, Y_grid = np.meshgrid(axes_ridge[0], axes_ridge[1])
                contour = axes[1].contourf(X_grid, Y_grid, pdp_ridge_2d, levels=20, cmap='plasma')
                axes[1].set_xlabel(best_static_name, fontsize=12)
                axes[1].set_ylabel(best_ts_name, fontsize=12)
                axes[1].set_title('Ridge Regression - 2D Partial Dependence')
                plt.colorbar(contour, ax=axes[1])
                
            except Exception as e:
                axes[1].text(0.5, 0.5, f'Ridge Error: {str(e)[:30]}...', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Ridge Regression - Error')
            
            plt.tight_layout()
            plt.savefig(f'./pdp_2d_combined_{target_name.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    [OK] 2D PDP plots created for {target_name}")
            
        except Exception as e:
            print(f"    [ERROR] Error creating 2D PDP plots for {target_name}: {e}")

def create_partial_dependence_plots(models, df, target_cols, static_cols, feature_cols):
    """Create Partial Dependence Plots for static feature analysis - excludes target-derived features"""
    print("[PLOT] Creating Partial Dependence Plots for static feature effects...")
    
    # Get target-derived features to exclude
    target_derived_features = get_target_derived_features(target_cols)
    
    # Filter feature_cols to exclude target-derived features
    safe_feature_cols = [col for col in feature_cols if col not in target_derived_features]
    print(f"  Excluding {len(target_derived_features)} target-derived features from PDP analysis")
    print(f"  Using {len(safe_feature_cols)} safe features for PDP analysis")
    
    # Prepare data with safe features only
    X = df[safe_feature_cols].values
    y = df[target_cols].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create new scaler for safe features
    X_train_scaled, X_test_scaled, safe_scaler_X = create_safe_scaler_and_transform(X_train, X_test, safe_feature_cols)
    
    # Get static feature indices from safe features
    static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
    static_feature_names = [safe_feature_cols[i] for i in static_indices]
    
    # Analyze top 3 targets
    for i, target_name in enumerate(target_cols[:3]):
        print(f"  Creating PDP plots for {target_name}...")
        
        try:
            # Get the target-specific scaler_y
            model_data = models[target_name]
            scaler_y = model_data['scaler_y']
            
            # Scale target data using the same scaler as training
            y_train_clean = np.nan_to_num(y_train[:, i], nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Check for constant target
            if np.var(y_train_clean) == 0:
                print(f"      Warning: {target_name} has zero variance, adding small noise...")
                y_train_clean += np.random.normal(0, 1e-8, len(y_train_clean))
            
            y_train_scaled = scaler_y.transform(y_train_clean.reshape(-1, 1)).flatten()
            
            # Train surrogate models on safe features
            if XGBOOST_AVAILABLE:
                surrogate_xgb = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=1,
                    eval_metric='rmse'
                )
                surrogate_xgb.fit(X_train_scaled, y_train_scaled)
            else:
                surrogate_xgb = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
                surrogate_xgb.fit(X_train_scaled, y_train_scaled)
            
            surrogate_ridge = Ridge(alpha=1.0, random_state=42)
            surrogate_ridge.fit(X_train_scaled, y_train_scaled)
            
            # Select top 6 static features based on XGB importance (from safe features)
            if hasattr(surrogate_xgb, 'feature_importances_'):
                xgb_importance = surrogate_xgb.feature_importances_
                safe_static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
                static_importance = xgb_importance[safe_static_indices] if safe_static_indices else np.array([])
            else:
                # Fallback for non-tree models
                safe_static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
                static_importance = np.ones(len(safe_static_indices)) / max(1, len(safe_static_indices)) if safe_static_indices else np.array([])
            
            if len(static_importance) > 0:
                top_static_indices = np.argsort(static_importance)[-6:]
            else:
                print(f"    [WARNING] No safe static features available for PDP analysis of {target_name}")
                continue
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Partial Dependence Plots: {target_name} - Static Features (Surrogate Models)', fontsize=16)
            axes = axes.flatten()
            
            for j, local_idx in enumerate(top_static_indices):
                global_idx = safe_static_indices[local_idx]
                feature_name = safe_feature_cols[global_idx]
                
                try:
                    # XGB PDP
                    pdp_xgb, axes_xgb = partial_dependence(
                        surrogate_xgb, X_train_scaled, [global_idx], 
                        kind='average', grid_resolution=50
                    )
                    
                    # Ridge PDP
                    pdp_ridge, axes_ridge = partial_dependence(
                        surrogate_ridge, X_train_scaled, [global_idx], 
                        kind='average', grid_resolution=50
                    )
                    
                    # Plot both models
                    ax = axes[j]
                    ax.plot(axes_xgb[0], pdp_xgb[0], 'b-', label='XGBoost Surrogate', linewidth=2)
                    ax.plot(axes_ridge[0], pdp_ridge[0], 'r--', label='Ridge Surrogate', linewidth=2)
                    ax.set_xlabel(feature_name, fontsize=10)
                    ax.set_ylabel('Partial Dependence', fontsize=10)
                    ax.set_title(f'{feature_name[:20]}...', fontsize=10)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                except Exception as e:
                    axes[j].text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                               ha='center', va='center', transform=axes[j].transAxes)
                    axes[j].set_title(f'Error: {feature_name[:15]}...')
            
            plt.tight_layout()
            plt.savefig(f'./pdp_static_features_{target_name.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    [OK] PDP plots created for {target_name} using surrogate models")
            
        except Exception as e:
            print(f"    [ERROR] Error creating PDP plots for {target_name}: {e}")

def create_2d_partial_dependence_plots(models, df, target_cols, static_cols, ts_cols, feature_cols):
    """Create 2D Partial Dependence Plots for best combined static and non-static feature effects - excludes target-derived features"""
    print("[PLOT] Creating 2D Partial Dependence Plots for combined feature effects...")
    
    # Get target-derived features to exclude
    target_derived_features = get_target_derived_features(target_cols)
    
    # Filter feature_cols to exclude target-derived features
    safe_feature_cols = [col for col in feature_cols if col not in target_derived_features]
    print(f"  Excluding {len(target_derived_features)} target-derived features from 2D PDP analysis")
    print(f"  Using {len(safe_feature_cols)} safe features for 2D PDP analysis")
    
    # Prepare data with safe features only
    X = df[safe_feature_cols].values
    y = df[target_cols].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create new scaler for safe features
    X_train_scaled, X_test_scaled, safe_scaler_X = create_safe_scaler_and_transform(X_train, X_test, safe_feature_cols)
    
    # Get feature indices from safe features
    static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
    ts_indices = [i for i, col in enumerate(safe_feature_cols) if col in ts_cols]
    static_feature_names = [safe_feature_cols[i] for i in static_indices]
    ts_feature_names = [safe_feature_cols[i] for i in ts_indices]
    
    # Analyze top 3 targets
    for i, target_name in enumerate(target_cols[:3]):
        print(f"  Creating 2D PDP plots for {target_name}...")
        
        try:
            # Get the target-specific scaler_y
            model_data = models[target_name]
            scaler_y = model_data['scaler_y']
            
            # Scale target data using the same scaler as training
            y_train_clean = np.nan_to_num(y_train[:, i], nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Check for constant target
            if np.var(y_train_clean) == 0:
                print(f"      Warning: {target_name} has zero variance, adding small noise...")
                y_train_clean += np.random.normal(0, 1e-8, len(y_train_clean))
            
            y_train_scaled = scaler_y.transform(y_train_clean.reshape(-1, 1)).flatten()
            
            # Train surrogate models on safe features
            if XGBOOST_AVAILABLE:
                surrogate_xgb = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=1,
                    eval_metric='rmse'
                )
                surrogate_xgb.fit(X_train_scaled, y_train_scaled)
            else:
                surrogate_xgb = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
                surrogate_xgb.fit(X_train_scaled, y_train_scaled)
            
            surrogate_ridge = Ridge(alpha=1.0, random_state=42)
            surrogate_ridge.fit(X_train_scaled, y_train_scaled)
            
            # Get feature importance for non-static features only (from safe features)
            if hasattr(surrogate_xgb, 'feature_importances_'):
                xgb_importance = surrogate_xgb.feature_importances_
                # Map to safe feature indices
                safe_ts_indices = [i for i, col in enumerate(safe_feature_cols) if col in ts_cols]
                safe_static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
                ts_importance = xgb_importance[safe_ts_indices] if safe_ts_indices else np.array([])
                static_importance = xgb_importance[safe_static_indices] if safe_static_indices else np.array([])
            else:
                # Fallback for non-tree models
                safe_ts_indices = [i for i, col in enumerate(safe_feature_cols) if col in ts_cols]
                safe_static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
                ts_importance = np.ones(len(safe_ts_indices)) / max(1, len(safe_ts_indices)) if safe_ts_indices else np.array([])
                static_importance = np.ones(len(safe_static_indices)) / max(1, len(safe_static_indices)) if safe_static_indices else np.array([])
            
            # Find best static and non-static features from safe features
            if len(static_importance) > 0 and len(ts_importance) > 0:
                best_static_idx = safe_static_indices[np.argmax(static_importance)]
                best_ts_idx = safe_ts_indices[np.argmax(ts_importance)]
                
                best_static_name = safe_feature_cols[best_static_idx]
                best_ts_name = safe_feature_cols[best_ts_idx]
            else:
                print(f"    [WARNING] No safe features available for 2D PDP analysis of {target_name}")
                continue
            
            print(f"    Best static feature: {best_static_name} (importance: {static_importance[np.argmax(static_importance)]:.4f})")
            print(f"    Best non-static feature: {best_ts_name} (importance: {ts_importance[np.argmax(ts_importance)]:.4f})")
            
            # Create 2D PDP plots
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(f'2D Partial Dependence: {target_name} - {best_static_name} vs {best_ts_name} (Surrogate Models)', fontsize=16)
            
            # XGBoost 2D PDP
            try:
                pdp_xgb_2d, axes_xgb = partial_dependence(
                    surrogate_xgb, X_train_scaled, [best_static_idx, best_ts_idx], 
                    kind='average', grid_resolution=20
                )
                
                # Create contour plot for XGB
                X_grid, Y_grid = np.meshgrid(axes_xgb[0], axes_xgb[1])
                contour = axes[0].contourf(X_grid, Y_grid, pdp_xgb_2d, levels=20, cmap='viridis')
                axes[0].set_xlabel(best_static_name, fontsize=12)
                axes[0].set_ylabel(best_ts_name, fontsize=12)
                axes[0].set_title('XGBoost Surrogate - 2D Partial Dependence')
                plt.colorbar(contour, ax=axes[0])
                
            except Exception as e:
                axes[0].text(0.5, 0.5, f'XGB Error: {str(e)[:30]}...', 
                           ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title('XGB Surrogate - Error')
            
            # Ridge Regression 2D PDP
            try:
                pdp_ridge_2d, axes_ridge = partial_dependence(
                    surrogate_ridge, X_train_scaled, [best_static_idx, best_ts_idx], 
                    kind='average', grid_resolution=20
                )
                
                # Create contour plot for Ridge
                X_grid, Y_grid = np.meshgrid(axes_ridge[0], axes_ridge[1])
                contour = axes[1].contourf(X_grid, Y_grid, pdp_ridge_2d, levels=20, cmap='plasma')
                axes[1].set_xlabel(best_static_name, fontsize=12)
                axes[1].set_ylabel(best_ts_name, fontsize=12)
                axes[1].set_title('Ridge Surrogate - 2D Partial Dependence')
                plt.colorbar(contour, ax=axes[1])
                
            except Exception as e:
                axes[1].text(0.5, 0.5, f'Ridge Error: {str(e)[:30]}...', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Ridge Surrogate - Error')
            
            plt.tight_layout()
            plt.savefig(f'./pdp_2d_combined_{target_name.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    [OK] 2D PDP plots created for {target_name} using surrogate models")
            
        except Exception as e:
            print(f"    [ERROR] Error creating 2D PDP plots for {target_name}: {e}")

def create_feature_importance_analysis(models, df, target_cols, static_cols, ts_cols, feature_cols):
    """Create comprehensive feature importance analysis - excludes target-derived features to prevent data leakage"""
    print("[INFO] Creating comprehensive feature importance analysis...")
    
    # Get target-derived features to exclude
    target_derived_features = get_target_derived_features(target_cols)
    
    # Filter feature_cols to exclude target-derived features
    safe_feature_cols = [col for col in feature_cols if col not in target_derived_features]
    print(f"  Excluding {len(target_derived_features)} target-derived features to prevent data leakage")
    print(f"  Using {len(safe_feature_cols)} safe features for importance analysis")
    
    # Prepare data with safe features only
    X = df[safe_feature_cols].values
    y = df[target_cols].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create new scaler for safe features
    X_train_scaled, X_test_scaled, safe_scaler_X = create_safe_scaler_and_transform(X_train, X_test, safe_feature_cols)
    
    # Get feature indices for safe features
    static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
    ts_indices = [i for i, col in enumerate(safe_feature_cols) if col in ts_cols]
    static_feature_names = [safe_feature_cols[i] for i in static_indices]
    ts_feature_names = [safe_feature_cols[i] for i in ts_indices]
    
    # Create comprehensive feature importance analysis
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Feature Importance Analysis - XGB (Non-Static) vs Ridge (Static) - Surrogate Models', fontsize=16)
    
    # Collect importance data
    xgb_importance_data = []  # Non-static features only for XGB
    ridge_importance_data = []  # Static features only for Ridge
    target_names = []
    
    for i, target_name in enumerate(target_cols[:3]):
        print(f"  Analyzing feature importance for {target_name}...")
        
        try:
            # Get the target-specific scaler_y
            model_data = models[target_name]
            scaler_y = model_data['scaler_y']
            
            # Scale target data using the same scaler as training
            y_train_clean = np.nan_to_num(y_train[:, i], nan=0.0, posinf=1e6, neginf=-1e6)
            y_test_clean = np.nan_to_num(y_test[:, i], nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Check for constant target
            if np.var(y_train_clean) == 0:
                print(f"      Warning: {target_name} has zero variance, adding small noise...")
                y_train_clean += np.random.normal(0, 1e-8, len(y_train_clean))
                y_test_clean += np.random.normal(0, 1e-8, len(y_test_clean))
            
            y_train_scaled = scaler_y.transform(y_train_clean.reshape(-1, 1)).flatten()
            y_test_scaled = scaler_y.transform(y_test_clean.reshape(-1, 1)).flatten()
            
            # Train surrogate models on safe features
            if XGBOOST_AVAILABLE:
                surrogate_xgb = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=1,
                    eval_metric='rmse'
                )
                surrogate_xgb.fit(X_train_scaled, y_train_scaled)
            else:
                surrogate_xgb = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
                surrogate_xgb.fit(X_train_scaled, y_train_scaled)
            
            surrogate_ridge = Ridge(alpha=1.0, random_state=42)
            surrogate_ridge.fit(X_train_scaled, y_train_scaled)
            
            # XGBoost feature importance - NON-STATIC features only (excluding target-derived features)
            if hasattr(surrogate_xgb, 'feature_importances_'):
                xgb_importance = surrogate_xgb.feature_importances_
                # Map original feature indices to safe feature indices
                safe_ts_indices = [i for i, col in enumerate(safe_feature_cols) if col in ts_cols]
                if safe_ts_indices:
                    xgb_importance_data.append(xgb_importance[safe_ts_indices])
                else:
                    xgb_importance_data.append(np.array([]))
            else:
                # Fallback for non-tree models
                safe_ts_indices = [i for i, col in enumerate(safe_feature_cols) if col in ts_cols]
                xgb_importance_data.append(np.ones(len(safe_ts_indices)) / max(1, len(safe_ts_indices)))
            
            # Ridge Regression feature importance - STATIC features only (excluding target-derived features)
            ridge_coef = np.abs(surrogate_ridge.coef_)
            safe_static_indices = [i for i, col in enumerate(safe_feature_cols) if col in static_cols]
            if safe_static_indices:
                ridge_importance_data.append(ridge_coef[safe_static_indices])
            else:
                ridge_importance_data.append(np.array([]))
            
            target_names.append(target_name)
            
        except Exception as e:
            print(f"    [ERROR] Error analyzing {target_name}: {e}")
    
    if xgb_importance_data and ridge_importance_data:
        # Plot 1: XGBoost importance heatmap (NON-STATIC features, excluding target-derived)
        xgb_importance_matrix = np.array(xgb_importance_data)
        safe_ts_feature_names = [col for col in safe_feature_cols if col in ts_cols]
        if len(xgb_importance_matrix) > 0 and xgb_importance_matrix.shape[1] > 0:
            im1 = axes[0, 0].imshow(xgb_importance_matrix, cmap='YlOrRd', aspect='auto')
            axes[0, 0].set_title('XGBoost Surrogate - Non-Static Feature Importance (No Target Features)')
            axes[0, 0].set_xlabel('Non-Static Features')
            axes[0, 0].set_ylabel('Targets')
            axes[0, 0].set_xticks(range(len(safe_ts_feature_names)))
            axes[0, 0].set_xticklabels(safe_ts_feature_names, rotation=45, ha='right')
            axes[0, 0].set_yticks(range(len(target_names)))
            axes[0, 0].set_yticklabels(target_names)
            plt.colorbar(im1, ax=axes[0, 0])
        else:
            axes[0, 0].text(0.5, 0.5, 'No safe non-static features available', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('XGBoost Surrogate - No Safe Features')
        
        # Plot 2: Ridge Regression importance heatmap (STATIC features, excluding target-derived)
        ridge_importance_matrix = np.array(ridge_importance_data)
        safe_static_feature_names = [col for col in safe_feature_cols if col in static_cols]
        if len(ridge_importance_matrix) > 0 and ridge_importance_matrix.shape[1] > 0:
            im2 = axes[0, 1].imshow(ridge_importance_matrix, cmap='YlOrRd', aspect='auto')
            axes[0, 1].set_title('Ridge Surrogate - Static Feature Importance (No Target Features)')
            axes[0, 1].set_xlabel('Static Features')
            axes[0, 1].set_ylabel('Targets')
            axes[0, 1].set_xticks(range(len(safe_static_feature_names)))
            axes[0, 1].set_xticklabels(safe_static_feature_names, rotation=45, ha='right')
            axes[0, 1].set_yticks(range(len(target_names)))
            axes[0, 1].set_yticklabels(target_names)
            plt.colorbar(im2, ax=axes[0, 1])
        else:
            axes[0, 1].text(0.5, 0.5, 'No safe static features available', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Ridge Surrogate - No Safe Features')
        
        # Plot 3: Average importance across targets (XGB - Non-Static, excluding target-derived)
        if len(xgb_importance_matrix) > 0 and xgb_importance_matrix.shape[1] > 0:
            avg_xgb_importance = np.mean(xgb_importance_matrix, axis=0)
            sorted_indices = np.argsort(avg_xgb_importance)[::-1][:10]
            axes[1, 0].barh(range(len(sorted_indices)), avg_xgb_importance[sorted_indices])
            axes[1, 0].set_yticks(range(len(sorted_indices)))
            axes[1, 0].set_yticklabels([safe_ts_feature_names[i] for i in sorted_indices])
            axes[1, 0].set_title('Average XGB Surrogate Importance - Non-Static (Top 10, No Target Features)')
            axes[1, 0].invert_yaxis()
        else:
            axes[1, 0].text(0.5, 0.5, 'No safe non-static features', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('XGB Surrogate Importance - No Safe Features')
        
        # Plot 4: Average importance across targets (Ridge - Static, excluding target-derived)
        if len(ridge_importance_matrix) > 0 and ridge_importance_matrix.shape[1] > 0:
            avg_ridge_importance = np.mean(ridge_importance_matrix, axis=0)
            sorted_indices = np.argsort(avg_ridge_importance)[::-1][:10]
            axes[1, 1].barh(range(len(sorted_indices)), avg_ridge_importance[sorted_indices])
            axes[1, 1].set_yticks(range(len(sorted_indices)))
            axes[1, 1].set_yticklabels([safe_static_feature_names[i] for i in sorted_indices])
            axes[1, 1].set_title('Average Ridge Surrogate Importance - Static (Top 10, No Target Features)')
            axes[1, 1].invert_yaxis()
        else:
            axes[1, 1].text(0.5, 0.5, 'No safe static features', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Ridge Surrogate Importance - No Safe Features')
    
    plt.tight_layout()
    plt.savefig('./feature_importance_analysis_surrogate.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("[OK] Feature importance analysis completed using surrogate models!")

def create_error_analysis_plots(y_true, y_pred, target_name, model_name):
    """Create comprehensive error analysis plots"""
    errors = y_pred - y_true
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Error Analysis: {target_name} - {model_name}', fontsize=16)
    
    # 1. Prediction vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Prediction vs Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add R² to plot
    r2 = r2_score(y_true, y_pred)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error Distribution
    axes[1, 0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Prediction Errors')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    axes[1, 0].text(0.05, 0.95, f'MAE = {mae:.3f}\nRMSE = {rmse:.3f}', 
                    transform=axes[1, 0].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 4. Q-Q Plot for normality check
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./error_analysis_{target_name.replace(" ", "_")}_{model_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_model_comparison_plot(results, target_cols):
    """Create comprehensive model comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    # Prepare data for plotting
    models = ['XGB', 'Ridge', 'Ensemble']
    metrics = ['r2', 'mae', 'rmse', 'rmae']
    metric_names = ['R²', 'MAE', 'RMSE', 'RMAE']
    
    # 1. R² Comparison
    r2_data = {model: [] for model in models}
    for target in target_cols:
        if target in results:
            r2_data['XGB'].append(results[target]['xgb_metrics']['r2'])
            r2_data['Ridge'].append(results[target]['ridge_metrics']['r2'])
            r2_data['Ensemble'].append(results[target]['ensemble_metrics']['r2'])
    
    axes[0, 0].boxplot([r2_data[model] for model in models], labels=models)
    axes[0, 0].set_title('R² Score Distribution')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. RMAE Comparison
    rmae_data = {model: [] for model in models}
    for target in target_cols:
        if target in results:
            rmae_data['XGB'].append(results[target]['xgb_metrics']['rmae'])
            rmae_data['Ridge'].append(results[target]['ridge_metrics']['rmae'])
            rmae_data['Ensemble'].append(results[target]['ensemble_metrics']['rmae'])
    
    axes[0, 1].boxplot([rmae_data[model] for model in models], labels=models)
    axes[0, 1].set_title('RMAE Distribution')
    axes[0, 1].set_ylabel('RMAE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. MAE Comparison
    mae_data = {model: [] for model in models}
    for target in target_cols:
        if target in results:
            mae_data['XGB'].append(results[target]['xgb_metrics']['mae'])
            mae_data['Ridge'].append(results[target]['ridge_metrics']['mae'])
            mae_data['Ensemble'].append(results[target]['ensemble_metrics']['mae'])
    
    axes[1, 0].boxplot([mae_data[model] for model in models], labels=models)
    axes[1, 0].set_title('MAE Distribution')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. RMSE Comparison
    rmse_data = {model: [] for model in models}
    for target in target_cols:
        if target in results:
            rmse_data['XGB'].append(results[target]['xgb_metrics']['rmse'])
            rmse_data['Ridge'].append(results[target]['ridge_metrics']['rmse'])
            rmse_data['Ensemble'].append(results[target]['ensemble_metrics']['rmse'])
    
    axes[1, 1].boxplot([rmse_data[model] for model in models], labels=models)
    axes[1, 1].set_title('RMSE Distribution')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./model_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_single_visualization(args):
    """Create visualization for a single target (for parallel processing)"""
    target_name, ebm, static_cols, ebm_result, ax_idx = args
    
    # Get feature importance
    importance = ebm.feature_importances_
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1][:10]
    
    # Create subplot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot
    ax.barh(range(len(sorted_idx)), importance[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([static_cols[idx] for idx in sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{target_name}\nR² = {ebm_result["r2"]:.3f}, RMAE = {ebm_result["rmae"]:.3f}')
    ax.invert_yaxis()
    
    # Save individual plot
    plt.tight_layout()
    plt.savefig(f'./ebm_feature_importance_{target_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return target_name, importance, sorted_idx

def generate_error_analysis_for_models(models, df, target_cols, results):
    """Generate comprehensive error analysis for forecasting models"""
    print("[INFO] Generating error analysis for forecasting models...")
    
    # Prepare features for evaluation
    feature_cols = [col for col in df.columns if col not in target_cols + ['date']]
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use the same scaler_X as training (global for all targets)
    first_target = target_cols[0]
    scaler_X = models[first_target]['scaler_X']
    
    # Clean data before scaling
    X_train_clean = clean_data_for_scaling(X_train)
    X_test_clean = clean_data_for_scaling(X_test)
    
    X_train_scaled = scaler_X.transform(X_train_clean)
    X_test_scaled = scaler_X.transform(X_test_clean)
    
    # Generate predictions and error analysis for each target
    for i, target_name in enumerate(target_cols[:3]):  # Limit to 3 for detailed analysis
        print(f"  Analyzing errors for {target_name}...")
        
        # Get model data
        model_data = models[target_name]
        xgb_model = model_data['xgb']
        ridge = model_data['ridge']
        scaler_y = model_data['scaler_y']  # Use each target's own scaler_y
        
        # Generate predictions
        xgb_pred_scaled = xgb_model.predict(X_test_scaled)
        ridge_pred_scaled = ridge.predict(X_test_scaled)
        ensemble_pred_scaled = (xgb_pred_scaled + ridge_pred_scaled) / 2
        
        # Inverse transform predictions using the target's own scaler_y
        xgb_pred = scaler_y.inverse_transform(
            np.column_stack([xgb_pred_scaled if j == i else np.zeros_like(xgb_pred_scaled) 
                           for j in range(len(target_cols))])
        )[:, i]
        
        ridge_pred = scaler_y.inverse_transform(
            np.column_stack([ridge_pred_scaled if j == i else np.zeros_like(ridge_pred_scaled) 
                           for j in range(len(target_cols))])
        )[:, i]
        
        ensemble_pred = scaler_y.inverse_transform(
            np.column_stack([ensemble_pred_scaled if j == i else np.zeros_like(ensemble_pred_scaled) 
                           for j in range(len(target_cols))])
        )[:, i]
        
        # Get actual values (no scaling needed for actual values)
        y_actual = y_test[:, i]
        
        # Create error analysis plots
        create_error_analysis_plots(y_actual, xgb_pred, target_name, "XGBoost")
        create_error_analysis_plots(y_actual, ridge_pred, target_name, "Ridge Regression")
        create_error_analysis_plots(y_actual, ensemble_pred, target_name, "Ensemble")

def create_visualizations_parallel(ebm_models, static_cols, target_cols, ebm_results, n_jobs=None):
    """Create visualizations in parallel"""
    print("[PLOT] Creating visualizations in parallel...")
    
    if n_jobs is None:
        n_jobs = min(cpu_count(), len(target_cols))
    
    print(f"   Using {n_jobs} CPU cores for parallel processing")
    
    # Prepare arguments for parallel processing
    args_list = []
    for i, target_name in enumerate(target_cols[:6]):  # Limit to 6 for visualization
        args_list.append((target_name, ebm_models[target_name], static_cols, 
                         ebm_results[target_name], i))
    
    # Create visualizations in parallel
    start_time = time.time()
    with Pool(processes=n_jobs) as pool:
        viz_results = pool.map(create_single_visualization, args_list)
    
    viz_time = time.time() - start_time
    print(f"[OK] Visualizations created in {viz_time:.2f} seconds!")
    
    # Create combined visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for target_name, importance, sorted_idx in viz_results:
        ax = axes[viz_results.index((target_name, importance, sorted_idx))]
        
        # Plot
        ax.barh(range(len(sorted_idx)), importance[sorted_idx])
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([static_cols[idx] for idx in sorted_idx])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{target_name}\nR² = {ebm_results[target_name]["r2"]:.3f}, RMAE = {ebm_results[target_name]["rmae"]:.3f}')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('./ebm_feature_importance_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("[OK] All visualizations saved!")

def create_error_analysis_summary(results, ebm_results, target_cols):
    """Create a comprehensive error analysis summary"""
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)
    
    # Forecasting models error analysis
    print("\n[INFO] Forecasting Models Error Analysis:")
    print("-" * 40)
    
    for target_name in target_cols[:3]:  # Analyze top 3 targets
        if target_name in results:
            result = results[target_name]
            print(f"\n  {target_name}:")
            
            # Find best performing model
            xgb_r2 = result['xgb_metrics']['r2']
            ridge_r2 = result['ridge_metrics']['r2']
            ensemble_r2 = result['ensemble_metrics']['r2']
            
            best_r2 = max(xgb_r2, ridge_r2, ensemble_r2)
            best_model = 'XGB' if xgb_r2 == best_r2 else ('Ridge' if ridge_r2 == best_r2 else 'Ensemble')
            
            print(f"    Best Model: {best_model} (R² = {best_r2:.3f})")
            
            # Error patterns analysis
            xgb_rmae = result['xgb_metrics']['rmae']
            ridge_rmae = result['ridge_metrics']['rmae']
            ensemble_rmae = result['ensemble_metrics']['rmae']
            
            print(f"    RMAE Comparison:")
            print(f"      XGB: {xgb_rmae:.3f}, Ridge: {ridge_rmae:.3f}, Ensemble: {ensemble_rmae:.3f}")
            
            # Error consistency
            xgb_mape = result['xgb_metrics']['mape']
            ridge_mape = result['ridge_metrics']['mape']
            ensemble_mape = result['ensemble_metrics']['mape']
            
            print(f"    MAPE (Mean Absolute Percentage Error):")
            print(f"      XGB: {xgb_mape:.2f}%, Ridge: {ridge_mape:.2f}%, Ensemble: {ensemble_mape:.2f}%")
    
    # EBM models error analysis
    print("\n[INFO] EBM Models Error Analysis:")
    print("-" * 40)
    
    for target_name in target_cols[:3]:
        if target_name in ebm_results:
            result = ebm_results[target_name]
            print(f"\n  {target_name}:")
            print(f"    R² = {result['r2']:.3f}")
            print(f"    RMAE = {result['rmae']:.3f}")
            print(f"    MAPE = {result['mape']:.2f}%")
            print(f"    Max Error = {result['max_error']:.3f}")
            
            # Error quality assessment
            if result['r2'] > 0.8:
                print(f"    [OK] Excellent fit (R² > 0.8)")
            elif result['r2'] > 0.6:
                print(f"    [WARNING]  Good fit (R² > 0.6)")
            else:
                print(f"    [ERROR] Poor fit (R² < 0.6)")
            
            if result['mape'] < 10:
                print(f"    [OK] Low error rate (MAPE < 10%)")
            elif result['mape'] < 20:
                print(f"    [WARNING]  Moderate error rate (MAPE < 20%)")
            else:
                print(f"    [ERROR] High error rate (MAPE > 20%)")
    
    print("\n[PLOT] Error Analysis Insights:")
    print("-" * 40)
    print("  • RMAE provides a more robust error measure than RMSE for outliers")
    print("  • MAPE shows percentage-based error for business interpretation")
    print("  • R² indicates model fit quality (closer to 1.0 is better)")
    print("  • Error distribution plots help identify systematic biases")
    print("  • Q-Q plots reveal if errors follow normal distribution")
    print("  • Residual plots show if model assumptions are violated")

def optimize_data_processing(df, target_cols, ts_cols, static_cols, lookback=7):
    """Optimized data processing with vectorized operations"""
    print("[PROCESS] Optimizing data processing...")
    
    # Vectorized lag creation
    print("  Creating lagged features...")
    for col in ts_cols + target_cols:
        lag_data = np.column_stack([df[col].shift(i) for i in range(1, lookback + 1)])
        lag_cols = [f'{col}_lag_{i}' for i in range(1, lookback + 1)]
        df[lag_cols] = lag_data
    
    # Vectorized rolling statistics
    print("  Creating rolling statistics...")
    rolling_data = {}
    for col in target_cols:
        rolling_data[f'{col}_rolling_mean_7'] = df[col].rolling(window=7).mean()
        rolling_data[f'{col}_rolling_std_7'] = df[col].rolling(window=7).std()
    
    # Add rolling data to dataframe
    for col, data in rolling_data.items():
        df[col] = data
    
    # Drop rows with NaN values from lagged features
    df = df.dropna()
    
    print(f"[OK] Optimized data processing completed: {len(df)} rows after feature engineering")
    return df

def main():
    """Main execution with parallel processing"""
    print("="*60)
    print("ENHANCED EXPLAINABLE FORECASTING ANALYSIS")
    print("="*60)
    
    # Get system info
    n_cores = cpu_count()
    print(f"[SYSTEM]  System has {n_cores} CPU cores available")
    
    try:
        start_time = time.time()
        
        # Step 1: Load data
        df, target_cols, static_cols, ts_cols = load_and_prepare_data()
        
        # Step 2: Optimized data processing
        df = optimize_data_processing(df, target_cols, ts_cols, static_cols)
        
        # Step 3: Train forecasting models in parallel
        models, results = train_forecasting_models_parallel(df, target_cols, static_cols, ts_cols)
        
        # Step 4: Apply enhanced data cleaning for explainability
        df_clean = clean_data_for_explainability(df, target_cols)
        
        # Step 5: Generate parallel forecasts
        forecasts = generate_parallel_forecasts(models, df_clean, target_cols)
        
        # Step 6: Create comprehensive explainable analysis
        feature_cols = [col for col in df_clean.columns if col not in target_cols + ['date']]
        shap_results = create_shap_explanations(models, df_clean, target_cols, static_cols, feature_cols)
        lime_results = create_lime_explanations(models, df_clean, target_cols, static_cols, feature_cols, results)
        create_partial_dependence_plots(models, df_clean, target_cols, static_cols, feature_cols)
        create_2d_partial_dependence_plots(models, df_clean, target_cols, static_cols, ts_cols, feature_cols)
        create_feature_importance_analysis(models, df_clean, target_cols, static_cols, ts_cols, feature_cols)
        
        # Step 7: Generate comprehensive error analysis
        generate_error_analysis_for_models(models, df_clean, target_cols, results)
        
        # Step 8: Create model comparison plots
        create_model_comparison_plot(results, target_cols)
        
        # Step 9: Generate comprehensive error analysis summary
        create_error_analysis_summary(results, {}, target_cols)
        
        total_time = time.time() - start_time
        
        # Step 10: Print comprehensive results
        print("\n" + "="*60)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*60)
        
        print(f"\n[TIME]  Total execution time: {total_time:.2f} seconds")
        print(f"[SYSTEM]  CPU cores utilized: {min(n_cores, len(target_cols))}")
        
        print("\n[PLOT] Forecasting Performance (Comprehensive Metrics):")
        print("-" * 60)
        for target_name, result in results.items():
            print(f"\n  {target_name}:")
            print(f"    XGBoost:")
            print(f"      R² = {result['xgb_metrics']['r2']:.3f}, MAE = {result['xgb_metrics']['mae']:.3f}")
            print(f"      RMSE = {result['xgb_metrics']['rmse']:.3f}, RMAE = {result['xgb_metrics']['rmae']:.3f}")
            print(f"    Ridge Regression:")
            print(f"      R² = {result['ridge_metrics']['r2']:.3f}, MAE = {result['ridge_metrics']['mae']:.3f}")
            print(f"      RMSE = {result['ridge_metrics']['rmse']:.3f}, RMAE = {result['ridge_metrics']['rmae']:.3f}")
            print(f"    Ensemble:")
            print(f"      R² = {result['ensemble_metrics']['r2']:.3f}, MAE = {result['ensemble_metrics']['mae']:.3f}")
            print(f"      RMSE = {result['ensemble_metrics']['rmse']:.3f}, RMAE = {result['ensemble_metrics']['rmae']:.3f}")
            print(f"    Best Model: {result['best_model']}")
        
        print("\n[INFO] Explainable Analysis Results:")
        print("-" * 60)
        if shap_results:
            print("  [OK] SHAP analysis completed for static feature importance")
            for target_name, result in shap_results.items():
                if 'error' not in result:
                    print(f"    {target_name}: SHAP explanations generated")
                    print(f"      Top static feature impact analyzed")
        
        if lime_results:
            print("  [OK] LIME analysis completed for local explanations")
            for target_name, result in lime_results.items():
                if 'error' not in result:
                    print(f"    {target_name}: {len(result.get('explanations', []))} local explanations")
        
        print("  [OK] Partial Dependence Plots created for feature effects")
        print("  [OK] 2D Partial Dependence Plots created for combined static/non-static effects")
        print("  [OK] Feature importance analysis completed (XGB: non-static, Ridge: static)")
        
        print("\n[FORECAST] Forecast Results:")
        print("-" * 60)
        for target_name, forecast in forecasts.items():
            print(f"  {target_name}: {forecast:.2f}")
        
        print("\n📈 Enhanced Analysis Features:")
        print("-" * 60)
        print("  • SHAP explanations for model interpretability")
        print("  • LIME local explanations for individual predictions")
        print("  • Partial Dependence Plots for static feature effects")
        print("  • 2D Partial Dependence Plots for combined static/non-static feature effects")
        print("  • Feature importance analysis (XGB: non-static features, Ridge: static features)")
        print("  • Error analysis plots generated for top 3 targets")
        print("  • Model comparison plots created")
        print("  • Comprehensive metrics calculated including RMAE")
        print("  • Residual analysis and Q-Q plots generated")
        
        print("\n[OK] Comprehensive parallel multitarget analysis completed successfully!")
        print("   All visualizations and error analysis saved to current directory.")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
