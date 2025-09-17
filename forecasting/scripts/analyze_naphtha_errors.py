# -*- coding: utf-8 -*-
"""
Diagnostic script to analyze errors in Total_Stabilized_Naphtha_Product_Flowrate for EBM model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from interpret.glassbox import ExplainableBoostingRegressor
import warnings

warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load data and analyze Total_Stabilized_Naphtha_Product_Flowrate specifically"""
    print("🔍 Loading and analyzing Total_Stabilized_Naphtha_Product_Flowrate...")
    
    # Load data
    df = pd.read_csv('final_concatenated_data_mice_imputed.csv')
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Target column
    target_col = 'Total_Stabilized_Naphtha_Product_Flowrate'
    
    # Check if target exists
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found!")
        print("Available columns:", df.columns.tolist())
        return None, None, None
    
    # Analyze target variable
    print(f"\n📊 Analysis of {target_col}:")
    print("-" * 50)
    
    target_data = df[target_col]
    print(f"Data type: {target_data.dtype}")
    print(f"Non-null values: {target_data.count()}")
    print(f"Null values: {target_data.isnull().sum()}")
    print(f"Zero values: {(target_data == 0).sum()}")
    print(f"Negative values: {(target_data < 0).sum()}")
    
    print(f"\nDescriptive Statistics:")
    print(f"  Mean: {target_data.mean():.3f}")
    print(f"  Median: {target_data.median():.3f}")
    print(f"  Std: {target_data.std():.3f}")
    print(f"  Min: {target_data.min():.3f}")
    print(f"  Max: {target_data.max():.3f}")
    print(f"  Range: {target_data.max() - target_data.min():.3f}")
    
    # Check for outliers using IQR method
    Q1 = target_data.quantile(0.25)
    Q3 = target_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = target_data[(target_data < lower_bound) | (target_data > upper_bound)]
    print(f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(target_data)*100:.1f}%)")
    
    # Check data distribution
    print(f"\nDistribution Analysis:")
    print(f"  Skewness: {target_data.skew():.3f}")
    print(f"  Kurtosis: {target_data.kurtosis():.3f}")
    
    # Identify static columns (crude oil features)
    static_cols = [col for col in df.columns if col.startswith('crude_') or col in ['API', 'Sulphur', 'blend_id']]
    print(f"\nStatic features available: {len(static_cols)}")
    print(f"Static columns: {static_cols[:5]}...")  # Show first 5
    
    return df, target_col, static_cols

def analyze_feature_relationships(df, target_col, static_cols):
    """Analyze relationships between features and target"""
    print(f"\n🔗 Feature-Target Relationship Analysis:")
    print("-" * 50)
    
    # Calculate correlations with target
    correlations = {}
    for col in static_cols:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            corr = df[col].corr(df[target_col])
            if not np.isnan(corr):
                correlations[col] = corr
    
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"Top 10 most correlated features:")
    for i, (feature, corr) in enumerate(sorted_corrs[:10]):
        print(f"  {i+1:2d}. {feature:30s}: {corr:6.3f}")
    
    # Check for constant features
    constant_features = []
    for col in static_cols:
        if col in df.columns:
            if df[col].nunique() <= 1:
                constant_features.append(col)
    
    if constant_features:
        print(f"\n⚠️  Constant features found: {len(constant_features)}")
        print(f"Constant features: {constant_features[:5]}...")
    else:
        print(f"\n✅ No constant features found")
    
    return correlations, constant_features

def diagnose_ebm_training(df, target_col, static_cols):
    """Diagnose EBM training issues"""
    print(f"\n🔧 EBM Training Diagnosis:")
    print("-" * 50)
    
    # Prepare data
    X_static = df[static_cols].values
    y_target = df[target_col].values
    
    print(f"Feature matrix shape: {X_static.shape}")
    print(f"Target vector shape: {y_target.shape}")
    
    # Check for missing values
    missing_features = np.isnan(X_static).sum(axis=0)
    missing_target = np.isnan(y_target).sum()
    
    print(f"Missing values in features: {missing_features.sum()}")
    print(f"Missing values in target: {missing_target}")
    
    if missing_features.sum() > 0:
        print(f"Features with missing values:")
        for i, missing_count in enumerate(missing_features):
            if missing_count > 0:
                print(f"  {static_cols[i]}: {missing_count} missing")
    
    # Check for infinite values
    inf_features = np.isinf(X_static).sum(axis=0)
    inf_target = np.isinf(y_target).sum()
    
    print(f"Infinite values in features: {inf_features.sum()}")
    print(f"Infinite values in target: {inf_target}")
    
    # Check feature variance
    feature_vars = np.var(X_static, axis=0)
    zero_var_features = np.where(feature_vars == 0)[0]
    
    if len(zero_var_features) > 0:
        print(f"Zero variance features: {len(zero_var_features)}")
        for idx in zero_var_features[:5]:  # Show first 5
            print(f"  {static_cols[idx]}")
    
    return X_static, y_target

def train_ebm_with_diagnostics(X_static, y_target, static_cols, target_col):
    """Train EBM with detailed diagnostics"""
    print(f"\n🚀 Training EBM with diagnostics:")
    print("-" * 50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_static, y_target, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    try:
        # Train EBM with different configurations
        configs = [
            {"interactions": 0, "max_bins": 32, "name": "Simple EBM"},
            {"interactions": 1, "max_bins": 64, "name": "Medium EBM"},
            {"interactions": 3, "max_bins": 128, "name": "Complex EBM"}
        ]
        
        results = {}
        
        for config in configs:
            print(f"\n  Training {config['name']}...")
            
            try:
                ebm = ExplainableBoostingRegressor(
                    interactions=config['interactions'],
                    max_bins=config['max_bins'],
                    max_interaction_bins=8,
                    random_state=42,
                    n_jobs=1
                )
                
                ebm.fit(X_train, y_train)
                y_pred = ebm.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                rmae = np.sqrt(mae)
                
                results[config['name']] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'rmae': rmae,
                    'r2': r2,
                    'model': ebm
                }
                
                print(f"    R² = {r2:.3f}, MAE = {mae:.3f}, RMAE = {rmae:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error training {config['name']}: {e}")
                results[config['name']] = {'error': str(e)}
        
        return results, X_test, y_test
        
    except Exception as e:
        print(f"❌ Error in EBM training: {e}")
        return None, None, None

def create_diagnostic_plots(df, target_col, static_cols, results, X_test, y_test):
    """Create diagnostic plots"""
    print(f"\n📊 Creating diagnostic plots...")
    
    # 1. Target distribution
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Diagnostic Analysis: {target_col}', fontsize=16)
    
    # Target histogram
    axes[0, 0].hist(df[target_col], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Target Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Target boxplot
    axes[0, 1].boxplot(df[target_col])
    axes[0, 1].set_title('Target Boxplot')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Target Q-Q plot
    from scipy import stats
    stats.probplot(df[target_col], dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Target Q-Q Plot')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Feature importance (if EBM trained successfully)
    if results and 'Simple EBM' in results and 'error' not in results['Simple EBM']:
        ebm = results['Simple EBM']['model']
        importance = ebm.feature_importances_
        sorted_idx = np.argsort(importance)[::-1][:10]
        
        axes[1, 0].barh(range(len(sorted_idx)), importance[sorted_idx])
        axes[1, 0].set_yticks(range(len(sorted_idx)))
        axes[1, 0].set_yticklabels([static_cols[idx] for idx in sorted_idx])
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Top 10 Feature Importance')
        axes[1, 0].invert_yaxis()
    
    # Prediction vs Actual (if EBM trained successfully)
    if results and 'Simple EBM' in results and 'error' not in results['Simple EBM']:
        ebm = results['Simple EBM']['model']
        y_pred = ebm.predict(X_test)
        
        axes[1, 1].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual')
        axes[1, 1].set_ylabel('Predicted')
        axes[1, 1].set_title('Prediction vs Actual')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = results['Simple EBM']['r2']
        axes[1, 1].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1, 1].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Error distribution (if EBM trained successfully)
    if results and 'Simple EBM' in results and 'error' not in results['Simple EBM']:
        ebm = results['Simple EBM']['model']
        y_pred = ebm.predict(X_test)
        errors = y_pred - y_test
        
        axes[1, 2].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 2].set_xlabel('Prediction Errors')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Error Distribution')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./naphtha_diagnostic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Diagnostic plots saved as 'naphtha_diagnostic_analysis.png'")

def main():
    """Main diagnostic function"""
    print("="*60)
    print("TOTAL STABILIZED NAPHTHA EBM ERROR DIAGNOSIS")
    print("="*60)
    
    try:
        # Step 1: Load and analyze data
        df, target_col, static_cols = load_and_analyze_data()
        if df is None:
            return
        
        # Step 2: Analyze feature relationships
        correlations, constant_features = analyze_feature_relationships(df, target_col, static_cols)
        
        # Step 3: Diagnose EBM training
        X_static, y_target = diagnose_ebm_training(df, target_col, static_cols)
        
        # Step 4: Train EBM with diagnostics
        results, X_test, y_test = train_ebm_with_diagnostics(X_static, y_target, static_cols, target_col)
        
        # Step 5: Create diagnostic plots
        create_diagnostic_plots(df, target_col, static_cols, results, X_test, y_test)
        
        # Step 6: Summary and recommendations
        print(f"\n" + "="*60)
        print("DIAGNOSIS SUMMARY & RECOMMENDATIONS")
        print("="*60)
        
        if results:
            print(f"\n🔍 EBM Training Results:")
            for config_name, result in results.items():
                if 'error' in result:
                    print(f"  {config_name}: ❌ FAILED - {result['error']}")
                else:
                    print(f"  {config_name}: ✅ SUCCESS")
                    print(f"    R² = {result['r2']:.3f}, MAE = {result['mae']:.3f}, RMAE = {result['rmae']:.3f}")
        
        print(f"\n💡 Potential Issues & Solutions:")
        print(f"  1. Data Quality Issues:")
        print(f"     - Check for outliers and extreme values")
        print(f"     - Verify data preprocessing (scaling, normalization)")
        print(f"     - Ensure no missing or infinite values")
        
        print(f"  2. Feature Engineering:")
        print(f"     - Remove constant features: {len(constant_features)} found")
        print(f"     - Consider feature selection based on correlation")
        print(f"     - Check for multicollinearity")
        
        print(f"  3. Model Configuration:")
        print(f"     - Try different EBM parameters")
        print(f"     - Consider feature scaling")
        print(f"     - Check target variable distribution")
        
        print(f"  4. Data Distribution:")
        print(f"     - Target may be highly skewed or have outliers")
        print(f"     - Consider log transformation if appropriate")
        print(f"     - Check for data leakage or temporal issues")
        
        print(f"\n✅ Diagnosis completed! Check 'naphtha_diagnostic_analysis.png' for visual analysis.")
        
    except Exception as e:
        print(f"❌ Error in diagnosis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

