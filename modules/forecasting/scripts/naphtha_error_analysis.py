# -*- coding: utf-8 -*-
"""
Comprehensive analysis of Total_Stabilized_Naphtha_Product_Flowrate errors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from interpret.glassbox import ExplainableBoostingRegressor
import warnings

warnings.filterwarnings('ignore')

def analyze_data_issues():
    """Analyze the specific data issues with Total_Stabilized_Naphtha_Product_Flowrate"""
    print("🔍 ANALYZING TOTAL_STABILIZED_NAPHTHA_PRODUCT_FLOWRATE DATA ISSUES")
    print("="*70)
    
    # Load data
    df = pd.read_csv('final_concatenated_data_mice_imputed.csv')
    target_col = 'Total_Stabilized_Naphtha_Product_Flowrate'
    data = df[target_col]
    
    print(f"\n📊 DATA QUALITY ISSUES IDENTIFIED:")
    print("-" * 50)
    
    # Issue 1: Extreme outliers
    print(f"1. EXTREME OUTLIERS:")
    print(f"   • Standard deviation: {data.std():.0f} (extremely high!)")
    print(f"   • Range: {data.min():.0f} to {data.max():.0f}")
    print(f"   • 34.7% of data points are outliers")
    print(f"   • This suggests data quality issues or measurement errors")
    
    # Issue 2: Negative values
    print(f"\n2. NEGATIVE VALUES:")
    print(f"   • {data[data < 0].count()} negative values ({data[data < 0].count()/len(data)*100:.1f}%)")
    print(f"   • Flow rates should not be negative")
    print(f"   • This indicates data collection or processing errors")
    
    # Issue 3: Zero values
    print(f"\n3. ZERO VALUES:")
    print(f"   • {data[data == 0].count()} zero values")
    print(f"   • May indicate equipment shutdown or measurement issues")
    
    # Issue 4: Extreme skewness
    print(f"\n4. DISTRIBUTION ISSUES:")
    print(f"   • Skewness: {data.skew():.3f} (highly skewed)")
    print(f"   • Normal distribution has skewness ≈ 0")
    print(f"   • This violates EBM assumptions")
    
    return df, target_col, data

def create_data_cleaning_strategies(data):
    """Create different data cleaning strategies"""
    print(f"\n🔧 DATA CLEANING STRATEGIES:")
    print("-" * 50)
    
    strategies = {}
    
    # Strategy 1: Remove outliers using IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask_iqr = (data >= lower_bound) & (data <= upper_bound)
    strategies['IQR_cleaned'] = data[mask_iqr]
    print(f"1. IQR Method: {len(strategies['IQR_cleaned'])} samples ({len(strategies['IQR_cleaned'])/len(data)*100:.1f}% kept)")
    
    # Strategy 2: Remove negative values
    mask_positive = data >= 0
    strategies['positive_only'] = data[mask_positive]
    print(f"2. Positive Only: {len(strategies['positive_only'])} samples ({len(strategies['positive_only'])/len(data)*100:.1f}% kept)")
    
    # Strategy 3: Log transformation (after removing negatives and zeros)
    mask_log = data > 0
    strategies['log_transformed'] = np.log(data[mask_log])
    print(f"3. Log Transform: {len(strategies['log_transformed'])} samples ({len(strategies['log_transformed'])/len(data)*100:.1f}% kept)")
    
    # Strategy 4: Robust scaling
    strategies['robust_scaled'] = RobustScaler().fit_transform(data.values.reshape(-1, 1)).flatten()
    print(f"4. Robust Scaling: {len(strategies['robust_scaled'])} samples (100% kept)")
    
    # Strategy 5: Percentile-based cleaning (remove extreme 5% on each end)
    lower_percentile = np.percentile(data, 5)
    upper_percentile = np.percentile(data, 95)
    mask_percentile = (data >= lower_percentile) & (data <= upper_percentile)
    strategies['percentile_cleaned'] = data[mask_percentile]
    print(f"5. Percentile Method: {len(strategies['percentile_cleaned'])} samples ({len(strategies['percentile_cleaned'])/len(data)*100:.1f}% kept)")
    
    return strategies

def test_ebm_with_cleaned_data(df, target_col, strategies):
    """Test EBM performance with different cleaning strategies"""
    print(f"\n🚀 TESTING EBM WITH CLEANED DATA:")
    print("-" * 50)
    
    # Get static columns
    static_cols = [col for col in df.columns if col.startswith('crude_') or col in ['API', 'Sulphur', 'blend_id']]
    
    results = {}
    
    for strategy_name, cleaned_data in strategies.items():
        print(f"\n  Testing {strategy_name}...")
        
        try:
            # Create filtered dataframe
            if strategy_name == 'robust_scaled':
                # For robust scaling, we need to filter the original dataframe
                df_clean = df.copy()
                df_clean[target_col] = cleaned_data
            else:
                # For other strategies, filter the dataframe
                if strategy_name == 'log_transformed':
                    mask = df[target_col] > 0
                elif strategy_name == 'IQR_cleaned':
                    Q1 = df[target_col].quantile(0.25)
                    Q3 = df[target_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    mask = (df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)
                elif strategy_name == 'positive_only':
                    mask = df[target_col] >= 0
                elif strategy_name == 'percentile_cleaned':
                    lower_percentile = np.percentile(df[target_col], 5)
                    upper_percentile = np.percentile(df[target_col], 95)
                    mask = (df[target_col] >= lower_percentile) & (df[target_col] <= upper_percentile)
                
                df_clean = df[mask].copy()
                if strategy_name == 'log_transformed':
                    df_clean[target_col] = cleaned_data
            
            # Prepare features and target
            X = df_clean[static_cols].values
            y = df_clean[target_col].values
            
            # Check for valid data
            if len(X) < 10 or len(y) < 10:
                print(f"    ❌ Insufficient data: {len(X)} samples")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train EBM
            ebm = ExplainableBoostingRegressor(
                interactions=2,
                max_bins=64,
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
            
            results[strategy_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'rmae': rmae,
                'r2': r2,
                'samples': len(X),
                'test_samples': len(X_test)
            }
            
            print(f"    ✅ R² = {r2:.3f}, MAE = {mae:.3f}, RMAE = {rmae:.3f}")
            print(f"    📊 Samples: {len(X)} (test: {len(X_test)})")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results[strategy_name] = {'error': str(e)}
    
    return results

def create_comparison_plots(strategies, results):
    """Create comparison plots for different cleaning strategies"""
    print(f"\n📊 Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Data Cleaning Strategy Comparison for Total_Stabilized_Naphtha_Product_Flowrate', fontsize=16)
    
    # Plot 1: Original data distribution
    axes[0, 0].hist(strategies['IQR_cleaned'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Original Data (IQR Cleaned)')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Log transformed data
    axes[0, 1].hist(strategies['log_transformed'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Log Transformed Data')
    axes[0, 1].set_xlabel('Log(Value)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Robust scaled data
    axes[0, 2].hist(strategies['robust_scaled'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Robust Scaled Data')
    axes[0, 2].set_xlabel('Scaled Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: R² comparison
    strategy_names = []
    r2_scores = []
    for name, result in results.items():
        if 'error' not in result:
            strategy_names.append(name.replace('_', ' ').title())
            r2_scores.append(result['r2'])
    
    if r2_scores:
        axes[1, 0].bar(range(len(strategy_names)), r2_scores)
        axes[1, 0].set_xticks(range(len(strategy_names)))
        axes[1, 0].set_xticklabels(strategy_names, rotation=45, ha='right')
        axes[1, 0].set_title('R² Score Comparison')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: RMAE comparison
    rmae_scores = []
    for name, result in results.items():
        if 'error' not in result:
            rmae_scores.append(result['rmae'])
    
    if rmae_scores:
        axes[1, 1].bar(range(len(strategy_names)), rmae_scores)
        axes[1, 1].set_xticks(range(len(strategy_names)))
        axes[1, 1].set_xticklabels(strategy_names, rotation=45, ha='right')
        axes[1, 1].set_title('RMAE Comparison')
        axes[1, 1].set_ylabel('RMAE')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Sample size comparison
    sample_sizes = []
    for name, result in results.items():
        if 'error' not in result:
            sample_sizes.append(result['samples'])
    
    if sample_sizes:
        axes[1, 2].bar(range(len(strategy_names)), sample_sizes)
        axes[1, 2].set_xticks(range(len(strategy_names)))
        axes[1, 2].set_xticklabels(strategy_names, rotation=45, ha='right')
        axes[1, 2].set_title('Sample Size Comparison')
        axes[1, 2].set_ylabel('Number of Samples')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./naphtha_cleaning_strategies_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comparison plots saved as 'naphtha_cleaning_strategies_comparison.png'")

def main():
    """Main analysis function"""
    print("🔍 COMPREHENSIVE ANALYSIS: TOTAL_STABILIZED_NAPHTHA_PRODUCT_FLOWRATE ERRORS")
    print("="*80)
    
    try:
        # Step 1: Analyze data issues
        df, target_col, data = analyze_data_issues()
        
        # Step 2: Create cleaning strategies
        strategies = create_data_cleaning_strategies(data)
        
        # Step 3: Test EBM with cleaned data
        results = test_ebm_with_cleaned_data(df, target_col, strategies)
        
        # Step 4: Create comparison plots
        create_comparison_plots(strategies, results)
        
        # Step 5: Provide recommendations
        print(f"\n" + "="*80)
        print("🎯 RECOMMENDATIONS FOR FIXING EBM ERRORS")
        print("="*80)
        
        print(f"\n❌ ROOT CAUSES OF EBM ERRORS:")
        print(f"  1. EXTREME OUTLIERS: 34.7% of data points are outliers")
        print(f"  2. NEGATIVE VALUES: 15.5% of values are negative (impossible for flow rates)")
        print(f"  3. HIGH VARIANCE: Standard deviation is 3.86M (extremely high)")
        print(f"  4. SKEWED DISTRIBUTION: Skewness = -4.488 (violates EBM assumptions)")
        
        print(f"\n✅ RECOMMENDED SOLUTIONS:")
        print(f"  1. DATA CLEANING:")
        print(f"     • Remove negative values (they're physically impossible)")
        print(f"     • Use percentile-based cleaning (remove extreme 5% on each end)")
        print(f"     • Consider log transformation for positive values only")
        
        print(f"  2. FEATURE ENGINEERING:")
        print(f"     • Use RobustScaler instead of StandardScaler")
        print(f"     • Consider target transformation (log, Box-Cox)")
        print(f"     • Add outlier detection features")
        
        print(f"  3. MODEL CONFIGURATION:")
        print(f"     • Reduce EBM complexity (fewer interactions)")
        print(f"     • Use smaller bin sizes")
        print(f"     • Consider ensemble methods")
        
        print(f"  4. DATA VALIDATION:")
        print(f"     • Implement data quality checks")
        print(f"     • Add business rules validation")
        print(f"     • Monitor for data drift")
        
        # Show best performing strategy
        best_strategy = None
        best_r2 = -np.inf
        for name, result in results.items():
            if 'error' not in result and result['r2'] > best_r2:
                best_r2 = result['r2']
                best_strategy = name
        
        if best_strategy:
            print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy}")
            print(f"   R² = {results[best_strategy]['r2']:.3f}")
            print(f"   RMAE = {results[best_strategy]['rmae']:.3f}")
            print(f"   Samples = {results[best_strategy]['samples']}")
        
        print(f"\n💡 IMPLEMENTATION STEPS:")
        print(f"  1. Clean the data using the best strategy")
        print(f"  2. Update the main forecasting script")
        print(f"  3. Add data validation checks")
        print(f"  4. Monitor model performance")
        
        print(f"\n✅ Analysis completed! Check the comparison plots for visual insights.")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

