# XGBoost and L1/L2 Regression Comparison

This script provides a comprehensive comparison between XGBoost, L1/L2 regularization methods, and baseline models for time series forecasting in petroleum refining processes.

## Overview

The comparison script evaluates the performance of:
- **XGBoost** with hyperparameter tuning
- **L1 Regularization (Lasso)** 
- **L2 Regularization (Ridge)**
- **ElasticNet** (L1 + L2 combination)
- **Random Forest** (baseline)
- **Linear Regression** (baseline)
- **Ensemble** model combining all approaches

## Features

### 🚀 Performance
- **Parallel Processing**: Utilizes multiple CPU cores for efficient training
- **Hyperparameter Tuning**: Grid search optimization for all models
- **Comprehensive Metrics**: R², MAE, RMSE, RMAE, MAPE, Max Error

### 📊 Analysis
- **Feature Importance Comparison**: Across different model types
- **Performance Visualization**: Box plots and detailed comparisons
- **Detailed Results Table**: CSV export with all metrics
- **Forecast Generation**: Using all trained models

### 🔧 Technical Features
- **Time Series Features**: Lagged variables and rolling statistics
- **Data Scaling**: StandardScaler for features and targets
- **Cross-Validation**: 3-fold CV for hyperparameter tuning
- **Error Handling**: Robust error handling and logging

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_xgboost_comparison.txt
```

2. Ensure data file is available:
   - `final_concatenated_data_mice_imputed.csv`

## Usage

### Quick Start
```bash
python run_xgboost_comparison.py
```

### Direct Execution
```bash
python xgboost_l1_l2_comparison.py
```

## Output Files

The script generates several output files:

### 📈 Visualizations
- `model_performance_comparison.png` - Comprehensive performance comparison
- `feature_importance_comparison_*.png` - Feature importance for top 3 targets

### 📋 Data
- `model_performance_comparison.csv` - Detailed results table

### 🔍 Analysis
- Console output with detailed performance metrics
- Best model recommendations for each target

## Model Details

### XGBoost
- **Hyperparameters Tuned**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- **Early Stopping**: 10 rounds
- **Evaluation Metric**: RMSE

### L1/L2 Regularization
- **Lasso (L1)**: Alpha values [0.001, 0.01, 0.1, 1.0, 10.0]
- **Ridge (L2)**: Alpha values [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
- **ElasticNet**: Alpha [0.001, 0.01, 0.1, 1.0] × L1_ratio [0.1, 0.3, 0.5, 0.7, 0.9]

### Baseline Models
- **Random Forest**: 100 estimators, default parameters
- **Linear Regression**: Standard OLS regression

## Performance Metrics

The script evaluates models using:

- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **RMAE**: Root Mean Absolute Error (robust to outliers)
- **MAPE**: Mean Absolute Percentage Error
- **Max Error**: Maximum prediction error

## Target Variables

The script analyzes 11 target variables:
- Total_Stabilized_Naphtha_Product_Flowrate
- Total_Kerosene_Product_Flowrate
- Jet_Fuel_Product_Train1_Flowrate
- Total_Light_Diesel_Product_Flowrate
- Total_Heavy_Diesel_Product_Flowrate
- Total_Atmospheric_Residue_Flowrate
- Blend_Yield_Gas & LPG
- Blend_Yield_Kerosene
- Blend_Yield_Light Diesel
- Blend_Yield_Heavy Diesel
- Blend_Yield_RCO

## Time Series Features

The script creates comprehensive time series features:
- **Lagged Variables**: 1-7 day lags for all time series
- **Rolling Statistics**: 7-day rolling mean and standard deviation
- **Static Features**: Crude oil properties and blend characteristics

## Parallel Processing

The script uses multiprocessing to:
- Train models for different targets in parallel
- Utilize all available CPU cores
- Reduce total execution time

## Error Handling

- Graceful handling of missing dependencies
- Robust error reporting
- Fallback mechanisms for failed models

## Comparison with Original Script

This script extends the original `simple_forecasting_script_parallel.py` by:
- Adding XGBoost with hyperparameter tuning
- Including L1/L2 regularization methods
- Providing comprehensive model comparison
- Enhanced feature importance analysis
- Detailed performance metrics

## Requirements

- Python 3.7+
- 8GB+ RAM recommended
- Multi-core CPU for parallel processing
- XGBoost library

## Troubleshooting

### Common Issues

1. **XGBoost Import Error**
   ```bash
   pip install xgboost
   ```

2. **Memory Issues**
   - Reduce number of parallel processes
   - Use smaller hyperparameter grids

3. **Data File Missing**
   - Ensure `final_concatenated_data_mice_imputed.csv` is in the current directory

### Performance Tips

- Use SSD storage for faster I/O
- Ensure sufficient RAM (8GB+ recommended)
- Close other applications to free up CPU cores

## Results Interpretation

### Best Model Selection
- Higher R² scores indicate better fit
- Lower MAE/RMSE indicate better accuracy
- Lower MAPE indicates better percentage accuracy
- RMAE is more robust to outliers than RMSE

### Feature Importance
- Compare importance across different model types
- Identify consistent important features
- Understand model-specific feature preferences

### Ensemble Performance
- Ensemble often outperforms individual models
- Provides robust predictions by combining strengths
- Reduces overfitting risk

## Future Enhancements

Potential improvements:
- Bayesian optimization for hyperparameter tuning
- Additional regularization methods (ElasticNet variants)
- Time series cross-validation
- Feature selection based on importance
- Model stacking techniques





