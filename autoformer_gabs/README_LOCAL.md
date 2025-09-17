# Local Autoformer EBM Analysis

This is a modified version of the Autoformer EBM notebook that runs in your local environment without Colab dependencies.

## Files

- `local_autoformer_ebm_script.py` - Main analysis script
- `requirements_local.txt` - Python package dependencies
- `run_local_analysis.py` - Automated setup and run script
- `run_analysis.bat` - Windows batch file for easy execution
- `README_LOCAL.md` - This file

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Data file**: `final_concatenated_data_mice_imputed.csv` in the same directory
3. **Internet connection** for downloading packages

## Quick Start

### Option 1: Automated (Recommended)
1. Double-click `run_analysis.bat` (Windows) or run `python run_local_analysis.py`
2. The script will automatically install dependencies and run the analysis

### Option 2: Manual
1. Install dependencies:
   ```bash
   pip install -r requirements_local.txt
   ```

2. Run the analysis:
   ```bash
   python local_autoformer_ebm_script.py
   ```

## What the Analysis Does

1. **Data Loading**: Loads and preprocesses the concatenated interpolated data
2. **Autoformer Training**: Trains a Hugging Face Autoformer for multivariate time series forecasting
3. **EBM Training**: Trains Explainable Boosting Machine models for blend effects explanation
4. **Visualization**: Creates feature importance plots
5. **Results**: Provides comprehensive performance metrics and explanations

## Output Files

- `./dataset/custom/custom.csv` - Processed dataset for Autoformer
- `./dataset/custom/dataset_info.json` - Dataset metadata
- `./dataset/custom/dataset_summary.txt` - Dataset summary
- `./autoformer_results/` - Trained model checkpoints
- `./logs/` - Training logs
- `./ebm_feature_importance.png` - Feature importance visualization

## Key Features

- **Multivariate Prediction**: Predicts all 11 target variables simultaneously
- **Hugging Face Autoformer**: Uses the latest transformer-based time series model
- **EBM Explanation**: Explains how blend characteristics affect each target
- **Local Execution**: No Colab or cloud dependencies required
- **Comprehensive Evaluation**: Multiple metrics and visualizations

## Troubleshooting

### Common Issues

1. **"File not found" error**: Ensure `final_concatenated_data_mice_imputed.csv` is in the same directory
2. **Package installation fails**: Try updating pip: `python -m pip install --upgrade pip`
3. **Memory issues**: The script uses reduced batch sizes for local execution. If you have more memory, you can increase them in the script
4. **CUDA errors**: The script will automatically use CPU if CUDA is not available

### Performance Notes

- The script uses reduced parameters (5 epochs, smaller batch sizes) for local execution
- Training time depends on your hardware (CPU/GPU)
- For production use, consider increasing epochs and batch sizes

## Customization

You can modify the following parameters in `local_autoformer_ebm_script.py`:

- `seq_len`: Sequence length for time series (default: 60)
- `pred_len`: Prediction length (default: 7)
- `num_train_epochs`: Training epochs (default: 5)
- `per_device_train_batch_size`: Batch size (default: 8)

## Support

If you encounter issues:
1. Check that all dependencies are installed correctly
2. Ensure the data file is in the correct location
3. Check the console output for specific error messages
4. Verify Python version compatibility (3.8+)

