# -*- coding: utf-8 -*-
"""
Local Autoformer EBM Script
Modified to run in local environment without Colab dependencies
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoformerConfig, AutoformerForPrediction, Trainer, TrainingArguments
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from interpret.glassbox import ExplainableBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load and prepare data for Autoformer training
    """
    print("🔍 Starting data preparation for Autoformer...")

    # Define the path to the MICE interpolated and concatenated data
    file_path = 'final_concatenated_data_mice_imputed.csv'

    # List of specified target columns
    target_cols_list = [
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

    # Load the combined CSV file with robust error handling
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File not found - {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded {file_path}: {len(df)} rows, {len(df.columns)} columns")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Error: File {file_path} is empty.")
    except Exception as e:
        raise RuntimeError(f"Error loading file {file_path}: {e}")

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")

    # Check if 'date' column exists and parse dates
    if 'date' in df.columns:
        print("✅ Found 'date' column, parsing dates...")
        # Convert to string first to handle mixed types
        df['date'] = df['date'].astype(str)
        # Parse dates, coercing errors to NaT
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Drop rows with invalid dates
        initial_rows = len(df)
        df = df.dropna(subset=['date'])
        if len(df) == 0:
            raise ValueError("Error: No valid date entries after parsing.")
        print(f"✅ Kept {len(df)} rows with valid dates (dropped {initial_rows - len(df)} invalid)")
    else:
        print("⚠️  No 'date' column found, creating synthetic dates...")
        # Create synthetic date column starting from a reasonable date
        start_date = datetime(2024, 1, 1)
        df['date'] = [start_date + timedelta(days=i) for i in range(len(df))]

    # Sort by date and reset index
    df = df.sort_values('date').reset_index(drop=True)

    # Drop 'Tag' column if it exists
    if "Tag" in df.columns:
        df = df.drop(columns=["Tag"])
        print("✅ Dropped 'Tag' column")

    # Display date range
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")

    # Check if all specified target columns exist in the DataFrame
    existing_targets = [col for col in target_cols_list if col in df.columns]
    missing_targets = [col for col in target_cols_list if col not in df.columns]

    if missing_targets:
        print(f"⚠️  Missing target columns: {missing_targets}")

    if not existing_targets:
        raise ValueError("❌ No target columns found in the dataset!")

    print(f"✅ Found {len(existing_targets)} target columns: {existing_targets}")

    # Identify feature columns (all columns except 'date' and target columns)
    feature_cols = [col for col in df.columns if col != 'date' and col not in existing_targets]
    print(f"📊 Feature columns: {len(feature_cols)} columns")

    # Convert all non-date columns to numeric
    print("🔧 Converting columns to numeric...")
    non_numeric_cols = []

    for col in df.columns:
        if col != 'date':
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"✅ Converted {col} to numeric")
                except Exception as e:
                    non_numeric_cols.append(col)
                    print(f"❌ Failed to convert {col}: {e}")

    # Drop columns that couldn't be converted
    if non_numeric_cols:
        print(f"⚠️  Dropping {len(non_numeric_cols)} non-numeric columns")
        df = df.drop(columns=non_numeric_cols)
        # Update feature and target lists
        feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
        existing_targets = [col for col in existing_targets if col not in non_numeric_cols]

    # Handle missing values
    print("🔧 Handling missing values...")
    # For numeric columns, use forward fill then backward fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    # Fill any remaining NaN values with 0 for numeric columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Check for remaining NaN values
    initial_rows = len(df)
    nan_rows = df.isnull().any(axis=1).sum()
    
    if nan_rows > 0:
        print(f"⚠️  Found {nan_rows} rows with NaN values, filling with 0...")
        df = df.fillna(0)
    
    final_rows = len(df)
    print(f"✅ Final dataset: {final_rows} rows after cleaning")

    if df.empty:
        raise ValueError("❌ All data was removed during cleaning!")

    # Final verification
    print(f"✅ Final dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"✅ Target columns: {len(existing_targets)}")
    print(f"✅ Feature columns: {len(feature_cols)}")

    # Prepare datasets for Autoformer
    # Autoformer expects: date column + all other numeric columns
    # The target columns should be at the end for easier processing

    # Reorder columns: date + features + targets
    final_columns = ['date'] + feature_cols + existing_targets
    df_final = df[final_columns].copy()

    # Create output directory
    os.makedirs('./dataset/custom/', exist_ok=True)

    # Save the prepared dataset
    output_path = './dataset/custom/custom.csv'
    df_final.to_csv(output_path, index=False)

    # Create dataset info file
    dataset_info = {
        'total_rows': len(df_final),
        'total_columns': len(df_final.columns) - 1,  # Exclude date column
        'target_columns': existing_targets,
        'num_targets': len(existing_targets),
        'feature_columns': feature_cols[:20],  # First 20 for brevity
        'num_features': len(feature_cols),
        'date_range_start': str(df_final['date'].min()),
        'date_range_end': str(df_final['date'].max()),
        'primary_target': existing_targets[0] if existing_targets else 'Unknown'
    }

    # Save dataset info
    info_path = './dataset/custom/dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)

    # Create summary file
    summary_path = './dataset/custom/dataset_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Autoformer Dataset Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total rows: {dataset_info['total_rows']}\n")
        f.write(f"Total columns (excluding date): {dataset_info['total_columns']}\n")
        f.write(f"Target columns: {len(existing_targets)}\n")
        f.write(f"Feature columns: {len(feature_cols)}\n")
        f.write(f"Date range: {dataset_info['date_range_start']} to {dataset_info['date_range_end']}\n")
        f.write(f"Primary target: {dataset_info['primary_target']}\n")

    print(f"✅ Dataset prepared and saved to {output_path}")
    print(f"✅ Dataset info saved to {info_path}")
    print(f"✅ Summary saved to {summary_path}")

    return df_final, existing_targets, feature_cols, dataset_info

class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_cols, ts_cols, static_cols, seq_len=60, pred_len=7):
        self.data = data
        self.target_cols = target_cols
        self.ts_cols = ts_cols
        self.static_cols = static_cols
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Prepare features and targets
        self.features = data[ts_cols].values
        self.targets = data[target_cols].values
        self.statics = data[static_cols].values
        
        # Normalize features
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.features_scaled = self.feature_scaler.fit_transform(self.features)
        self.targets_scaled = self.target_scaler.fit_transform(self.targets)
        
        # Create sequences
        self.sequences = []
        self.labels = []
        self.static_features = []
        
        for i in range(len(self.features_scaled) - seq_len - pred_len + 1):
            seq = self.features_scaled[i:i+seq_len]
            label = self.targets_scaled[i+seq_len:i+seq_len+pred_len]
            static = self.statics[i+seq_len]  # Use static features at prediction time
            
            self.sequences.append(seq)
            self.labels.append(label)
            self.static_features.append(static)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'past_values': torch.FloatTensor(self.sequences[idx]),
            'past_time_features': torch.zeros(self.seq_len, 1),  # Dummy time features
            'static_categorical_features': torch.zeros(len(self.static_cols)),  # Dummy categorical
            'static_real_features': torch.FloatTensor(self.static_features[idx]),
            'future_values': torch.FloatTensor(self.labels[idx]),
            'future_time_features': torch.zeros(self.pred_len, 1),  # Dummy time features
            'past_observed_mask': torch.ones(self.seq_len, len(self.ts_cols)),  # All values observed
            'future_observed_mask': torch.ones(self.pred_len, len(self.target_cols))  # All values observed
        }
    
    def inverse_transform_targets(self, scaled_targets):
        return self.target_scaler.inverse_transform(scaled_targets)

def train_autoformer_model(df_processed, target_cols, ts_cols, static_cols):
    """
    Train Autoformer model using Hugging Face Transformers
    """
    print("🚀 Starting Autoformer training...")
    
    # Create dataset
    dataset = TimeSeriesDataset(df_processed, target_cols, ts_cols, static_cols, seq_len=60, pred_len=7)
    print(f"Dataset size: {len(dataset)}")
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Configure Autoformer
    config = AutoformerConfig(
        context_length=60,
        prediction_length=7,
        num_time_features=1,
        num_static_categorical_features=0,
        num_static_real_features=len(static_cols),
        cardinality=[],
        embedding_dimension=[],
        d_model=64,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_layers=2,
        decoder_layers=2,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        activation_function="gelu",
        dropout=0.1,
        encoder_layerdrop=0.1,
        decoder_layerdrop=0.1,
        use_cache=True,
        num_parallel_samples=100,
        init_std=0.02,
        num_patches=16,
        patch_size="auto",
        patch_stride="auto",
        num_default_real_val_embeddings=1,
        scaling="std",
        embedding_dimension_multiplier=2.0,
        distr_output="student_t",
        loss="nll",
        input_size=len(ts_cols),
        num_targets=len(target_cols),
        output_size=len(target_cols),
    )
    
    print("✅ Autoformer configuration created")
    
    # Create model and training setup
    model = AutoformerForPrediction(config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./autoformer_results",
        num_train_epochs=5,  # Reduced for local testing
        per_device_train_batch_size=8,  # Reduced for local testing
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Custom compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Calculate metrics for each target
        mse = mean_squared_error(labels.flatten(), predictions.flatten())
        mae = mean_absolute_error(labels.flatten(), predictions.flatten())
        r2 = r2_score(labels.flatten(), predictions.flatten())
        
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "rmse": np.sqrt(mse)
        }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("✅ Trainer created")
    
    # Train the model
    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training completed!")
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")
    
    return trainer, test_results, test_dataset

def train_ebm_models(df_processed, target_cols, static_cols):
    """
    Train EBM models for blend effects explanation
    """
    print("🔍 Starting EBM training...")
    
    # Prepare data for EBM
    X_static = df_processed[static_cols].values
    y_targets = df_processed[target_cols].values
    
    print(f"EBM data shape: X={X_static.shape}, y={y_targets.shape}")
    
    ebm_models = {}
    ebm_results = {}
    
    for i, target_name in enumerate(target_cols):
        print(f"Training EBM for {target_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_static, y_targets[:, i], test_size=0.2, random_state=42
        )
        
        # Train EBM
        ebm = ExplainableBoostingRegressor(
            interactions=5,  # Reduced for local testing
            max_bins=128,
            max_interaction_bins=16,
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
        
        print(f"  R² = {r2:.3f}, MAE = {mae:.3f}")
    
    print("✅ EBM models trained successfully!")
    return ebm_models, ebm_results

def plot_results(ebm_models, static_cols, target_cols, ebm_results):
    """
    Create visualization plots
    """
    print("📊 Creating visualizations...")
    
    # Plot EBM feature importance
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, target_name in enumerate(target_cols[:6]):  # Show first 6 targets
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
    
    print("✅ Visualizations saved")

def main():
    """
    Main execution function
    """
    print("="*60)
    print("LOCAL AUTOFORMER + EBM ANALYSIS")
    print("="*60)
    
    try:
        # Step 1: Load and prepare data
        df_processed, target_cols, feature_cols, dataset_info = load_and_prepare_data()
        
        # Define static and time series columns
        static_cols = [col for col in feature_cols if col.startswith('crude_') or col in ['API', 'Sulphur', 'blend_id']]
        ts_cols = [col for col in feature_cols if col not in static_cols]
        
        print(f"Static columns: {len(static_cols)}")
        print(f"Time series columns: {len(ts_cols)}")
        
        # Step 2: Train Autoformer model
        trainer, test_results, test_dataset = train_autoformer_model(df_processed, target_cols, ts_cols, static_cols)
        
        # Step 3: Train EBM models
        ebm_models, ebm_results = train_ebm_models(df_processed, target_cols, static_cols)
        
        # Step 4: Create visualizations
        plot_results(ebm_models, static_cols, target_cols, ebm_results)
        
        # Step 5: Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        print("\n📊 Autoformer Performance:")
        for key, value in test_results.items():
            if 'eval_' in key:
                print(f"  {key}: {value:.4f}")
        
        print("\n🔍 EBM Model Performance:")
        for target_name, results in ebm_results.items():
            print(f"  {target_name}:")
            print(f"    R² = {results['r2']:.3f}")
            print(f"    MAE = {results['mae']:.3f}")
            print(f"    MSE = {results['mse']:.3f}")
        
        print("\n✅ Analysis completed successfully!")
        print("\nKey Findings:")
        print("1. Autoformer provides multivariate time series forecasting")
        print("2. EBM models explain how blend characteristics affect each target")
        print("3. Feature importance shows which crude properties matter most")
        print("4. Local explanations reveal specific blend effects")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
