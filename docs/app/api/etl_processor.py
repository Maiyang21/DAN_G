"""
ETL Processor for Raw Data
Processes raw refinery data and prepares it for forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ETLProcessor:
    """ETL processor for refinery data."""
    
    def __init__(self):
        self.processed_data = None
        self.feature_columns = []
        self.target_columns = []
        
    def process_file(self, file_path):
        """Process uploaded file and return cleaned data."""
        try:
            # Read file based on extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Process the data
            processed_df = self._process_dataframe(df)
            
            # Store processed data
            self.processed_data = processed_df
            
            return processed_df
            
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
    
    def _process_dataframe(self, df):
        """Process and clean the dataframe."""
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Convert date columns to datetime
        date_columns = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp']
        for col in date_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                df_processed.set_index(col, inplace=True)
                break
        
        # If no date column found, create a synthetic one
        if df_processed.index.dtype == 'object' or not hasattr(df_processed.index, 'date'):
            df_processed.index = pd.date_range(
                start='2024-01-01',
                periods=len(df_processed),
                freq='D'
            )
        
        # Identify feature and target columns
        self._identify_columns(df_processed)
        
        # Clean data
        df_processed = self._clean_data(df_processed)
        
        # Feature engineering
        df_processed = self._engineer_features(df_processed)
        
        return df_processed
    
    def _identify_columns(self, df):
        """Identify feature and target columns."""
        # Common target columns (yields)
        target_keywords = [
            'yield', 'Yield', 'YIELD',
            'quality', 'Quality', 'QUALITY',
            'flowrate', 'Flowrate', 'FLOWRATE',
            'production', 'Production', 'PRODUCTION'
        ]
        
        # Common feature columns (operating parameters)
        feature_keywords = [
            'temperature', 'Temperature', 'TEMP',
            'pressure', 'Pressure', 'PRESSURE',
            'flow', 'Flow', 'FLOW',
            'crude', 'Crude', 'CRUDE',
            'composition', 'Composition', 'COMPOSITION',
            'density', 'Density', 'DENSITY',
            'viscosity', 'Viscosity', 'VISCOSITY'
        ]
        
        # Identify target columns
        self.target_columns = []
        for col in df.columns:
            if any(keyword in col for keyword in target_keywords):
                self.target_columns.append(col)
        
        # Identify feature columns
        self.feature_columns = []
        for col in df.columns:
            if any(keyword in col for keyword in feature_keywords):
                self.feature_columns.append(col)
        
        # If no targets identified, use last few columns as targets
        if not self.target_columns:
            self.target_columns = df.columns[-3:].tolist()
        
        # If no features identified, use remaining columns as features
        if not self.feature_columns:
            self.feature_columns = [col for col in df.columns if col not in self.target_columns]
    
    def _clean_data(self, df):
        """Clean the data by handling missing values and outliers."""
        # Handle missing values using interpolation
        df_cleaned = df.interpolate(method='linear', limit_direction='both')
        
        # Fill remaining NaN values with forward fill
        df_cleaned = df_cleaned.fillna(method='ffill')
        
        # Remove outliers using IQR method
        for col in df_cleaned.select_dtypes(include=[np.number]).columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)
        
        return df_cleaned
    
    def _engineer_features(self, df):
        """Engineer additional features for better forecasting."""
        df_engineered = df.copy()
        
        # Add lag features for targets
        for target in self.target_columns:
            if target in df_engineered.columns:
                for lag in [1, 2, 3, 7]:  # 1, 2, 3, and 7-day lags
                    df_engineered[f'{target}_lag_{lag}'] = df_engineered[target].shift(lag)
        
        # Add rolling statistics for features
        for feature in self.feature_columns:
            if feature in df_engineered.columns:
                # 3-day rolling mean
                df_engineered[f'{feature}_rolling_3d'] = df_engineered[feature].rolling(window=3).mean()
                # 7-day rolling mean
                df_engineered[f'{feature}_rolling_7d'] = df_engineered[feature].rolling(window=7).mean()
                # 3-day rolling std
                df_engineered[f'{feature}_rolling_std_3d'] = df_engineered[feature].rolling(window=3).std()
        
        # Add time-based features
        df_engineered['day_of_week'] = df_engineered.index.dayofweek
        df_engineered['month'] = df_engineered.index.month
        df_engineered['quarter'] = df_engineered.index.quarter
        
        # Add crude composition features (if not present)
        if 'crude_composition' not in df_engineered.columns:
            # Create synthetic crude composition features
            np.random.seed(42)
            df_engineered['crude_light_fraction'] = np.random.uniform(0.3, 0.7, len(df_engineered))
            df_engineered['crude_heavy_fraction'] = 1 - df_engineered['crude_light_fraction']
            df_engineered['crude_sulfur_content'] = np.random.uniform(0.5, 3.0, len(df_engineered))
            df_engineered['crude_api_gravity'] = np.random.uniform(20, 45, len(df_engineered))
        
        # Fill any remaining NaN values
        df_engineered = df_engineered.fillna(method='ffill').fillna(method='bfill')
        
        return df_engineered
    
    def get_processed_data(self):
        """Get the processed data."""
        return self.processed_data
    
    def get_feature_columns(self):
        """Get feature column names."""
        return self.feature_columns
    
    def get_target_columns(self):
        """Get target column names."""
        return self.target_columns
    
    def get_data_summary(self):
        """Get summary statistics of processed data."""
        if self.processed_data is None:
            return None
        
        summary = {
            'total_rows': len(self.processed_data),
            'total_columns': len(self.processed_data.columns),
            'feature_columns': len(self.feature_columns),
            'target_columns': len(self.target_columns),
            'date_range': {
                'start': str(self.processed_data.index.min()),
                'end': str(self.processed_data.index.max())
            },
            'missing_values': self.processed_data.isnull().sum().sum(),
            'data_types': self.processed_data.dtypes.to_dict()
        }
        
        return summary

