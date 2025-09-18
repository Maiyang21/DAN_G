"""
Enhanced ETL Processor for Real-time Data Processing
Handles data ingestion, cleaning, validation, and transformation
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import hashlib
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ETLProcessor:
    """Enhanced ETL processor with real-time capabilities and data quality monitoring"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.data_quality_threshold = 0.8
        self.processing_stats = {}
        
    def process_file(self, file) -> pd.DataFrame:
        """Process uploaded file with comprehensive validation and cleaning"""
        try:
            logger.info(f"Processing file: {file.filename}")
            
            # Read file based on extension
            if file.filename.endswith('.csv'):
                data = pd.read_csv(file)
            elif file.filename.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file)
            elif file.filename.endswith('.json'):
                data = pd.read_json(file)
            else:
                raise ValueError(f"Unsupported file format: {file.filename}")
            
            # Store original data info
            original_shape = data.shape
            logger.info(f"Original data shape: {original_shape}")
            
            # Comprehensive data processing pipeline
            processed_data = self._comprehensive_processing_pipeline(data)
            
            # Calculate processing statistics
            self.processing_stats = self._calculate_processing_stats(data, processed_data)
            
            logger.info(f"Processing completed. Final shape: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            raise
    
    def _comprehensive_processing_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data processing pipeline"""
        
        # Step 1: Data validation and type detection
        data = self._validate_and_detect_types(data)
        
        # Step 2: Handle missing values with advanced imputation
        data = self._handle_missing_values(data)
        
        # Step 3: Detect and handle outliers
        data = self._handle_outliers(data)
        
        # Step 4: Feature engineering
        data = self._engineer_features(data)
        
        # Step 5: Data normalization and scaling
        data = self._normalize_data(data)
        
        # Step 6: Final validation
        data = self._final_validation(data)
        
        return data
    
    def _validate_and_detect_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data and detect appropriate data types"""
        logger.info("Validating data and detecting types...")
        
        # Remove completely empty rows and columns
        data = data.dropna(how='all').dropna(axis=1, how='all')
        
        # Detect and convert numeric columns
        for col in data.columns:
            if data[col].dtype == 'object':
                # Try to convert to numeric
                numeric_data = pd.to_numeric(data[col], errors='coerce')
                if not numeric_data.isna().all():
                    data[col] = numeric_data
                    logger.info(f"Converted column '{col}' to numeric")
        
        # Detect datetime columns
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                    if not data[col].isna().all():
                        logger.info(f"Converted column '{col}' to datetime")
                except:
                    pass
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Advanced missing value handling"""
        logger.info("Handling missing values...")
        
        missing_before = data.isnull().sum().sum()
        
        # For numeric columns, use KNN imputation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=5)
            data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        
        # For categorical columns, use mode imputation
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().any():
                mode_value = data[col].mode()
                if len(mode_value) > 0:
                    data[col].fillna(mode_value[0], inplace=True)
                else:
                    data[col].fillna('Unknown', inplace=True)
        
        missing_after = data.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using IQR method"""
        logger.info("Handling outliers...")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_count += outliers
            
            # Cap outliers instead of removing them
            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
        
        logger.info(f"Handled {outlier_count} outliers")
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features for better forecasting"""
        logger.info("Engineering features...")
        
        # Create time-based features if datetime column exists
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            data[f'{col}_year'] = data[col].dt.year
            data[f'{col}_month'] = data[col].dt.month
            data[f'{col}_day'] = data[col].dt.day
            data[f'{col}_dayofweek'] = data[col].dt.dayofweek
            data[f'{col}_hour'] = data[col].dt.hour
        
        # Create lag features for time series data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            for lag in [1, 2, 3, 7]:  # 1, 2, 3, and 7-day lags
                data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        # Create rolling statistics
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            for window in [3, 7, 14]:  # 3, 7, and 14-day windows
                data[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window).mean()
                data[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window).std()
        
        # Create interaction features
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            data[f'{col1}_{col2}_interaction'] = data[col1] * data[col2]
            data[f'{col1}_{col2}_ratio'] = data[col1] / (data[col2] + 1e-8)
        
        logger.info(f"Feature engineering completed. New shape: {data.shape}")
        return data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data for better model performance"""
        logger.info("Normalizing data...")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Standardize numeric columns
        if len(numeric_cols) > 0:
            data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        
        # Encode categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
        
        return data
    
    def _final_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Final data validation and quality check"""
        logger.info("Performing final validation...")
        
        # Check for any remaining missing values
        if data.isnull().any().any():
            logger.warning("Warning: Missing values still present after processing")
            data = data.fillna(0)  # Fill any remaining NaN with 0
        
        # Check for infinite values
        if np.isinf(data.select_dtypes(include=[np.number])).any().any():
            logger.warning("Warning: Infinite values detected, replacing with 0")
            data = data.replace([np.inf, -np.inf], 0)
        
        # Ensure all numeric columns are finite
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        return data
    
    def _calculate_processing_stats(self, original: pd.DataFrame, processed: pd.DataFrame) -> Dict:
        """Calculate processing statistics"""
        stats = {
            'original_shape': original.shape,
            'processed_shape': processed.shape,
            'rows_removed': original.shape[0] - processed.shape[0],
            'columns_added': processed.shape[1] - original.shape[1],
            'missing_values_before': original.isnull().sum().sum(),
            'missing_values_after': processed.isnull().sum().sum(),
            'data_quality_score': self._calculate_data_quality_score(processed),
            'processing_time': datetime.now().isoformat()
        }
        
        return stats
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        
        # Check for constant columns (low variance)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        variance_score = 1.0
        if len(numeric_cols) > 0:
            low_variance_cols = data[numeric_cols].var() < 1e-10
            variance_score = 1 - (low_variance_cols.sum() / len(numeric_cols))
        
        # Check for duplicate rows
        duplicate_score = 1 - (data.duplicated().sum() / len(data))
        
        # Overall quality score (weighted average)
        quality_score = (completeness * 0.4 + variance_score * 0.3 + duplicate_score * 0.3)
        
        return round(quality_score, 3)
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return self.processing_stats
    
    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate data quality and return status"""
        quality_score = self._calculate_data_quality_score(data)
        
        if quality_score >= self.data_quality_threshold:
            return True, f"Data quality acceptable (Score: {quality_score:.3f})"
        else:
            return False, f"Data quality below threshold (Score: {quality_score:.3f} < {self.data_quality_threshold})"
    
    def generate_data_report(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive data report"""
        report = {
            'basic_info': {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict()
            },
            'quality_metrics': {
                'completeness': 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])),
                'duplicates': data.duplicated().sum(),
                'unique_values': data.nunique().to_dict()
            },
            'statistical_summary': data.describe().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else {},
            'processing_stats': self.processing_stats
        }
        
        return report