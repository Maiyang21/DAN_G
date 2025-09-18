"""
AWS Client for SageMaker and S3 Integration
Handles model inference and data storage
"""

import boto3
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import io
import pickle

logger = logging.getLogger(__name__)

class AWSClient:
    """AWS client for SageMaker and S3 operations"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
        self.s3_bucket = os.environ.get('AWS_S3_BUCKET', 'dan-g-refinery-data')
        self.sagemaker_endpoint = os.environ.get('AWS_SAGEMAKER_ENDPOINT', 'dan-g-forecasting-endpoint')
        self.region = os.environ.get('AWS_REGION', 'us-east-1')
        
    def upload_to_s3(self, data: pd.DataFrame, s3_key: str) -> bool:
        """Upload DataFrame to S3"""
        try:
            # Convert DataFrame to CSV
            csv_buffer = io.StringIO()
            data.to_csv(csv_buffer, index=False)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )
            
            logger.info(f"Data uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            return False
    
    def download_from_s3(self, s3_key: str) -> pd.DataFrame:
        """Download DataFrame from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=s3_key
            )
            
            # Read CSV from S3
            data = pd.read_csv(io.BytesIO(response['Body'].read()))
            
            logger.info(f"Data downloaded from S3: s3://{self.s3_bucket}/{s3_key}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            raise
    
    def invoke_sagemaker_model(self, data: pd.DataFrame) -> pd.DataFrame:
        """Invoke SageMaker model for forecasting"""
        try:
            # Prepare data for SageMaker
            input_data = self._prepare_sagemaker_input(data)
            
            # Invoke SageMaker endpoint
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.sagemaker_endpoint,
                ContentType='application/json',
                Body=json.dumps(input_data)
            )
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            
            # Convert to DataFrame
            forecast_df = self._parse_sagemaker_output(result)
            
            logger.info("SageMaker model invoked successfully")
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error invoking SageMaker model: {str(e)}")
            # Fallback to local model
            return self._fallback_forecast(data)
    
    def _prepare_sagemaker_input(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for SageMaker input"""
        try:
            # Select numeric features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                raise ValueError("Insufficient numeric features for forecasting")
            
            # Prepare features and target
            feature_cols = numeric_cols[:-1]
            target_col = numeric_cols[-1]
            
            X = data[feature_cols].fillna(0).values
            y = data[target_col].fillna(0).values
            
            # Create input payload
            input_data = {
                'instances': X.tolist(),
                'features': feature_cols.tolist(),
                'target': target_col,
                'forecast_horizon': 7
            }
            
            return input_data
            
        except Exception as e:
            logger.error(f"Error preparing SageMaker input: {str(e)}")
            raise
    
    def _parse_sagemaker_output(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Parse SageMaker output to DataFrame"""
        try:
            # Extract forecast data from SageMaker response
            predictions = result.get('predictions', [])
            confidence_intervals = result.get('confidence_intervals', [])
            
            # Create forecast DataFrame
            forecast_data = []
            for i, pred in enumerate(predictions):
                forecast_data.append({
                    'date': (datetime.now() + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    'forecast_value': pred,
                    'confidence_lower': confidence_intervals[i][0] if i < len(confidence_intervals) else pred * 0.9,
                    'confidence_upper': confidence_intervals[i][1] if i < len(confidence_intervals) else pred * 1.1,
                    'day_ahead': i + 1
                })
            
            return pd.DataFrame(forecast_data)
            
        except Exception as e:
            logger.error(f"Error parsing SageMaker output: {str(e)}")
            raise
    
    def _fallback_forecast(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback forecasting when SageMaker is unavailable"""
        try:
            logger.warning("Using fallback forecasting model")
            
            # Simple trend-based forecast
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns for forecasting")
            
            target_col = numeric_cols[-1]
            last_value = data[target_col].iloc[-1]
            
            # Generate simple forecast
            forecast_data = []
            for i in range(7):
                # Simple trend with some variation
                trend_factor = 1 + (i * 0.02)  # 2% increase per day
                forecast_value = last_value * trend_factor
                
                forecast_data.append({
                    'date': (datetime.now() + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    'forecast_value': forecast_value,
                    'confidence_lower': forecast_value * 0.9,
                    'confidence_upper': forecast_value * 1.1,
                    'day_ahead': i + 1
                })
            
            return pd.DataFrame(forecast_data)
            
        except Exception as e:
            logger.error(f"Error in fallback forecast: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check AWS services health"""
        try:
            # Check S3 access
            s3_status = "healthy"
            try:
                self.s3_client.head_bucket(Bucket=self.s3_bucket)
            except Exception:
                s3_status = "unhealthy"
            
            # Check SageMaker endpoint
            sagemaker_status = "healthy"
            try:
                self.sagemaker_runtime.describe_endpoint(EndpointName=self.sagemaker_endpoint)
            except Exception:
                sagemaker_status = "unhealthy"
            
            return {
                's3': s3_status,
                'sagemaker': sagemaker_status,
                'region': self.region
            }
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {
                's3': 'error',
                'sagemaker': 'error',
                'error': str(e)
            }
    
    def upload_model_artifacts(self, model_data: Dict[str, Any], s3_key: str) -> bool:
        """Upload model artifacts to S3"""
        try:
            # Serialize model data
            model_bytes = pickle.dumps(model_data)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=model_bytes,
                ContentType='application/octet-stream'
            )
            
            logger.info(f"Model artifacts uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading model artifacts: {str(e)}")
            return False
    
    def download_model_artifacts(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """Download model artifacts from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=s3_key
            )
            
            # Deserialize model data
            model_data = pickle.loads(response['Body'].read())
            
            logger.info(f"Model artifacts downloaded from S3: s3://{self.s3_bucket}/{s3_key}")
            return model_data
            
        except Exception as e:
            logger.error(f"Error downloading model artifacts: {str(e)}")
            return None
    
    def list_user_data(self, user_id: str) -> List[str]:
        """List user's data files in S3"""
        try:
            prefix = f"processed_data/{user_id}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append(obj['Key'])
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing user data: {str(e)}")
            return []
    
    def delete_user_data(self, user_id: str, s3_key: str) -> bool:
        """Delete user's data file from S3"""
        try:
            # Verify the file belongs to the user
            if not s3_key.startswith(f"processed_data/{user_id}/"):
                raise ValueError("Unauthorized access to file")
            
            self.s3_client.delete_object(
                Bucket=self.s3_bucket,
                Key=s3_key
            )
            
            logger.info(f"Data deleted from S3: s3://{self.s3_bucket}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting data: {str(e)}")
            return False
