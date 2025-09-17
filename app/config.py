import os

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Use either S3_BUCKET *or* S3_ACCESS_POINT_ARN (preferred for shared VPC access)
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ACCESS_POINT_ARN = os.getenv("S3_ACCESS_POINT_ARN", "")  # e.g., arn:aws:s3:us-east-1:123456789012:accesspoint/my-ap
S3_PREFIX = os.getenv("S3_PREFIX", "refinery-forecast/input/")

# SageMaker Configuration
SM_ENDPOINT = os.getenv("SM_ENDPOINT", "autoformer-ebm-endpoint")

# Model Configuration
# Target columns that the model will predict
TARGET_COLS = [
    "target_RCO_flow", "target_Heavy_Diesel_flow", "target_Light_Diesel_flow", 
    "target_Kero_flow", "target_Naphtha_flow",
    "target_RCO_Yield", "target_Heavy_Diesel_Yield", "target_Light_Diesel_Yield", 
    "target_Kero_Yield"
]

# ETL Configuration
STATIC_PREFIX = "static_"
FUTURE_PREFIX = "future_"
TIME_INDEX_COL = "date"

# File Processing Configuration
ALLOWED_FILE_TYPES = ["xlsx", "xls", "csv"]
MAX_FILE_SIZE_MB = 100

# UI Configuration
APP_TITLE = "Refinery Forecast App — Autoformer + EBM (AWS)"
APP_ICON = "🏭"

