"""
AWS utilities for S3 and SageMaker integration
"""
from typing import Dict, Tuple
import io
import json
import pandas as pd
import boto3
from botocore.config import Config as BotoConfig
from config import AWS_REGION, S3_BUCKET, S3_ACCESS_POINT_ARN, S3_PREFIX, SM_ENDPOINT

# Configure S3 client with proper settings
_s3_cfg = BotoConfig(
    s3={"addressing_style": "virtual"}, 
    s3_use_arn_region=True, 
    retries={"max_attempts": 10}
)
s3 = boto3.client("s3", region_name=AWS_REGION, config=_s3_cfg)
smr = boto3.client("sagemaker-runtime", region_name=AWS_REGION)


def upload_df_csv(df: pd.DataFrame, key: str) -> str:
    """
    Upload a DataFrame as CSV to S3 bucket or Access Point. 
    Returns the S3 URI used.
    
    Args:
        df: DataFrame to upload
        key: S3 key (path) for the file
        
    Returns:
        S3 URI of the uploaded file
        
    Raises:
        RuntimeError: If neither S3_ACCESS_POINT_ARN nor S3_BUCKET is configured
    """
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    
    if S3_ACCESS_POINT_ARN:
        s3.put_object(Bucket=S3_ACCESS_POINT_ARN, Key=key, Body=csv_bytes)
        return f"s3://{S3_ACCESS_POINT_ARN}/{key}"
    elif S3_BUCKET:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=csv_bytes)
        return f"s3://{S3_BUCKET}/{key}"
    else:
        raise RuntimeError("Neither S3_ACCESS_POINT_ARN nor S3_BUCKET configured.")


def invoke_endpoint(payload: Dict) -> Dict:
    """
    Invoke SageMaker endpoint with JSON payload. 
    Returns JSON dict.
    
    Args:
        payload: Dictionary containing the prediction request
        
    Returns:
        Dictionary containing the endpoint response
    """
    resp = smr.invoke_endpoint(
        EndpointName=SM_ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = resp["Body"].read()
    try:
        return json.loads(body)
    except Exception:
        return {"raw": body.decode("utf-8", errors="ignore")}


def check_aws_credentials() -> bool:
    """
    Check if AWS credentials are properly configured.
    
    Returns:
        True if credentials are available, False otherwise
    """
    try:
        # Try to get caller identity
        sts = boto3.client("sts", region_name=AWS_REGION)
        sts.get_caller_identity()
        return True
    except Exception:
        return False


def list_s3_objects(prefix: str = None) -> list:
    """
    List objects in S3 bucket/access point with given prefix.
    
    Args:
        prefix: S3 prefix to filter objects
        
    Returns:
        List of S3 object keys
    """
    try:
        if S3_ACCESS_POINT_ARN:
            response = s3.list_objects_v2(Bucket=S3_ACCESS_POINT_ARN, Prefix=prefix)
        elif S3_BUCKET:
            response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        else:
            return []
            
        return [obj["Key"] for obj in response.get("Contents", [])]
    except Exception:
        return []

