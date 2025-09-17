import boto3
import os
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
import pandas as pd
from io import StringIO, BytesIO

# --- Load AWS Credentials from Environment Variables ---
# These should be set in your Heroku config vars
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1') # Default region if not set

def get_s3_client():
    """Initializes and returns a boto3 S3 client."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        # Test connection briefly (optional)
        s3_client.list_buckets()
        print("S3 client initialized successfully.")
        return s3_client
    except (NoCredentialsError, PartialCredentialsError):
        print("Error: AWS credentials not found. Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are set.")
        return None
    except ClientError as e:
        if e.response['Error']['Code'] == 'InvalidAccessKeyId':
            print("Error: Invalid AWS Access Key ID provided.")
        elif e.response['Error']['Code'] == 'SignatureDoesNotMatch':
            print("Error: Invalid AWS Secret Access Key provided.")
        else:
            print(f"Error connecting to S3: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error initializing S3 client: {e}")
        return None


def download_s3_file(s3_client, bucket_name, s3_key, download_path):
    """Downloads a file from S3 to a local path."""
    if not s3_client: return False
    print(f"Attempting to download s3://{bucket_name}/{s3_key} to {download_path}...")
    try:
        # Ensure download directory exists
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        s3_client.download_file(bucket_name, s3_key, download_path)
        print(f"Successfully downloaded {s3_key}.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"Error: File not found in S3: s3://{bucket_name}/{s3_key}")
        else:
            print(f"Error downloading {s3_key} from S3: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error downloading {s3_key}: {e}")
        return False


def upload_df_to_s3(s3_client, df, bucket_name, s3_key, index=True):
    """Uploads a pandas DataFrame to S3 as CSV."""
    if not s3_client: return False
    print(f"Attempting to upload DataFrame to s3://{bucket_name}/{s3_key}...")
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=index)
        csv_content = csv_buffer.getvalue()
        # Use put_object which handles content directly
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_content, ContentType='text/csv')
        print(f"Successfully uploaded DataFrame to {s3_key}.")
        return True
    except ClientError as e:
        print(f"Error uploading DataFrame to S3: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error uploading DataFrame: {e}")
        return False