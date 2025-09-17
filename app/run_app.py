#!/usr/bin/env python3
"""
Startup script for the Refinery Forecast App
"""
import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import boto3
        import plotly
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_aws_config():
    """Check AWS configuration"""
    aws_region = os.getenv('AWS_REGION')
    s3_bucket = os.getenv('S3_BUCKET')
    s3_access_point = os.getenv('S3_ACCESS_POINT_ARN')
    sm_endpoint = os.getenv('SM_ENDPOINT')
    
    print("🔧 AWS Configuration:")
    print(f"   Region: {aws_region or 'Not set'}")
    print(f"   S3 Bucket: {s3_bucket or 'Not set'}")
    print(f"   S3 Access Point: {s3_access_point or 'Not set'}")
    print(f"   SageMaker Endpoint: {sm_endpoint or 'Not set'}")
    
    if not any([s3_bucket, s3_access_point]):
        print("⚠️  Warning: Neither S3_BUCKET nor S3_ACCESS_POINT_ARN is set")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🚀 Starting Refinery Forecast App...")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check AWS configuration
    check_aws_config()
    
    # Get the directory containing this script
    app_dir = Path(__file__).parent
    
    # Change to app directory
    os.chdir(app_dir)
    
    print("\n🌐 Starting Streamlit server...")
    print("   Open your browser to: http://localhost:8501")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
















