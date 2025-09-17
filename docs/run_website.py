#!/usr/bin/env python3
"""
DAN_G Refinery Forecasting Website Runner
Simple script to run the forecasting website.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import flask
        import pandas
        import numpy
        import sklearn
        import plotly
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ['uploads', 'static', 'templates']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def run_website():
    """Run the Flask website."""
    print("🚀 Starting DAN_G Refinery Forecasting Website...")
    print("📍 Website will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Change to the website directory
        os.chdir(Path(__file__).parent)
        
        # Run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n🛑 Website stopped by user")
    except Exception as e:
        print(f"❌ Error running website: {e}")
        sys.exit(1)

def main():
    """Main function."""
    print("=" * 60)
    print("🏭 DAN_G Refinery Forecasting Website")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run website
    run_website()

if __name__ == "__main__":
    main()

