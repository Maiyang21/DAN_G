#!/usr/bin/env python3
"""
Local Autoformer EBM Analysis Runner
This script installs dependencies and runs the analysis
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_local.txt"])
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def check_data_file():
    """Check if data file exists"""
    data_file = "final_concatenated_data_mice_imputed.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure the data file is in the current directory")
        return False
    print(f"✅ Data file found: {data_file}")
    return True

def run_analysis():
    """Run the main analysis"""
    print("🚀 Starting Autoformer EBM analysis...")
    try:
        import local_autoformer_ebm_script
        local_autoformer_ebm_script.main()
        print("✅ Analysis completed successfully!")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("="*60)
    print("LOCAL AUTOFORMER EBM ANALYSIS SETUP")
    print("="*60)
    
    # Check if data file exists
    if not check_data_file():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Run analysis
    run_analysis()

if __name__ == "__main__":
    main()

