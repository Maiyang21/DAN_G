#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner script for XGBoost and L1/L2 Regression Comparison
This script provides an easy way to run the comparison analysis
"""

import sys
import os
import subprocess
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 
        'seaborn', 'scipy', 'xgboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements_xgboost_comparison.txt")
        return False
    
    print("✅ All dependencies are available!")
    return True

def check_data_file():
    """Check if the required data file exists"""
    data_file = 'final_concatenated_data_mice_imputed.csv'
    
    if os.path.exists(data_file):
        print(f"✅ Data file found: {data_file}")
        return True
    else:
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure the data file is in the current directory.")
        return False

def run_comparison():
    """Run the XGBoost comparison analysis"""
    print("\n" + "="*60)
    print("STARTING XGBOOST AND L1/L2 REGRESSION COMPARISON")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Import and run the comparison
        from xgboost_l1_l2_comparison import main
        main()
        
        end_time = time.time()
        print(f"\n✅ Analysis completed successfully in {end_time - start_time:.2f} seconds!")
        
    except Exception as e:
        print(f"❌ Error running comparison: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main runner function"""
    print("🚀 XGBoost and L1/L2 Regression Comparison Runner")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return 1
    
    # Check data file
    if not check_data_file():
        print("\n❌ Please ensure the data file is available and try again.")
        return 1
    
    # Run comparison
    if not run_comparison():
        print("\n❌ Comparison failed. Please check the error messages above.")
        return 1
    
    print("\n🎉 All done! Check the generated files:")
    print("  • model_performance_comparison.png")
    print("  • model_performance_comparison.csv")
    print("  • feature_importance_comparison_*.png")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

