#!/usr/bin/env python3
"""
Test script to verify the local environment setup
"""

import sys
import os

def test_python_version():
    """Test Python version"""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    return True

def test_data_file():
    """Test if data file exists"""
    data_file = "final_concatenated_data_mice_imputed.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False
    print(f"✅ Data file found: {data_file}")
    return True

def test_imports():
    """Test if required packages can be imported"""
    try:
        import pandas as pd
        import numpy as np
        import torch
        print("✅ Basic packages imported successfully")
        
        # Test transformers
        try:
            from transformers import AutoformerConfig, AutoformerForPrediction
            print("✅ Transformers imported successfully")
        except ImportError:
            print("⚠️  Transformers not installed - will be installed during setup")
        
        # Test interpret
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
            print("✅ Interpret imported successfully")
        except ImportError:
            print("⚠️  Interpret not installed - will be installed during setup")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test if data can be loaded"""
    try:
        import pandas as pd
        df = pd.read_csv("final_concatenated_data_mice_imputed.csv", nrows=5)
        print(f"✅ Data loading test successful: {df.shape}")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("LOCAL ENVIRONMENT TEST")
    print("="*50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Data File", test_data_file),
        ("Package Imports", test_imports),
        ("Data Loading", test_data_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Ready to run analysis.")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main()

