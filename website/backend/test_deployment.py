#!/usr/bin/env python3
"""
Test script to verify backend deployment configuration
"""

import sys
import os
import importlib.util

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    required_modules = [
        'flask',
        'flask_sqlalchemy', 
        'flask_cors',
        'flask_socketio',
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'boto3',
        'psutil',
        'joblib',
        'scipy',
        'werkzeug',
        'python_socketio',
        'eventlet',
        'gunicorn',
        'dotenv',
        'redis'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            if module == 'sklearn':
                import sklearn
            elif module == 'python_socketio':
                import socketio
            else:
                __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

def test_app_structure():
    """Test if the Flask app can be initialized"""
    print("\nTesting Flask app structure...")
    
    try:
        # Add the current directory to Python path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Try to import the app
        from app import app
        print("✅ Flask app imported successfully")
        
        # Test basic app configuration
        if hasattr(app, 'config'):
            print("✅ App has configuration")
        else:
            print("❌ App missing configuration")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to import Flask app: {e}")
        return False

def main():
    """Run all tests"""
    print("DAN_G Backend Deployment Test")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test app structure
    app_ok = test_app_structure()
    
    print("\n" + "=" * 40)
    if imports_ok and app_ok:
        print("✅ All tests passed! Backend is ready for deployment.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
