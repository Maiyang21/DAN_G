#!/usr/bin/env python3
"""
Test script to check dependencies for XGBoost comparison
"""

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing dependencies...")
    
    dependencies = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization',
        'scipy': 'Scientific computing',
        'xgboost': 'Gradient boosting'
    }
    
    results = {}
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"  ✅ {package:12s} - {description}")
            results[package] = True
        except ImportError:
            print(f"  ❌ {package:12s} - {description} (MISSING)")
            results[package] = False
    
    print(f"\n📊 Summary:")
    print(f"  Available: {sum(results.values())}/{len(results)}")
    print(f"  Missing: {len(results) - sum(results.values())}")
    
    if not results['xgboost']:
        print(f"\n⚠️  XGBoost is not available. The script will run without XGBoost.")
        print(f"   Install with: pip install xgboost")
    
    return results

if __name__ == "__main__":
    test_dependencies()





