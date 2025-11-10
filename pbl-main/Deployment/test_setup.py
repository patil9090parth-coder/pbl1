#!/usr/bin/env python3
"""
Test script to verify the setup and dependencies
"""

import sys
import subprocess
import importlib
import warnings
warnings.filterwarnings('ignore')

def test_import(module_name):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - FAILED: {e}")
        return False

def test_streamlit():
    """Test Streamlit installation"""
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} - OK")
        return True
    except ImportError:
        print("❌ Streamlit - FAILED")
        return False

def main():
    print("🧪 Testing Real Estate Analytics Platform Setup")
    print("=" * 50)
    
    # Test Python version
    print(f"🐍 Python Version: {sys.version}")
    print("-" * 30)
    
    # Test critical dependencies
    critical_deps = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'folium',
        'requests',
        'PIL'
    ]
    
    optional_deps = [
        'streamlit_option_menu',
        'streamlit_folium',
        'scikit-learn'
    ]
    
    print("🔍 Testing Critical Dependencies:")
    critical_ok = 0
    for dep in critical_deps:
        if test_import(dep):
            critical_ok += 1
    
    print("\n🔍 Testing Optional Dependencies:")
    optional_ok = 0
    for dep in optional_deps:
        if test_import(dep):
            optional_ok += 1
    
    # Summary
    print("\n📊 Summary:")
    print(f"Critical Dependencies: {critical_ok}/{len(critical_deps)} working")
    print(f"Optional Dependencies: {optional_ok}/{len(optional_deps)} working")
    
    if critical_ok == len(critical_deps):
        print("✅ All critical dependencies are working!")
        print("🚀 Ready to run the application!")
        
        print("\n🎯 Next Steps:")
        print("1. Run the working app: streamlit run app_working.py")
        print("2. Or try the enhanced app: streamlit run app_enhanced.py")
        print("3. Access at: http://localhost:8501")
        
    else:
        print("❌ Some critical dependencies are missing!")
        print("💡 Please install missing packages using:")
        print("   pip install -r requirements_enhanced.txt")
    
    # Test data loading
    print("\n📋 Testing Data Loading:")
    try:
        # Create sample data
        import pandas as pd
        import numpy as np
        
        locations = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad']
        data = {
            'Location': np.random.choice(locations, 100),
            'Price': np.random.randint(50, 200, 100),
            'Area': np.random.randint(500, 2000, 100)
        }
        df = pd.DataFrame(data)
        print(f"✅ Sample data created: {len(df)} rows, {len(df.columns)} columns")
        print(f"✅ Data preview:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")

if __name__ == "__main__":
    main()