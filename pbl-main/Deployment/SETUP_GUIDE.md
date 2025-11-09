# Real Estate Price Prediction App - Setup Guide

## Quick Fix for Installation Issues

If you're experiencing installation problems, follow these steps:

### Option 1: Use Pre-compiled Wheels (Recommended)
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install packages individually to avoid conflicts
pip install streamlit
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install plotly
pip install folium
pip install streamlit-folium
```

### Option 2: Use Conda (Alternative)
```bash
# If you have Anaconda/Miniconda installed
conda install streamlit pandas numpy scikit-learn matplotlib seaborn plotly folium
pip install streamlit-folium
```

### Option 3: Use Pre-built Environment
```bash
# Download and install pre-built wheels
pip install --only-binary=all streamlit pandas numpy scikit-learn matplotlib seaborn plotly folium streamlit-folium
```

## Model Compatibility Warning Fix

The warnings about scikit-learn version differences are normal and won't affect functionality. The models will still work correctly.

## Running the App

After successful installation:
```bash
cd Deployment
streamlit run app.py
```

## Streamlit Cloud Deployment

For Streamlit Cloud deployment, the simplified requirements.txt should work automatically. The platform handles the compilation issues for you.

## Troubleshooting

1. **Python Version**: Use Python 3.8-3.11 for best compatibility
2. **Memory Issues**: Close other applications if you have limited RAM
3. **Network Issues**: Use a stable internet connection for package downloads

## App URLs (When Running Locally)
- Local: http://localhost:8501
- Network: http://192.168.1.101:8501