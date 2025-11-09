# Real Estate Price Prediction - Comprehensive Agent Documentation

## Project Overview
This is a comprehensive machine learning project for predicting real estate prices in Mumbai, India. The project encompasses web scraping, data cleaning, exploratory data analysis, model building with multiple algorithms, and deployment through a Streamlit web application.

## Project Structure and Architecture

### Core Components
- **Data Pipeline**: Web scraping → Data cleaning → Analysis → Model building → Deployment
- **Technology Stack**: Python 3.x, Jupyter Notebooks, Streamlit, scikit-learn, pandas, numpy
- **Deployment**: Streamlit multi-page web application with interactive UI

### Directory Structure
```
Project-Real-Estate-Price-Prediction-main/
├── Jupyter Notebooks/           # Data science workflow
│   ├── 99acres Web Scraping.ipynb          # Data collection from 99acres.com
│   ├── 99acres Data Cleaning.ipynb         # Data preprocessing and cleaning
│   ├── Real Estate Price Analysis.ipynb    # Exploratory data analysis
│   ├── Geocoders Maps.ipynb               # Location mapping and geocoding
│   └── Project Model Building.ipynb       # ML model development
├── Datasets/                  # Raw and processed data
│   ├── Raw_Property.csv                   # Initial scraped data
│   ├── Property_Location.csv               # Location-specific data
│   ├── Final_Project.csv                  # Cleaned dataset for modeling
│   └── Various property subsets (Prop_001to050.csv, etc.)
├── Deployment/                 # Production-ready application
│   ├── app.py                              # Main Streamlit application
│   ├── eda_app.py                         # Data analysis module
│   ├── ml_app.py                          # ML prediction module
│   ├── regression_model.pkl               # Trained model artifact
│   ├── requirements.txt                   # Production dependencies
│   └── IMG/                               # Visualization assets
└── private/                    # Virtual environment
```

## Development Environment Setup

### Virtual Environment Management
```bash
# Create virtual environment
python -m venv private

# Activate virtual environment (Windows)
private\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate requirements file after adding new packages
pip freeze > requirements.txt
```

### Running the Application
```bash
# Navigate to Deployment directory
cd Deployment

# Run Streamlit application
streamlit run app.py

# Or run from project root with working directory
streamlit run Deployment\app.py --cwd Deployment
```

## Data Science Workflow

### 1. Data Collection (Web Scraping)
- **Source**: www.99acres.com (Mumbai property listings)
- **Notebook**: `99acres Web Scraping.ipynb`
- **Output**: Raw property data with attributes like price, area, location, bedrooms, etc.

### 2. Data Cleaning and Preprocessing
- **Notebook**: `99acres Data Cleaning.ipynb`
- **Processes**: Handle missing values, standardize formats, remove duplicates
- **Key Features**: Price normalization, area standardization, location categorization

### 3. Exploratory Data Analysis
- **Notebook**: `Real Estate Price Analysis.ipynb`
- **Focus**: Price distributions, location analysis, correlation studies
- **Visualizations**: Price range distributions, regional comparisons, feature relationships

### 4. Geocoding and Location Mapping
- **Notebook**: `Geocoders Maps.ipynb`
- **Purpose**: Convert location names to coordinates for spatial analysis
- **Tools**: Geocoding APIs for Mumbai region mapping

### 5. Model Building and Evaluation
- **Notebook**: `Project Model Building.ipynb`
- **Algorithms Compared**:
  - Linear Regression (baseline)
  - Decision Tree Regression
  - Random Forest Regression
  - Polynomial Features with Linear Regression
- **Evaluation Metrics**: RMSE, R² Score, Cross-validation
- **Final Model**: Best performing model saved as `regression_model.pkl`

## Machine Learning Implementation

### Feature Engineering
- **Numerical Features**: Area (SqFt), Floor Number, Property Age
- **Categorical Features**: Location/Region, Number of Bedrooms, Number of Bathrooms
- **Target Variable**: Property Price (in Lakhs)

### Model Architecture (ml_app.py)
```python
def predict_price(Area_SqFt, Floor_No, Bedroom):
    x = np.zeros(7)  # Feature vector initialization
    x[0] = Area_SqFt
    x[1] = Floor_No  
    x[2] = Bedroom
    return reg.predict([x])[0]  # Predict price in Lakhs
```

### Model Performance
- **Best Algorithm**: Random Forest Regression (based on project structure)
- **Prediction Range**: Property prices in Mumbai market
- **Input Features**: Area, Floor, Bedrooms, Location, Bathrooms, Property Age

## Streamlit Application Architecture

### Main Application (app.py)
- **Navigation**: Sidebar menu with Home, Data Analysis, Prediction, About sections
- **UI Components**: Images, text, expandable sections, social media links
- **Multi-page Structure**: Modular design with separate apps for EDA and ML

### Data Analysis Module (eda_app.py)
- **Descriptive Analytics**: Dataset overview, data types, summary statistics
- **Location Analysis**: Regional distribution of properties
- **Visualizations**: Pre-generated plots for price analysis
  - Price range distributions
  - Floor-wise price analysis
  - Bedroom/Bathroom correlations
  - Property age impact
  - Area vs Price scatter plots
  - Regional price comparisons (Central Mumbai, South Mumbai, Thane)

### Prediction Module (ml_app.py)
- **User Inputs**: 
  - Location selection (dropdown from unique regions)
  - Area in SqFt (slider with range 500-max_area)
  - Floor number (dropdown)
  - Number of bathrooms (dropdown)
  - Number of bedrooms (dropdown)
  - Property age (dropdown)
- **Prediction Output**: Price in Lakhs with success message
- **Model Loading**: Pre-trained model loaded via pickle

## Code Standards and Patterns

### Import Structure
```python
# Standard library imports
import pickle

# Third-party imports
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Local imports
from ml_app import run_ml_app
from eda_app import run_eda_app
```

### Streamlit UI Patterns
- **Selection Components**: `st.selectbox()` for categorical choices
- **Range Inputs**: `st.slider()` for numerical ranges
- **Action Buttons**: `st.button()` for triggering predictions
- **Expandable Sections**: `st.expander()` for organized content display
- **Status Messages**: `st.success()`, `st.write()` for user feedback

### Data Loading Patterns
```python
# CSV data loading
df = pd.read_csv("Final_Project.csv")

# Model loading
pickle_in = open('regression_model.pkl','rb')
reg = pickle.load(pickle_in)
```

### File Path Conventions
- **Windows Paths**: Use backslashes for file paths (`IMG\filename.png`)
- **Relative Paths**: All paths relative to Deployment directory
- **Asset Organization**: Images stored in `IMG/` subdirectory

## Dependencies and Requirements

### Core Dependencies (requirements.txt)
- **Data Science**: pandas, numpy, scikit-learn
- **Web Framework**: streamlit
- **Image Processing**: Pillow (PIL)
- **Visualization**: matplotlib (implied), altair
- **Utilities**: requests, python-dateutil

### Version Management
- **Production**: Use specific versions in requirements.txt
- **Development**: Latest compatible versions for new features
- **Compatibility**: Python 3.x with Windows environment support

## Common Development Tasks

### Adding New Features
1. Update the relevant module (eda_app.py or ml_app.py)
2. Test functionality locally with sample data
3. Update requirements.txt if new dependencies added
4. Test full application flow

### Data Updates
1. Run web scraping notebook for new data
2. Execute data cleaning pipeline
3. Retrain model with updated dataset
4. Replace model pickle file in Deployment/
5. Update visualizations if needed

### UI Modifications
1. Modify Streamlit components in relevant .py files
2. Update image assets in IMG/ directory if needed
3. Test responsive behavior and user interactions
4. Ensure consistent styling across all pages

## Troubleshooting Guide

### Common Issues
- **Module Import Errors**: Check virtual environment activation
- **Model Loading Issues**: Verify pickle file path and compatibility
- **Data Loading Problems**: Check CSV file paths and formats
- **Streamlit Port Conflicts**: Use different port with `--server.port` flag

### Performance Considerations
- **Model Loading**: Load model once at module level
- **Data Caching**: Streamlit handles data caching automatically
- **Image Optimization**: Use appropriately sized images for web display

## Deployment and Production

### Local Development
- Use virtual environment for dependency isolation
- Test with sample data before full deployment
- Monitor Streamlit console for errors and warnings

### Production Deployment
- **Streamlit Cloud**: Deploy via streamlit.io platform
- **Environment Variables**: Configure any API keys or sensitive data
- **Monitoring**: Set up logging and error tracking
- **Updates**: Version control model files and dependencies

This documentation serves as a comprehensive guide for agents working on the Real Estate Price Prediction project, covering all aspects from development setup to production deployment.
