# Real Estate Price Prediction App

A comprehensive real estate price prediction application built with Streamlit and machine learning.

## Features

### 🏠 **Basic Prediction**
- Simple interface for quick price predictions
- Uses polynomial regression model
- Basic feature inputs (Area, Floor, Bedrooms, Bathrooms, Age, Location)

### 🚀 **Enhanced Prediction** 
- Advanced prediction with RandomForest model
- Feature engineering for better accuracy
- Market insights and confidence intervals
- Property analysis and recommendations

### 📊 **Data Analysis**
- Interactive data visualization
- Price distribution analysis
- Location-based insights
- Correlation analysis

## Model Performance

### Enhanced Model Results:
- **Training R² Score**: 0.9986
- **Test R² Score**: 0.9905
- **Mean Absolute Error**: 3.27 Lakhs
- **Cross-validation Score**: 0.9923 ± 0.0028

## Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Maps**: folium
- **Deployment**: Streamlit Cloud

## Local Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Data Sources

The application uses Mumbai real estate data including:
- Property details (area, bedrooms, bathrooms, floor)
- Location information (Central Mumbai, South Mumbai, Thane)
- Age and construction status
- Price information

## Deployment

The app is deployed on Streamlit Cloud and can be accessed at: [Your Streamlit Cloud URL]

## Usage

1. **Basic Prediction**: For quick estimates using the original model
2. **Enhanced Prediction**: For accurate predictions with detailed insights
3. **Data Analysis**: For exploring market trends and patterns

---

*This project demonstrates end-to-end machine learning deployment with a focus on real estate price prediction.*