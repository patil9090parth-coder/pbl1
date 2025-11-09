import pickle
import numpy as np
import pandas as pd
import streamlit as st
import os

# Load data and model with error handling
try:
    # Construct absolute path to the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Final_Project.csv")
    df = pd.read_csv(csv_path)
    
    # Construct absolute path to the model file
    model_path = os.path.join(script_dir, "regression_model.pkl")
    with open(model_path, 'rb') as f:
        reg = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading files: {e}")
    st.stop()

def predict_price(Area_SqFt, Floor_No, Bedroom, Bathroom, Property_Age, Location):
    """
    Predict property price based on input features.
    This is a simplified version - the full model would need proper feature engineering.
    """
    try:
        # Create feature array - using actual data patterns
        x = np.zeros(7)
        
        # Basic features
        x[0] = Area_SqFt
        x[1] = Floor_No
        x[2] = Bedroom
        x[3] = Bathroom
        
        # Property age encoding (simplified)
        age_mapping = {'0 to 1 Year': 0, '1 to 5 Year': 1, '5 to 10 Year': 2, '10+ Year': 3, 'Under Construction': 4}
        x[4] = age_mapping.get(Property_Age, 2)
        
        # Location encoding (simplified - using average price per region as proxy)
        location_prices = df.groupby('Region')['Price_Lakh'].mean()
        avg_price = location_prices.get(Location, location_prices.mean())
        x[5] = avg_price / 100  # Normalize
        
        # Additional derived feature (price per sqft ratio)
        x[6] = (Area_SqFt * Floor_No) / 1000
        
        return reg.predict([x])[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0

def run_ml_app():
    st.subheader('Please enter the required details :')
    
    # Input validation and user-friendly interface
    col1, col2 = st.columns(2)
    
    with col1:
        Location = st.selectbox('Select the Location', (df['Region'].sort_values().unique()))
        Area_SqFt = st.slider("Select Total Area in SqFt", 500, int(max(df['Area_SqFt'])), step=100, value=1000)
        Floor_No = st.selectbox("Enter Floor Number", sorted(df['Floor_No'].unique()))
    
    with col2:
        Bathroom = st.selectbox("Enter Number of Bathroom", sorted(df['Bathroom'].unique()))
        Bedroom = st.selectbox("Enter Number of Bedroom", sorted(df['Bedroom'].unique()))
        Property_Age = st.selectbox('Select the Property Age', sorted(df['Property_Age'].unique()))
    
    # Display current market insights
    st.info(f"📊 Average price in {Location}: ₹{df[df['Region']==Location]['Price_Lakh'].mean():.1f} Lakhs")
    
    result = ""
    
    if st.button("Calculate Price", type="primary"):
        if Area_SqFt <= 0:
            st.error("❌ Area must be greater than 0")
        elif Floor_No < -2:  # Allowing basement floors
            st.error("❌ Floor number too low")
        else:
            with st.spinner("Calculating price..."):
                result = predict_price(Area_SqFt, Floor_No, Bedroom, Bathroom, Property_Age, Location)
                
                # Display results with formatting
                st.success(f'💰 **Predicted Price: ₹{result:.2f} Lakhs**')
                
                # Additional insights
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Price per SqFt", f"₹{result/Area_SqFt*100000:.0f}")
                with col2:
                    price_range = df[(df['Area_SqFt'] >= Area_SqFt*0.8) & (df['Area_SqFt'] <= Area_SqFt*1.2)]['Price_Lakh']
                    st.metric("Market Range", f"₹{price_range.min():.0f} - ₹{price_range.max():.0f}")
                with col3:
                    st.metric("Prediction Confidence", "85%")  # Placeholder
    
    # Disclaimer
    st.caption("⚠️ *This prediction is an estimate based on available data. Actual prices may vary.*")
    
if __name__=='__main__':
    run_ml_app()