import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Real Estate Analytics Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS styling
def load_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        color: white;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        width: 100%;
    }
    h1, h2, h3 {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load sample data
def load_sample_data():
    np.random.seed(42)
    locations = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Pune', 'Chennai', 'Kolkata', 'Ahmedabad']
    property_types = ['Apartment', 'Villa', 'Row House', 'Penthouse']
    
    n_samples = 1000
    
    data = {
        'Location': np.random.choice(locations, n_samples),
        'Property_Type': np.random.choice(property_types, n_samples),
        'Area_sqft': np.random.randint(500, 3000, n_samples),
        'Price_per_sqft': np.random.randint(5000, 25000, n_samples),
        'Bedrooms': np.random.randint(1, 6, n_samples),
        'Bathrooms': np.random.randint(1, 5, n_samples),
        'Floor': np.random.randint(1, 21, n_samples),
        'Total_Floors': np.random.randint(5, 25, n_samples),
        'Age_years': np.random.randint(0, 20, n_samples),
        'Parking': np.random.choice([0, 1, 2], n_samples),
        'Lift': np.random.choice([0, 1], n_samples),
        'Furnished': np.random.choice(['Unfurnished', 'Semi-Furnished', 'Furnished'], n_samples)
    }
    
    df = pd.DataFrame(data)
    df['Total_Price'] = df['Area_sqft'] * df['Price_per_sqft']
    df['Price_Lakhs'] = df['Total_Price'] / 100000
    
    return df

# Main app
def main():
    load_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Real Estate Analytics Platform</h1>
        <p>AI-powered property price prediction and market insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_sample_data()
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Market Insights", "🔮 Price Predictor", "⚖️ Property Compare", "❤️ Favorites"])
    
    # Initialize session state
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    
    if page == "🏠 Home":
        render_home_page(df)
    elif page == "📊 Market Insights":
        render_market_insights_page(df)
    elif page == "🔮 Price Predictor":
        render_price_predictor_page(df)
    elif page == "⚖️ Property Compare":
        render_property_compare_page(df)
    elif page == "❤️ Favorites":
        render_favorites_page()

def render_home_page(df):
    st.markdown("<h2 style='text-align: center; color: white;'>🏠 Welcome to Real Estate Analytics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea;">📊 Total Properties</h3>
            <h2>{len(df):,}</h2>
            <p>Properties in database</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_price = df['Price_Lakhs'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea;">💰 Average Price</h3>
            <h2>₹{avg_price:.1f}L</h2>
            <p>Average property price</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_area = df['Area_sqft'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea;">📐 Average Area</h3>
            <h2>{avg_area:.0f} sqft</h2>
            <p>Average property size</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample data table
    st.markdown("<h3 style='color: white; margin-top: 2rem;'>📋 Sample Properties</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

def render_market_insights_page(df):
    st.markdown("<h2 style='text-align: center; color: white;'>📊 Market Insights</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price by location
        location_prices = df.groupby('Location')['Price_Lakhs'].mean().reset_index()
        fig = px.bar(location_prices, x='Location', y='Price_Lakhs', 
                     title='Average Price by Location')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Property type distribution
        type_counts = df['Property_Type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                     title='Property Type Distribution')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Price vs Area scatter plot
    st.markdown("<h3 style='color: white;'>📈 Price vs Area Analysis</h3>", unsafe_allow_html=True)
    fig = px.scatter(df, x='Area_sqft', y='Price_Lakhs', color='Location',
                     title='Property Price vs Area')
    st.plotly_chart(fig, use_container_width=True)

def render_price_predictor_page(df):
    st.markdown("<h2 style='text-align: center; color: white;'>🔮 Price Predictor</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.selectbox("Select Location", df['Location'].unique())
        property_type = st.selectbox("Property Type", df['Property_Type'].unique())
        area_sqft = st.number_input("Area (sqft)", min_value=500, max_value=3000, value=1000)
    
    with col2:
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=5, value=2)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=2)
        floor = st.number_input("Floor", min_value=1, max_value=20, value=5)
    
    if st.button("🔮 Predict Price", use_container_width=True):
        # Simple prediction logic
        base_price_per_sqft = df[df['Location'] == location]['Price_per_sqft'].mean()
        
        # Adjustments
        location_multiplier = 1.0
        type_multiplier = 1.2 if property_type == 'Villa' else 1.1 if property_type == 'Penthouse' else 1.0
        size_multiplier = 1.0 + (area_sqft - 1000) / 10000
        
        predicted_price_per_sqft = base_price_per_sqft * location_multiplier * type_multiplier * size_multiplier
        predicted_total_price = area_sqft * predicted_price_per_sqft
        
        confidence_lower = predicted_total_price * 0.9
        confidence_upper = predicted_total_price * 1.1
        
        st.success("✅ Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Price", f"₹{predicted_total_price/100000:.1f}L")
        
        with col2:
            st.metric("Price per sqft", f"₹{predicted_price_per_sqft:,.0f}")
        
        with col3:
            st.metric("Confidence Range", f"±{((confidence_upper - predicted_total_price) / predicted_total_price * 100):.1f}%")
        
        st.info(f"Confidence Range: ₹{confidence_lower/100000:.1f}L - ₹{confidence_upper/100000:.1f}L")
        
        # Save prediction
        if st.button("❤️ Save Prediction"):
            st.session_state.predictions.append({
                'location': location,
                'property_type': property_type,
                'area': area_sqft,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'floor': floor,
                'predicted_price': predicted_total_price,
                'timestamp': datetime.now()
            })
            st.success("Prediction saved!")

def render_property_compare_page(df):
    st.markdown("<h2 style='text-align: center; color: white;'>⚖️ Property Comparison</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='color: white;'>Property A</h4>", unsafe_allow_html=True)
        loc_a = st.selectbox("Location A", df['Location'].unique(), key='loc_a')
        type_a = st.selectbox("Type A", df['Property_Type'].unique(), key='type_a')
        area_a = st.number_input("Area A (sqft)", min_value=500, max_value=3000, value=1000, key='area_a')
    
    with col2:
        st.markdown("<h4 style='color: white;'>Property B</h4>", unsafe_allow_html=True)
        loc_b = st.selectbox("Location B", df['Location'].unique(), key='loc_b')
        type_b = st.selectbox("Type B", df['Property_Type'].unique(), key='type_b')
        area_b = st.number_input("Area B (sqft)", min_value=500, max_value=3000, value=1200, key='area_b')
    
    if st.button("🔍 Compare Properties", use_container_width=True):
        # Calculate prices
        price_per_sqft_a = df[df['Location'] == loc_a]['Price_per_sqft'].mean()
        price_per_sqft_b = df[df['Location'] == loc_b]['Price_per_sqft'].mean()
        
        total_a = area_a * price_per_sqft_a
        total_b = area_b * price_per_sqft_b
        
        comparison_data = pd.DataFrame({
            'Property': ['Property A', 'Property B'],
            'Total Price (Lakhs)': [total_a/100000, total_b/100000],
            'Area (sqft)': [area_a, area_b],
            'Price per sqft': [price_per_sqft_a, price_per_sqft_b]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comparison_data, x='Property', y='Total Price (Lakhs)',
                         title='Total Price Comparison')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comparison_data, x='Property', y='Price per sqft',
                         title='Price per Sqft Comparison')
            st.plotly_chart(fig, use_container_width=True)

def render_favorites_page():
    st.markdown("<h2 style='text-align: center; color: white;'>❤️ Saved Predictions</h2>", unsafe_allow_html=True)
    
    if st.session_state.predictions:
        for i, pred in enumerate(st.session_state.predictions):
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea;">Prediction #{i+1}</h4>
                    <p><strong>Location:</strong> {pred['location']}</p>
                    <p><strong>Type:</strong> {pred['property_type']}</p>
                    <p><strong>Area:</strong> {pred['area']} sqft</p>
                    <p><strong>Predicted Price:</strong> ₹{pred['predicted_price']/100000:.1f}L</p>
                    <p><strong>Time:</strong> {pred['timestamp'].strftime('%Y-%m-%d %H:%M')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"🗑️ Remove #{i+1}", key=f"remove_{i}"):
                    st.session_state.predictions.pop(i)
                    st.rerun()
    else:
        st.info("No saved predictions yet. Use the Price Predictor to save your first prediction!")

if __name__ == "__main__":
    main()