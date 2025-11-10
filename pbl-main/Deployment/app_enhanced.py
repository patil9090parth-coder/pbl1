import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import requests
import json
from datetime import datetime, timedelta
import time
from streamlit_option_menu import option_menu
import base64
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="RealEstate AI - Smart Property Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': 'AI-powered Real Estate Price Prediction Platform'
    }
)

# Custom CSS for modern design
def load_css():
    st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1rem;
    }
    
    /* Card Styles */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    
    .metric-change {
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
    }
    
    /* Input Styles */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in-left {
        animation: slideInLeft 0.6s ease-out;
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #764ba2, #667eea);
    }
    </style>
    """, unsafe_allow_html=True)

# Load custom CSS
load_css()

# Mock data for demonstration
def load_sample_data():
    """Load sample data for demonstration"""
    np.random.seed(42)
    
    # Sample property data
    locations = ['Andheri West', 'Bandra West', 'Juhu', 'Goregaon East', 'Malad West', 
                'Kandivali West', 'Borivali West', 'Dahisar', 'Mira Road', 'Bhayander']
    
    property_types = ['Apartment', 'Villa', 'Row House', 'Studio', 'Penthouse']
    
    data = {
        'Location': np.random.choice(locations, 1000),
        'Property_Type': np.random.choice(property_types, 1000),
        'Area_sqft': np.random.randint(400, 3000, 1000),
        'Bedrooms': np.random.randint(1, 6, 1000),
        'Bathrooms': np.random.randint(1, 5, 1000),
        'Floor': np.random.randint(1, 25, 1000),
        'Age': np.random.randint(0, 30, 1000),
        'Price_per_sqft': np.random.randint(8000, 25000, 1000),
        'Total_Price': np.zeros(1000)
    }
    
    df = pd.DataFrame(data)
    df['Total_Price'] = df['Area_sqft'] * df['Price_per_sqft']
    
    return df

# Load data
@st.cache_data
def get_data():
    return load_sample_data()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# Header section
def render_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="main-header fade-in">
            <h1 class="header-title">🏠 RealEstate AI</h1>
            <p class="header-subtitle">Smart Property Price Prediction Platform</p>
            <p style="color: #888; font-size: 1rem;">
                Powered by Advanced Machine Learning & Market Intelligence
            </p>
        </div>
        """, unsafe_allow_html=True)

# Navigation menu
def render_navigation():
    selected = option_menu(
        menu_title=None,
        options=["Home", "Market Insights", "Price Predictor", "Property Compare", "Favorites", "About"],
        icons=["house", "graph-up", "calculator", "arrow-left-right", "heart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "rgba(255,255,255,0.1)", "border-radius": "15px"},
            "icon": {"color": "white", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0px",
                "padding": "10px 15px",
                "border-radius": "10px",
                "color": "white",
                "background-color": "transparent"
            },
            "nav-link-selected": {"background-color": "rgba(102, 126, 234, 0.8)", "color": "white"},
            "nav-link-hover": {"background-color": "rgba(255,255,255,0.2)"}
        }
    )
    st.session_state.page = selected
    return selected

# Home page
def render_home():
    st.markdown("""
    <div class="fade-in">
        <h2 style="text-align: center; color: white; margin-bottom: 2rem;">
            🚀 Welcome to the Future of Real Estate
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card slide-in-left">
            <div class="metric-value">99.05%</div>
            <div class="metric-label">Prediction Accuracy</div>
            <div class="metric-change" style="color: #28a745;">↑ 2.3%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="animation-delay: 0.1s;">
            <div class="metric-value">1,247</div>
            <div class="metric-label">Properties Analyzed</div>
            <div class="metric-change" style="color: #28a745;">↑ 15.2%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="animation-delay: 0.2s;">
            <div class="metric-value">₹2.4Cr</div>
            <div class="metric-label">Avg. Property Value</div>
            <div class="metric-change" style="color: #dc3545;">↓ 3.1%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="animation-delay: 0.3s;">
            <div class="metric-value">4.8★</div>
            <div class="metric-label">User Rating</div>
            <div class="metric-change" style="color: #28a745;">↑ 0.3★</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("""
    <div class="fade-in" style="margin-top: 3rem;">
        <h3 style="text-align: center; color: white; margin-bottom: 2rem;">
            🌟 Why Choose RealEstate AI?
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
            <h4 style="color: #667eea; margin-bottom: 1rem;">AI-Powered Predictions</h4>
            <p style="color: #666;">Advanced machine learning algorithms provide highly accurate price predictions with confidence intervals.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h4 style="color: #667eea; margin-bottom: 1rem;">Market Intelligence</h4>
            <p style="color: #666;">Real-time market trends, neighborhood insights, and investment opportunity analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
            <h4 style="color: #667eea; margin-bottom: 1rem;">Personalized Experience</h4>
            <p style="color: #666;">Tailored recommendations based on your preferences and investment goals.</p>
        </div>
        """, unsafe_allow_html=True)

# Market Insights page
def render_market_insights():
    st.markdown("""
    <div class="fade-in">
        <h2 style="color: white; margin-bottom: 2rem;">📈 Market Intelligence Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = get_data()
    
    # Market overview charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by location
        fig = px.box(df, x='Location', y='Total_Price', 
                     title='Property Price Distribution by Location',
                     color_discrete_sequence=['#667eea'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=18, color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price per sqft analysis
        fig = px.scatter(df, x='Area_sqft', y='Price_per_sqft', 
                        color='Location', size='Bedrooms',
                        title='Price per Sqft vs Area Analysis',
                        color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=18, color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Market trends
    st.markdown("""
    <div class="fade-in" style="margin-top: 2rem;">
        <h3 style="color: white; margin-bottom: 1rem;">📊 Market Trends</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate trend data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
    trend_data = pd.DataFrame({
        'Date': dates,
        'Avg_Price': np.random.normal(15000000, 2000000, len(dates)) + np.linspace(0, 2000000, len(dates)),
        'Volume': np.random.normal(150, 30, len(dates))
    })
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average Property Prices', 'Transaction Volume'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=trend_data['Date'], y=trend_data['Avg_Price'], 
                  mode='lines+markers', name='Avg Price', line=dict(color='#667eea', width=3)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=trend_data['Date'], y=trend_data['Volume'], 
               name='Volume', marker_color='#764ba2'),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(size=18, color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Price Predictor page
def render_price_predictor():
    st.markdown("""
    <div class="fade-in">
        <h2 style="color: white; margin-bottom: 2rem;">🎯 Smart Price Predictor</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #667eea; margin-bottom: 1rem;">Property Details</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Property form
        with st.form("property_form"):
            col_left, col_right = st.columns(2)
            
            with col_left:
                location = st.selectbox("Location", 
                                        ['Andheri West', 'Bandra West', 'Juhu', 'Goregaon East', 'Malad West'])
                property_type = st.selectbox("Property Type", 
                                           ['Apartment', 'Villa', 'Row House', 'Studio', 'Penthouse'])
                area_sqft = st.number_input("Area (sqft)", min_value=200, max_value=5000, value=1000)
                bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
            
            with col_right:
                bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
                floor = st.number_input("Floor", min_value=1, max_value=50, value=5)
                age = st.number_input("Property Age (years)", min_value=0, max_value=50, value=5)
                furnishing = st.selectbox("Furnishing", ['Unfurnished', 'Semi-furnished', 'Fully furnished'])
            
            submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #667eea; margin-bottom: 1rem;">Prediction Results</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if submitted:
            # Mock prediction calculation
            base_price = area_sqft * np.random.randint(8000, 15000)
            location_multiplier = np.random.uniform(0.8, 1.5)
            type_multiplier = np.random.uniform(0.9, 1.3)
            
            predicted_price = base_price * location_multiplier * type_multiplier
            confidence_interval = (predicted_price * 0.9, predicted_price * 1.1)
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            st.metric(
                label="Predicted Price",
                value=f"₹{predicted_price:,.0f}",
                delta=f"±{((confidence_interval[1] - predicted_price) / predicted_price * 100):.1f}%"
            )
            
            st.write(f"**Confidence Range:**")
            st.write(f"₹{confidence_interval[0]:,.0f} - ₹{confidence_interval[1]:,.0f}")
            
            # Add to favorites button
            if st.button("❤️ Save Prediction"):
                st.session_state.predictions.append({
                    'location': location,
                    'type': property_type,
                    'area': area_sqft,
                    'price': predicted_price,
                    'timestamp': datetime.now()
                })
                st.success("Prediction saved!")

# Property Compare page
def render_property_compare():
    st.markdown("""
    <div class="fade-in">
        <h2 style="color: white; margin-bottom: 2rem;">⚖️ Property Comparison Tool</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = get_data()
    
    # Property selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #667eea; margin-bottom: 1rem;">Property A</h4>
        </div>
        """, unsafe_allow_html=True)
        
        prop_a_location = st.selectbox("Location A", df['Location'].unique(), key="prop_a_loc")
        prop_a_type = st.selectbox("Type A", df['Property_Type'].unique(), key="prop_a_type")
        prop_a_area = st.number_input("Area A (sqft)", min_value=200, max_value=3000, value=1000, key="prop_a_area")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #764ba2; margin-bottom: 1rem;">Property B</h4>
        </div>
        """, unsafe_allow_html=True)
        
        prop_b_location = st.selectbox("Location B", df['Location'].unique(), key="prop_b_loc")
        prop_b_type = st.selectbox("Type B", df['Property_Type'].unique(), key="prop_b_type")
        prop_b_area = st.number_input("Area B (sqft)", min_value=200, max_value=3000, value=1200, key="prop_b_area")
    
    if st.button("🔍 Compare Properties", use_container_width=True):
        # Mock comparison data
        prop_a_price = prop_a_area * np.random.randint(8000, 15000)
        prop_b_price = prop_b_area * np.random.randint(8000, 15000)
        
        # Comparison chart
        comparison_data = pd.DataFrame({
            'Property': ['Property A', 'Property B'],
            'Price': [prop_a_price, prop_b_price],
            'Area': [prop_a_area, prop_b_area],
            'Price_per_sqft': [prop_a_price/prop_a_area, prop_b_price/prop_b_area]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comparison_data, x='Property', y='Price',
                        title='Total Price Comparison',
                        color='Property',
                        color_discrete_sequence=['#667eea', '#764ba2'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=18, color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comparison_data, x='Property', y='Price_per_sqft',
                        title='Price per Sqft Comparison',
                        color='Property',
                        color_discrete_sequence=['#667eea', '#764ba2'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=18, color='white')
            )
            st.plotly_chart(fig, use_container_width=True)

# Favorites page
def render_favorites():
    st.markdown("""
    <div class="fade-in">
        <h2 style="color: white; margin-bottom: 2rem;">❤️ Saved Predictions</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.predictions:
        for i, prediction in enumerate(st.session_state.predictions):
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea;">Prediction #{i+1}</h4>
                    <p><strong>Location:</strong> {prediction['location']}</p>
                    <p><strong>Type:</strong> {prediction['type']}</p>
                    <p><strong>Area:</strong> {prediction['area']} sqft</p>
                    <p><strong>Predicted Price:</strong> ₹{prediction['price']:,.0f}</p>
                    <p><strong>Time:</strong> {prediction['timestamp'].strftime('%Y-%m-%d %H:%M')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🗺️ View on Map", key=f"map_{i}"):
                        st.info("Map integration would show location here")
                with col2:
                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.predictions.pop(i)
                        st.rerun()
    else:
        st.info("No saved predictions yet. Use the Price Predictor to save your first prediction!")

# About page
def render_about():
    st.markdown("""
    <div class="fade-in">
        <h2 style="color: white; margin-bottom: 2rem;">ℹ️ About RealEstate AI</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #667eea; margin-bottom: 1rem;">Our Mission</h4>
        <p style="margin-bottom: 1.5rem;">
            RealEstate AI is revolutionizing the real estate industry by providing accurate, 
            data-driven property price predictions using advanced machine learning algorithms.
        </p>
        
        <h4 style="color: #667eea; margin-bottom: 1rem;">Key Features</h4>
        <ul style="margin-bottom: 1.5rem;">
            <li>🔮 AI-powered price predictions with confidence intervals</li>
            <li>📊 Real-time market intelligence and trends</li>
            <li>⚖️ Property comparison tools</li>
            <li>🎯 Personalized recommendations</li>
            <li>📱 Mobile-responsive design</li>
            <li>🔒 Secure and private data handling</li>
        </ul>
        
        <h4 style="color: #667eea; margin-bottom: 1rem;">Technology Stack</h4>
        <p style="margin-bottom: 1rem;">
            Built with Python, Streamlit, Scikit-learn, Plotly, and deployed on modern cloud infrastructure.
        </p>
        
        <h4 style="color: #667eea; margin-bottom: 1rem;">Contact Us</h4>
        <p>
            📧 Email: info@realestateai.com<br>
            🌐 Website: www.realestateai.com<br>
            📱 Phone: +91-XXXXXXXXXX
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main app logic
def main():
    # Render header
    render_header()
    
    # Render navigation
    selected_page = render_navigation()
    
    # Render selected page
    if selected_page == "Home":
        render_home()
    elif selected_page == "Market Insights":
        render_market_insights()
    elif selected_page == "Price Predictor":
        render_price_predictor()
    elif selected_page == "Property Compare":
        render_property_compare()
    elif selected_page == "Favorites":
        render_favorites()
    elif selected_page == "About":
        render_about()

if __name__ == "__main__":
    main()