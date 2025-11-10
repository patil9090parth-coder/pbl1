import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import requests
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class MarketInsightsDashboard:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.insights = {}
        self.market_trends = {}
        
    def load_market_data(self):
        """Load and generate synthetic market data for demonstration"""
        np.random.seed(42)
        
        # Generate comprehensive market data
        locations = ['Andheri West', 'Bandra West', 'Juhu', 'Goregaon East', 'Malad West', 
                    'Kandivali West', 'Borivali West', 'Dahisar', 'Mira Road', 'Bhayander',
                    'Powai', 'Vile Parle', 'Santacruz', 'Khar', 'Versova']
        
        property_types = ['Apartment', 'Villa', 'Row House', 'Studio', 'Penthouse', 'Duplex']
        
        # Generate 2 years of data
        dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
        
        data = []
        for date in dates:
            n_properties = np.random.randint(5, 15)
            for _ in range(n_properties):
                location = np.random.choice(locations)
                property_type = np.random.choice(property_types)
                
                # Location-based price multipliers
                location_multiplier = {
                    'Bandra West': 1.5, 'Juhu': 1.4, 'Powai': 1.3,
                    'Andheri West': 1.2, 'Vile Parle': 1.1, 'Santacruz': 1.1,
                    'Goregaon East': 1.0, 'Malad West': 0.95, 'Khar': 1.2,
                    'Versova': 1.1, 'Kandivali West': 0.9, 'Borivali West': 0.85,
                    'Dahisar': 0.8, 'Mira Road': 0.75, 'Bhayander': 0.7
                }
                
                area_sqft = np.random.randint(400, 3000)
                base_price_per_sqft = np.random.randint(8000, 20000)
                adjusted_price_per_sqft = base_price_per_sqft * location_multiplier.get(location, 1.0)
                
                # Add seasonal trends
                seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
                
                # Add market trend
                trend_factor = 1.0 + 0.15 * (date.year - 2022) + 0.02 * date.month
                
                final_price_per_sqft = adjusted_price_per_sqft * seasonal_factor * trend_factor
                total_price = area_sqft * final_price_per_sqft
                
                data.append({
                    'Date': date,
                    'Location': location,
                    'Property_Type': property_type,
                    'Area_sqft': area_sqft,
                    'Bedrooms': np.random.randint(1, 6),
                    'Bathrooms': np.random.randint(1, 5),
                    'Floor': np.random.randint(1, 25),
                    'Age': np.random.randint(0, 30),
                    'Price_per_sqft': final_price_per_sqft,
                    'Total_Price': total_price,
                    'Furnishing': np.random.choice(['Unfurnished', 'Semi-furnished', 'Fully furnished']),
                    'Parking': np.random.randint(0, 3),
                    'Balcony': np.random.randint(0, 4),
                    'Amenities_Score': np.random.randint(1, 11),
                    'Maintenance_Score': np.random.randint(1, 11)
                })
        
        self.data = pd.DataFrame(data)
        self.process_market_data()
        return self.data
    
    def process_market_data(self):
        """Process and enrich market data"""
        if self.data is None:
            return
        
        df = self.data.copy()
        
        # Add derived features
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.quarter
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        df['Week_of_Year'] = df['Date'].dt.isocalendar().week
        
        # Price categories
        df['Price_Category'] = pd.cut(df['Total_Price'], 
                                     bins=[0, 10000000, 25000000, 50000000, 100000000, float('inf')],
                                     labels=['Budget', 'Mid-Range', 'Premium', 'Luxury', 'Ultra-Luxury'])
        
        # Size categories
        df['Size_Category'] = pd.cut(df['Area_sqft'],
                                      bins=[0, 500, 1000, 1500, 3000, float('inf')],
                                      labels=['Compact', 'Medium', 'Large', 'Extra Large', 'Luxury'])
        
        # Age categories
        df['Age_Category'] = pd.cut(df['Age'],
                                   bins=[0, 5, 10, 20, 50, float('inf')],
                                   labels=['New', 'Recent', 'Moderate', 'Old', 'Very Old'])
        
        self.processed_data = df
        return df
    
    def generate_market_insights(self):
        """Generate comprehensive market insights"""
        if self.processed_data is None:
            return {}
        
        df = self.processed_data
        
        # Basic statistics
        insights = {
            'total_properties': len(df),
            'avg_price': df['Total_Price'].mean(),
            'median_price': df['Total_Price'].median(),
            'avg_price_per_sqft': df['Price_per_sqft'].mean(),
            'price_volatility': df['Total_Price'].std() / df['Total_Price'].mean(),
            'avg_area': df['Area_sqft'].mean(),
            'most_popular_location': df['Location'].mode()[0],
            'most_popular_type': df['Property_Type'].mode()[0]
        }
        
        # Location-wise analysis
        location_stats = df.groupby('Location').agg({
            'Total_Price': ['mean', 'median', 'count'],
            'Price_per_sqft': 'mean',
            'Area_sqft': 'mean'
        }).round(2)
        
        location_stats.columns = ['Avg_Price', 'Median_Price', 'Property_Count', 'Avg_Price_per_sqft', 'Avg_Area']
        insights['location_stats'] = location_stats.sort_values('Avg_Price', ascending=False)
        
        # Price trends
        monthly_trends = df.groupby([df['Date'].dt.to_period('M')]).agg({
            'Total_Price': 'mean',
            'Price_per_sqft': 'mean'
        }).round(2)
        
        insights['monthly_trends'] = monthly_trends
        
        # Seasonal analysis
        seasonal_stats = df.groupby('Month').agg({
            'Total_Price': 'mean',
            'Price_per_sqft': 'mean'
        }).round(2)
        
        insights['seasonal_stats'] = seasonal_stats
        
        self.insights = insights
        return insights
    
    def detect_market_anomalies(self):
        """Detect market anomalies and outliers"""
        if self.processed_data is None:
            return None
        
        df = self.processed_data
        
        # Use Isolation Forest for anomaly detection
        features = ['Total_Price', 'Price_per_sqft', 'Area_sqft', 'Age']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])
        
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = isolation_forest.fit_predict(scaled_features)
        
        df['Is_Anomaly'] = anomalies == -1
        anomaly_data = df[df['Is_Anomaly']].copy()
        
        return anomaly_data
    
    def predict_market_trends(self, days_ahead=30):
        """Predict future market trends"""
        if self.processed_data is None:
            return None
        
        df = self.processed_data
        
        # Simple trend prediction based on historical data
        recent_data = df[df['Date'] >= (df['Date'].max() - timedelta(days=90))]
        
        # Calculate trend
        daily_avg = recent_data.groupby(recent_data['Date'].dt.date)['Total_Price'].mean()
        trend_slope = np.polyfit(range(len(daily_avg)), daily_avg.values, 1)[0]
        
        # Generate predictions
        future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), 
                                    periods=days_ahead, freq='D')
        
        predictions = []
        for i, date in enumerate(future_dates):
            base_price = daily_avg.iloc[-1]
            predicted_price = base_price + (trend_slope * (i + 1))
            predictions.append({
                'Date': date,
                'Predicted_Price': predicted_price,
                'Confidence_Lower': predicted_price * 0.95,
                'Confidence_Upper': predicted_price * 1.05
            })
        
        return pd.DataFrame(predictions)

# Streamlit interface
def create_market_insights_dashboard():
    """Create the market insights dashboard interface"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
        <h2 style="color: white; text-align: center; margin-bottom: 1rem;">
            📊 Market Intelligence Dashboard
        </h2>
        <p style="color: rgba(255,255,255,0.9); text-align: center; font-size: 1.1rem;">
            Comprehensive real estate market analysis and insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize dashboard
    @st.cache_resource
    def load_dashboard():
        dashboard = MarketInsightsDashboard()
        dashboard.load_market_data()
        dashboard.generate_market_insights()
        return dashboard
    
    dashboard = load_dashboard()
    insights = dashboard.insights
    
    # Key metrics
    st.markdown("### 📈 Key Market Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Properties",
            value=f"{insights['total_properties']:,}",
            delta="+12.3%"
        )
    
    with col2:
        st.metric(
            label="Average Price",
            value=f"₹{insights['avg_price']/1000000:.1f}Cr",
            delta="+8.7%"
        )
    
    with col3:
        st.metric(
            label="Price per sqft",
            value=f"₹{insights['avg_price_per_sqft']/1000:.1f}K",
            delta="+5.2%"
        )
    
    with col4:
        st.metric(
            label="Market Volatility",
            value=f"{insights['price_volatility']:.2f}",
            delta="-2.1%"
        )
    
    # Market trends visualization
    st.markdown("### 📊 Market Trends Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Price Trends", "Location Analysis", "Seasonal Patterns", "Market Anomalies"])
    
    with tab1:
        # Price trends over time
        monthly_data = dashboard.processed_data.groupby(dashboard.processed_data['Date'].dt.to_period('M')).agg({
            'Total_Price': ['mean', 'median'],
            'Price_per_sqft': 'mean'
        }).round(2)
        
        monthly_data.columns = ['Avg_Price', 'Median_Price', 'Avg_Price_per_sqft']
        monthly_data.index = monthly_data.index.to_timestamp()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Average Property Prices Over Time', 'Price per sqft Trends'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index, 
                y=monthly_data['Avg_Price']/1000000,
                mode='lines+markers', 
                name='Average Price',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index, 
                y=monthly_data['Median_Price']/1000000,
                mode='lines+markers', 
                name='Median Price',
                line=dict(color='#764ba2', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index, 
                y=monthly_data['Avg_Price_per_sqft']/1000,
                mode='lines+markers', 
                name='Price per sqft',
                line=dict(color='#f093fb', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=16, color='white')
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price (Cr)", row=1, col=1)
        fig.update_yaxes(title_text="Price per sqft (K)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Location analysis
        location_stats = insights['location_stats']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average price by location
            fig = px.bar(
                location_stats.head(10).reset_index(),
                x='Location',
                y='Avg_Price',
                title='Average Property Prices by Location',
                color='Avg_Price',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=16, color='white'),
                xaxis_tickangle=-45
            )
            
            fig.update_yaxes(title_text="Average Price (₹)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Property count by location
            fig = px.bar(
                location_stats.head(10).reset_index(),
                x='Location',
                y='Property_Count',
                title='Number of Properties by Location',
                color='Property_Count',
                color_continuous_scale='plasma'
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=16, color='white'),
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Seasonal analysis
        seasonal_stats = insights['seasonal_stats']
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Average Prices by Month', 'Price per sqft by Month'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Bar(
                x=seasonal_stats.index,
                y=seasonal_stats['Total_Price']/1000000,
                name='Average Price',
                marker_color='#667eea'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=seasonal_stats.index,
                y=seasonal_stats['Price_per_sqft']/1000,
                name='Price per sqft',
                marker_color='#764ba2'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=16, color='white')
        )
        
        fig.update_xaxes(title_text="Month", row=1, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Average Price (Cr)", row=1, col=1)
        fig.update_yaxes(title_text="Price per sqft (K)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Market anomalies
        anomalies = dashboard.detect_market_anomalies()
        
        if anomalies is not None and len(anomalies) > 0:
            st.warning(f"⚠️ Detected {len(anomalies)} potential market anomalies")
            
            # Anomaly visualization
            fig = px.scatter(
                anomalies,
                x='Area_sqft',
                y='Total_Price',
                color='Location',
                size='Price_per_sqft',
                title='Market Anomalies Detection',
                hover_data=['Location', 'Property_Type', 'Age']
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=16, color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly details
            with st.expander("🔍 View Anomaly Details"):
                st.dataframe(anomalies[['Location', 'Property_Type', 'Total_Price', 'Price_per_sqft', 'Area_sqft']].head(10))
        else:
            st.success("✅ No significant market anomalies detected")
    
    # Market predictions
    st.markdown("### 🔮 Market Predictions")
    
    with st.expander("📈 View 30-Day Market Forecast"):
        predictions = dashboard.predict_market_trends(30)
        
        if predictions is not None:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted_Price']/1000000,
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='#667eea', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Confidence_Lower']/1000000,
                mode='lines',
                name='Lower Confidence',
                line=dict(color='rgba(102, 126, 234, 0.3)', width=1),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Confidence_Upper']/1000000,
                mode='lines',
                name='Upper Confidence',
                line=dict(color='rgba(102, 126, 234, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.1)',
                showlegend=False
            ))
            
            fig.update_layout(
                title='30-Day Market Price Prediction',
                xaxis_title='Date',
                yaxis_title='Predicted Price (Cr)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=16, color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction summary
            latest_prediction = predictions.iloc[-1]
            st.info(f"""
            **30-Day Prediction Summary:**
            - Predicted Price: ₹{latest_prediction['Predicted_Price']/1000000:.2f} Cr
            - Confidence Range: ₹{latest_prediction['Confidence_Lower']/1000000:.2f} Cr - ₹{latest_prediction['Confidence_Upper']/1000000:.2f} Cr
            - Expected Change: {((latest_prediction['Predicted_Price'] - predictions.iloc[0]['Predicted_Price']) / predictions.iloc[0]['Predicted_Price'] * 100):.1f}%
            """)
    
    # Interactive filters
    st.markdown("### 🔍 Interactive Data Explorer")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_locations = st.multiselect(
            "Select Locations",
            options=dashboard.processed_data['Location'].unique(),
            default=dashboard.processed_data['Location'].unique()[:5]
        )
    
    with col2:
        selected_types = st.multiselect(
            "Property Types",
            options=dashboard.processed_data['Property_Type'].unique(),
            default=dashboard.processed_data['Property_Type'].unique()
        )
    
    with col3:
        price_range = st.slider(
            "Price Range (Cr)",
            min_value=float(dashboard.processed_data['Total_Price'].min()/10000000),
            max_value=float(dashboard.processed_data['Total_Price'].max()/10000000),
            value=(1.0, 10.0),
            step=0.5
        )
    
    with col4:
        area_range = st.slider(
            "Area Range (sqft)",
            min_value=int(dashboard.processed_data['Area_sqft'].min()),
            max_value=int(dashboard.processed_data['Area_sqft'].max()),
            value=(500, 2000),
            step=100
        )
    
    # Filter data
    filtered_data = dashboard.processed_data[
        (dashboard.processed_data['Location'].isin(selected_locations)) &
        (dashboard.processed_data['Property_Type'].isin(selected_types)) &
        (dashboard.processed_data['Total_Price'] >= price_range[0] * 10000000) &
        (dashboard.processed_data['Total_Price'] <= price_range[1] * 10000000) &
        (dashboard.processed_data['Area_sqft'] >= area_range[0]) &
        (dashboard.processed_data['Area_sqft'] <= area_range[1])
    ]
    
    # Custom visualization based on filtered data
    if len(filtered_data) > 0:
        fig = px.scatter_3d(
            filtered_data,
            x='Area_sqft',
            y='Price_per_sqft',
            z='Total_Price',
            color='Location',
            size='Age',
            title='3D Property Analysis (Filtered Data)',
            hover_data=['Location', 'Property_Type', 'Bedrooms', 'Bathrooms']
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Area (sqft)',
                yaxis_title='Price per sqft',
                zaxis_title='Total Price'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=16, color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data download
    st.markdown("### 💾 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Market Data"):
            csv = dashboard.processed_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download Insights Report"):
            insights_json = json.dumps({
                'timestamp': datetime.now().isoformat(),
                'key_metrics': {
                    'total_properties': insights['total_properties'],
                    'avg_price': insights['avg_price'],
                    'median_price': insights['median_price'],
                    'avg_price_per_sqft': insights['avg_price_per_sqft']
                },
                'location_stats': insights['location_stats'].to_dict('index'),
                'top_locations': insights['location_stats'].head().to_dict('index')
            }, indent=2, default=str)
            
            st.download_button(
                label="Download JSON Report",
                data=insights_json,
                file_name=f"market_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Create the dashboard
if __name__ == "__main__":
    create_market_insights_dashboard()