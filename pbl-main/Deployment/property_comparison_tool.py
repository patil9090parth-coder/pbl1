import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class PropertyComparisonTool:
    def __init__(self):
        self.properties = []
        self.comparison_matrix = None
        self.similarity_scores = None
        
    def add_property(self, property_data):
        """Add a property to the comparison list"""
        self.properties.append({
            'id': len(self.properties) + 1,
            'name': property_data.get('name', f"Property {len(self.properties) + 1}"),
            'location': property_data.get('location', 'Unknown'),
            'area_sqft': property_data.get('area_sqft', 0),
            'bedrooms': property_data.get('bedrooms', 0),
            'bathrooms': property_data.get('bathrooms', 0),
            'floor': property_data.get('floor', 0),
            'age': property_data.get('age', 0),
            'price': property_data.get('price', 0),
            'price_per_sqft': property_data.get('price_per_sqft', 0),
            'furnishing': property_data.get('furnishing', 'Unknown'),
            'parking': property_data.get('parking', 0),
            'balcony': property_data.get('balcony', 0),
            'amenities_score': property_data.get('amenities_score', 0),
            'maintenance_score': property_data.get('maintenance_score', 0),
            'property_type': property_data.get('property_type', 'Unknown'),
            'coordinates': property_data.get('coordinates', None),
            'image_url': property_data.get('image_url', ''),
            'description': property_data.get('description', ''),
            'added_date': datetime.now()
        })
    
    def generate_comparison_matrix(self):
        """Generate comparison matrix with normalized scores"""
        if not self.properties:
            return None
        
        df = pd.DataFrame(self.properties)
        
        # Calculate derived features
        df['price_per_sqft'] = df['price'] / df['area_sqft']
        df['space_efficiency'] = df['area_sqft'] / df['bedrooms'] if df['bedrooms'].any() > 0 else 0
        df['bathroom_ratio'] = df['bathrooms'] / df['bedrooms'] if df['bedrooms'].any() > 0 else 0
        
        # Normalize scores (higher is better for most metrics)
        features_to_normalize = [
            'area_sqft', 'price_per_sqft', 'amenities_score', 'maintenance_score',
            'space_efficiency', 'bathroom_ratio', 'parking', 'balcony'
        ]
        
        normalized_df = df.copy()
        
        for feature in features_to_normalize:
            if feature in df.columns and df[feature].max() != df[feature].min():
                normalized_df[f'{feature}_normalized'] = (
                    (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
                ) * 100
            else:
                normalized_df[f'{feature}_normalized'] = 50  # Default score
        
        # Age score (newer is better)
        if df['age'].max() != df['age'].min():
            normalized_df['age_normalized'] = (
                (df['age'].max() - df['age']) / (df['age'].max() - df['age'].min())
            ) * 100
        else:
            normalized_df['age_normalized'] = 50
        
        # Floor score (higher floors preferred, but with diminishing returns)
        if df['floor'].max() != df['floor'].min():
            normalized_df['floor_normalized'] = np.minimum(
                (df['floor'] / df['floor'].max()) * 100, 90
            )
        else:
            normalized_df['floor_normalized'] = 50
        
        self.comparison_matrix = normalized_df
        return normalized_df
    
    def calculate_similarity_scores(self, reference_property=None):
        """Calculate similarity scores between properties"""
        if self.comparison_matrix is None:
            self.generate_comparison_matrix()
        
        df = self.comparison_matrix
        
        # Select features for similarity calculation
        similarity_features = [
            'area_sqft_normalized', 'age_normalized', 'floor_normalized',
            'amenities_score_normalized', 'maintenance_score_normalized',
            'space_efficiency_normalized', 'bathroom_ratio_normalized'
        ]
        
        # Create feature matrix
        feature_matrix = df[similarity_features].fillna(0).values
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(feature_matrix)
        
        # If reference property is specified, return similarities to it
        if reference_property is not None and 0 <= reference_property < len(self.properties):
            similarity_scores = similarity_matrix[reference_property]
            return similarity_scores
        
        self.similarity_scores = similarity_matrix
        return similarity_matrix
    
    def get_best_value_properties(self, budget_range=None, priority_features=None):
        """Find properties with best value for money"""
        if self.comparison_matrix is None:
            self.generate_comparison_matrix()
        
        df = self.comparison_matrix.copy()
        
        # Filter by budget if specified
        if budget_range:
            df = df[(df['price'] >= budget_range[0]) & (df['price'] <= budget_range[1])]
        
        # Calculate value score
        df['value_score'] = (
            df['area_sqft_normalized'] * 0.25 +
            df['amenities_score_normalized'] * 0.20 +
            df['maintenance_score_normalized'] * 0.15 +
            df['age_normalized'] * 0.15 +
            df['floor_normalized'] * 0.10 +
            df['space_efficiency_normalized'] * 0.10 +
            df['bathroom_ratio_normalized'] * 0.05
        )
        
        # Adjust for price (lower price = higher value)
        df['price_score'] = (
            (df['price'].max() - df['price']) / (df['price'].max() - df['price'].min())
        ) * 100 if df['price'].max() != df['price'].min() else 50
        
        df['final_value_score'] = (df['value_score'] * 0.7) + (df['price_score'] * 0.3)
        
        return df.sort_values('final_value_score', ascending=False)
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.properties:
            return None
        
        df = self.generate_comparison_matrix()
        
        report = {
            'total_properties': len(self.properties),
            'price_range': {
                'min': df['price'].min(),
                'max': df['price'].max(),
                'avg': df['price'].mean()
            },
            'area_range': {
                'min': df['area_sqft'].min(),
                'max': df['area_sqft'].max(),
                'avg': df['area_sqft'].mean()
            },
            'best_value_properties': self.get_best_value_properties().head(3),
            'highest_scoring_properties': df.nlargest(3, 'area_sqft_normalized'),
            'newest_properties': df.nlargest(3, 'age_normalized'),
            'most_amenities': df.nlargest(3, 'amenities_score_normalized'),
            'location_distribution': df['location'].value_counts().to_dict(),
            'property_type_distribution': df['property_type'].value_counts().to_dict()
        }
        
        return report

def create_property_comparison_interface():
    """Create the property comparison tool interface"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
        <h2 style="color: white; text-align: center; margin-bottom: 1rem;">
            🏠 Property Comparison Tool
        </h2>
        <p style="color: rgba(255,255,255,0.9); text-align: center; font-size: 1.1rem;">
            Compare properties side-by-side with intelligent scoring and recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize comparison tool in session state
    if 'comparison_tool' not in st.session_state:
        st.session_state.comparison_tool = PropertyComparisonTool()
        
        # Add sample properties for demonstration
        sample_properties = [
            {
                'name': 'Luxury Sea View Apartment',
                'location': 'Bandra West',
                'area_sqft': 1500,
                'bedrooms': 3,
                'bathrooms': 3,
                'floor': 15,
                'age': 2,
                'price': 45000000,
                'furnishing': 'Fully furnished',
                'parking': 2,
                'balcony': 2,
                'amenities_score': 9,
                'maintenance_score': 8,
                'property_type': 'Apartment',
                'description': 'Stunning sea view apartment with premium amenities'
            },
            {
                'name': 'Cozy Family Home',
                'location': 'Andheri West',
                'area_sqft': 1200,
                'bedrooms': 2,
                'bathrooms': 2,
                'floor': 8,
                'age': 5,
                'price': 28000000,
                'furnishing': 'Semi-furnished',
                'parking': 1,
                'balcony': 1,
                'amenities_score': 7,
                'maintenance_score': 8,
                'property_type': 'Apartment',
                'description': 'Perfect family home in a prime location'
            },
            {
                'name': 'Modern Penthouse',
                'location': 'Juhu',
                'area_sqft': 2500,
                'bedrooms': 4,
                'bathrooms': 4,
                'floor': 20,
                'age': 1,
                'price': 85000000,
                'furnishing': 'Fully furnished',
                'parking': 3,
                'balcony': 3,
                'amenities_score': 10,
                'maintenance_score': 9,
                'property_type': 'Penthouse',
                'description': 'Ultra-luxury penthouse with panoramic city views'
            }
        ]
        
        for prop in sample_properties:
            st.session_state.comparison_tool.add_property(prop)
    
    comparison_tool = st.session_state.comparison_tool
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Add Properties", "📊 Compare Properties", "⭐ Best Value", "🎯 Similar Properties", "📋 Reports"
    ])
    
    with tab1:
        st.markdown("### Add New Property for Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            property_name = st.text_input("Property Name", placeholder="Enter property name")
            location = st.selectbox("Location", 
                ["Bandra West", "Andheri West", "Juhu", "Goregaon East", "Malad West", 
                 "Kandivali West", "Borivali West", "Powai", "Vile Parle", "Santacruz"])
            area_sqft = st.number_input("Area (sqft)", min_value=100, max_value=10000, value=1000)
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
            floor = st.number_input("Floor", min_value=1, max_value=50, value=5)
            
        with col2:
            age = st.number_input("Age (years)", min_value=0, max_value=50, value=5)
            price = st.number_input("Price (₹)", min_value=1000000, max_value=1000000000, value=30000000, step=1000000)
            furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Fully furnished"])
            parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)
            balcony = st.number_input("Balconies", min_value=0, max_value=5, value=1)
            amenities_score = st.slider("Amenities Score", min_value=1, max_value=10, value=5)
            maintenance_score = st.slider("Maintenance Score", min_value=1, max_value=10, value=5)
        
        property_type = st.selectbox("Property Type", 
            ["Apartment", "Villa", "Row House", "Studio", "Penthouse", "Duplex"])
        
        description = st.text_area("Description", placeholder="Enter property description")
        
        if st.button("➕ Add Property", type="primary"):
            new_property = {
                'name': property_name,
                'location': location,
                'area_sqft': area_sqft,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'floor': floor,
                'age': age,
                'price': price,
                'furnishing': furnishing,
                'parking': parking,
                'balcony': balcony,
                'amenities_score': amenities_score,
                'maintenance_score': maintenance_score,
                'property_type': property_type,
                'description': description
            }
            
            comparison_tool.add_property(new_property)
            st.success(f"✅ Property '{property_name}' added successfully!")
            st.balloons()
    
    with tab2:
        st.markdown("### Side-by-Side Property Comparison")
        
        if len(comparison_tool.properties) < 2:
            st.warning("⚠️ Please add at least 2 properties to compare")
        else:
            # Property selection
            col1, col2 = st.columns(2)
            
            with col1:
                property1_idx = st.selectbox(
                    "Select First Property",
                    range(len(comparison_tool.properties)),
                    format_func=lambda x: comparison_tool.properties[x]['name']
                )
            
            with col2:
                property2_idx = st.selectbox(
                    "Select Second Property",
                    range(len(comparison_tool.properties)),
                    format_func=lambda x: comparison_tool.properties[x]['name'],
                    index=1 if len(comparison_tool.properties) > 1 else 0
                )
            
            if property1_idx != property2_idx:
                # Generate comparison matrix
                comparison_df = comparison_tool.generate_comparison_matrix()
                
                if comparison_df is not None:
                    # Get selected properties
                    prop1 = comparison_df.iloc[property1_idx]
                    prop2 = comparison_df.iloc[property2_idx]
                    
                    # Display comparison
                    st.markdown("#### 📊 Detailed Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{prop1['name']}**")
                        st.write(f"📍 Location: {prop1['location']}")
                        st.write(f"🏠 Type: {prop1['property_type']}")
                        st.write(f"📐 Area: {prop1['area_sqft']:,} sqft")
                        st.write(f"🛏️ Bedrooms: {prop1['bedrooms']}")
                        st.write(f"🚿 Bathrooms: {prop1['bathrooms']}")
                        st.write(f"🏢 Floor: {prop1['floor']}")
                        st.write(f"🏗️ Age: {prop1['age']} years")
                        st.write(f"💰 Price: ₹{prop1['price']:,}")
                        st.write(f"💳 Price/sqft: ₹{prop1['price_per_sqft']:,.0f}")
                        st.write(f"🪑 Furnishing: {prop1['furnishing']}")
                        st.write(f"🚗 Parking: {prop1['parking']} spaces")
                        st.write(f"🏞️ Balconies: {prop1['balcony']}")
                        st.write(f"⭐ Amenities: {prop1['amenities_score']}/10")
                        st.write(f"🔧 Maintenance: {prop1['maintenance_score']}/10")
                    
                    with col2:
                        st.markdown(f"**{prop2['name']}**")
                        st.write(f"📍 Location: {prop2['location']}")
                        st.write(f"🏠 Type: {prop2['property_type']}")
                        st.write(f"📐 Area: {prop2['area_sqft']:,} sqft")
                        st.write(f"🛏️ Bedrooms: {prop2['bedrooms']}")
                        st.write(f"🚿 Bathrooms: {prop2['bathrooms']}")
                        st.write(f"🏢 Floor: {prop2['floor']}")
                        st.write(f"🏗️ Age: {prop2['age']} years")
                        st.write(f"💰 Price: ₹{prop2['price']:,}")
                        st.write(f"💳 Price/sqft: ₹{prop2['price_per_sqft']:,.0f}")
                        st.write(f"🪑 Furnishing: {prop2['furnishing']}")
                        st.write(f"🚗 Parking: {prop2['parking']} spaces")
                        st.write(f"🏞️ Balconies: {prop2['balcony']}")
                        st.write(f"⭐ Amenities: {prop2['amenities_score']}/10")
                        st.write(f"🔧 Maintenance: {prop2['maintenance_score']}/10")
                    
                    # Radar chart comparison
                    st.markdown("#### 📈 Feature Comparison Radar Chart")
                    
                    categories = ['Area', 'Price/sqft', 'Amenities', 'Maintenance', 'Age', 'Floor']
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[
                            prop1['area_sqft_normalized'],
                            prop1['price_per_sqft_normalized'],
                            prop1['amenities_score_normalized'],
                            prop1['maintenance_score_normalized'],
                            prop1['age_normalized'],
                            prop1['floor_normalized']
                        ],
                        theta=categories,
                        fill='toself',
                        name=prop1['name']
                    ))
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[
                            prop2['area_sqft_normalized'],
                            prop2['price_per_sqft_normalized'],
                            prop2['amenities_score_normalized'],
                            prop2['maintenance_score_normalized'],
                            prop2['age_normalized'],
                            prop2['floor_normalized']
                        ],
                        theta=categories,
                        fill='toself',
                        name=prop2['name']
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=True,
                        title="Property Feature Comparison"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendation
                    st.markdown("#### 🎯 Recommendation")
                    
                    # Calculate overall scores
                    prop1_score = (
                        prop1['area_sqft_normalized'] * 0.3 +
                        prop1['amenities_score_normalized'] * 0.2 +
                        prop1['maintenance_score_normalized'] * 0.2 +
                        prop1['age_normalized'] * 0.15 +
                        prop1['floor_normalized'] * 0.15
                    )
                    
                    prop2_score = (
                        prop2['area_sqft_normalized'] * 0.3 +
                        prop2['amenities_score_normalized'] * 0.2 +
                        prop2['maintenance_score_normalized'] * 0.2 +
                        prop2['age_normalized'] * 0.15 +
                        prop2['floor_normalized'] * 0.15
                    )
                    
                    if prop1_score > prop2_score:
                        st.success(f"🏆 **{prop1['name']}** is recommended based on overall features")
                        st.info(f"Score: {prop1_score:.1f} vs {prop2_score:.1f}")
                    else:
                        st.success(f"🏆 **{prop2['name']}** is recommended based on overall features")
                        st.info(f"Score: {prop2_score:.1f} vs {prop1_score:.1f}")
    
    with tab3:
        st.markdown("### Best Value Properties")
        
        budget_range = st.slider(
            "Budget Range (₹)",
            min_value=1000000,
            max_value=100000000,
            value=(20000000, 60000000),
            step=1000000
        )
        
        best_value_props = comparison_tool.get_best_value_properties(budget_range)
        
        if best_value_props is not None and len(best_value_props) > 0:
            st.markdown("#### 🏆 Top Value Properties")
            
            for idx, (_, prop) in enumerate(best_value_props.head(3).iterrows()):
                with st.expander(f"#{idx+1} {prop['name']} - Score: {prop['final_value_score']:.1f}/100"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"📍 **Location:** {prop['location']}")
                        st.write(f"📐 **Area:** {prop['area_sqft']:,} sqft")
                        st.write(f"💰 **Price:** ₹{prop['price']:,}")
                        st.write(f"💳 **Price/sqft:** ₹{prop['price_per_sqft']:,.0f}")
                        st.write(f"⭐ **Value Score:** {prop['final_value_score']:.1f}/100")
                    
                    with col2:
                        st.write(f"🛏️ **Bedrooms:** {prop['bedrooms']}")
                        st.write(f"🚿 **Bathrooms:** {prop['bathrooms']}")
                        st.write(f"🏢 **Floor:** {prop['floor']}")
                        st.write(f"🏗️ **Age:** {prop['age']} years")
                        st.write(f"⭐ **Amenities:** {prop['amenities_score']}/10")
        else:
            st.warning("No properties found in the selected budget range")
    
    with tab4:
        st.markdown("### Find Similar Properties")
        
        if len(comparison_tool.properties) < 2:
            st.warning("⚠️ Please add at least 2 properties to find similar ones")
        else:
            reference_idx = st.selectbox(
                "Select Reference Property",
                range(len(comparison_tool.properties)),
                format_func=lambda x: comparison_tool.properties[x]['name']
            )
            
            similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            
            similarities = comparison_tool.calculate_similarity_scores(reference_idx)
            
            if similarities is not None:
                # Create similarity dataframe
                similarity_df = pd.DataFrame({
                    'Property': [prop['name'] for prop in comparison_tool.properties],
                    'Similarity': similarities,
                    'Location': [prop['location'] for prop in comparison_tool.properties],
                    'Price': [prop['price'] for prop in comparison_tool.properties],
                    'Area': [prop['area_sqft'] for prop in comparison_tool.properties]
                })
                
                # Filter similar properties
                similar_props = similarity_df[
                    (similarity_df['Similarity'] >= similarity_threshold) & 
                    (similarity_df.index != reference_idx)
                ].sort_values('Similarity', ascending=False)
                
                if len(similar_props) > 0:
                    st.markdown(f"#### 🔍 Properties Similar to {comparison_tool.properties[reference_idx]['name']}")
                    
                    for idx, (_, prop) in enumerate(similar_props.iterrows()):
                        similarity_pct = prop['Similarity'] * 100
                        st.info(f"**{prop['Property']}** - Similarity: {similarity_pct:.1f}%")
                        st.write(f"📍 Location: {prop['Location']} | 💰 Price: ₹{prop['Price']:,} | 📐 Area: {prop['Area']:,} sqft")
                        st.write("---")
                else:
                    st.warning("No similar properties found with the current threshold")
    
    with tab5:
        st.markdown("### Comparison Reports")
        
        report = comparison_tool.generate_comparison_report()
        
        if report:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Properties", report['total_properties'])
                st.metric("Price Range", f"₹{report['price_range']['min']/1000000:.1f}M - ₹{report['price_range']['max']/10000000:.1f}Cr")
            
            with col2:
                st.metric("Avg Area", f"{report['area_range']['avg']:.0f} sqft")
                st.metric("Area Range", f"{report['area_range']['min']:.0f} - {report['area_range']['max']:.0f} sqft")
            
            with col3:
                most_common_location = max(report['location_distribution'], key=report['location_distribution'].get)
                most_common_type = max(report['property_type_distribution'], key=report['property_type_distribution'].get)
                st.metric("Most Common Location", most_common_location)
                st.metric("Most Common Type", most_common_type)
            
            # Location distribution chart
            st.markdown("#### 📍 Location Distribution")
            location_data = pd.DataFrame(list(report['location_distribution'].items()), columns=['Location', 'Count'])
            fig = px.bar(location_data, x='Location', y='Count', title='Properties by Location')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Property type distribution
            st.markdown("#### 🏠 Property Type Distribution")
            type_data = pd.DataFrame(list(report['property_type_distribution'].items()), columns=['Type', 'Count'])
            fig = px.pie(type_data, values='Count', names='Type', title='Properties by Type')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Export report
            if st.button("📊 Export Report"):
                report_json = json.dumps(report, indent=2, default=str)
                st.download_button(
                    label="Download Report (JSON)",
                    data=report_json,
                    file_name=f"property_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Create the comparison tool
if __name__ == "__main__":
    create_property_comparison_interface()