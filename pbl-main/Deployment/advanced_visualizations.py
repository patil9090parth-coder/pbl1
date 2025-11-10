import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from plotly.offline import iplot
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import networkx as nx
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualizationSuite:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.visualization_configs = {}
        
    def load_sample_data(self):
        """Load comprehensive sample data for visualizations"""
        np.random.seed(42)
        
        # Generate comprehensive real estate data
        locations = ['Bandra West', 'Andheri West', 'Juhu', 'Powai', 'Goregaon East', 
                    'Malad West', 'Kandivali West', 'Borivali West', 'Vile Parle', 'Santacruz']
        
        property_types = ['Apartment', 'Villa', 'Row House', 'Studio', 'Penthouse', 'Duplex']
        furnishing_types = ['Unfurnished', 'Semi-furnished', 'Fully furnished']
        
        # Generate 2000 properties
        n_properties = 2000
        data = []
        
        for i in range(n_properties):
            location = np.random.choice(locations)
            property_type = np.random.choice(property_types)
            furnishing = np.random.choice(furnishing_types)
            
            # Location-based price multipliers
            location_multiplier = {
                'Bandra West': 1.4, 'Juhu': 1.3, 'Powai': 1.2,
                'Andheri West': 1.1, 'Vile Parle': 1.1, 'Santacruz': 1.0,
                'Goregaon East': 0.9, 'Malad West': 0.85, 'Kandivali West': 0.8,
                'Borivali West': 0.75
            }
            
            area_sqft = np.random.randint(400, 3000)
            base_price_per_sqft = np.random.randint(8000, 20000)
            adjusted_price_per_sqft = base_price_per_sqft * location_multiplier.get(location, 1.0)
            
            # Add some correlation between features
            bedrooms = max(1, min(6, int(area_sqft / 400) + np.random.randint(-1, 2)))
            bathrooms = max(1, min(5, bedrooms + np.random.randint(0, 2)))
            
            data.append({
                'property_id': f'PROP_{i+1:04d}',
                'location': location,
                'property_type': property_type,
                'area_sqft': area_sqft,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'floor': np.random.randint(1, 25),
                'age': np.random.randint(0, 30),
                'price_per_sqft': adjusted_price_per_sqft,
                'total_price': area_sqft * adjusted_price_per_sqft,
                'furnishing': furnishing,
                'parking_spaces': np.random.randint(0, 3),
                'balconies': np.random.randint(0, 4),
                'amenities_score': np.random.randint(1, 11),
                'maintenance_score': np.random.randint(1, 11),
                'latitude': self._get_location_coordinates(location)[0] + np.random.normal(0, 0.01),
                'longitude': self._get_location_coordinates(location)[1] + np.random.normal(0, 0.01),
                'listing_date': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                'seller_type': np.random.choice(['Individual', 'Builder', 'Agent']),
                'negotiable': np.random.choice([True, False]),
                'loan_available': np.random.choice([True, False])
            })
        
        self.data = pd.DataFrame(data)
        self.process_data()
        return self.data
    
    def _get_location_coordinates(self, location):
        """Get approximate coordinates for Mumbai locations"""
        coordinates = {
            'Bandra West': (19.0596, 72.8295),
            'Andheri West': (19.1136, 72.8697),
            'Juhu': (19.1075, 72.8258),
            'Powai': (19.1197, 72.9053),
            'Goregaon East': (19.1553, 72.8826),
            'Malad West': (19.1863, 72.8279),
            'Kandivali West': (19.2035, 72.8434),
            'Borivali West': (19.2344, 72.8388),
            'Vile Parle': (19.0896, 72.8656),
            'Santacruz': (19.0823, 72.8511)
        }
        return coordinates.get(location, (19.0760, 72.8777))  # Default to Mumbai center
    
    def process_data(self):
        """Process and enrich data for visualizations"""
        if self.data is None:
            return
        
        df = self.data.copy()
        
        # Add derived features
        df['price_category'] = pd.cut(df['total_price'], 
                                     bins=[0, 15000000, 30000000, 60000000, 120000000, float('inf')],
                                     labels=['Budget', 'Mid-Range', 'Premium', 'Luxury', 'Ultra-Luxury'])
        
        df['area_category'] = pd.cut(df['area_sqft'],
                                      bins=[0, 600, 1200, 1800, 3000, float('inf')],
                                      labels=['Compact', 'Medium', 'Large', 'Extra Large', 'Mansion'])
        
        df['age_category'] = pd.cut(df['age'],
                                     bins=[0, 5, 10, 20, 50, float('inf')],
                                     labels=['New', 'Recent', 'Mature', 'Old', 'Heritage'])
        
        df['floor_category'] = pd.cut(df['floor'],
                                       bins=[0, 5, 10, 20, 50, float('inf')],
                                       labels=['Low', 'Mid', 'High', 'Sky', 'Penthouse'])
        
        df['space_efficiency'] = df['area_sqft'] / df['bedrooms']
        df['price_efficiency'] = df['total_price'] / df['area_sqft']
        df['amenities_to_price_ratio'] = df['amenities_score'] / (df['total_price'] / 1000000)
        
        self.processed_data = df
        return df
    
    def create_3d_scatter_plot(self):
        """Create 3D scatter plot with multiple dimensions"""
        if self.processed_data is None:
            return None
        
        df = self.processed_data
        
        fig = px.scatter_3d(
            df,
            x='area_sqft',
            y='price_per_sqft',
            z='total_price',
            color='location',
            size='amenities_score',
            hover_data=['property_type', 'bedrooms', 'bathrooms', 'age'],
            title='3D Property Analysis: Area vs Price per sqft vs Total Price'
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Area (sqft)',
                yaxis_title='Price per sqft (₹)',
                zaxis_title='Total Price (₹)'
            ),
            height=700
        )
        
        return fig
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap for numerical features"""
        if self.processed_data is None:
            return None
        
        # Select numerical columns
        numerical_cols = ['area_sqft', 'bedrooms', 'bathrooms', 'floor', 'age', 
                         'price_per_sqft', 'total_price', 'amenities_score', 
                         'maintenance_score', 'space_efficiency']
        
        correlation_matrix = self.processed_data[numerical_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            labels=dict(color="Correlation"),
            x=numerical_cols,
            y=numerical_cols,
            color_continuous_scale='RdBu',
            aspect="auto",
            title='Feature Correlation Heatmap'
        )
        
        fig.update_layout(height=600)
        
        return fig
    
    def create_parallel_coordinates_plot(self):
        """Create parallel coordinates plot for multi-dimensional analysis"""
        if self.processed_data is None:
            return None
        
        df = self.processed_data.copy()
        
        # Normalize numerical features for better visualization
        numerical_features = ['area_sqft', 'total_price', 'amenities_score', 'age', 'floor']
        
        # Create normalized dataset
        normalized_df = df.copy()
        for col in numerical_features:
            normalized_df[f'{col}_norm'] = (
                (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            ) * 100
        
        fig = px.parallel_coordinates(
            normalized_df,
            dimensions=[f'{col}_norm' for col in numerical_features],
            color='total_price',
            labels={f'{col}_norm': col.replace('_', ' ').title() for col in numerical_features},
            color_continuous_scale='viridis',
            title='Multi-dimensional Property Analysis'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_clustering_visualization(self):
        """Create clustering visualization using PCA and K-means"""
        if self.processed_data is None:
            return None
        
        df = self.processed_data.copy()
        
        # Select features for clustering
        features = ['area_sqft', 'total_price', 'amenities_score', 'age', 'floor', 'bedrooms']
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(scaled_features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Add to dataframe
        df['pca_1'] = pca_features[:, 0]
        df['pca_2'] = pca_features[:, 1]
        df['cluster'] = clusters
        
        fig = px.scatter(
            df,
            x='pca_1',
            y='pca_2',
            color='cluster',
            size='total_price',
            hover_data=['location', 'property_type', 'area_sqft', 'bedrooms'],
            title='Property Clustering Analysis (PCA + K-means)'
        )
        
        fig.update_layout(height=600)
        
        return fig
    
    def create_anomaly_detection_plot(self):
        """Create anomaly detection visualization"""
        if self.processed_data is None:
            return None
        
        df = self.processed_data.copy()
        
        # Select features for anomaly detection
        features = ['area_sqft', 'total_price', 'price_per_sqft', 'amenities_score']
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])
        
        # Perform anomaly detection
        isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        anomalies = isolation_forest.fit_predict(scaled_features)
        
        df['is_anomaly'] = anomalies == -1
        
        fig = px.scatter(
            df,
            x='area_sqft',
            y='total_price',
            color='is_anomaly',
            size='amenities_score',
            hover_data=['location', 'property_type', 'age', 'floor'],
            title='Property Anomaly Detection',
            color_discrete_map={True: 'red', False: 'blue'}
        )
        
        fig.update_layout(height=600)
        
        return fig
    
    def create_geographical_visualization(self):
        """Create geographical visualization with Folium"""
        if self.processed_data is None:
            return None
        
        df = self.processed_data.copy()
        
        # Create base map
        m = folium.Map(location=[19.0760, 72.8777], zoom_start=11)
        
        # Create color map based on price
        price_colors = ['green', 'yellow', 'orange', 'red', 'purple']
        price_ranges = pd.qcut(df['total_price'], q=5, labels=price_colors)
        
        # Add markers for each property
        for idx, row in df.iterrows():
            color = price_ranges.iloc[idx] if pd.notna(price_ranges.iloc[idx]) else 'blue'
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=f"""
                <b>{row['property_type']}</b><br>
                Location: {row['location']}<br>
                Area: {row['area_sqft']} sqft<br>
                Price: ₹{row['total_price']:,}<br>
                Bedrooms: {row['bedrooms']}<br>
                Age: {row['age']} years
                """,
                color=color,
                fill=True,
                fillColor=color
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white;">
        <p><i class="fa fa-circle" style="color:green"></i> Budget</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Mid-Range</p>
        <p><i class="fa fa-circle" style="color:red"></i> Premium</p>
        <p><i class="fa fa-circle" style="color:purple"></i> Luxury</p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def create_statistical_distribution_plots(self):
        """Create statistical distribution plots"""
        if self.processed_data is None:
            return None
        
        df = self.processed_data.copy()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Distribution', 'Area Distribution', 
                          'Price per sqft Distribution', 'Age Distribution'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Price distribution
        fig.add_trace(
            go.Histogram(x=df['total_price'], name='Price', nbinsx=50, opacity=0.7),
            row=1, col=1
        )
        
        # Add normal distribution overlay
        price_mean, price_std = df['total_price'].mean(), df['total_price'].std()
        x_price = np.linspace(df['total_price'].min(), df['total_price'].max(), 100)
        y_price = stats.norm.pdf(x_price, price_mean, price_std) * len(df) * (df['total_price'].max() - df['total_price'].min()) / 50
        
        fig.add_trace(
            go.Scatter(x=x_price, y=y_price, mode='lines', name='Price Normal', line=dict(color='red')),
            row=1, col=1
        )
        
        # Area distribution
        fig.add_trace(
            go.Histogram(x=df['area_sqft'], name='Area', nbinsx=50, opacity=0.7),
            row=1, col=2
        )
        
        # Price per sqft distribution
        fig.add_trace(
            go.Histogram(x=df['price_per_sqft'], name='Price/sqft', nbinsx=50, opacity=0.7),
            row=2, col=1
        )
        
        # Age distribution
        fig.add_trace(
            go.Histogram(x=df['age'], name='Age', nbinsx=30, opacity=0.7),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        
        return fig
    
    def create_time_series_analysis(self):
        """Create time series analysis plots"""
        if self.processed_data is None:
            return None
        
        df = self.processed_data.copy()
        
        # Group by listing date (simulate monthly data)
        df['listing_month'] = df['listing_date'].dt.to_period('M')
        monthly_data = df.groupby('listing_month').agg({
            'total_price': ['mean', 'median', 'count'],
            'area_sqft': 'mean',
            'amenities_score': 'mean'
        }).round(2)
        
        monthly_data.columns = ['avg_price', 'median_price', 'property_count', 'avg_area', 'avg_amenities']
        monthly_data.index = monthly_data.index.to_timestamp()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Price Trend', 'Property Count Trend', 
                          'Median Price Trend', 'Average Area Trend'),
            vertical_spacing=0.1
        )
        
        # Average price trend
        fig.add_trace(
            go.Scatter(x=monthly_data.index, y=monthly_data['avg_price'], 
                      mode='lines+markers', name='Avg Price'),
            row=1, col=1
        )
        
        # Property count trend
        fig.add_trace(
            go.Scatter(x=monthly_data.index, y=monthly_data['property_count'], 
                      mode='lines+markers', name='Property Count'),
            row=1, col=2
        )
        
        # Median price trend
        fig.add_trace(
            go.Scatter(x=monthly_data.index, y=monthly_data['median_price'], 
                      mode='lines+markers', name='Median Price'),
            row=2, col=1
        )
        
        # Average area trend
        fig.add_trace(
            go.Scatter(x=monthly_data.index, y=monthly_data['avg_area'], 
                      mode='lines+markers', name='Avg Area'),
            row=2, col=2
        )
        
        fig.update_layout(height=700, showlegend=False)
        
        return fig
    
    def create_network_analysis(self):
        """Create network analysis visualization"""
        if self.processed_data is None:
            return None
        
        df = self.processed_data.copy()
        
        # Create network based on location similarities
        G = nx.Graph()
        
        # Add nodes (properties)
        for idx, row in df.iterrows():
            G.add_node(idx, 
                      location=row['location'],
                      price=row['total_price'],
                      area=row['area_sqft'],
                      type=row['property_type'])
        
        # Add edges based on similarity (same location and similar price)
        for i, row1 in df.iterrows():
            for j, row2 in df.iterrows():
                if i < j:  # Avoid duplicate edges
                    if (row1['location'] == row2['location'] and 
                        abs(row1['total_price'] - row2['total_price']) < 5000000):  # Within 50L price difference
                        G.add_edge(i, j)
        
        # Calculate network metrics
        centrality = nx.degree_centrality(G)
        
        # Create visualization
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Create edges trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create nodes trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_info = df.loc[node]
            node_text.append(f"Location: {node_info['location']}<br>Price: ₹{node_info['total_price']:,}<br>Area: {node_info['area_sqft']} sqft")
            node_color.append(centrality[node])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[str(node) for node in G.nodes()],
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Node Centrality',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        
        node_trace.text = node_text
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Property Network Analysis (Location & Price Similarity)',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Network of properties connected by location and price similarity",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002 )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig

def create_advanced_visualizations_dashboard():
    """Create the advanced visualizations dashboard"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
        <h2 style="color: white; text-align: center; margin-bottom: 1rem;">
            📊 Advanced Data Visualization Suite
        </h2>
        <p style="color: rgba(255,255,255,0.9); text-align: center; font-size: 1.1rem;">
            Interactive visualizations for deep data analysis and insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize visualization suite
    @st.cache_resource
    def load_visualization_suite():
        suite = AdvancedVisualizationSuite()
        suite.load_sample_data()
        return suite
    
    suite = load_visualization_suite()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "3D Analysis", "Correlations", "Clustering", "Anomalies", "Statistics", "Time Series", "Network Analysis"
    ])
    
    with tab1:
        st.markdown("### 3D Property Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig_3d = suite.create_3d_scatter_plot()
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            st.markdown("#### Controls")
            st.info("Use mouse to rotate, zoom, and pan the 3D visualization")
            
            if st.checkbox("Show location labels"):
                st.write("Location distribution in the dataset:")
                location_counts = suite.processed_data['location'].value_counts()
                st.dataframe(location_counts)
    
    with tab2:
        st.markdown("### Feature Correlation Analysis")
        
        fig_corr = suite.create_correlation_heatmap()
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Additional correlation insights
        st.markdown("#### Key Correlations")
        
        corr_matrix = suite.processed_data[['area_sqft', 'total_price', 'price_per_sqft', 'amenities_score', 'age']].corr()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strongest_corr = corr_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates()
            strongest_corr = strongest_corr[strongest_corr < 1.0].head(1)
            st.metric("Strongest Correlation", f"{strongest_corr.iloc[0]:.3f}")
        
        with col2:
            price_area_corr = corr_matrix.loc['area_sqft', 'total_price']
            st.metric("Price-Area Correlation", f"{price_area_corr:.3f}")
        
        with col3:
            price_amenity_corr = corr_matrix.loc['amenities_score', 'total_price']
            st.metric("Price-Amenity Correlation", f"{price_amenity_corr:.3f}")
    
    with tab3:
        st.markdown("### Property Clustering Analysis")
        
        fig_cluster = suite.create_clustering_visualization()
        if fig_cluster:
            st.plotly_chart(fig_cluster, use_container_width=True)
        
        st.markdown("#### Clustering Insights")
        
        # Perform clustering analysis
        features = ['area_sqft', 'total_price', 'amenities_score', 'age', 'floor', 'bedrooms']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(suite.processed_data[features])
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        cluster_analysis = pd.DataFrame({
            'Cluster': range(5),
            'Count': pd.Series(clusters).value_counts().sort_index(),
            'Avg_Price': [suite.processed_data[clusters == i]['total_price'].mean() for i in range(5)],
            'Avg_Area': [suite.processed_data[clusters == i]['area_sqft'].mean() for i in range(5)]
        })
        
        st.dataframe(cluster_analysis)
    
    with tab4:
        st.markdown("### Anomaly Detection")
        
        fig_anomaly = suite.create_anomaly_detection_plot()
        if fig_anomaly:
            st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Show anomaly statistics
        features = ['area_sqft', 'total_price', 'price_per_sqft', 'amenities_score']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(suite.processed_data[features])
        
        isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        anomalies = isolation_forest.fit_predict(scaled_features)
        
        anomaly_count = sum(anomalies == -1)
        anomaly_percentage = (anomaly_count / len(anomalies)) * 100
        
        st.warning(f"Detected {anomaly_count} anomalies ({anomaly_percentage:.1f}% of data)")
        
        if st.checkbox("Show anomaly details"):
            anomaly_data = suite.processed_data[anomalies == -1]
            st.dataframe(anomaly_data[['location', 'property_type', 'total_price', 'area_sqft', 'price_per_sqft']].head(10))
    
    with tab5:
        st.markdown("### Statistical Distribution Analysis")
        
        fig_stats = suite.create_statistical_distribution_plots()
        if fig_stats:
            st.plotly_chart(fig_stats, use_container_width=True)
        
        # Statistical summary
        st.markdown("#### Statistical Summary")
        
        stats_summary = suite.processed_data[['total_price', 'area_sqft', 'price_per_sqft', 'age']].describe()
        st.dataframe(stats_summary)
        
        # Normality tests
        st.markdown("#### Normality Tests (Shapiro-Wilk)")
        
        for col in ['total_price', 'area_sqft', 'price_per_sqft']:
            # Sample data for normality test (Shapiro-Wilk requires < 5000 samples)
            sample_data = suite.processed_data[col].sample(min(1000, len(suite.processed_data)), random_state=42)
            stat, p_value = stats.shapiro(sample_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**{col}:**")
            with col2:
                st.metric("Test Statistic", f"{stat:.4f}")
            with col3:
                distribution = "Normal" if p_value > 0.05 else "Not Normal"
                st.metric("Distribution", distribution)
    
    with tab6:
        st.markdown("### Time Series Analysis")
        
        fig_time = suite.create_time_series_analysis()
        if fig_time:
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Trend analysis
        st.markdown("#### Trend Analysis")
        
        monthly_data = suite.processed_data.groupby(
            suite.processed_data['listing_date'].dt.to_period('M')
        ).agg({
            'total_price': ['mean', 'median', 'count'],
            'area_sqft': 'mean',
            'amenities_score': 'mean'
        })
        
        monthly_data.columns = ['avg_price', 'median_price', 'property_count', 'avg_area', 'avg_amenities']
        monthly_data.index = monthly_data.index.to_timestamp()
        
        # Calculate trends
        price_trend = np.polyfit(range(len(monthly_data)), monthly_data['avg_price'], 1)[0]
        count_trend = np.polyfit(range(len(monthly_data)), monthly_data['property_count'], 1)[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Price Trend", f"₹{price_trend:,.0f}/month")
        
        with col2:
            st.metric("Listing Count Trend", f"{count_trend:.1f} properties/month")
    
    with tab7:
        st.markdown("### Network Analysis")
        
        fig_network = suite.create_network_analysis()
        if fig_network:
            st.plotly_chart(fig_network, use_container_width=True)
        
        st.markdown("#### Network Properties")
        
        # Calculate network statistics
        df = suite.processed_data.copy()
        G = nx.Graph()
        
        # Add nodes and edges
        for idx, row in df.iterrows():
            G.add_node(idx, location=row['location'], price=row['total_price'])
        
        for i, row1 in df.iterrows():
            for j, row2 in df.iterrows():
                if i < j and row1['location'] == row2['location'] and abs(row1['total_price'] - row2['total_price']) < 5000000:
                    G.add_edge(i, j)
        
        if G.number_of_nodes() > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Network Nodes", G.number_of_nodes())
            
            with col2:
                st.metric("Network Edges", G.number_of_edges())
            
            with col3:
                avg_clustering = nx.average_clustering(G) if G.number_of_nodes() > 2 else 0
                st.metric("Clustering Coefficient", f"{avg_clustering:.3f}")
    
    # Export functionality
    st.markdown("### 💾 Export Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Data Summary"):
            summary = {
                'total_properties': len(suite.processed_data),
                'locations': suite.processed_data['location'].nunique(),
                'property_types': suite.processed_data['property_type'].nunique(),
                'price_range': {
                    'min': suite.processed_data['total_price'].min(),
                    'max': suite.processed_data['total_price'].max(),
                    'avg': suite.processed_data['total_price'].mean()
                },
                'area_range': {
                    'min': suite.processed_data['area_sqft'].min(),
                    'max': suite.processed_data['area_sqft'].max(),
                    'avg': suite.processed_data['area_sqft'].mean()
                }
            }
            
            summary_json = json.dumps(summary, indent=2, default=str)
            st.download_button(
                label="Download Summary (JSON)",
                data=summary_json,
                file_name=f"data_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📈 Generate Report"):
            st.info("Report generation feature coming soon!")

# Create the dashboard
if __name__ == "__main__":
    create_advanced_visualizations_dashboard()