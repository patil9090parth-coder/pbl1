import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import base64
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class MobileOptimizationSuite:
    def __init__(self):
        self.device_type = self._detect_device_type()
        self.screen_size = self._get_screen_size()
        self.optimization_settings = self._get_optimization_settings()
    
    def _detect_device_type(self):
        """Detect device type based on user agent"""
        try:
            # This is a simplified detection - in production, you'd use JavaScript
            user_agent = st.runtime.get_instance().browser_server_address
            if 'mobile' in str(user_agent).lower():
                return 'mobile'
            elif 'tablet' in str(user_agent).lower():
                return 'tablet'
            else:
                return 'desktop'
        except:
            return 'desktop'
    
    def _get_screen_size(self):
        """Get estimated screen size"""
        # In a real implementation, you'd get this from JavaScript
        return {
            'width': st.session_state.get('screen_width', 1920),
            'height': st.session_state.get('screen_height', 1080)
        }
    
    def _get_optimization_settings(self):
        """Get optimization settings based on device type"""
        if self.device_type == 'mobile':
            return {
                'chart_height': 300,
                'font_size': 12,
                'margin': dict(l=20, r=20, t=30, b=20),
                'showlegend': False,
                'max_data_points': 100,
                'enable_lazy_loading': True,
                'compress_images': True,
                'reduce_animations': True
            }
        elif self.device_type == 'tablet':
            return {
                'chart_height': 400,
                'font_size': 14,
                'margin': dict(l=30, r=30, t=40, b=30),
                'showlegend': True,
                'max_data_points': 500,
                'enable_lazy_loading': True,
                'compress_images': True,
                'reduce_animations': False
            }
        else:
            return {
                'chart_height': 600,
                'font_size': 16,
                'margin': dict(l=50, r=50, t=60, b=50),
                'showlegend': True,
                'max_data_points': 2000,
                'enable_lazy_loading': False,
                'compress_images': False,
                'reduce_animations': False
            }
    
    def optimize_chart(self, fig, chart_type='scatter'):
        """Optimize chart for mobile viewing"""
        settings = self.optimization_settings
        
        # Update layout for mobile optimization
        fig.update_layout(
            height=settings['chart_height'],
            font=dict(size=settings['font_size']),
            margin=settings['margin'],
            showlegend=settings['showlegend']
        )
        
        # Reduce data points if necessary
        if hasattr(fig, 'data') and len(fig.data) > 0:
            for trace in fig.data:
                if hasattr(trace, 'x') and len(trace.x) > settings['max_data_points']:
                    # Sample data points
                    sample_indices = np.linspace(0, len(trace.x) - 1, settings['max_data_points'], dtype=int)
                    if hasattr(trace, 'x'):
                        trace.x = [trace.x[i] for i in sample_indices]
                    if hasattr(trace, 'y'):
                        trace.y = [trace.y[i] for i in sample_indices]
        
        # Reduce animations for mobile
        if settings['reduce_animations']:
            fig.update_layout(
                transition=dict(duration=0),
                uirevision='static'
            )
        
        return fig
    
    def create_mobile_responsive_chart(self, data, chart_type='line'):
        """Create a mobile-responsive chart from scratch"""
        settings = self.optimization_settings
        
        if chart_type == 'line':
            fig = go.Figure()
            
            if isinstance(data, dict):
                for name, values in data.items():
                    fig.add_trace(go.Scatter(
                        x=list(range(len(values))),
                        y=values,
                        mode='lines+markers',
                        name=name
                    ))
            
        elif chart_type == 'bar':
            fig = go.Figure()
            
            if isinstance(data, dict):
                for name, values in data.items():
                    fig.add_trace(go.Bar(
                        x=list(range(len(values))),
                        y=values,
                        name=name
                    ))
        
        elif chart_type == 'scatter':
            fig = go.Figure()
            
            if isinstance(data, dict):
                fig.add_trace(go.Scatter(
                    x=data.get('x', []),
                    y=data.get('y', []),
                    mode='markers',
                    marker=dict(size=8),
                    text=data.get('text', [])
                ))
        
        # Apply mobile optimization
        fig = self.optimize_chart(fig, chart_type)
        
        return fig
    
    def create_mobile_friendly_layout(self, content_type='dashboard'):
        """Create mobile-friendly layout components"""
        settings = self.optimization_settings
        
        if content_type == 'dashboard':
            # Mobile-friendly dashboard layout
            st.markdown("""
            <style>
            /* Mobile-first responsive design */
            .mobile-dashboard {
                padding: 10px;
                margin: 0 auto;
                max-width: 100%;
            }
            
            .mobile-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                color: white;
            }
            
            .mobile-metric {
                font-size: 24px;
                font-weight: bold;
                margin: 5px 0;
            }
            
            .mobile-label {
                font-size: 14px;
                opacity: 0.9;
            }
            
            @media (max-width: 768px) {
                .stTabs [data-baseweb="tab-list"] {
                    flex-direction: column;
                }
                
                .stTabs [data-baseweb="tab"] {
                    width: 100%;
                    margin: 2px 0;
                }
                
                .stColumn {
                    flex-direction: column;
                }
                
                .mobile-card {
                    padding: 10px;
                    margin: 5px 0;
                }
                
                .mobile-metric {
                    font-size: 20px;
                }
            }
            </style>
            """, unsafe_allow_html=True)
        
        elif content_type == 'form':
            # Mobile-friendly form layout
            st.markdown("""
            <style>
            .mobile-form {
                padding: 15px;
                margin: 0 auto;
                max-width: 100%;
            }
            
            .mobile-form .stTextInput > div > div > input {
                font-size: 16px; /* Prevents zoom on iOS */
                padding: 12px;
                border-radius: 8px;
            }
            
            .mobile-form .stNumberInput > div > div > input {
                font-size: 16px;
                padding: 12px;
                border-radius: 8px;
            }
            
            .mobile-form .stSelectbox > div > div > div {
                font-size: 16px;
                padding: 12px;
                border-radius: 8px;
            }
            
            .mobile-form .stButton > button {
                font-size: 16px;
                padding: 12px 24px;
                border-radius: 8px;
                width: 100%;
                margin: 5px 0;
            }
            
            @media (max-width: 768px) {
                .mobile-form {
                    padding: 10px;
                }
            }
            </style>
            """, unsafe_allow_html=True)
    
    def create_mobile_navigation(self):
        """Create mobile-optimized navigation"""
        st.markdown("""
        <style>
        /* Mobile navigation styles */
        .mobile-nav {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #fff;
            border-top: 1px solid #e0e0e0;
            padding: 10px 0;
            z-index: 999;
            display: none;
        }
        
        .mobile-nav-item {
            flex: 1;
            text-align: center;
            padding: 5px;
            cursor: pointer;
            font-size: 12px;
        }
        
        .mobile-nav-icon {
            font-size: 20px;
            display: block;
            margin-bottom: 2px;
        }
        
        @media (max-width: 768px) {
            .mobile-nav {
                display: flex;
            }
            
            /* Add padding to main content to account for bottom nav */
            .main .block-container {
                padding-bottom: 80px;
            }
        }
        </style>
        
        <div class="mobile-nav">
            <div class="mobile-nav-item" onclick="window.location.href='?page=home'">
                <span class="mobile-nav-icon">🏠</span>
                <div>Home</div>
            </div>
            <div class="mobile-nav-item" onclick="window.location.href='?page=predict'">
                <span class="mobile-nav-icon">🔮</span>
                <div>Predict</div>
            </div>
            <div class="mobile-nav-item" onclick="window.location.href='?page=compare'">
                <span class="mobile-nav-icon">⚖️</span>
                <div>Compare</div>
            </div>
            <div class="mobile-nav-item" onclick="window.location.href='?page=insights'">
                <span class="mobile-nav-icon">📊</span>
                <div>Insights</div>
            </div>
            <div class="mobile-nav-item" onclick="window.location.href='?page=about'">
                <span class="mobile-nav-icon">ℹ️</span>
                <div>About</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def optimize_data_loading(self, data_size_threshold=1000):
        """Optimize data loading for mobile devices"""
        settings = self.optimization_settings
        
        if settings['enable_lazy_loading']:
            # Implement lazy loading
            @st.cache_data(show_spinner=False)
            def load_data_chunk(chunk_size=data_size_threshold):
                # Simulate loading data in chunks
                return pd.DataFrame()  # Placeholder
            
            return load_data_chunk
        
        return None
    
    def create_touch_friendly_controls(self):
        """Create touch-friendly controls for mobile devices"""
        st.markdown("""
        <style>
        /* Touch-friendly controls */
        .touch-slider .stSlider > div > div > div {
            height: 8px;
        }
        
        .touch-slider .stSlider > div > div > div > div {
            width: 24px;
            height: 24px;
            border-radius: 50%;
        }
        
        .touch-button .stButton > button {
            min-height: 44px; /* iOS recommended touch target size */
            min-width: 44px;
            font-size: 16px;
            border-radius: 8px;
            margin: 5px;
        }
        
        .touch-checkbox .stCheckbox > div > div > div {
            width: 24px;
            height: 24px;
        }
        
        .touch-radio .stRadio > div > div > div {
            padding: 12px;
            margin: 5px 0;
        }
        
        /* Larger touch targets for mobile */
        @media (max-width: 768px) {
            .stSlider > div > div > div {
                height: 10px;
            }
            
            .stSlider > div > div > div > div {
                width: 28px;
                height: 28px;
            }
            
            .stButton > button {
                padding: 15px 20px;
                font-size: 18px;
            }
            
            .stSelectbox > div > div > div {
                padding: 15px;
                font-size: 16px;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def create_progressive_web_app_features(self):
        """Create Progressive Web App (PWA) features"""
        st.markdown("""
        <script>
        // Progressive Web App features
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/service-worker.js')
                .then(function(registration) {
                    console.log('ServiceWorker registration successful');
                })
                .catch(function(err) {
                    console.log('ServiceWorker registration failed');
                });
        }
        
        // Offline detection
        window.addEventListener('online', function() {
            console.log('Application is online');
        });
        
        window.addEventListener('offline', function() {
            console.log('Application is offline');
            alert('You are currently offline. Some features may be limited.');
        });
        
        // Screen orientation detection
        window.addEventListener('orientationchange', function() {
            console.log('Screen orientation changed');
            // Trigger re-layout if needed
        });
        </script>
        
        <link rel="manifest" href="/manifest.json">
        <meta name="theme-color" content="#667eea">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="default">
        <meta name="apple-mobile-web-app-title" content="Real Estate Analytics">
        """, unsafe_allow_html=True)
    
    def create_performance_monitoring(self):
        """Create performance monitoring for mobile optimization"""
        st.markdown("""
        <script>
        // Performance monitoring
        window.addEventListener('load', function() {
            setTimeout(function() {
                const perfData = window.performance.timing;
                const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
                const connectTime = perfData.responseEnd - perfData.requestStart;
                const renderTime = perfData.domComplete - perfData.domLoading;
                
                console.log('Performance Metrics:');
                console.log('Page Load Time:', pageLoadTime, 'ms');
                console.log('Connect Time:', connectTime, 'ms');
                console.log('Render Time:', renderTime, 'ms');
                
                // Send to analytics (in production)
                if (pageLoadTime > 3000) {
                    console.warn('Slow page load detected');
                }
            }, 0);
        });
        
        // Memory usage monitoring
        if ('memory' in performance) {
            setInterval(function() {
                const memoryInfo = performance.memory;
                console.log('Memory Usage:', {
                    used: Math.round(memoryInfo.usedJSHeapSize / 1048576) + ' MB',
                    total: Math.round(memoryInfo.totalJSHeapSize / 1048576) + ' MB',
                    limit: Math.round(memoryInfo.jsHeapSizeLimit / 1048576) + ' MB'
                });
            }, 30000); // Check every 30 seconds
        }
        </script>
        """, unsafe_allow_html=True)
    
    def create_mobile_testing_interface(self):
        """Create interface for testing mobile optimizations"""
        st.markdown("### 📱 Mobile Optimization Testing")
        
        # Device detection info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Device Type", self.device_type)
        
        with col2:
            st.metric("Screen Width", f"{self.screen_size['width']}px")
        
        with col3:
            st.metric("Screen Height", f"{self.screen_size['height']}px")
        
        # Optimization settings
        st.markdown("#### Optimization Settings")
        st.json(self.optimization_settings)
        
        # Test different chart types
        st.markdown("#### Mobile-Optimized Charts")
        
        # Generate sample data
        sample_data = {
            'x': list(range(50)),
            'y': np.random.randn(50).cumsum(),
            'text': [f'Point {i}' for i in range(50)]
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Mobile-optimized line chart
            mobile_chart = self.create_mobile_responsive_chart(
                {'Line Data': sample_data['y']}, 
                'line'
            )
            st.plotly_chart(mobile_chart, use_container_width=True)
        
        with col2:
            # Mobile-optimized bar chart
            bar_data = {'Bar Data': np.random.randint(1, 10, 10)}
            mobile_bar = self.create_mobile_responsive_chart(bar_data, 'bar')
            st.plotly_chart(mobile_bar, use_container_width=True)
        
        # Touch-friendly controls
        st.markdown("#### Touch-Friendly Controls")
        
        with st.form("mobile_test_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                slider_value = st.slider("Touch Slider", 0, 100, 50, 
                                       key="touch_slider", 
                                       help="Designed for touch interaction")
                st.write(f"Slider value: {slider_value}")
            
            with col2:
                checkbox_value = st.checkbox("Touch Checkbox", 
                                           key="touch_checkbox",
                                           help="Large touch target")
                st.write(f"Checkbox: {checkbox_value}")
            
            submitted = st.form_submit_button("Submit Form", 
                                            help="Large touch-friendly button")
            
            if submitted:
                st.success("Form submitted successfully!")
        
        # Performance monitoring
        st.markdown("#### Performance Monitoring")
        
        if st.button("Test Performance"):
            start_time = datetime.now()
            
            # Simulate some processing
            data = np.random.randn(1000, 1000)
            result = np.corrcoef(data)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            st.metric("Processing Time", f"{processing_time:.3f} seconds")
            st.metric("Data Points Processed", "1,000,000")
            
            if processing_time > 1.0:
                st.warning("Processing took longer than expected - consider optimization")
            else:
                st.success("Processing completed efficiently")

def create_mobile_optimized_interface():
    """Create the main mobile-optimized interface"""
    
    # Initialize mobile optimization
    mobile_suite = MobileOptimizationSuite()
    
    # Apply mobile optimizations
    mobile_suite.create_mobile_friendly_layout('dashboard')
    mobile_suite.create_touch_friendly_controls()
    mobile_suite.create_mobile_navigation()
    mobile_suite.create_progressive_web_app_features()
    mobile_suite.create_performance_monitoring()
    
    st.markdown("""
    <div class="mobile-dashboard">
        <h1 style="text-align: center; color: #667eea; margin-bottom: 2rem;">
            📱 Mobile-Optimized Dashboard
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Mobile-friendly metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="mobile-card">
            <div class="mobile-label">Total Properties</div>
            <div class="mobile-metric">1,247</div>
            <div style="font-size: 12px; opacity: 0.8;">+12.3% from last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="mobile-card">
            <div class="mobile-label">Avg Price</div>
            <div class="mobile-metric">₹2.8Cr</div>
            <div style="font-size: 12px; opacity: 0.8;">+8.7% from last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Mobile-optimized charts
    st.markdown("### 📊 Mobile-Optimized Charts")
    
    # Sample data for mobile charts
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    prices = [2.5, 2.7, 2.8, 3.1, 2.9, 3.2]
    
    mobile_line_chart = mobile_suite.create_mobile_responsive_chart(
        {'Average Price': prices}, 'line'
    )
    st.plotly_chart(mobile_line_chart, use_container_width=True)
    
    # Mobile-friendly data table
    st.markdown("### 📋 Quick Stats")
    
    quick_stats = pd.DataFrame({
        'Location': ['Bandra', 'Andheri', 'Juhu', 'Powai'],
        'Avg Price': ['₹4.2Cr', '₹3.1Cr', '₹5.8Cr', '₹2.9Cr'],
        'Properties': [156, 243, 89, 178]
    })
    
    # Optimize table for mobile viewing
    st.dataframe(quick_stats, use_container_width=True)
    
    # Mobile testing interface
    with st.expander("🔧 Mobile Optimization Testing"):
        mobile_suite.create_mobile_testing_interface()
    
    st.markdown("""
    <div style="height: 60px;"></div>
    """, unsafe_allow_html=True)

# Create the mobile-optimized interface
if __name__ == "__main__":
    create_mobile_optimized_interface()