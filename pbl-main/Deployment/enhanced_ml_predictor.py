import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import lime
import lime.lime_tabular
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedRealEstatePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = None
        self.confidence_intervals = {}
        self.model_performance = {}
        
    def load_data(self, file_path):
        """Load and preprocess data"""
        try:
            df = pd.read_csv(file_path)
            st.success(f"✅ Data loaded successfully: {df.shape[0]} records, {df.shape[1]} features")
            return df
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, df):
        """Advanced data preprocessing"""
        st.info("🔧 Preprocessing data...")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if df_processed[col].isnull().any():
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                st.write(f"Filled {col} missing values with median: {median_val:.2f}")
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if df_processed[col].isnull().any():
                mode_val = df_processed[col].mode()[0]
                df_processed[col].fillna(mode_val, inplace=True)
                st.write(f"Filled {col} missing values with mode: {mode_val}")
        
        # Advanced feature engineering
        df_processed = self.engineer_features(df_processed)
        
        # Remove outliers using IQR method
        df_processed = self.remove_outliers(df_processed)
        
        st.success("✅ Data preprocessing completed")
        return df_processed
    
    def engineer_features(self, df):
        """Create advanced features"""
        st.info("⚙️ Engineering advanced features...")
        
        # Price per square foot (if not exists)
        if 'Area_sqft' in df.columns and 'Total_Price' in df.columns:
            df['Price_per_sqft'] = df['Total_Price'] / df['Area_sqft']
        
        # Total rooms
        if 'Bedrooms' in df.columns and 'Bathrooms' in df.columns:
            df['Total_Rooms'] = df['Bedrooms'] + df['Bathrooms']
        
        # Property age categories
        if 'Age' in df.columns:
            df['Age_Category'] = pd.cut(df['Age'], 
                                        bins=[0, 5, 10, 20, 50, 100], 
                                        labels=['New', 'Recent', 'Moderate', 'Old', 'Very Old'])
        
        # Floor categories
        if 'Floor' in df.columns:
            df['Floor_Category'] = pd.cut(df['Floor'], 
                                        bins=[0, 5, 10, 20, 50], 
                                        labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
        
        # Size categories
        if 'Area_sqft' in df.columns:
            df['Size_Category'] = pd.cut(df['Area_sqft'], 
                                        bins=[0, 500, 1000, 1500, 3000, 5000], 
                                        labels=['Compact', 'Medium', 'Large', 'Extra Large', 'Luxury'])
        
        # Location premium indicator
        if 'Location' in df.columns:
            location_avg_price = df.groupby('Location')['Total_Price'].mean()
            df['Location_Premium'] = df['Location'].map(location_avg_price)
            df['Location_Premium_Category'] = pd.cut(df['Location_Premium'], 
                                                     bins=3, 
                                                     labels=['Budget', 'Mid-Range', 'Premium'])
        
        # Property type multiplier
        if 'Property_Type' in df.columns:
            type_avg_price = df.groupby('Property_Type')['Total_Price'].mean()
            df['Type_Premium'] = df['Property_Type'].map(type_avg_price)
        
        st.success("✅ Feature engineering completed")
        return df
    
    def remove_outliers(self, df, columns=None, method='iqr'):
        """Remove outliers using IQR method"""
        if columns is None:
            columns = ['Total_Price', 'Area_sqft', 'Price_per_sqft']
        
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                
                st.write(f"Removed {len(outliers)} outliers from {col}")
        
        return df_clean
    
    def encode_categorical_features(self, df, fit_encoders=False):
        """Encode categorical features"""
        df_encoded = df.copy()
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != 'Total_Price':  # Don't encode target variable
                if fit_encoders or col not in self.encoders:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = le
                else:
                    le = self.encoders[col]
                    # Handle unseen labels
                    df_encoded[col] = df_encoded[col].astype(str)
                    mask = df_encoded[col].isin(le.classes_)
                    df_encoded.loc[~mask, col] = le.classes_[0]  # Replace unseen with first class
                    df_encoded[col] = le.transform(df_encoded[col])
        
        return df_encoded
    
    def train_models(self, X, y):
        """Train multiple models with cross-validation"""
        st.info("🤖 Training advanced ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=42)
        }
        
        # Train and evaluate models
        for name, model in models.items():
            st.write(f"Training {name}...")
            
            # Train model
            if name == 'Random Forest':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
            
            self.models[name] = model
            self.model_performance[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'CV_Score': -cv_scores.mean()
            }
            
            st.success(f"✅ {name} trained - R²: {r2:.4f}, MAE: ₹{mae:,.0f}")
        
        # Store feature importance (from Random Forest)
        if 'Random Forest' in self.models:
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.models['Random Forest'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        st.success("✅ All models trained successfully!")
        return X_test, y_test
    
    def predict_with_confidence(self, X_new, model_name='Random Forest', confidence_level=0.95):
        """Predict with confidence intervals"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Prepare input data
        if model_name == 'Gradient Boosting':
            X_new_scaled = self.scalers['main'].transform(X_new)
            predictions = model.predict(X_new_scaled)
        else:
            predictions = model.predict(X_new)
        
        # Calculate confidence intervals using bootstrap
        n_bootstrap = 100
        bootstrap_predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(len(X_new), size=len(X_new), replace=True)
            X_bootstrap = X_new.iloc[bootstrap_indices]
            
            if model_name == 'Gradient Boosting':
                X_bootstrap_scaled = self.scalers['main'].transform(X_bootstrap)
                bootstrap_pred = model.predict(X_bootstrap_scaled)
            else:
                bootstrap_pred = model.predict(X_bootstrap)
            
            bootstrap_predictions.append(bootstrap_pred)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate confidence intervals
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
        
        return {
            'prediction': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_interval': (lower_bound, upper_bound),
            'margin_of_error': (upper_bound - lower_bound) / 2
        }
    
    def explain_prediction(self, X_new, model_name='Random Forest'):
        """Explain prediction using SHAP and LIME"""
        explanations = {}
        
        if model_name in self.models:
            model = self.models[model_name]
            
            # SHAP explanation
            try:
                if model_name == 'Random Forest':
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_new)
                    explanations['shap_values'] = shap_values
                    explanations['shap_expected_value'] = explainer.expected_value
            except Exception as e:
                st.warning(f"SHAP explanation failed: {str(e)}")
            
            # LIME explanation
            try:
                # For LIME, we need to use the original training data
                # This is a simplified version - in practice, you'd store training data
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.random.rand(100, X_new.shape[1]),
                    feature_names=X_new.columns.tolist(),
                    mode='regression'
                )
                explanations['lime_explainer'] = lime_explainer
            except Exception as e:
                st.warning(f"LIME explanation failed: {str(e)}")
        
        return explanations
    
    def generate_prediction_report(self, X_new, predictions, confidence_results):
        """Generate comprehensive prediction report"""
        report = {
            'timestamp': datetime.now(),
            'input_features': X_new.to_dict('records'),
            'predictions': predictions,
            'confidence_intervals': confidence_results,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None
        }
        
        return report
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance
        }
        
        joblib.dump(model_data, filepath)
        st.success(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.encoders = model_data['encoders']
            self.feature_importance = model_data['feature_importance']
            self.model_performance = model_data['model_performance']
            st.success("✅ Model loaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False

def create_enhanced_ml_interface():
    """Create the enhanced ML prediction interface for Streamlit"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
        <h2 style="color: white; text-align: center; margin-bottom: 1rem;">
            🧠 Enhanced ML Price Predictor
        </h2>
        <p style="color: rgba(255,255,255,0.9); text-align: center; font-size: 1.1rem;">
            Advanced machine learning with confidence intervals and explainable AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize predictor
    @st.cache_resource
    def load_predictor():
        return EnhancedRealEstatePredictor()
    
    predictor = load_predictor()
    
    # Sidebar for model configuration
    with st.sidebar:
        st.markdown("### ⚙️ Model Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "Select Model",
            ['Random Forest', 'Gradient Boosting'],
            help="Choose the machine learning model for predictions"
        )
        
        # Confidence level
        confidence_level = st.slider(
            "Confidence Level",
            min_value=0.8,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Confidence level for prediction intervals"
        )
        
        # Show feature importance
        show_features = st.checkbox("Show Feature Importance", value=True)
        
        # Show model performance
        show_performance = st.checkbox("Show Model Performance", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏠 Property Details")
        
        with st.form("enhanced_prediction_form"):
            col_left, col_right = st.columns(2)
            
            with col_left:
                location = st.selectbox(
                    "Location",
                    ['Andheri West', 'Bandra West', 'Juhu', 'Goregaon East', 'Malad West',
                     'Kandivali West', 'Borivali West', 'Dahisar', 'Mira Road', 'Bhayander'],
                    help="Select property location"
                )
                
                property_type = st.selectbox(
                    "Property Type",
                    ['Apartment', 'Villa', 'Row House', 'Studio', 'Penthouse'],
                    help="Type of property"
                )
                
                area_sqft = st.number_input(
                    "Area (sqft)",
                    min_value=200,
                    max_value=5000,
                    value=1000,
                    step=50,
                    help="Total area in square feet"
                )
                
                bedrooms = st.number_input(
                    "Bedrooms",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Number of bedrooms"
                )
            
            with col_right:
                bathrooms = st.number_input(
                    "Bathrooms",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Number of bathrooms"
                )
                
                floor = st.number_input(
                    "Floor",
                    min_value=1,
                    max_value=50,
                    value=5,
                    help="Floor number"
                )
                
                age = st.number_input(
                    "Property Age (years)",
                    min_value=0,
                    max_value=50,
                    value=5,
                    help="Age of the property"
                )
                
                furnishing = st.selectbox(
                    "Furnishing",
                    ['Unfurnished', 'Semi-furnished', 'Fully furnished'],
                    help="Furnishing status"
                )
            
            # Advanced features
            with st.expander("🔧 Advanced Features (Optional)"):
                col_adv1, col_adv2 = st.columns(2)
                
                with col_adv1:
                    parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)
                    balcony = st.number_input("Balconies", min_value=0, max_value=5, value=1)
                
                with col_adv2:
                    amenities_score = st.slider("Amenities Score", min_value=1, max_value=10, value=5)
                    maintenance_score = st.slider("Maintenance Score", min_value=1, max_value=10, value=7)
            
            submitted = st.form_submit_button(
                "🔮 Predict with Confidence",
                use_container_width=True,
                help="Get AI-powered price prediction with confidence intervals"
            )
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if submitted:
            with st.spinner("🤖 AI is analyzing your property..."):
                # Create input dataframe
                input_data = pd.DataFrame({
                    'Location': [location],
                    'Property_Type': [property_type],
                    'Area_sqft': [area_sqft],
                    'Bedrooms': [bedrooms],
                    'Bathrooms': [bathrooms],
                    'Floor': [floor],
                    'Age': [age],
                    'Furnishing': [furnishing],
                    'Parking': [parking],
                    'Balcony': [balcony],
                    'Amenities_Score': [amenities_score],
                    'Maintenance_Score': [maintenance_score]
                })
                
                # For demonstration, create synthetic training data
                # In real implementation, load actual training data
                np.random.seed(42)
                n_samples = 1000
                synthetic_data = pd.DataFrame({
                    'Location': np.random.choice(['Andheri West', 'Bandra West', 'Juhu'], n_samples),
                    'Property_Type': np.random.choice(['Apartment', 'Villa', 'Row House'], n_samples),
                    'Area_sqft': np.random.randint(400, 3000, n_samples),
                    'Bedrooms': np.random.randint(1, 6, n_samples),
                    'Bathrooms': np.random.randint(1, 5, n_samples),
                    'Floor': np.random.randint(1, 25, n_samples),
                    'Age': np.random.randint(0, 30, n_samples),
                    'Furnishing': np.random.choice(['Unfurnished', 'Semi-furnished', 'Fully furnished'], n_samples),
                    'Parking': np.random.randint(0, 3, n_samples),
                    'Balcony': np.random.randint(0, 3, n_samples),
                    'Amenities_Score': np.random.randint(1, 11, n_samples),
                    'Maintenance_Score': np.random.randint(1, 11, n_samples),
                    'Total_Price': np.random.randint(5000000, 50000000, n_samples)
                })
                
                # Train model with synthetic data (for demonstration)
                y = synthetic_data['Total_Price']
                X = synthetic_data.drop('Total_Price', axis=1)
                
                # Preprocess and train
                X_processed = predictor.preprocess_data(X)
                X_encoded = predictor.encode_categorical_features(X_processed, fit_encoders=True)
                predictor.train_models(X_encoded, y)
                
                # Make prediction
                input_processed = predictor.preprocess_data(input_data)
                input_encoded = predictor.encode_categorical_features(input_processed)
                
                # Get prediction with confidence
                confidence_results = predictor.predict_with_confidence(
                    input_encoded, 
                    model_name=model_choice,
                    confidence_level=confidence_level
                )
                
                prediction = confidence_results['prediction'][0]
                lower_bound = confidence_results['lower_bound'][0]
                upper_bound = confidence_results['upper_bound'][0]
                margin_of_error = confidence_results['margin_of_error'][0]
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                # Main prediction
                st.metric(
                    label="Predicted Price",
                    value=f"₹{prediction:,.0f}",
                    delta=f"±{margin_of_error/prediction*100:.1f}% margin"
                )
                
                # Confidence interval
                st.write(f"**{int(confidence_level*100)}% Confidence Interval:**")
                st.write(f"₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}")
                
                # Price breakdown
                price_per_sqft = prediction / area_sqft
                st.write(f"**Price per sqft:** ₹{price_per_sqft:,.0f}")
                
                # Market comparison
                market_avg = np.random.randint(8000, 15000)  # Mock market average
                comparison = ((price_per_sqft - market_avg) / market_avg) * 100
                
                if comparison > 0:
                    st.write(f"**vs Market:** {comparison:.1f}% above average")
                else:
                    st.write(f"**vs Market:** {abs(comparison):.1f}% below average")
                
                # Save prediction
                if st.button("💾 Save Prediction", key="save_enhanced"):
                    if 'enhanced_predictions' not in st.session_state:
                        st.session_state.enhanced_predictions = []
                    
                    st.session_state.enhanced_predictions.append({
                        'timestamp': datetime.now(),
                        'input': input_data.to_dict('records')[0],
                        'prediction': prediction,
                        'confidence_interval': (lower_bound, upper_bound),
                        'model': model_choice,
                        'confidence_level': confidence_level
                    })
                    st.success("✅ Prediction saved!")
        
        else:
            st.info("👆 Fill in property details and click 'Predict with Confidence' to get started")
    
    # Model performance and insights
    if show_performance and predictor.model_performance:
        st.markdown("### 📈 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            performance = predictor.model_performance[model_choice]
            st.metric("R² Score", f"{performance['R2']:.4f}")
        
        with col2:
            st.metric("MAE", f"₹{performance['MAE']:,.0f}")
        
        with col3:
            st.metric("RMSE", f"₹{performance['RMSE']:,.0f}")
    
    # Feature importance
    if show_features and predictor.feature_importance is not None:
        st.markdown("### 🔍 Feature Importance")
        
        # Top 10 features
        top_features = predictor.feature_importance.head(10)
        
        fig = px.bar(
            top_features, 
            x='importance', 
            y='feature',
            orientation='h',
            title=f'Top 10 Most Important Features ({model_choice})',
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=16, color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions
    if 'enhanced_predictions' in st.session_state and st.session_state.enhanced_predictions:
        st.markdown("### 🕒 Recent Predictions")
        
        recent_predictions = st.session_state.enhanced_predictions[-5:]  # Last 5 predictions
        
        for i, pred in enumerate(reversed(recent_predictions)):
            with st.expander(f"Prediction {len(recent_predictions)-i} - {pred['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Location:** {pred['input']['Location']}")
                    st.write(f"**Type:** {pred['input']['Property_Type']}")
                    st.write(f"**Area:** {pred['input']['Area_sqft']} sqft")
                
                with col2:
                    st.write(f"**Prediction:** ₹{pred['prediction']:,.0f}")
                    st.write(f"**Confidence:** {int(pred['confidence_level']*100)}%")
                    st.write(f"**Model:** {pred['model']}")

# Create the enhanced ML predictor interface
if __name__ == "__main__":
    create_enhanced_ml_interface()