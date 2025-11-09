import pickle
import numpy as np
import pandas as pd
import streamlit as st
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

class RealEstatePredictor:
    """Enhanced Real Estate Price Predictor with proper feature engineering"""
    
    def __init__(self):
        self.df = None
        self.model = None
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.is_trained = False
    
    def load_data(self, filename="Final_Project.csv"):
        """Load and validate data"""
        try:
            # Construct absolute path to the CSV file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, filename)
            self.df = pd.read_csv(filepath)
            print(f"✅ Loaded {len(self.df)} property records")
            return True
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Advanced feature engineering and preprocessing"""
        if self.df is None:
            return False
        
        # Create a copy for processing
        df_processed = self.df.copy()
        
        # Handle missing values
        df_processed = df_processed.dropna(subset=['Price_Lakh', 'Area_SqFt', 'Floor_No', 'Bedroom', 'Bathroom'])
        
        # Remove outliers (properties with unrealistic prices)
        Q1 = df_processed['Price_Lakh'].quantile(0.25)
        Q3 = df_processed['Price_Lakh'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_processed = df_processed[(df_processed['Price_Lakh'] >= lower_bound) & 
                                   (df_processed['Price_Lakh'] <= upper_bound)]
        
        # Feature engineering
        # 1. Price per square foot
        df_processed['Price_per_sqft'] = df_processed['Price_Lakh'] * 100000 / df_processed['Area_SqFt']
        
        # 2. Total rooms
        df_processed['Total_rooms'] = df_processed['Bedroom'] + df_processed['Bathroom']
        
        # 3. Floor category (basement, low, mid, high, penthouse)
        def categorize_floor(floor):
            if floor < 0:
                return 'Basement'
            elif floor <= 5:
                return 'Low'
            elif floor <= 15:
                return 'Mid'
            elif floor <= 25:
                return 'High'
            else:
                return 'Penthouse'
        
        df_processed['Floor_category'] = df_processed['Floor_No'].apply(categorize_floor)
        
        # 4. Property size category
        def categorize_size(area):
            if area < 600:
                return 'Compact'
            elif area < 1200:
                return 'Medium'
            elif area < 2000:
                return 'Large'
            else:
                return 'Luxury'
        
        df_processed['Size_category'] = df_processed['Area_SqFt'].apply(categorize_size)
        
        # 5. Age in years (simplified from Property_Age)
        age_mapping = {
            'Under Construction': 0,
            '0 to 1 Year': 0.5,
            '1 to 5 Year': 3,
            '5 to 10 Year': 7.5,
            '10+ Year': 15
        }
        df_processed['Age_years'] = df_processed['Property_Age'].map(age_mapping)
        
        # 6. Location tier (based on average prices)
        location_avg_price = df_processed.groupby('Region')['Price_Lakh'].mean()
        location_tiers = pd.qcut(location_avg_price, q=3, labels=['Budget', 'Mid-tier', 'Premium'])
        df_processed['Location_tier'] = df_processed['Region'].map(location_tiers)
        
        self.df_processed = df_processed
        print(f"✅ Preprocessed data: {len(df_processed)} records after cleaning")
        return True
    
    def train_model(self):
        """Train a new Random Forest model with proper feature engineering"""
        if not hasattr(self, 'df_processed'):
            if not self.preprocess_data():
                return False
        
        # Prepare features
        feature_columns = [
            'Area_SqFt', 'Floor_No', 'Bedroom', 'Bathroom', 'Age_years',
            'Price_per_sqft', 'Total_rooms'
        ]
        
        # Encode categorical variables
        le_floor = LabelEncoder()
        le_size = LabelEncoder()
        le_location = LabelEncoder()
        
        self.df_processed['Floor_category_encoded'] = le_floor.fit_transform(self.df_processed['Floor_category'])
        self.df_processed['Size_category_encoded'] = le_size.fit_transform(self.df_processed['Size_category'])
        self.df_processed['Location_tier_encoded'] = le_location.fit_transform(self.df_processed['Location_tier'])
        
        feature_columns.extend(['Floor_category_encoded', 'Size_category_encoded', 'Location_tier_encoded'])
        
        # Prepare data
        X = self.df_processed[feature_columns].copy()
        y = self.df_processed['Price_Lakh'].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Store training data for analysis
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully")
        print(f"   Mean Absolute Error: {mae:.2f} Lakhs")
        print(f"   R² Score: {r2:.3f}")
        
        # Store artifacts
        self.scalers = {'main': scaler}
        self.encoders = {
            'floor': le_floor,
            'size': le_size,
            'location': le_location
        }
        self.feature_columns = feature_columns
        self.is_trained = True
        
        return True
    
    def predict_enhanced(self, area_sqft, floor_no, bedroom, bathroom, property_age, location):
        """Enhanced prediction with proper feature engineering"""
        if not self.is_trained:
            return None
        
        try:
            # Create input features
            input_data = pd.DataFrame({
                'Area_SqFt': [area_sqft],
                'Floor_No': [floor_no],
                'Bedroom': [bedroom],
                'Bathroom': [bathroom]
            })
            
            # Add derived features
            input_data['Age_years'] = {'Under Construction': 0, '0 to 1 Year': 0.5, 
                                       '1 to 5 Year': 3, '5 to 10 Year': 7.5, '10+ Year': 15}.get(property_age, 7.5)
            
            # Calculate price per sqft based on location averages
            location_avg_price = self.df_processed.groupby('Region')['Price_Lakh'].mean()
            avg_price = location_avg_price.get(location, location_avg_price.mean())
            input_data['Price_per_sqft'] = (avg_price * 100000) / area_sqft
            
            input_data['Total_rooms'] = bedroom + bathroom
            
            # Encode categorical features
            def categorize_floor(floor):
                if floor < 0:
                    return 'Basement'
                elif floor <= 5:
                    return 'Low'
                elif floor <= 15:
                    return 'Mid'
                elif floor <= 25:
                    return 'High'
                else:
                    return 'Penthouse'
            
            def categorize_size(area):
                if area < 600:
                    return 'Compact'
                elif area < 1200:
                    return 'Medium'
                elif area < 2000:
                    return 'Large'
                else:
                    return 'Luxury'
            
            location_avg_price = self.df_processed.groupby('Region')['Price_Lakh'].mean()
            location_tiers = pd.qcut(location_avg_price, q=3, labels=['Budget', 'Mid-tier', 'Premium'])
            location_tier = location_tiers.get(location, 'Mid-tier')
            
            input_data['Floor_category_encoded'] = self.encoders['floor'].transform([categorize_floor(floor_no)])[0]
            input_data['Size_category_encoded'] = self.encoders['size'].transform([categorize_size(area_sqft)])[0]
            input_data['Location_tier_encoded'] = self.encoders['location'].transform([location_tier])[0]
            
            # Ensure correct column order
            input_features = input_data[self.feature_columns]
            
            # Scale features
            input_scaled = self.scalers['main'].transform(input_features)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            # Add confidence interval (simplified)
            confidence_range = prediction * 0.15  # ±15% confidence interval
            
            return {
                'predicted_price': max(0, prediction),  # Ensure non-negative
                'confidence_lower': max(0, prediction - confidence_range),
                'confidence_upper': prediction + confidence_range,
                'confidence_percentage': 85
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def get_market_insights(self, location, area_sqft):
        """Get market insights for a specific location and area"""
        if self.df_processed is None:
            return None
        
        location_data = self.df_processed[self.df_processed['Region'] == location]
        if len(location_data) == 0:
            location_data = self.df_processed
        
        similar_properties = location_data[
            (location_data['Area_SqFt'] >= area_sqft * 0.8) & 
            (location_data['Area_SqFt'] <= area_sqft * 1.2)
        ]
        
        if len(similar_properties) == 0:
            similar_properties = location_data
        
        return {
            'avg_price': similar_properties['Price_Lakh'].mean(),
            'min_price': similar_properties['Price_Lakh'].min(),
            'max_price': similar_properties['Price_Lakh'].max(),
            'count': len(similar_properties),
            'avg_price_per_sqft': (similar_properties['Price_Lakh'] * 100000 / similar_properties['Area_SqFt']).mean()
        }
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if hasattr(self, 'model') and hasattr(self, 'feature_names'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df.set_index('feature')['importance']
        else:
            return pd.Series()
    
    def get_cv_scores(self):
        """Get cross-validation scores"""
        if hasattr(self, 'model') and hasattr(self, 'X_train') and hasattr(self, 'y_train'):
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='r2')
            return cv_scores
        else:
            return np.array([0, 0, 0, 0, 0])

# Global predictor instance
predictor = RealEstatePredictor()

def load_enhanced_predictor():
    """Load the enhanced predictor"""
    try:
        predictor_instance = RealEstatePredictor()
        predictor_instance.load_data()
        predictor_instance.train_model()
        return predictor_instance
    except Exception as e:
        st.error(f"Error loading enhanced predictor: {str(e)}")
        return None

def predict_price_enhanced(area_sqft, floor_no, bedroom, bathroom, property_age, location):
    """Enhanced prediction function using the improved model"""
    if not predictor.is_trained:
        return None
    
    return predictor.predict_enhanced(area_sqft, floor_no, bedroom, bathroom, property_age, location)

def run_ml_app_enhanced():
    """Enhanced ML app with improved predictions"""
    st.subheader('🏠 Enhanced Real Estate Price Predictor')
    
    # Initialize predictor if not already done
    if not predictor.is_trained:
        with st.spinner("🚀 Initializing enhanced prediction engine..."):
            if not predictor.load_data() or not predictor.train_model():
                st.error("Failed to initialize predictor")
                return
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.selectbox('📍 Select Location', 
                               sorted(predictor.df['Region'].unique()))
        area_sqft = st.slider("📐 Area (SqFt)", 
                             300, int(predictor.df['Area_SqFt'].max()), 
                             step=50, value=800)
        floor_no = st.number_input("🏢 Floor Number", 
                                  min_value=-2, max_value=50, value=5)
    
    with col2:
        bedroom = st.selectbox("🛏️ Bedrooms", 
                              sorted(predictor.df['Bedroom'].unique()))
        bathroom = st.selectbox("🚿 Bathrooms", 
                             sorted(predictor.df['Bathroom'].unique()))
        property_age = st.selectbox('🏗️ Property Age', 
                                   sorted(predictor.df['Property_Age'].unique()))
    
    # Get market insights
    insights = predictor.get_market_insights(location, area_sqft)
    
    if insights and insights['count'] > 0:
        st.info(f"📊 Market Insights for {location}:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Price", f"₹{insights['avg_price']:.1f}L")
        with col2:
            st.metric("Price Range", f"₹{insights['min_price']:.0f}L - ₹{insights['max_price']:.0f}L")
        with col3:
            st.metric("₹ per SqFt", f"₹{insights['avg_price_per_sqft']:.0f}")
    
    # Prediction button
    if st.button("🔮 Predict Price", type="primary"):
        # Validate inputs
        if area_sqft <= 0:
            st.error("❌ Area must be greater than 0")
            return
        
        if bedroom <= 0 or bathroom <= 0:
            st.error("❌ Bedrooms and bathrooms must be at least 1")
            return
        
        # Make prediction
        with st.spinner("🧠 Analyzing market data..."):
            result = predict_price_enhanced(
                area_sqft, floor_no, bedroom, bathroom, property_age, location
            )
        
        if result:
            # Display results
            st.success(f"💰 **Predicted Price: ₹{result['predicted_price']:.2f} Lakhs**")
            
            # Confidence interval
            st.info(f"📈 Confidence Range: ₹{result['confidence_lower']:.2f}L - ₹{result['confidence_upper']:.2f}L")
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                price_per_sqft = (result['predicted_price'] * 100000) / area_sqft
                st.metric("Price per SqFt", f"₹{price_per_sqft:.0f}")
            with col2:
                if insights and insights['count'] > 0:
                    market_comparison = ((result['predicted_price'] - insights['avg_price']) / insights['avg_price']) * 100
                    st.metric("vs Market Avg", f"{market_comparison:+.1f}%")
            with col3:
                st.metric("Confidence", f"{result['confidence_percentage']}%")
            
            # Property analysis
            with st.expander("🔍 Property Analysis"):
                st.write(f"- **Property Type**: {bedroom}BHK, {bathroom} Bathroom(s)")
                st.write(f"- **Size Category**: {area_sqft} sqft")
                st.write(f"- **Floor Position**: {floor_no} ({'Basement' if floor_no < 0 else 'Low' if floor_no <= 5 else 'Mid' if floor_no <= 15 else 'High'} floor)")
                st.write(f"- **Age**: {property_age}")
                st.write(f"- **Location Tier**: {location}")
        else:
            st.error("❌ Prediction failed. Please check your inputs.")
    
    # Disclaimer
    st.caption("⚠️ *This prediction is based on historical data and market trends. Actual prices may vary.*")

if __name__ == '__main__':
    run_ml_app_enhanced()