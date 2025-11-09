#!/usr/bin/env python3
"""
Test script for Real Estate Price Prediction Model
Tests prediction accuracy and model functionality
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_model_and_data():
    """Load the trained model and test data"""
    try:
        df = pd.read_csv("Final_Project.csv")
        with open('regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return df, model
    except Exception as e:
        print(f"Error loading model/data: {e}")
        return None, None

def create_test_cases(df):
    """Create realistic test cases based on actual data patterns"""
    test_cases = []
    
    # Test case 1: Small apartment in Central Mumbai
    test_cases.append({
        'Area_SqFt': 650,
        'Floor_No': 5,
        'Bedroom': 2,
        'Bathroom': 2,
        'Property_Age': '5 to 10 Year',
        'Location': 'Central Mumbai',
        'expected_range': (150, 300)
    })
    
    # Test case 2: Large apartment in South Mumbai
    test_cases.append({
        'Area_SqFt': 1500,
        'Floor_No': 10,
        'Bedroom': 3,
        'Bathroom': 3,
        'Property_Age': '0 to 1 Year',
        'Location': 'South Mumbai',
        'expected_range': (400, 800)
    })
    
    # Test case 3: Budget apartment in Thane
    test_cases.append({
        'Area_SqFt': 800,
        'Floor_No': 3,
        'Bedroom': 2,
        'Bathroom': 2,
        'Property_Age': '10+ Year',
        'Location': 'Thane',
        'expected_range': (80, 180)
    })
    
    # Test case 4: Studio apartment
    test_cases.append({
        'Area_SqFt': 400,
        'Floor_No': 2,
        'Bedroom': 1,
        'Bathroom': 1,
        'Property_Age': '1 to 5 Year',
        'Location': 'Malad Mumbai',
        'expected_range': (60, 150)
    })
    
    return test_cases

def predict_price_improved(Area_SqFt, Floor_No, Bedroom, Bathroom, Property_Age, Location, df, model):
    """Improved prediction function that uses all features"""
    try:
        # Create feature array - using actual data patterns
        x = np.zeros(7)
        
        # Basic features
        x[0] = Area_SqFt
        x[1] = Floor_No
        x[2] = Bedroom
        x[3] = Bathroom
        
        # Property age encoding
        age_mapping = {'0 to 1 Year': 0, '1 to 5 Year': 1, '5 to 10 Year': 2, '10+ Year': 3, 'Under Construction': 4}
        x[4] = age_mapping.get(Property_Age, 2)
        
        # Location encoding (using average price per region as proxy)
        location_prices = df.groupby('Region')['Price_Lakh'].mean()
        avg_price = location_prices.get(Location, location_prices.mean())
        x[5] = avg_price / 100  # Normalize
        
        # Additional derived feature
        x[6] = (Area_SqFt * Floor_No) / 1000
        
        return model.predict([x])[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def test_model_accuracy(df, model):
    """Test model accuracy against actual data"""
    print("🔍 Testing Model Accuracy...")
    
    # Sample 100 random properties for testing
    test_sample = df.sample(min(100, len(df)), random_state=42)
    
    predictions = []
    actual_prices = []
    
    for idx, row in test_sample.iterrows():
        try:
            pred = predict_price_improved(
                row['Area_SqFt'], 
                row['Floor_No'], 
                row['Bedroom'], 
                row['Bathroom'], 
                row['Property_Age'], 
                row['Region'],
                df, 
                model
            )
            if pred is not None:
                predictions.append(pred)
                actual_prices.append(row['Price_Lakh'])
        except:
            continue
    
    if len(predictions) > 0:
        mae = mean_absolute_error(actual_prices, predictions)
        mse = mean_squared_error(actual_prices, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_prices, predictions)
        
        print(f"📊 Model Accuracy Metrics:")
        print(f"   Mean Absolute Error: {mae:.2f} Lakhs")
        print(f"   Root Mean Square Error: {rmse:.2f} Lakhs")
        print(f"   R² Score: {r2:.3f}")
        print(f"   Mean Actual Price: {np.mean(actual_prices):.2f} Lakhs")
        print(f"   Mean Predicted Price: {np.mean(predictions):.2f} Lakhs")
        
        return mae, rmse, r2, actual_prices, predictions
    else:
        print("❌ No valid predictions could be made")
        return None, None, None, None, None

def test_individual_predictions(df, model):
    """Test individual prediction cases"""
    print("\n🧪 Testing Individual Predictions...")
    
    test_cases = create_test_cases(df)
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case['Location']} - {case['Bedroom']}BHK")
        print(f"   Area: {case['Area_SqFt']} sqft, Floor: {case['Floor_No']}, Age: {case['Property_Age']}")
        
        prediction = predict_price_improved(
            case['Area_SqFt'],
            case['Floor_No'],
            case['Bedroom'],
            case['Bathroom'],
            case['Property_Age'],
            case['Location'],
            df,
            model
        )
        
        if prediction:
            min_exp, max_exp = case['expected_range']
            status = "✅ PASS" if min_exp <= prediction <= max_exp else "❌ FAIL"
            
            print(f"   Predicted: ₹{prediction:.2f} Lakhs")
            print(f"   Expected Range: ₹{min_exp} - ₹{max_exp} Lakhs")
            print(f"   Status: {status}")
            
            results.append({
                'test_case': i,
                'prediction': prediction,
                'expected_min': min_exp,
                'expected_max': max_exp,
                'status': status == "✅ PASS"
            })
        else:
            print("   ❌ Prediction failed")
            results.append({'test_case': i, 'status': False})
    
    return results

def create_accuracy_plot(actual_prices, predictions):
    """Create accuracy visualization"""
    if actual_prices and predictions:
        plt.figure(figsize=(12, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(actual_prices, predictions, alpha=0.6, color='blue')
        plt.plot([min(actual_prices), max(actual_prices)], [min(actual_prices), max(actual_prices)], 'r--', lw=2)
        plt.xlabel('Actual Price (Lakhs)')
        plt.ylabel('Predicted Price (Lakhs)')
        plt.title('Actual vs Predicted Prices')
        plt.grid(True, alpha=0.3)
        
        # Error distribution
        plt.subplot(1, 2, 2)
        errors = np.array(predictions) - np.array(actual_prices)
        plt.hist(errors, bins=20, alpha=0.7, color='green')
        plt.xlabel('Prediction Error (Lakhs)')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_accuracy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Accuracy plot saved as 'model_accuracy_analysis.png'")

def main():
    """Main testing function"""
    print("🏠 Real Estate Price Prediction - Model Testing")
    print("=" * 50)
    
    # Load model and data
    df, model = load_model_and_data()
    if df is None or model is None:
        print("❌ Failed to load model or data")
        return
    
    print(f"✅ Loaded dataset with {len(df)} properties")
    print(f"✅ Loaded prediction model: {type(model).__name__}")
    
    # Test individual predictions
    individual_results = test_individual_predictions(df, model)
    
    # Test model accuracy
    accuracy_metrics = test_model_accuracy(df, model)
    
    if accuracy_metrics[0] is not None:
        mae, rmse, r2, actual_prices, predictions = accuracy_metrics
        
        # Summary
        print("\n" + "="*50)
        print("📋 TEST SUMMARY")
        print("="*50)
        
        # Individual test results
        passed_tests = sum(1 for r in individual_results if r.get('status', False))
        total_tests = len(individual_results)
        print(f"Individual Tests: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
        
        # Model accuracy
        print(f"Model R² Score: {r2:.3f}")
        print(f"Average Error: {mae:.2f} Lakhs")
        
        # Overall assessment
        if r2 > 0.7 and passed_tests >= total_tests * 0.7:
            print("🎉 Overall Assessment: EXCELLENT - Model is ready for production")
        elif r2 > 0.5 and passed_tests >= total_tests * 0.5:
            print("👍 Overall Assessment: GOOD - Model needs minor improvements")
        else:
            print("⚠️ Overall Assessment: NEEDS IMPROVEMENT - Model requires significant enhancement")
        
        # Create visualization
        create_accuracy_plot(actual_prices, predictions)
        
    else:
        print("❌ Could not complete accuracy testing")

if __name__ == "__main__":
    main()