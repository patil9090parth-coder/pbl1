import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from ml_app_enhanced import load_enhanced_predictor, predict_price_enhanced
import warnings
warnings.filterwarnings('ignore')

def test_enhanced_model():
    """Test the enhanced real estate prediction model"""
    
    print("🚀 Testing Enhanced Real Estate Price Prediction Model")
    print("=" * 60)
    
    # Load data and model
    try:
        predictor = load_enhanced_predictor()
        df = pd.read_csv('Final_Project.csv')
        print(f"✅ Data loaded successfully: {len(df)} records")
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading data/model: {e}")
        return
    
    # Test 1: Model accuracy on training data
    print("\n📊 Test 1: Model Performance on Training Data")
    print("-" * 40)
    
    # Prepare features for training data
    X_train = predictor.X_train
    y_train = predictor.y_train
    
    # Predict on training data
    y_pred_train = predictor.model.predict(X_train)
    
    # Calculate metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    print(f"Training MAE: {mae_train:.2f} Lakhs")
    print(f"Training RMSE: {rmse_train:.2f} Lakhs") 
    print(f"Training R² Score: {r2_train:.4f}")
    
    # Test 2: Model accuracy on test data
    print("\n📊 Test 2: Model Performance on Test Data")
    print("-" * 40)
    
    X_test = predictor.X_test
    y_test = predictor.y_test
    
    y_pred_test = predictor.model.predict(X_test)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"Test MAE: {mae_test:.2f} Lakhs")
    print(f"Test RMSE: {rmse_test:.2f} Lakhs")
    print(f"Test R² Score: {r2_test:.4f}")
    
    # Test 3: Individual prediction tests
    print("\n🎯 Test 3: Individual Prediction Tests")
    print("-" * 40)
    
    test_cases = [
        {"area_sqft": 800, "floor_no": 5, "bedroom": 2, "bathroom": 2, "property_age": "1 to 5 Year", "location": "Thane"},
        {"area_sqft": 1200, "floor_no": 10, "bedroom": 3, "bathroom": 3, "property_age": "0 to 1 Year", "location": "South Mumbai"},
        {"area_sqft": 600, "floor_no": 3, "bedroom": 1, "bathroom": 1, "property_age": "5 to 10 Year", "location": "Central Mumbai"},
        {"area_sqft": 2000, "floor_no": 15, "bedroom": 4, "bathroom": 4, "property_age": "Under Construction", "location": "South Mumbai"},
    ]
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = predictor.predict_enhanced(**test_case)
            if result:
                print(f"Test {i}: ✅ PASS")
                print(f"  Input: {test_case}")
                print(f"  Predicted Price: ₹{result['predicted_price']:.2f} Lakhs")
                print(f"  Confidence: {result['confidence_percentage']:.1f}%")
                passed_tests += 1
            else:
                print(f"Test {i}: ❌ FAIL - No prediction returned")
        except Exception as e:
            print(f"Test {i}: ❌ ERROR - {str(e)}")
        print()
    
    # Test 4: Feature importance analysis
    print("📊 Test 4: Feature Importance Analysis")
    print("-" * 40)
    
    feature_importance = predictor.get_feature_importance()
    print("Top 5 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance.head().items(), 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    # Test 5: Model validation with cross-validation
    print("\n📊 Test 5: Cross-Validation Results")
    print("-" * 40)
    
    cv_scores = predictor.get_cv_scores()
    print(f"Cross-validation R² scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"Mean CV R² Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted (Test Data)
    plt.subplot(2, 3, 1)
    plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (Lakhs)')
    plt.ylabel('Predicted Price (Lakhs)')
    plt.title('Actual vs Predicted (Test Data)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    plt.subplot(2, 3, 2)
    residuals = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price (Lakhs)')
    plt.ylabel('Residuals (Lakhs)')
    plt.title('Residual Plot (Test Data)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Feature Importance
    plt.subplot(2, 3, 3)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features.values)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Training vs Test Performance
    plt.subplot(2, 3, 4)
    metrics = ['MAE', 'RMSE', 'R²']
    train_values = [mae_train, rmse_train, r2_train]
    test_values = [mae_test, rmse_test, r2_test]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, train_values, width, label='Training', alpha=0.8)
    plt.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Training vs Test Performance')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Price Distribution Comparison
    plt.subplot(2, 3, 5)
    plt.hist(y_test, bins=30, alpha=0.7, label='Actual', color='blue')
    plt.hist(y_pred_test, bins=30, alpha=0.7, label='Predicted', color='red')
    plt.xlabel('Price (Lakhs)')
    plt.ylabel('Frequency')
    plt.title('Price Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Cross-validation scores
    plt.subplot(2, 3, 6)
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', label=f'Mean: {np.mean(cv_scores):.4f}')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.title('Cross-Validation Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_model_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print("\n📋 SUMMARY REPORT")
    print("=" * 60)
    print(f"✅ Individual Tests: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
    print(f"📊 Test R² Score: {r2_test:.4f}")
    print(f"📊 Test RMSE: {rmse_test:.2f} Lakhs")
    print(f"📊 Test MAE: {mae_test:.2f} Lakhs")
    print(f"📊 Mean CV R² Score: {np.mean(cv_scores):.4f}")
    
    # Model assessment
    if r2_test > 0.8:
        assessment = "EXCELLENT"
    elif r2_test > 0.7:
        assessment = "GOOD"
    elif r2_test > 0.5:
        assessment = "ACCEPTABLE"
    else:
        assessment = "NEEDS IMPROVEMENT"
    
    print(f"\n🎯 Model Assessment: {assessment}")
    
    if assessment == "NEEDS IMPROVEMENT":
        print("💡 Recommendations:")
        print("- Collect more training data")
        print("- Try different algorithms (XGBoost, LightGBM)")
        print("- Add more relevant features")
        print("- Perform hyperparameter tuning")
    else:
        print("✅ Model is ready for deployment!")
    
    print(f"\n📈 Visualization saved as: enhanced_model_validation.png")

if __name__ == "__main__":
    test_enhanced_model()