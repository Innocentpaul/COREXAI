"""
Debug script to test the AI Explainability Dashboard components
"""

import numpy as np
import joblib
import os
import sys
import traceback

def test_models():
    """Test if models can be loaded and used"""
    print("Testing model loading and predictions...")
    print("-" * 50)
    
    models = {}
    model_files = {
        "intrusion": "random_forest_model.pkl",
        "fraud": "FFD_rf_model.pkl"
    }
    
    for name, file_path in model_files.items():
        print(f"\nTesting {name} model:")
        if not os.path.exists(file_path):
            print(f"  ✗ Model file '{file_path}' not found!")
            continue
            
        try:
            # Load model
            model = joblib.load(file_path)
            models[name] = model
            print(f"  ✓ Model loaded successfully")
            
            # Check model attributes
            if hasattr(model, 'n_features_in_'):
                print(f"  → Expected features: {model.n_features_in_}")
            if hasattr(model, 'n_estimators'):
                print(f"  → Number of trees: {model.n_estimators}")
            if hasattr(model, 'classes_'):
                print(f"  → Classes: {model.classes_}")
                
            # Test prediction
            test_data = np.random.rand(1, 30)
            pred = model.predict(test_data)
            proba = model.predict_proba(test_data)
            print(f"  ✓ Prediction test passed - Prediction: {pred[0]}, Probabilities: {proba[0]}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            traceback.print_exc()
    
    return models

def test_shap():
    """Test SHAP functionality"""
    print("\n\nTesting SHAP...")
    print("-" * 50)
    
    try:
        import shap
        print("✓ SHAP imported successfully")
        
        # Create a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        X = np.random.rand(100, 30)
        y = (X[:, 0] > 0.5).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X[:1])
        
        print("✓ SHAP TreeExplainer works")
        print(f"  → SHAP values shape: {shap_values.values.shape}")
        print(f"  → Base values shape: {shap_values.base_values.shape}")
        
    except Exception as e:
        print(f"✗ SHAP error: {str(e)}")
        traceback.print_exc()

def test_lime():
    """Test LIME functionality"""
    print("\n\nTesting LIME...")
    print("-" * 50)
    
    try:
        import lime
        import lime.lime_tabular
        print("✓ LIME imported successfully")
        
        # Create a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        X = np.random.rand(100, 30)
        y = (X[:, 0] > 0.5).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test LIME
        feature_names = [f"Feature_{i}" for i in range(30)]
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=feature_names,
            class_names=["Class 0", "Class 1"],
            mode="classification"
        )
        
        exp = explainer.explain_instance(
            X[0], 
            model.predict_proba, 
            num_features=10
        )
        
        print("✓ LIME explainer works")
        print(f"  → Explanation available for {len(exp.as_list())} features")
        
    except Exception as e:
        print(f"✗ LIME error: {str(e)}")
        traceback.print_exc()

def test_matplotlib():
    """Test matplotlib functionality"""
    print("\n\nTesting Matplotlib...")
    print("-" * 50)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
        
        # Create a simple plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close()
        
        print("✓ Matplotlib plotting works")
        
    except Exception as e:
        print(f"✗ Matplotlib error: {str(e)}")
        traceback.print_exc()

def test_plotly():
    """Test plotly functionality"""
    print("\n\nTesting Plotly...")
    print("-" * 50)
    
    try:
        import plotly.graph_objects as go
        print("✓ Plotly imported successfully")
        
        # Create a simple 3D plot
        fig = go.Figure(data=[go.Scatter3d(
            x=[1, 2, 3],
            y=[1, 2, 3],
            z=[1, 2, 3],
            mode='markers'
        )])
        
        html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        print("✓ Plotly 3D plotting works")
        print(f"  → HTML output length: {len(html)} characters")
        
    except Exception as e:
        print(f"✗ Plotly error: {str(e)}")
        traceback.print_exc()

def test_api_endpoint():
    """Test the API endpoint locally"""
    print("\n\nTesting API endpoint simulation...")
    print("-" * 50)
    
    try:
        # Load a model
        if os.path.exists("random_forest_model.pkl"):
            model = joblib.load("random_forest_model.pkl")
            print("✓ Model loaded for API test")
            
            # Simulate API request data
            request_data = {
                "model_type": "intrusion",
                "explanation_type": "shap_3d",
                "num_samples": 1,
                "top_features": 10
            }
            
            # Generate sample data
            sample_data = np.random.rand(1, 30)
            
            # Test prediction
            predictions = model.predict(sample_data)
            probabilities = model.predict_proba(sample_data)
            
            print(f"✓ Predictions work: {predictions[0]}, Probabilities: {probabilities[0]}")
            
        else:
            print("✗ No model file found for API test")
            
    except Exception as e:
        print(f"✗ API test error: {str(e)}")
        traceback.print_exc()

def main():
    """Run all tests"""
    print("COREXAI Dashboard - Debug Testing")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    # Run tests
    models = test_models()
    test_shap()
    test_lime()
    test_matplotlib()
    test_plotly()
    test_api_endpoint()
    
    print("\n" + "=" * 50)
    print("Debug testing complete!")
    
    # Recommendations
    print("\nRecommendations:")
    if not models:
        print("• Run 'python create_models.py' to generate model files")
    
    print("• Make sure all dependencies are installed correctly")
    print("• Check the Flask app logs when running for more detailed error messages")

if __name__ == "__main__":
    main()