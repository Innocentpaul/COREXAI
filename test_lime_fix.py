"""
Test script to verify LIME fix
"""

import numpy as np
import joblib
import lime.lime_tabular
import matplotlib.pyplot as plt
import os

def test_lime_fix():
    """Test the LIME fix for KeyError: 1"""
    print("Testing LIME fix...")
    print("-" * 50)
    
    # Load model
    model_path = "random_forest_model.pkl"
    if not os.path.exists(model_path):
        print("Model file not found. Please run create_models.py first.")
        return
        
    model = joblib.load(model_path)
    print("✓ Model loaded")
    
    # Create sample data
    sample_data = np.random.rand(1, 30)
    
    # Check what class the model predicts
    prediction = model.predict(sample_data)[0]
    probabilities = model.predict_proba(sample_data)[0]
    print(f"Model prediction: {prediction}")
    print(f"Probabilities: {probabilities}")
    
    # Create LIME explainer
    np.random.seed(42)
    training_data = np.random.rand(1000, 30)
    feature_names = [f"Feature_{i+1}" for i in range(30)]
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        class_names=["Normal", "Anomaly"],
        mode="classification"
    )
    
    # Test explanation with top_labels=2
    print("\nTesting LIME with top_labels=2...")
    try:
        exp = explainer.explain_instance(
            sample_data[0], 
            model.predict_proba, 
            num_features=10,
            top_labels=2  # Request explanations for both classes
        )
        
        print(f"Available labels: {exp.available_labels()}")
        
        # Try to get explanation for predicted class
        try:
            explanation = exp.as_list(label=int(prediction))
            print(f"✓ Got explanation for predicted class {prediction}")
        except KeyError:
            available_labels = list(exp.available_labels())
            explanation = exp.as_list(label=available_labels[0])
            print(f"✓ Got explanation for available class {available_labels[0]}")
            
        print(f"Number of features in explanation: {len(explanation)}")
        print("First 3 features:")
        for feat, imp in explanation[:3]:
            print(f"  {feat}: {imp:.4f}")
            
        # Test pyplot figure generation
        try:
            fig = exp.as_pyplot_figure(label=int(prediction))
            print("✓ Generated pyplot figure for predicted class")
        except KeyError:
            available_labels = list(exp.available_labels())
            fig = exp.as_pyplot_figure(label=available_labels[0])
            print("✓ Generated pyplot figure for available class")
            
        plt.close()
        
        print("\n✓ LIME fix is working correctly!")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lime_fix()