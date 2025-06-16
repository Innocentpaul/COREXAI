"""
Create sample Random Forest models for testing the AI Explainability Dashboard.
This script generates two models: one for intrusion detection and one for fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_intrusion_detection_model():
    """Create a sample intrusion detection model with 30 features."""
    print("Creating Intrusion Detection Model...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 5000
    n_features = 30
    
    # Create features - mix of different patterns
    X = np.random.randn(n_samples, n_features)
    
    # Create labels based on some logical rules
    # Intrusion if: high values in features 0-4 OR low values in features 25-29
    intrusion_condition1 = np.sum(X[:, :5] > 1.5, axis=1) >= 3
    intrusion_condition2 = np.sum(X[:, 25:] < -1.5, axis=1) >= 3
    
    y = (intrusion_condition1 | intrusion_condition2).astype(int)
    
    # Add some noise to make it more realistic
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Print accuracy
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Intrusion Detection Model - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
    
    # Save the model
    joblib.dump(model, 'random_forest_model.pkl')
    print("Saved intrusion detection model as 'random_forest_model.pkl'")
    
    return model

def create_fraud_detection_model():
    """Create a sample fraud detection model with 30 features."""
    print("\nCreating Fraud Detection Model...")
    
    # Generate synthetic data
    np.random.seed(123)
    n_samples = 5000
    n_features = 30
    
    # Create features - different patterns than intrusion
    X = np.random.randn(n_samples, n_features)
    
    # Create labels based on different rules
    # Fraud if: unusual pattern in transaction features
    fraud_condition1 = (X[:, 5] > 2) & (X[:, 10] < -1.5)  # Unusual amount + time
    fraud_condition2 = np.sum(X[:, 15:20] > 1.8, axis=1) >= 3  # Multiple risk factors
    fraud_condition3 = (X[:, 0] < -2) & (X[:, 1] < -2)  # Unusual location pattern
    
    y = (fraud_condition1 | fraud_condition2 | fraud_condition3).astype(int)
    
    # Add some noise
    noise_indices = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    # Make fraud less common (imbalanced dataset)
    fraud_indices = np.where(y == 1)[0]
    keep_fraud = np.random.choice(fraud_indices, size=int(0.3 * len(fraud_indices)), replace=False)
    remove_fraud = np.setdiff1d(fraud_indices, keep_fraud)
    y[remove_fraud] = 0
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle imbalanced data
        random_state=123
    )
    
    model.fit(X_train, y_train)
    
    # Print accuracy
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Fraud Detection Model - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
    
    # Save the model
    joblib.dump(model, 'FFD_rf_model.pkl')
    print("Saved fraud detection model as 'FFD_rf_model.pkl'")
    
    return model

def verify_models():
    """Verify that the models work correctly."""
    print("\nVerifying models...")
    
    # Test intrusion detection model
    if os.path.exists('random_forest_model.pkl'):
        intrusion_model = joblib.load('random_forest_model.pkl')
        test_data = np.random.randn(1, 30)
        prediction = intrusion_model.predict(test_data)
        probability = intrusion_model.predict_proba(test_data)
        print(f"Intrusion model test - Prediction: {prediction[0]}, Probabilities: {probability[0]}")
        print(f"Intrusion model features: {intrusion_model.n_features_in_}")
    
    # Test fraud detection model
    if os.path.exists('FFD_rf_model.pkl'):
        fraud_model = joblib.load('FFD_rf_model.pkl')
        test_data = np.random.randn(1, 30)
        prediction = fraud_model.predict(test_data)
        probability = fraud_model.predict_proba(test_data)
        print(f"Fraud model test - Prediction: {prediction[0]}, Probabilities: {probability[0]}")
        print(f"Fraud model features: {fraud_model.n_features_in_}")

def main():
    """Main function to create both models."""
    print("Creating sample models for AI Explainability Dashboard")
    print("=" * 50)
    
    # Check if models already exist
    if os.path.exists('random_forest_model.pkl'):
        response = input("Intrusion detection model already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping intrusion detection model creation.")
        else:
            create_intrusion_detection_model()
    else:
        create_intrusion_detection_model()
    
    if os.path.exists('FFD_rf_model.pkl'):
        response = input("Fraud detection model already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping fraud detection model creation.")
        else:
            create_fraud_detection_model()
    else:
        create_fraud_detection_model()
    
    # Verify both models
    verify_models()
    
    print("\nModel creation complete! You can now run the dashboard with: python app3.py")

if __name__ == "__main__":
    main()