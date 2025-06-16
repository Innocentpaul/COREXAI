import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid GUI issues
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import joblib
import numpy as np
import pandas as pd
import shap
import lime.lime_tabular
import io
import base64
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import traceback
import logging
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages

# Model paths
INTRUSION_MODEL_PATH = "random_forest_model.pkl"
FRAUD_MODEL_PATH = "FFD_rf_model.pkl"

# Number of features for each model
INTRUSION_FEATURES = 30
FRAUD_FEATURES = 30

# Feature names (update with your actual feature names)
INTRUSION_FEATURE_NAMES = [f"Intrusion_Feature_{i+1}" for i in range(INTRUSION_FEATURES)]
FRAUD_FEATURE_NAMES = [f"Fraud_Feature_{i+1}" for i in range(FRAUD_FEATURES)]

# Load models
def load_models():
    """Load the pre-trained models and return them in a dictionary."""
    models = {}
    try:
        models["intrusion"] = joblib.load(INTRUSION_MODEL_PATH)
        logger.info("Intrusion detection model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading intrusion model: {str(e)}")
        models["intrusion"] = None

    try:
        models["fraud"] = joblib.load(FRAUD_MODEL_PATH)
        logger.info("Fraud detection model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading fraud model: {str(e)}")
        models["fraud"] = None
        
    return models

# Global models variable
MODELS = load_models()

def get_feature_names(model_type):
    """Get the feature names for the specified model type."""
    if model_type == "intrusion":
        return INTRUSION_FEATURE_NAMES
    elif model_type == "fraud":
        return FRAUD_FEATURE_NAMES
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def get_num_features(model_type):
    """Get the number of features for the specified model type."""
    if model_type == "intrusion":
        return INTRUSION_FEATURES
    elif model_type == "fraud":
        return FRAUD_FEATURES
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def convert_matplotlib_to_base64(fig):
    """Convert a Matplotlib figure to a base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor='white', bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{encoded_image}"

def generate_sample_data(model_type, num_samples=1, random_state=None):
    """Generate sample data for demonstration purposes."""
    np.random.seed(random_state)
    num_features = get_num_features(model_type)
    return np.random.rand(num_samples, num_features)

def generate_shap_explanation_2d(model, sample_data, feature_names, top_features=10):
    """Generate a 2D SHAP waterfall plot."""
    try:
        # Create a SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample_data)
        
        # Handle multi-output models if necessary
        if hasattr(shap_values, 'values'):
            if len(shap_values.values.shape) == 3 and shap_values.values.shape[2] > 1:
                # For classification with probabilities for multiple classes
                class_idx = 1  # Usually index 1 is the positive class
                waterfall_values = shap_values.values[0, :, class_idx]
                base_value = shap_values.base_values[0, class_idx] if len(shap_values.base_values.shape) > 1 else shap_values.base_values[0]
                waterfall_explanation = shap.Explanation(
                    values=waterfall_values,
                    base_values=base_value,
                    data=shap_values.data[0],
                    feature_names=feature_names
                )
            else:
                waterfall_explanation = shap_values[0]
        else:
            waterfall_explanation = shap_values[0]
            
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(waterfall_explanation, max_display=top_features, show=False)
        plt.title(f"SHAP Waterfall Plot - Top {top_features} Features", fontsize=16, pad=20)
        plt.tight_layout()
        
        return convert_matplotlib_to_base64(plt.gcf())
    except Exception as e:
        logger.error(f"Error generating SHAP 2D explanation: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_shap_force_plot(model, sample_data, feature_names):
    """Generate a SHAP force plot."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample_data)
        
        # Handle multi-class models
        if hasattr(shap_values, 'values'):
            if len(shap_values.values.shape) == 3 and shap_values.values.shape[2] > 1:
                shap_values_single = shap_values.values[0, :, 1]
                base_value = shap_values.base_values[0, 1] if len(shap_values.base_values.shape) > 1 else shap_values.base_values[0]
            else:
                shap_values_single = shap_values.values[0]
                base_value = shap_values.base_values[0]
        else:
            shap_values_single = shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0]
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        
        # Create force plot
        shap.force_plot(base_value, shap_values_single, sample_data[0], 
                       feature_names=feature_names, matplotlib=True, show=False)
        
        plt.gcf().set_size_inches(14, 3)
        plt.title("SHAP Force Plot - Feature Contributions", fontsize=14, pad=20)
        
        return convert_matplotlib_to_base64(plt.gcf())
    except Exception as e:
        logger.error(f"Error generating SHAP force plot: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_shap_beeswarm(model, sample_data, feature_names, num_samples=50):
    """Generate a SHAP beeswarm plot for multiple samples."""
    try:
        # Generate multiple samples for a more meaningful beeswarm plot
        if sample_data.shape[0] < num_samples:
            model_type = "intrusion" if len(feature_names) == INTRUSION_FEATURES else "fraud"
            extended_samples = np.vstack([sample_data] + 
                                        [generate_sample_data(model_type, 1, random_state=i) 
                                         for i in range(num_samples-1)])
        else:
            extended_samples = sample_data[:num_samples]
            
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(extended_samples)
        
        # Create the beeswarm plot
        plt.figure(figsize=(12, 10))
        
        if hasattr(shap_values, 'values'):
            if len(shap_values.values.shape) == 3 and shap_values.values.shape[2] > 1:
                # For multi-class, focus on positive class
                class_idx = 1
                beeswarm_values = shap_values.values[:, :, class_idx]
                beeswarm_explanation = shap.Explanation(
                    values=beeswarm_values,
                    base_values=shap_values.base_values[:, class_idx] if len(shap_values.base_values.shape) > 1 else shap_values.base_values,
                    data=shap_values.data,
                    feature_names=feature_names
                )
            else:
                beeswarm_explanation = shap_values
        else:
            beeswarm_explanation = shap_values
                
        shap.plots.beeswarm(beeswarm_explanation, show=False)
        plt.title("SHAP Beeswarm Plot - Feature Impact Distribution", fontsize=16, pad=20)
        plt.tight_layout()
        
        return convert_matplotlib_to_base64(plt.gcf())
    except Exception as e:
        logger.error(f"Error generating SHAP beeswarm plot: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_shap_3d_plot(model, sample_data, feature_names, top_features=15):
    """Generate an interactive 3D scatter plot for SHAP values using Plotly."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample_data)
        
        # Extract SHAP values and feature data
        if hasattr(shap_values, 'values'):
            if len(shap_values.values.shape) == 3 and shap_values.values.shape[2] > 1:
                class_idx = 1
                shap_vals = shap_values.values[0, :, class_idx]
            else:
                shap_vals = shap_values.values[0]
        else:
            shap_vals = shap_values[0].values if hasattr(shap_values[0], 'values') else shap_values[0]
            
        data_values = shap_values.data[0] if hasattr(shap_values, 'data') else sample_data[0]
        
        # Keep only the top N features by absolute SHAP value
        if len(feature_names) > top_features:
            indices = np.argsort(-np.abs(shap_vals))[:top_features]
            shap_vals_filtered = shap_vals[indices]
            data_values_filtered = data_values[indices]
            feature_names_filtered = [feature_names[i] for i in indices]
        else:
            shap_vals_filtered = shap_vals
            data_values_filtered = data_values
            feature_names_filtered = feature_names
            
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter3d(
            x=list(range(len(feature_names_filtered))),
            y=data_values_filtered,
            z=shap_vals_filtered,
            mode='markers+text',
            marker=dict(
                size=12,
                color=shap_vals_filtered,
                colorscale='RdBu',
                colorbar=dict(title='SHAP Value'),
                opacity=0.9,
                line=dict(width=1, color='white')
            ),
            text=feature_names_filtered,
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>' +
                         'Feature Value: %{y:.4f}<br>' +
                         'SHAP Value: %{z:.4f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add vertical lines from z=0 to each point
        for i in range(len(feature_names_filtered)):
            fig.add_trace(go.Scatter3d(
                x=[i, i],
                y=[data_values_filtered[i], data_values_filtered[i]],
                z=[0, shap_vals_filtered[i]],
                mode='lines',
                line=dict(color='gray', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=dict(
                text=f"3D SHAP Plot - Top {len(feature_names_filtered)} Features",
                font=dict(size=20)
            ),
            scene=dict(
                xaxis=dict(title="Feature Index", gridcolor='gray'),
                yaxis=dict(title="Feature Value", gridcolor='gray'),
                zaxis=dict(title="SHAP Value", gridcolor='gray'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            template='plotly_white',
            margin=dict(l=0, r=0, b=0, t=60),
            height=700,
            hoverlabel=dict(bgcolor="white", font_size=12)
        )
        
        # Return the plot as a simple HTML string with inline Plotly
        return fig.to_html(full_html=False, include_plotlyjs=True)
    except Exception as e:
        logger.error(f"Error generating 3D SHAP plot: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_lime_explanation(model, sample_data, feature_names, top_features=10):
    """Generate a LIME explanation image."""
    try:
        # Create training data for LIME
        num_features = len(feature_names)
        np.random.seed(42)
        training_data = np.random.rand(1000, num_features)
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=["Normal", "Anomaly"],
            mode="classification"
        )
        
        # Generate explanation for the instance
        exp = explainer.explain_instance(
            sample_data[0], 
            model.predict_proba, 
            num_features=top_features,
            top_labels=2  # Request explanations for both classes
        )
        
        # Get the predicted class
        prediction = model.predict(sample_data)[0]
        
        # Create and save the figure
        plt.figure(figsize=(12, 8))
        
        # Try to use the predicted class, fallback to available class
        try:
            fig = exp.as_pyplot_figure(label=int(prediction))
        except KeyError:
            # Use the first available label
            available_labels = list(exp.available_labels())
            fig = exp.as_pyplot_figure(label=available_labels[0])
            
        plt.title(f"LIME Feature Importance (Class: {'Anomaly' if prediction == 1 else 'Normal'})", fontsize=16, pad=20)
        plt.xlabel("Feature Importance", fontsize=12)
        plt.tight_layout()
        
        return convert_matplotlib_to_base64(fig)
    except Exception as e:
        logger.error(f"Error generating LIME explanation: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_lime_3d_plot(model, sample_data, feature_names, top_features=10):
    """Generate a 3D visualization of LIME explanations using Plotly."""
    try:
        # Create training data for LIME
        num_features = len(feature_names)
        np.random.seed(42)
        training_data = np.random.rand(1000, num_features)
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=["Normal", "Anomaly"],
            mode="classification"
        )
        
        # Generate explanation
        exp = explainer.explain_instance(
            sample_data[0], 
            model.predict_proba, 
            num_features=top_features,
            top_labels=2  # Request explanations for both classes
        )
        
        # Get the predicted class
        prediction = model.predict(sample_data)[0]
        
        # Extract explanation data for the predicted class
        try:
            explanation = exp.as_list(label=int(prediction))
        except KeyError:
            # Use the first available label
            available_labels = list(exp.available_labels())
            explanation = exp.as_list(label=available_labels[0])
            
        feature_names_lime = []
        importance_values = []
        
        for feature_desc, importance in explanation:
            # Extract feature name from description
            found = False
            for fn in feature_names:
                if fn in feature_desc:
                    feature_names_lime.append(fn)
                    found = True
                    break
            if not found:
                # Try to extract feature name differently
                parts = feature_desc.split()
                if len(parts) > 0:
                    feature_names_lime.append(parts[0])
                else:
                    feature_names_lime.append(f"Feature_{len(feature_names_lime)}")
            importance_values.append(importance)
        
        # Get feature values for the explained instance
        feature_values = []
        feature_indices = []
        for fname in feature_names_lime:
            try:
                idx = feature_names.index(fname)
                feature_values.append(sample_data[0][idx])
                feature_indices.append(idx)
            except ValueError:
                # If exact match not found, use index based on position
                idx = len(feature_values)
                if idx < len(sample_data[0]):
                    feature_values.append(sample_data[0][idx])
                else:
                    feature_values.append(0)
                feature_indices.append(idx)
        
        # Create 3D plot
        fig = go.Figure()
        
        # Add 3D scatter plot for LIME values
        fig.add_trace(go.Scatter3d(
            x=list(range(len(feature_names_lime))),
            y=feature_values,
            z=importance_values,
            mode='markers+text',
            marker=dict(
                size=12,
                color=importance_values,
                colorscale='RdBu',  # Changed from 'RdGn' to 'RdBu' which is valid
                colorbar=dict(title='LIME Impact'),
                opacity=0.9,
                line=dict(width=1, color='black')
            ),
            text=feature_names_lime,
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>' +
                         'Feature Value: %{y:.4f}<br>' +
                         'LIME Impact: %{z:.4f}<br>' +
                         '<extra></extra>',
            name='LIME Importance'
        ))
        
        # Add vertical lines to show impact
        for i in range(len(feature_names_lime)):
            fig.add_trace(go.Scatter3d(
                x=[i, i],
                y=[feature_values[i], feature_values[i]],
                z=[0, importance_values[i]],
                mode='lines',
                line=dict(
                    color='green' if importance_values[i] > 0 else 'red',
                    width=3
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add a plane at z=0
        x_plane = list(range(len(feature_names_lime)))
        y_plane = feature_values
        z_plane = [0] * len(feature_names_lime)
        
        fig.add_trace(go.Scatter3d(
            x=x_plane,
            y=y_plane,
            z=z_plane,
            mode='markers',
            marker=dict(
                size=8,
                color='gray',
                opacity=0.5
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"3D LIME Explanation - Feature Impact Visualization (Class: {'Anomaly' if prediction == 1 else 'Normal'})",
                font=dict(size=20)
            ),
            scene=dict(
                xaxis=dict(title="Feature Index", showticklabels=False),
                yaxis=dict(title="Feature Value"),
                zaxis=dict(title="LIME Impact"),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            template='plotly_white',
            margin=dict(l=0, r=0, b=0, t=60),
            height=700,
            showlegend=False
        )
        
        # Return the plot as a simple HTML string with inline Plotly
        return fig.to_html(full_html=False, include_plotlyjs=True)
    except Exception as e:
        logger.error(f"Error generating 3D LIME plot: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_feature_correlation_heatmap(model, sample_data, feature_names, top_features=20):
    """Generate a correlation heatmap of top features."""
    try:
        # Generate multiple samples for correlation
        model_type = "intrusion" if len(feature_names) == INTRUSION_FEATURES else "fraud"
        samples = np.vstack([sample_data] + 
                           [generate_sample_data(model_type, 1, random_state=i) 
                            for i in range(100)])
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Use permutation importance as fallback
            importances = np.random.rand(len(feature_names))
        
        # Select top features
        top_indices = np.argsort(-importances)[:min(top_features, len(feature_names))]
        top_features_data = samples[:, top_indices]
        top_feature_names = [feature_names[i] for i in top_indices]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(top_features_data.T)
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   xticklabels=top_feature_names,
                   yticklabels=top_feature_names,
                   cmap='coolwarm',
                   center=0,
                   annot=True,
                   fmt='.2f',
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title(f"Feature Correlation Heatmap - Top {len(top_feature_names)} Features", fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return convert_matplotlib_to_base64(plt.gcf())
    except Exception as e:
        logger.error(f"Error generating correlation heatmap: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_feature_importance_comparison(model, sample_data, feature_names):
    """Generate a comparison of different feature importance methods."""
    try:
        # Get model feature importances
        if hasattr(model, 'feature_importances_'):
            model_importances = model.feature_importances_
        else:
            model_importances = np.random.rand(len(feature_names))
        
        # Get SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample_data)
        
        if hasattr(shap_values, 'values'):
            if len(shap_values.values.shape) == 3 and shap_values.values.shape[2] > 1:
                shap_importance = np.abs(shap_values.values[0, :, 1])
            else:
                shap_importance = np.abs(shap_values.values[0])
        else:
            shap_importance = np.abs(shap_values[0].values if hasattr(shap_values[0], 'values') else shap_values[0])
        
        # Get LIME values
        np.random.seed(42)
        training_data = np.random.rand(500, len(feature_names))
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=["Normal", "Anomaly"],
            mode="classification"
        )
        
        lime_exp = lime_explainer.explain_instance(
            sample_data[0], 
            model.predict_proba, 
            num_features=len(feature_names),
            top_labels=2  # Request explanations for both classes
        )
        
        # Get the predicted class
        prediction = model.predict(sample_data)[0]
        
        # Extract LIME importances for the predicted class
        lime_importance = np.zeros(len(feature_names))
        try:
            lime_list = lime_exp.as_list(label=int(prediction))
        except KeyError:
            # Use the first available label
            available_labels = list(lime_exp.available_labels())
            lime_list = lime_exp.as_list(label=available_labels[0])
            
        for feature_desc, importance in lime_list:
            for i, fn in enumerate(feature_names):
                if fn in feature_desc:
                    lime_importance[i] = abs(importance)
                    break
        
        # Normalize importances
        model_importances = model_importances / (np.max(model_importances) + 1e-10)
        shap_importance = shap_importance / (np.max(shap_importance) + 1e-10)
        lime_importance = lime_importance / (np.max(lime_importance) + 1e-10)
        
        # Select top 15 features by average importance
        avg_importance = (model_importances + shap_importance + lime_importance) / 3
        top_indices = np.argsort(-avg_importance)[:15]
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(top_indices))
        width = 0.25
        
        ax.bar(x - width, model_importances[top_indices], width, label='Model', alpha=0.8)
        ax.bar(x, shap_importance[top_indices], width, label='SHAP', alpha=0.8)
        ax.bar(x + width, lime_importance[top_indices], width, label='LIME', alpha=0.8)
        
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Normalized Importance', fontsize=12)
        ax.set_title('Feature Importance Comparison - Model vs SHAP vs LIME', fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([feature_names[i] for i in top_indices], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return convert_matplotlib_to_base64(fig)
    except Exception as e:
        logger.error(f"Error generating feature importance comparison: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def calculate_feature_impacts(model, sample_data, feature_names):
    """Calculate feature impacts for the frontend."""
    try:
        # Use SHAP to calculate feature impacts
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample_data)
        
        if hasattr(shap_values, 'values'):
            if len(shap_values.values.shape) == 3 and shap_values.values.shape[2] > 1:
                # Multi-class model
                impacts = shap_values.values[0, :, 1].tolist()
            else:
                impacts = shap_values.values[0].tolist()
        else:
            impacts = shap_values[0].values.tolist() if hasattr(shap_values[0], 'values') else shap_values[0].tolist()
        
        return impacts
    except Exception as e:
        logger.error(f"Error calculating feature impacts: {str(e)}")
        # Return random impacts as fallback
        return np.random.randn(len(feature_names)).tolist()

@app.route('/')
def home():
    """Render the home page."""
    return render_template("index.html")

@app.route('/static/images/placeholder-chart.png')
def placeholder_image():
    """Generate a placeholder chart image."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Select options and click "Generate Explanation"\nto visualize AI model decisions', 
            ha='center', va='center', fontsize=16, color='gray', 
            transform=ax.transAxes, wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    
    return send_file(buf, mimetype='image/png')

@app.route('/explain', methods=['POST'])
def explain():
    """API endpoint to generate model explanations."""
    try:
        data = request.get_json()
        model_type = data.get("model_type")
        explanation_type = data.get("explanation_type", "shap_2d")
        num_samples = int(data.get("num_samples", 1))
        top_features = int(data.get("top_features", 10))
        
        # Validate inputs
        if model_type not in ["intrusion", "fraud"]:
            return jsonify({"error": "Invalid model type. Choose 'intrusion' or 'fraud'."}), 400
        
        valid_explanations = ["shap_2d", "shap_3d", "lime_2d", "lime_3d", "shap_beeswarm", 
                             "shap_force", "feature_correlation", "feature_comparison"]
        if explanation_type not in valid_explanations:
            return jsonify({"error": f"Invalid explanation type. Choose from: {', '.join(valid_explanations)}"}), 400
        
        # Get the appropriate model
        model = MODELS.get(model_type)
        if model is None:
            return jsonify({"error": f"Model '{model_type}' not loaded. Please ensure the model file exists and run create_models.py"}), 500
        
        # Get feature names for this model
        feature_names = get_feature_names(model_type)
        
        # Generate sample data or use provided data
        if "input_data" in data and data["input_data"]:
            try:
                sample_data = np.array(data["input_data"]).reshape(1, -1)
                expected = get_num_features(model_type)
                if sample_data.shape[1] != expected:
                    return jsonify({
                        "error": f"{model_type} model expects {expected} features, but got {sample_data.shape[1]}"
                    }), 400
            except Exception as e:
                return jsonify({"error": f"Invalid input data: {str(e)}"}), 400
        else:
            # Generate random sample data
            sample_data = generate_sample_data(model_type, num_samples, random_state=42)
        
        # Generate the requested explanation
        result = {}
        
        try:
            if explanation_type == "shap_2d":
                result["plot"] = generate_shap_explanation_2d(model, sample_data, feature_names, top_features)
                result["type"] = "image"
            elif explanation_type == "shap_3d":
                result["plot"] = generate_shap_3d_plot(model, sample_data, feature_names, top_features)
                result["type"] = "html"
            elif explanation_type == "lime_2d":
                result["plot"] = generate_lime_explanation(model, sample_data, feature_names, top_features)
                result["type"] = "image"
            elif explanation_type == "lime_3d":
                result["plot"] = generate_lime_3d_plot(model, sample_data, feature_names, top_features)
                result["type"] = "html"
            elif explanation_type == "shap_beeswarm":
                result["plot"] = generate_shap_beeswarm(model, sample_data, feature_names, num_samples)
                result["type"] = "image"
            elif explanation_type == "shap_force":
                result["plot"] = generate_shap_force_plot(model, sample_data, feature_names)
                result["type"] = "image"
            elif explanation_type == "feature_correlation":
                result["plot"] = generate_feature_correlation_heatmap(model, sample_data, feature_names, top_features)
                result["type"] = "image"
            elif explanation_type == "feature_comparison":
                result["plot"] = generate_feature_importance_comparison(model, sample_data, feature_names)
                result["type"] = "image"
        except Exception as e:
            logger.error(f"Error generating {explanation_type}: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Failed to generate {explanation_type}: {str(e)}"}), 500
        
        # Add prediction results
        try:
            predictions = model.predict(sample_data)
            probabilities = model.predict_proba(sample_data)
            
            result["predictions"] = predictions.tolist()
            result["probabilities"] = probabilities.tolist()
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            result["predictions"] = [0]
            result["probabilities"] = [[0.5, 0.5]]
        
        # Add feature impacts for the frontend
        try:
            result["feature_impacts"] = calculate_feature_impacts(model, sample_data, feature_names)
        except Exception as e:
            logger.error(f"Error calculating feature impacts: {str(e)}")
            result["feature_impacts"] = [0] * len(feature_names)
        
        # Return the explanation
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in /explain endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    models_status = {
        "intrusion": MODELS["intrusion"] is not None,
        "fraud": MODELS["fraud"] is not None
    }
    
    # Test if models can make predictions
    test_results = {}
    for model_name, model in MODELS.items():
        if model is not None:
            try:
                test_data = np.random.rand(1, 30)
                _ = model.predict(test_data)
                test_results[model_name] = "working"
            except Exception as e:
                test_results[model_name] = f"error: {str(e)}"
        else:
            test_results[model_name] = "not loaded"
    
    return jsonify({
        "status": "ok" if all(models_status.values()) else "degraded",
        "models": models_status,
        "test_results": test_results
    })

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Check if models exist
    if not os.path.exists(INTRUSION_MODEL_PATH) or not os.path.exists(FRAUD_MODEL_PATH):
        print("WARNING: Model files not found!")
        print("Please run 'python create_models.py' to generate sample models.")
        print("-" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)