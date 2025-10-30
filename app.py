"""
Simple ML Model Monitoring Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.base import BaseEstimator
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tempfile
import time

# Page config
st.set_page_config(
    page_title="ML Model Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'performance_results' not in st.session_state:
    st.session_state.performance_results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
# New: Model comparison
if 'models' not in st.session_state:
    st.session_state.models = []  # List of {name, model, model_type}
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
# New: Dataset analysis
if 'datasets' not in st.session_state:
    st.session_state.datasets = []  # List of {name, data, target_column}
if 'dataset_analysis' not in st.session_state:
    st.session_state.dataset_analysis = None

def load_model(file_path):
    """Load model from file."""
    try:
        # Try pickle first
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
        except:
            # Fallback to joblib
            model = joblib.load(file_path)
        
        # Check if it's a valid model
        if hasattr(model, 'predict'):
            return model
        else:
            st.error("Uploaded file doesn't contain a valid model with predict method.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def extract_model_metadata(model):
    """Extract comprehensive metadata from a model."""
    metadata = {
        'class_name': type(model).__name__,
        'model_type': None,  # Will be auto-detected
        'algorithm': None,
        'n_features': None,
        'feature_names': None,
        'n_classes': None,
        'class_names': None,
        'parameters': {},
        'is_pipeline': False,
        'preprocessing_steps': []
    }
    
    try:
        # Detect if it's a pipeline
        if hasattr(model, 'steps'):
            metadata['is_pipeline'] = True
            # Extract preprocessing steps
            for step_name, step_obj in model.steps[:-1]:  # All but last step
                metadata['preprocessing_steps'].append({
                    'name': step_name,
                    'type': type(step_obj).__name__
                })
            # Get the actual model (last step)
            actual_model = model.steps[-1][1]
        else:
            actual_model = model
        
        # Update class name with actual model
        metadata['algorithm'] = type(actual_model).__name__
        
        # Auto-detect model type based on common sklearn patterns
        classification_indicators = [
            'Classifier', 'SVC', 'SVM', 'LogisticRegression', 
            'NaiveBayes', 'KNeighborsClassifier', 'DecisionTreeClassifier',
            'RandomForestClassifier', 'GradientBoostingClassifier',
            'AdaBoostClassifier', 'XGBClassifier', 'LGBMClassifier',
            'CatBoostClassifier', 'MLPClassifier'
        ]
        
        regression_indicators = [
            'Regressor', 'SVR', 'LinearRegression', 'Ridge', 'Lasso',
            'ElasticNet', 'KNeighborsRegressor', 'DecisionTreeRegressor',
            'RandomForestRegressor', 'GradientBoostingRegressor',
            'AdaBoostRegressor', 'XGBRegressor', 'LGBMRegressor',
            'CatBoostRegressor', 'MLPRegressor'
        ]
        
        model_name = metadata['algorithm']
        
        if any(indicator in model_name for indicator in classification_indicators):
            metadata['model_type'] = 'classification'
        elif any(indicator in model_name for indicator in regression_indicators):
            metadata['model_type'] = 'regression'
        
        # Extract number of features
        if hasattr(actual_model, 'n_features_in_'):
            metadata['n_features'] = actual_model.n_features_in_
        elif hasattr(actual_model, 'coef_'):
            if len(actual_model.coef_.shape) > 1:
                metadata['n_features'] = actual_model.coef_.shape[1]
            else:
                metadata['n_features'] = len(actual_model.coef_)
        
        # Extract feature names if available
        if hasattr(actual_model, 'feature_names_in_'):
            metadata['feature_names'] = list(actual_model.feature_names_in_)
        elif hasattr(model, 'feature_names_in_'):
            metadata['feature_names'] = list(model.feature_names_in_)
        
        # Extract number of classes for classification
        if hasattr(actual_model, 'n_classes_'):
            metadata['n_classes'] = actual_model.n_classes_
        elif hasattr(actual_model, 'classes_'):
            metadata['n_classes'] = len(actual_model.classes_)
            metadata['class_names'] = list(actual_model.classes_)
        
        # Extract key parameters
        if hasattr(actual_model, 'get_params'):
            all_params = actual_model.get_params()
            # Filter important parameters (exclude None and default values)
            important_params = ['n_estimators', 'max_depth', 'learning_rate', 
                              'C', 'gamma', 'kernel', 'alpha', 'l1_ratio',
                              'hidden_layer_sizes', 'activation', 'solver',
                              'max_iter', 'n_neighbors', 'metric']
            
            for param in important_params:
                if param in all_params and all_params[param] is not None:
                    metadata['parameters'][param] = all_params[param]
        
        # Additional model-specific metadata
        if hasattr(actual_model, 'n_estimators'):
            metadata['n_estimators'] = actual_model.n_estimators
        if hasattr(actual_model, 'max_depth'):
            metadata['max_depth'] = actual_model.max_depth
        if hasattr(actual_model, 'feature_importances_'):
            metadata['has_feature_importance'] = True
        
    except Exception as e:
        # If extraction fails, return basic metadata
        pass
    
    return metadata

def display_model_metadata(metadata):
    """Display model metadata in a nice format."""
    st.markdown("### 📋 Model Information (Auto-Detected)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Information")
        st.info(f"**Algorithm**: {metadata['algorithm']}")
        
        if metadata['model_type']:
            model_type_emoji = "📊" if metadata['model_type'] == 'classification' else "📈"
            st.info(f"**Type**: {model_type_emoji} {metadata['model_type'].title()}")
        
        if metadata['is_pipeline']:
            st.info("**Pipeline**: Yes ✅")
            if metadata['preprocessing_steps']:
                steps_str = " → ".join([s['type'] for s in metadata['preprocessing_steps']])
                st.caption(f"Steps: {steps_str}")
        
        if metadata['n_features']:
            st.info(f"**Features**: {metadata['n_features']}")
        
        if metadata['n_classes']:
            st.info(f"**Classes**: {metadata['n_classes']}")
    
    with col2:
        st.markdown("#### Parameters")
        if metadata['parameters']:
            for param, value in metadata['parameters'].items():
                st.caption(f"**{param}**: {value}")
        else:
            st.caption("No parameters extracted")
        
        if metadata.get('has_feature_importance'):
            st.success("✅ Supports feature importance")
    
    # Feature names if available
    if metadata['feature_names']:
        with st.expander("🔍 Feature Names"):
            st.write(", ".join(metadata['feature_names'][:20]))  # Show first 20
            if len(metadata['feature_names']) > 20:
                st.caption(f"... and {len(metadata['feature_names']) - 20} more")
    
    # Class names if available
    if metadata['class_names']:
        with st.expander("🏷️ Class Names"):
            st.write(", ".join([str(c) for c in metadata['class_names']]))
    
    return metadata

def calculate_classification_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    try:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        return metrics
    except Exception as e:
        st.error(f"Error calculating classification metrics: {str(e)}")
        return None

def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    try:
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return metrics
    except Exception as e:
        st.error(f"Error calculating regression metrics: {str(e)}")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">ML Model Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    nav_options = [
        "Home",
        "Model Upload", 
        "Dataset Upload",
        "Results & Analytics",
        "Model Comparison",
        "Dataset Analysis",
        "Data Generation",
        "Performance Testing",
        "Model Explainability"
    ]
    
    # Get the current page index
    current_index = nav_options.index(st.session_state.current_page) if st.session_state.current_page in nav_options else 0
    
    selected_page = st.sidebar.radio(
        "Choose a page:",
        nav_options,
        index=current_index,
        key="page_nav_radio"
    )
    
    # Only update if the radio button selection is different from current page
    # This prevents the radio from overriding button clicks
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
    
    # Route to pages
    if st.session_state.current_page == "Home":
        show_home_page()
    elif st.session_state.current_page == "Model Upload":
        show_model_upload_page()
    elif st.session_state.current_page == "Dataset Upload":
        show_dataset_upload_page()
    elif st.session_state.current_page == "Results & Analytics":
        show_results_page()
    elif st.session_state.current_page == "Model Comparison":
        show_model_comparison_page()
    elif st.session_state.current_page == "Dataset Analysis":
        show_dataset_analysis_page()
    elif st.session_state.current_page == "Data Generation":
        show_data_generation_page()
    elif st.session_state.current_page == "Performance Testing":
        show_performance_testing_page()
    elif st.session_state.current_page == "Model Explainability":
        show_shap_page()

def show_home_page():
    """Display the home page."""
    
    st.markdown("## Welcome to the ML Model Monitoring Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About This Dashboard
        
        A comprehensive tool to upload your machine learning models and test their performance with advanced analytics.
        
        ### Workflow:
        
        1. **Model Upload**: Upload your trained model (.pkl file)
        2. **Dataset Upload**: Upload your test dataset (CSV file) 
        3. **Results & Analytics**: View performance metrics and visualizations
        4. **Model Comparison**: Compare multiple models on the same dataset 🆕
        5. **Dataset Analysis**: Analyze multiple datasets for quality and imbalances 🆕
        6. **Data Generation**: Generate synthetic test data
        7. **Performance Testing**: Test model latency and throughput
        8. **Model Explainability**: Understand model predictions
        
        ### Supported Models
        
        - Classification models (Random Forest, SVM, etc.)
        - Regression models (Linear Regression, etc.)
        - Any scikit-learn compatible model
        """)
    
    with col2:
        st.markdown("### Current Status")
        
        # Model status
        if st.session_state.model is not None:
            st.success("Model Loaded")
            if st.session_state.model_type:
                st.info(f"Type: {st.session_state.model_type.title()}")
        else:
            st.warning("No Model Loaded")
        
        # Dataset status
        if st.session_state.dataset is not None:
            st.success("Dataset Loaded")
            st.info(f"Rows: {len(st.session_state.dataset)}")
            if st.session_state.target_column:
                st.info(f"Target: {st.session_state.target_column}")
        else:
            st.warning("No Dataset Loaded")
        
        # Synthetic data status
        if st.session_state.synthetic_data is not None:
            st.success("Synthetic Data Generated")
            st.info(f"Samples: {len(st.session_state.synthetic_data)}")
        else:
            st.warning("No Synthetic Data")
        
        # Next steps
        st.markdown("### Next Steps")
        if st.session_state.model is None:
            if st.button("Upload Model", key="home_model_btn"):
                st.session_state.current_page = "Model Upload"
                st.rerun()
        elif st.session_state.dataset is None:
            if st.button("Upload Dataset", key="home_dataset_btn"):
                st.session_state.current_page = "Dataset Upload"
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View Results", key="home_results_btn"):
                    st.session_state.current_page = "Results & Analytics"
                    st.rerun()
            with col2:
                if st.button("SHAP Analysis", key="home_shap_btn"):
                    st.session_state.current_page = "Model Explainability"
                    st.rerun()

def show_model_upload_page():
    """Display the model upload page with automatic metadata extraction."""
    
    st.markdown("## Model Upload")
    st.markdown("Upload your trained machine learning model (.pkl file). Model details will be automatically extracted!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a model file",
        type=['pkl'],
        help="Upload a scikit-learn model saved as .pkl file"
    )
    
    if uploaded_file is not None:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
        
        # Load the model
        with st.spinner("Loading model and extracting metadata..."):
            model = load_model(temp_path)
        
        if model is not None:
            st.session_state.model = model
            st.success("✅ Model loaded successfully!")
            
            # Extract comprehensive metadata
            metadata = extract_model_metadata(model)
            
            # Display metadata
            display_model_metadata(metadata)
            
            # Auto-detect model type
            if metadata['model_type']:
                st.session_state.model_type = metadata['model_type']
                st.success(f"🤖 Auto-detected model type: **{metadata['model_type'].title()}**")
                
                # Store metadata in session state
                if 'model_metadata' not in st.session_state:
                    st.session_state.model_metadata = {}
                st.session_state.model_metadata = metadata
                
                # Show confirmation
                st.markdown("---")
                st.markdown("### ✅ Model Ready")
                st.info(f"**{metadata['algorithm']}** model is ready for evaluation!")
                
                # Next steps
                st.markdown("### Next Steps")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Upload Dataset", key="goto_dataset_btn", use_container_width=True):
                        st.session_state.current_page = "Dataset Upload"
                        st.rerun()
                with col2:
                    if st.button("🏆 Compare Models", key="goto_compare_btn", use_container_width=True):
                        st.session_state.current_page = "Model Comparison"
                        st.rerun()
                
            else:
                # Manual selection fallback
                st.warning("⚠️ Could not auto-detect model type. Please select manually:")
                
                model_type = st.radio(
                    "Select model type:",
                    ["classification", "regression"],
                    key="model_type_selector"
                )
                
                if model_type:
                    st.session_state.model_type = model_type
                    st.success(f"Model type set to: {model_type.title()}")
                    
                    # Next steps
                    st.markdown("### Next Steps")
                    if st.button("Upload Dataset", key="goto_dataset_btn_manual"):
                        st.session_state.current_page = "Dataset Upload"
                        st.rerun()
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
    
    # Current model status
    if st.session_state.model is not None:
        st.markdown("---")
        st.markdown("### 📦 Current Model Status")
        st.success("✅ Model loaded and ready")
        
        if st.session_state.model_type:
            st.info(f"**Type**: {st.session_state.model_type.title()}")
        
        # Show metadata if available
        if hasattr(st.session_state, 'model_metadata') and st.session_state.model_metadata:
            with st.expander("View Model Details"):
                metadata = st.session_state.model_metadata
                st.json({
                    'Algorithm': metadata['algorithm'],
                    'Type': metadata['model_type'],
                    'Features': metadata['n_features'],
                    'Classes': metadata['n_classes'] if metadata['n_classes'] else 'N/A',
                    'Is Pipeline': metadata['is_pipeline'],
                    'Parameters': metadata['parameters']
                })

def show_dataset_upload_page():
    """Display the dataset upload page."""
    
    st.markdown("## Dataset Upload")
    st.markdown("Upload your test dataset with true labels to evaluate model performance.")
    
    # Check if model is loaded
    if st.session_state.model is None:
        st.error("Please upload a model first")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file with your test dataset",
        type=['csv'],
        help="Upload your test dataset containing features and true labels"
    )
    
    if uploaded_file is not None:
        try:
            # Load dataset
            dataset = pd.read_csv(uploaded_file)
            st.session_state.dataset = dataset
            
            st.success(f"Dataset loaded successfully! Shape: {dataset.shape}")
            
            # Dataset preview
            st.markdown("### Dataset Preview")
            st.dataframe(dataset.head())
            
            # Dataset info
            st.markdown("### Dataset Info")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", len(dataset))
                st.metric("Columns", len(dataset.columns))
            with col2:
                st.metric("Missing Values", dataset.isnull().sum().sum())
                st.metric("Duplicates", dataset.duplicated().sum())
            
            # Feature information
            st.markdown("### Feature Information")
            st.info(f"Available columns: {', '.join(dataset.columns.tolist())}")
            
            # Target column selection
            st.markdown("### Target Column Selection")
            target_column = st.selectbox(
                "Select the target column (true labels):",
                options=dataset.columns.tolist(),
                key="target_column_selector"
            )
            
            if target_column:
                st.session_state.target_column = target_column
                st.info(f"Target column set to: {target_column}")
                
                # Show feature columns that will be used for prediction
                feature_columns = [col for col in dataset.columns if col != target_column]
                st.session_state.feature_columns = feature_columns  # Store feature columns in session state
                st.info(f"Features for prediction: {', '.join(feature_columns)}")
                
                # Target distribution
                st.markdown("### Target Distribution")
                if st.session_state.model_type == "classification":
                    value_counts = dataset[target_column].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title="Target Class Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(dataset, x=target_column, title="Target Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Next steps
                st.markdown("### Next Steps")
                if st.button("View Results", key="goto_results_btn"):
                    st.session_state.current_page = "Results & Analytics"
                    st.rerun()
                
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    # Current dataset status
    if st.session_state.dataset is not None:
        st.markdown("### Current Dataset Status")
        st.success(f"Dataset loaded: {len(st.session_state.dataset)} rows")
        if st.session_state.target_column:
            st.info(f"Target: {st.session_state.target_column}")

def show_results_page():
    """Display the results and analytics page."""
    
    st.markdown("## Results & Analytics")
    
    # Check if both model and dataset are loaded
    if st.session_state.model is None:
        st.error("Please upload a model first")
        return
    
    if st.session_state.dataset is None:
        st.error("Please upload a dataset first")
        return
    
    if st.session_state.target_column is None:
        st.error("Please select a target column first")
        return
    
    # Prepare data
    dataset = st.session_state.dataset
    target_column = st.session_state.target_column
    feature_columns = [col for col in dataset.columns if col != target_column]
    
    X = dataset[feature_columns]
    y_true = dataset[target_column]
    
    # Show data preparation info
    st.markdown("### Data Preparation")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Features used: {len(feature_columns)}")
        st.info(f"Samples: {len(X)}")
    with col2:
        st.info(f"Target column: {target_column}")
        st.info(f"Model type: {st.session_state.model_type}")
    
    # Evaluate model button
    if st.button("Evaluate Model", type="primary", key="evaluate_btn"):
        with st.spinner("Evaluating model performance..."):
            try:
                # Make predictions
                y_pred = st.session_state.model.predict(X)
                
                # Calculate metrics based on model type
                if st.session_state.model_type == "classification":
                    metrics = calculate_classification_metrics(y_true, y_pred)
                else:
                    metrics = calculate_regression_metrics(y_true, y_pred)
                
                if metrics is not None:
                    st.session_state.results = {
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'metrics': metrics,
                        'model_type': st.session_state.model_type
                    }
                    st.success("Model evaluation completed!")
                
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
    
    # Display results if available
    if st.session_state.results is not None:
        results = st.session_state.results
        
        if results['model_type'] == "classification":
            display_classification_results(results)
        else:
            display_regression_results(results)
    
    # Next steps
    if st.session_state.results is not None:
        st.markdown("### Next Steps")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Synthetic Data", key="goto_generation_btn"):
                st.session_state.current_page = "Data Generation"
                st.rerun()
        with col2:
            if st.button("SHAP Analysis", key="goto_shap_btn"):
                st.session_state.current_page = "Model Explainability"
                st.rerun()

def display_classification_results(results):
    """Display classification results."""
    
    metrics = results['metrics']
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    st.markdown("### Classification Metrics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1']:.3f}")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Confusion Matrix")
        cm = metrics['confusion_matrix']
        fig = px.imshow(cm, text_auto=True, aspect="auto", 
                       title="Confusion Matrix",
                       labels={'x': 'Predicted', 'y': 'Actual'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Prediction Distribution")
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        true_counts = pd.Series(y_true).value_counts().sort_index()
        
        fig = go.Figure()
        fig.add_bar(x=pred_counts.index, y=pred_counts.values, 
                   name='Predicted', opacity=0.7)
        fig.add_bar(x=true_counts.index, y=true_counts.values, 
                   name='Actual', opacity=0.7)
        fig.update_layout(title="Predicted vs Actual Distribution",
                         barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.markdown("### Classification Report")
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    st.dataframe(report_df.round(3))

def display_regression_results(results):
    """Display regression results."""
    
    metrics = results['metrics']
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    st.markdown("### Regression Metrics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{metrics['r2']:.3f}")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.3f}")
    with col3:
        st.metric("MAE", f"{metrics['mae']:.3f}")
    with col4:
        st.metric("MAPE", f"{metrics['mape']:.1f}%")
    
    # Scatter plot
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Predicted vs Actual")
        fig = px.scatter(x=y_true, y=y_pred, 
                        title="Predicted vs Actual Values",
                        labels={'x': 'Actual', 'y': 'Predicted'})
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_shape(type="line", x0=min_val, y0=min_val, 
                     x1=max_val, y1=max_val, 
                     line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Residuals")
        residuals = y_true - y_pred
        fig = px.scatter(x=y_pred, y=residuals, 
                        title="Residuals vs Predicted",
                        labels={'x': 'Predicted', 'y': 'Residuals'})
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    # Error distribution
    st.markdown("### Error Distribution")
    fig = px.histogram(x=residuals, title="Distribution of Residuals")
    st.plotly_chart(fig, use_container_width=True)

def show_data_generation_page():
    """Display the data generation page."""
    
    st.markdown("## Data Generation")
    st.markdown("Generate synthetic test data for performance testing.")
    
    if st.session_state.model is None:
        st.error("Please upload a model first")
        return
    
    if st.session_state.dataset is None:
        st.error("Please upload a dataset first to understand the feature structure")
        return
    
    # Get feature information from existing dataset
    target_column = st.session_state.target_column
    feature_columns = [col for col in st.session_state.dataset.columns if col != target_column]
    
    st.markdown("### Data Generation Based on Your Dataset")
    st.info(f"Generating data with {len(feature_columns)} features: {', '.join(feature_columns)}")
    
    # Data generation parameters
    st.markdown("### Generation Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        num_samples = st.number_input("Number of samples:", min_value=10, max_value=10000, value=1000)
        random_seed = st.number_input("Random seed:", min_value=0, value=42)
    
    with col2:
        noise_level = st.slider("Noise level:", 0.0, 1.0, 0.1)
        distribution = st.selectbox("Distribution:", ["normal", "uniform", "exponential"])
    
    if st.button("Generate Data", key="generate_data_btn"):
        with st.spinner("Generating synthetic data..."):
            try:
                # Generate synthetic data based on existing dataset statistics
                np.random.seed(random_seed)
                
                original_data = st.session_state.dataset[feature_columns]
                synthetic_data = {}
                
                for col in feature_columns:
                    # Analyze the original column
                    col_data = original_data[col]
                    
                    if pd.api.types.is_numeric_dtype(col_data):
                        # For numeric columns, use statistics from original data
                        mean_val = col_data.mean()
                        std_val = col_data.std()
                        min_val = col_data.min()
                        max_val = col_data.max()
                        
                        if distribution == "normal":
                            generated = np.random.normal(mean_val, std_val, num_samples)
                        elif distribution == "uniform":
                            generated = np.random.uniform(min_val, max_val, num_samples)
                        else:  # exponential
                            generated = np.random.exponential(abs(mean_val) if mean_val != 0 else 1, num_samples)
                        
                        # Add noise
                        noise = np.random.normal(0, noise_level * std_val, num_samples)
                        synthetic_data[col] = generated + noise
                    
                    else:
                        # For categorical columns, sample from existing values
                        unique_values = col_data.unique()
                        synthetic_data[col] = np.random.choice(unique_values, num_samples)
                
                synthetic_df = pd.DataFrame(synthetic_data)
                st.session_state.synthetic_data = synthetic_df
                
                st.success(f"Generated {num_samples} synthetic samples!")
                
                # Display preview
                st.markdown("### Generated Data Preview")
                st.dataframe(synthetic_df.head())
                
                # Comparison with original data
                st.markdown("### Comparison with Original Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Data Statistics:**")
                    st.dataframe(original_data.describe())
                
                with col2:
                    st.markdown("**Generated Data Statistics:**")
                    st.dataframe(synthetic_df.describe())
                
                # Next steps
                st.markdown("### Next Steps")
                if st.button("Performance Testing", key="goto_performance_btn"):
                    st.session_state.current_page = "Performance Testing"
                    st.rerun()
                
            except Exception as e:
                st.error(f"Error generating data: {str(e)}")
    
    # Display current synthetic data
    if st.session_state.synthetic_data is not None:
        st.markdown("### Current Synthetic Data")
        st.success(f"Generated dataset with {len(st.session_state.synthetic_data)} samples")

def show_performance_testing_page():
    """Display the performance testing page."""
    
    st.markdown("## Performance Testing")
    st.markdown("Test your model's latency, throughput, and error rates using synthetic data.")
    
    if st.session_state.model is None:
        st.error("Please upload a model first")
        return
    
    if st.session_state.synthetic_data is None:
        st.error("Please generate synthetic data first")
        return
    
    # Performance testing parameters
    st.markdown("### Test Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        num_predictions = st.number_input("Number of predictions:", min_value=10, max_value=10000, 
                                        value=min(1000, len(st.session_state.synthetic_data)))
        batch_size = st.number_input("Batch size:", min_value=1, max_value=1000, value=100)
    
    with col2:
        num_iterations = st.number_input("Number of iterations:", min_value=1, max_value=100, value=10)
        simulate_network = st.checkbox("Simulate network latency", value=False)
    
    # Use synthetic data for testing
    test_data = st.session_state.synthetic_data.head(num_predictions)
    
    st.info(f"Using {len(test_data)} samples from generated synthetic data")
    
    if st.button("Run Performance Test", type="primary", key="perf_test_btn"):
        with st.spinner("Running performance tests..."):
            try:
                latencies = []
                throughputs = []
                error_count = 0
                
                progress_bar = st.progress(0)
                
                for i in range(num_iterations):
                    # Update progress
                    progress_bar.progress((i + 1) / num_iterations)
                    
                    # Simulate network latency if enabled
                    if simulate_network:
                        time.sleep(np.random.uniform(0.001, 0.01))  # 1-10ms network delay
                    
                    # Batch processing
                    start_time = time.time()
                    
                    try:
                        for j in range(0, len(test_data), batch_size):
                            batch = test_data.iloc[j:j+batch_size]
                            predictions = st.session_state.model.predict(batch.values)
                    except Exception:
                        error_count += 1
                    
                    end_time = time.time()
                    
                    latency = (end_time - start_time) * 1000  # ms
                    throughput = len(test_data) / (end_time - start_time)  # samples/sec
                    
                    latencies.append(latency)
                    throughputs.append(throughput)
                
                # Calculate error rate
                error_rate = (error_count / num_iterations) * 100
                
                # Store results
                performance_results = {
                    'latencies': latencies,
                    'throughputs': throughputs,
                    'avg_latency': np.mean(latencies),
                    'avg_throughput': np.mean(throughputs),
                    'min_latency': np.min(latencies),
                    'max_latency': np.max(latencies),
                    'error_rate': error_rate,
                    'num_predictions': num_predictions,
                    'batch_size': batch_size,
                    'num_iterations': num_iterations
                }
                
                st.session_state.performance_results = performance_results
                
                # Display results
                st.success("Performance test completed!")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Latency", f"{performance_results['avg_latency']:.2f} ms")
                with col2:
                    st.metric("Avg Throughput", f"{performance_results['avg_throughput']:.1f} samples/sec")
                with col3:
                    st.metric("Error Rate", f"{performance_results['error_rate']:.1f}%")
                with col4:
                    st.metric("Latency Range", f"{performance_results['min_latency']:.1f}-{performance_results['max_latency']:.1f} ms")
                
                # Performance charts
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(y=latencies, title="Latency per Iteration", 
                                 labels={'x': 'Iteration', 'y': 'Latency (ms)'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.line(y=throughputs, title="Throughput per Iteration",
                                 labels={'x': 'Iteration', 'y': 'Throughput (samples/sec)'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Latency distribution
                st.markdown("### Latency Distribution")
                fig = px.histogram(x=latencies, title="Distribution of Latencies (ms)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Next steps
                st.markdown("### Next Steps")
                if st.button("SHAP Analysis", key="goto_shap_from_perf_btn"):
                    st.session_state.current_page = "Model Explainability"
                    st.rerun()
                
            except Exception as e:
                st.error(f"Error during performance testing: {str(e)}")

def show_shap_page():
    """Display the SHAP explainability page."""
    
    st.markdown("## Model Explainability")
    st.markdown("Understand your model's predictions using feature importance analysis.")
    
    if st.session_state.model is None:
        st.error("Please upload a model first")
        return
    
    # Check if synthetic data is available
    if st.session_state.synthetic_data is None:
        st.error("No synthetic data available. Please generate synthetic data first.")
        return
        
    # Use synthetic data for explanation
    explain_data = st.session_state.synthetic_data.copy()
    
    st.info("Using synthetic data for SHAP explanation analysis")
    
    # Validate data types and handle any missing values
    for col in explain_data.columns:
        if explain_data[col].dtype == 'object':
            # Convert categorical columns to numeric using label encoding
            explain_data[col] = pd.Categorical(explain_data[col]).codes
        # Fill any missing values with 0 (or you could use mean/median)
        explain_data[col] = explain_data[col].fillna(0)
    
    # Sample size for explanation
    max_samples = min(100, len(explain_data))
    num_samples = st.slider("Number of samples to explain:", 1, max_samples, min(10, max_samples))
    
    if st.button("Generate SHAP Explanations", type="primary", key="shap_btn"):
        with st.spinner("Generating SHAP explanations..."):
            try:
                # Simple feature importance (correlation-based for simplicity)
                sample_data = explain_data.head(num_samples)
                
                try:
                    # Make predictions
                    predictions = st.session_state.model.predict(sample_data.values)
                    
                    # Calculate feature importance using correlation with predictions
                    feature_importance = {}
                    
                    # Handle different prediction shapes (single output or multi-output)
                    if predictions.ndim > 1:
                        # For multi-output models, use mean prediction across outputs
                        predictions_for_corr = np.mean(predictions, axis=1)
                    else:
                        predictions_for_corr = predictions
                    
                    # Calculate correlations for each feature
                    for col in sample_data.columns:
                        try:
                            correlation = np.corrcoef(sample_data[col].values, predictions_for_corr)[0, 1]
                            feature_importance[col] = abs(correlation) if not np.isnan(correlation) else 0
                        except Exception as col_error:
                            st.warning(f"Could not calculate importance for feature '{col}': {str(col_error)}")
                            feature_importance[col] = 0
                            
                except Exception as pred_error:
                    st.error(f"Error making predictions: {str(pred_error)}")
                    return
                
                # Display feature importance
                st.markdown("### Feature Importance")
                
                # Create dataframe and normalize importance scores
                importance_df = pd.DataFrame(list(feature_importance.items()), 
                                          columns=['Feature', 'Importance'])
                # Add small epsilon to avoid division by zero
                importance_df['Importance'] = np.abs(importance_df['Importance'])
                max_importance = importance_df['Importance'].max()
                if max_importance > 0:
                    importance_df['Importance'] = importance_df['Importance'] / max_importance
                
                # Sort by importance descending for better visualization
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Create feature importance bar plot
                fig = px.bar(importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Feature Importance (Correlation-based)",
                            labels={'Importance': 'Normalized Importance Score',
                                   'Feature': 'Feature Name'},
                            color='Importance',
                            color_continuous_scale='viridis')
                
                # Update layout for better readability
                fig.update_layout(
                    showlegend=False,
                    xaxis_title="Normalized Importance Score",
                    yaxis_title="Feature Name",
                    yaxis={'categoryorder': 'total descending'},  # Changed to descending to show most important at top
                    height=max(400, len(importance_df) * 30),  # Dynamic height based on number of features
                    yaxis_autorange='reversed'  # This ensures most important features appear at the top
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sample predictions table
                st.markdown("### Sample Predictions")
                result_df = sample_data.copy()
                result_df['Prediction'] = predictions
                st.dataframe(result_df)
                
                # Feature correlation heatmap
                if len(sample_data.select_dtypes(include=[np.number]).columns) > 1:
                    st.markdown("### Feature Correlation Heatmap")
                    numeric_data = sample_data.select_dtypes(include=[np.number])
                    correlation_matrix = numeric_data.corr()
                    fig = px.imshow(correlation_matrix, title="Feature Correlation Matrix", 
                                   color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Top important features
                st.markdown("### Top 5 Most Important Features")
                
                # Get top 5 features and create a dedicated visualization for them
                top_features = importance_df.head(5)  # Using head() since we sorted descending
                
                # Create a horizontal bar chart for top 5 features
                fig_top = px.bar(top_features,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title="Top 5 Most Important Features",
                                labels={'Importance': 'Normalized Importance Score',
                                       'Feature': 'Feature Name'},
                                color='Importance',
                                color_continuous_scale='viridis')
                
                fig_top.update_layout(
                    showlegend=False,
                    xaxis_title="Normalized Importance Score",
                    yaxis_title="Feature Name",
                    height=400,
                    yaxis={'categoryorder': 'total descending'},  # Show most important at top
                    yaxis_autorange='reversed'  # This ensures most important features appear at the top
                )
                
                st.plotly_chart(fig_top, use_container_width=True)
                
                # Display detailed info for top features
                for idx, row in top_features.iterrows():
                    st.info(f"**{row['Feature']}**: Normalized importance score {row['Importance']:.3f}")
                
                st.success("Feature importance analysis generated!")
                st.info("Note: This is a simplified explanation using correlation analysis. For more advanced analysis, consider installing the SHAP library for deeper model interpretability.")
                
            except Exception as e:
                st.error(f"Error generating explanations: {str(e)}")

def show_model_comparison_page():
    """Display the model comparison page with auto-detection."""
    
    st.markdown("## Model Comparison")
    st.markdown("Upload multiple models and compare their performance. Model details auto-detected! 🚀")
    
    # Check if dataset is loaded
    if st.session_state.dataset is None or st.session_state.target_column is None:
        st.warning("⚠️ Please upload a dataset first (use Dataset Upload page)")
        if st.button("Go to Dataset Upload", key="goto_dataset_from_compare"):
            st.session_state.current_page = "Dataset Upload"
            st.rerun()
        return
    
    # Display current dataset info
    st.info(f"📊 Current dataset: **{len(st.session_state.dataset)} rows**, Target: **{st.session_state.target_column}**")
    
    # Model upload section
    st.markdown("### Upload Models for Comparison")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        model_name = st.text_input(
            "Model Name (optional - will use algorithm name if empty):", 
            placeholder="e.g., Random Forest v1", 
            key="compare_model_name"
        )
    
    uploaded_file = st.file_uploader(
        "Upload model (.pkl file)",
        type=['pkl'],
        key="compare_model_uploader",
        help="Model type will be auto-detected from the file"
    )
    
    if uploaded_file is not None:
        if st.button("📥 Add Model", key="add_model_btn", type="primary"):
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            # Load the model and extract metadata
            with st.spinner("Loading model and detecting type..."):
                model = load_model(temp_path)
            
            if model is not None:
                # Extract metadata
                metadata = extract_model_metadata(model)
                
                # Use algorithm name if no custom name provided
                final_name = model_name if model_name else f"{metadata['algorithm']} #{len(st.session_state.models) + 1}"
                
                # Use detected model type or fallback
                detected_type = metadata['model_type'] if metadata['model_type'] else 'classification'
                
                # Add to models list with metadata
                st.session_state.models.append({
                    'name': final_name,
                    'model': model,
                    'model_type': detected_type,
                    'metadata': metadata
                })
                
                st.success(f"✅ Model '{final_name}' added!")
                st.info(f"🤖 Detected: **{metadata['algorithm']}** ({detected_type})")
                
                if metadata['n_features']:
                    st.caption(f"Features: {metadata['n_features']}")
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    # Display current models
    if len(st.session_state.models) > 0:
        st.markdown("### Current Models")
        
        models_data = []
        for m in st.session_state.models:
            meta = m.get('metadata', {})
            models_data.append({
                'Name': m['name'],
                'Algorithm': meta.get('algorithm', type(m['model']).__name__),
                'Type': m['model_type'],
                'Features': meta.get('n_features', 'N/A'),
                'Classes': meta.get('n_classes', 'N/A') if m['model_type'] == 'classification' else 'N/A'
            })
        
        models_df = pd.DataFrame(models_data)
        st.dataframe(models_df, use_container_width=True)
        
        # Remove model option
        col1, col2 = st.columns([3, 1])
        with col1:
            model_to_remove = st.selectbox("Select model to remove:", 
                                          [m['name'] for m in st.session_state.models],
                                          key="remove_model_select")
        with col2:
            if st.button("Remove", key="remove_model_btn"):
                st.session_state.models = [m for m in st.session_state.models if m['name'] != model_to_remove]
                st.success(f"Removed '{model_to_remove}'")
                st.rerun()
        
        # Compare models button
        st.markdown("### Run Comparison")
        
        if len(st.session_state.models) >= 2:
            if st.button("Compare All Models", type="primary", key="compare_models_btn"):
                with st.spinner("Comparing models..."):
                    try:
                        # Prepare data
                        dataset = st.session_state.dataset
                        target_column = st.session_state.target_column
                        feature_columns = [col for col in dataset.columns if col != target_column]
                        
                        X = dataset[feature_columns]
                        y_true = dataset[target_column]
                        
                        comparison_results = []
                        
                        for model_info in st.session_state.models:
                            model = model_info['model']
                            model_type = model_info['model_type']
                            model_name = model_info['name']
                            
                            try:
                                # Make predictions
                                y_pred = model.predict(X)
                                
                                # Calculate metrics based on model type
                                if model_type == "classification":
                                    metrics = calculate_classification_metrics(y_true, y_pred)
                                    result = {
                                        'name': model_name,
                                        'type': model_type,
                                        'accuracy': metrics['accuracy'],
                                        'precision': metrics['precision'],
                                        'recall': metrics['recall'],
                                        'f1': metrics['f1']
                                    }
                                else:  # regression
                                    metrics = calculate_regression_metrics(y_true, y_pred)
                                    result = {
                                        'name': model_name,
                                        'type': model_type,
                                        'r2': metrics['r2'],
                                        'rmse': metrics['rmse'],
                                        'mae': metrics['mae'],
                                        'mape': metrics['mape']
                                    }
                                
                                comparison_results.append(result)
                            
                            except Exception as e:
                                st.error(f"Error evaluating {model_name}: {str(e)}")
                        
                        st.session_state.comparison_results = comparison_results
                        st.success("Comparison completed!")
                    
                    except Exception as e:
                        st.error(f"Error during comparison: {str(e)}")
        else:
            st.info("Add at least 2 models to enable comparison")
    
    # Display comparison results
    if st.session_state.comparison_results is not None and len(st.session_state.comparison_results) > 0:
        st.markdown("### Comparison Results")
        
        results = st.session_state.comparison_results
        
        # Separate classification and regression results
        class_results = [r for r in results if r['type'] == 'classification']
        reg_results = [r for r in results if r['type'] == 'regression']
        
        # Display classification results
        if class_results:
            st.markdown("#### Classification Models")
            
            class_df = pd.DataFrame(class_results)
            class_df = class_df.round(4)
            st.dataframe(class_df, use_container_width=True)
            
            # Find best model
            best_model = max(class_results, key=lambda x: x['accuracy'])
            st.success(f"🏆 Best Classification Model: **{best_model['name']}** (Accuracy: {best_model['accuracy']:.4f})")
            
            # Comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(class_df, x='name', y='accuracy', 
                           title="Accuracy Comparison",
                           labels={'name': 'Model', 'accuracy': 'Accuracy'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(class_df, x='name', y='f1',
                           title="F1-Score Comparison",
                           labels={'name': 'Model', 'f1': 'F1-Score'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics comparison
            st.markdown("#### Detailed Metrics Comparison")
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            
            fig = go.Figure()
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric.capitalize(),
                    x=[r['name'] for r in class_results],
                    y=[r[metric] for r in class_results]
                ))
            
            fig.update_layout(
                title="All Classification Metrics Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display regression results
        if reg_results:
            st.markdown("#### Regression Models")
            
            reg_df = pd.DataFrame(reg_results)
            reg_df = reg_df.round(4)
            st.dataframe(reg_df, use_container_width=True)
            
            # Find best model (highest R²)
            best_model = max(reg_results, key=lambda x: x['r2'])
            st.success(f"🏆 Best Regression Model: **{best_model['name']}** (R²: {best_model['r2']:.4f})")
            
            # Comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(reg_df, x='name', y='r2',
                           title="R² Score Comparison",
                           labels={'name': 'Model', 'r2': 'R² Score'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(reg_df, x='name', y='rmse',
                           title="RMSE Comparison (Lower is Better)",
                           labels={'name': 'Model', 'rmse': 'RMSE'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics comparison
            st.markdown("#### Detailed Metrics Comparison")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("R² Score (Higher is Better)", "Error Metrics (Lower is Better)")
            )
            
            # R² scores
            fig.add_trace(
                go.Bar(name='R²', x=[r['name'] for r in reg_results], 
                      y=[r['r2'] for r in reg_results]),
                row=1, col=1
            )
            
            # Error metrics
            for metric, name in [('rmse', 'RMSE'), ('mae', 'MAE')]:
                fig.add_trace(
                    go.Bar(name=name, x=[r['name'] for r in reg_results],
                          y=[r[metric] for r in reg_results]),
                    row=1, col=2
                )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Export comparison
        st.markdown("### Export Results")
        
        if st.button("Clear All Models", key="clear_models_btn"):
            st.session_state.models = []
            st.session_state.comparison_results = None
            st.success("All models cleared!")
            st.rerun()

def show_dataset_analysis_page():
    """Display the dataset analysis page."""
    
    st.markdown("## Dataset Analysis")
    st.markdown("Upload multiple datasets to analyze data quality, detect imbalances, and compare distributions.")
    
    # Dataset upload section
    st.markdown("### Upload Datasets")
    
    col1, col2 = st.columns(2)
    with col1:
        dataset_name = st.text_input("Dataset Name:", placeholder="e.g., Training Data", key="dataset_name")
    
    uploaded_file = st.file_uploader(
        "Upload dataset (CSV file)",
        type=['csv'],
        key="dataset_uploader"
    )
    
    if uploaded_file is not None and dataset_name:
        if st.button("Add Dataset", key="add_dataset_btn"):
            try:
                # Load dataset
                dataset = pd.read_csv(uploaded_file)
                
                # Add to datasets list
                st.session_state.datasets.append({
                    'name': dataset_name,
                    'data': dataset
                })
                st.success(f"Dataset '{dataset_name}' added successfully!")
            
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
    
    # Display current datasets
    if len(st.session_state.datasets) > 0:
        st.markdown("### Current Datasets")
        
        datasets_info = []
        for ds in st.session_state.datasets:
            datasets_info.append({
                'Name': ds['name'],
                'Rows': len(ds['data']),
                'Columns': len(ds['data'].columns),
                'Missing Values': ds['data'].isnull().sum().sum(),
                'Duplicates': ds['data'].duplicated().sum()
            })
        
        datasets_df = pd.DataFrame(datasets_info)
        st.dataframe(datasets_df, use_container_width=True)
        
        # Remove dataset option
        col1, col2 = st.columns([3, 1])
        with col1:
            dataset_to_remove = st.selectbox("Select dataset to remove:",
                                           [ds['name'] for ds in st.session_state.datasets],
                                           key="remove_dataset_select")
        with col2:
            if st.button("Remove", key="remove_dataset_btn"):
                st.session_state.datasets = [ds for ds in st.session_state.datasets if ds['name'] != dataset_to_remove]
                st.success(f"Removed '{dataset_to_remove}'")
                st.rerun()
        
        # Analysis section
        st.markdown("### Dataset Analysis")
        
        # Select target column for imbalance analysis
        if len(st.session_state.datasets) > 0:
            sample_dataset = st.session_state.datasets[0]['data']
            target_column = st.selectbox(
                "Select target column for imbalance analysis:",
                options=sample_dataset.columns.tolist(),
                key="analysis_target_column"
            )
            
            if st.button("Analyze Datasets", type="primary", key="analyze_datasets_btn"):
                with st.spinner("Analyzing datasets..."):
                    try:
                        analysis_results = []
                        
                        for ds_info in st.session_state.datasets:
                            dataset = ds_info['data']
                            name = ds_info['name']
                            
                            # Basic statistics
                            result = {
                                'name': name,
                                'rows': len(dataset),
                                'columns': len(dataset.columns),
                                'missing_values': dataset.isnull().sum().sum(),
                                'missing_percentage': (dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns))) * 100,
                                'duplicates': dataset.duplicated().sum(),
                                'duplicate_percentage': (dataset.duplicated().sum() / len(dataset)) * 100
                            }
                            
                            # Class imbalance analysis (if target column exists)
                            if target_column in dataset.columns:
                                value_counts = dataset[target_column].value_counts()
                                result['target_classes'] = len(value_counts)
                                result['class_distribution'] = value_counts.to_dict()
                                
                                # Calculate imbalance ratio
                                if len(value_counts) > 1:
                                    max_count = value_counts.max()
                                    min_count = value_counts.min()
                                    result['imbalance_ratio'] = max_count / min_count
                                else:
                                    result['imbalance_ratio'] = 1.0
                            
                            analysis_results.append(result)
                        
                        st.session_state.dataset_analysis = analysis_results
                        st.success("Analysis completed!")
                    
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    
    # Display analysis results
    if st.session_state.dataset_analysis is not None:
        st.markdown("### Analysis Results")
        
        results = st.session_state.dataset_analysis
        
        # Data quality overview
        st.markdown("#### Data Quality Overview")
        
        quality_df = pd.DataFrame([{
            'Dataset': r['name'],
            'Rows': r['rows'],
            'Columns': r['columns'],
            'Missing Values': r['missing_values'],
            'Missing %': f"{r['missing_percentage']:.2f}%",
            'Duplicates': r['duplicates'],
            'Duplicate %': f"{r['duplicate_percentage']:.2f}%"
        } for r in results])
        
        st.dataframe(quality_df, use_container_width=True)
        
        # Missing values comparison
        st.markdown("#### Missing Values Comparison")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Missing Values',
            x=[r['name'] for r in results],
            y=[r['missing_values'] for r in results],
            text=[f"{r['missing_percentage']:.1f}%" for r in results],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Missing Values by Dataset",
            xaxis_title="Dataset",
            yaxis_title="Number of Missing Values"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Duplicate records comparison
        st.markdown("#### Duplicate Records Comparison")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Duplicates',
            x=[r['name'] for r in results],
            y=[r['duplicates'] for r in results],
            text=[f"{r['duplicate_percentage']:.1f}%" for r in results],
            textposition='auto',
            marker_color='indianred'
        ))
        
        fig.update_layout(
            title="Duplicate Records by Dataset",
            xaxis_title="Dataset",
            yaxis_title="Number of Duplicates"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Class imbalance analysis
        if 'target_classes' in results[0]:
            st.markdown("#### Class Imbalance Analysis")
            
            # Imbalance ratio comparison
            imbalance_df = pd.DataFrame([{
                'Dataset': r['name'],
                'Classes': r['target_classes'],
                'Imbalance Ratio': f"{r['imbalance_ratio']:.2f}:1"
            } for r in results])
            
            st.dataframe(imbalance_df, use_container_width=True)
            
            # Identify datasets with high imbalance
            high_imbalance = [r for r in results if r.get('imbalance_ratio', 1) > 2]
            if high_imbalance:
                st.warning(f"⚠️ {len(high_imbalance)} dataset(s) have high class imbalance (ratio > 2:1)")
                for r in high_imbalance:
                    st.info(f"**{r['name']}**: Imbalance ratio {r['imbalance_ratio']:.2f}:1")
            
            # Class distribution visualization
            st.markdown("#### Class Distribution by Dataset")
            
            for r in results:
                if 'class_distribution' in r:
                    st.markdown(f"**{r['name']}**")
                    
                    dist = r['class_distribution']
                    dist_df = pd.DataFrame(list(dist.items()), columns=['Class', 'Count'])
                    
                    fig = px.pie(dist_df, values='Count', names='Class',
                               title=f"{r['name']} - Class Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Side-by-side comparison
            if len(results) > 1:
                st.markdown("#### Side-by-Side Class Distribution")
                
                fig = go.Figure()
                
                for r in results:
                    if 'class_distribution' in r:
                        dist = r['class_distribution']
                        fig.add_trace(go.Bar(
                            name=r['name'],
                            x=list(dist.keys()),
                            y=list(dist.values())
                        ))
                
                fig.update_layout(
                    title="Class Distribution Comparison",
                    xaxis_title="Class",
                    yaxis_title="Count",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Dataset size comparison
        st.markdown("#### Dataset Size Comparison")
        
        fig = px.bar(
            x=[r['name'] for r in results],
            y=[r['rows'] for r in results],
            title="Number of Rows by Dataset",
            labels={'x': 'Dataset', 'y': 'Rows'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### Recommendations")
        
        for r in results:
            st.markdown(f"**{r['name']}:**")
            recommendations = []
            
            if r['missing_percentage'] > 5:
                recommendations.append(f"⚠️ High missing values ({r['missing_percentage']:.1f}%) - Consider imputation or removal")
            
            if r['duplicate_percentage'] > 1:
                recommendations.append(f"⚠️ Duplicate records found ({r['duplicate_percentage']:.1f}%) - Consider removing duplicates")
            
            if r.get('imbalance_ratio', 1) > 3:
                recommendations.append(f"⚠️ Severe class imbalance ({r['imbalance_ratio']:.1f}:1) - Consider oversampling, undersampling, or SMOTE")
            elif r.get('imbalance_ratio', 1) > 2:
                recommendations.append(f"⚠️ Moderate class imbalance ({r['imbalance_ratio']:.1f}:1) - Monitor model performance carefully")
            
            if not recommendations:
                recommendations.append("✅ Dataset looks good - no major issues detected")
            
            for rec in recommendations:
                st.write(f"  {rec}")
        
        # Clear button
        if st.button("Clear All Datasets", key="clear_datasets_btn"):
            st.session_state.datasets = []
            st.session_state.dataset_analysis = None
            st.success("All datasets cleared!")
            st.rerun()

if __name__ == "__main__":
    main()
