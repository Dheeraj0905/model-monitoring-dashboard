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
    page_icon="🤖",
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
    st.session_state.current_page = "🏠 Home"

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
    st.markdown('<h1 class="main-header">🤖 ML Model Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    nav_options = [
        "🏠 Home",
        "📁 Model Upload", 
        "📊 Dataset Upload",
        "📈 Results & Analytics",
        "🎲 Data Generation",
        "⚡ Performance Testing",
        "🔍 SHAP Explainability"
    ]
    
    selected_page = st.sidebar.radio(
        "Choose a page:",
        nav_options,
        index=nav_options.index(st.session_state.current_page) if st.session_state.current_page in nav_options else 0,
        key="page_nav_radio"
    )
    
    st.session_state.current_page = selected_page
    
    # Route to pages
    if st.session_state.current_page == "🏠 Home":
        show_home_page()
    elif st.session_state.current_page == "📁 Model Upload":
        show_model_upload_page()
    elif st.session_state.current_page == "📊 Dataset Upload":
        show_dataset_upload_page()
    elif st.session_state.current_page == "📈 Results & Analytics":
        show_results_page()
    elif st.session_state.current_page == "🎲 Data Generation":
        show_data_generation_page()
    elif st.session_state.current_page == "⚡ Performance Testing":
        show_performance_testing_page()
    elif st.session_state.current_page == "🔍 SHAP Explainability":
        show_shap_page()

def show_home_page():
    """Display the home page."""
    
    st.markdown("## Welcome to the ML Model Monitoring Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What is this dashboard?
        
        A comprehensive tool to upload your machine learning models and test their performance with advanced analytics.
        
        ### 🚀 Workflow:
        
        1. **📁 Model Upload**: Upload your trained model (.pkl file)
        2. **📊 Dataset Upload**: Upload your test dataset (CSV file) 
        3. **📈 Results & Analytics**: View performance metrics and visualizations
        4. **🎲 Data Generation**: Generate synthetic test data
        5. **⚡ Performance Testing**: Test model latency and throughput
        6. **🔍 SHAP Explainability**: Understand model predictions
        
        ### 🔧 Supported Models
        
        - Classification models (Random Forest, SVM, etc.)
        - Regression models (Linear Regression, etc.)
        - Any scikit-learn compatible model
        """)
    
    with col2:
        st.markdown("### 📊 Current Status")
        
        # Model status
        if st.session_state.model is not None:
            st.success("✅ Model Loaded")
            if st.session_state.model_type:
                st.info(f"🔧 Type: {st.session_state.model_type.title()}")
        else:
            st.warning("⚠️ No Model Loaded")
        
        # Dataset status
        if st.session_state.dataset is not None:
            st.success("✅ Dataset Loaded")
            st.info(f"📊 Rows: {len(st.session_state.dataset)}")
            if st.session_state.target_column:
                st.info(f"🎯 Target: {st.session_state.target_column}")
        else:
            st.warning("⚠️ No Dataset Loaded")
        
        # Synthetic data status
        if st.session_state.synthetic_data is not None:
            st.success("✅ Synthetic Data Generated")
            st.info(f"🎲 Samples: {len(st.session_state.synthetic_data)}")
        else:
            st.warning("⚠️ No Synthetic Data")
        
        # Next steps
        st.markdown("### 🔄 Next Steps")
        if st.session_state.model is None:
            if st.button("📁 Upload Model", key="home_model_btn"):
                st.session_state.current_page = "📁 Model Upload"
                st.rerun()
        elif st.session_state.dataset is None:
            if st.button("📊 Upload Dataset", key="home_dataset_btn"):
                st.session_state.current_page = "📊 Dataset Upload"
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📈 View Results", key="home_results_btn"):
                    st.session_state.current_page = "📈 Results & Analytics"
                    st.rerun()
            with col2:
                if st.button("🔍 SHAP Analysis", key="home_shap_btn"):
                    st.session_state.current_page = "🔍 SHAP Explainability"
                    st.rerun()

def show_model_upload_page():
    """Display the model upload page."""
    
    st.markdown("## 📁 Model Upload")
    st.markdown("Upload your trained machine learning model (.pkl file) to get started.")
    
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
        model = load_model(temp_path)
        
        if model is not None:
            st.session_state.model = model
            st.success("✅ Model loaded successfully!")
            
            # Model type selection
            st.markdown("### 🔧 Model Type")
            st.markdown("Please specify what type of model this is:")
            
            model_type = st.radio(
                "Select model type:",
                ["classification", "regression"],
                key="model_type_selector"
            )
            
            if model_type:
                st.session_state.model_type = model_type
                st.info(f"Model type set to: {model_type.title()}")
                
                # Show model information
                st.markdown("### 📊 Model Information")
                st.info(f"Model Class: {type(model).__name__}")
                
                # Next steps
                st.markdown("### 🔄 Next Steps")
                if st.button("📊 Upload Dataset", key="goto_dataset_btn"):
                    st.session_state.current_page = "📊 Dataset Upload"
                    st.rerun()
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
    
    # Current model status
    if st.session_state.model is not None:
        st.markdown("### 📊 Current Model Status")
        st.success("✅ Model loaded and ready")
        if st.session_state.model_type:
            st.info(f"🔧 Type: {st.session_state.model_type.title()}")

def show_dataset_upload_page():
    """Display the dataset upload page."""
    
    st.markdown("## 📊 Dataset Upload")
    st.markdown("Upload your test dataset with true labels to evaluate model performance.")
    
    # Check if model is loaded
    if st.session_state.model is None:
        st.error("❌ Please upload a model first")
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
            
            st.success(f"✅ Dataset loaded successfully! Shape: {dataset.shape}")
            
            # Dataset preview
            st.markdown("### 📋 Dataset Preview")
            st.dataframe(dataset.head())
            
            # Dataset info
            st.markdown("### 📊 Dataset Info")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", len(dataset))
                st.metric("Columns", len(dataset.columns))
            with col2:
                st.metric("Missing Values", dataset.isnull().sum().sum())
                st.metric("Duplicates", dataset.duplicated().sum())
            
            # Feature information
            st.markdown("### 🔧 Feature Information")
            st.info(f"Available columns: {', '.join(dataset.columns.tolist())}")
            
            # Target column selection
            st.markdown("### 🎯 Target Column Selection")
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
                st.info(f"Features for prediction: {', '.join(feature_columns)}")
                
                # Target distribution
                st.markdown("### 📈 Target Distribution")
                if st.session_state.model_type == "classification":
                    value_counts = dataset[target_column].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title="Target Class Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(dataset, x=target_column, title="Target Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Next steps
                st.markdown("### 🔄 Next Steps")
                if st.button("📈 View Results", key="goto_results_btn"):
                    st.session_state.current_page = "📈 Results & Analytics"
                    st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
    
    # Current dataset status
    if st.session_state.dataset is not None:
        st.markdown("### 📊 Current Dataset Status")
        st.success(f"✅ Dataset loaded: {len(st.session_state.dataset)} rows")
        if st.session_state.target_column:
            st.info(f"🎯 Target: {st.session_state.target_column}")

def show_results_page():
    """Display the results and analytics page."""
    
    st.markdown("## 📈 Results & Analytics")
    
    # Check if both model and dataset are loaded
    if st.session_state.model is None:
        st.error("❌ Please upload a model first")
        return
    
    if st.session_state.dataset is None:
        st.error("❌ Please upload a dataset first")
        return
    
    if st.session_state.target_column is None:
        st.error("❌ Please select a target column first")
        return
    
    # Prepare data
    dataset = st.session_state.dataset
    target_column = st.session_state.target_column
    feature_columns = [col for col in dataset.columns if col != target_column]
    
    X = dataset[feature_columns]
    y_true = dataset[target_column]
    
    # Show data preparation info
    st.markdown("### 📊 Data Preparation")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Features used: {len(feature_columns)}")
        st.info(f"Samples: {len(X)}")
    with col2:
        st.info(f"Target column: {target_column}")
        st.info(f"Model type: {st.session_state.model_type}")
    
    # Evaluate model button
    if st.button("🚀 Evaluate Model", type="primary", key="evaluate_btn"):
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
                    st.success("✅ Model evaluation completed!")
                
            except Exception as e:
                st.error(f"❌ Error during evaluation: {str(e)}")
    
    # Display results if available
    if st.session_state.results is not None:
        results = st.session_state.results
        
        if results['model_type'] == "classification":
            display_classification_results(results)
        else:
            display_regression_results(results)
    
    # Next steps
    if st.session_state.results is not None:
        st.markdown("### 🔄 Next Steps")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎲 Generate Synthetic Data", key="goto_generation_btn"):
                st.session_state.current_page = "🎲 Data Generation"
                st.rerun()
        with col2:
            if st.button("🔍 SHAP Analysis", key="goto_shap_btn"):
                st.session_state.current_page = "🔍 SHAP Explainability"
                st.rerun()

def display_classification_results(results):
    """Display classification results."""
    
    metrics = results['metrics']
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    st.markdown("### 📊 Classification Metrics")
    
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
        st.markdown("### 🔢 Confusion Matrix")
        cm = metrics['confusion_matrix']
        fig = px.imshow(cm, text_auto=True, aspect="auto", 
                       title="Confusion Matrix",
                       labels={'x': 'Predicted', 'y': 'Actual'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Prediction Distribution")
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
    st.markdown("### 📋 Classification Report")
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    st.dataframe(report_df.round(3))

def display_regression_results(results):
    """Display regression results."""
    
    metrics = results['metrics']
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    st.markdown("### 📊 Regression Metrics")
    
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
        st.markdown("### 📈 Predicted vs Actual")
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
        st.markdown("### 📉 Residuals")
        residuals = y_true - y_pred
        fig = px.scatter(x=y_pred, y=residuals, 
                        title="Residuals vs Predicted",
                        labels={'x': 'Predicted', 'y': 'Residuals'})
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    # Error distribution
    st.markdown("### 📊 Error Distribution")
    fig = px.histogram(x=residuals, title="Distribution of Residuals")
    st.plotly_chart(fig, use_container_width=True)

def show_data_generation_page():
    """Display the data generation page."""
    
    st.markdown("## 🎲 Data Generation")
    st.markdown("Generate synthetic test data for performance testing.")
    
    if st.session_state.model is None:
        st.error("❌ Please upload a model first")
        return
    
    if st.session_state.dataset is None:
        st.error("❌ Please upload a dataset first to understand the feature structure")
        return
    
    # Get feature information from existing dataset
    target_column = st.session_state.target_column
    feature_columns = [col for col in st.session_state.dataset.columns if col != target_column]
    
    st.markdown("### 📊 Data Generation Based on Your Dataset")
    st.info(f"Generating data with {len(feature_columns)} features: {', '.join(feature_columns)}")
    
    # Data generation parameters
    st.markdown("### ⚙️ Generation Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        num_samples = st.number_input("Number of samples:", min_value=10, max_value=10000, value=1000)
        random_seed = st.number_input("Random seed:", min_value=0, value=42)
    
    with col2:
        noise_level = st.slider("Noise level:", 0.0, 1.0, 0.1)
        distribution = st.selectbox("Distribution:", ["normal", "uniform", "exponential"])
    
    if st.button("🎲 Generate Data", key="generate_data_btn"):
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
                
                st.success(f"✅ Generated {num_samples} synthetic samples!")
                
                # Display preview
                st.markdown("### 📋 Generated Data Preview")
                st.dataframe(synthetic_df.head())
                
                # Comparison with original data
                st.markdown("### 📊 Comparison with Original Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Data Statistics:**")
                    st.dataframe(original_data.describe())
                
                with col2:
                    st.markdown("**Generated Data Statistics:**")
                    st.dataframe(synthetic_df.describe())
                
                # Next steps
                st.markdown("### 🔄 Next Steps")
                if st.button("⚡ Performance Testing", key="goto_performance_btn"):
                    st.session_state.current_page = "⚡ Performance Testing"
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating data: {str(e)}")
    
    # Display current synthetic data
    if st.session_state.synthetic_data is not None:
        st.markdown("### 📊 Current Synthetic Data")
        st.success(f"✅ Generated dataset with {len(st.session_state.synthetic_data)} samples")

def show_performance_testing_page():
    """Display the performance testing page."""
    
    st.markdown("## ⚡ Performance Testing")
    st.markdown("Test your model's latency, throughput, and error rates using synthetic data.")
    
    if st.session_state.model is None:
        st.error("❌ Please upload a model first")
        return
    
    if st.session_state.synthetic_data is None:
        st.error("❌ Please generate synthetic data first")
        return
    
    # Performance testing parameters
    st.markdown("### ⚙️ Test Parameters")
    
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
    
    if st.button("🚀 Run Performance Test", type="primary", key="perf_test_btn"):
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
                st.success("✅ Performance test completed!")
                
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
                st.markdown("### 📊 Latency Distribution")
                fig = px.histogram(x=latencies, title="Distribution of Latencies (ms)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Next steps
                st.markdown("### 🔄 Next Steps")
                if st.button("🔍 SHAP Analysis", key="goto_shap_from_perf_btn"):
                    st.session_state.current_page = "🔍 SHAP Explainability"
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during performance testing: {str(e)}")

def show_shap_page():
    """Display the SHAP explainability page."""
    
    st.markdown("## 🔍 SHAP Explainability")
    st.markdown("Understand your model's predictions using feature importance analysis.")
    
    if st.session_state.model is None:
        st.error("❌ Please upload a model first")
        return
    
    # Check available data sources
    available_data = []
    if st.session_state.dataset is not None:
        available_data.append("Original Dataset")
    if st.session_state.synthetic_data is not None:
        available_data.append("Synthetic Data")
    
    if not available_data:
        st.error("❌ No data available for explanation. Please upload a dataset or generate synthetic data.")
        return
    
    # Data source selection
    data_source = st.selectbox("Select data for explanation:", available_data)
    
    # Get the selected data
    if data_source == "Original Dataset":
        explain_data = st.session_state.dataset.copy()
        if st.session_state.target_column:
            explain_data = explain_data.drop(columns=[st.session_state.target_column])
    else:
        explain_data = st.session_state.synthetic_data.copy()
    
    # Sample size for explanation
    max_samples = min(100, len(explain_data))
    num_samples = st.slider("Number of samples to explain:", 1, max_samples, min(10, max_samples))
    
    if st.button("🔍 Generate SHAP Explanations", type="primary", key="shap_btn"):
        with st.spinner("Generating SHAP explanations..."):
            try:
                # Simple feature importance (correlation-based for simplicity)
                sample_data = explain_data.head(num_samples)
                predictions = st.session_state.model.predict(sample_data.values)
                
                # Calculate feature importance using correlation with predictions
                feature_importance = {}
                for col in sample_data.columns:
                    if pd.api.types.is_numeric_dtype(sample_data[col]):
                        correlation = np.corrcoef(sample_data[col].values, predictions)[0, 1]
                        feature_importance[col] = abs(correlation) if not np.isnan(correlation) else 0
                    else:
                        # For categorical features, use a simple encoding correlation
                        encoded_values = pd.factorize(sample_data[col])[0]
                        correlation = np.corrcoef(encoded_values, predictions)[0, 1]
                        feature_importance[col] = abs(correlation) if not np.isnan(correlation) else 0
                
                # Display feature importance
                st.markdown("### 📊 Feature Importance")
                
                importance_df = pd.DataFrame(list(feature_importance.items()), 
                                           columns=['Feature', 'Importance'])
                importance_df = importance_df.sort_values('Importance', ascending=True)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h', title="Feature Importance (Correlation-based)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Sample predictions table
                st.markdown("### 🎯 Sample Predictions")
                result_df = sample_data.copy()
                result_df['Prediction'] = predictions
                st.dataframe(result_df)
                
                # Feature correlation heatmap
                if len(sample_data.select_dtypes(include=[np.number]).columns) > 1:
                    st.markdown("### 🔥 Feature Correlation Heatmap")
                    numeric_data = sample_data.select_dtypes(include=[np.number])
                    correlation_matrix = numeric_data.corr()
                    fig = px.imshow(correlation_matrix, title="Feature Correlation Matrix", 
                                   color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Top important features
                st.markdown("### 🏆 Top 5 Most Important Features")
                top_features = importance_df.tail(5)
                for idx, row in top_features.iterrows():
                    st.info(f"**{row['Feature']}**: Importance score {row['Importance']:.3f}")
                
                st.success("✅ SHAP explanations generated!")
                st.info("ℹ️ Note: This is a simplified explanation using correlation analysis. For full SHAP analysis, install the SHAP library.")
                
            except Exception as e:
                st.error(f"❌ Error generating explanations: {str(e)}")

if __name__ == "__main__":
    main()
