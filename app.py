"""
ML Model Monitoring Dashboard - Phase 1
Main Streamlit application for interactive model testing and monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import json
import time
from datetime import datetime
import logging

# Import our custom modules
from model_utils import ModelLoader
from schema_utils import SchemaManager, SchemaDefinition
from synthetic_data import DataGenerator
from metrics import MetricsAggregator
from explainability import ExplainabilityAnalyzer
from storage import SessionManager, ResultsStorage, DataExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
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
    .warning-message {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loader' not in st.session_state:
    st.session_state.model_loader = ModelLoader()
if 'schema_manager' not in st.session_state:
    st.session_state.schema_manager = SchemaManager()
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = None
if 'metrics_aggregator' not in st.session_state:
    st.session_state.metrics_aggregator = MetricsAggregator()
if 'explainability_analyzer' not in st.session_state:
    st.session_state.explainability_analyzer = ExplainabilityAnalyzer()
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionManager()
if 'results_storage' not in st.session_state:
    st.session_state.results_storage = ResultsStorage()
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Home"

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 ML Model Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📁 Model Upload", "📋 Schema Definition", "🎲 Data Generation", 
         "🧪 Model Testing", "📊 Results & Analytics", "📈 Explainability", "💾 Session Management"],
        index=["🏠 Home", "📁 Model Upload", "📋 Schema Definition", "🎲 Data Generation", 
               "🧪 Model Testing", "📊 Results & Analytics", "📈 Explainability", "💾 Session Management"].index(st.session_state.current_page)
    )
    
    # Update session state if page changed via sidebar
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        st.rerun()
    
    # Route to appropriate page
    if st.session_state.current_page == "🏠 Home":
        show_home_page()
    elif st.session_state.current_page == "📁 Model Upload":
        show_model_upload_page()
    elif st.session_state.current_page == "📋 Schema Definition":
        show_schema_definition_page()
    elif st.session_state.current_page == "🎲 Data Generation":
        show_data_generation_page()
    elif st.session_state.current_page == "🧪 Model Testing":
        show_model_testing_page()
    elif st.session_state.current_page == "📊 Results & Analytics":
        show_results_page()
    elif st.session_state.current_page == "📈 Explainability":
        show_explainability_page()
    elif st.session_state.current_page == "💾 Session Management":
        show_session_management_page()

def show_home_page():
    """Display the home page with overview and instructions."""
    
    st.markdown("## Welcome to the ML Model Monitoring Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What is this dashboard?
        
        This dashboard provides a comprehensive platform for monitoring and evaluating machine learning models. 
        You can upload your trained models, define input schemas, generate synthetic test data, and run 
        automated tests to measure performance, detect drift, and understand model behavior.
        
        ### 🚀 Key Features
        
        - **Model Upload**: Upload and validate your trained models (.pkl files)
        - **Schema Definition**: Define or auto-detect input feature schemas
        - **Synthetic Data Generation**: Generate test data with various distributions
        - **Performance Testing**: Measure latency, throughput, and accuracy
        - **Drift Detection**: Simulate and detect data drift
        - **Explainability**: SHAP-based model explanations
        - **Session Management**: Track and compare test runs over time
        
        ### 📋 Getting Started
        
        1. **Upload a Model**: Start by uploading your trained model file
        2. **Define Schema**: Specify your input feature schema
        3. **Generate Data**: Create synthetic test data
        4. **Run Tests**: Execute comprehensive model evaluation
        5. **Analyze Results**: Review performance metrics and explanations
        """)
    
    with col2:
        st.markdown("### 📊 Current Status")
        
        # Show current session status
        if st.session_state.current_session_id:
            session_data = st.session_state.session_manager.get_current_session()
            if session_data:
                st.success(f"✅ Active Session: {session_data['session_name']}")
                st.info(f"📅 Created: {session_data['created_at'][:10]}")
                st.info(f"🧪 Test Runs: {len(session_data['test_runs'])}")
            else:
                st.warning("⚠️ Session data not found")
        else:
            st.info("ℹ️ No active session")
        
        # Show model status
        if st.session_state.model_loader.model is not None:
            model_info = st.session_state.model_loader.get_model_info()
            st.success("✅ Model Loaded")
            st.info(f"📊 Features: {model_info['n_features']}")
            st.info(f"🔧 Type: {model_info['model_info']['type']}")
        else:
            st.info("ℹ️ No model loaded")
        
        # Show schema status
        if st.session_state.schema_manager.schema:
            schema_summary = st.session_state.schema_manager.get_summary()
            st.success("✅ Schema Defined")
            st.info(f"📋 Features: {schema_summary['n_features']}")
        else:
            st.info("ℹ️ No schema defined")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🆕 Create New Session", use_container_width=True):
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_id = st.session_state.session_manager.create_session(session_name)
            st.session_state.current_session_id = session_id
            st.success(f"Created new session: {session_name}")
            st.rerun()
    
    with col2:
        if st.button("📁 Upload Model", use_container_width=True):
            st.session_state.current_page = "📁 Model Upload"
            st.rerun()
    
    with col3:
        if st.button("🧪 Run Tests", use_container_width=True):
            if st.session_state.model_loader.model is not None:
                st.session_state.current_page = "🧪 Model Testing"
                st.rerun()
            else:
                st.error("Please upload a model first")

def show_model_upload_page():
    """Display the model upload page."""
    
    st.markdown("## 📁 Model Upload")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a model file (.pkl)",
        type=['pkl'],
        help="Upload your trained model file in pickle format"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with st.spinner("Processing uploaded model..."):
            try:
                # Save to temporary file
                temp_path = f"temp_model_{int(time.time())}.pkl"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load model
                success = st.session_state.model_loader.load_model(temp_path)
                
                if success:
                    st.success("✅ Model loaded successfully!")
                    
                    # Display model information
                    model_info = st.session_state.model_loader.get_model_info()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Model Information")
                        st.json({
                            "Model Type": model_info['model_info']['type'],
                            "Module": model_info['model_info']['module'],
                            "Number of Features": model_info['n_features'],
                            "Has Probabilities": model_info['has_proba']
                        })
                    
                    with col2:
                        st.markdown("### 🔧 Feature Information")
                        if model_info['feature_names']:
                            st.write("**Detected Features:**")
                            for i, (name, dtype) in enumerate(zip(model_info['feature_names'], model_info['feature_types'])):
                                st.write(f"{i+1}. {name} ({dtype})")
                        else:
                            st.info("No feature information detected")
                    
                    # Auto-create schema if features detected
                    if model_info['feature_names']:
                        st.markdown("### 🔄 Auto-Create Schema")
                        if st.button("Create Schema from Model", type="primary"):
                            # Create schema from model features
                            st.session_state.schema_manager.schema = []
                            for name, dtype in zip(model_info['feature_names'], model_info['feature_types']):
                                feature = SchemaDefinition(
                                    name=name,
                                    data_type=dtype,
                                    description=f"Auto-detected from model"
                                )
                                st.session_state.schema_manager.add_feature(feature)
                            
                            st.success("✅ Schema created from model features!")
                            st.rerun()
                    
                    # Update session with model info
                    if st.session_state.current_session_id:
                        st.session_state.session_manager.update_model_info(model_info)
                
                else:
                    st.error("❌ Failed to load model. Please check the file format.")
                
                # Clean up temporary file
                import os
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
    
    # Sample model creation
    st.markdown("---")
    st.markdown("### 🧪 Create Sample Model")
    
    if st.button("Generate Sample Model for Testing"):
        with st.spinner("Creating sample model..."):
            try:
                sample_path = st.session_state.model_loader.create_sample_model()
                success = st.session_state.model_loader.load_model(sample_path)
                
                if success:
                    st.success("✅ Sample model created and loaded!")
                    st.info("This is a RandomForestClassifier trained on synthetic data for demonstration purposes.")
                    
                    # Auto-create schema
                    model_info = st.session_state.model_loader.get_model_info()
                    st.session_state.schema_manager.schema = []
                    for name, dtype in zip(model_info['feature_names'], model_info['feature_types']):
                        feature = SchemaDefinition(
                            name=name,
                            data_type=dtype,
                            min_val=-10.0,
                            max_val=10.0,
                            description=f"Sample feature {name}"
                        )
                        st.session_state.schema_manager.add_feature(feature)
                    
                    st.success("✅ Schema created for sample model!")
                    st.rerun()
                else:
                    st.error("❌ Failed to create sample model")
                    
            except Exception as e:
                st.error(f"❌ Error creating sample model: {str(e)}")

def show_schema_definition_page():
    """Display the schema definition page."""
    
    st.markdown("## 📋 Schema Definition")
    
    # Current schema status
    if st.session_state.schema_manager.schema:
        st.success("✅ Schema is defined")
        schema_summary = st.session_state.schema_manager.get_summary()
        st.info(f"Number of features: {schema_summary['n_features']}")
    else:
        st.warning("⚠️ No schema defined")
    
    # Schema definition options
    tab1, tab2, tab3 = st.tabs(["Manual Definition", "CSV Auto-Detection", "Default Schema"])
    
    with tab1:
        st.markdown("### ✏️ Manual Schema Definition")
        
        # Add new feature
        with st.expander("Add New Feature"):
            col1, col2 = st.columns(2)
            
            with col1:
                feature_name = st.text_input("Feature Name", key="new_feature_name")
                feature_type = st.selectbox("Data Type", ["float64", "int64", "category"], key="new_feature_type")
            
            with col2:
                if feature_type in ["float64", "int64"]:
                    min_val = st.number_input("Minimum Value", key="new_min_val")
                    max_val = st.number_input("Maximum Value", key="new_max_val")
                else:
                    min_val = max_val = None
                
                if feature_type == "category":
                    categories_input = st.text_input("Categories (comma-separated)", key="new_categories")
                    categories = [cat.strip() for cat in categories_input.split(",")] if categories_input else None
                else:
                    categories = None
            
            description = st.text_input("Description", key="new_description")
            
            if st.button("Add Feature"):
                if feature_name:
                    feature = SchemaDefinition(
                        name=feature_name,
                        data_type=feature_type,
                        min_val=min_val,
                        max_val=max_val,
                        categories=categories,
                        description=description
                    )
                    st.session_state.schema_manager.add_feature(feature)
                    st.success(f"✅ Added feature: {feature_name}")
                    st.rerun()
                else:
                    st.error("Please enter a feature name")
        
        # Display current features
        if st.session_state.schema_manager.schema:
            st.markdown("### 📋 Current Features")
            
            for i, feature in enumerate(st.session_state.schema_manager.schema):
                with st.expander(f"{i+1}. {feature.name} ({feature.data_type})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Name:** {feature.name}")
                        st.write(f"**Type:** {feature.data_type}")
                        st.write(f"**Description:** {feature.description}")
                    
                    with col2:
                        if feature.data_type in ["float64", "int64"]:
                            st.write(f"**Range:** {feature.min_val} to {feature.max_val}")
                        if feature.categories:
                            st.write(f"**Categories:** {', '.join(feature.categories)}")
                    
                    if st.button(f"Remove {feature.name}", key=f"remove_{i}"):
                        st.session_state.schema_manager.remove_feature(feature.name)
                        st.success(f"✅ Removed feature: {feature.name}")
                        st.rerun()
    
    with tab2:
        st.markdown("### 📄 CSV Auto-Detection")
        
        uploaded_csv = st.file_uploader(
            "Upload CSV file for schema auto-detection",
            type=['csv'],
            help="Upload a CSV file to automatically detect the schema"
        )
        
        if uploaded_csv is not None:
            with st.spinner("Detecting schema from CSV..."):
                try:
                    # Save temporary CSV
                    temp_csv = f"temp_schema_{int(time.time())}.csv"
                    with open(temp_csv, "wb") as f:
                        f.write(uploaded_csv.getbuffer())
                    
                    # Auto-detect schema
                    success = st.session_state.schema_manager.auto_detect_from_csv(temp_csv)
                    
                    if success:
                        st.success("✅ Schema auto-detected from CSV!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to auto-detect schema")
                    
                    # Clean up
                    import os
                    if os.path.exists(temp_csv):
                        os.remove(temp_csv)
                        
                except Exception as e:
                    st.error(f"❌ Error processing CSV: {str(e)}")
    
    with tab3:
        st.markdown("### 🎯 Default Schema")
        
        n_features = st.number_input("Number of features", min_value=1, max_value=20, value=5)
        
        if st.button("Create Default Schema"):
            success = st.session_state.schema_manager.create_default_schema(n_features)
            if success:
                st.success(f"✅ Created default schema with {n_features} features!")
                st.rerun()
            else:
                st.error("❌ Failed to create default schema")
    
    # Schema export/import
    if st.session_state.schema_manager.schema:
        st.markdown("---")
        st.markdown("### 💾 Schema Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export schema
            schema_json = st.session_state.schema_manager.to_json()
            st.download_button(
                label="📥 Download Schema (JSON)",
                data=schema_json,
                file_name=f"schema_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Import schema
            uploaded_schema = st.file_uploader(
                "Upload Schema (JSON)",
                type=['json'],
                help="Upload a previously saved schema file"
            )
            
            if uploaded_schema is not None:
                try:
                    schema_data = json.load(uploaded_schema)
                    st.session_state.schema_manager.from_dict(schema_data)
                    st.success("✅ Schema imported successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error importing schema: {str(e)}")

def show_data_generation_page():
    """Display the data generation page."""
    
    st.markdown("## 🎲 Synthetic Data Generation")
    
    # Check if schema is defined
    if not st.session_state.schema_manager.schema:
        st.error("❌ Please define a schema first before generating data")
        return
    
    # Initialize data generator if not exists
    if st.session_state.data_generator is None:
        st.session_state.data_generator = DataGenerator(st.session_state.schema_manager)
    
    # Generation parameters
    st.markdown("### ⚙️ Generation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.number_input("Number of samples", min_value=10, max_value=10000, value=1000)
        distribution_type = st.selectbox(
            "Distribution type",
            ["uniform", "normal", "mixed"],
            help="Choose the distribution for generating numeric features"
        )
    
    with col2:
        add_noise = st.checkbox("Add noise", value=True)
        noise_level = st.slider("Noise level", 0.0, 1.0, 0.1, 0.1)
        seed = st.number_input("Random seed", min_value=0, value=42)
    
    # Generate data
    if st.button("🎲 Generate Data", type="primary"):
        with st.spinner("Generating synthetic data..."):
            try:
                # Generate data
                synthetic_data = st.session_state.data_generator.generate_data(
                    n_samples=n_samples,
                    distribution_type=distribution_type,
                    add_noise=add_noise,
                    noise_level=noise_level,
                    seed=seed
                )
                
                # Store in session state
                st.session_state.synthetic_data = synthetic_data
                
                st.success(f"✅ Generated {len(synthetic_data)} samples with {len(synthetic_data.columns)} features!")
                
                # Display data preview
                st.markdown("### 📊 Data Preview")
                st.dataframe(synthetic_data.head(10))
                
                # Display data statistics
                st.markdown("### 📈 Data Statistics")
                st.dataframe(synthetic_data.describe())
                
            except Exception as e:
                st.error(f"❌ Error generating data: {str(e)}")
    
    # Drift simulation
    if 'synthetic_data' in st.session_state and st.session_state.synthetic_data is not None:
        st.markdown("---")
        st.markdown("### 🌊 Drift Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            drift_type = st.selectbox(
                "Drift type",
                ["mean_shift", "variance_shift", "distribution_shift"],
                help="Type of drift to simulate"
            )
            drift_magnitude = st.slider("Drift magnitude", 0.0, 2.0, 0.5, 0.1)
        
        with col2:
            # Feature selection for drift
            numeric_features = [
                f.name for f in st.session_state.schema_manager.schema 
                if f.data_type in ["float64", "int64"]
            ]
            
            if numeric_features:
                affected_features = st.multiselect(
                    "Affected features",
                    numeric_features,
                    default=numeric_features[:2] if len(numeric_features) >= 2 else numeric_features
                )
            else:
                affected_features = []
                st.info("No numeric features available for drift simulation")
        
        if st.button("🌊 Generate Drift Data") and affected_features:
            with st.spinner("Generating drift data..."):
                try:
                    drift_data = st.session_state.data_generator.generate_drift_data(
                        st.session_state.synthetic_data,
                        drift_type=drift_type,
                        drift_magnitude=drift_magnitude,
                        affected_features=affected_features
                    )
                    
                    st.session_state.drift_data = drift_data
                    
                    st.success(f"✅ Generated drift data with {drift_type}!")
                    
                    # Compare original vs drift data
                    st.markdown("### 📊 Original vs Drift Data Comparison")
                    
                    for feature in affected_features:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Original {feature}:**")
                            st.write(f"Mean: {st.session_state.synthetic_data[feature].mean():.3f}")
                            st.write(f"Std: {st.session_state.synthetic_data[feature].std():.3f}")
                        
                        with col2:
                            st.write(f"**Drift {feature}:**")
                            st.write(f"Mean: {drift_data[feature].mean():.3f}")
                            st.write(f"Std: {drift_data[feature].std():.3f}")
                    
                except Exception as e:
                    st.error(f"❌ Error generating drift data: {str(e)}")
    
    # Data export
    if 'synthetic_data' in st.session_state and st.session_state.synthetic_data is not None:
        st.markdown("---")
        st.markdown("### 💾 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export original data
            csv_data = st.session_state.synthetic_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Original Data (CSV)",
                data=csv_data,
                file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export drift data if available
            if 'drift_data' in st.session_state and st.session_state.drift_data is not None:
                drift_csv_data = st.session_state.drift_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Drift Data (CSV)",
                    data=drift_csv_data,
                    file_name=f"drift_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def show_model_testing_page():
    """Display the model testing page."""
    
    st.markdown("## 🧪 Model Testing")
    
    # Check prerequisites
    if st.session_state.model_loader.model is None:
        st.error("❌ Please upload a model first")
        return
    
    if not st.session_state.schema_manager.schema:
        st.error("❌ Please define a schema first")
        return
    
    if 'synthetic_data' not in st.session_state or st.session_state.synthetic_data is None:
        st.error("❌ Please generate synthetic data first")
        return
    
    # Test configuration
    st.markdown("### ⚙️ Test Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_name = st.text_input("Test Name", value=f"test_{datetime.now().strftime('%H%M%S')}")
        n_iterations = st.number_input("Timing iterations", min_value=1, max_value=10, value=3)
    
    with col2:
        include_drift_test = st.checkbox("Include drift detection", value=True)
        include_explainability = st.checkbox("Include explainability analysis", value=True)
    
    # Run tests
    if st.button("🚀 Run Comprehensive Tests", type="primary"):
        with st.spinner("Running comprehensive model tests..."):
            try:
                # Prepare data
                X = st.session_state.synthetic_data.values
                reference_data = st.session_state.synthetic_data
                
                # Run comprehensive evaluation
                results = st.session_state.metrics_aggregator.run_comprehensive_evaluation(
                    model=st.session_state.model_loader.model,
                    X=X,
                    reference_data=reference_data if include_drift_test else None,
                    test_name=test_name
                )
                
                # Add explainability if requested
                if include_explainability:
                    try:
                        explainability_results = st.session_state.explainability_analyzer.analyze_model(
                            model=st.session_state.model_loader.model,
                            X=X,
                            feature_names=st.session_state.schema_manager.get_feature_names()
                        )
                        results['explainability'] = explainability_results
                    except Exception as e:
                        st.warning(f"Explainability analysis failed: {str(e)}")
                        results['explainability'] = {'error': str(e)}
                
                # Store results
                st.session_state.test_results = results
                
                # Add to session
                if st.session_state.current_session_id:
                    st.session_state.session_manager.add_test_run(results, test_name)
                
                st.success("✅ Tests completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error running tests: {str(e)}")
    
    # Display results if available
    if st.session_state.test_results:
        st.markdown("---")
        st.markdown("### 📊 Test Results")
        
        # Performance metrics
        if 'performance' in st.session_state.test_results:
            performance = st.session_state.test_results['performance']
            
            st.markdown("#### ⚡ Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Latency (ms)", f"{performance.get('latency_mean_ms', 0):.2f}")
            with col2:
                st.metric("Throughput (pred/s)", f"{performance.get('throughput_preds_per_sec', 0):.0f}")
            with col3:
                st.metric("Error Rate", f"{performance.get('error_rate', 0):.2%}")
            with col4:
                accuracy = performance.get('accuracy', 0)
                if accuracy is not None:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                else:
                    st.metric("Accuracy", "N/A")
        
        # Drift detection results
        if 'drift' in st.session_state.test_results:
            drift = st.session_state.test_results['drift']
            
            st.markdown("#### 🌊 Drift Detection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                drift_detected = drift.get('overall_drift_detected', False)
                drift_score = drift.get('overall_drift_score', 0)
                
                if drift_detected:
                    st.error(f"🚨 Drift Detected! Score: {drift_score:.2f}")
                else:
                    st.success(f"✅ No Significant Drift. Score: {drift_score:.2f}")
            
            with col2:
                summary = drift.get('summary', {})
                st.info(f"Features with drift: {summary.get('n_features_with_drift', 0)}/{summary.get('total_features', 0)}")

def show_results_page():
    """Display the results and analytics page."""
    
    st.markdown("## 📊 Results & Analytics")
    
    # Check if we have test results
    if not st.session_state.test_results:
        st.info("ℹ️ No test results available. Run some tests first.")
        return
    
    results = st.session_state.test_results
    
    # Results overview
    st.markdown("### 📈 Results Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test Name", results.get('test_name', 'Unknown'))
    with col2:
        st.metric("Samples", results.get('n_samples', 0))
    with col3:
        st.metric("Features", results.get('n_features', 0))
    
    # Performance metrics visualization
    if 'performance' in results:
        st.markdown("### ⚡ Performance Metrics")
        
        performance = results['performance']
        
        # Create performance metrics chart
        metrics_data = {
            'Metric': ['Latency (ms)', 'Throughput (pred/s)', 'Error Rate (%)'],
            'Value': [
                performance.get('latency_mean_ms', 0),
                performance.get('throughput_preds_per_sec', 0),
                performance.get('error_rate', 0) * 100
            ]
        }
        
        fig = px.bar(
            x=metrics_data['Metric'],
            y=metrics_data['Value'],
            title="Performance Metrics",
            color=metrics_data['Value'],
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed performance metrics
        with st.expander("📋 Detailed Performance Metrics"):
            st.json(performance)
    
    # Drift analysis
    if 'drift' in results:
        st.markdown("### 🌊 Drift Analysis")
        
        drift = results['drift']
        
        # Overall drift score
        drift_score = drift.get('overall_drift_score', 0)
        
        # Create drift score gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = drift_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Drift Score"},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature-level drift
        if 'feature_drift' in drift:
            feature_drift_data = []
            for feature, drift_info in drift['feature_drift'].items():
                feature_drift_data.append({
                    'Feature': feature,
                    'Drift Score': drift_info.get('drift_score', 0),
                    'Drift Detected': drift_info.get('drift_detected', False)
                })
            
            if feature_drift_data:
                df_drift = pd.DataFrame(feature_drift_data)
                
                fig = px.bar(
                    df_drift,
                    x='Feature',
                    y='Drift Score',
                    color='Drift Detected',
                    title="Feature-level Drift Scores",
                    color_discrete_map={True: 'red', False: 'green'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Export results
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as JSON
        results_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📥 Download Results (JSON)",
            data=results_json,
            file_name=f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Export as CSV (performance metrics only)
        if 'performance' in results:
            perf_data = results['performance']
            perf_df = pd.DataFrame([perf_data])
            csv_data = perf_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Performance Metrics (CSV)",
                data=csv_data,
                file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def show_explainability_page():
    """Display the explainability page."""
    
    st.markdown("## 📈 Model Explainability")
    
    # Check if we have explainability results
    if 'explainability' not in st.session_state.test_results:
        st.info("ℹ️ No explainability results available. Run tests with explainability enabled.")
        return
    
    explainability = st.session_state.test_results['explainability']
    
    if 'error' in explainability:
        st.error(f"❌ Explainability analysis failed: {explainability['error']}")
        return
    
    # Feature importance
    if 'explanations' in explainability and 'feature_importance' in explainability['explanations']:
        feature_importance = explainability['explanations']['feature_importance']
        
        st.markdown("### 🎯 Feature Importance")
        
        if feature_importance:
            # Create feature importance chart
            features = list(feature_importance.keys())
            importance_values = list(feature_importance.values())
            
            fig = px.bar(
                x=importance_values,
                y=features,
                orientation='h',
                title="Feature Importance (SHAP Values)",
                labels={'x': 'Importance', 'y': 'Features'}
            )
            fig.update_layout(height=max(400, len(features) * 30))
            st.plotly_chart(fig, use_container_width=True)
            
            # Top features
            st.markdown("#### 🏆 Top 5 Most Important Features")
            top_features = list(feature_importance.items())[:5]
            
            for i, (feature, importance) in enumerate(top_features, 1):
                st.write(f"{i}. **{feature}**: {importance:.4f}")
        else:
            st.warning("No feature importance data available")
    
    # Visualizations
    if 'visualizations' in explainability:
        visualizations = explainability['visualizations']
        
        st.markdown("### 📊 SHAP Visualizations")
        
        if 'summary_plot' in visualizations:
            st.markdown("#### 📈 SHAP Summary Plot")
            st.image(f"data:image/png;base64,{visualizations['summary_plot']}", use_column_width=True)
        
        if 'waterfall_plot' in visualizations:
            st.markdown("#### 🌊 SHAP Waterfall Plot (Sample 1)")
            st.image(f"data:image/png;base64,{visualizations['waterfall_plot']}", use_column_width=True)
    
    # Summary
    if 'summary' in explainability:
        summary = explainability['summary']
        
        st.markdown("### 📋 Explainability Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Features", summary.get('total_features', 0))
            st.metric("Max Importance", f"{summary.get('max_importance', 0):.4f}")
        
        with col2:
            st.metric("Mean Importance", f"{summary.get('mean_importance', 0):.4f}")
            if 'top_feature_percentage' in summary:
                st.metric("Top Feature %", f"{summary.get('top_feature_percentage', 0):.1f}%")

def show_session_management_page():
    """Display the session management page."""
    
    st.markdown("## 💾 Session Management")
    
    # Current session info
    if st.session_state.current_session_id:
        session_data = st.session_state.session_manager.get_current_session()
        if session_data:
            st.success(f"✅ Active Session: {session_data['session_name']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Created", session_data['created_at'][:10])
            with col2:
                st.metric("Test Runs", len(session_data['test_runs']))
            with col3:
                st.metric("Status", session_data['status'])
        else:
            st.warning("⚠️ Session data not found")
    else:
        st.info("ℹ️ No active session")
    
    # Session actions
    st.markdown("### 🎛️ Session Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🆕 Create New Session", use_container_width=True):
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_id = st.session_state.session_manager.create_session(session_name)
            st.session_state.current_session_id = session_id
            st.success(f"Created new session: {session_name}")
            st.rerun()
    
    with col2:
        if st.button("📊 View Session History", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("💾 Export Session", use_container_width=True):
            if st.session_state.current_session_id:
                export_path = f"session_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                success = st.session_state.session_manager.export_session(
                    st.session_state.current_session_id, export_path
                )
                if success:
                    st.success(f"Session exported to {export_path}")
                else:
                    st.error("Failed to export session")
            else:
                st.error("No active session to export")
    
    # Session history
    st.markdown("### 📚 Session History")
    
    sessions = st.session_state.session_manager.list_sessions()
    
    if sessions:
        # Create sessions DataFrame
        sessions_df = pd.DataFrame(sessions)
        
        # Display sessions table
        st.dataframe(
            sessions_df[['session_name', 'created_at', 'total_test_runs', 'status']],
            use_container_width=True
        )
        
        # Session selection
        selected_session = st.selectbox(
            "Select a session to load:",
            options=[s['session_id'] for s in sessions],
            format_func=lambda x: next(s['session_name'] for s in sessions if s['session_id'] == x)
        )
        
        if st.button("📂 Load Selected Session"):
            success = st.session_state.session_manager.load_session(selected_session)
            if success:
                st.session_state.current_session_id = selected_session
                st.success("Session loaded successfully!")
                st.rerun()
            else:
                st.error("Failed to load session")
    else:
        st.info("No sessions found. Create a new session to get started.")
    
    # Test runs history
    if st.session_state.current_session_id:
        st.markdown("### 🧪 Test Runs History")
        
        test_runs = st.session_state.session_manager.get_test_runs()
        
        if test_runs:
            # Display test runs
            for i, test_run in enumerate(test_runs):
                with st.expander(f"Test Run {i+1}: {test_run['test_name']} ({test_run['timestamp'][:16]})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Test Name:** {test_run['test_name']}")
                        st.write(f"**Timestamp:** {test_run['timestamp']}")
                        st.write(f"**Status:** {test_run['status']}")
                    
                    with col2:
                        results = test_run.get('results', {})
                        if 'performance' in results:
                            perf = results['performance']
                            st.write(f"**Latency:** {perf.get('latency_mean_ms', 0):.2f} ms")
                            st.write(f"**Throughput:** {perf.get('throughput_preds_per_sec', 0):.0f} pred/s")
                            st.write(f"**Accuracy:** {perf.get('accuracy', 0):.2%}" if perf.get('accuracy') else "**Accuracy:** N/A")
        else:
            st.info("No test runs in current session")

if __name__ == "__main__":
    main()
