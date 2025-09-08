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
from storage import ResultsStorage, DataExporter  # Remove SessionManager import

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
    .option-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .feature-item {
        background-color: #ffffff;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #e9ecef;
        margin: 0.25rem 0;
    }
    .session-item {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        transition: box-shadow 0.2s;
    }
    .session-item:hover {
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
if 'results_storage' not in st.session_state:
    st.session_state.results_storage = ResultsStorage()
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Home"

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 ML Model Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation with buttons
    st.sidebar.title("Navigation")
    
    # Navigation buttons - simplified version
    nav_buttons = {
        "🏠 Home": "show_home_page",
        "📁 Model Upload": "show_model_upload_page",
        "📋 Schema Definition": "show_schema_definition_page",
        "🎲 Data Generation": "show_data_generation_page",
        "🧪 Model Testing": "show_model_testing_page",
        "📊 Results & Analytics": "show_results_page",
        "📈 Explainability": "show_explainability_page"
    }
    
    # Create navigation buttons
    with st.sidebar:
        for nav_label in nav_buttons.keys():
            button_type = "primary" if nav_label == st.session_state.current_page else "secondary"
            if st.button(
                nav_label,
                type=button_type,
                key=f"nav_{nav_label}",
                use_container_width=True
            ):
                st.session_state.current_page = nav_label
    
    # Route to appropriate page based on current_page state
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
        
        ### 📋 Getting Started
        
        1. **Upload a Model**: Start by uploading your trained model file
        2. **Define Schema**: Specify your input feature schema
        3. **Generate Data**: Create synthetic test data
        4. **Run Tests**: Execute comprehensive model evaluation
        5. **Analyze Results**: Review performance metrics and explanations
        """)
    
    with col2:
        st.markdown("### 📊 Current Status")
        
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📁 Upload Model", use_container_width=True):
            st.session_state.current_page = "📁 Model Upload"
    
    with col2:
        if st.button("🧪 Run Tests", use_container_width=True):
            if st.session_state.model_loader.model is not None:
                st.session_state.current_page = "🧪 Model Testing"
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
                from model_utils import create_sample_model
                sample_path = create_sample_model()
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
    st.markdown("### 📝 Schema Definition Options")
    
    # Create a more visual selection
    option = st.radio(
        "Choose how to define your schema:",
        ["Manual Definition", "CSV Auto-Detection", "Default Schema"],
        horizontal=True
    )
    
    if option == "Manual Definition":
        st.markdown("---")
        st.markdown("#### ✏️ Manual Schema Definition")
        
        # Add new feature
        with st.expander("Add New Feature"):
            col1, col2 = st.columns(2)
            
            with col1:
                feature_name = st.text_input("Feature Name", key="new_feature_name")
                st.write("**Data Type:**")
                feature_type = st.radio("", ["float64", "int64", "category"], key="new_feature_type", horizontal=True)
            
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
    
    elif option == "CSV Auto-Detection":
        st.markdown("---")
        st.markdown("#### 📄 CSV Auto-Detection")
        
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
    
    elif option == "Default Schema":
        st.markdown("---")
        st.markdown("#### 🎯 Default Schema")
        
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
        st.write("**Distribution type:**")
        distribution_type = st.radio(
            "",
            ["uniform", "normal", "mixed"],
            help="Choose the distribution for generating numeric features",
            horizontal=True
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
            st.write("**Drift type:**")
            drift_type = st.radio(
                "",
                ["mean_shift", "variance_shift", "distribution_shift"],
                help="Type of drift to simulate",
                horizontal=True
            )
            drift_magnitude = st.slider("Drift magnitude", 0.0, 2.0, 0.5, 0.1)
        
        with col2:
            # Feature selection for drift
            numeric_features = [
                f.name for f in st.session_state.schema_manager.schema 
                if f.data_type in ["float64", "int64"]
            ]
            
            if numeric_features:
                st.write("**Affected features:**")
                affected_features = []
                for feature in numeric_features:
                    if st.checkbox(feature, value=feature in numeric_features[:2] if len(numeric_features) >= 2 else feature == numeric_features[0]):
                        affected_features.append(feature)
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
                    
                    # Statistical comparison
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
                    
                    # Visual comparison graphs
                    st.markdown("#### 📈 Distribution Comparison")
                    
                    # Create comparison plots for each affected feature
                    for feature in affected_features:
                        st.markdown(f"**{feature} Distribution Comparison**")
                        
                        # Create histogram comparison
                        fig = go.Figure()
                        
                        # Original data
                        fig.add_trace(go.Histogram(
                            x=st.session_state.synthetic_data[feature],
                            name='Original Data',
                            opacity=0.7,
                            nbinsx=30
                        ))
                        
                        # Drift data
                        fig.add_trace(go.Histogram(
                            x=drift_data[feature],
                            name='Drift Data',
                            opacity=0.7,
                            nbinsx=30
                        ))
                        
                        fig.update_layout(
                            title=f'{feature} - Original vs Drift Distribution',
                            xaxis_title=feature,
                            yaxis_title='Frequency',
                            barmode='overlay',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Box plot comparison
                    st.markdown("#### 📦 Box Plot Comparison")
                    
                    # Prepare data for box plot
                    comparison_data = []
                    for feature in affected_features:
                        # Original data
                        for value in st.session_state.synthetic_data[feature]:
                            comparison_data.append({
                                'Feature': feature,
                                'Value': value,
                                'Type': 'Original'
                            })
                        
                        # Drift data
                        for value in drift_data[feature]:
                            comparison_data.append({
                                'Feature': feature,
                                'Value': value,
                                'Type': 'Drift'
                            })
                    
                    if comparison_data:
                        df_comparison = pd.DataFrame(comparison_data)
                        
                        # Create box plot
                        fig = px.box(
                            df_comparison,
                            x='Feature',
                            y='Value',
                            color='Type',
                            title='Feature Distribution Comparison (Box Plots)',
                            height=500
                        )
                        
                        fig.update_layout(
                            xaxis_title='Features',
                            yaxis_title='Values'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics table
                    st.markdown("#### 📋 Summary Statistics")
                    
                    summary_data = []
                    for feature in affected_features:
                        original_mean = st.session_state.synthetic_data[feature].mean()
                        original_std = st.session_state.synthetic_data[feature].std()
                        drift_mean = drift_data[feature].mean()
                        drift_std = drift_data[feature].std()
                        
                        # Calculate change
                        mean_change = ((drift_mean - original_mean) / original_mean * 100) if original_mean != 0 else 0
                        std_change = ((drift_std - original_std) / original_std * 100) if original_std != 0 else 0
                        
                        summary_data.append({
                            'Feature': feature,
                            'Original Mean': f"{original_mean:.3f}",
                            'Drift Mean': f"{drift_mean:.3f}",
                            'Mean Change %': f"{mean_change:.1f}%",
                            'Original Std': f"{original_std:.3f}",
                            'Drift Std': f"{drift_std:.3f}",
                            'Std Change %': f"{std_change:.1f}%"
                        })
                    
                    if summary_data:
                        df_summary = pd.DataFrame(summary_data)
                        st.dataframe(df_summary, use_container_width=True)
                    
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
        include_explainability = st.checkbox("Include explainability analysis", value=True)
    
    # Run tests
    if st.button("🚀 Run Comprehensive Tests", type="primary"):
        with st.spinner("Running comprehensive model tests..."):
            try:
                # Prepare data
                X = st.session_state.synthetic_data.values
                
                # Run comprehensive evaluation
                results = st.session_state.metrics_aggregator.run_comprehensive_evaluation(
                    model=st.session_state.model_loader.model,
                    X=X,
                    test_name=test_name,
                    n_iterations=n_iterations
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
                
                # Save results to storage
                file_path = st.session_state.results_storage.save_test_results(results, test_name)
                st.success(f"✅ Tests completed successfully! Results saved to {file_path}")
                
            except Exception as e:
                st.error(f"❌ Error running tests: {str(e)}")
                logger.error(f"Test error: {str(e)}", exc_info=True)
    
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

if __name__ == "__main__":
    main()
