"""
Example script to create a scikit-learn pipeline for testing the dashboard.

This script demonstrates how to create a complete ML pipeline that includes
preprocessing steps and a trained model, then save it as a .pkl file for
use with the model monitoring dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

def create_classification_pipeline():
    """Create a classification pipeline with preprocessing."""
    
    # Generate sample data with mixed feature types
    np.random.seed(42)
    
    # Create numerical features
    X_num, y = make_classification(
        n_samples=1000, 
        n_features=3, 
        n_informative=2, 
        n_redundant=1, 
        n_classes=2, 
        random_state=42
    )
    
    # Add categorical features
    categories = ['Category_A', 'Category_B', 'Category_C']
    cat_feature = np.random.choice(categories, size=1000)
    
    # Create DataFrame with meaningful column names
    df = pd.DataFrame(X_num, columns=['income', 'age', 'credit_score'])
    df['category'] = cat_feature
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['income', 'age', 'credit_score']),
            ('cat', OneHotEncoder(drop='first'), ['category'])
        ]
    )
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    
    # Train pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"Classification Pipeline Performance:")
    print(f"Training Score: {train_score:.3f}")
    print(f"Test Score: {test_score:.3f}")
    
    # Save pipeline
    pipeline_path = "classification_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"Pipeline saved to: {pipeline_path}")
    
    # Save sample data for testing
    test_data_path = "classification_test_data.csv"
    X_test.to_csv(test_data_path, index=False)
    print(f"Test data saved to: {test_data_path}")
    
    return pipeline, X_test, y_test

def create_regression_pipeline():
    """Create a regression pipeline with preprocessing."""
    
    # Generate sample data
    np.random.seed(42)
    
    # Create numerical features
    X_num, y = make_regression(
        n_samples=1000, 
        n_features=4, 
        noise=0.1, 
        random_state=42
    )
    
    # Add categorical feature
    regions = ['North', 'South', 'East', 'West']
    region_feature = np.random.choice(regions, size=1000)
    
    # Create DataFrame
    df = pd.DataFrame(X_num, columns=['house_size', 'bedrooms', 'bathrooms', 'lot_size'])
    df['region'] = region_feature
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['house_size', 'bedrooms', 'bathrooms', 'lot_size']),
            ('cat', OneHotEncoder(drop='first'), ['region'])
        ]
    )
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    
    # Train pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"\nRegression Pipeline Performance:")
    print(f"Training Score: {train_score:.3f}")
    print(f"Test Score: {test_score:.3f}")
    
    # Save pipeline
    pipeline_path = "regression_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"Pipeline saved to: {pipeline_path}")
    
    # Save sample data for testing
    test_data_path = "regression_test_data.csv"
    X_test.to_csv(test_data_path, index=False)
    print(f"Test data saved to: {test_data_path}")
    
    return pipeline, X_test, y_test

def create_simple_pipeline():
    """Create a simple pipeline for basic testing."""
    
    # Generate simple data
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=5, n_classes=2, random_state=42)
    
    # Create DataFrame with simple column names
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    
    # Simple pipeline with just scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    test_score = pipeline.score(X_test, y_test)
    print(f"\nSimple Pipeline Performance:")
    print(f"Test Score: {test_score:.3f}")
    
    # Save
    pipeline_path = "simple_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"Pipeline saved to: {pipeline_path}")
    
    # Save test data
    test_data_path = "simple_test_data.csv"
    X_test.to_csv(test_data_path, index=False)
    print(f"Test data saved to: {test_data_path}")
    
    return pipeline, X_test, y_test

if __name__ == "__main__":
    print("Creating example pipelines for the Model Monitoring Dashboard...")
    print("=" * 60)
    
    # Create different types of pipelines
    try:
        # Classification with mixed features
        print("1. Creating Classification Pipeline with Mixed Features:")
        create_classification_pipeline()
        
        # Regression with mixed features
        print("\n2. Creating Regression Pipeline with Mixed Features:")
        create_regression_pipeline()
        
        # Simple pipeline
        print("\n3. Creating Simple Pipeline:")
        create_simple_pipeline()
        
        print("\n" + "=" * 60)
        print("✅ All pipelines created successfully!")
        print("\nTo use these with the dashboard:")
        print("1. Run the dashboard: python -m streamlit run app.py")
        print("2. Upload one of the .pkl files")
        print("3. Use the corresponding CSV file for raw data testing")
        print("\nFiles created:")
        print("- classification_pipeline.pkl + classification_test_data.csv")
        print("- regression_pipeline.pkl + regression_test_data.csv") 
        print("- simple_pipeline.pkl + simple_test_data.csv")
        
    except Exception as e:
        print(f"❌ Error creating pipelines: {str(e)}")
        raise
