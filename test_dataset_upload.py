#!/usr/bin/env python3
"""
Test script to verify dataset upload functionality
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_utils import ModelLoader
import pandas as pd
import numpy as np

def test_dataset_upload_functionality():
    """Test the key components of dataset upload functionality"""
    
    print("=== Testing Dataset Upload Functionality ===\n")
    
    # Test 1: Load the test model
    print("1. Testing model loading...")
    model_loader = ModelLoader()
    
    try:
        model_path = "example_client/test_model_with_features.pkl"
        success = model_loader.load_model(model_path)
        if success:
            model_info = model_loader.model_info
            print(f"✅ Model loaded successfully: {model_info.get('type', 'Unknown')}")
            print(f"   Pipeline detected: {model_info.get('is_pipeline', False)}")
            print(f"   Feature names: {getattr(model_loader, 'feature_names', [])[:5] if hasattr(model_loader, 'feature_names') else 'Not available'}...")
        else:
            print("❌ Model loading failed")
            return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test 2: Load test dataset
    print("\n2. Testing dataset loading...")
    try:
        dataset_path = "example_client/test_dataset_with_target.csv"
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # Test 3: Test prediction
    print("\n3. Testing model prediction...")
    try:
        # Get feature data (excluding target column)
        target_column = 'target'  # Adjust based on your dataset
        if target_column in df.columns:
            X_test = df.drop(columns=[target_column])
            y_true = df[target_column]
        else:
            X_test = df
            y_true = None
        
        predictions = model_loader.predict(X_test.iloc[:5])  # Test with first 5 rows
        print(f"✅ Predictions generated: {predictions[:5] if len(predictions) > 5 else predictions}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False
    
    # Test 4: Test metrics calculation (if we have true labels)
    if y_true is not None:
        print("\n4. Testing metrics calculation...")
        try:
            full_predictions = model_loader.predict(X_test)
            metrics = model_loader.calculate_metrics(y_true, full_predictions)
            print(f"✅ Metrics calculated successfully:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
        except Exception as e:
            print(f"❌ Metrics calculation failed: {e}")
            return False
    
    print("\n=== All tests passed! Dataset upload functionality is working ===")
    return True

if __name__ == "__main__":
    success = test_dataset_upload_functionality()
    if success:
        print("\n🎉 The dataset upload section should be working properly in the Streamlit app!")
        print("📊 You can now test it by:")
        print("   1. Opening http://localhost:8501 in your browser")
        print("   2. Navigating to '📊 Dataset Upload' in the sidebar")
        print("   3. Uploading the model file: example_client/test_model_with_features.pkl")
        print("   4. Uploading the dataset file: example_client/test_dataset_with_target.csv")
        print("   5. Selecting 'target' as the target column")
        print("   6. Clicking 'Evaluate Model' to see the results")
    else:
        print("\n❌ There are still issues with the dataset upload functionality")
        print("   Please check the error messages above")
