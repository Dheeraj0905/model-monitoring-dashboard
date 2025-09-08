"""
Example client script for API-driven model testing.
This script demonstrates how to use the dashboard programmatically.
"""

import requests
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import time


def create_sample_model():
    """Create a sample model for testing."""
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, 
                              n_redundant=1, n_classes=2, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save the model
    model_path = "sample_model_for_api.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Created sample model: {model_path}")
    return model_path


def send_model_to_dashboard(model_path, api_url="http://localhost:8501/api/run_tests"):
    """
    Send model to dashboard API for testing.
    
    Args:
        model_path: Path to the model file
        api_url: Dashboard API endpoint URL
    """
    try:
        # Prepare the request
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Create sample schema (optional)
        schema = {
            "features": [
                {"name": "feature_0", "data_type": "float64", "min_val": -10.0, "max_val": 10.0},
                {"name": "feature_1", "data_type": "float64", "min_val": -10.0, "max_val": 10.0},
                {"name": "feature_2", "data_type": "float64", "min_val": -10.0, "max_val": 10.0},
                {"name": "feature_3", "data_type": "float64", "min_val": -10.0, "max_val": 10.0},
                {"name": "feature_4", "data_type": "float64", "min_val": -10.0, "max_val": 10.0}
            ]
        }
        
        # Prepare files for upload
        files = {
            'model': ('model.pkl', model_data, 'application/octet-stream')
        }
        
        # Prepare data
        data = {
            'test_name': f'api_test_{int(time.time())}',
            'n_samples': 1000,
            'include_drift_test': 'true',
            'include_explainability': 'true',
            'schema': json.dumps(schema)
        }
        
        print(f"Sending model to dashboard API: {api_url}")
        print(f"Test name: {data['test_name']}")
        
        # Send request
        response = requests.post(api_url, files=files, data=data, timeout=300)
        
        if response.status_code == 200:
            results = response.json()
            print("✅ Model testing completed successfully!")
            print("\n📊 Results Summary:")
            print(f"Test Name: {results.get('test_name', 'Unknown')}")
            print(f"Samples: {results.get('n_samples', 0)}")
            print(f"Features: {results.get('n_features', 0)}")
            
            # Performance metrics
            if 'performance' in results:
                perf = results['performance']
                print(f"\n⚡ Performance Metrics:")
                print(f"  Latency: {perf.get('latency_mean_ms', 0):.2f} ms")
                print(f"  Throughput: {perf.get('throughput_preds_per_sec', 0):.0f} pred/s")
                print(f"  Error Rate: {perf.get('error_rate', 0):.2%}")
                if perf.get('accuracy'):
                    print(f"  Accuracy: {perf.get('accuracy', 0):.2%}")
            
            # Drift detection
            if 'drift' in results:
                drift = results['drift']
                print(f"\n🌊 Drift Detection:")
                print(f"  Overall Drift Score: {drift.get('overall_drift_score', 0):.2f}")
                print(f"  Drift Detected: {drift.get('overall_drift_detected', False)}")
            
            # Explainability
            if 'explainability' in results:
                explain = results['explainability']
                if 'explanations' in explain and 'feature_importance' in explain['explanations']:
                    importance = explain['explanations']['feature_importance']
                    print(f"\n📈 Top 3 Most Important Features:")
                    for i, (feature, imp) in enumerate(list(importance.items())[:3], 1):
                        print(f"  {i}. {feature}: {imp:.4f}")
            
            return results
            
        else:
            print(f"❌ API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def main():
    """Main function to demonstrate API usage."""
    print("🚀 ML Model Monitoring Dashboard - API Client Example")
    print("=" * 60)
    
    # Create a sample model
    print("\n1. Creating sample model...")
    model_path = create_sample_model()
    
    # Note: This is a placeholder URL since we're implementing Phase 1 (UI only)
    # In Phase 2, this would be the actual API endpoint
    api_url = "http://localhost:8501/api/run_tests"
    
    print(f"\n2. Sending model to dashboard API...")
    print("Note: This is a demonstration. The actual API endpoint will be available in Phase 2.")
    print("For now, you can use the Streamlit dashboard UI to test your models.")
    
    # Simulate what the API response would look like
    print("\n📊 Simulated API Response:")
    simulated_results = {
        "test_name": f"api_test_{int(time.time())}",
        "timestamp": "2024-01-01T12:00:00",
        "n_samples": 1000,
        "n_features": 5,
        "performance": {
            "latency_mean_ms": 2.5,
            "throughput_preds_per_sec": 400,
            "error_rate": 0.0,
            "accuracy": 0.95
        },
        "drift": {
            "overall_drift_score": 0.15,
            "overall_drift_detected": False,
            "summary": {
                "n_features_with_drift": 0,
                "total_features": 5
            }
        },
        "explainability": {
            "explanations": {
                "feature_importance": {
                    "feature_2": 0.25,
                    "feature_0": 0.20,
                    "feature_1": 0.18,
                    "feature_3": 0.15,
                    "feature_4": 0.12
                }
            }
        }
    }
    
    print(json.dumps(simulated_results, indent=2))
    
    print("\n" + "=" * 60)
    print("✅ Example completed!")
    print("\nTo use the actual dashboard:")
    print("1. Run: streamlit run app.py")
    print("2. Open: http://localhost:8501")
    print("3. Upload your model and run tests through the UI")
    print("\nPhase 2 will add the API endpoints for programmatic access.")


if __name__ == "__main__":
    main()
