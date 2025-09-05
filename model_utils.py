"""
Model utilities for loading, introspection, and prediction.
Handles .pkl model files and provides schema inference capabilities.
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles model loading and introspection."""
    
    def __init__(self):
        self.model = None
        self.model_info = {}
        self.feature_names = []
        self.feature_types = []
        
    def load_model(self, model_path: str) -> bool:
        """
        Load a model from a .pkl file.
        
        Args:
            model_path: Path to the .pkl file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try different loading methods
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except:
                # Fallback to joblib
                self.model = joblib.load(model_path)
            
            logger.info(f"Successfully loaded model from {model_path}")
            self._introspect_model()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def _introspect_model(self):
        """Extract information about the model."""
        if self.model is None:
            return
            
        self.model_info = {
            'type': type(self.model).__name__,
            'module': type(self.model).__module__
        }
        
        # Try to extract feature information
        self._extract_feature_info()
        
    def _extract_feature_info(self):
        """Extract feature names and types from the model."""
        if self.model is None:
            return
            
        # Handle different model types
        if hasattr(self.model, 'feature_names_in_'):
            # scikit-learn 1.0+ models
            self.feature_names = list(self.model.feature_names_in_)
            self.feature_types = ['float64'] * len(self.feature_names)
            
        elif hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            n_features = len(self.model.feature_importances_)
            self.feature_names = [f'feature_{i}' for i in range(n_features)]
            self.feature_types = ['float64'] * n_features
            
        elif hasattr(self.model, 'coef_'):
            # Linear models
            if self.model.coef_.ndim == 1:
                n_features = len(self.model.coef_)
            else:
                n_features = self.model.coef_.shape[1]
            self.feature_names = [f'feature_{i}' for i in range(n_features)]
            self.feature_types = ['float64'] * n_features
            
        elif hasattr(self.model, 'n_features_in_'):
            # Generic scikit-learn models
            n_features = self.model.n_features_in_
            self.feature_names = [f'feature_{i}' for i in range(n_features)]
            self.feature_types = ['float64'] * n_features
            
        elif isinstance(self.model, Pipeline):
            # Handle pipelines
            self._extract_pipeline_info()
            
        else:
            # Fallback: assume 10 features
            logger.warning("Could not determine number of features, defaulting to 10")
            self.feature_names = [f'feature_{i}' for i in range(10)]
            self.feature_types = ['float64'] * 10
    
    def _extract_pipeline_info(self):
        """Extract feature info from sklearn Pipeline."""
        try:
            # Get the final estimator
            final_estimator = self.model.steps[-1][1]
            
            # Try to get feature names from the pipeline
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
                self.feature_types = ['float64'] * len(self.feature_names)
            else:
                # Estimate from the final estimator
                if hasattr(final_estimator, 'n_features_in_'):
                    n_features = final_estimator.n_features_in_
                elif hasattr(final_estimator, 'feature_importances_'):
                    n_features = len(final_estimator.feature_importances_)
                else:
                    n_features = 10  # fallback
                    
                self.feature_names = [f'feature_{i}' for i in range(n_features)]
                self.feature_types = ['float64'] * n_features
                
        except Exception as e:
            logger.warning(f"Could not extract pipeline info: {str(e)}")
            self.feature_names = [f'feature_{i}' for i in range(10)]
            self.feature_types = ['float64'] * 10
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("No model loaded")
            
        try:
            # Convert numpy array to DataFrame if model expects it
            if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
                # Model expects DataFrame with specific column names
                X_df = pd.DataFrame(X, columns=self.model.feature_names_in_)
                return self.model.predict(X_df)
            else:
                # Model can handle numpy array
                return self.model.predict(X)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Get prediction probabilities if available.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray or None: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded")
            
        try:
            if hasattr(self.model, 'predict_proba'):
                # Convert numpy array to DataFrame if model expects it
                if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
                    # Model expects DataFrame with specific column names
                    X_df = pd.DataFrame(X, columns=self.model.feature_names_in_)
                    return self.model.predict_proba(X_df)
                else:
                    # Model can handle numpy array
                    return self.model.predict_proba(X)
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not get probabilities: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_info': self.model_info,
            'feature_names': self.feature_names,
            'feature_types': self.feature_types,
            'n_features': len(self.feature_names),
            'has_proba': hasattr(self.model, 'predict_proba') if self.model else False
        }
    
    def validate_input_shape(self, X: np.ndarray) -> bool:
        """
        Validate that input has the correct shape for the model.
        
        Args:
            X: Input features
            
        Returns:
            bool: True if shape is valid
        """
        if self.model is None:
            return False
            
        expected_features = len(self.feature_names)
        if X.ndim == 1:
            return X.shape[0] == expected_features
        else:
            return X.shape[1] == expected_features


def create_sample_model() -> str:
    """
    Create a sample model for testing purposes.
    
    Returns:
        str: Path to the created model file
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, 
                              n_redundant=1, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = "sample_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Created sample model at {model_path}")
    return model_path


if __name__ == "__main__":
    # Test the model loader
    loader = ModelLoader()
    
    # Create and load a sample model
    model_path = create_sample_model()
    success = loader.load_model(model_path)
    
    if success:
        print("Model loaded successfully!")
        print("Model info:", loader.get_model_info())
        
        # Test prediction
        X_test = np.random.random((5, 5))
        predictions = loader.predict(X_test)
        print("Sample predictions:", predictions)
    else:
        print("Failed to load model")
