"""
Simple model utilities for loading and prediction.
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModelLoader:
    """Simple model loader for basic model operations."""
    
    def __init__(self):
        self.model = None
        self.model_info = {}
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a model from a file.
        
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
            self._extract_model_info()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def _extract_model_info(self):
        """Extract basic information about the model."""
        if self.model is None:
            return
        
        self.model_info = {
            'type': type(self.model).__name__,
            'module': type(self.model).__module__,
            'has_predict': hasattr(self.model, 'predict'),
            'has_predict_proba': hasattr(self.model, 'predict_proba')
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            X: Input features as DataFrame
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        try:
            # Convert DataFrame to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            
            predictions = self.model.predict(X_array)
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get prediction probabilities if available.
        
        Args:
            X: Input features as DataFrame
            
        Returns:
            np.ndarray or None: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        if not hasattr(self.model, 'predict_proba'):
            return None
        
        try:
            # Convert DataFrame to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            
            probabilities = self.model.predict_proba(X_array)
            return probabilities
            
        except Exception as e:
            logger.error(f"Probability prediction failed: {str(e)}")
            return None
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate classification metrics."""
        try:
            results = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
            return results
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {str(e)}")
            return {"error": str(e)}
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate regression metrics."""
        try:
            results = {
                'r2': r2_score(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
            }
            return results
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {str(e)}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {"error": "No model loaded"}
        
        return self.model_info


# For backward compatibility
ModelLoader = SimpleModelLoader
