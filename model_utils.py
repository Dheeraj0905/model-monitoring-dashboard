"""
Model utilities for loading and prediction.
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles model loading and predictions with pipeline support."""
    
    def __init__(self):
        self.model = None
        self.model_info = {}
        self.feature_names = []
        self.feature_types = []
        self.is_pipeline = False
        self.raw_feature_names = []  # For pipeline: original raw feature names
        
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
        
        # Check if model is a pipeline
        self.is_pipeline = isinstance(self.model, Pipeline)
        
        self.model_info = {
            'type': type(self.model).__name__,
            'module': type(self.model).__module__,
            'is_pipeline': self.is_pipeline
        }
        
        # Try to extract feature information
        self._extract_feature_info()
        
    def _extract_feature_info(self):
        """Extract feature names and types from the model."""
        if self.model is None:
            return
        
        if self.is_pipeline:
            # For pipelines, we need to extract from the first step (preprocessing)
            # and final step (model) separately
            self._extract_pipeline_features()
        else:
            # Handle regular models
            self._extract_regular_model_features()
    
    def _extract_pipeline_features(self):
        """Extract feature information from a scikit-learn pipeline."""
        try:
            # Multiple strategies to extract feature names
            
            # Strategy 1: From the pipeline itself
            if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
                self.raw_feature_names = list(self.model.feature_names_in_)
                self.feature_names = list(self.model.feature_names_in_)
                self.feature_types = ['object'] * len(self.feature_names)
                return
            
            # Strategy 2: From the first transformer
            if len(self.model.steps) > 0:
                first_step = self.model.steps[0][1]
                if hasattr(first_step, 'feature_names_in_') and first_step.feature_names_in_ is not None:
                    self.raw_feature_names = list(first_step.feature_names_in_)
                
                # Strategy 3: From the final estimator
                final_estimator = self.model.steps[-1][1]
                if hasattr(final_estimator, 'feature_names_in_') and final_estimator.feature_names_in_ is not None:
                    self.feature_names = list(final_estimator.feature_names_in_)
                elif hasattr(final_estimator, 'n_features_in_'):
                    n_features = final_estimator.n_features_in_
                    self.feature_names = [f'feature_{i}' for i in range(n_features)]
                
                # Infer feature types based on transformers
                self._infer_feature_types_from_pipeline()
            
            # Fallback: use generic names if nothing detected
            if not self.raw_feature_names and self.feature_names:
                self.raw_feature_names = [f'raw_feature_{i}' for i in range(len(self.feature_names))]
            elif not self.feature_names:
                self._fallback_feature_extraction()
                
        except Exception as e:
            logger.warning(f"Could not extract pipeline features: {e}")
            self._fallback_feature_extraction()
    
    def _infer_feature_types_from_pipeline(self):
        """Infer feature types from pipeline transformers."""
        if not self.feature_names:
            return
        
        # Default to mixed types for pipelines
        self.feature_types = ['object'] * len(self.feature_names)
        
        # Try to infer from specific transformers
        for step_name, transformer in self.model.steps[:-1]:  # Exclude final estimator
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    feature_names_out = transformer.get_feature_names_out()
                    if len(feature_names_out) == len(self.feature_names):
                        # Check for numerical vs categorical patterns
                        for i, name in enumerate(feature_names_out):
                            if 'num__' in name or 'scaler' in step_name.lower():
                                self.feature_types[i] = 'float64'
                            elif 'cat__' in name or 'encoder' in step_name.lower():
                                self.feature_types[i] = 'category'
                except:
                    pass
    
    def _extract_regular_model_features(self):
        """Extract feature information from regular (non-pipeline) models."""
        try:
            # Multiple strategies for feature detection
            
            # Strategy 1: scikit-learn 1.0+ models with feature_names_in_
            if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
                self.feature_names = list(self.model.feature_names_in_)
                # Try to infer types from feature names
                self.feature_types = []
                for name in self.feature_names:
                    if any(keyword in name.lower() for keyword in ['category', 'cat', 'type', 'class']):
                        self.feature_types.append('category')
                    elif any(keyword in name.lower() for keyword in ['id', 'count', 'num', 'age', 'year']):
                        self.feature_types.append('int64')
                    else:
                        self.feature_types.append('float64')
                return
            
            # Strategy 2: Tree-based models with feature_importances_
            elif hasattr(self.model, 'feature_importances_') and self.model.feature_importances_ is not None:
                n_features = len(self.model.feature_importances_)
                self.feature_names = [f'feature_{i}' for i in range(n_features)]
                self.feature_types = ['float64'] * n_features
                return
            
            # Strategy 3: Linear models with coefficients
            elif hasattr(self.model, 'coef_') and self.model.coef_ is not None:
                if self.model.coef_.ndim == 1:
                    n_features = len(self.model.coef_)
                else:
                    n_features = self.model.coef_.shape[1]
                self.feature_names = [f'feature_{i}' for i in range(n_features)]
                self.feature_types = ['float64'] * n_features
                return
            
            # Strategy 4: Generic scikit-learn models with n_features_in_
            elif hasattr(self.model, 'n_features_in_') and self.model.n_features_in_ is not None:
                n_features = self.model.n_features_in_
                self.feature_names = [f'feature_{i}' for i in range(n_features)]
                self.feature_types = ['float64'] * n_features
                return
            
            # Strategy 5: Try to inspect the model for other attributes
            else:
                self._inspect_model_attributes()
                
        except Exception as e:
            logger.warning(f"Error in regular model feature extraction: {e}")
            self._fallback_feature_extraction()
    
    def _inspect_model_attributes(self):
        """Inspect model for any available feature information."""
        try:
            # Check various common attributes
            for attr in ['n_features_', 'n_features', 'input_features_', 'feature_count_']:
                if hasattr(self.model, attr):
                    n_features = getattr(self.model, attr)
                    if isinstance(n_features, int) and n_features > 0:
                        self.feature_names = [f'feature_{i}' for i in range(n_features)]
                        self.feature_types = ['float64'] * n_features
                        logger.info(f"Found {n_features} features using attribute '{attr}'")
                        return
            
            # If nothing found, use fallback
            self._fallback_feature_extraction()
            
        except Exception as e:
            logger.warning(f"Error inspecting model attributes: {e}")
            self._fallback_feature_extraction()
    
    def _fallback_feature_extraction(self):
        """Fallback method when feature extraction fails."""
        logger.warning("Using fallback feature extraction")
        # Default to 5 float features
        self.feature_names = [f'feature_{i}' for i in range(5)]
        self.feature_types = ['float64'] * 5
        if self.is_pipeline:
            self.raw_feature_names = [f'raw_feature_{i}' for i in range(5)]
    
    
    def predict_with_raw_data(self, raw_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using raw data with a pipeline.
        
        Args:
            raw_data: Raw input data as DataFrame
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        if not self.is_pipeline:
            raise ValueError("Model is not a pipeline. Use predict() method instead.")
        
        try:
            # Pipeline handles all preprocessing automatically
            return self.model.predict(raw_data)
        except Exception as e:
            logger.error(f"Pipeline prediction failed: {str(e)}")
            raise
    
    def predict_proba_with_raw_data(self, raw_data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get prediction probabilities using raw data with a pipeline.
        
        Args:
            raw_data: Raw input data as DataFrame
            
        Returns:
            np.ndarray or None: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        if not self.is_pipeline:
            raise ValueError("Model is not a pipeline. Use predict_proba() method instead.")
        
        try:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(raw_data)
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not get probabilities from pipeline: {str(e)}")
            return None
    
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
            'raw_feature_names': self.raw_feature_names if self.is_pipeline else self.feature_names,
            'n_features': len(self.feature_names),
            'n_raw_features': len(self.raw_feature_names) if self.is_pipeline else len(self.feature_names),
            'is_pipeline': self.is_pipeline,
            'has_proba': hasattr(self.model, 'predict_proba') if self.model else False,
            'model_type': self._detect_model_type(),
            'task_type': self._detect_task_type()
        }
    
    def _detect_model_type(self) -> str:
        """Detect the type of model (classifier, regressor, etc.)."""
        if self.model is None:
            return "unknown"
        
        model_name = type(self.model).__name__.lower()
        
        if self.is_pipeline:
            # For pipelines, check the final estimator
            final_estimator = self.model.steps[-1][1]
            model_name = type(final_estimator).__name__.lower()
        
        if 'classifier' in model_name or 'classification' in model_name:
            return "classifier"
        elif 'regressor' in model_name or 'regression' in model_name:
            return "regressor"
        elif hasattr(self.model, 'predict_proba'):
            return "classifier"
        else:
            return "regressor"  # Default assumption
    
    def _detect_task_type(self) -> str:
        """Detect whether this is a classification or regression task."""
        model_type = self._detect_model_type()
        if model_type == "classifier":
            return "classification"
        elif model_type == "regressor":
            return "regression"
        else:
            return "unknown"
    
    def calculate_metrics(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Calculate appropriate metrics based on model type."""
        if self.model is None:
            raise ValueError("No model loaded")
        
        task_type = self._detect_task_type()
        
        try:
            if self.is_pipeline and isinstance(X, pd.DataFrame):
                predictions = self.predict_with_raw_data(X)
                probabilities = self.predict_proba_with_raw_data(X) if hasattr(self.model, 'predict_proba') else None
            else:
                predictions = self.predict(X)
                probabilities = self.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
            
            if task_type == "classification":
                return self._calculate_classification_metrics(y_true, predictions, probabilities)
            elif task_type == "regression":
                return self._calculate_regression_metrics(y_true, predictions)
            else:
                return {"error": "Unknown task type"}
                
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate classification metrics."""
        import sklearn.metrics as metrics
        
        try:
            results = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
                'unique_classes': sorted(list(set(y_true)) + list(set(y_pred)))
            }
            
            # Add AUC if probabilities are available and it's binary classification
            if y_proba is not None and len(np.unique(y_true)) == 2:
                try:
                    results['auc_roc'] = float(metrics.roc_auc_score(y_true, y_proba[:, 1]))
                except:
                    pass
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate regression metrics."""
        
        try:
            results = {
                'r2_score': float(r2_score(y_true, y_pred)),
                'mean_squared_error': float(mean_squared_error(y_true, y_pred)),
                'root_mean_squared_error': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mean_absolute_error': float(mean_absolute_error(y_true, y_pred)),
                'mean_absolute_percentage_error': float(np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8))) * 100),
                'max_error': float(np.max(np.abs(y_true - y_pred))),
                'residuals': (y_true - y_pred).tolist()[:100]  # First 100 residuals for plotting
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {str(e)}")
            return {"error": str(e)}
    
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
