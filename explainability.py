"""
Explainability utilities using SHAP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
import warnings
from io import BytesIO
import base64

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
plt.style.use('default')


class SHAPExplainer:
    """Handles SHAP-based model explainability."""
    
    def __init__(self):
        self.explainer = None
        self.feature_names = []
        self.model = None
        self.background_data = None
        
    def setup_explainer(self, model, X_background: np.ndarray, 
                       feature_names: Optional[List[str]] = None):
        """
        Set up SHAP explainer with background data.
        
        Args:
            model: Trained model
            X_background: Background data for explainer
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not available. Please install it with: pip install shap")
        
        self.model = model
        self.background_data = X_background
        
        if feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(X_background.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Choose appropriate explainer based on model type
        try:
            if hasattr(model, 'predict_proba'):
                # Classification model
                self.explainer = shap.Explainer(model, X_background)
            else:
                # Regression model
                self.explainer = shap.Explainer(model, X_background)
        except Exception as e:
            logger.warning(f"Failed to create SHAP explainer: {str(e)}")
            # Fallback to TreeExplainer for tree-based models
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception as e2:
                logger.warning(f"TreeExplainer also failed: {str(e2)}")
                # Final fallback to LinearExplainer
                try:
                    self.explainer = shap.LinearExplainer(model, X_background)
                except Exception as e3:
                    logger.error(f"All SHAP explainers failed: {str(e3)}")
                    raise
        
        logger.info(f"SHAP explainer set up successfully with {len(self.feature_names)} features")
    
    def explain_predictions(self, X: np.ndarray, max_samples: int = 100) -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions.
        
        Args:
            X: Input data to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            Dict with SHAP values and explanations
        """
        if self.explainer is None:
            raise ValueError("Explainer not set up. Call setup_explainer() first.")
        
        # Limit samples for performance
        if len(X) > max_samples:
            X_sample = X[:max_samples]
            logger.info(f"Limited explanation to {max_samples} samples for performance")
        else:
            X_sample = X
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer(X_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-output model
                shap_values_array = shap_values[0].values if len(shap_values) > 0 else np.array([])
            else:
                # Single output model
                shap_values_array = shap_values.values
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(shap_values_array)
            
            # Get prediction explanations
            explanations = self._get_prediction_explanations(X_sample, shap_values_array)
            
            results = {
                'shap_values': shap_values_array.tolist() if shap_values_array.size > 0 else [],
                'feature_importance': feature_importance,
                'explanations': explanations,
                'n_samples_explained': len(X_sample),
                'feature_names': self.feature_names
            }
            
            return results
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance from SHAP values."""
        if shap_values.size == 0:
            return {}
        
        # Calculate mean absolute SHAP values
        if shap_values.ndim > 2:
            # Multi-output case
            importance = np.mean(np.abs(shap_values), axis=(0, 1))
        else:
            # Single output case
            importance = np.mean(np.abs(shap_values), axis=0)
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, imp in enumerate(importance):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
            feature_importance[feature_name] = float(imp)
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def _get_prediction_explanations(self, X: np.ndarray, shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """Get explanations for individual predictions."""
        explanations = []
        
        if shap_values.size == 0:
            return explanations
        
        for i in range(len(X)):
            explanation = {
                'sample_index': i,
                'input_values': X[i].tolist(),
                'feature_contributions': {}
            }
            
            # Get SHAP values for this sample
            if shap_values.ndim > 2:
                sample_shap = shap_values[i, :, 0]  # First output for multi-output
            else:
                sample_shap = shap_values[i, :]
            
            # Create feature contributions
            for j, shap_val in enumerate(sample_shap):
                feature_name = self.feature_names[j] if j < len(self.feature_names) else f'feature_{j}'
                explanation['feature_contributions'][feature_name] = {
                    'shap_value': float(shap_val),
                    'input_value': float(X[i, j]) if j < X.shape[1] else 0.0
                }
            
            explanations.append(explanation)
        
        return explanations
    
    def create_summary_plot(self, shap_values: np.ndarray, 
                           max_display: int = 10) -> str:
        """
        Create SHAP summary plot as base64 encoded image.
        
        Args:
            shap_values: SHAP values array
            max_display: Maximum number of features to display
            
        Returns:
            str: Base64 encoded image
        """
        if not SHAP_AVAILABLE or shap_values.size == 0:
            return ""
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[0] if len(shap_values) > 0 else None
            else:
                shap_values_plot = shap_values
            
            if shap_values_plot is not None:
                # Create summary plot
                shap.summary_plot(shap_values_plot, 
                                feature_names=self.feature_names,
                                max_display=max_display,
                                show=False)
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to create summary plot: {str(e)}")
            return ""
    
    def create_waterfall_plot(self, sample_idx: int, shap_values: np.ndarray, 
                             X: np.ndarray) -> str:
        """
        Create SHAP waterfall plot for a specific sample.
        
        Args:
            sample_idx: Index of the sample to explain
            shap_values: SHAP values array
            X: Input data
            
        Returns:
            str: Base64 encoded image
        """
        if not SHAP_AVAILABLE or shap_values.size == 0:
            return ""
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get SHAP values for the sample
            if shap_values.ndim > 2:
                sample_shap = shap_values[sample_idx, :, 0]
            else:
                sample_shap = shap_values[sample_idx, :]
            
            # Create waterfall plot
            shap.waterfall_plot(shap.Explanation(values=sample_shap,
                                               data=X[sample_idx],
                                               feature_names=self.feature_names),
                               show=False)
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to create waterfall plot: {str(e)}")
            return ""


class ExplainabilityAnalyzer:
    """Comprehensive explainability analysis."""
    
    def __init__(self):
        self.shap_explainer = SHAPExplainer()
    
    def analyze_model(self, model, X: np.ndarray, 
                     feature_names: Optional[List[str]] = None,
                     background_samples: int = 100) -> Dict[str, Any]:
        """Perform explainability analysis."""
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not available. Please install with: pip install shap'}
        
        try:
            # Prepare background data
            if len(X) > background_samples:
                background_data = X[:background_samples]
            else:
                background_data = X
            
            # Set up explainer
            self.shap_explainer.setup_explainer(model, background_data, feature_names)
            
            # Generate explanations
            explanations = self.shap_explainer.explain_predictions(X)
            
            # Create visualizations
            visualizations = self._create_visualizations(X, explanations)
            
            # Compile results
            results = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'n_samples': len(X),
                'n_features': X.shape[1] if X.ndim > 1 else 1,
                'explanations': explanations,
                'visualizations': visualizations,
                'summary': self._create_summary(explanations)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Explainability analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _create_visualizations(self, X: np.ndarray, explanations: Dict[str, Any]) -> Dict[str, str]:
        """Create explainability visualizations."""
        visualizations = {}
        
        try:
            # Get SHAP values for plotting
            shap_values = np.array(explanations.get('shap_values', []))
            
            if shap_values.size > 0:
                # Summary plot
                summary_plot = self.shap_explainer.create_summary_plot(shap_values)
                if summary_plot:
                    visualizations['summary_plot'] = summary_plot
                
                # Waterfall plot for first sample
                if len(X) > 0:
                    waterfall_plot = self.shap_explainer.create_waterfall_plot(0, shap_values, X)
                    if waterfall_plot:
                        visualizations['waterfall_plot'] = waterfall_plot
            
        except Exception as e:
            logger.warning(f"Visualization creation failed: {str(e)}")
        
        return visualizations
    
    def _create_summary(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of explainability results."""
        feature_importance = explanations.get('feature_importance', {})
        
        if not feature_importance:
            return {'message': 'No feature importance data available'}
        
        # Get top features
        top_features = list(feature_importance.items())[:5]
        
        # Calculate statistics
        importance_values = list(feature_importance.values())
        total_importance = sum(importance_values)
        
        summary = {
            'top_features': top_features,
            'total_features': len(feature_importance),
            'max_importance': max(importance_values) if importance_values else 0,
            'min_importance': min(importance_values) if importance_values else 0,
            'mean_importance': np.mean(importance_values) if importance_values else 0,
            'importance_std': np.std(importance_values) if importance_values else 0
        }
        
        # Calculate feature contribution percentages
        if total_importance > 0:
            summary['top_feature_percentage'] = (top_features[0][1] / total_importance * 100) if top_features else 0
            summary['top_3_features_percentage'] = sum(imp for _, imp in top_features[:3]) / total_importance * 100
        
        return summary
    
    def compare_models(self, model1_results: Dict[str, Any], 
                      model2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare explainability between two models."""
        try:
            importance1 = model1_results.get('explanations', {}).get('feature_importance', {})
            importance2 = model2_results.get('explanations', {}).get('feature_importance', {})
            
            # Find common features
            common_features = set(importance1.keys()) & set(importance2.keys())
            
            comparison = {
                'common_features': len(common_features),
                'feature_importance_correlation': 0.0,
                'top_feature_agreement': False,
                'detailed_comparison': {}
            }
            
            if len(common_features) > 1:
                # Calculate correlation
                common_imp1 = [importance1[f] for f in common_features]
                common_imp2 = [importance2[f] for f in common_features]
                
                correlation = np.corrcoef(common_imp1, common_imp2)[0, 1]
                comparison['feature_importance_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
                
                # Check top feature agreement
                top_feature1 = max(importance1.items(), key=lambda x: x[1])[0]
                top_feature2 = max(importance2.items(), key=lambda x: x[1])[0]
                comparison['top_feature_agreement'] = top_feature1 == top_feature2
                
                # Detailed comparison
                for feature in common_features:
                    comparison['detailed_comparison'][feature] = {
                        'model1_importance': importance1[feature],
                        'model2_importance': importance2[feature],
                        'difference': abs(importance1[feature] - importance2[feature])
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            return {'error': str(e)}


def create_sample_explainability_data() -> Tuple[np.ndarray, np.ndarray]:
    """Create sample data for testing explainability."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=5, n_classes=2, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return X, model


if __name__ == "__main__":
    # Test the explainability modules
    if SHAP_AVAILABLE:
        X, model = create_sample_explainability_data()
        
        # Test SHAP explainer
        explainer = SHAPExplainer()
        explainer.setup_explainer(model, X[:100])
        
        explanations = explainer.explain_predictions(X[:50])
        print("SHAP explanations generated successfully")
        print("Feature importance:", explanations.get('feature_importance', {}))
        
        # Test comprehensive analysis
        analyzer = ExplainabilityAnalyzer()
        results = analyzer.analyze_model(model, X[:100])
        print("Comprehensive analysis completed")
        print("Summary:", results.get('summary', {}))
    else:
        print("SHAP not available. Install with: pip install shap")
