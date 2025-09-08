"""
Performance and drift metrics calculation.
Handles latency, throughput, error rate, accuracy, and drift detection.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix,
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from io import BytesIO
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('default')


class PerformanceMetrics:
    """Calculates performance metrics for model predictions."""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_prediction_metrics(self, model, X: np.ndarray, 
                                   y_true: Optional[np.ndarray] = None,
                                   n_iterations: int = 1) -> Dict[str, Any]:
        """
        Calculate comprehensive prediction metrics.
        
        Args:
            model: Trained model object
            X: Input features
            y_true: True labels (optional, for accuracy calculation)
            n_iterations: Number of iterations for timing measurements
            
        Returns:
            Dict with performance metrics
        """
        metrics = {}
        
        # Timing metrics
        timing_metrics = self._calculate_timing_metrics(model, X, n_iterations)
        metrics.update(timing_metrics)
        
        # Prediction metrics
        predictions = self._get_predictions(model, X)
        metrics['predictions'] = predictions.tolist()
        
        # Accuracy metrics (if true labels provided)
        if y_true is not None:
            accuracy_metrics = self._calculate_accuracy_metrics(predictions, y_true)
            metrics.update(accuracy_metrics)
        
        # Error rate
        error_rate = self._calculate_error_rate(model, X)
        metrics['error_rate'] = error_rate
        
        # Prediction statistics
        pred_stats = self._calculate_prediction_statistics(predictions)
        metrics.update(pred_stats)
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_timing_metrics(self, model, X: np.ndarray, n_iterations: int) -> Dict[str, float]:
        """Calculate timing-related metrics."""
        times = []
        
        try:
            # Get expected feature names from model
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                n_expected = len(expected_features)
                n_provided = X.shape[1]
                
                if n_expected != n_provided:
                    logger.warning(f"Feature count mismatch. Model expects {n_expected} features, but got {n_provided}")
                    # Use only the first n_expected features
                    X = X[:, :n_expected]
            
            # Run predictions
            for _ in range(n_iterations):
                start_time = time.time()
                _ = self._get_predictions(model, X)
                end_time = time.time()
                times.append(end_time - start_time)
            
            times = np.array(times)
            
            return {
                'latency_mean_ms': np.mean(times) * 1000,
                'latency_std_ms': np.std(times) * 1000,
                'latency_min_ms': np.min(times) * 1000,
                'latency_max_ms': np.max(times) * 1000,
                'latency_p95_ms': np.percentile(times, 95) * 1000,
                'latency_p99_ms': np.percentile(times, 99) * 1000,
                'throughput_preds_per_sec': len(X) / np.mean(times),
                'total_prediction_time_sec': np.sum(times)
            }
            
        except Exception as e:
            logger.error(f"Error in timing metrics calculation: {str(e)}")
            return {
                'latency_mean_ms': 0,
                'latency_std_ms': 0,
                'latency_min_ms': 0,
                'latency_max_ms': 0,
                'latency_p95_ms': 0,
                'latency_p99_ms': 0,
                'throughput_preds_per_sec': 0,
                'total_prediction_time_sec': 0
            }
    
    def _get_predictions(self, model, X: np.ndarray) -> np.ndarray:
        """Get model predictions safely."""
        try:
            # Handle feature name matching if model expects specific features
            if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
                n_expected = len(model.feature_names_in_)
                if X.shape[1] != n_expected:
                    logger.warning(f"Adjusting input features to match model expectations")
                    X = X[:, :n_expected]
                    
                # Create DataFrame with expected feature names
                X_df = pd.DataFrame(X, columns=model.feature_names_in_[:n_expected])
                return model.predict(X_df)
            else:
                # Model can handle numpy array directly
                return model.predict(X)
                
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return np.array([])
    
    def _calculate_accuracy_metrics(self, predictions: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy-related metrics."""
        try:
            # Handle different prediction types
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                # Multi-class or multi-output
                if y_true.ndim > 1 and y_true.shape[1] > 1:
                    # Multi-output regression
                    return self._calculate_regression_metrics(predictions, y_true)
                else:
                    # Multi-class classification
                    pred_classes = np.argmax(predictions, axis=1)
                    return {
                        'accuracy': accuracy_score(y_true, pred_classes),
                        'precision_macro': precision_score(y_true, pred_classes, average='macro', zero_division=0),
                        'recall_macro': recall_score(y_true, pred_classes, average='macro', zero_division=0),
                        'f1_macro': f1_score(y_true, pred_classes, average='macro', zero_division=0)
                    }
            else:
                # Binary classification or regression
                if self._is_classification_problem(y_true):
                    return {
                        'accuracy': accuracy_score(y_true, predictions),
                        'precision': precision_score(y_true, predictions, zero_division=0),
                        'recall': recall_score(y_true, predictions, zero_division=0),
                        'f1': f1_score(y_true, predictions, zero_division=0)
                    }
                else:
                    return self._calculate_regression_metrics(predictions, y_true)
                    
        except Exception as e:
            logger.warning(f"Could not calculate accuracy metrics: {str(e)}")
            return {'accuracy': None}
    
    def _is_classification_problem(self, y: np.ndarray) -> bool:
        """Determine if this is a classification problem."""
        # Check if all values are integers and have a reasonable number of unique values
        unique_values = np.unique(y)
        return (len(unique_values) <= 20 and 
                np.all(np.equal(np.mod(unique_values, 1), 0)))
    
    def _calculate_regression_metrics(self, predictions: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        try:
            return {
                'mse': mean_squared_error(y_true, predictions),
                'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
                'mae': mean_absolute_error(y_true, predictions),
                'r2': r2_score(y_true, predictions)
            }
        except Exception as e:
            logger.warning(f"Could not calculate regression metrics: {str(e)}")
            return {'mse': None, 'rmse': None, 'mae': None, 'r2': None}
    
    def _calculate_error_rate(self, model, X: np.ndarray) -> float:
        """Calculate error rate (percentage of failed predictions)."""
        try:
            # Convert numpy array to DataFrame if needed
            if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
                # Model expects DataFrame with specific column names
                X_df = pd.DataFrame(X, columns=model.feature_names_in_)
                predictions = model.predict(X_df)
            else:
                # Model can handle numpy array
                predictions = model.predict(X)
            return 0.0  # No errors if prediction succeeds
        except Exception as e:
            logger.warning(f"Prediction error: {str(e)}")
            return 1.0  # 100% error rate if prediction fails
    
    def _calculate_prediction_statistics(self, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistics of predictions."""
        if len(predictions) == 0:
            return {'pred_mean': None, 'pred_std': None, 'pred_min': None, 'pred_max': None}
        
        return {
            'pred_mean': float(np.mean(predictions)),
            'pred_std': float(np.std(predictions)),
            'pred_min': float(np.min(predictions)),
            'pred_max': float(np.max(predictions)),
            'pred_median': float(np.median(predictions))
        }


class DriftDetector:
    """Detects and measures data drift between datasets."""
    
    def __init__(self):
        self.reference_data = None
        self.reference_stats = {}
    
    def set_reference_data(self, data: pd.DataFrame):
        """Set reference data for drift detection."""
        self.reference_data = data.copy()
        self.reference_stats = self._calculate_data_statistics(data)
        logger.info(f"Set reference data with {len(data)} samples and {len(data.columns)} features")
    
    def detect_drift(self, new_data: pd.DataFrame, 
                    methods: List[str] = ['statistical', 'distribution']) -> Dict[str, Any]:
        """
        Detect drift between reference and new data.
        
        Args:
            new_data: New data to compare against reference
            methods: List of drift detection methods to use
            
        Returns:
            Dict with drift detection results
        """
        if self.reference_data is None:
            raise ValueError("No reference data set. Call set_reference_data() first.")
        
        drift_results = {
            'overall_drift_detected': False,
            'feature_drift': {},
            'summary': {}
        }
        
        # Calculate statistics for new data
        new_stats = self._calculate_data_statistics(new_data)
        
        # Detect drift for each feature
        for feature in self.reference_data.columns:
            if feature in new_data.columns:
                feature_drift = self._detect_feature_drift(
                    self.reference_data[feature], 
                    new_data[feature], 
                    methods
                )
                drift_results['feature_drift'][feature] = feature_drift
        
        # Calculate overall drift score
        drift_scores = [result['drift_score'] for result in drift_results['feature_drift'].values()]
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
        
        drift_results['overall_drift_detected'] = overall_drift_score > 0.5
        drift_results['overall_drift_score'] = overall_drift_score
        
        # Summary statistics
        drift_results['summary'] = {
            'n_features_with_drift': sum(1 for result in drift_results['feature_drift'].values() 
                                       if result['drift_detected']),
            'total_features': len(drift_results['feature_drift']),
            'drift_percentage': (sum(1 for result in drift_results['feature_drift'].values() 
                                   if result['drift_detected']) / 
                               len(drift_results['feature_drift']) * 100) if drift_results['feature_drift'] else 0
        }
        
        return drift_results
    
    def _calculate_data_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive statistics for a dataset."""
        stats = {}
        
        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                stats[column] = {
                    'mean': float(data[column].mean()),
                    'std': float(data[column].std()),
                    'min': float(data[column].min()),
                    'max': float(data[column].max()),
                    'median': float(data[column].median()),
                    'skewness': float(data[column].skew()),
                    'kurtosis': float(data[column].kurtosis())
                }
            else:
                # Categorical data
                value_counts = data[column].value_counts()
                stats[column] = {
                    'unique_count': int(data[column].nunique()),
                    'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_common_freq': float(value_counts.iloc[0]) if len(value_counts) > 0 else 0.0
                }
        
        return stats
    
    def _detect_feature_drift(self, ref_data: pd.Series, new_data: pd.Series, 
                             methods: List[str]) -> Dict[str, Any]:
        """Detect drift for a single feature."""
        drift_result = {
            'drift_detected': False,
            'drift_score': 0.0,
            'methods': {}
        }
        
        if pd.api.types.is_numeric_dtype(ref_data) and pd.api.types.is_numeric_dtype(new_data):
            # Numeric feature drift detection
            for method in methods:
                if method == 'statistical':
                    stat_result = self._statistical_drift_test(ref_data, new_data)
                    drift_result['methods']['statistical'] = stat_result
                    
                elif method == 'distribution':
                    dist_result = self._distribution_drift_test(ref_data, new_data)
                    drift_result['methods']['distribution'] = dist_result
        
        else:
            # Categorical feature drift detection
            cat_result = self._categorical_drift_test(ref_data, new_data)
            drift_result['methods']['categorical'] = cat_result
        
        # Calculate overall drift score
        method_scores = [result.get('drift_score', 0) for result in drift_result['methods'].values()]
        drift_result['drift_score'] = np.mean(method_scores) if method_scores else 0.0
        drift_result['drift_detected'] = drift_result['drift_score'] > 0.5
        
        return drift_result
    
    def _statistical_drift_test(self, ref_data: pd.Series, new_data: pd.Series) -> Dict[str, Any]:
        """Perform statistical tests for drift detection."""
        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_data.dropna(), new_data.dropna())
            
            # Mann-Whitney U test
            mw_stat, mw_pvalue = stats.mannwhitneyu(ref_data.dropna(), new_data.dropna(), 
                                                   alternative='two-sided')
            
            # Calculate drift score based on p-values
            drift_score = 1.0 - min(ks_pvalue, mw_pvalue)
            
            return {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'mw_statistic': float(mw_stat),
                'mw_pvalue': float(mw_pvalue),
                'drift_score': float(drift_score),
                'drift_detected': ks_pvalue < 0.05 or mw_pvalue < 0.05
            }
            
        except Exception as e:
            logger.warning(f"Statistical drift test failed: {str(e)}")
            return {'drift_score': 0.0, 'drift_detected': False}
    
    def _distribution_drift_test(self, ref_data: pd.Series, new_data: pd.Series) -> Dict[str, Any]:
        """Perform distribution-based drift detection."""
        try:
            # Calculate distribution statistics
            ref_mean, ref_std = ref_data.mean(), ref_data.std()
            new_mean, new_std = new_data.mean(), new_data.std()
            
            # Mean shift
            mean_shift = abs(new_mean - ref_mean) / (ref_std + 1e-8)
            
            # Variance shift
            variance_shift = abs(new_std - ref_std) / (ref_std + 1e-8)
            
            # Combined drift score
            drift_score = min(1.0, (mean_shift + variance_shift) / 2)
            
            return {
                'mean_shift': float(mean_shift),
                'variance_shift': float(variance_shift),
                'drift_score': float(drift_score),
                'drift_detected': drift_score > 0.5
            }
            
        except Exception as e:
            logger.warning(f"Distribution drift test failed: {str(e)}")
            return {'drift_score': 0.0, 'drift_detected': False}
    
    def _categorical_drift_test(self, ref_data: pd.Series, new_data: pd.Series) -> Dict[str, Any]:
        """Perform categorical drift detection."""
        try:
            # Get value counts
            ref_counts = ref_data.value_counts()
            new_counts = new_data.value_counts()
            
            # Get all unique values
            all_values = set(ref_counts.index) | set(new_counts.index)
            
            # Calculate chi-square test
            observed = []
            expected = []
            
            for value in all_values:
                ref_freq = ref_counts.get(value, 0)
                new_freq = new_counts.get(value, 0)
                
                observed.extend([ref_freq, new_freq])
                
                # Expected frequencies (proportional to sample sizes)
                total_ref = len(ref_data)
                total_new = len(new_data)
                total = total_ref + total_new
                
                if total > 0:
                    expected_freq = (ref_freq + new_freq) * total_ref / total
                    expected.extend([expected_freq, (ref_freq + new_freq) - expected_freq])
                else:
                    expected.extend([0, 0])
            
            # Perform chi-square test
            if len(observed) > 0 and sum(expected) > 0:
                chi2_stat, chi2_pvalue = stats.chisquare(observed, expected)
                drift_score = 1.0 - chi2_pvalue
            else:
                drift_score = 0.0
                chi2_pvalue = 1.0
            
            return {
                'chi2_statistic': float(chi2_stat) if 'chi2_stat' in locals() else 0.0,
                'chi2_pvalue': float(chi2_pvalue),
                'drift_score': float(drift_score),
                'drift_detected': chi2_pvalue < 0.05
            }
            
        except Exception as e:
            logger.warning(f"Categorical drift test failed: {str(e)}")
            return {'drift_score': 0.0, 'drift_detected': False}


class MetricsAggregator:
    """Aggregates performance metrics."""
    
    def __init__(self):
        self.performance_metrics = PerformanceMetrics()
    
    def run_comprehensive_evaluation(self, 
                              model, 
                              X: np.ndarray,
                              test_name: str = "default_test",
                              y_true: Optional[np.ndarray] = None,
                              n_iterations: int = 3) -> Dict[str, Any]:
        """
        Run comprehensive model evaluation.
        
        Args:
            model: The model to evaluate
            X: Input features
            test_name: Name for this test run
            y_true: Optional true labels for accuracy metrics
            n_iterations: Number of iterations for timing metrics
            
        Returns:
            Dict containing evaluation results
        """
        results = {
            'test_name': test_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_features': X.shape[1] if len(X.shape) > 1 else 1
        }
        
        try:
            # Check if model is a dictionary and extract the actual model
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
                
            # Ensure we have a valid model
            if not hasattr(model, 'predict'):
                raise ValueError("Model does not have predict method")
                
            # Get performance metrics
            performance_results = self.performance_metrics.calculate_prediction_metrics(
                model=model,
                X=X,
                y_true=y_true,
                n_iterations=n_iterations
            )
            results['performance'] = performance_results
            
            # Get predictions
            predictions = model.predict(X)
            
            if y_true is not None:
                # Classification report
                report = classification_report(y_true, predictions, output_dict=True)
                results['classification_report'] = report
                
                # Confusion matrix
                cm = confusion_matrix(y_true, predictions)
                
                # Create confusion matrix plot
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                # Convert plot to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                cm_image = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                results['confusion_matrix'] = {
                    'values': cm.tolist(),
                    'plot': cm_image
                }
            
            # Add prediction statistics
            prediction_stats = {
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions))
            }
            results['prediction_statistics'] = prediction_stats
            
            return results
        
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {str(e)}")
            raise
