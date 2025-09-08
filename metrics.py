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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


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
        
        # Convert numpy array to DataFrame if needed
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            # Model expects DataFrame with specific column names
            X_df = pd.DataFrame(X, columns=model.feature_names_in_)
        else:
            # Model can handle numpy array
            X_df = X
        
        for _ in range(n_iterations):
            start_time = time.time()
            _ = model.predict(X_df)
            end_time = time.time()
            times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            'latency_mean_ms': np.mean(times) * 1000,  # Convert to milliseconds
            'latency_std_ms': np.std(times) * 1000,
            'latency_min_ms': np.min(times) * 1000,
            'latency_max_ms': np.max(times) * 1000,
            'latency_p95_ms': np.percentile(times, 95) * 1000,
            'latency_p99_ms': np.percentile(times, 99) * 1000,
            'throughput_preds_per_sec': len(X) / np.mean(times),
            'total_prediction_time_sec': np.sum(times)
        }
    
    def _get_predictions(self, model, X: np.ndarray) -> np.ndarray:
        """Get model predictions safely."""
        try:
            # Convert numpy array to DataFrame if needed
            if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
                # Model expects DataFrame with specific column names
                X_df = pd.DataFrame(X, columns=model.feature_names_in_)
                return model.predict(X_df)
            else:
                # Model can handle numpy array
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
    """Aggregates and manages multiple types of metrics."""
    
    def __init__(self):
        self.performance_metrics = PerformanceMetrics()
        self.drift_detector = DriftDetector()
        self.metrics_history = []
    
    def run_comprehensive_evaluation(self, model, X: np.ndarray, 
                                   y_true: Optional[np.ndarray] = None,
                                   reference_data: Optional[pd.DataFrame] = None,
                                   test_name: str = "default_test") -> Dict[str, Any]:
        """
        Run comprehensive model evaluation including performance and drift metrics.
        
        Args:
            model: Trained model
            X: Input features
            y_true: True labels (optional)
            reference_data: Reference data for drift detection (optional)
            test_name: Name for this test run
            
        Returns:
            Dict with comprehensive evaluation results
        """
        results = {
            'test_name': test_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_samples': len(X),
            'n_features': X.shape[1] if X.ndim > 1 else 1
        }
        
        # Performance metrics
        performance_results = self.performance_metrics.calculate_prediction_metrics(
            model, X, y_true
        )
        results['performance'] = performance_results
        
        # Drift detection (if reference data provided)
        if reference_data is not None:
            try:
                # Convert X to DataFrame for drift detection
                if isinstance(X, np.ndarray):
                    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                else:
                    X_df = X
                
                self.drift_detector.set_reference_data(reference_data)
                drift_results = self.drift_detector.detect_drift(X_df)
                results['drift'] = drift_results
            except Exception as e:
                logger.warning(f"Drift detection failed: {str(e)}")
                results['drift'] = {'error': str(e)}
        
        # Store in history
        self.metrics_history.append(results)
        
        return results
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics history."""
        if not self.metrics_history:
            return {'message': 'No metrics history available'}
        
        summary = {
            'total_tests': len(self.metrics_history),
            'latest_test': self.metrics_history[-1]['test_name'],
            'performance_trends': self._calculate_performance_trends(),
            'drift_trends': self._calculate_drift_trends()
        }
        
        return summary
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        if len(self.metrics_history) < 2:
            return {'message': 'Insufficient data for trends'}
        
        trends = {}
        
        # Extract performance metrics
        latencies = [test['performance'].get('latency_mean_ms', 0) for test in self.metrics_history]
        throughputs = [test['performance'].get('throughput_preds_per_sec', 0) for test in self.metrics_history]
        
        if latencies:
            trends['latency_trend'] = 'improving' if latencies[-1] < latencies[0] else 'degrading'
            trends['latency_change_percent'] = ((latencies[-1] - latencies[0]) / latencies[0] * 100) if latencies[0] > 0 else 0
        
        if throughputs:
            trends['throughput_trend'] = 'improving' if throughputs[-1] > throughputs[0] else 'degrading'
            trends['throughput_change_percent'] = ((throughputs[-1] - throughputs[0]) / throughputs[0] * 100) if throughputs[0] > 0 else 0
        
        return trends
    
    def _calculate_drift_trends(self) -> Dict[str, Any]:
        """Calculate drift trends over time."""
        if len(self.metrics_history) < 2:
            return {'message': 'Insufficient data for trends'}
        
        drift_scores = [test.get('drift', {}).get('overall_drift_score', 0) for test in self.metrics_history]
        
        if drift_scores:
            return {
                'drift_trend': 'increasing' if drift_scores[-1] > drift_scores[0] else 'decreasing',
                'current_drift_score': drift_scores[-1],
                'max_drift_score': max(drift_scores)
            }
        
        return {'message': 'No drift data available'}


if __name__ == "__main__":
    # Test the metrics modules
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = X[:800], X[800:], y[:800], y[800:]
    
    # Train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Test performance metrics
    perf_metrics = PerformanceMetrics()
    results = perf_metrics.calculate_prediction_metrics(model, X_test, y_test)
    print("Performance metrics:", results)
    
    # Test drift detection
    drift_detector = DriftDetector()
    reference_data = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    new_data = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    
    drift_detector.set_reference_data(reference_data)
    drift_results = drift_detector.detect_drift(new_data)
    print("Drift detection results:", drift_results)
    
    # Test comprehensive evaluation
    aggregator = MetricsAggregator()
    comprehensive_results = aggregator.run_comprehensive_evaluation(
        model, X_test, y_test, reference_data, "test_run_1"
    )
    print("Comprehensive evaluation:", comprehensive_results)
