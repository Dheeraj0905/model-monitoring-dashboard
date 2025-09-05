"""
Tests for metrics.py
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Add parent directory to path to import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import PerformanceMetrics, DriftDetector, MetricsAggregator


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for PerformanceMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = PerformanceMetrics()
        
        # Create a sample model
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        self.X_test = X[:20]
        self.y_test = y[:20]
    
    def test_calculate_prediction_metrics(self):
        """Test prediction metrics calculation."""
        results = self.metrics.calculate_prediction_metrics(
            self.model, self.X_test, self.y_test
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'latency_mean_ms', 'latency_std_ms', 'throughput_preds_per_sec',
            'error_rate', 'predictions'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, results)
        
        # Check metric values are reasonable
        self.assertGreater(results['latency_mean_ms'], 0)
        self.assertGreater(results['throughput_preds_per_sec'], 0)
        self.assertGreaterEqual(results['error_rate'], 0)
        self.assertLessEqual(results['error_rate'], 1)
        
        # Check predictions
        self.assertEqual(len(results['predictions']), len(self.X_test))
    
    def test_calculate_prediction_metrics_without_labels(self):
        """Test prediction metrics calculation without true labels."""
        results = self.metrics.calculate_prediction_metrics(
            self.model, self.X_test
        )
        
        # Should still have performance metrics
        self.assertIn('latency_mean_ms', results)
        self.assertIn('throughput_preds_per_sec', results)
        self.assertIn('predictions', results)
        
        # Should not have accuracy metrics
        self.assertNotIn('accuracy', results)
    
    def test_metrics_history(self):
        """Test metrics history tracking."""
        # Run multiple tests
        for i in range(3):
            self.metrics.calculate_prediction_metrics(
                self.model, self.X_test, self.y_test
            )
        
        self.assertEqual(len(self.metrics.metrics_history), 3)


class TestDriftDetector(unittest.TestCase):
    """Test cases for DriftDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.drift_detector = DriftDetector()
        
        # Create reference data
        np.random.seed(42)
        self.reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Create new data with some drift
        self.new_data = pd.DataFrame({
            'feature1': np.random.normal(1, 1, 100),  # Mean shift
            'feature2': np.random.normal(5, 3, 100),  # Variance shift
            'feature3': np.random.choice(['A', 'B', 'C'], 100)
        })
    
    def test_set_reference_data(self):
        """Test setting reference data."""
        self.drift_detector.set_reference_data(self.reference_data)
        
        self.assertIsNotNone(self.drift_detector.reference_data)
        self.assertEqual(len(self.drift_detector.reference_stats), 3)
    
    def test_detect_drift(self):
        """Test drift detection."""
        self.drift_detector.set_reference_data(self.reference_data)
        drift_results = self.drift_detector.detect_drift(self.new_data)
        
        # Check structure of results
        self.assertIn('overall_drift_detected', drift_results)
        self.assertIn('overall_drift_score', drift_results)
        self.assertIn('feature_drift', drift_results)
        self.assertIn('summary', drift_results)
        
        # Check that drift is detected (since we introduced drift)
        self.assertIsInstance(drift_results['overall_drift_detected'], bool)
        self.assertGreaterEqual(drift_results['overall_drift_score'], 0)
        self.assertLessEqual(drift_results['overall_drift_score'], 1)
    
    def test_detect_drift_no_reference(self):
        """Test drift detection without reference data."""
        with self.assertRaises(ValueError):
            self.drift_detector.detect_drift(self.new_data)
    
    def test_statistical_drift_test(self):
        """Test statistical drift test."""
        ref_data = pd.Series(np.random.normal(0, 1, 100))
        new_data = pd.Series(np.random.normal(1, 1, 100))  # Mean shift
        
        result = self.drift_detector._statistical_drift_test(ref_data, new_data)
        
        self.assertIn('ks_statistic', result)
        self.assertIn('ks_pvalue', result)
        self.assertIn('drift_score', result)
        self.assertIn('drift_detected', result)
        
        # Should detect drift due to mean shift
        self.assertGreater(result['drift_score'], 0)
    
    def test_distribution_drift_test(self):
        """Test distribution drift test."""
        ref_data = pd.Series(np.random.normal(0, 1, 100))
        new_data = pd.Series(np.random.normal(1, 1, 100))  # Mean shift
        
        result = self.drift_detector._distribution_drift_test(ref_data, new_data)
        
        self.assertIn('mean_shift', result)
        self.assertIn('variance_shift', result)
        self.assertIn('drift_score', result)
        self.assertIn('drift_detected', result)
        
        # Should detect mean shift
        self.assertGreater(result['mean_shift'], 0)
    
    def test_categorical_drift_test(self):
        """Test categorical drift test."""
        ref_data = pd.Series(['A', 'B', 'C'] * 33 + ['A'])
        new_data = pd.Series(['A', 'A', 'A'] * 33 + ['B'])  # Distribution shift
        
        result = self.drift_detector._categorical_drift_test(ref_data, new_data)
        
        self.assertIn('chi2_statistic', result)
        self.assertIn('chi2_pvalue', result)
        self.assertIn('drift_score', result)
        self.assertIn('drift_detected', result)


class TestMetricsAggregator(unittest.TestCase):
    """Test cases for MetricsAggregator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = MetricsAggregator()
        
        # Create a sample model
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        self.X_test = X[:20]
        self.y_test = y[:20]
        self.reference_data = pd.DataFrame(X[:50], columns=[f'feature_{i}' for i in range(5)])
    
    def test_run_comprehensive_evaluation(self):
        """Test comprehensive evaluation."""
        results = self.aggregator.run_comprehensive_evaluation(
            self.model, self.X_test, self.y_test, self.reference_data, "test_run"
        )
        
        # Check structure
        self.assertIn('test_name', results)
        self.assertIn('timestamp', results)
        self.assertIn('n_samples', results)
        self.assertIn('n_features', results)
        self.assertIn('performance', results)
        self.assertIn('drift', results)
        
        # Check values
        self.assertEqual(results['test_name'], "test_run")
        self.assertEqual(results['n_samples'], len(self.X_test))
        self.assertEqual(results['n_features'], self.X_test.shape[1])
    
    def test_run_comprehensive_evaluation_without_drift(self):
        """Test comprehensive evaluation without drift detection."""
        results = self.aggregator.run_comprehensive_evaluation(
            self.model, self.X_test, self.y_test, test_name="test_run"
        )
        
        # Should have performance but not drift
        self.assertIn('performance', results)
        self.assertNotIn('drift', results)
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        # Run some evaluations first
        for i in range(3):
            self.aggregator.run_comprehensive_evaluation(
                self.model, self.X_test, self.y_test, test_name=f"test_{i}"
            )
        
        summary = self.aggregator.get_metrics_summary()
        
        self.assertIn('total_tests', summary)
        self.assertIn('latest_test', summary)
        self.assertIn('performance_trends', summary)
        self.assertIn('drift_trends', summary)
        
        self.assertEqual(summary['total_tests'], 3)
    
    def test_metrics_history(self):
        """Test metrics history tracking."""
        # Run evaluation
        self.aggregator.run_comprehensive_evaluation(
            self.model, self.X_test, self.y_test, test_name="test_run"
        )
        
        self.assertEqual(len(self.aggregator.metrics_history), 1)
        
        # Check history content
        history_item = self.aggregator.metrics_history[0]
        self.assertEqual(history_item['test_name'], "test_run")
        self.assertIn('performance', history_item)


if __name__ == '__main__':
    unittest.main()
