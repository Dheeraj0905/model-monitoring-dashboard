"""
Tests for model_utils.py
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle

# Add parent directory to path to import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import ModelLoader


class TestModelLoader(unittest.TestCase):
    """Test cases for ModelLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = ModelLoader()
        
        # Create a sample model for testing
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        with open(self.temp_file.name, 'wb') as f:
            pickle.dump(self.model, f)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_load_model_success(self):
        """Test successful model loading."""
        success = self.loader.load_model(self.temp_file.name)
        self.assertTrue(success)
        self.assertIsNotNone(self.loader.model)
        self.assertEqual(type(self.loader.model).__name__, 'RandomForestClassifier')
    
    def test_load_model_failure(self):
        """Test model loading failure."""
        success = self.loader.load_model('nonexistent_file.pkl')
        self.assertFalse(success)
        self.assertIsNone(self.loader.model)
    
    def test_model_introspection(self):
        """Test model introspection."""
        self.loader.load_model(self.temp_file.name)
        model_info = self.loader.get_model_info()
        
        self.assertIn('model_info', model_info)
        self.assertIn('feature_names', model_info)
        self.assertIn('feature_types', model_info)
        self.assertIn('n_features', model_info)
        self.assertIn('has_proba', model_info)
        
        self.assertEqual(model_info['n_features'], 5)
        self.assertTrue(model_info['has_proba'])
    
    def test_predict(self):
        """Test model prediction."""
        self.loader.load_model(self.temp_file.name)
        
        # Test prediction
        X_test = np.random.random((10, 5))
        predictions = self.loader.predict(X_test)
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_predict_proba(self):
        """Test prediction probabilities."""
        self.loader.load_model(self.temp_file.name)
        
        X_test = np.random.random((5, 5))
        probabilities = self.loader.predict_proba(X_test)
        
        self.assertIsNotNone(probabilities)
        self.assertEqual(probabilities.shape, (5, 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
    
    def test_validate_input_shape(self):
        """Test input shape validation."""
        self.loader.load_model(self.temp_file.name)
        
        # Valid shape
        X_valid = np.random.random((5, 5))
        self.assertTrue(self.loader.validate_input_shape(X_valid))
        
        # Invalid shape
        X_invalid = np.random.random((5, 3))
        self.assertFalse(self.loader.validate_input_shape(X_invalid))


if __name__ == '__main__':
    unittest.main()
