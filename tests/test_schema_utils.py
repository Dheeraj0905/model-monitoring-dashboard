"""
Tests for schema_utils.py
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os

# Add parent directory to path to import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema_utils import SchemaManager, SchemaDefinition


class TestSchemaDefinition(unittest.TestCase):
    """Test cases for SchemaDefinition class."""
    
    def test_schema_definition_creation(self):
        """Test schema definition creation."""
        feature = SchemaDefinition(
            name='test_feature',
            data_type='float64',
            min_val=0.0,
            max_val=10.0,
            description='Test feature'
        )
        
        self.assertEqual(feature.name, 'test_feature')
        self.assertEqual(feature.data_type, 'float64')
        self.assertEqual(feature.min_val, 0.0)
        self.assertEqual(feature.max_val, 10.0)
        self.assertEqual(feature.description, 'Test feature')
    
    def test_schema_definition_to_dict(self):
        """Test schema definition to dictionary conversion."""
        feature = SchemaDefinition(
            name='test_feature',
            data_type='float64',
            min_val=0.0,
            max_val=10.0,
            categories=['A', 'B', 'C'],
            description='Test feature'
        )
        
        feature_dict = feature.to_dict()
        
        self.assertEqual(feature_dict['name'], 'test_feature')
        self.assertEqual(feature_dict['data_type'], 'float64')
        self.assertEqual(feature_dict['min_val'], 0.0)
        self.assertEqual(feature_dict['max_val'], 10.0)
        self.assertEqual(feature_dict['categories'], ['A', 'B', 'C'])
        self.assertEqual(feature_dict['description'], 'Test feature')
    
    def test_schema_definition_from_dict(self):
        """Test schema definition creation from dictionary."""
        feature_dict = {
            'name': 'test_feature',
            'data_type': 'float64',
            'min_val': 0.0,
            'max_val': 10.0,
            'categories': ['A', 'B', 'C'],
            'description': 'Test feature'
        }
        
        feature = SchemaDefinition.from_dict(feature_dict)
        
        self.assertEqual(feature.name, 'test_feature')
        self.assertEqual(feature.data_type, 'float64')
        self.assertEqual(feature.min_val, 0.0)
        self.assertEqual(feature.max_val, 10.0)
        self.assertEqual(feature.categories, ['A', 'B', 'C'])
        self.assertEqual(feature.description, 'Test feature')


class TestSchemaManager(unittest.TestCase):
    """Test cases for SchemaManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.schema_manager = SchemaManager()
    
    def test_add_feature(self):
        """Test adding features to schema."""
        feature = SchemaDefinition('test_feature', 'float64', 0.0, 10.0)
        self.schema_manager.add_feature(feature)
        
        self.assertEqual(len(self.schema_manager.schema), 1)
        self.assertEqual(self.schema_manager.schema[0].name, 'test_feature')
    
    def test_remove_feature(self):
        """Test removing features from schema."""
        feature = SchemaDefinition('test_feature', 'float64', 0.0, 10.0)
        self.schema_manager.add_feature(feature)
        
        self.schema_manager.remove_feature('test_feature')
        self.assertEqual(len(self.schema_manager.schema), 0)
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        feature1 = SchemaDefinition('feature1', 'float64', 0.0, 10.0)
        feature2 = SchemaDefinition('feature2', 'int64', 0, 100)
        
        self.schema_manager.add_feature(feature1)
        self.schema_manager.add_feature(feature2)
        
        names = self.schema_manager.get_feature_names()
        self.assertEqual(names, ['feature1', 'feature2'])
    
    def test_get_feature_types(self):
        """Test getting feature types."""
        feature1 = SchemaDefinition('feature1', 'float64', 0.0, 10.0)
        feature2 = SchemaDefinition('feature2', 'int64', 0, 100)
        
        self.schema_manager.add_feature(feature1)
        self.schema_manager.add_feature(feature2)
        
        types = self.schema_manager.get_feature_types()
        self.assertEqual(types, ['float64', 'int64'])
    
    def test_validate_data(self):
        """Test data validation."""
        # Create schema
        feature1 = SchemaDefinition('feature1', 'float64', 0.0, 10.0)
        feature2 = SchemaDefinition('feature2', 'int64', 0, 100)
        self.schema_manager.add_feature(feature1)
        self.schema_manager.add_feature(feature2)
        
        # Valid data
        valid_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [10, 20, 30]
        })
        
        validation_result = self.schema_manager.validate_data(valid_data)
        self.assertTrue(validation_result['valid'])
        
        # Invalid data (missing feature)
        invalid_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0]
        })
        
        validation_result = self.schema_manager.validate_data(invalid_data)
        self.assertFalse(validation_result['valid'])
        self.assertGreater(len(validation_result['errors']), 0)
    
    def test_auto_detect_from_csv(self):
        """Test automatic schema detection from CSV."""
        # Create temporary CSV file
        temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        csv_content = """feature1,feature2,feature3
1.0,10,category_a
2.0,20,category_b
3.0,30,category_a"""
        
        temp_csv.write(csv_content)
        temp_csv.close()
        
        try:
            # Test auto-detection
            success = self.schema_manager.auto_detect_from_csv(temp_csv.name)
            self.assertTrue(success)
            self.assertEqual(len(self.schema_manager.schema), 3)
            
            # Check feature types
            types = self.schema_manager.get_feature_types()
            self.assertIn('float64', types)
            self.assertIn('int64', types)
            self.assertIn('category', types)
            
        finally:
            os.unlink(temp_csv.name)
    
    def test_create_default_schema(self):
        """Test creating default schema."""
        success = self.schema_manager.create_default_schema(5)
        self.assertTrue(success)
        self.assertEqual(len(self.schema_manager.schema), 5)
        
        # Check that all features are float64
        types = self.schema_manager.get_feature_types()
        self.assertTrue(all(t == 'float64' for t in types))
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        # Create schema
        feature1 = SchemaDefinition('feature1', 'float64', 0.0, 10.0)
        feature2 = SchemaDefinition('feature2', 'category', categories=['A', 'B'])
        self.schema_manager.add_feature(feature1)
        self.schema_manager.add_feature(feature2)
        
        # Test to JSON
        json_str = self.schema_manager.to_json()
        self.assertIsInstance(json_str, str)
        
        # Test from JSON
        new_schema_manager = SchemaManager()
        new_schema_manager.from_json(json_str)
        
        self.assertEqual(len(new_schema_manager.schema), 2)
        self.assertEqual(new_schema_manager.get_feature_names(), ['feature1', 'feature2'])
    
    def test_get_summary(self):
        """Test getting schema summary."""
        feature1 = SchemaDefinition('feature1', 'float64', 0.0, 10.0)
        feature2 = SchemaDefinition('feature2', 'int64', 0, 100)
        self.schema_manager.add_feature(feature1)
        self.schema_manager.add_feature(feature2)
        
        summary = self.schema_manager.get_summary()
        
        self.assertEqual(summary['n_features'], 2)
        self.assertIn('float64', summary['feature_types'])
        self.assertIn('int64', summary['feature_types'])
        self.assertEqual(summary['feature_types']['float64'], 1)
        self.assertEqual(summary['feature_types']['int64'], 1)


if __name__ == '__main__':
    unittest.main()
