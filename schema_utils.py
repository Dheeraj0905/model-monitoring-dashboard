"""
Schema utilities for defining and validating input schemas.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchemaDefinition:
    """Represents a feature schema definition."""
    
    def __init__(self, name: str, data_type: str, 
                 min_val: Optional[float] = None, 
                 max_val: Optional[float] = None,
                 categories: Optional[List[str]] = None,
                 description: str = ""):
        self.name = name
        self.data_type = data_type
        self.min_val = min_val
        self.max_val = max_val
        self.categories = categories
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'data_type': self.data_type,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'categories': self.categories,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaDefinition':
        """Create from dictionary representation."""
        return cls(
            name=data['name'],
            data_type=data['data_type'],
            min_val=data.get('min_val'),
            max_val=data.get('max_val'),
            categories=data.get('categories'),
            description=data.get('description', '')
        )


class SchemaManager:
    """Manages feature schemas for model input validation."""
    
    def __init__(self):
        self.schema: List[SchemaDefinition] = []
        self.auto_detected = False
    
    def add_feature(self, feature: SchemaDefinition):
        """Add a feature to the schema."""
        self.schema.append(feature)
        logger.info(f"Added feature: {feature.name} ({feature.data_type})")
    
    def remove_feature(self, feature_name: str):
        """Remove a feature from the schema."""
        self.schema = [f for f in self.schema if f.name != feature_name]
        logger.info(f"Removed feature: {feature_name}")
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [f.name for f in self.schema]
    
    def get_feature_types(self) -> List[str]:
        """Get list of feature data types."""
        return [f.data_type for f in self.schema]
    
    def get_feature_by_name(self, name: str) -> Optional[SchemaDefinition]:
        """Get a feature definition by name."""
        for feature in self.schema:
            if feature.name == name:
                return feature
        return None
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data against the schema.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dict with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if all required features are present
        required_features = set(self.get_feature_names())
        actual_features = set(data.columns)
        
        missing_features = required_features - actual_features
        if missing_features:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Missing features: {list(missing_features)}")
        
        extra_features = actual_features - required_features
        if extra_features:
            validation_results['warnings'].append(f"Extra features: {list(extra_features)}")
        
        # Validate each feature
        for feature in self.schema:
            if feature.name in data.columns:
                feature_validation = self._validate_feature(data[feature.name], feature)
                if not feature_validation['valid']:
                    validation_results['valid'] = False
                    validation_results['errors'].extend(feature_validation['errors'])
                validation_results['warnings'].extend(feature_validation['warnings'])
        
        return validation_results
    
    def _validate_feature(self, series: pd.Series, feature: SchemaDefinition) -> Dict[str, Any]:
        """Validate a single feature."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        # Check data type
        if feature.data_type == 'float64':
            if not pd.api.types.is_numeric_dtype(series):
                result['valid'] = False
                result['errors'].append(f"Feature {feature.name} should be numeric")
        elif feature.data_type == 'int64':
            if not pd.api.types.is_integer_dtype(series):
                result['warnings'].append(f"Feature {feature.name} contains non-integer values")
        elif feature.data_type == 'category':
            if feature.categories:
                invalid_categories = set(series.dropna()) - set(feature.categories)
                if invalid_categories:
                    result['valid'] = False
                    result['errors'].append(f"Feature {feature.name} contains invalid categories: {list(invalid_categories)}")
        
        # Check value ranges
        if feature.data_type in ['float64', 'int64'] and pd.api.types.is_numeric_dtype(series):
            if feature.min_val is not None:
                below_min = series < feature.min_val
                if below_min.any():
                    result['warnings'].append(f"Feature {feature.name} has values below minimum ({feature.min_val})")
            
            if feature.max_val is not None:
                above_max = series > feature.max_val
                if above_max.any():
                    result['warnings'].append(f"Feature {feature.name} has values above maximum ({feature.max_val})")
        
        return result
    
    def auto_detect_from_csv(self, csv_path: str, sample_size: int = 1000) -> bool:
        """
        Automatically detect schema from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            sample_size: Number of rows to sample for detection
            
        Returns:
            bool: True if successful
        """
        try:
            # Read a sample of the data
            df = pd.read_csv(csv_path, nrows=sample_size)
            
            # Clear existing schema
            self.schema = []
            
            # Detect schema for each column
            for column in df.columns:
                feature = self._detect_feature_schema(column, df[column])
                self.add_feature(feature)
            
            self.auto_detected = True
            logger.info(f"Auto-detected schema from {csv_path} with {len(self.schema)} features")
            return True
            
        except Exception as e:
            logger.error(f"Failed to auto-detect schema: {str(e)}")
            return False
    
    def _detect_feature_schema(self, name: str, series: pd.Series) -> SchemaDefinition:
        """Detect schema for a single feature."""
        # Determine data type
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                data_type = 'int64'
            else:
                data_type = 'float64'
            
            # Get min/max values
            min_val = float(series.min()) if not series.empty else None
            max_val = float(series.max()) if not series.empty else None
            
            return SchemaDefinition(
                name=name,
                data_type=data_type,
                min_val=min_val,
                max_val=max_val,
                description=f"Auto-detected from data"
            )
        
        else:
            # Categorical feature
            unique_values = series.dropna().unique()
            categories = list(unique_values)[:20]  # Limit to 20 categories
            
            return SchemaDefinition(
                name=name,
                data_type='category',
                categories=categories,
                description=f"Auto-detected from data ({len(unique_values)} unique values)"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation."""
        return {
            'features': [f.to_dict() for f in self.schema],
            'auto_detected': self.auto_detected,
            'n_features': len(self.schema)
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Load schema from dictionary representation."""
        self.schema = [SchemaDefinition.from_dict(f) for f in data['features']]
        self.auto_detected = data.get('auto_detected', False)
    
    def to_json(self) -> str:
        """Convert schema to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def from_json(self, json_str: str):
        """Load schema from JSON string."""
        data = json.loads(json_str)
        self.from_dict(data)
    
    def create_default_schema(self, n_features: int = 5) -> bool:
        """
        Create a default schema with the specified number of features.
        
        Args:
            n_features: Number of features to create
            
        Returns:
            bool: True if successful
        """
        try:
            self.schema = []
            
            for i in range(n_features):
                feature = SchemaDefinition(
                    name=f'feature_{i}',
                    data_type='float64',
                    min_val=-10.0,
                    max_val=10.0,
                    description=f"Default feature {i}"
                )
                self.add_feature(feature)
            
            self.auto_detected = False
            logger.info(f"Created default schema with {n_features} features")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create default schema: {str(e)}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current schema."""
        feature_types = {}
        for feature in self.schema:
            feature_types[feature.data_type] = feature_types.get(feature.data_type, 0) + 1
        
        return {
            'n_features': len(self.schema),
            'feature_types': feature_types,
            'auto_detected': self.auto_detected,
            'feature_names': self.get_feature_names()
        }


def create_sample_csv(schema: SchemaManager, n_samples: int = 100, 
                     output_path: str = "sample_data.csv") -> bool:
    """
    Create a sample CSV file based on the schema.
    
    Args:
        schema: SchemaManager instance
        n_samples: Number of samples to generate
        output_path: Output file path
        
    Returns:
        bool: True if successful
    """
    try:
        data = {}
        
        for feature in schema.schema:
            if feature.data_type == 'float64':
                # Generate random float values
                min_val = feature.min_val if feature.min_val is not None else -10.0
                max_val = feature.max_val if feature.max_val is not None else 10.0
                data[feature.name] = np.random.uniform(min_val, max_val, n_samples)
                
            elif feature.data_type == 'int64':
                # Generate random integer values
                min_val = int(feature.min_val) if feature.min_val is not None else -100
                max_val = int(feature.max_val) if feature.max_val is not None else 100
                data[feature.name] = np.random.randint(min_val, max_val + 1, n_samples)
                
            elif feature.data_type == 'category':
                # Generate random categorical values
                if feature.categories:
                    data[feature.name] = np.random.choice(feature.categories, n_samples)
                else:
                    data[feature.name] = np.random.choice(['A', 'B', 'C'], n_samples)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Created sample CSV with {n_samples} samples at {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample CSV: {str(e)}")
        return False


if __name__ == "__main__":
    # Test the schema manager
    schema_manager = SchemaManager()
    
    # Create a default schema
    schema_manager.create_default_schema(3)
    print("Default schema created:", schema_manager.get_summary())
    
    # Test JSON serialization
    json_str = schema_manager.to_json()
    print("JSON representation:", json_str)
    
    # Create sample CSV
    create_sample_csv(schema_manager, 10, "test_sample.csv")
    
    # Test auto-detection
    schema_manager.auto_detect_from_csv("test_sample.csv")
    print("Auto-detected schema:", schema_manager.get_summary())
