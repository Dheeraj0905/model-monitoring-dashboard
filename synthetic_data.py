"""
Synthetic data generation utilities.
Generates test data based on schema definitions and user parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
from schema_utils import SchemaManager, SchemaDefinition

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerator:
    """Generates synthetic data based on schema and parameters."""
    
    def __init__(self, schema_manager: SchemaManager):
        self.schema_manager = schema_manager
        self.generation_params = {}
    
    def set_generation_params(self, params: Dict[str, Any]):
        """Set parameters for data generation."""
        self.generation_params = params
        logger.info(f"Set generation parameters: {params}")
    
    def generate_data(self, n_samples: int, 
                     distribution_type: str = 'uniform',
                     add_noise: bool = True,
                     noise_level: float = 0.1,
                     seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic data based on the schema.
        
        Args:
            n_samples: Number of samples to generate
            distribution_type: Type of distribution ('uniform', 'normal', 'mixed')
            add_noise: Whether to add noise to the data
            noise_level: Level of noise to add (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Generated synthetic data
        """
        if seed is not None:
            np.random.seed(seed)
        
        data = {}
        
        for feature in self.schema_manager.schema:
            feature_data = self._generate_feature_data(
                feature, n_samples, distribution_type, add_noise, noise_level
            )
            data[feature.name] = feature_data
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {n_samples} samples with {len(self.schema_manager.schema)} features")
        return df
    
    def _generate_feature_data(self, feature: SchemaDefinition, n_samples: int,
                              distribution_type: str, add_noise: bool, 
                              noise_level: float) -> np.ndarray:
        """Generate data for a single feature."""
        
        if feature.data_type == 'category':
            return self._generate_categorical_data(feature, n_samples)
        
        elif feature.data_type in ['float64', 'int64']:
            return self._generate_numeric_data(
                feature, n_samples, distribution_type, add_noise, noise_level
            )
        
        else:
            # Fallback to uniform distribution
            return self._generate_numeric_data(
                feature, n_samples, 'uniform', add_noise, noise_level
            )
    
    def _generate_categorical_data(self, feature: SchemaDefinition, n_samples: int) -> np.ndarray:
        """Generate categorical data."""
        if feature.categories:
            # Use provided categories
            categories = feature.categories
        else:
            # Generate default categories
            categories = [f'cat_{i}' for i in range(3)]
        
        # Generate random choices
        return np.random.choice(categories, n_samples)
    
    def _generate_numeric_data(self, feature: SchemaDefinition, n_samples: int,
                              distribution_type: str, add_noise: bool, 
                              noise_level: float) -> np.ndarray:
        """Generate numeric data."""
        
        # Get min/max values
        min_val = feature.min_val if feature.min_val is not None else -10.0
        max_val = feature.max_val if feature.max_val is not None else 10.0
        
        if distribution_type == 'uniform':
            data = np.random.uniform(min_val, max_val, n_samples)
            
        elif distribution_type == 'normal':
            # Normal distribution centered in the range
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 6  # 99.7% of data within range
            data = np.random.normal(mean, std, n_samples)
            # Clip to bounds
            data = np.clip(data, min_val, max_val)
            
        elif distribution_type == 'mixed':
            # Mix of different distributions
            n1 = n_samples // 3
            n2 = n_samples // 3
            n3 = n_samples - n1 - n2
            
            # Uniform component
            data1 = np.random.uniform(min_val, min_val + (max_val - min_val) * 0.3, n1)
            
            # Normal component
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 8
            data2 = np.random.normal(mean, std, n2)
            data2 = np.clip(data2, min_val, max_val)
            
            # Exponential-like component
            data3 = np.random.exponential((max_val - min_val) / 4, n3)
            data3 = np.clip(data3 + min_val, min_val, max_val)
            
            data = np.concatenate([data1, data2, data3])
            np.random.shuffle(data)
            
        else:
            # Default to uniform
            data = np.random.uniform(min_val, max_val, n_samples)
        
        # Add noise if requested
        if add_noise and noise_level > 0:
            noise_std = (max_val - min_val) * noise_level
            noise = np.random.normal(0, noise_std, n_samples)
            data = data + noise
            data = np.clip(data, min_val, max_val)
        
        # Convert to appropriate type
        if feature.data_type == 'int64':
            data = data.astype(int)
        
        return data
    
    def generate_drift_data(self, base_data: pd.DataFrame, 
                           drift_type: str = 'mean_shift',
                           drift_magnitude: float = 0.5,
                           affected_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate data with drift for testing drift detection.
        
        Args:
            base_data: Original data to apply drift to
            drift_type: Type of drift ('mean_shift', 'variance_shift', 'distribution_shift')
            drift_magnitude: Magnitude of the drift (0.0 to 1.0)
            affected_features: List of features to apply drift to (None for all numeric features)
            
        Returns:
            pd.DataFrame: Data with applied drift
        """
        drift_data = base_data.copy()
        
        if affected_features is None:
            # Apply to all numeric features
            affected_features = [
                f.name for f in self.schema_manager.schema 
                if f.data_type in ['float64', 'int64']
            ]
        
        for feature_name in affected_features:
            if feature_name in drift_data.columns:
                feature = self.schema_manager.get_feature_by_name(feature_name)
                if feature and feature.data_type in ['float64', 'int64']:
                    drift_data[feature_name] = self._apply_drift_to_feature(
                        drift_data[feature_name], feature, drift_type, drift_magnitude
                    )
        
        logger.info(f"Applied {drift_type} drift to {len(affected_features)} features")
        return drift_data
    
    def _apply_drift_to_feature(self, data: pd.Series, feature: SchemaDefinition,
                               drift_type: str, drift_magnitude: float) -> pd.Series:
        """Apply drift to a single feature."""
        
        min_val = feature.min_val if feature.min_val is not None else data.min()
        max_val = feature.max_val if feature.max_val is not None else data.max()
        range_size = max_val - min_val
        
        if drift_type == 'mean_shift':
            # Shift the mean
            shift = range_size * drift_magnitude
            shifted_data = data + shift
            # Clip to bounds
            shifted_data = np.clip(shifted_data, min_val, max_val)
            
        elif drift_type == 'variance_shift':
            # Change the variance
            current_std = data.std()
            new_std = current_std * (1 + drift_magnitude)
            # Normalize and rescale
            normalized = (data - data.mean()) / current_std
            shifted_data = normalized * new_std + data.mean()
            # Clip to bounds
            shifted_data = np.clip(shifted_data, min_val, max_val)
            
        elif drift_type == 'distribution_shift':
            # Change the distribution shape
            if drift_magnitude > 0.5:
                # More skewed
                shifted_data = np.power(data - min_val, 1 + drift_magnitude) + min_val
            else:
                # More uniform
                shifted_data = data * (1 - drift_magnitude) + np.random.uniform(
                    min_val, max_val, len(data)
                ) * drift_magnitude
            
            shifted_data = np.clip(shifted_data, min_val, max_val)
            
        else:
            # Default to mean shift
            shift = range_size * drift_magnitude
            shifted_data = data + shift
            shifted_data = np.clip(shifted_data, min_val, max_val)
        
        # Convert back to original type
        if feature.data_type == 'int64':
            shifted_data = shifted_data.astype(int)
        
        return pd.Series(shifted_data, index=data.index)
    
    def generate_correlated_data(self, n_samples: int, 
                                correlation_matrix: Optional[np.ndarray] = None,
                                seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate data with specified correlations between features.
        
        Args:
            n_samples: Number of samples to generate
            correlation_matrix: Correlation matrix (None for random correlations)
            seed: Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Generated correlated data
        """
        if seed is not None:
            np.random.seed(seed)
        
        numeric_features = [
            f for f in self.schema_manager.schema 
            if f.data_type in ['float64', 'int64']
        ]
        
        if len(numeric_features) < 2:
            # Not enough numeric features for correlation
            return self.generate_data(n_samples)
        
        n_numeric = len(numeric_features)
        
        if correlation_matrix is None:
            # Generate random correlation matrix
            correlation_matrix = self._generate_random_correlation_matrix(n_numeric)
        
        # Generate multivariate normal data
        mean = np.zeros(n_numeric)
        cov_matrix = self._correlation_to_covariance(correlation_matrix, numeric_features)
        
        try:
            multivariate_data = np.random.multivariate_normal(mean, cov_matrix, n_samples)
        except np.linalg.LinAlgError:
            # Fallback to independent data if correlation matrix is not positive definite
            logger.warning("Correlation matrix not positive definite, using independent data")
            multivariate_data = np.random.normal(0, 1, (n_samples, n_numeric))
        
        # Scale to feature ranges
        data = {}
        for i, feature in enumerate(numeric_features):
            min_val = feature.min_val if feature.min_val is not None else -10.0
            max_val = feature.max_val if feature.max_val is not None else 10.0
            
            # Normalize to [0, 1] then scale to feature range
            normalized = (multivariate_data[:, i] - multivariate_data[:, i].min()) / \
                        (multivariate_data[:, i].max() - multivariate_data[:, i].min())
            scaled = normalized * (max_val - min_val) + min_val
            
            if feature.data_type == 'int64':
                scaled = scaled.astype(int)
            
            data[feature.name] = scaled
        
        # Add categorical features
        for feature in self.schema_manager.schema:
            if feature.data_type == 'category':
                data[feature.name] = self._generate_categorical_data(feature, n_samples)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {n_samples} correlated samples")
        return df
    
    def _generate_random_correlation_matrix(self, n_features: int) -> np.ndarray:
        """Generate a random valid correlation matrix."""
        # Generate random matrix
        A = np.random.randn(n_features, n_features)
        # Make it symmetric and positive definite
        correlation_matrix = np.dot(A, A.T)
        # Normalize to correlation matrix
        std = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(std, std)
        return correlation_matrix
    
    def _correlation_to_covariance(self, correlation_matrix: np.ndarray, 
                                  features: List[SchemaDefinition]) -> np.ndarray:
        """Convert correlation matrix to covariance matrix."""
        stds = []
        for feature in features:
            min_val = feature.min_val if feature.min_val is not None else -10.0
            max_val = feature.max_val if feature.max_val is not None else 10.0
            # Estimate standard deviation as 1/6 of the range
            std = (max_val - min_val) / 6
            stds.append(std)
        
        stds = np.array(stds)
        return np.outer(stds, stds) * correlation_matrix


def create_sample_schema() -> SchemaManager:
    """Create a sample schema for testing."""
    schema_manager = SchemaManager()
    
    # Add some sample features
    features = [
        SchemaDefinition('age', 'int64', 18, 80, description="Age in years"),
        SchemaDefinition('income', 'float64', 20000, 200000, description="Annual income"),
        SchemaDefinition('education', 'category', categories=['high_school', 'bachelor', 'master', 'phd']),
        SchemaDefinition('score', 'float64', 0, 100, description="Credit score"),
        SchemaDefinition('experience', 'int64', 0, 40, description="Years of experience")
    ]
    
    for feature in features:
        schema_manager.add_feature(feature)
    
    return schema_manager


if __name__ == "__main__":
    # Test the data generator
    schema_manager = create_sample_schema()
    generator = DataGenerator(schema_manager)
    
    # Generate basic data
    data = generator.generate_data(100, distribution_type='normal', seed=42)
    print("Generated data shape:", data.shape)
    print("Data types:", data.dtypes)
    print("Sample data:")
    print(data.head())
    
    # Generate drift data
    drift_data = generator.generate_drift_data(data, drift_type='mean_shift', drift_magnitude=0.3)
    print("\nDrift data sample:")
    print(drift_data.head())
    
    # Generate correlated data
    correlated_data = generator.generate_correlated_data(100, seed=42)
    print("\nCorrelated data sample:")
    print(correlated_data.head())
