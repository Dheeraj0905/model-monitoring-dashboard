"""
Configuration settings for the Model Monitoring Dashboard.
"""
from pathlib import Path
import logging
import os

# Base directory
BASE_DIR = Path(__file__).parent

# Storage directories
STORAGE_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
STORAGE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Session Management Settings
SESSION_CLEANUP_DAYS = int(os.getenv('SESSION_CLEANUP_DAYS', '30'))  # Clean up sessions older than 30 days
AUTO_CLEANUP_ON_START = os.getenv('AUTO_CLEANUP_ON_START', 'false').lower() == 'true'

# Storage Settings
SESSIONS_DIR = os.getenv('SESSIONS_DIR', 'sessions')
RESULTS_DIR = os.getenv('RESULTS_DIR', 'results')

# Model Settings
MAX_MODEL_SIZE_MB = int(os.getenv('MAX_MODEL_SIZE_MB', '100'))  # Maximum model file size in MB
SUPPORTED_MODEL_FORMATS = ['.pkl', '.joblib', '.pickle']

# Data Generation Settings
DEFAULT_SAMPLES = int(os.getenv('DEFAULT_SAMPLES', '1000'))
MAX_SAMPLES = int(os.getenv('MAX_SAMPLES', '10000'))
DEFAULT_FEATURES = int(os.getenv('DEFAULT_FEATURES', '5'))

# Performance Settings
DEFAULT_TIMING_ITERATIONS = int(os.getenv('DEFAULT_TIMING_ITERATIONS', '3'))
MAX_TIMING_ITERATIONS = int(os.getenv('MAX_TIMING_ITERATIONS', '10'))

# Explainability Settings
SHAP_MAX_SAMPLES = int(os.getenv('SHAP_MAX_SAMPLES', '100'))
SHAP_BACKGROUND_SAMPLES = int(os.getenv('SHAP_BACKGROUND_SAMPLES', '100'))

# UI Settings
DEFAULT_PAGE = os.getenv('DEFAULT_PAGE', '🏠 Home')
SHOW_DEBUG_INFO = os.getenv('SHOW_DEBUG_INFO', 'false').lower() == 'true'

# Export Settings
DEFAULT_EXPORT_FORMAT = os.getenv('DEFAULT_EXPORT_FORMAT', 'json')  # json, csv, excel
MAX_EXPORT_SIZE_MB = int(os.getenv('MAX_EXPORT_SIZE_MB', '50'))

# Default test settings
DEFAULT_TEST_ITERATIONS = 3
MAX_SAMPLES_VISUALIZATION = 1000

# File extensions
ALLOWED_MODEL_EXTENSIONS = ['.pkl', '.joblib', '.h5']
ALLOWED_DATA_EXTENSIONS = ['.csv', '.parquet', '.feather']))

def get_cleanup_config():
    """Get cleanup configuration."""
    return {
        'session_cleanup_days': SESSION_CLEANUP_DAYS,
        'auto_cleanup_on_start': AUTO_CLEANUP_ON_START,
        'sessions_dir': SESSIONS_DIR,
        'results_dir': RESULTS_DIR
    }

def get_model_config():
    """Get model configuration."""
    return {
        'max_size_mb': MAX_MODEL_SIZE_MB,
        'supported_formats': SUPPORTED_MODEL_FORMATS
    }

def get_data_config():
    """Get data generation configuration."""
    return {
        'default_samples': DEFAULT_SAMPLES,
        'max_samples': MAX_SAMPLES,
        'default_features': DEFAULT_FEATURES
    }

def get_performance_config():
    """Get performance testing configuration."""
    return {
        'default_timing_iterations': DEFAULT_TIMING_ITERATIONS,
        'max_timing_iterations': MAX_TIMING_ITERATIONS
    }

def get_explainability_config():
    """Get explainability configuration."""
    return {
        'shap_max_samples': SHAP_MAX_SAMPLES,
        'shap_background_samples': SHAP_BACKGROUND_SAMPLES
    }
