"""
Storage utilities for model monitoring results.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
import os
import glob
import atexit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_temp_files():
    """Clean up temporary files when the application exits."""
    try:
        # Get all temp files
        temp_files = glob.glob("temp_*")
        
        # Remove each temp file
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {str(e)}")

# Register the cleanup function to run on exit
atexit.register(cleanup_temp_files)

class ResultsStorage:
    """Handles storage of test results and visualizations."""
    
    def __init__(self, storage_dir: str = "results"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        # Clean any existing temp files on initialization
        cleanup_temp_files()
    
    def save_test_results(self, test_results: Dict[str, Any], 
                         test_name: Optional[str] = None) -> str:
        """
        Save test results to file.
        
        Args:
            test_results: Results dictionary to save
            test_name: Optional name for the test
            
        Returns:
            str: Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if test_name is None:
            test_name = f"test_{timestamp}"
            
        filename = f"{test_name}_{timestamp}.json"
        file_path = self.storage_dir / filename
        
        # Add metadata
        results_with_metadata = {
            'test_name': test_name,
            'timestamp': datetime.now().isoformat(),
            'results': test_results
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(results_with_metadata, f, indent=2, default=str)
            
            logger.info(f"Saved test results to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save test results: {str(e)}")
            raise
    
    def load_test_results(self, filename: str) -> Dict[str, Any]:
        """Load test results from file."""
        file_path = self.storage_dir / filename
        
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Loaded results from {file_path}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load results: {str(e)}")
            raise
    
    def save_visualization(self, image_data: str, 
                         filename: Optional[str] = None) -> str:
        """
        Save visualization image.
        
        Args:
            image_data: Base64 encoded image data
            filename: Optional filename
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"viz_{timestamp}.png"
        
        file_path = self.storage_dir / filename
        
        try:
            import base64
            image_bytes = base64.b64decode(image_data)
            
            with open(file_path, 'wb') as f:
                f.write(image_bytes)
            
            logger.info(f"Saved visualization to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save visualization: {str(e)}")
            raise
    
    def list_results(self) -> List[Dict[str, Any]]:
        """List all saved test results."""
        results_list = []
        
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                results_list.append({
                    'filename': file_path.name,
                    'test_name': data.get('test_name', ''),
                    'timestamp': data.get('timestamp', ''),
                    'size': file_path.stat().st_size
                })
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {str(e)}")
        
        # Sort by timestamp
        results_list.sort(key=lambda x: x['timestamp'], reverse=True)
        return results_list


class DataExporter:
    """Handles export of data in various formats."""
    
    @staticmethod
    def export_to_csv(data: Union[pd.DataFrame, np.ndarray], 
                     filename: str, 
                     feature_names: Optional[List[str]] = None) -> str:
        """Export data to CSV format."""
        if isinstance(data, np.ndarray):
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=feature_names)
        else:
            df = data
        
        df.to_csv(filename, index=False)
        logger.info(f"Exported data to {filename}")
        return filename
    
    @staticmethod
    def export_test_results(results_list: List[Dict[str, Any]], 
                          filename: str,
                          format: str = 'csv') -> str:
        """
        Export test results to CSV/Excel format.
        
        Args:
            results_list: List of test results
            filename: Output filename
            format: 'csv' or 'excel'
            
        Returns:
            str: Path to saved file
        """
        # Flatten results data
        flattened_data = []
        
        for result in results_list:
            row = {
                'test_name': result.get('test_name', ''),
                'timestamp': result.get('timestamp', '')
            }
            
            # Add metrics from results
            metrics = result.get('results', {})
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        row[f'{key}_{sub_key}'] = sub_value
                else:
                    row[key] = value
                    
            flattened_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(flattened_data)
        
        # Save based on format
        if format == 'excel':
            df.to_excel(filename, index=False)
        else:
            df.to_csv(filename, index=False)
            
        logger.info(f"Exported test results to {filename}")
        return filename


if __name__ == "__main__":
    # Test the storage functionality
    storage = ResultsStorage()
    
    # Save some test results
    test_results = {
        'performance': {
            'latency_mean_ms': 10.5,
            'throughput_preds_per_sec': 1000,
            'accuracy': 0.95
        }
    }
    
    file_path = storage.save_test_results(test_results, "performance_test")
    print(f"Saved test results to: {file_path}")
    
    # List available results
    results = storage.list_results()
    print("\nAvailable test results:")
    for result in results:
        print(f"- {result['test_name']} ({result['timestamp']})")
    
    # Export results
    export_path = DataExporter.export_test_results(results, "test_results.csv")
    print(f"\nExported results to: {export_path}")
