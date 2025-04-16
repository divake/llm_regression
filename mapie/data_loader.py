import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import logging
import yaml

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and preprocessing data for conformal prediction."""
    
    def __init__(self, config_path):
        """Initialize with configuration file path."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config['output']['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_data(self):
        """Load train, validation, and test datasets."""
        logger.info("Loading datasets...")
        
        train_data = pd.read_csv(self.config['data']['train_path'])
        validation_data = pd.read_csv(self.config['data']['validation_path'])
        test_data = pd.read_csv(self.config['data']['test_path'])
        
        # Extract feature column names (exclude text_features and target)
        feature_cols = [col for col in train_data.columns if col.startswith('x')]
        target_col = 'target'
        
        logger.info(f"Using features: {feature_cols}")
        logger.info(f"Target column: {target_col}")
        
        # Extract features and target
        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        
        X_val = validation_data[feature_cols].values
        y_val = validation_data[target_col].values
        
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values
        
        logger.info(f"Loaded datasets: Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def load_model(self):
        """Load the pre-trained model."""
        logger.info(f"Loading model from {self.config['data']['model_path']}...")
        model = joblib.load(self.config['data']['model_path'])
        logger.info(f"Loaded model: {type(model).__name__}")
        return model 