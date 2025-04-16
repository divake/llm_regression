import os
import argparse
import logging
import torch
import joblib
import yaml
import numpy as np

from utils import load_config, load_data, create_torch_datasets
from regression_scoring_functions import get_scoring_function
from regression_trainer import ConformalRegressionTrainer, create_rf_wrapper

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Conformal Prediction for Regression')
    
    # Use absolute path to config.yaml in the current directory
    default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    parser.add_argument('--config', type=str, default=default_config_path,
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset (overrides config)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained regression model')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--scoring_function', type=str, default=None,
                        help='Scoring function type (overrides config)')
    parser.add_argument('--symmetric', action='store_true',
                        help='Use symmetric intervals (overrides config)')
    parser.add_argument('--asymmetric', action='store_false', dest='symmetric',
                        help='Use asymmetric intervals (overrides config)')
    parser.add_argument('--target_coverage', type=float, default=None,
                        help='Target coverage level (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    
    return parser.parse_args()

def setup_environment(config):
    """Setup environment"""
    # Set random seed for reproducibility
    seed = config['data']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create required directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    
    # Check if Friedman dataset directory exists
    friedman_dir = config['paths']['friedman_data_dir']
    if not os.path.exists(friedman_dir):
        logging.warning(f"Friedman dataset directory {friedman_dir} does not exist. Creating it...")
        os.makedirs(friedman_dir, exist_ok=True)
        
        # Provide instruction to user if directory was just created
        logging.warning(f"Please ensure Friedman dataset files are placed in {friedman_dir} directory.")
        logging.warning("Expected files: train_data.csv, validation_data.csv, test_data.csv")
    
    # Set up logging
    log_file = os.path.join(config['paths']['output_dir'], 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_or_train_base_model(config, data, model_path=None):
    """
    Load or train base regression model
    
    Args:
        config: Configuration dictionary
        data: Data dictionary
        model_path: Path to pre-trained model (optional)
        
    Returns:
        model: Trained/loaded base model
    """
    if model_path is not None:
        logging.info(f"Loading pre-trained base model from {model_path}")
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            logging.warning("Training new model instead")
    
    # For this demo, we'll use a pre-trained random forest
    # In a real application, you would train your base model here
    from sklearn.ensemble import RandomForestRegressor
    
    logging.info("Training new Random Forest regression model")
    x_train, y_train = data['train']
    
    rf_params = config['rf_model']
    model = RandomForestRegressor(
        n_estimators=rf_params['n_estimators'],
        max_depth=rf_params['max_depth'],
        random_state=rf_params['random_state']
    )
    
    model.fit(x_train, y_train)
    
    # Save the trained model
    model_save_path = os.path.join(config['paths']['model_dir'], 'base_model.joblib')
    joblib.dump(model, model_save_path)
    logging.info(f"Saved base model to {model_save_path}")
    
    return model

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_path:
        config['paths']['data_dir'] = os.path.dirname(args.data_path)
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.scoring_function:
        config['scoring_functions']['type'] = args.scoring_function
    if args.symmetric is not None:
        config['scoring_functions']['symmetric'] = args.symmetric
    if args.target_coverage:
        config['training']['target_coverage'] = args.target_coverage
    if args.seed:
        config['data']['seed'] = args.seed
    
    # Setup environment
    setup_environment(config)
    
    # Log configuration
    logging.info("Configuration:")
    logging.info(yaml.dump(config, default_flow_style=False))
    
    # Load and split data
    data = load_data(config, args.data_path)
    
    # Load or train base model
    base_model = load_or_train_base_model(config, data, args.model_path)
    
    # Create PyTorch datasets and dataloaders
    dataloaders = create_torch_datasets(data, config)
    
    # Create base model wrapper for PyTorch compatibility
    base_model_wrapper = create_rf_wrapper(base_model)
    
    # Determine feature dimension from data
    x_train, _ = data['train']
    feature_dim = x_train.shape[1]
    
    # Get scoring function
    scoring_fn = get_scoring_function(config, feature_dim=feature_dim)
    logging.info(f"Using scoring function: {type(scoring_fn).__name__}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Move models to device
    base_model_wrapper.to(device)
    scoring_fn.to(device)
    
    # Create trainer
    trainer = ConformalRegressionTrainer(
        base_model=base_model_wrapper,
        scoring_fn=scoring_fn,
        train_loader=dataloaders['train'],
        cal_loader=dataloaders['calibration'],
        test_loader=dataloaders['test'],
        device=device,
        config=config
    )
    
    # Train scoring function
    logging.info("Starting training")
    history, best_model_state, metrics = trainer.train()
    
    # Log final results
    logging.info("Training completed!")
    logging.info(f"Final test metrics:")
    for key, value in metrics.items():
        logging.info(f"  {key}: {value:.4f}")
    
    # Save configuration with results
    results_config = config.copy()
    results_config['results'] = metrics
    with open(os.path.join(config['paths']['output_dir'], 'results.yaml'), 'w') as f:
        yaml.dump(results_config, f, default_flow_style=False)
    
    logging.info(f"Results saved to {config['paths']['output_dir']}")

if __name__ == "__main__":
    main() 