import os
import logging
import yaml
import time
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor

from data_loader import DataLoader
from scoring_functions import get_conformity_score
from methods import get_conformal_method
from evaluation import ConformalEvaluator

def train_quantile_models(X_train, y_train, alpha, quantile_model_type='gradient_boosting'):
    """
    Train low and high quantile regression models for CQR method.
    
    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target values.
    alpha : float
        Significance level (1-alpha = coverage level).
    quantile_model_type : str, default='gradient_boosting'
        Type of quantile regression model to use.
        
    Returns
    -------
    tuple
        Tuple of (low_quantile_model, high_quantile_model).
    """
    # Set quantile levels
    low_quantile = alpha / 2
    high_quantile = 1 - alpha / 2
    
    logger = logging.getLogger(__name__)
    logger.info(f"Training quantile models (low={low_quantile}, high={high_quantile})")
    
    # Train low quantile model
    start_time = time.time()
    if quantile_model_type == 'quantile_regressor':
        low_model = QuantileRegressor(
            quantile=low_quantile,
            alpha=0.0,  # No regularization
            solver='highs'
        )
    else:  # default to gradient_boosting
        low_model = GradientBoostingRegressor(
            loss='quantile',
            alpha=low_quantile,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
    
    low_model.fit(X_train, y_train)
    logger.info(f"Low quantile model trained in {time.time() - start_time:.2f} seconds")
    
    # Train high quantile model
    start_time = time.time()
    if quantile_model_type == 'quantile_regressor':
        high_model = QuantileRegressor(
            quantile=high_quantile,
            alpha=0.0,  # No regularization
            solver='highs'
        )
    else:  # default to gradient_boosting
        high_model = GradientBoostingRegressor(
            loss='quantile',
            alpha=high_quantile,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
    
    high_model.fit(X_train, y_train)
    logger.info(f"High quantile model trained in {time.time() - start_time:.2f} seconds")
    
    return low_model, high_model

def main(config_path):
    """
    Run the conformal prediction pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file.
        
    Returns
    -------
    dict
        Dictionary with evaluation results.
    """
    # Ensure config_path is an absolute path
    if not os.path.isabs(config_path):
        # If it's a relative path, make it relative to the current working directory
        config_path = os.path.abspath(config_path)
    
    print(f"Using configuration file: {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    log_level = config['output']['log_level']
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting conformal prediction experiment: {config['experiment_name']}")
    
    # Create output directory
    output_dir = config['output']['results_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_loader = DataLoader(config_path)
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_data()
    
    # Get method and parameters
    method_name = config['conformal_prediction'].get('method', "split")
    alpha = config['conformal_prediction']['alpha']
    
    # Special handling for CQR method
    if method_name.startswith('cqr'):
        logger.info(f"Setting up CQR method: {method_name}")
        symmetric = method_name == "cqr_symmetric"
        score_name = 'quantile_residual_symmetric' if symmetric else 'quantile_residual'
        
        # Load the original Random Forest model for point predictions
        logger.info("Loading the original Random Forest model for point predictions")
        base_model = data_loader.load_model()
        
        # Check if we need to train models or load existing ones
        use_existing_models = config.get('use_existing_quantile_models', False)
        models_dir = os.path.dirname(config['output']['results_dir'])
        low_model_path = os.path.join(models_dir, 'low_quantile_model.joblib')
        high_model_path = os.path.join(models_dir, 'high_quantile_model.joblib')
        
        if use_existing_models and os.path.exists(low_model_path) and os.path.exists(high_model_path):
            logger.info(f"Loading existing quantile models")
            low_model = joblib.load(low_model_path)
            high_model = joblib.load(high_model_path)
        else:
            # Train quantile models
            logger.info(f"Training new quantile models")
            quantile_model_type = config.get('quantile_model_type', 'gradient_boosting')
            low_model, high_model = train_quantile_models(X_train, y_train, alpha, quantile_model_type)
            
            # Save trained models
            logger.info(f"Saving quantile models to {models_dir}")
            os.makedirs(models_dir, exist_ok=True)
            joblib.dump(low_model, low_model_path)
            joblib.dump(high_model, high_model_path)
        
        # Create model tuple for CQR
        model = (low_model, high_model)
        conformity_score = get_conformity_score(score_name)
        
        # Additional parameters
        kwargs = {'base_model': base_model}  # Pass the base model for point predictions
        if 'low_quantile' in config['conformal_prediction']:
            kwargs['low_quantile'] = config['conformal_prediction']['low_quantile']
        if 'high_quantile' in config['conformal_prediction']:
            kwargs['high_quantile'] = config['conformal_prediction']['high_quantile']
    else:
        # For standard methods, just load the regular model
        model = data_loader.load_model()
        score_name = config['conformal_prediction']['scoring_function']
        conformity_score = get_conformity_score(score_name)
        kwargs = {}
    
    logger.info(f"Running method: {method_name}")
    
    # Get the conformal method
    method = get_conformal_method(method_name, model, conformity_score, alpha, **kwargs)
    
    # Calibrate the method
    logger.info(f"Calibrating {method_name} method...")
    start_time = time.time()
    method.calibrate(X_train, y_train, X_val, y_val)
    calibration_time = time.time() - start_time
    logger.info(f"Calibration completed in {calibration_time:.2f} seconds")
    
    # Evaluate the method
    logger.info("Evaluating on test data...")
    evaluator = ConformalEvaluator(config)
    results = evaluator.evaluate(method, X_test, y_test)
    
    # Save results and create visualizations
    results_path = evaluator.save_results()
    logger.info(f"Results saved to {results_path}")
    
    # Generate and save prediction interval plot
    plot_path = evaluator.plot_prediction_intervals()
    logger.info(f"Visualization created at: {plot_path}")
    
    print("\nEvaluation Results:")
    print(f"  Method: {results['method_name']}")
    print(f"  Theoretical coverage: {results['theoretical_coverage']}")
    print(f"  Empirical coverage: {results['empirical_coverage']:.4f}")
    print(f"  Average interval width: {results['interval_width']:.4f}")
    print(f"  Mean absolute error: {results['mae']:.4f}")
    print(f"  Efficiency (width/MAE ratio): {results['efficiency']:.4f}")
    print(f"  Prediction time: {results['prediction_time']:.4f} seconds")
    print(f"\nPrediction interval plot saved to: {plot_path}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run conformal prediction experiment')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Convert relative path to absolute path if needed
    config_path = args.config
    if not os.path.isabs(config_path):
        # Get the directory of the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    main(config_path)