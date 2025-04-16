import os
import logging
import yaml
import time
import numpy as np
from pathlib import Path

from data_loader import DataLoader
from scoring_functions import get_conformity_score
from methods import get_conformal_method
from evaluation import ConformalEvaluator

def main(config_path):
    """
    Run the conformal prediction pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    """
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
    
    # Load data and model
    data_loader = DataLoader(config_path)
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_data()
    model = data_loader.load_model()
    
    # Get conformity score
    score_name = config['conformal_prediction']['scoring_function']
    conformity_score = get_conformity_score(score_name)
    
    # Get conformal method
    method_name = config['conformal_prediction']['method']
    alpha = config['conformal_prediction']['alpha']
    method = get_conformal_method(method_name, model, conformity_score, alpha)
    
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
    
    # Explicitly generate and save prediction interval plot
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
    
    main(args.config) 