import numpy as np
import pandas as pd
import time
import logging
import os
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

class ConformalEvaluator:
    """Class for evaluating conformal prediction methods."""
    
    def __init__(self, config):
        """
        Initialize the evaluator with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary.
        """
        self.config = config
        self.results = {}
    
    def evaluate(self, method, X_test, y_test):
        """
        Evaluate a conformal prediction method on test data.
        
        Parameters
        ----------
        method : ConformalMethod
            Calibrated conformal method.
        X_test : array-like
            Test feature matrix.
        y_test : array-like
            Test target values.
            
        Returns
        -------
        dict
            Evaluation results.
        """
        start_time = time.time()
        
        # Generate prediction intervals
        lower_bound, upper_bound, y_pred = method.predict(X_test)
        
        prediction_time = time.time() - start_time
        
        # Calculate coverage
        in_interval = (y_test >= lower_bound) & (y_test <= upper_bound)
        empirical_coverage = np.mean(in_interval)
        
        # Calculate average interval width
        interval_width = np.mean(upper_bound - lower_bound)
        
        # Calculate average absolute error of point predictions
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Calculate prediction interval efficiency (width/MAE ratio)
        efficiency = interval_width / mae if mae > 0 else float('inf')
        
        # Record results
        self.results = {
            'method_name': method.__class__.__name__,
            'alpha': method.alpha,
            'theoretical_coverage': method.theoretical_coverage,
            'empirical_coverage': empirical_coverage,
            'interval_width': interval_width,
            'mae': mae,
            'efficiency': efficiency,
            'prediction_time': prediction_time,
            'training_cost': method.training_cost(len(X_test)),
            'evaluation_cost': method.evaluation_cost(len(X_test), len(X_test)),
        }
        
        logger.info(f"Evaluation results for {method.__class__.__name__}:")
        logger.info(f"  Theoretical coverage: {method.theoretical_coverage}")
        logger.info(f"  Empirical coverage: {empirical_coverage:.4f}")
        logger.info(f"  Average interval width: {interval_width:.4f}")
        logger.info(f"  Mean absolute error: {mae:.4f}")
        logger.info(f"  Efficiency (width/MAE ratio): {efficiency:.4f}")
        logger.info(f"  Prediction time: {prediction_time:.4f} seconds")
        
        return self.results
    
    def save_results(self, output_dir=None):
        """
        Save evaluation results to a CSV file and create visualizations.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save results. If None, use the one from config.
        """
        if output_dir is None:
            output_dir = self.config['output']['results_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results as CSV
        experiment_name = self.config['experiment_name']
        results_path = os.path.join(output_dir, f"{experiment_name}_results.csv")
        pd.DataFrame([self.results]).to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Create visualization of prediction intervals
        # This would be part of a more complete implementation
        
        return results_path 