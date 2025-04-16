import numpy as np
import pandas as pd
import time
import logging
import os
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

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
        
        # Save prediction data for plotting
        self.prediction_data = {
            'y_test': y_test,
            'y_pred': y_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'in_interval': in_interval
        }
        
        logger.info(f"Evaluation results for {method.__class__.__name__}:")
        logger.info(f"  Theoretical coverage: {method.theoretical_coverage}")
        logger.info(f"  Empirical coverage: {empirical_coverage:.4f}")
        logger.info(f"  Average interval width: {interval_width:.4f}")
        logger.info(f"  Mean absolute error: {mae:.4f}")
        logger.info(f"  Efficiency (width/MAE ratio): {efficiency:.4f}")
        logger.info(f"  Prediction time: {prediction_time:.4f} seconds")
        
        return self.results
    
    def plot_prediction_intervals(self, output_dir=None, figsize=(12, 8), sort_by_prediction=True):
        """
        Create a visualization of prediction intervals.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save the plot. If None, use the one from config.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        sort_by_prediction : bool, optional
            Whether to sort samples by prediction value for better visualization.
        
        Returns
        -------
        str
            Path to the saved plot.
        """
        if not hasattr(self, 'prediction_data'):
            raise ValueError("No prediction data available. Run evaluate() first.")
            
        if output_dir is None:
            output_dir = self.config['output']['results_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract prediction data
        y_test = self.prediction_data['y_test']
        y_pred = self.prediction_data['y_pred']
        lower_bound = self.prediction_data['lower_bound']
        upper_bound = self.prediction_data['upper_bound']
        in_interval = self.prediction_data['in_interval']
        
        # Sort data by prediction value if requested
        if sort_by_prediction:
            sort_idx = np.argsort(y_pred)
            y_test = y_test[sort_idx]
            y_pred = y_pred[sort_idx]
            lower_bound = lower_bound[sort_idx]
            upper_bound = upper_bound[sort_idx]
            in_interval = in_interval[sort_idx]
        
        # Calculate interval width for each sample
        interval_width = upper_bound - lower_bound
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot predictions as a line
        plt.plot(range(len(y_pred)), y_pred, 'b-', label='Predictions')
        
        # Plot prediction intervals
        plt.fill_between(
            range(len(y_pred)), 
            lower_bound, 
            upper_bound, 
            alpha=0.2, 
            color='blue', 
            label='Prediction Intervals'
        )
        
        # Plot true values
        scatter = plt.scatter(
            range(len(y_test)), 
            y_test, 
            c=['green' if covered else 'red' for covered in in_interval],
            marker='o', 
            s=20, 
            label='True Values (green=covered)'
        )
        
        # Set plot title and labels
        coverage = np.mean(in_interval)
        target_coverage = 1 - self.results['alpha']
        plt.title(f"Prediction Intervals (Coverage: {coverage:.4f}, Target: {target_coverage:.4f})")
        plt.xlabel("Sample Index (sorted by prediction)" if sort_by_prediction else "Sample Index")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        experiment_name = self.config['experiment_name']
        plot_path = os.path.join(output_dir, f"{experiment_name}_intervals.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction interval plot saved to {plot_path}")
        
        # Create second plot showing interval widths
        plt.figure(figsize=figsize)
        plt.bar(range(len(interval_width)), interval_width, alpha=0.7)
        plt.title(f"Prediction Interval Widths (Average: {np.mean(interval_width):.4f})")
        plt.xlabel("Sample Index (sorted by prediction)" if sort_by_prediction else "Sample Index")
        plt.ylabel("Interval Width")
        plt.grid(True, axis='y')
        
        # Save the second plot
        width_plot_path = os.path.join(output_dir, f"{experiment_name}_interval_widths.png")
        plt.savefig(width_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Interval width plot saved to {width_plot_path}")
        
        plt.close('all')
        return plot_path
    
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
        if hasattr(self, 'prediction_data'):
            self.plot_prediction_intervals(output_dir)
        
        return results_path 