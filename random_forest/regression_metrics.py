import torch
import numpy as np

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_prediction_intervals(predictions, widths):
    """
    Compute prediction intervals given predictions and widths
    
    Args:
        predictions: Model point predictions
        widths: Interval widths (half-width for symmetric intervals)
        
    Returns:
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
    """
    lower_bounds = predictions - widths
    upper_bounds = predictions + widths
    
    return lower_bounds, upper_bounds

def compute_coverage(targets, lower_bounds, upper_bounds):
    """
    Compute empirical coverage of prediction intervals
    
    Args:
        targets: True target values
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
        
    Returns:
        coverage: Fraction of targets within intervals
    """
    # Check if targets are within intervals
    in_interval = (targets >= lower_bounds) & (targets <= upper_bounds)
    
    # Compute coverage
    coverage = in_interval.float().mean()
    
    return coverage

def compute_average_width(lower_bounds, upper_bounds):
    """
    Compute average width of prediction intervals
    
    Args:
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
        
    Returns:
        avg_width: Average width of prediction intervals
    """
    widths = upper_bounds - lower_bounds
    avg_width = widths.mean()
    
    return avg_width

def compute_conformal_calibration(scoring_fn, calibration_errors, target_coverage=0.9):
    """
    Compute calibrated scaling factor using conformal prediction
    
    Args:
        scoring_fn: Trained scoring function 
        calibration_errors: Absolute errors on calibration set
        target_coverage: Target coverage level (default: 0.9)
        
    Returns:
        scale_factor: Scaling factor for prediction intervals to achieve target coverage
    """
    scoring_fn.eval()
    
    with torch.no_grad():
        # Predict widths for calibration errors
        predicted_widths = scoring_fn(calibration_errors).squeeze()
        
        # Get calibration scores (predicted width / actual error)
        calibration_scores = predicted_widths / calibration_errors.squeeze()
        
        # Sort scores
        sorted_scores, _ = torch.sort(calibration_scores)
        
        # Find quantile
        n = len(sorted_scores)
        q_index = int(np.ceil((n + 1) * target_coverage)) - 1
        q_index = min(max(0, q_index), n - 1)  # Ensure index is valid
        
        # Get scaling factor
        scale_factor = sorted_scores[q_index]
        
    return scale_factor

def evaluate_prediction_intervals(targets, predictions, lower_bounds, upper_bounds):
    """
    Evaluate prediction intervals
    
    Args:
        targets: True target values
        predictions: Model point predictions
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Convert to numpy if tensors
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(lower_bounds, torch.Tensor):
        lower_bounds = lower_bounds.detach().cpu().numpy()
    if isinstance(upper_bounds, torch.Tensor):
        upper_bounds = upper_bounds.detach().cpu().numpy()
    
    # Compute coverage
    in_interval = (targets >= lower_bounds) & (targets <= upper_bounds)
    coverage = np.mean(in_interval)
    
    # Compute average width
    widths = upper_bounds - lower_bounds
    avg_width = np.mean(widths)
    
    # Compute mean absolute error
    mae = np.mean(np.abs(targets - predictions))
    
    # Compute root mean squared error
    rmse = np.sqrt(np.mean(np.square(targets - predictions)))
    
    # Compute interval sharpness (smaller is better)
    sharpness = np.mean(widths)
    
    # Compute interval score (combines coverage and width)
    alpha = 1 - coverage  # Non-coverage probability
    unit_intervals = (upper_bounds - lower_bounds)
    penalties = (2/alpha) * (np.maximum(0, lower_bounds - targets) + np.maximum(0, targets - upper_bounds))
    interval_score = np.mean(unit_intervals + penalties)
    
    # Compute normalized interval score (divide by target range)
    target_range = np.max(targets) - np.min(targets)
    normalized_interval_score = interval_score / target_range if target_range > 0 else interval_score
    
    # Compute efficiency (coverage / width ratio - higher is better)
    efficiency = coverage / (avg_width + 1e-8)
    
    # Calculate miscoverage rate
    miscoverage = 1 - coverage
    
    # Calculate average widths for covered and non-covered points
    covered_widths = widths[in_interval]
    uncovered_widths = widths[~in_interval]
    avg_covered_width = np.mean(covered_widths) if len(covered_widths) > 0 else 0
    avg_uncovered_width = np.mean(uncovered_widths) if len(uncovered_widths) > 0 else 0
    
    # Compute correlation between width and error
    abs_errors = np.abs(targets - predictions)
    width_error_corr = np.corrcoef(widths, abs_errors)[0, 1] if len(widths) > 1 else 0
    
    metrics = {
        'coverage': coverage,
        'avg_width': avg_width,
        'interval_score': interval_score, 
        'norm_interval_score': normalized_interval_score,
        'efficiency': efficiency,
        'miscoverage': miscoverage,
        'mae': mae,
        'rmse': rmse,
        'sharpness': sharpness,
        'avg_covered_width': avg_covered_width,
        'avg_uncovered_width': avg_uncovered_width,
        'width_error_corr': width_error_corr
    }
    
    return metrics 