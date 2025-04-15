import os
import yaml
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

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

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create required directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    
    # Ensure numeric values are properly converted from strings
    if isinstance(config['training']['learning_rate'], str):
        config['training']['learning_rate'] = float(config['training']['learning_rate'])
    if isinstance(config['training']['target_coverage'], str):
        config['training']['target_coverage'] = float(config['training']['target_coverage'])
    if isinstance(config['training']['coverage_weight'], str):
        config['training']['coverage_weight'] = float(config['training']['coverage_weight'])
    if isinstance(config['training']['width_weight'], str):
        config['training']['width_weight'] = float(config['training']['width_weight'])
    if isinstance(config['training']['grad_clip'], str):
        config['training']['grad_clip'] = float(config['training']['grad_clip'])
    if isinstance(config['training']['batch_size'], str):
        config['training']['batch_size'] = int(config['training']['batch_size'])
    if isinstance(config['training']['num_epochs'], str):
        config['training']['num_epochs'] = int(config['training']['num_epochs'])
    
    # Convert scheduler parameters if needed
    if isinstance(config['scheduler']['factor'], str):
        config['scheduler']['factor'] = float(config['scheduler']['factor'])
    if isinstance(config['scheduler']['patience'], str):
        config['scheduler']['patience'] = int(config['scheduler']['patience'])
    if isinstance(config['scheduler']['early_stop_patience'], str):
        config['scheduler']['early_stop_patience'] = int(config['scheduler']['early_stop_patience'])
    
    return config

def load_data(config, dataset_path=None):
    """
    Load dataset and split into train, calibration, and test sets
    
    Args:
        config: Configuration dictionary
        dataset_path: Path to the dataset (optional)
        
    Returns:
        data: Dictionary containing data splits
    """
    logging.info("Loading and splitting dataset...")
    
    # Use Friedman dataset files that already exist
    friedman_dir = config['paths']['friedman_data_dir']
    
    # Convert relative path to absolute if necessary
    if not os.path.isabs(friedman_dir):
        friedman_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), friedman_dir)
    
    # Check if we're using pre-split datasets
    train_path = os.path.join(friedman_dir, "train_data.csv")
    val_path = os.path.join(friedman_dir, "validation_data.csv")
    test_path = os.path.join(friedman_dir, "test_data.csv")
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        logging.info("Using pre-split Friedman datasets")
        
        # Load pre-split datasets
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # For demonstration, assume the target is the last column
        feature_columns = train_df.columns[:-1].tolist()
        target_column = train_df.columns[-1]
        
        # Convert all feature columns to numeric, handling any non-numeric values
        for col in feature_columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            val_df[col] = pd.to_numeric(val_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        
        # Replace any NaN values with 0 (from failed numeric conversion)
        train_df.fillna(0, inplace=True)
        val_df.fillna(0, inplace=True)
        test_df.fillna(0, inplace=True)
        
        # Extract features and targets
        x_train = train_df[feature_columns].values
        y_train = train_df[target_column].values
        
        x_val = val_df[feature_columns].values
        y_val = val_df[target_column].values
        
        x_test = test_df[feature_columns].values
        y_test = test_df[target_column].values
    else:
        # If no path provided or pre-split files don't exist, try the raw dataset
        raw_path = os.path.join(friedman_dir, "friedman2_raw.csv")
        
        if dataset_path is not None:
            # Override with user-provided path
            raw_path = dataset_path
        
        logging.info(f"Loading dataset from {raw_path}")
        
        try:
            df = pd.read_csv(raw_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {raw_path}")
        
        # For demonstration, we'll assume the dataset has features and target columns
        # Adjust this based on your actual dataset structure
        if 'target' in df.columns:
            feature_columns = [col for col in df.columns if col != 'target']
            target_column = 'target'
        else:
            feature_columns = df.columns[:-1].tolist()  # Assume last column is target
            target_column = df.columns[-1]
        
        # Convert all feature columns to numeric
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Replace any NaN values with 0
        df.fillna(0, inplace=True)
        
        features = df[feature_columns].values
        targets = df[target_column].values
        
        # Split data
        train_ratio = config['data']['train_ratio']
        val_ratio = config['data']['val_ratio']
        
        # First split to get training data
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            features, targets, 
            test_size=config['data']['test_ratio'],
            random_state=config['data']['seed']
        )
        
        # Then split training data to get validation (calibration) set
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, 
            test_size=val_ratio_adjusted,
            random_state=config['data']['seed']
        )
    
    # Normalize features if specified
    if config['data']['normalize']:
        # Compute statistics from training set
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        
        # Apply normalization
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std
        
        # Store normalization parameters for later use
        normalization_params = {'mean': mean, 'std': std}
    else:
        normalization_params = None
    
    data = {
        'train': (x_train, y_train),
        'calibration': (x_val, y_val),
        'test': (x_test, y_test),
        'feature_names': feature_columns,
        'target_name': target_column,
        'normalization_params': normalization_params
    }
    
    logging.info(f"Data loaded and split: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")
    
    return data

def create_torch_datasets(data, config):
    """
    Create PyTorch datasets and dataloaders from data splits
    
    Args:
        data: Dictionary containing data splits
        config: Configuration dictionary
        
    Returns:
        dataloaders: Dictionary containing dataloaders
    """
    batch_size = config['training']['batch_size']
    
    # Unpack data
    x_train, y_train = data['train']
    x_cal, y_cal = data['calibration']
    x_test, y_test = data['test']
    
    # Create tensor datasets
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    
    cal_dataset = TensorDataset(
        torch.tensor(x_cal, dtype=torch.float32),
        torch.tensor(y_cal, dtype=torch.float32).unsqueeze(1)
    )
    
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    cal_loader = DataLoader(
        cal_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    dataloaders = {
        'train': train_loader,
        'calibration': cal_loader,
        'test': test_loader
    }
    
    return dataloaders

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
    
    # Plot error vs width relationship
    abs_errors = np.abs(targets - predictions)
    
    # Ensure both arrays have the same shape by flattening them
    widths_flat = widths.flatten()
    abs_errors_flat = abs_errors.flatten()
    
    # Initialize correlation with default value
    correlation = 0
    
    # Check if shapes match
    if len(widths_flat) == len(abs_errors_flat):
        plt.figure(figsize=(10, 6))
        plt.scatter(abs_errors_flat, widths_flat, alpha=0.5, s=10)
        
        # Add trend line
        try:
            z = np.polyfit(abs_errors_flat, widths_flat, 1)
            p = np.poly1d(z)
            sorted_errors = np.sort(abs_errors_flat)
            plt.plot(sorted_errors, p(sorted_errors), "r--", linewidth=2)
            
            # Compute correlation
            correlation = np.corrcoef(abs_errors_flat, widths_flat)[0, 1]
            corr_text = f'Correlation: {correlation:.4f}'
        except:
            corr_text = 'Correlation: N/A'
        
        plt.title(f'Width vs Error Relationship ({corr_text})')
        plt.xlabel('Absolute Error')
        plt.ylabel('Interval Width')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'width_vs_error.png'), dpi=150)
        plt.close()
    else:
        logging.warning(f"Cannot create width vs error plot: Arrays have different shapes ({len(abs_errors_flat)} vs {len(widths_flat)})")
    
    metrics = {
        'coverage': coverage,
        'avg_width': avg_width,
        'interval_score': interval_score, 
        'norm_interval_score': normalized_interval_score,
        'efficiency': efficiency,
        'miscoverage': miscoverage,
        'mae': mae,
        'rmse': rmse,
        'avg_covered_width': avg_covered_width,
        'avg_uncovered_width': avg_uncovered_width,
        'width_error_corr': correlation
    }
    
    return metrics

def compute_prediction_intervals(predictions, scores, tau, symmetric=True):
    """
    Compute prediction intervals using nonconformity scores and calibration factor
    
    Args:
        predictions: Model point predictions
        scores: Nonconformity scores 
        tau: Calibration factor
        symmetric: Whether to use symmetric intervals
        
    Returns:
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
    """
    if symmetric:
        # For symmetric intervals, use same width for both sides
        lower_bounds = predictions - tau * scores
        upper_bounds = predictions + tau * scores
    else:
        # For asymmetric intervals, use separate scores for lower and upper
        assert scores.shape[1] == 2, "Asymmetric intervals require scores with shape [..., 2]"
        lower_scores, upper_scores = scores[:, 0], scores[:, 1]
        lower_bounds = predictions - tau[0] * lower_scores
        upper_bounds = predictions + tau[1] * upper_scores
    
    return lower_bounds, upper_bounds

def plot_training_progress(history, config, output_dir):
    """
    Plot training progress
    
    Args:
        history: Dictionary containing training history
        config: Configuration dictionary
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 10))
    
    # Plot train loss
    plt.subplot(2, 3, 1)
    plt.plot(history['epoch'], history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot coverage
    plt.subplot(2, 3, 2)
    plt.plot(history['epoch'], history['train_coverage'], label='Train')
    plt.plot(history['epoch'], history['val_coverage'], label='Validation')
    target_coverage = config['training']['target_coverage']
    plt.axhline(y=target_coverage, color='r', linestyle='--', label=f'Target ({target_coverage})')
    plt.title('Coverage')
    plt.xlabel('Epoch')
    plt.ylabel('Coverage')
    plt.legend()
    plt.grid(True)
    
    # Plot width
    plt.subplot(2, 3, 3)
    plt.plot(history['epoch'], history['train_width'], label='Train')
    plt.plot(history['epoch'], history['val_width'], label='Validation')
    plt.title('Average Width')
    plt.xlabel('Epoch')
    plt.ylabel('Width')
    plt.legend()
    plt.grid(True)
    
    # Plot calibration factor
    plt.subplot(2, 3, 4)
    if isinstance(history['calibration_factor'][0], (list, tuple, np.ndarray)):
        # For asymmetric intervals
        lower_factors = [factor[0] for factor in history['calibration_factor']]
        upper_factors = [factor[1] for factor in history['calibration_factor']]
        plt.plot(history['epoch'], lower_factors, label='Lower')
        plt.plot(history['epoch'], upper_factors, label='Upper')
        plt.legend()
    else:
        # For symmetric intervals
        plt.plot(history['epoch'], history['calibration_factor'])
    plt.title('Calibration Factor')
    plt.xlabel('Epoch')
    plt.ylabel('Factor')
    plt.grid(True)
    
    # Plot interval score
    plt.subplot(2, 3, 5)
    plt.plot(history['epoch'], history['val_interval_score'])
    plt.title('Validation Interval Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score (lower is better)')
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 3, 6)
    plt.plot(history['epoch'], history['learning_rate'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=150)
    plt.close()

def plot_prediction_intervals(targets, predictions, lower_bounds, upper_bounds, config, output_dir):
    """
    Plot prediction intervals
    
    Args:
        targets: True target values
        predictions: Model point predictions
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
        config: Configuration dictionary
        output_dir: Directory to save plots
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
    
    # Handle different shapes of bounds
    n_samples = targets.size if isinstance(targets, np.ndarray) else len(targets)
    targets_flat = targets.flatten()
    
    if lower_bounds.ndim > 1 and lower_bounds.shape[0] == n_samples and lower_bounds.shape[1] == n_samples:
        # Case where bounds have shape (n_samples, n_samples)
        # For each target, check if it's within ANY of the intervals
        targets_expanded = targets_flat[:, np.newaxis]
        in_interval = np.any((targets_expanded >= lower_bounds) & (targets_expanded <= upper_bounds), axis=1)
        
        # Use only the diagonal elements for plotting
        lower_bounds = np.diag(lower_bounds)
        upper_bounds = np.diag(upper_bounds)
    else:
        # Standard case where bounds have same shape as targets
        in_interval = (targets >= lower_bounds) & (targets <= upper_bounds)
    
    coverage = np.mean(in_interval)
    widths = upper_bounds - lower_bounds
    
    # Attempt to create width vs error plot
    try:
        # Create a plot of interval width vs absolute error
        abs_error = np.abs(targets.flatten() - predictions.flatten())
        
        plt.figure(figsize=(10, 6))
        plt.scatter(widths.flatten(), abs_error.flatten(), alpha=0.3, s=10)
        plt.title('Interval Width vs Absolute Error')
        plt.xlabel('Interval Width')
        plt.ylabel('Absolute Error')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'width_vs_error.png'), dpi=150)
        plt.close()
    except Exception as e:
        logging.warning(f"Cannot create width vs error plot: {str(e)}")
    
    # Sort by prediction for better visualization
    idx = np.argsort(predictions.flatten())
    targets_sorted = targets.flatten()[idx]
    preds_sorted = predictions.flatten()[idx]
    lower_sorted = lower_bounds.flatten()[idx]
    upper_sorted = upper_bounds.flatten()[idx]
    in_interval_sorted = in_interval.flatten()[idx]
    
    # Take a subset of samples for clarity
    n_samples = min(500, len(idx))
    step = max(1, len(idx) // n_samples)
    subset_idx = np.arange(0, len(idx), step)[:n_samples]
    
    # Plot intervals
    plt.figure(figsize=(12, 8))
    plt.fill_between(
        range(len(subset_idx)),
        lower_sorted[subset_idx],
        upper_sorted[subset_idx],
        alpha=0.3,
        color='blue',
        label='Prediction Intervals'
    )
    
    # Plot predictions
    plt.plot(
        range(len(subset_idx)),
        preds_sorted[subset_idx],
        'b-',
        label='Predictions'
    )
    
    # Plot targets
    plt.scatter(
        range(len(subset_idx)),
        targets_sorted[subset_idx],
        color=['g' if covered else 'r' for covered in in_interval_sorted[subset_idx]],
        alpha=0.7,
        s=15,
        label='True Values (green=covered)'
    )
    
    target_coverage = config['training']['target_coverage']
    plt.title(f'Prediction Intervals (Coverage: {coverage:.4f}, Target: {target_coverage:.4f})')
    plt.xlabel('Sample Index (sorted by prediction)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_intervals.png'), dpi=150)
    plt.close()
    
    # Plot the distribution of widths
    plt.figure(figsize=(10, 6))
    plt.hist(widths.flatten(), bins=50, alpha=0.7)
    plt.title('Distribution of Interval Widths')
    plt.xlabel('Width')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'width_distribution.png'), dpi=150)
    plt.close()

def save_results(targets, predictions, lower_bounds, upper_bounds, scores, metrics, config, output_dir):
    """
    Save prediction results
    
    Args:
        targets: True target values
        predictions: Model point predictions
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
        scores: Nonconformity scores
        metrics: Evaluation metrics
        config: Configuration dictionary
        output_dir: Directory to save results
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
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    
    # Ensure all arrays are properly flattened to 1D
    targets_flat = targets.flatten()
    predictions_flat = predictions.flatten()
    
    # Ensure all arrays have the same length
    n_samples = len(targets_flat)
    logging.info(f"Number of samples: {n_samples}")
    logging.info(f"Shapes - targets: {targets.shape}, predictions: {predictions.shape}, "
                f"lower_bounds: {lower_bounds.shape}, upper_bounds: {upper_bounds.shape}")
    
    # Handle different shapes of bounds
    if lower_bounds.ndim > 1 and lower_bounds.shape[0] == n_samples and lower_bounds.shape[1] == n_samples:
        # Case where bounds have shape (n_samples, n_samples)
        # For each target, check if it's within ANY of the intervals
        targets_expanded = targets_flat[:, np.newaxis]
        in_interval = np.any((targets_expanded >= lower_bounds) & (targets_expanded <= upper_bounds), axis=1)
        
        # Use only the diagonal elements for width calculation (as an approximation)
        lower_bounds_flat = np.diag(lower_bounds)
        upper_bounds_flat = np.diag(upper_bounds)
    else:
        # Standard case where bounds have same shape as targets
        lower_bounds_flat = lower_bounds.flatten()
        upper_bounds_flat = upper_bounds.flatten()
        in_interval = (targets_flat >= lower_bounds_flat) & (targets_flat <= upper_bounds_flat)
    
    abs_error = np.abs(targets_flat - predictions_flat)
    width = upper_bounds_flat - lower_bounds_flat
    
    # Prepare data for DataFrame
    data = {
        'target': targets_flat,
        'prediction': predictions_flat,
        'abs_error': abs_error,
        'lower_bound': lower_bounds_flat,
        'upper_bound': upper_bounds_flat,
        'width': width,
        'in_interval': in_interval
    }
    
    # Add scores to data
    if config['scoring_functions']['symmetric']:
        # Make sure scores is flattened
        data['score'] = scores.flatten()
    else:
        try:
            # Check if scores has the right shape for asymmetric intervals
            if scores.shape[1] == 2:
                data['lower_score'] = scores[:, 0].flatten()
                data['upper_score'] = scores[:, 1].flatten()
            else:
                logging.warning(f"Unexpected shape for asymmetric scores: {scores.shape}. Skipping score data in results.")
        except (IndexError, AttributeError) as e:
            logging.warning(f"Error processing scores: {e}. Skipping score data in results.")
    
    # Check array lengths before creating DataFrame
    lengths = {k: len(v) for k, v in data.items()}
    if len(set(lengths.values())) > 1:
        logging.warning(f"Arrays have inconsistent lengths: {lengths}")
        
        # Fix length issues by truncating to shortest array
        min_length = min(lengths.values())
        for k in data:
            if len(data[k]) > min_length:
                logging.warning(f"Truncating {k} from {len(data[k])} to {min_length}")
                data[k] = data[k][:min_length]
    
    # Create DataFrame
    results_df = pd.DataFrame(data)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'prediction_intervals.csv'), index=False)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.yml'), 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    return results_df 