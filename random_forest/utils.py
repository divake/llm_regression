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

def generate_friedman_data(output_dir, n_samples=1000, seed=42):
    """
    Generate synthetic Friedman #2 dataset
    
    Args:
        output_dir: Directory to save generated data
        n_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test) - numpy arrays
    """
    from sklearn.datasets import make_friedman2
    from sklearn.model_selection import train_test_split
    
    logging.info(f"Generating synthetic Friedman #2 dataset with {n_samples} samples")
    
    # Set random seed
    np.random.seed(seed)
    
    # Generate dataset
    X, y = make_friedman2(n_samples=n_samples, noise=0.1, random_state=seed)
    
    # Define feature names
    feature_names = [f'x{i+1}' for i in range(X.shape[1])]
    
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=seed)
    
    # Create dataframes
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    
    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df['target'] = y_val
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    
    # Save to files
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation_data.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
    
    logging.info(f"Saved Friedman dataset to {output_dir}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_data(config, data_path=None):
    """
    Load and prepare data for conformal prediction
    
    Args:
        config: Configuration dictionary
        data_path: Path to dataset (optional)
        
    Returns:
        data: Dictionary with train, calibration, and test data
    """
    if data_path:
        logging.info(f"Loading data from {data_path}")
        # Load data from CSV or similar
        df = pd.read_csv(data_path)
        
        # Assuming last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Get feature names
        feature_names = df.columns[:-1].tolist()
    else:
        # For this example, use Friedman 2 dataset
        friedman_dir = config.get('paths', {}).get('friedman_data_dir', 'friedman2_dataset')
        logging.info(f"Loading Friedman dataset from {friedman_dir}")
        
        # Check if dataset files exist
        train_path = os.path.join(friedman_dir, "train_data.csv")
        val_path = os.path.join(friedman_dir, "validation_data.csv")
        test_path = os.path.join(friedman_dir, "test_data.csv")
        
        # Generate synthetic data if files don't exist
        if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
            logging.warning("Friedman dataset files not found. Generating synthetic data...")
            generate_friedman_data(friedman_dir, n_samples=2000, seed=config['data']['seed'])
        
        # Load datasets
        train_df = pd.read_csv(os.path.join(friedman_dir, "train_data.csv"))
        val_df = pd.read_csv(os.path.join(friedman_dir, "validation_data.csv"))
        test_df = pd.read_csv(os.path.join(friedman_dir, "test_data.csv"))
        
        # Extract feature names from config or use defaults
        config_feature_names = config.get('data', {}).get('feature_names', None)
        
        # If feature_names is None in config, use all columns except 'target'
        if config_feature_names is None:
            feature_names = [col for col in train_df.columns if col != 'target']
            logging.info(f"Using feature names from dataset: {feature_names}")
        else:
            feature_names = config_feature_names
            logging.info(f"Using feature names from config: {feature_names}")
        
        # Extract features and targets
        X_train = train_df[feature_names].values
        y_train = train_df['target'].values
        X_val = val_df[feature_names].values
        y_val = val_df['target'].values
        X_test = test_df[feature_names].values
        y_test = test_df['target'].values
        
        # Combine train and validation for our purpose
        X = np.vstack([X_train, X_val, X_test])
        y = np.concatenate([y_train, y_val, y_test])
    
    # Check if any feature contains string values and parse them
    # This handles text features like "x1: 96.99, x2: 1700.53, x3: 0.9915, x4: 9.24"
    for col_idx in range(X.shape[1]):
        if X.dtype == object or isinstance(X[0, col_idx], str):
            sample_val = X[0, col_idx]
            if isinstance(sample_val, str) and ': ' in sample_val and ',' in sample_val:
                logging.info(f"Detected complex text feature at column {col_idx}: '{sample_val}'")
                
                # Check if this is a structured text feature containing multiple values
                # First, find all the feature keys in the first sample
                parts = sample_val.split(', ')
                if len(parts) > 1 and all(':' in p for p in parts):
                    # This appears to be a composite feature with multiple values
                    feature_keys = [p.split(':', 1)[0].strip() for p in parts]
                    logging.info(f"Extracting {len(feature_keys)} sub-features: {feature_keys}")
                    
                    # Create new columns for each sub-feature
                    new_columns = {key: [] for key in feature_keys}
                    
                    # Process all rows
                    for row_idx in range(X.shape[0]):
                        if isinstance(X[row_idx, col_idx], str):
                            # Parse each part of the text feature
                            row_parts = X[row_idx, col_idx].split(', ')
                            for part in row_parts:
                                if ':' in part:
                                    try:
                                        key, value_str = part.split(':', 1)
                                        key = key.strip()
                                        if key in feature_keys:
                                            new_columns[key].append(float(value_str.strip()))
                                    except (ValueError, IndexError):
                                        # If parsing fails for this part, use 0
                                        if key in feature_keys:
                                            new_columns[key].append(0.0)
                            
                            # Check if any keys are missing in this row
                            for key in feature_keys:
                                if len(new_columns[key]) <= row_idx:
                                    new_columns[key].append(0.0)
                        else:
                            # Handle non-string values
                            for key in feature_keys:
                                new_columns[key].append(0.0)
                    
                    # Replace the original column with the first extracted feature
                    first_key = feature_keys[0]
                    X[:, col_idx] = np.array(new_columns[first_key])
                    
                    # Add additional columns for the remaining features if there are any
                    if len(feature_keys) > 1:
                        for i, key in enumerate(feature_keys[1:], 1):
                            feature_name = f"{feature_names[col_idx]}_{key}"
                            logging.info(f"Adding new feature column: {feature_name}")
                            new_col = np.array(new_columns[key]).reshape(-1, 1)
                            X = np.hstack((X, new_col))
                            # Extend feature_names list to match new X matrix
                            feature_names.append(feature_name)
                else:
                    # Simple string feature, use basic parsing
                    logging.info(f"Converting text feature at column {col_idx} to numeric values")
                    numeric_col = []
                    for row_idx in range(X.shape[0]):
                        if isinstance(X[row_idx, col_idx], str) and ':' in X[row_idx, col_idx]:
                            # Extract numeric value for the first mentioned feature
                            try:
                                # Parse the first numeric value from a string like "x1: 96.99, x2: 1700.53..."
                                value = float(X[row_idx, col_idx].split(':', 1)[1].split(',')[0].strip())
                                numeric_col.append(value)
                            except (ValueError, IndexError):
                                # If parsing fails, use 0 as fallback
                                numeric_col.append(0.0)
                        else:
                            # If it's already numeric, or can be converted directly
                            try:
                                numeric_col.append(float(X[row_idx, col_idx]))
                            except (ValueError, TypeError):
                                numeric_col.append(0.0)
                    
                    # Replace the column with numeric values
                    X[:, col_idx] = np.array(numeric_col)
            else:
                # Simple string feature that doesn't match the pattern, try basic conversion
                logging.info(f"Converting basic text feature at column {col_idx} to numeric values")
                numeric_col = []
                for row_idx in range(X.shape[0]):
                    try:
                        numeric_col.append(float(X[row_idx, col_idx]))
                    except (ValueError, TypeError):
                        numeric_col.append(0.0)
                X[:, col_idx] = np.array(numeric_col)
    
    # Ensure X is a float array
    X = X.astype(float)
    
    # Log about the features
    logging.info(f"Features after processing: {X.shape[1]} columns, feature names: {feature_names}")
    
    # Split data according to config ratios
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / total_ratio
    val_ratio = val_ratio / total_ratio
    test_ratio = test_ratio / total_ratio
    
    # Calculate split indices
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Shuffle data
    np.random.seed(config['data']['seed'])
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    # Normalize features if specified in config
    if config['data'].get('normalize', True):
        X = preprocess_features(X, train_end, config)
    
    # Split data
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logging.info(f"Data split: Train {X_train.shape[0]}, Calibration {X_val.shape[0]}, Test {X_test.shape[0]}")
    
    return {
        'train': (X_train, y_train),
        'calibration': (X_val, y_val),
        'test': (X_test, y_test),
        'feature_names': feature_names  # This list may have been extended with new features
    }

def preprocess_features(X, train_end=None, config=None):
    """
    Preprocess features: normalize and optionally add interaction terms
    
    Args:
        X: Input features
        train_end: Index where training data ends (for normalization)
        config: Configuration dictionary
        
    Returns:
        X_processed: Processed features
    """
    from sklearn.preprocessing import StandardScaler
    
    if train_end is None:
        train_end = X.shape[0]
    
    # Standardize features
    scaler = StandardScaler()
    X_train = X[:train_end]
    X_train_scaled = scaler.fit_transform(X_train)
    X_scaled = scaler.transform(X)
    
    # Check if we should add interaction terms
    add_interactions = False
    if config is not None:
        add_interactions = config.get('data', {}).get('add_interactions', False)
    
    if add_interactions:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
        X_with_interactions = poly.fit_transform(X_scaled)
        return X_with_interactions
    else:
        return X_scaled

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

def evaluate_prediction_intervals(targets, predictions, lower_bounds, upper_bounds, output_dir=None):
    """
    Evaluate prediction intervals
    
    Args:
        targets: True target values
        predictions: Model point predictions
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
        output_dir: Directory to save plots (optional)
        
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
    
    # Check if shapes match and output_dir is provided
    if len(widths_flat) == len(abs_errors_flat) and output_dir is not None:
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
    elif len(widths_flat) != len(abs_errors_flat):
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
    # Ensure predictions have shape [n_samples, 1]
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1)
    
    # Ensure scores have appropriate shape
    if scores.ndim == 1:
        scores = scores.unsqueeze(1)
    
    # Log shapes for debugging
    logging.debug(f"Predictions shape: {predictions.shape}, Scores shape: {scores.shape}")
    
    if symmetric:
        # For symmetric intervals, use same width for both sides
        if scores.shape[1] != 1:
            logging.warning(f"Expected scores to have shape [n_samples, 1] for symmetric intervals, "
                           f"but got {scores.shape}. Using first column.")
            scores = scores[:, 0].unsqueeze(1)
        
        lower_bounds = predictions - tau * scores
        upper_bounds = predictions + tau * scores
    else:
        # For asymmetric intervals, use separate scores for lower and upper
        if scores.shape[1] != 2:
            logging.warning(f"Asymmetric intervals require scores with shape [n_samples, 2], "
                           f"but got {scores.shape}. Attempting to reshape...")
            if scores.shape[1] > 2:
                # Take first two columns
                scores = scores[:, :2]
            else:
                # Duplicate single column for both lower and upper
                scores = scores.repeat(1, 2)
        
        lower_scores, upper_scores = scores[:, 0].unsqueeze(1), scores[:, 1].unsqueeze(1)
        
        if isinstance(tau, (list, tuple)):
            lower_tau, upper_tau = tau
        else:
            logging.warning(f"Expected tau to be a tuple for asymmetric intervals, but got {type(tau)}. "
                           "Using same value for both bounds.")
            lower_tau = upper_tau = tau
        
        lower_bounds = predictions - lower_tau * lower_scores
        upper_bounds = predictions + upper_tau * upper_scores
    
    # Final shape check - ensure output shapes match predictions
    if lower_bounds.shape != predictions.shape:
        logging.warning(f"Shape mismatch: lower_bounds {lower_bounds.shape} vs predictions {predictions.shape}")
        
    if upper_bounds.shape != predictions.shape:
        logging.warning(f"Shape mismatch: upper_bounds {upper_bounds.shape} vs predictions {predictions.shape}")
    
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