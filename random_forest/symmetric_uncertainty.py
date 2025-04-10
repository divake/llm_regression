import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from regression_scoring_function import RegressionScoringMLP
from regression_trainer import RegressionUncertaintyTrainer
from regression_metrics import AverageMeter, evaluate_prediction_intervals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Constants
OUTPUT_DIR = "friedman2_rf_regression"
FRIEDMAN_OUTPUT_DIR = "friedman2_dataset"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "uncertainty_results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")

# Make sure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Training parameters
CONFIG = {
    'learning_rate': 1e-3,
    'target_coverage': 0.9,  # 90% target coverage
    'coverage_weight': 1.0,  # Weight for coverage loss
    'width_weight': 0.1,    # Weight for width loss (lower to favor coverage)
    'grad_clip': 1.0,
    'schedule_factor': 0.5,
    'schedule_patience': 5,
    'early_stop_patience': 10
}

def load_data():
    """Load dataset and RF model predictions"""
    logging.info("Loading Random Forest model and predictions...")
    
    # Load the Random Forest model
    rf_model_path = os.path.join(OUTPUT_DIR, "random_forest_model.joblib")
    rf_model = joblib.load(rf_model_path)
    
    # Load original datasets with the same split as used for training the RF model
    train_df = pd.read_csv(os.path.join(FRIEDMAN_OUTPUT_DIR, "train_data.csv"))
    val_df = pd.read_csv(os.path.join(FRIEDMAN_OUTPUT_DIR, "validation_data.csv"))
    test_df = pd.read_csv(os.path.join(FRIEDMAN_OUTPUT_DIR, "test_data.csv"))
    
    # Feature names
    feature_names = ['x1', 'x2', 'x3', 'x4']
    
    # Extract features and targets from each dataset
    X_train = train_df[feature_names].values
    y_train = train_df['target'].values
    
    X_val = val_df[feature_names].values  # Validation data will be used for calibration
    y_val = val_df['target'].values
    
    X_test = test_df[feature_names].values
    y_test = test_df['target'].values
    
    # Generate predictions for each dataset
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)
    y_test_pred = rf_model.predict(X_test)
    
    # Calculate absolute errors for each dataset
    train_errors = np.abs(y_train - y_train_pred).reshape(-1, 1)
    val_errors = np.abs(y_val - y_val_pred).reshape(-1, 1)
    test_errors = np.abs(y_test - y_test_pred).reshape(-1, 1)
    
    logging.info(f"Loaded train: {len(train_df)}, validation: {len(val_df)}, test: {len(test_df)} data points")
    
    # Create data dictionaries
    train_data = {
        'features': X_train,
        'targets': y_train,
        'predictions': y_train_pred,
        'errors': train_errors
    }
    
    val_data = {
        'features': X_val,
        'targets': y_val,
        'predictions': y_val_pred,
        'errors': val_errors
    }
    
    test_data = {
        'features': X_test,
        'targets': y_test,
        'predictions': y_test_pred,
        'errors': test_errors
    }
    
    return rf_model, train_data, val_data, test_data, feature_names

def create_torch_datasets(train_data, val_data, test_data):
    """Create PyTorch datasets for training the scoring function using the original data splits"""
    logging.info("Creating training, calibration, and test datasets...")
    
    # Create PyTorch TensorDatasets
    train_dataset = TensorDataset(
        torch.tensor(train_data['errors'], dtype=torch.float32),
        torch.tensor(train_data['targets'], dtype=torch.float32)
    )
    
    # Use validation dataset as calibration dataset
    cal_dataset = TensorDataset(
        torch.tensor(val_data['errors'], dtype=torch.float32),
        torch.tensor(val_data['targets'], dtype=torch.float32)
    )
    
    test_dataset = TensorDataset(
        torch.tensor(test_data['errors'], dtype=torch.float32),
        torch.tensor(test_data['targets'], dtype=torch.float32)
    )
    
    # Create DataLoaders
    batch_size = 64
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    cal_loader = DataLoader(
        cal_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Store original data for later evaluation
    original_data = {
        'train': (train_data['features'], train_data['targets'], train_data['predictions']),
        'cal': (val_data['features'], val_data['targets'], val_data['predictions']),
        'test': (test_data['features'], test_data['targets'], test_data['predictions'])
    }
    
    logging.info(f"Created datasets - Train: {len(train_dataset)}, Cal: {len(cal_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, cal_loader, test_loader, original_data

def create_rf_wrapper(rf_model):
    """Create a PyTorch wrapper for the Random Forest model"""
    class RFWrapper(torch.nn.Module):
        def __init__(self, rf_model):
            super().__init__()
            self.rf_model = rf_model
            
        def forward(self, x):
            """Convert tensor to numpy for prediction with sklearn model"""
            x_np = x.detach().cpu().numpy()
            preds = self.rf_model.predict(x_np)
            return torch.tensor(preds, dtype=torch.float32).unsqueeze(1)
    
    return RFWrapper(rf_model)

def train_scoring_function(rf_model, train_loader, cal_loader, test_loader):
    """Train the scoring function for uncertainty estimation"""
    logging.info("Training the uncertainty scoring function...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create scoring function
    scoring_fn = RegressionScoringMLP(
        input_dim=1,          # Input is absolute error
        hidden_dims=[64, 32], # Hidden layer dimensions
        output_dim=1,         # Output is uncertainty width
        dropout_rate=0.2      # Dropout for regularization
    ).to(device)
    
    logging.info(f"Created scoring function: {scoring_fn}")
    
    # Create RF model wrapper
    rf_wrapper = create_rf_wrapper(rf_model)
    
    # Create trainer
    trainer = RegressionUncertaintyTrainer(
        rf_model=rf_wrapper,
        scoring_fn=scoring_fn,
        train_loader=train_loader,
        cal_loader=cal_loader,
        test_loader=test_loader,
        device=device,
        config=CONFIG
    )
    
    # Train for specified number of epochs
    num_epochs = 50
    logging.info(f"Starting training for {num_epochs} epochs")
    
    history, best_model_state = trainer.train(
        num_epochs=num_epochs,
        output_dir=RESULTS_DIR
    )
    
    return trainer, scoring_fn, history

def evaluate_on_original_data(trainer, original_data):
    """Evaluate the trained model on original data with features"""
    logging.info("Evaluating on original test data...")
    
    # Unpack test data
    test_features, test_targets, test_predictions = original_data['test']
    
    # Evaluate and create detailed plots
    metrics = trainer.evaluate_on_original_predictions(
        feature_data=test_features,
        target_data=test_targets,
        prediction_data=test_predictions,
        output_dir=RESULTS_DIR
    )
    
    logging.info(f"Final test metrics:")
    logging.info(f"  Coverage: {metrics['coverage']:.4f} (Target: {CONFIG['target_coverage']:.4f})")
    logging.info(f"  Average width: {metrics['avg_width']:.4f}")
    logging.info(f"  Interval score: {metrics['interval_score']:.4f}")
    logging.info(f"  Efficiency: {metrics['efficiency']:.4f}")
    logging.info(f"  RMSE: {metrics['rmse']:.4f}")
    logging.info(f"  Width-error correlation: {metrics['width_error_corr']:.4f}")
    
    return metrics

def analyze_results():
    """Analyze saved prediction intervals"""
    logging.info("Analyzing prediction intervals...")
    
    # Load prediction intervals
    intervals_path = os.path.join(RESULTS_DIR, "prediction_intervals.csv")
    if not os.path.exists(intervals_path):
        logging.warning(f"Prediction intervals file not found: {intervals_path}")
        return
    
    results_df = pd.read_csv(intervals_path)
    
    # Calculate overall statistics
    coverage = results_df['in_interval'].mean()
    avg_width = results_df['calibrated_width'].mean()
    efficiency = coverage / avg_width if avg_width > 0 else 0
    
    logging.info(f"Overall statistics:")
    logging.info(f"  Total samples: {len(results_df)}")
    logging.info(f"  Coverage: {coverage:.4f}")
    logging.info(f"  Average width: {avg_width:.4f}")
    logging.info(f"  Efficiency (coverage/width): {efficiency:.4f}")
    
    # Analyze feature relationships
    if 'feature_1' in results_df.columns:
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        feature_titles = ['x1', 'x2', 'x3', 'x4']
        
        # Bin features and analyze coverage by bins
        plt.figure(figsize=(15, 10))
        
        for i, (feature, title) in enumerate(zip(feature_names, feature_titles)):
            plt.subplot(2, 2, i+1)
            
            # Create 10 bins for the feature
            bins = np.linspace(results_df[feature].min(), results_df[feature].max(), 11)
            bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
            
            # Assign bin to each data point
            results_df['bin'] = pd.cut(results_df[feature], bins=bins, labels=bin_labels)
            
            # Calculate coverage and width by bin
            bin_stats = results_df.groupby('bin').agg({
                'in_interval': 'mean',
                'calibrated_width': 'mean'
            }).reset_index()
            
            # Plot coverage by bin
            plt.bar(range(len(bin_stats)), bin_stats['in_interval'], alpha=0.7)
            plt.axhline(y=CONFIG['target_coverage'], color='r', linestyle='--', 
                       label=f"Target ({CONFIG['target_coverage']:.2f})")
            
            plt.title(f'Coverage by {title} Range')
            plt.xlabel(f'{title} Bins')
            plt.ylabel('Coverage')
            plt.xticks(range(len(bin_stats)), bin_stats['bin'], rotation=90)
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'coverage_by_feature.png'), dpi=150)
        plt.close()
    
    # Plot the calibration curve
    target_coverages = np.linspace(0.5, 1.0, 11)
    empirical_coverages = []
    
    for target in target_coverages:
        # Scale widths to achieve different target coverages
        scale_factor = target / CONFIG['target_coverage']
        lower_bounds = results_df['predicted'] - scale_factor * results_df['calibrated_width']
        upper_bounds = results_df['predicted'] + scale_factor * results_df['calibrated_width']
        in_interval = (results_df['actual'] >= lower_bounds) & (results_df['actual'] <= upper_bounds)
        empirical_coverages.append(in_interval.mean())
    
    plt.figure(figsize=(8, 8))
    plt.plot(target_coverages, empirical_coverages, 'bo-', label='Empirical')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal')
    plt.title('Calibration Curve')
    plt.xlabel('Target Coverage')
    plt.ylabel('Empirical Coverage')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'calibration_curve.png'), dpi=150)
    plt.close()
    
    logging.info("Analysis complete!")

def main():
    """Main execution function"""
    logging.info("Starting symmetric uncertainty estimation")
    
    # Load data
    rf_model, train_data, val_data, test_data, feature_names = load_data()
    
    # Create datasets
    train_loader, cal_loader, test_loader, original_data = create_torch_datasets(
        train_data, val_data, test_data
    )
    
    # Train scoring function
    trainer, scoring_fn, history = train_scoring_function(
        rf_model, train_loader, cal_loader, test_loader
    )
    
    # Evaluate on original data
    metrics = evaluate_on_original_data(trainer, original_data)
    
    # Analyze results 
    analyze_results()
    
    logging.info("Symmetric uncertainty estimation completed successfully!")
    return metrics

if __name__ == "__main__":
    main() 