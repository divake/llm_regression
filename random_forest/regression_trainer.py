import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from regression_metrics import (
    AverageMeter, 
    compute_prediction_intervals, 
    compute_coverage,
    compute_average_width,
    compute_conformal_calibration,
    evaluate_prediction_intervals
)

class RegressionUncertaintyTrainer:
    def __init__(self, rf_model, scoring_fn, train_loader, cal_loader, 
                 test_loader, device, config=None):
        """
        Initialize the trainer for regression uncertainty estimation
        
        Args:
            rf_model: Random Forest regression model
            scoring_fn: Scoring function that predicts uncertainty width
            train_loader: Training data loader
            cal_loader: Calibration data loader
            test_loader: Test data loader
            device: Device to run on
            config: Configuration dictionary containing training parameters
        """
        self.rf_model = rf_model  # This should be an RF model wrapper
        self.scoring_fn = scoring_fn
        self.train_loader = train_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        self.device = device
        
        # Default configuration
        default_config = {
            'learning_rate': 1e-3,
            'target_coverage': 0.9,
            'coverage_weight': 1.0,
            'width_weight': 0.1,
            'grad_clip': 1.0,
            'schedule_factor': 0.5,
            'schedule_patience': 5
        }
        
        # Use provided config or default
        self.config = config if config is not None else default_config
        
        # Extract key parameters
        self.learning_rate = self.config.get('learning_rate', default_config['learning_rate'])
        self.target_coverage = self.config.get('target_coverage', default_config['target_coverage'])
        self.coverage_weight = self.config.get('coverage_weight', default_config['coverage_weight'])
        self.width_weight = self.config.get('width_weight', default_config['width_weight'])
        self.grad_clip = self.config.get('grad_clip', default_config['grad_clip'])
        
    def train(self, num_epochs, output_dir=None):
        """
        Train the scoring function for the specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
            output_dir: Directory to save results and plots
        
        Returns:
            history: Dictionary containing training history
            best_model_state: State dict of the best model
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Setup optimizer
        optimizer = optim.Adam(self.scoring_fn.parameters(), lr=self.learning_rate)
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.config.get('schedule_factor', 0.5),
            patience=self.config.get('schedule_patience', 5),
            verbose=True
        )
        
        # Initialize tracking variables
        best_model_state = None
        best_score = float('inf')
        early_stop_counter = 0
        early_stop_patience = self.config.get('early_stop_patience', 10)
        
        # Initialize history
        history = {
            'epoch': [],
            'train_loss': [],
            'train_coverage': [],
            'train_width': [],
            'val_coverage': [],
            'val_width': [],
            'val_interval_score': [],
            'calibration_factor': [],
            'learning_rate': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Train one epoch
            train_loss, train_coverage, train_width = self._train_epoch(optimizer)
            
            # Compute calibration factor on calibration set
            cal_factor = self._compute_calibration_factor()
            
            # Evaluate on validation set
            val_metrics = self._evaluate(self.test_loader, cal_factor)
            val_coverage = val_metrics['coverage']
            val_width = val_metrics['avg_width']
            val_interval_score = val_metrics['interval_score']
            
            # Log progress
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Train Coverage: {train_coverage:.4f}, "
                        f"Train Width: {train_width:.4f}, "
                        f"Val Coverage: {val_coverage:.4f}, "
                        f"Val Width: {val_width:.4f}, "
                        f"Val Interval Score: {val_interval_score:.4f}, "
                        f"Calibration Factor: {cal_factor:.4f}, "
                        f"LR: {lr:.6f}")
            
            # Update history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_coverage'].append(train_coverage)
            history['train_width'].append(train_width)
            history['val_coverage'].append(val_coverage)
            history['val_width'].append(val_width)
            history['val_interval_score'].append(val_interval_score)
            history['calibration_factor'].append(cal_factor)
            history['learning_rate'].append(lr)
            
            # Update scheduler
            scheduler.step(val_interval_score)
            
            # Check for best model (based on validation interval score)
            if val_interval_score < best_score:
                best_score = val_interval_score
                best_model_state = self.scoring_fn.state_dict()
                early_stop_counter = 0
                
                # Save best model
                if output_dir is not None:
                    torch.save(best_model_state, os.path.join(output_dir, 'best_scoring_model.pt'))
                    logging.info(f"Saved best model with interval score: {best_score:.4f}")
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= early_stop_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Plot training progress
            if output_dir is not None and (epoch + 1) % 5 == 0:
                self._plot_training_progress(history, output_dir)
                
        # Restore best model
        if best_model_state is not None:
            self.scoring_fn.load_state_dict(best_model_state)
        
        # Final evaluation and plots
        if output_dir is not None:
            self._plot_training_progress(history, output_dir)
            self._plot_prediction_intervals(self.test_loader, cal_factor, output_dir)
            self._save_prediction_results(self.test_loader, cal_factor, output_dir)
        
        return history, best_model_state
            
    def _train_epoch(self, optimizer):
        """
        Train for one epoch
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            loss_avg: Average loss for the epoch
            coverage_avg: Average coverage for the epoch
            width_avg: Average width for the epoch
        """
        self.scoring_fn.train()
        
        # Initialize meters
        loss_meter = AverageMeter()
        coverage_meter = AverageMeter()
        width_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader)
        for abs_errors, targets in pbar:
            abs_errors = abs_errors.to(self.device)
            targets = targets.to(self.device)
            batch_size = abs_errors.size(0)
            
            # Forward pass (predicted widths)
            widths = self.scoring_fn(abs_errors)
            widths = widths.squeeze()
            
            # Normalize abs_errors for numerical stability
            normalized_errors = abs_errors.squeeze()
            
            # Coverage loss: penalize if width < error (uncovered point)
            # This encourages widths to be large enough to cover the errors
            coverage_loss = torch.relu(normalized_errors - widths).mean()
            
            # Width loss: penalize large widths
            width_loss = widths.mean()
            
            # Combined loss
            loss = (
                self.coverage_weight * coverage_loss + 
                self.width_weight * width_loss
            )
            
            # Add L2 regularization if available
            if hasattr(self.scoring_fn, 'l2_reg'):
                loss = loss + self.scoring_fn.l2_reg
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.scoring_fn.parameters(), self.grad_clip)
            
            optimizer.step()
            
            # Compute coverage (for monitoring)
            with torch.no_grad():
                # Simulate coverage with current widths
                covered = (widths >= normalized_errors).float().mean()
            
            # Update meters
            loss_meter.update(loss.item(), batch_size)
            coverage_meter.update(covered.item(), batch_size)
            width_meter.update(widths.mean().item(), batch_size)
            
            # Update progress bar
            pbar.set_description(
                f"Train Loss: {loss_meter.avg:.4f}, "
                f"Coverage: {coverage_meter.avg:.4f}, "
                f"Width: {width_meter.avg:.4f}"
            )
        
        return loss_meter.avg, coverage_meter.avg, width_meter.avg
    
    def _compute_calibration_factor(self):
        """
        Compute calibration factor on calibration set to ensure target coverage
        
        Returns:
            calibration_factor: Factor to scale prediction intervals
        """
        self.scoring_fn.eval()
        
        all_errors = []
        all_widths = []
        
        # Collect errors and widths on calibration set
        with torch.no_grad():
            for abs_errors, _ in self.cal_loader:
                abs_errors = abs_errors.to(self.device)
                
                # Predict widths
                widths = self.scoring_fn(abs_errors).squeeze()
                
                all_errors.extend(abs_errors.squeeze().cpu().numpy())
                all_widths.extend(widths.cpu().numpy())
        
        # Convert to numpy arrays
        all_errors = np.array(all_errors)
        all_widths = np.array(all_widths)
        
        # Calculate conformity scores as actual/predicted
        conformal_scores = all_errors / (all_widths + 1e-8)
        
        # Sort scores
        sorted_scores = np.sort(conformal_scores)
        
        # Find quantile for target coverage
        n = len(sorted_scores)
        q_index = int(np.ceil((n + 1) * self.target_coverage)) - 1
        q_index = min(max(0, q_index), n - 1)  # Ensure index is valid
        
        # Get calibration factor
        calibration_factor = sorted_scores[q_index]
        
        return calibration_factor
    
    def _evaluate(self, data_loader, calibration_factor):
        """
        Evaluate on a dataset
        
        Args:
            data_loader: DataLoader for evaluation
            calibration_factor: Factor to scale prediction intervals
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.scoring_fn.eval()
        
        all_targets = []
        all_errors = []
        all_widths = []
        
        # Collect predictions on test set
        with torch.no_grad():
            for abs_errors, targets in data_loader:
                abs_errors = abs_errors.to(self.device)
                targets = targets.to(self.device)
                
                # Predict widths
                widths = self.scoring_fn(abs_errors).squeeze()
                
                # Apply calibration factor
                calibrated_widths = calibration_factor * widths
                
                all_targets.extend(targets.cpu().numpy())
                all_errors.extend(abs_errors.squeeze().cpu().numpy())
                all_widths.extend(calibrated_widths.cpu().numpy())
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_errors = np.array(all_errors)
        all_widths = np.array(all_widths)
        
        # Compute predictions from errors (target - error relationship is not available here)
        # Since we're working with absolute errors, we need to infer predictions
        rf_predictions = all_targets - all_errors  # This is approximate
        
        # Compute lower and upper bounds
        lower_bounds = rf_predictions - all_widths
        upper_bounds = rf_predictions + all_widths
        
        # Evaluate metrics
        metrics = evaluate_prediction_intervals(
            all_targets, rf_predictions, lower_bounds, upper_bounds
        )
        
        return metrics
    
    def evaluate_on_original_predictions(self, feature_data, target_data, prediction_data, output_dir=None):
        """
        Evaluate on original RF predictions with features
        
        Args:
            feature_data: Features used for RF predictions
            target_data: True target values
            prediction_data: RF model predictions
            output_dir: Directory to save results and plots
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.scoring_fn.eval()
        
        # Compute absolute errors
        abs_errors = np.abs(target_data - prediction_data).reshape(-1, 1)
        abs_errors_tensor = torch.tensor(abs_errors, dtype=torch.float32, device=self.device)
        
        # Compute calibration factor
        calibration_factor = self._compute_calibration_factor()
        
        # Predict widths
        with torch.no_grad():
            widths = self.scoring_fn(abs_errors_tensor).squeeze().cpu().numpy()
            
        # Apply calibration factor
        calibrated_widths = calibration_factor * widths
        
        # Compute intervals
        lower_bounds = prediction_data - calibrated_widths
        upper_bounds = prediction_data + calibrated_widths
        
        # Evaluate metrics
        metrics = evaluate_prediction_intervals(
            target_data, prediction_data, lower_bounds, upper_bounds
        )
        
        # Save results if output_dir provided
        if output_dir is not None:
            # Create results DataFrame
            results_df = pd.DataFrame({
                'feature_1': feature_data[:, 0],
                'feature_2': feature_data[:, 1],
                'feature_3': feature_data[:, 2],
                'feature_4': feature_data[:, 3],
                'actual': target_data,
                'predicted': prediction_data,
                'abs_error': abs_errors.squeeze(),
                'predicted_width': widths,
                'calibrated_width': calibrated_widths,
                'lower_bound': lower_bounds,
                'upper_bound': upper_bounds,
                'in_interval': (target_data >= lower_bounds) & (target_data <= upper_bounds)
            })
            
            # Save results
            results_df.to_csv(os.path.join(output_dir, 'prediction_intervals.csv'), index=False)
            
            # Plot results
            self._plot_detailed_results(results_df, metrics, output_dir)
        
        return metrics
    
    def _plot_training_progress(self, history, output_dir):
        """
        Plot training progress
        
        Args:
            history: Dictionary containing training history
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
        plt.axhline(y=self.target_coverage, color='r', linestyle='--', label=f'Target ({self.target_coverage})')
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
    
    def _plot_prediction_intervals(self, data_loader, calibration_factor, output_dir):
        """
        Plot prediction intervals
        
        Args:
            data_loader: DataLoader for evaluation
            calibration_factor: Factor to scale prediction intervals
            output_dir: Directory to save plots
        """
        self.scoring_fn.eval()
        
        all_targets = []
        all_errors = []
        all_widths = []
        
        # Collect predictions on test set
        with torch.no_grad():
            for abs_errors, targets in data_loader:
                abs_errors = abs_errors.to(self.device)
                targets = targets.to(self.device)
                
                # Predict widths
                widths = self.scoring_fn(abs_errors).squeeze()
                
                # Apply calibration factor
                calibrated_widths = calibration_factor * widths
                
                all_targets.extend(targets.cpu().numpy())
                all_errors.extend(abs_errors.squeeze().cpu().numpy())
                all_widths.extend(calibrated_widths.cpu().numpy())
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_errors = np.array(all_errors)
        all_widths = np.array(all_widths)
        
        # Compute predictions from errors
        # Since we're working with absolute errors, we need to infer predictions
        rf_predictions = all_targets - all_errors  # This is approximate
        
        # Compute lower and upper bounds
        lower_bounds = rf_predictions - all_widths
        upper_bounds = rf_predictions + all_widths
        
        # Calculate in_interval
        in_interval = (all_targets >= lower_bounds) & (all_targets <= upper_bounds)
        coverage = np.mean(in_interval)
        
        # Plot prediction intervals
        plt.figure(figsize=(12, 8))
        
        # Sort by prediction for better visualization
        idx = np.argsort(rf_predictions)
        targets_sorted = all_targets[idx]
        preds_sorted = rf_predictions[idx]
        lower_sorted = lower_bounds[idx]
        upper_sorted = upper_bounds[idx]
        in_interval_sorted = in_interval[idx]
        
        # Take a subset of samples for clarity
        n_samples = min(500, len(idx))
        step = len(idx) // n_samples
        subset_idx = np.arange(0, len(idx), step)[:n_samples]
        
        # Plot intervals
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
            label='RF Predictions'
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
        
        plt.title(f'Prediction Intervals (Coverage: {coverage:.4f}, Target: {self.target_coverage:.4f})')
        plt.xlabel('Sample Index (sorted by prediction)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_intervals.png'), dpi=150)
        plt.close()
        
        # Plot the distribution of widths
        plt.figure(figsize=(10, 6))
        plt.hist(all_widths, bins=50, alpha=0.7)
        plt.title('Distribution of Interval Widths')
        plt.xlabel('Width')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'width_distribution.png'), dpi=150)
        plt.close()
        
        # Plot width vs error relationship
        plt.figure(figsize=(10, 6))
        plt.scatter(all_errors, all_widths, alpha=0.5, s=10)
        
        # Add trend line
        z = np.polyfit(all_errors, all_widths, 1)
        p = np.poly1d(z)
        plt.plot(np.sort(all_errors), p(np.sort(all_errors)), "r--", linewidth=2)
        
        # Compute correlation
        correlation = np.corrcoef(all_errors, all_widths)[0, 1]
        
        plt.title(f'Width vs Error Relationship (Correlation: {correlation:.4f})')
        plt.xlabel('Absolute Error')
        plt.ylabel('Predicted Width')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'width_vs_error.png'), dpi=150)
        plt.close()
    
    def _save_prediction_results(self, data_loader, calibration_factor, output_dir):
        """
        Save prediction results
        
        Args:
            data_loader: DataLoader for evaluation
            calibration_factor: Factor to scale prediction intervals
            output_dir: Directory to save results
        """
        self.scoring_fn.eval()
        
        all_targets = []
        all_errors = []
        all_widths = []
        all_calibrated_widths = []
        
        # Collect predictions on test set
        with torch.no_grad():
            for abs_errors, targets in data_loader:
                abs_errors = abs_errors.to(self.device)
                targets = targets.to(self.device)
                
                # Predict widths
                widths = self.scoring_fn(abs_errors).squeeze()
                
                # Apply calibration factor
                calibrated_widths = calibration_factor * widths
                
                all_targets.extend(targets.cpu().numpy())
                all_errors.extend(abs_errors.squeeze().cpu().numpy())
                all_widths.extend(widths.cpu().numpy())
                all_calibrated_widths.extend(calibrated_widths.cpu().numpy())
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_errors = np.array(all_errors)
        all_widths = np.array(all_widths)
        all_calibrated_widths = np.array(all_calibrated_widths)
        
        # Compute predictions from errors
        rf_predictions = all_targets - all_errors  # This is approximate
        
        # Compute lower and upper bounds
        lower_bounds = rf_predictions - all_calibrated_widths
        upper_bounds = rf_predictions + all_calibrated_widths
        
        # Calculate in_interval
        in_interval = (all_targets >= lower_bounds) & (all_targets <= upper_bounds)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'actual': all_targets,
            'predicted': rf_predictions,
            'abs_error': all_errors,
            'predicted_width': all_widths,
            'calibrated_width': all_calibrated_widths,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'in_interval': in_interval
        })
        
        # Save results
        results_df.to_csv(os.path.join(output_dir, 'test_prediction_intervals.csv'), index=False)
        
    def _plot_detailed_results(self, results_df, metrics, output_dir):
        """
        Plot detailed results
        
        Args:
            results_df: DataFrame containing prediction results
            metrics: Dictionary of evaluation metrics
            output_dir: Directory to save plots
        """
        # Create a figure for detailed results
        plt.figure(figsize=(15, 12))
        
        # Actual vs Predicted
        plt.subplot(2, 2, 1)
        plt.scatter(
            results_df['predicted'], 
            results_df['actual'], 
            c=results_df['in_interval'].map({True: 'g', False: 'r'}),
            alpha=0.5,
            s=15
        )
        min_val = min(results_df['actual'].min(), results_df['predicted'].min())
        max_val = max(results_df['actual'].max(), results_df['predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.grid(True)
        
        # Error vs Width
        plt.subplot(2, 2, 2)
        plt.scatter(
            results_df['abs_error'],
            results_df['calibrated_width'],
            c=results_df['in_interval'].map({True: 'g', False: 'r'}),
            alpha=0.5,
            s=15
        )
        max_x = max(results_df['abs_error'].max(), results_df['calibrated_width'].max()) * 1.1
        plt.plot([0, max_x], [0, max_x], 'k--')
        plt.title('Error vs Predicted Width')
        plt.xlabel('Absolute Error')
        plt.ylabel('Calibrated Width')
        plt.grid(True)
        
        # Width Distribution
        plt.subplot(2, 2, 3)
        plt.hist(results_df['calibrated_width'], bins=50, alpha=0.7)
        plt.axvline(x=results_df['calibrated_width'].mean(), color='r', linestyle='--', 
                    label=f'Mean: {results_df["calibrated_width"].mean():.2f}')
        plt.title('Distribution of Interval Widths')
        plt.xlabel('Calibrated Width')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        # Error Distribution
        plt.subplot(2, 2, 4)
        plt.hist(results_df['abs_error'], bins=50, alpha=0.7)
        plt.axvline(x=results_df['abs_error'].mean(), color='r', linestyle='--', 
                    label=f'Mean: {results_df["abs_error"].mean():.2f}')
        plt.title('Distribution of Absolute Errors')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        # Add metrics as text
        plt.figtext(0.5, 0.01, 
                    f"Coverage: {metrics['coverage']:.4f} (Target: {self.target_coverage:.4f}) | "
                    f"Avg Width: {metrics['avg_width']:.4f} | "
                    f"Interval Score: {metrics['interval_score']:.4f} | "
                    f"Efficiency: {metrics['efficiency']:.4f}",
                    ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'detailed_results.png'), dpi=150)
        plt.close()
        
        # Create feature importance plot
        if 'feature_1' in results_df.columns:
            # Plot errors across feature ranges
            plt.figure(figsize=(15, 10))
            
            features = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
            titles = ['x1', 'x2', 'x3', 'x4']
            
            for i, (feature, title) in enumerate(zip(features, titles)):
                plt.subplot(2, 2, i+1)
                plt.scatter(
                    results_df[feature],
                    results_df['abs_error'],
                    c=results_df['in_interval'].map({True: 'g', False: 'r'}),
                    alpha=0.5,
                    s=10
                )
                plt.title(f'Error vs {title}')
                plt.xlabel(title)
                plt.ylabel('Absolute Error')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'error_vs_features.png'), dpi=150)
            plt.close()
            
            # Plot widths across feature ranges
            plt.figure(figsize=(15, 10))
            
            for i, (feature, title) in enumerate(zip(features, titles)):
                plt.subplot(2, 2, i+1)
                plt.scatter(
                    results_df[feature],
                    results_df['calibrated_width'],
                    c=results_df['in_interval'].map({True: 'g', False: 'r'}),
                    alpha=0.5,
                    s=10
                )
                plt.title(f'Width vs {title}')
                plt.xlabel(title)
                plt.ylabel('Calibrated Width')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'width_vs_features.png'), dpi=150)
            plt.close() 