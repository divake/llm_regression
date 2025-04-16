import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import (
    AverageMeter, 
    evaluate_prediction_intervals,
    compute_prediction_intervals,
    plot_training_progress,
    plot_prediction_intervals,
    save_results
)

class ConformalRegressionTrainer:
    def __init__(self, base_model, scoring_fn, train_loader, cal_loader, 
                 test_loader, device, config):
        """
        Initialize the trainer for conformal regression uncertainty estimation
        
        Args:
            base_model: Base regression model
            scoring_fn: Scoring function that predicts nonconformity scores
            train_loader: Training data loader
            cal_loader: Calibration data loader
            test_loader: Test data loader
            device: Device to run on
            config: Configuration dictionary
        """
        self.base_model = base_model
        self.scoring_fn = scoring_fn
        self.train_loader = train_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Extract key parameters
        self.learning_rate = config['training']['learning_rate']
        self.target_coverage = config['training']['target_coverage']
        self.coverage_weight = config['training']['coverage_weight']
        self.width_weight = config['training']['width_weight']
        self.grad_clip = config['training']['grad_clip']
        self.symmetric = config['scoring_functions']['symmetric']
        
    def train(self, num_epochs=None, output_dir=None):
        """
        Train the scoring function for the specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train (default: from config)
            output_dir: Directory to save results and plots
        
        Returns:
            history: Dictionary containing training history
            best_model_state: State dict of the best model
        """
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
            
        if output_dir is None:
            output_dir = self.config['paths']['output_dir']
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup optimizer
        optimizer = optim.Adam(self.scoring_fn.parameters(), lr=self.learning_rate)
        
        # Setup learning rate scheduler with warmup
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
        
        # OneCycleLR scheduler with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate * 5,  # Peak at 5x base learning rate
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,  # Percentage of steps used for warmup
            div_factor=25,  # Initial LR = max_lr / div_factor
            final_div_factor=1e4,  # Final LR = max_lr / final_div_factor
            anneal_strategy='cos'  # Cosine annealing
        )
        
        # Early stopping scheduler based on validation metrics
        val_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # We want to maximize coverage
            factor=self.config['scheduler']['factor'],
            patience=self.config['scheduler']['patience'],
            verbose=True
        )
        
        # Initialize tracking variables
        best_model_state = None
        best_coverage = 0
        best_score = float('inf')
        early_stop_counter = 0
        early_stop_patience = self.config['scheduler']['early_stop_patience']
        min_coverage_threshold = self.target_coverage * 0.95  # 95% of target coverage must be met
        
        # Initialize history
        history = {
            'epoch': [],
            'train_loss': [],
            'train_coverage': [],
            'train_width': [],
            'coverage_loss': [],
            'width_loss': [],
            'val_coverage': [],
            'val_width': [],
            'val_interval_score': [],
            'calibration_factor': [],
            'learning_rate': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Train one epoch
            train_loss, train_coverage, train_width, coverage_loss, width_loss = self._train_epoch(
                optimizer, scheduler=scheduler
            )
            
            # Compute calibration factor on calibration set
            calibration_factor = self._compute_calibration_factor()
            
            # Evaluate on validation set
            val_metrics = self._evaluate(self.test_loader, calibration_factor)
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
                        f"LR: {lr:.6f}")
            
            # Update history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_coverage'].append(train_coverage)
            history['train_width'].append(train_width)
            history['coverage_loss'].append(coverage_loss)
            history['width_loss'].append(width_loss)
            history['val_coverage'].append(val_coverage)
            history['val_width'].append(val_width)
            history['val_interval_score'].append(val_interval_score)
            history['calibration_factor'].append(calibration_factor)
            history['learning_rate'].append(lr)
            
            # Update val scheduler based on coverage metric
            val_scheduler.step(val_coverage)
            
            # Check for best model (based on coverage first, then interval score)
            improved = False
            
            # First prioritize models that meet coverage threshold
            if val_coverage >= min_coverage_threshold:
                if val_coverage > best_coverage or (val_coverage == best_coverage and val_interval_score < best_score):
                    best_coverage = val_coverage
                    best_score = val_interval_score
                    best_model_state = self.scoring_fn.state_dict()
                    early_stop_counter = 0
                    improved = True
            # For models that don't meet coverage, just compare interval scores
            elif val_interval_score < best_score:
                best_score = val_interval_score
                best_model_state = self.scoring_fn.state_dict()
                early_stop_counter = 0
                improved = True
            
            if improved:
                # Save best model
                model_path = os.path.join(self.config['paths']['model_dir'], 'best_scoring_model.pt')
                if hasattr(self.scoring_fn, 'save'):
                    self.scoring_fn.save(model_path)
                else:
                    torch.save(best_model_state, model_path)
                    
                logging.info(f"Saved best model with coverage: {best_coverage:.4f}, interval score: {best_score:.4f}")
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= early_stop_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Plot training progress
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                plot_training_progress(history, self.config, output_dir)
                
        # Restore best model
        if best_model_state is not None:
            self.scoring_fn.load_state_dict(best_model_state)
        
        # Final evaluation and results
        final_results = self.evaluate_and_save_results(output_dir)
        
        return history, best_model_state, final_results
            
    def _train_epoch(self, optimizer, scheduler=None):
        """
        Train for one epoch
        
        Args:
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler (optional)
            
        Returns:
            loss_avg: Average loss for the epoch
            coverage_avg: Average coverage for the epoch
            width_avg: Average width for the epoch
        """
        self.scoring_fn.train()
        self.base_model.eval()  # Base model is fixed
        
        # Initialize meters
        loss_meter = AverageMeter()
        coverage_meter = AverageMeter()
        width_meter = AverageMeter()
        coverage_loss_meter = AverageMeter()
        width_loss_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc="Training")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = inputs.size(0)
            
            # Get base model predictions
            with torch.no_grad():
                predictions = self.base_model(inputs)
                
                # Ensure predictions have the right shape
                if predictions.ndim == 1:
                    predictions = predictions.unsqueeze(1)
                elif predictions.ndim > 2:
                    predictions = predictions.view(batch_size, -1)[:, 0:1]
                    
                # Ensure targets have consistent shape with predictions
                if targets.ndim == 1:
                    targets = targets.unsqueeze(1)
                elif targets.ndim > 2:
                    targets = targets.view(batch_size, -1)[:, 0:1]
            
            # Forward pass to get nonconformity scores
            scores = self.scoring_fn(inputs, predictions)
            
            # Calculate coverage stats (without gradient)
            with torch.no_grad():
                errors = torch.abs(targets - predictions)
                if self.symmetric:
                    # For symmetric intervals
                    covered = (scores.squeeze() >= errors.squeeze()).float()
                    covered_ratio = covered.mean()
                else:
                    # For asymmetric intervals
                    lower_scores, upper_scores = scores[:, 0], scores[:, 1]
                    neg_errors = torch.clamp(predictions - targets, min=0)  # When prediction > target
                    pos_errors = torch.clamp(targets - predictions, min=0)  # When target > prediction
                    
                    # Ensure consistent shapes by squeezing if needed
                    neg_errors = neg_errors.squeeze()
                    pos_errors = pos_errors.squeeze()
                    
                    lower_covered = (lower_scores >= neg_errors).float()
                    upper_covered = (upper_scores >= pos_errors).float()
                    covered = torch.min(torch.stack([lower_covered, upper_covered]), dim=0)[0]
                    covered_ratio = covered.mean()
            
            # New target-based coverage loss that prioritizes meeting coverage target
            coverage_gap = torch.relu(self.target_coverage - covered_ratio)
            coverage_loss = torch.exp(5.0 * coverage_gap) - 1.0
            
            # Width loss: only apply when coverage is acceptable
            if self.symmetric:
                if covered_ratio >= self.target_coverage:
                    width_loss = scores.mean()
                else:
                    width_loss = torch.zeros_like(scores.mean())
            else:
                if covered_ratio >= self.target_coverage:
                    width_loss = (lower_scores.mean() + upper_scores.mean()) / 2
                else:
                    width_loss = torch.zeros_like(lower_scores.mean())
            
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
            
            # Step scheduler if using OneCycleLR or similar per-batch scheduler
            if scheduler is not None and isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, 
                                                               torch.optim.lr_scheduler.CyclicLR)):
                scheduler.step()
            
            # Update meters
            loss_meter.update(loss.item(), batch_size)
            coverage_meter.update(covered_ratio.item(), batch_size)
            coverage_loss_meter.update(coverage_loss.item(), batch_size)
            
            if self.symmetric:
                width_meter.update(scores.mean().item(), batch_size)
                width_val = scores.mean().item() if covered_ratio >= self.target_coverage else 0
                width_loss_meter.update(width_val, batch_size)
            else:
                avg_width = (lower_scores.mean() + upper_scores.mean()) / 2
                width_meter.update(avg_width.item(), batch_size)
                width_val = avg_width.item() if covered_ratio >= self.target_coverage else 0
                width_loss_meter.update(width_val, batch_size)
            
            # Update progress bar
            pbar.set_description(
                f"Train Loss: {loss_meter.avg:.4f}, "
                f"Coverage: {coverage_meter.avg:.4f} (Target: {self.target_coverage:.2f}), "
                f"Width: {width_meter.avg:.4f}"
            )
        
        return loss_meter.avg, coverage_meter.avg, width_meter.avg, coverage_loss_meter.avg, width_loss_meter.avg
    
    def _compute_calibration_factor(self):
        """
        Compute calibration factor on calibration set to ensure target coverage
        
        Returns:
            calibration_factor: Factor to scale prediction intervals
        """
        self.scoring_fn.eval()
        self.base_model.eval()
        
        all_inputs = []
        all_targets = []
        all_predictions = []
        all_scores = []
        
        # Collect data on calibration set
        with torch.no_grad():
            for inputs, targets in self.cal_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get base model predictions
                predictions = self.base_model(inputs)
                
                # Ensure predictions have the right shape
                if predictions.ndim == 1:
                    predictions = predictions.unsqueeze(1)
                
                # Ensure targets have matching shape
                if targets.ndim == 1:
                    targets = targets.unsqueeze(1)
                
                # Get nonconformity scores
                scores = self.scoring_fn(inputs, predictions)
                
                all_inputs.append(inputs.cpu())
                all_targets.append(targets.cpu())
                all_predictions.append(predictions.cpu())
                all_scores.append(scores.cpu())
        
        # Concatenate data
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        
        # Compute errors
        errors = torch.abs(all_targets - all_predictions)
        
        if self.symmetric:
            # For symmetric intervals, compute ratio of actual to predicted scores
            # Q = |y - f(x)| / s(x)
            conformity_scores = errors.squeeze() / (all_scores.squeeze() + 1e-8)
            
            # Sort scores
            sorted_scores = torch.sort(conformity_scores).values
            
            # Find quantile for target coverage
            n = len(sorted_scores)
            q_index = int(np.ceil((n + 1) * self.target_coverage)) - 1
            q_index = min(max(0, q_index), n - 1)  # Ensure index is valid
            
            # Get calibration factor
            calibration_factor = sorted_scores[q_index].item()
            
            # Add safety margin to ensure target coverage
            calibration_factor *= 1.1  # Add 10% safety margin
        else:
            # For asymmetric intervals, compute separate calibration factors
            # Extract lower and upper scores
            lower_scores, upper_scores = all_scores[:, 0], all_scores[:, 1]
            
            # Compute signed errors
            neg_errors = torch.clamp(all_predictions - all_targets, min=0)  # When prediction > target
            pos_errors = torch.clamp(all_targets - all_predictions, min=0)  # When target > prediction
            
            # Compute conformity scores
            neg_errors = neg_errors.squeeze()
            pos_errors = pos_errors.squeeze()
            lower_conformity = neg_errors / (lower_scores + 1e-8)
            upper_conformity = pos_errors / (upper_scores + 1e-8)
            
            # Flatten tensors to ensure they're 1-dimensional
            lower_conformity = lower_conformity.flatten()
            upper_conformity = upper_conformity.flatten()
            
            # Sort scores
            sorted_lower = torch.sort(lower_conformity).values
            sorted_upper = torch.sort(upper_conformity).values
            
            # Find quantile for target coverage
            n_lower = len(sorted_lower)
            n_upper = len(sorted_upper)
            
            lower_idx = int(np.ceil((n_lower + 1) * self.target_coverage)) - 1
            upper_idx = int(np.ceil((n_upper + 1) * self.target_coverage)) - 1
            
            lower_idx = min(max(0, lower_idx), n_lower - 1)
            upper_idx = min(max(0, upper_idx), n_upper - 1)
            
            # Get calibration factors, ensuring we're selecting a single element
            lower_factor = sorted_lower[lower_idx].item()
            upper_factor = sorted_upper[upper_idx].item()
            
            # Add safety margin to ensure target coverage
            lower_factor *= 1.1  # Add 10% safety margin
            upper_factor *= 1.1  # Add 10% safety margin
            
            calibration_factor = (lower_factor, upper_factor)
        
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
        self.base_model.eval()
        
        all_inputs = []
        all_targets = []
        all_predictions = []
        all_scores = []
        
        # Collect predictions on test set
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get base model predictions
                predictions = self.base_model(inputs)
                
                # Ensure predictions have the right shape
                if predictions.ndim == 1:
                    predictions = predictions.unsqueeze(1)
                
                # Ensure targets have matching shape
                if targets.ndim == 1:
                    targets = targets.unsqueeze(1)
                
                # Get nonconformity scores
                scores = self.scoring_fn(inputs, predictions)
                
                all_inputs.append(inputs.cpu())
                all_targets.append(targets.cpu())
                all_predictions.append(predictions.cpu())
                all_scores.append(scores.cpu())
        
        # Concatenate data
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        
        # Compute prediction intervals
        lower_bounds, upper_bounds = compute_prediction_intervals(
            all_predictions, all_scores, calibration_factor, 
            symmetric=self.symmetric
        )
        
        # Evaluate metrics
        metrics = evaluate_prediction_intervals(
            all_targets, all_predictions, lower_bounds, upper_bounds,
            output_dir=None  # Will save plots when evaluate_and_save_results is called
        )
        
        return metrics
    
    def evaluate_and_save_results(self, output_dir=None):
        """
        Evaluate on test set and save results
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        if output_dir is None:
            output_dir = self.config['paths']['output_dir']
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute calibration factor
        calibration_factor = self._compute_calibration_factor()
        
        # Evaluate on test set
        self.scoring_fn.eval()
        self.base_model.eval()
        
        all_inputs = []
        all_targets = []
        all_predictions = []
        all_scores = []
        
        # Collect predictions on test set
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get base model predictions
                predictions = self.base_model(inputs)
                
                # Ensure predictions have the right shape
                if predictions.ndim == 1:
                    predictions = predictions.unsqueeze(1)
                
                # Ensure targets have matching shape
                if targets.ndim == 1:
                    targets = targets.unsqueeze(1)
                
                # Get nonconformity scores
                scores = self.scoring_fn(inputs, predictions)
                
                all_inputs.append(inputs.cpu())
                all_targets.append(targets.cpu())
                all_predictions.append(predictions.cpu())
                all_scores.append(scores.cpu())
        
        # Concatenate data
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        
        # Compute prediction intervals
        lower_bounds, upper_bounds = compute_prediction_intervals(
            all_predictions, all_scores, calibration_factor, 
            symmetric=self.symmetric
        )
        
        # Evaluate metrics
        metrics = evaluate_prediction_intervals(
            all_targets, all_predictions, lower_bounds, upper_bounds,
            output_dir=output_dir
        )
        
        # Log results
        logging.info(f"Test metrics:")
        logging.info(f"  Coverage: {metrics['coverage']:.4f} (Target: {self.target_coverage:.4f})")
        logging.info(f"  Average width: {metrics['avg_width']:.4f}")
        logging.info(f"  Interval score: {metrics['interval_score']:.4f}")
        logging.info(f"  Efficiency: {metrics['efficiency']:.4f}")
        logging.info(f"  RMSE: {metrics['rmse']:.4f}")
        logging.info(f"  MAE: {metrics['mae']:.4f}")
        logging.info(f"  Width-error correlation: {metrics['width_error_corr']:.4f}")
        
        # Calculate additional metrics
        coverage_gap = abs(metrics['coverage'] - self.target_coverage)
        logging.info(f"  Coverage gap: {coverage_gap:.4f}")
        
        # Calculate success criteria
        coverage_success = metrics['coverage'] >= self.target_coverage * 0.95
        efficiency_threshold = 0.05  # Adjust based on your data range
        efficiency_success = metrics['efficiency'] >= efficiency_threshold
        width_error_corr_success = metrics['width_error_corr'] > 0.3
        
        logging.info(f"  Coverage target met: {'Yes' if coverage_success else 'No'}")
        logging.info(f"  Efficiency target met: {'Yes' if efficiency_success else 'No'}")
        logging.info(f"  Width-error correlation target met: {'Yes' if width_error_corr_success else 'No'}")
        
        # Plot results
        plot_prediction_intervals(
            all_targets, all_predictions, lower_bounds, upper_bounds, 
            self.config, output_dir
        )
        
        # Compute additional plots for understanding model behavior
        self._plot_calibration_diagnostics(
            all_targets, all_predictions, all_scores, calibration_factor, output_dir
        )
        
        # Save results
        results_df = save_results(
            all_targets, all_predictions, lower_bounds, upper_bounds, 
            all_scores, metrics, self.config, output_dir
        )
        
        return metrics
        
    def _plot_calibration_diagnostics(self, targets, predictions, scores, calibration_factor, output_dir):
        """
        Create additional diagnostic plots for understanding model calibration
        
        Args:
            targets: True target values
            predictions: Model predictions
            scores: Nonconformity scores
            calibration_factor: Calibration factor
            output_dir: Directory to save plots
        """
        # Convert to numpy
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
            
        # Calculate errors
        errors = np.abs(targets - predictions)
        
        # 1. Plot scores vs. errors with calibration factor line
        plt.figure(figsize=(10, 8))
        if self.symmetric:
            scores_flat = scores.squeeze()
            plt.scatter(scores_flat, errors, alpha=0.3, s=10)
            
            # Draw calibration factor line
            max_val = max(scores_flat.max(), errors.max()) * 1.1
            cf_line = np.linspace(0, max_val, 100)
            plt.plot(cf_line, calibration_factor * cf_line, 'r--', 
                    label=f'Calibration Factor: {calibration_factor:.4f}')
            
            # Draw y=x line for reference
            plt.plot(cf_line, cf_line, 'g--', alpha=0.5, label='y=x (perfect calibration)')
            
            plt.title('Nonconformity Scores vs. Actual Errors')
            plt.xlabel('Predicted Score')
            plt.ylabel('Absolute Error')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'scores_vs_errors.png'), dpi=150)
            plt.close()
            
            # 2. Plot histograms of conformity scores
            plt.figure(figsize=(10, 6))
            conformity_scores = errors / (scores_flat + 1e-8)
            plt.hist(conformity_scores, bins=50, alpha=0.7)
            plt.axvline(x=calibration_factor, color='r', linestyle='--', 
                        label=f'Calibration Factor: {calibration_factor:.4f}')
            plt.title('Distribution of Conformity Scores (error/score)')
            plt.xlabel('Conformity Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'conformity_scores_distribution.png'), dpi=150)
            plt.close()
        else:
            # For asymmetric intervals
            lower_scores = scores[:, 0]
            upper_scores = scores[:, 1]
            neg_errors = np.maximum(predictions - targets, 0)
            pos_errors = np.maximum(targets - predictions, 0)
            
            # Create 2x1 subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Lower bound plot
            ax1.scatter(lower_scores, neg_errors, alpha=0.3, s=10)
            max_val = max(lower_scores.max(), neg_errors.max()) * 1.1
            cf_line = np.linspace(0, max_val, 100)
            lower_cf = calibration_factor[0]
            ax1.plot(cf_line, lower_cf * cf_line, 'r--', 
                   label=f'Lower CF: {lower_cf:.4f}')
            ax1.plot(cf_line, cf_line, 'g--', alpha=0.5, label='y=x')
            ax1.set_title('Lower Bound Scores vs. Negative Errors')
            ax1.set_xlabel('Lower Score')
            ax1.set_ylabel('Negative Error')
            ax1.legend()
            ax1.grid(True)
            
            # Upper bound plot
            ax2.scatter(upper_scores, pos_errors, alpha=0.3, s=10)
            max_val = max(upper_scores.max(), pos_errors.max()) * 1.1
            cf_line = np.linspace(0, max_val, 100)
            upper_cf = calibration_factor[1]
            ax2.plot(cf_line, upper_cf * cf_line, 'r--', 
                   label=f'Upper CF: {upper_cf:.4f}')
            ax2.plot(cf_line, cf_line, 'g--', alpha=0.5, label='y=x')
            ax2.set_title('Upper Bound Scores vs. Positive Errors')
            ax2.set_xlabel('Upper Score')
            ax2.set_ylabel('Positive Error')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'asymmetric_scores_vs_errors.png'), dpi=150)
            plt.close()

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
            return torch.tensor(preds, dtype=torch.float32, device=x.device).unsqueeze(1)
    
    return RFWrapper(rf_model) 