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
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.config['scheduler']['factor'],
            patience=self.config['scheduler']['patience'],
            verbose=True
        )
        
        # Initialize tracking variables
        best_model_state = None
        best_score = float('inf')
        early_stop_counter = 0
        early_stop_patience = self.config['scheduler']['early_stop_patience']
        
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
            history['val_coverage'].append(val_coverage)
            history['val_width'].append(val_width)
            history['val_interval_score'].append(val_interval_score)
            history['calibration_factor'].append(calibration_factor)
            history['learning_rate'].append(lr)
            
            # Update scheduler
            scheduler.step(val_interval_score)
            
            # Check for best model (based on validation interval score)
            if val_interval_score < best_score:
                best_score = val_interval_score
                best_model_state = self.scoring_fn.state_dict()
                early_stop_counter = 0
                
                # Save best model
                model_path = os.path.join(self.config['paths']['model_dir'], 'best_scoring_model.pt')
                if hasattr(self.scoring_fn, 'save'):
                    self.scoring_fn.save(model_path)
                else:
                    torch.save(best_model_state, model_path)
                    
                logging.info(f"Saved best model with interval score: {best_score:.4f}")
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
        self.base_model.eval()  # Base model is fixed
        
        # Initialize meters
        loss_meter = AverageMeter()
        coverage_meter = AverageMeter()
        width_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc="Training")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = inputs.size(0)
            
            # Get base model predictions
            with torch.no_grad():
                predictions = self.base_model(inputs)
            
            # Forward pass to get nonconformity scores
            scores = self.scoring_fn(inputs)
            
            # Calculate calibration factor for loss estimation (without gradient)
            with torch.no_grad():
                errors = torch.abs(targets - predictions)
                if self.symmetric:
                    # For symmetric intervals
                    cal_ratio = errors / (scores.squeeze() + 1e-8)
                    covered = (scores.squeeze() >= errors).float()
                else:
                    # For asymmetric intervals
                    lower_scores, upper_scores = scores[:, 0], scores[:, 1]
                    neg_errors = torch.clamp(predictions - targets, min=0)  # When prediction > target
                    pos_errors = torch.clamp(targets - predictions, min=0)  # When target > prediction
                    
                    lower_ratio = neg_errors / (lower_scores + 1e-8)
                    upper_ratio = pos_errors / (upper_scores + 1e-8)
                    
                    lower_covered = (lower_scores >= neg_errors).float()
                    upper_covered = (upper_scores >= pos_errors).float()
                    covered = torch.min(torch.stack([lower_covered, upper_covered]), dim=0)[0]
            
            # Coverage loss: penalize if score < error (uncovered point)
            # This encourages scores to be large enough to cover the errors
            if self.symmetric:
                coverage_loss = torch.relu(errors - scores.squeeze()).mean()
                # Width loss: penalize large scores
                width_loss = scores.mean()
            else:
                coverage_loss_lower = torch.relu(neg_errors - lower_scores).mean()
                coverage_loss_upper = torch.relu(pos_errors - upper_scores).mean()
                coverage_loss = coverage_loss_lower + coverage_loss_upper
                
                # Width loss: penalize large scores
                width_loss = (lower_scores.mean() + upper_scores.mean()) / 2
            
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
            
            # Update meters
            loss_meter.update(loss.item(), batch_size)
            coverage_meter.update(covered.mean().item(), batch_size)
            
            if self.symmetric:
                width_meter.update(scores.mean().item(), batch_size)
            else:
                avg_width = (lower_scores.mean() + upper_scores.mean()) / 2
                width_meter.update(avg_width.item(), batch_size)
            
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
                
                # Get nonconformity scores
                scores = self.scoring_fn(inputs)
                
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
            conformity_scores = errors / (all_scores.squeeze() + 1e-8)
            
            # Sort scores
            sorted_scores = torch.sort(conformity_scores).values
            
            # Find quantile for target coverage
            n = len(sorted_scores)
            q_index = int(np.ceil((n + 1) * self.target_coverage)) - 1
            q_index = min(max(0, q_index), n - 1)  # Ensure index is valid
            
            # Get calibration factor
            calibration_factor = sorted_scores[q_index].item()
        else:
            # For asymmetric intervals, compute separate calibration factors
            # Extract lower and upper scores
            lower_scores, upper_scores = all_scores[:, 0], all_scores[:, 1]
            
            # Compute signed errors
            neg_errors = torch.clamp(all_predictions - all_targets, min=0)  # When prediction > target
            pos_errors = torch.clamp(all_targets - all_predictions, min=0)  # When target > prediction
            
            # Compute conformity scores
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
                
                # Get nonconformity scores
                scores = self.scoring_fn(inputs)
                
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
            all_targets, all_predictions, lower_bounds, upper_bounds
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
                
                # Get nonconformity scores
                scores = self.scoring_fn(inputs)
                
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
            all_targets, all_predictions, lower_bounds, upper_bounds
        )
        
        # Log results
        logging.info(f"Test metrics:")
        logging.info(f"  Coverage: {metrics['coverage']:.4f} (Target: {self.target_coverage:.4f})")
        logging.info(f"  Average width: {metrics['avg_width']:.4f}")
        logging.info(f"  Interval score: {metrics['interval_score']:.4f}")
        logging.info(f"  Efficiency: {metrics['efficiency']:.4f}")
        logging.info(f"  RMSE: {metrics['rmse']:.4f}")
        logging.info(f"  Width-error correlation: {metrics['width_error_corr']:.4f}")
        
        # Plot results
        plot_prediction_intervals(
            all_targets, all_predictions, lower_bounds, upper_bounds, 
            self.config, output_dir
        )
        
        # Save results
        results_df = save_results(
            all_targets, all_predictions, lower_bounds, upper_bounds, 
            all_scores, metrics, self.config, output_dir
        )
        
        return metrics

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