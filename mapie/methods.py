import numpy as np
import time
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class ConformalMethod(ABC):
    """Base abstract class for conformal prediction methods."""
    
    def __init__(self, model, conformity_score, alpha=0.1):
        """
        Initialize the conformal method.
        
        Parameters
        ----------
        model : object
            Trained model with a predict method.
        conformity_score : ConformityScore
            Conformity score to use.
        alpha : float, default=0.1
            Significance level (1-alpha = coverage level).
        """
        self.model = model
        self.conformity_score = conformity_score
        self.alpha = alpha
        
    @abstractmethod
    def calibrate(self, X_train, y_train, X_val=None, y_val=None):
        """
        Calibrate the method on training/validation data.
        
        Parameters
        ----------
        X_train : array-like
            Training feature matrix.
        y_train : array-like
            Training target values.
        X_val : array-like, optional
            Validation feature matrix.
        y_val : array-like, optional
            Validation target values.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Generate prediction intervals for new data.
        
        Parameters
        ----------
        X : array-like
            Feature matrix to predict on.
            
        Returns
        -------
        tuple
            Lower and upper bounds of the prediction intervals.
        """
        pass
    
    @property
    @abstractmethod
    def theoretical_coverage(self):
        """Return the theoretical coverage of the method."""
        pass
    
    @abstractmethod
    def training_cost(self, n_samples):
        """
        Return the training cost in terms of model fits.
        
        Parameters
        ----------
        n_samples : int
            Number of training samples.
            
        Returns
        -------
        float or str
            Training cost.
        """
        pass
    
    @abstractmethod
    def evaluation_cost(self, n_train, n_test):
        """
        Return the evaluation cost.
        
        Parameters
        ----------
        n_train : int
            Number of training samples.
        n_test : int
            Number of test samples.
            
        Returns
        -------
        float or str
            Evaluation cost.
        """
        pass


class SplitMethod(ConformalMethod):
    """Implementation of the split conformal prediction method."""
    
    def __init__(self, model, conformity_score, alpha=0.1):
        super().__init__(model, conformity_score, alpha)
        self.quantile = None
    
    def calibrate(self, X_train, y_train, X_val=None, y_val=None):
        """
        Calibrate the method using validation data.
        
        The split method requires a separate calibration dataset (validation data)
        to compute residuals and quantiles, providing valid coverage guarantees.
        """
        start_time = time.time()
        
        if X_val is None or y_val is None:
            raise ValueError("Split method requires validation data for calibration. "
                            "Please provide X_val and y_val.")
        
        logger.info("Using validation data for calibration (split method)")
        # Make predictions on validation data
        y_pred = self.model.predict(X_val)
        
        # Compute conformity scores
        scores = self.conformity_score.compute(y_val, y_pred)
        
        # Calculate the quantile for the prediction intervals
        self.quantile = np.quantile(scores, 1 - self.alpha)
        
        calibration_time = time.time() - start_time
        logger.info(f"Split method calibration completed in {calibration_time:.2f} seconds")
        logger.info(f"Calibrated quantile: {self.quantile:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Generate prediction intervals for new data.
        """
        if self.quantile is None:
            raise ValueError("Method not calibrated. Call calibrate() first.")
        
        # Make point predictions
        y_pred = self.model.predict(X)
        
        # Compute prediction intervals
        lower_bound, upper_bound = self.conformity_score.get_prediction_intervals(
            y_pred, self.quantile
        )
        
        return lower_bound, upper_bound, y_pred
    
    @property
    def theoretical_coverage(self):
        """
        When using a proper validation set, the theoretical coverage is ≥ 1-alpha.
        """
        return f"≥ {1 - self.alpha} (asymptotic guarantee)"
    
    def training_cost(self, n_samples):
        """
        Training cost of the split method is just 1 model fit.
        """
        return 1
    
    def evaluation_cost(self, n_train, n_test):
        """
        Evaluation cost of the split method is just the cost of making
        predictions on the test set.
        """
        return f"{n_test}"


class CQRMethod(ConformalMethod):
    """Implementation of the Conformalized Quantile Regression (CQR) method."""
    
    def __init__(self, model, conformity_score, alpha=0.1, low_quantile=None, high_quantile=None, base_model=None):
        """
        Initialize the CQR method.
        
        Parameters
        ----------
        model : tuple
            Tuple of (low_quantile_model, high_quantile_model), where each model
            is a trained quantile regressor with a predict method.
        conformity_score : ConformityScore
            Conformity score to use (must be a QuantileResidualScore instance).
        alpha : float, default=0.1
            Significance level (1-alpha = coverage level).
        low_quantile : float, optional, default=alpha/2
            Lower quantile level used in training the quantile regressors.
        high_quantile : float, optional, default=1-alpha/2
            Upper quantile level used in training the quantile regressors.
        base_model : object, optional
            The base model to use for point predictions. If None, point predictions
            will be calculated as the average of quantile predictions.
        """
        # For CQR, the model should be a tuple of two quantile regressors
        if not isinstance(model, tuple) or len(model) != 2:
            raise ValueError("For CQR method, model must be a tuple of (low_quantile_model, high_quantile_model)")
        
        # The conformity score should be a QuantileResidualScore
        if not hasattr(conformity_score, 'symmetric'):
            raise ValueError("CQR method requires a QuantileResidualScore conformity score")
        
        super().__init__(model, conformity_score, alpha)
        
        # Unpack the models
        self.low_quantile_model, self.high_quantile_model = model
        
        # Store the base model for point predictions if provided
        self.base_model = base_model
        
        # Set the quantile levels if not provided
        self.low_quantile = low_quantile if low_quantile is not None else alpha / 2
        self.high_quantile = high_quantile if high_quantile is not None else 1 - alpha / 2
        
        # Initialize variables for storing calibration results
        self.low_quantile_value = None
        self.high_quantile_value = None
        self.quantile_value = None
    
    def calibrate(self, X_train, y_train, X_val=None, y_val=None):
        """
        Calibrate the CQR method using validation data.
        
        Parameters
        ----------
        X_train : array-like
            Training feature matrix (not used for calibration but kept for API consistency).
        y_train : array-like
            Training target values (not used for calibration but kept for API consistency).
        X_val : array-like
            Validation/calibration feature matrix.
        y_val : array-like
            Validation/calibration target values.
        """
        start_time = time.time()
        
        if X_val is None or y_val is None:
            raise ValueError("CQR method requires validation/calibration data. "
                           "Please provide X_val and y_val.")
        
        logger.info("Calibrating CQR method...")
        
        # Make predictions using both quantile models on the calibration set
        y_pred_low = self.low_quantile_model.predict(X_val)
        y_pred_high = self.high_quantile_model.predict(X_val)
        
        # Compute conformity scores
        if self.conformity_score.is_symmetric:
            # Symmetric variant
            scores = self.conformity_score.compute(y_val, y_pred_low, y_pred_high)
            
            # Calculate the quantile adjustment for proper coverage
            n_calib = len(y_val)
            adjusted_level = (1 - self.alpha) * (1 + 1/n_calib)
            adjusted_level = min(adjusted_level, 1.0)  # Ensure it doesn't exceed 1
            
            # Get the quantile of the scores
            self.quantile_value = np.quantile(scores, adjusted_level)
            
            logger.info(f"Calibrated symmetric quantile (level {adjusted_level:.4f}): {self.quantile_value:.4f}")
        else:
            # Asymmetric variant
            low_scores, high_scores = self.conformity_score.compute(y_val, y_pred_low, y_pred_high)
            
            # Calculate the quantile adjustment for proper coverage
            n_calib = len(y_val)
            adjusted_level = (1 - self.alpha) * (1 + 1/n_calib)
            adjusted_level = min(adjusted_level, 1.0)  # Ensure it doesn't exceed 1
            
            # Get the quantiles of both score sets
            self.low_quantile_value = np.quantile(low_scores, adjusted_level)
            self.high_quantile_value = np.quantile(high_scores, adjusted_level)
            
            logger.info(f"Calibrated lower quantile (level {adjusted_level:.4f}): {self.low_quantile_value:.4f}")
            logger.info(f"Calibrated upper quantile (level {adjusted_level:.4f}): {self.high_quantile_value:.4f}")
        
        calibration_time = time.time() - start_time
        logger.info(f"CQR calibration completed in {calibration_time:.2f} seconds")
        
        return self
    
    def predict(self, X):
        """
        Generate prediction intervals for new data using CQR method.
        
        Parameters
        ----------
        X : array-like
            Feature matrix to predict on.
            
        Returns
        -------
        tuple
            Lower and upper bounds of the prediction intervals, and center predictions.
        """
        if self.conformity_score.is_symmetric and self.quantile_value is None:
            raise ValueError("Method not calibrated. Call calibrate() first.")
        elif not self.conformity_score.is_symmetric and (self.low_quantile_value is None or self.high_quantile_value is None):
            raise ValueError("Method not calibrated. Call calibrate() first.")
        
        # Make predictions using both quantile models
        y_pred_low = self.low_quantile_model.predict(X)
        y_pred_high = self.high_quantile_model.predict(X)
        
        # Calculate center predictions
        if self.base_model is not None:
            # Use the base model for point predictions if provided
            y_pred = self.base_model.predict(X)
            logger.info("Using base model for point predictions")
        else:
            # Fall back to average of quantile predictions if no base model
            y_pred = (y_pred_low + y_pred_high) / 2
            logger.info("Using average of quantile predictions for point predictions")
        
        # Compute prediction intervals
        if self.conformity_score.is_symmetric:
            # For symmetric variant, use the same quantile value for both bounds
            lower_bound, upper_bound = self.conformity_score.get_prediction_intervals(
                (y_pred_low, y_pred_high), self.quantile_value
            )
        else:
            # For asymmetric variant, use different quantile values
            lower_bound, upper_bound = self.conformity_score.get_prediction_intervals(
                (y_pred_low, y_pred_high), (self.low_quantile_value, self.high_quantile_value)
            )
        
        return lower_bound, upper_bound, y_pred
    
    @property
    def theoretical_coverage(self):
        """
        When using a proper calibration set, the theoretical coverage is ≥ 1-alpha.
        """
        return f"≥ {1 - self.alpha} (asymptotic guarantee)"
    
    def training_cost(self, n_samples):
        """
        Training cost of the CQR method is actually the cost of training two quantile regressors,
        but this happens outside this class, so we return 'external'.
        """
        return "external (2 quantile models)"
    
    def evaluation_cost(self, n_train, n_test):
        """
        Evaluation cost of the CQR method is the cost of making
        predictions with two quantile models on the test set.
        """
        return f"2 × {n_test}"


# Factory function to get the appropriate conformal method
def get_conformal_method(method_name, model, conformity_score, alpha=0.1, **kwargs):
    """
    Factory function to get the conformal method by name.
    
    Parameters
    ----------
    method_name : str
        Name of the conformal method.
    model : object or tuple
        Trained model with a predict method, or tuple of models for CQR.
    conformity_score : ConformityScore
        Conformity score to use.
    alpha : float, default=0.1
        Significance level (1-alpha = coverage level).
    **kwargs : dict
        Additional parameters for the conformal method.
        
    Returns
    -------
    ConformalMethod
        Instance of the conformal method.
    """
    methods = {
        'split': SplitMethod(model, conformity_score, alpha),
        'cqr': CQRMethod(model, conformity_score, alpha, **kwargs),
        'cqr_symmetric': CQRMethod(model, conformity_score, alpha, **kwargs)
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown conformal method: {method_name}. Available methods: {list(methods.keys())}")
    
    logger.info(f"Using conformal method: {method_name}")
    return methods[method_name] 