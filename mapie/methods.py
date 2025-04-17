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


# Factory function to get the appropriate conformal method
def get_conformal_method(method_name, model, conformity_score, alpha=0.1, **kwargs):
    """
    Factory function to get the conformal method by name.
    
    Parameters
    ----------
    method_name : str
        Name of the conformal method.
    model : object
        Trained model with a predict method.
    conformity_score : ConformityScore
        Conformity score to use.
    alpha : float, default=0.1
        Significance level (1-alpha = coverage level).
        
    Returns
    -------
    ConformalMethod
        Instance of the conformal method.
    """
    methods = {
        'split': SplitMethod(model, conformity_score, alpha)
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown conformal method: {method_name}. Available methods: {list(methods.keys())}")
    
    logger.info(f"Using conformal method: {method_name}")
    return methods[method_name] 