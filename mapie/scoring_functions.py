import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class ConformityScore(ABC):
    """Base abstract class for conformity scores."""
    
    @abstractmethod
    def compute(self, y_true, y_pred, **kwargs):
        """
        Compute conformity scores.
        
        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.
        **kwargs : dict
            Additional arguments specific to the score.
            
        Returns
        -------
        array-like
            Conformity scores.
        """
        pass
    
    @property
    @abstractmethod
    def is_symmetric(self):
        """Return whether the score is symmetric or not."""
        pass
    
    def get_prediction_intervals(self, y_pred, quantile, **kwargs):
        """
        Compute prediction intervals based on the conformity score.
        
        Parameters
        ----------
        y_pred : array-like
            Predicted target values.
        quantile : float
            Quantile to use for interval construction.
        **kwargs : dict
            Additional arguments specific to the score.
            
        Returns
        -------
        tuple
            Lower and upper bounds of the prediction intervals.
        """
        pass


class AbsoluteResidualScore(ConformityScore):
    """Implementation of the absolute residual conformity score."""
    
    def compute(self, y_true, y_pred, **kwargs):
        """Compute absolute residual scores: |y_true - y_pred|."""
        return np.abs(y_true - y_pred)
    
    @property
    def is_symmetric(self):
        """The absolute residual score is symmetric."""
        return True
    
    def get_prediction_intervals(self, y_pred, quantile, **kwargs):
        """
        Compute prediction intervals based on the absolute residual score.
        
        The interval is [y_pred - quantile, y_pred + quantile].
        """
        lower_bound = y_pred - quantile
        upper_bound = y_pred + quantile
        return lower_bound, upper_bound


class QuantileResidualScore(ConformityScore):
    """
    Implementation of quantile residual conformity scores for CQR method.
    This class supports both asymmetric and symmetric variants.
    """
    
    def __init__(self, symmetric=False):
        """
        Initialize the quantile residual score.
        
        Parameters
        ----------
        symmetric : bool, default=False
            Whether to use the symmetric variant or not.
        """
        self.symmetric = symmetric
    
    def compute(self, y_true, y_pred_low, y_pred_high=None):
        """
        Compute quantile residual scores for CQR.
        
        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred_low : array-like
            Predictions from the lower quantile model.
        y_pred_high : array-like, optional
            Predictions from the upper quantile model.
            Required if symmetric=False.
            
        Returns
        -------
        array-like or tuple of array-like
            If symmetric=True, returns a single array of max residuals.
            If symmetric=False, returns a tuple of (lower_residuals, upper_residuals).
        """
        if not self.symmetric and y_pred_high is None:
            raise ValueError("Upper quantile predictions required for asymmetric CQR")
        
        # Calculate residuals
        lower_residuals = y_true - y_pred_low
        
        if self.symmetric:
            upper_residuals = y_pred_high - y_true
            # Take the maximum of lower and upper residuals
            return np.maximum(lower_residuals, upper_residuals)
        else:
            upper_residuals = y_pred_high - y_true
            return lower_residuals, upper_residuals
    
    @property
    def is_symmetric(self):
        """Return whether the score is symmetric or not."""
        return self.symmetric
    
    def get_prediction_intervals(self, y_pred_bounds, quantiles, **kwargs):
        """
        Compute prediction intervals based on the quantile residual score.
        
        Parameters
        ----------
        y_pred_bounds : tuple
            Tuple of (y_pred_low, y_pred_high) predictions from quantile models.
        quantiles : float or tuple
            If symmetric=True, a single quantile value.
            If symmetric=False, a tuple of (lower_quantile, upper_quantile).
            
        Returns
        -------
        tuple
            Lower and upper bounds of the prediction intervals.
        """
        y_pred_low, y_pred_high = y_pred_bounds
        
        if self.symmetric:
            # For symmetric variant, we add the same quantile to both bounds
            q = quantiles
            lower_bound = y_pred_low - q
            upper_bound = y_pred_high + q
        else:
            # For asymmetric variant, we use different quantiles
            q_low, q_high = quantiles
            lower_bound = y_pred_low - q_low
            upper_bound = y_pred_high + q_high
            
        return lower_bound, upper_bound


# Factory function to get the appropriate conformity score
def get_conformity_score(score_name, **kwargs):
    """
    Factory function to get the conformity score by name.
    
    Parameters
    ----------
    score_name : str
        Name of the conformity score.
    **kwargs : dict
        Additional parameters for the conformity score.
        
    Returns
    -------
    ConformityScore
        Instance of the conformity score.
    """
    scores = {
        'absolute_residual': AbsoluteResidualScore(),
        'quantile_residual': QuantileResidualScore(**kwargs),
        'quantile_residual_symmetric': QuantileResidualScore(symmetric=True),
        # Add more scores when implementing them
    }
    
    if score_name not in scores:
        raise ValueError(f"Unknown conformity score: {score_name}")
    
    logger.info(f"Using conformity score: {score_name}")
    return scores[score_name] 