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


# Factory function to get the appropriate conformity score
def get_conformity_score(score_name):
    """
    Factory function to get the conformity score by name.
    
    Parameters
    ----------
    score_name : str
        Name of the conformity score.
        
    Returns
    -------
    ConformityScore
        Instance of the conformity score.
    """
    scores = {
        'absolute_residual': AbsoluteResidualScore(),
        # Add more scores when implementing them
    }
    
    if score_name not in scores:
        raise ValueError(f"Unknown conformity score: {score_name}")
    
    logger.info(f"Using conformity score: {score_name}")
    return scores[score_name] 