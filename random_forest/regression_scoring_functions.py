import torch
import torch.nn as nn
import numpy as np
import logging

class BaseRegressionScorer:
    """Base class for all scoring functions"""
    def __init__(self):
        pass
    
    def fit(self, train_loader):
        """Train the scoring function (if needed)"""
        pass
    
    def score(self, inputs, targets=None, predictions=None):
        """Compute nonconformity scores"""
        raise NotImplementedError("Subclasses must implement score()")
    
    def save(self, path):
        """Save scorer to disk"""
        pass
    
    def load(self, path):
        """Load scorer from disk"""
        pass

class AbsoluteErrorScorer(BaseRegressionScorer):
    """
    Simple absolute error scorer
    
    Nonconformity score = |y_true - y_pred|
    """
    def __init__(self):
        super().__init__()
    
    def score(self, inputs, targets=None, predictions=None):
        """
        Compute absolute error as nonconformity score
        
        Args:
            inputs: Input features (unused in this scorer)
            targets: True target values
            predictions: Model predictions
            
        Returns:
            scores: Absolute error scores
        """
        if targets is None or predictions is None:
            raise ValueError("Both targets and predictions must be provided")
        
        return torch.abs(targets - predictions)

class SquaredErrorScorer(BaseRegressionScorer):
    """
    Squared error scorer
    
    Nonconformity score = (y_true - y_pred)²
    """
    def __init__(self):
        super().__init__()
    
    def score(self, inputs, targets=None, predictions=None):
        """
        Compute squared error as nonconformity score
        
        Args:
            inputs: Input features (unused in this scorer)
            targets: True target values
            predictions: Model predictions
            
        Returns:
            scores: Squared error scores
        """
        if targets is None or predictions is None:
            raise ValueError("Both targets and predictions must be provided")
        
        return torch.pow(targets - predictions, 2)

class NormalizedErrorScorer(BaseRegressionScorer):
    """
    Normalized absolute error scorer
    
    Normalizes errors by the standard deviation of the training residuals
    """
    def __init__(self):
        super().__init__()
        self.residual_std = None
    
    def fit(self, train_loader):
        """
        Compute standard deviation of residuals on training data
        
        Args:
            train_loader: DataLoader containing training data
        """
        all_residuals = []
        
        # Collect all residuals (assuming train_loader provides inputs, targets, predictions)
        for batch in train_loader:
            inputs, targets = batch
            predictions = inputs  # Placeholder - would use your base model here
            residuals = (targets - predictions).abs().cpu().numpy()
            all_residuals.extend(residuals)
        
        # Compute standard deviation
        self.residual_std = np.std(all_residuals) + 1e-8  # Add small constant to avoid division by zero
    
    def score(self, inputs, targets=None, predictions=None):
        """
        Compute normalized absolute error as nonconformity score
        
        Args:
            inputs: Input features (unused in this scorer)
            targets: True target values
            predictions: Model predictions
            
        Returns:
            scores: Normalized error scores
        """
        if targets is None or predictions is None:
            raise ValueError("Both targets and predictions must be provided")
        
        if self.residual_std is None:
            raise ValueError("Scorer not fitted. Call fit() before scoring.")
        
        return torch.abs(targets - predictions) / self.residual_std

class LearnableScoringMLP(nn.Module, BaseRegressionScorer):
    """
    MLP that learns to predict nonconformity scores directly from features
    
    Instead of using a predefined score, this model learns the relationship
    between features and the appropriate nonconformity scores.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16], output_dim=1, 
                 dropout_rate=0.2, use_batch_norm=True, activation='leaky_relu',
                 final_activation='softplus', l2_lambda=1e-5, min_score=0.05,
                 include_prediction=True):
        """
        Initialize learnable scoring function
        
        Args:
            input_dim: Input dimension (feature dimension)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for symmetric, 2 for asymmetric)
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            activation: Activation function to use ('relu', 'leaky_relu', 'tanh', 'sigmoid')
            final_activation: Final activation function ('softplus', 'exp', 'relu')
            l2_lambda: L2 regularization strength
            min_score: Minimum score to prevent degenerate intervals
            include_prediction: Whether to include base model prediction as input
        """
        nn.Module.__init__(self)
        BaseRegressionScorer.__init__(self)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l2_lambda = l2_lambda
        self.min_score = min_score
        self.include_prediction = include_prediction
        
        # If including prediction, increase input dimension by 1
        self.actual_input_dim = input_dim + 1 if include_prediction else input_dim
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Choose final activation function
        if final_activation == 'softplus':
            self.final_activation = nn.Softplus(beta=1)
        elif final_activation == 'exp':
            self.final_activation = lambda x: torch.exp(x)
        elif final_activation == 'relu':
            self.final_activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported final activation: {final_activation}")
        
        # Initialize weights function
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    # Use slightly higher positive bias to encourage higher initial scores
                    m.bias.data.fill_(0.5)
        
        # Build MLP layers
        layers = []
        prev_dim = self.actual_input_dim
        
        # Hidden layers with dropout and optional batch norm
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
                
            layers.append(self.activation)
            
            if i < len(hidden_dims) - 1:  # No dropout after last hidden layer
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = dim
        
        # Final layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Add final activation to ensure positive output
        # Nonconformity scores must be positive
        layers.append(self.final_activation)
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
        
        # Apply initialization
        for m in self.network:
            if isinstance(m, nn.Linear):
                init_weights(m)
        
        # Regularization
        self.l2_reg = 0
    
    def forward(self, x, predictions=None):
        """
        Forward pass to predict nonconformity scores
        
        Args:
            x: Input features with shape [batch_size, input_dim]
            predictions: Base model predictions (optional)
        
        Returns:
            scores: Predicted nonconformity scores
        """
        # Combine features with predictions if needed
        if self.include_prediction:
            if predictions is None:
                raise ValueError("predictions must be provided when include_prediction=True")
            
            # Ensure predictions have the right shape
            if predictions.ndim == 1:
                predictions = predictions.unsqueeze(1)
            
            # Concatenate features and predictions
            x = torch.cat([x, predictions], dim=1)
        
        scores = self.network(x)
        
        # Add L2 regularization
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
        # Store L2 regularization value (convert to scalar before multiplying)
        self.l2_reg = float(self.l2_lambda) * l2_reg
        
        # Ensure reasonable score values
        scores = torch.clamp(scores, min=self.min_score)
        
        # Ensure scores have the right shape (batch_size, output_dim)
        if scores.ndim == 1:
            # If scores are one-dimensional, reshape to (batch_size, 1)
            scores = scores.view(-1, 1)
        elif scores.ndim > 2:
            # If scores have too many dimensions, flatten to 2D
            batch_size = x.size(0)
            scores = scores.view(batch_size, -1)
            # Take only the first output_dim columns
            scores = scores[:, :self.output_dim]
            logging.warning(f"Reshaped scores from complex shape to {scores.shape}")
        
        # Final check to ensure shape matches expected output_dim
        if scores.size(1) != self.output_dim:
            logging.warning(f"Score dimension mismatch! Expected {self.output_dim}, got {scores.size(1)}. Reshaping...")
            # Take first output_dim columns or pad with min_score
            if scores.size(1) > self.output_dim:
                scores = scores[:, :self.output_dim]
            else:
                padding = torch.ones(scores.size(0), self.output_dim - scores.size(1), device=scores.device) * self.min_score
                scores = torch.cat([scores, padding], dim=1)
        
        return scores
    
    def score(self, inputs, targets=None, predictions=None):
        """
        Compute learned nonconformity scores
        
        Args:
            inputs: Input features
            targets: True target values (unused in this scorer)
            predictions: Model predictions (used if include_prediction=True)
            
        Returns:
            scores: Learned nonconformity scores
        """
        # At inference time, we don't need to compute regularization
        with torch.no_grad():
            scores = self.forward(inputs, predictions)
            
            # Ensure output shape is correct (batch_size, output_dim)
            if scores.ndim == 1:
                scores = scores.view(-1, 1)
            
            # Final check to ensure shape matches expected output_dim
            if scores.size(1) != self.output_dim:
                logging.warning(f"Score dimension mismatch in scoring! Expected {self.output_dim}, got {scores.size(1)}. Reshaping...")
                # Take first output_dim columns or pad with min_score
                if scores.size(1) > self.output_dim:
                    scores = scores[:, :self.output_dim]
                else:
                    padding = torch.ones(scores.size(0), self.output_dim - scores.size(1), device=scores.device) * self.min_score
                    scores = torch.cat([scores, padding], dim=1)
        
        return scores
    
    def save(self, path):
        """
        Save model to disk
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'l2_lambda': self.l2_lambda,
            'min_score': self.min_score,
            'include_prediction': self.include_prediction
        }, path)
    
    def load(self, path):
        """
        Load model from disk
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.input_dim = checkpoint['input_dim']
        self.output_dim = checkpoint['output_dim']
        self.l2_lambda = checkpoint['l2_lambda']
        self.min_score = checkpoint['min_score']
        self.include_prediction = checkpoint.get('include_prediction', False)
        self.actual_input_dim = self.input_dim + 1 if self.include_prediction else self.input_dim

class AsymmetricScoringMLP(LearnableScoringMLP):
    """
    Extension of LearnableScoringMLP that outputs separate scores for lower and upper bounds
    
    This allows for asymmetric prediction intervals, which are useful when the errors
    are not symmetrically distributed.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16], dropout_rate=0.2, 
                 use_batch_norm=True, activation='leaky_relu',
                 final_activation='softplus', l2_lambda=1e-5, min_score=0.05,
                 include_prediction=True):
        """
        Initialize asymmetric scoring function
        
        Args:
            input_dim: Input dimension (feature dimension)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            activation: Activation function to use
            final_activation: Final activation function
            l2_lambda: L2 regularization strength
            min_score: Minimum score to prevent degenerate intervals
            include_prediction: Whether to include base model prediction as input
        """
        # Output dimension is 2 for asymmetric intervals (lower, upper)
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=2,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            activation=activation,
            final_activation=final_activation,
            l2_lambda=l2_lambda,
            min_score=min_score,
            include_prediction=include_prediction
        )

class ResidualScoringMLP(LearnableScoringMLP):
    """
    MLP that learns to predict nonconformity scores from both features and residuals
    
    This combines the strengths of traditional residual-based scoring and
    feature-based scoring for more accurate uncertainty estimation.
    """
    def __init__(self, feature_dim, hidden_dims=[128, 64, 32, 16], output_dim=1, 
                 dropout_rate=0.2, use_batch_norm=True, activation='leaky_relu',
                 final_activation='softplus', l2_lambda=1e-5, min_score=0.05,
                 include_prediction=True):
        """
        Initialize residual-based scoring function
        
        Args:
            feature_dim: Feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for symmetric, 2 for asymmetric)
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            activation: Activation function to use
            final_activation: Final activation function
            l2_lambda: L2 regularization strength
            min_score: Minimum score to prevent degenerate intervals
            include_prediction: Whether to include base model prediction as input
        """
        # Input includes features, residual, and optionally prediction
        input_dim = feature_dim + 1  # +1 for residual
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            activation=activation,
            final_activation=final_activation,
            l2_lambda=l2_lambda,
            min_score=min_score,
            include_prediction=include_prediction
        )
    
    def score(self, inputs, targets=None, predictions=None):
        """
        Compute learned nonconformity scores based on features and residuals
        
        Args:
            inputs: Input features
            targets: True target values 
            predictions: Model predictions
            
        Returns:
            scores: Learned nonconformity scores
        """
        if targets is None or predictions is None:
            raise ValueError("Both targets and predictions must be provided for ResidualScoringMLP")
        
        # Compute residuals
        residuals = torch.abs(targets - predictions)
        
        # Concatenate features and residuals
        combined_input = torch.cat([inputs, residuals.unsqueeze(1)], dim=1)
        
        # Score using the combined input
        with torch.no_grad():
            scores = self.forward(combined_input, predictions)
        
        return scores

def get_scoring_function(config, feature_dim=None):
    """
    Factory function to create a scoring function based on configuration
    
    Args:
        config: Configuration dictionary
        feature_dim: Feature dimension (required for learnable scorers)
        
    Returns:
        scorer: Initialized scoring function
    """
    scorer_type = config['scoring_functions']['type']
    symmetric = config['scoring_functions']['symmetric']
    include_prediction = config.get('scoring_functions', {}).get('include_prediction', True)
    
    if scorer_type == 'abs_residual':
        return AbsoluteErrorScorer()
    
    elif scorer_type == 'squared_residual':
        return SquaredErrorScorer()
    
    elif scorer_type == 'normalized_residual':
        return NormalizedErrorScorer()
    
    elif scorer_type == 'learnable_mlp':
        if feature_dim is None:
            raise ValueError("feature_dim must be provided for learnable scorers")
        
        # Get MLP configuration
        mlp_config = config['mlp']
        hidden_dims = mlp_config['hidden_dims']
        dropout_rate = mlp_config['dropout_rate']
        use_batch_norm = mlp_config.get('batch_norm', True)
        activation = mlp_config.get('activation', 'leaky_relu')
        final_activation = mlp_config.get('final_activation', 'softplus')
        l2_lambda = mlp_config.get('l2_lambda', 1e-5)
        min_score = mlp_config.get('min_score', 0.05)
        
        if symmetric:
            return LearnableScoringMLP(
                input_dim=feature_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                activation=activation,
                final_activation=final_activation,
                l2_lambda=l2_lambda,
                min_score=min_score,
                include_prediction=include_prediction
            )
        else:
            return AsymmetricScoringMLP(
                input_dim=feature_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                activation=activation,
                final_activation=final_activation,
                l2_lambda=l2_lambda,
                min_score=min_score,
                include_prediction=include_prediction
            )
    
    elif scorer_type == 'residual_mlp':
        if feature_dim is None:
            raise ValueError("feature_dim must be provided for learnable scorers")
        
        # Get MLP configuration
        mlp_config = config['mlp']
        hidden_dims = mlp_config['hidden_dims']
        dropout_rate = mlp_config['dropout_rate']
        use_batch_norm = mlp_config.get('batch_norm', True)
        activation = mlp_config.get('activation', 'leaky_relu')
        final_activation = mlp_config.get('final_activation', 'softplus')
        l2_lambda = mlp_config.get('l2_lambda', 1e-5)
        min_score = mlp_config.get('min_score', 0.05)
        
        output_dim = 1 if symmetric else 2
        
        return ResidualScoringMLP(
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            activation=activation,
            final_activation=final_activation,
            l2_lambda=l2_lambda,
            min_score=min_score,
            include_prediction=include_prediction
        )
    
    else:
        raise ValueError(f"Unsupported scoring function: {scorer_type}") 