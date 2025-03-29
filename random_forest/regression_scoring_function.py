import torch
import torch.nn as nn
import numpy as np

class RegressionScoringMLP(nn.Module):
    """
    MLP that takes |Ground Truth - Prediction| as input and outputs symmetric uncertainty bounds
    """
    def __init__(self, input_dim=1, hidden_dims=[64, 32], output_dim=1, dropout_rate=0.2):
        """
        Initialize regression scoring function
        
        Args:
            input_dim: Input dimension (default: 1 for absolute error)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (default: 1 for interval width)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        # Initialize weights function
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        # Use LeakyReLU for hidden layers
        activation = nn.LeakyReLU(0.1)
        
        # Hidden layers with dropout
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(activation)
            
            if i < len(hidden_dims) - 1:  # No dropout after last hidden layer
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = dim
        
        # Final layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Add softplus to ensure positive output (uncertainty width must be positive)
        layers.append(nn.Softplus(beta=1))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
        
        # Apply initialization
        for m in self.network:
            if isinstance(m, nn.Linear):
                init_weights(m)
        
        # Regularization
        self.l2_lambda = 1e-5
        
    def forward(self, x):
        """
        Forward pass to predict uncertainty width
        
        Args:
            x: Absolute error |y_true - y_pred| with shape [batch_size, 1]
        
        Returns:
            width: Estimated uncertainty width for symmetric intervals
        """
        width = self.network(x)
        
        # Add L2 regularization
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
        self.l2_reg = self.l2_lambda * l2_reg
        
        # Ensure reasonable width values (min width to prevent too narrow intervals)
        min_width = 0.1  # Minimum width to prevent degenerate intervals
        width = torch.clamp(width, min=min_width)
        
        return width 