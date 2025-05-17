import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class PredictorConfig:
    """Configuration for predictor architecture."""
    # Dimensions
    input_dim: int = 384
    hidden_dim: int = 192
    output_dim: int = 384
    # Architecture
    num_layers: int = 2
    use_bias: bool = True
    use_bn: bool = True
    dropout: float = 0.0
    # Activation
    activation: str = "gelu"
    final_activation: Optional[str] = None
    # Memory optimization
    fused_operations: bool = True


class MLP_Predictor(nn.Module):
    """
    Memory-efficient MLP predictor for VJEPA model.
    Maps from context encoder outputs to target encoder predictions.
    """
    
    def __init__(self, config: Optional[PredictorConfig] = None):
        """
        Initialize MLP predictor.
        
        Args:
            config: Predictor configuration
        """
        super().__init__()
        
        # Use default config if not provided
        self.config = config or PredictorConfig()
        
        # Extract configuration
        input_dim = self.config.input_dim
        hidden_dim = self.config.hidden_dim
        output_dim = self.config.output_dim
        num_layers = self.config.num_layers
        use_bias = self.config.use_bias
        use_bn = self.config.use_bn
        dropout = self.config.dropout
        activation = self.config.activation
        final_activation = self.config.final_activation
        
        # Create layers
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        # Build MLP layers
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i+1], bias=use_bias))
            
            # Apply batch norm if configured (except for final layer)
            if use_bn and i < len(dims) - 2:
                # Use 1D batch norm for efficiency
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            # Apply activation (except for final layer if not specified)
            is_final_layer = i == len(dims) - 2
            
            if not is_final_layer or final_activation is not None:
                # Choose activation function
                if is_final_layer and final_activation is not None:
                    act_type = final_activation
                else:
                    act_type = activation
                
                # Add appropriate activation
                if act_type == "gelu":
                    layers.append(nn.GELU())
                elif act_type == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif act_type == "silu" or act_type == "swish":
                    layers.append(nn.SiLU())
                elif act_type == "tanh":
                    layers.append(nn.Tanh())
            
            # Add dropout if configured (except for final layer)
            if dropout > 0 and not is_final_layer:
                layers.append(nn.Dropout(dropout))
        
        # Create sequential model
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for predictor."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for better training stability
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, N, D] or [B, T, N, D]
            
        Returns:
            Predicted features with same shape as input
        """
        # Handle different input shapes
        input_shape = x.shape
        
        if len(input_shape) == 4:
            # Video format [B, T, N, D]
            B, T, N, D = input_shape
            
            # Reshape to [B*T*N, D]
            x_flat = x.reshape(-1, D)
            
            # Forward through MLP
            out_flat = self.mlp(x_flat)
            
            # Reshape back to [B, T, N, D']
            out = out_flat.reshape(B, T, N, -1)
        else:
            # Image format [B, N, D]
            B, N, D = input_shape
            
            # The batch norm layer expects [B, C, ...] format for proper statistics
            # So we need to handle batch norm specially if present
            has_bn = any(isinstance(m, nn.BatchNorm1d) for m in self.mlp)
            
            if has_bn:
                # Reshape to [B*N, D]
                x_flat = x.reshape(-1, D)
                
                # Forward through MLP
                out_flat = self.mlp(x_flat)
                
                # Reshape back to [B, N, D']
                out = out_flat.reshape(B, N, -1)
            else:
                # If no BN, we can process each token in parallel
                out = self.mlp(x)
        
        return out


class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor with self-attention.
    More powerful than MLP but requires more memory.
    """
    
    def __init__(self, config: Optional[PredictorConfig] = None):
        """
        Initialize transformer predictor.
        
        Args:
            config: Predictor configuration
        """
        super().__init__()
        
        # Use default config if not provided
        self.config = config or PredictorConfig()
        
        # Extract configuration
        input_dim = self.config.input_dim
        hidden_dim = self.config.hidden_dim
        output_dim = self.config.output_dim
        
        # Input projection (if dimensions differ)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Transformer layer for token interactions
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,  # Use fewer heads for memory efficiency
            dim_feedforward=hidden_dim * 2,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for predictor."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, N, D] or [B, T, N, D]
            
        Returns:
            Predicted features with same shape as input
        """
        # Handle different input shapes
        input_shape = x.shape
        
        if len(input_shape) == 4:
            # Video format [B, T, N, D]
            B, T, N, D = input_shape
            
            # Reshape to [B*T, N, D]
            x = x.reshape(B*T, N, D)
            
            # Input projection
            x = self.input_proj(x)
            
            # Apply transformer layer
            x = self.transformer_layer(x)
            
            # Output projection
            x = self.output_proj(x)
            
            # Reshape back to [B, T, N, D']
            x = x.reshape(B, T, N, -1)
        else:
            # Image format [B, N, D]
            # Input projection
            x = self.input_proj(x)
            
            # Apply transformer layer
            x = self.transformer_layer(x)
            
            # Output projection
            x = self.output_proj(x)
        
        return x


def create_predictor(config: Optional[PredictorConfig] = None) -> nn.Module:
    """
    Create predictor module based on configuration.
    
    Args:
        config: Predictor configuration
        
    Returns:
        Predictor module
    """
    # Use default config if not provided
    config = config or PredictorConfig()
    
    # Determine predictor type from number of layers
    if config.num_layers <= 0:
        # Use transformer predictor for more powerful predictions
        return TransformerPredictor(config)
    else:
        # Use MLP predictor for efficiency
        return MLP_Predictor(config)