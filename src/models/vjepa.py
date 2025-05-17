import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass

from .encoders import EncoderConfig, create_encoder, MemoryEfficientViT
from .predictors import PredictorConfig, create_predictor
from ..utils.memory import MemoryOptimizer


@dataclass
class VJEPAConfig:
    """Configuration for VJEPA model."""
    # General model parameters
    encoder_config: Optional[EncoderConfig] = None
    predictor_config: Optional[PredictorConfig] = None
    use_momentum_encoder: bool = True
    momentum: float = 0.99
    # Loss parameters
    use_cosine_loss: bool = False
    use_mask_weighting: bool = True
    # Optimization parameters
    target_update_interval: int = 1  # How often to update target encoder
    stop_gradient: bool = True  # Whether to stop gradients from target encoder
    # Memory optimization
    share_parameters: bool = False  # Whether to share encoder parameters


class VJEPA(nn.Module):
    """
    Video Joint Embedding Predictive Architecture.
    Memory-efficient implementation with M1 optimizations.
    """
    
    def __init__(self, config: Optional[VJEPAConfig] = None):
        """
        Initialize VJEPA model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        # Use default config if not provided
        self.config = config or VJEPAConfig()
        
        # Create encoder config if not provided
        if self.config.encoder_config is None:
            self.config.encoder_config = EncoderConfig()
        
        # Create predictor config if not provided
        if self.config.predictor_config is None:
            self.config.predictor_config = PredictorConfig(
                input_dim=self.config.encoder_config.embed_dim,
                hidden_dim=self.config.encoder_config.embed_dim // 2,
                output_dim=self.config.encoder_config.embed_dim,
            )
        
        # Create context encoder (student)
        self.context_encoder = create_encoder(self.config.encoder_config)
        
        # Create target encoder (teacher) - with parameter sharing if configured
        if self.config.share_parameters:
            # Share parameters but still maintain separate forward passes
            self.target_encoder = self.context_encoder
        else:
            # Create separate target encoder
            self.target_encoder = create_encoder(self.config.encoder_config)
            
            # Disable gradient for target encoder
            if self.config.stop_gradient:
                for param in self.target_encoder.parameters():
                    param.requires_grad = False
            
            # Initialize target encoder with same weights as context encoder
            self._copy_weights(self.context_encoder, self.target_encoder)
        
        # Create predictor
        self.predictor = create_predictor(self.config.predictor_config)
        
        # Initialize weights
        self._init_weights()
        
        # Track update steps for target update interval
        self.register_buffer('update_counter', torch.tensor(0))
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize predictor
        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _copy_weights(self, src_model: nn.Module, tgt_model: nn.Module):
        """
        Copy weights from source to target model.
        
        Args:
            src_model: Source model
            tgt_model: Target model
        """
        for tgt_param, src_param in zip(tgt_model.parameters(), src_model.parameters()):
            tgt_param.data.copy_(src_param.data)
    
    def _update_target_encoder(self):
        """
        Update target encoder using momentum update.
        Target = momentum * target + (1 - momentum) * context
        """
        # Return if we're sharing parameters or not using momentum
        if self.config.share_parameters or not self.config.use_momentum_encoder:
            return
        
        # Check if it's time to update
        if self.update_counter % self.config.target_update_interval != 0:
            return
        
        momentum = self.config.momentum
        
        # Update target encoder weights
        for tgt_param, src_param in zip(self.target_encoder.parameters(), 
                                       self.context_encoder.parameters()):
            tgt_param.data = momentum * tgt_param.data + (1 - momentum) * src_param.data
    
    def forward(self, 
               video: torch.Tensor, 
               masked_video: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            video: Original video tensor [B, C, T, H, W]
            masked_video: Masked video tensor [B, C, T, H, W]
            mask: Binary mask indicating masked tokens [B, N] or [B, T*N]
            
        Returns:
            Dictionary of model outputs
        """
        # If masked_video not provided, assume video is already masked
        if masked_video is None:
            masked_video = video
        
        # Get target features (no grad)
        with torch.no_grad() if self.config.stop_gradient else torch.enable_grad():
            # Pass through target encoder
            target_output = self.target_encoder(video)
        
        # Get context features
        context_output = self.context_encoder(masked_video)
        
        # Apply predictor to get predictions
        predictions = self.predictor(context_output)
        
        # Update target encoder
        self.update_counter += 1
        self._update_target_encoder()
        
        # Prepare outputs
        outputs = {
            "predictions": predictions,
            "targets": target_output,
            "mask": mask,
        }
        
        # Compute loss if all inputs are provided
        if mask is not None:
            loss = self.compute_loss(predictions, target_output, mask)
            outputs["loss"] = loss
        
        return outputs
    
    def compute_loss(self, 
                    predictions: torch.Tensor, 
                    targets: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Args:
            predictions: Predicted features [B, T, N, D] or [B, N, D]
            targets: Target features [B, T, N, D] or [B, N, D]
            mask: Binary mask indicating masked tokens (1=keep, 0=mask) [B, N] or [B, T*N]
            
        Returns:
            Loss tensor
        """
        # Check dimensions and reshape if needed
        if len(predictions.shape) == 4 and len(targets.shape) == 4:
            # Video format [B, T, N, D]
            B, T, N, D = predictions.shape
            
            # Reshape to [B, T*N, D]
            predictions = predictions.reshape(B, T*N, D)
            targets = targets.reshape(B, T*N, D)
        
        # Choose loss function
        if self.config.use_cosine_loss:
            # Cosine similarity loss
            # Normalize features
            predictions = F.normalize(predictions, dim=-1)
            targets = F.normalize(targets, dim=-1)
            
            # Compute cosine similarity
            similarity = torch.sum(predictions * targets, dim=-1)  # [B, N]
            
            # Convert to loss (1 - similarity)
            loss = 1.0 - similarity  # [B, N]
        else:
            # Default to MSE loss
            loss = F.mse_loss(predictions, targets, reduction='none')  # [B, N, D]
            loss = loss.mean(dim=-1)  # [B, N]
        
        # Apply mask weighting if provided
        if mask is not None and self.config.use_mask_weighting:
            # Ensure mask has same shape as loss
            if mask.shape != loss.shape:
                mask = mask.reshape(loss.shape)
            
            # Apply mask weighting (focus on masked tokens)
            inverse_mask = 1.0 - mask  # Invert to focus on masked tokens
            
            # Normalize mask
            mask_sum = inverse_mask.sum()
            if mask_sum > 0:
                loss = (loss * inverse_mask).sum() / mask_sum
            else:
                loss = loss.mean()
        else:
            # No mask weighting
            loss = loss.mean()
        
        return loss
    
    def encode(self, x: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """
        Encode input using specified encoder.
        
        Args:
            x: Input tensor
            use_target: Whether to use target encoder
            
        Returns:
            Encoded features
        """
        if use_target:
            with torch.no_grad():
                return self.target_encoder(x)
        else:
            return self.context_encoder(x)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension of model."""
        return self.config.encoder_config.embed_dim


# Memory-efficient inference version
class VJEPAInference(nn.Module):
    """
    Memory-efficient VJEPA model for inference.
    Uses only the target encoder for feature extraction.
    """
    
    def __init__(self, model: VJEPA):
        """
        Initialize inference model from trained VJEPA.
        
        Args:
            model: Trained VJEPA model
        """
        super().__init__()
        
        # Extract and optimize target encoder
        self.encoder = MemoryOptimizer.optimize_model_for_inference(
            model.target_encoder
        )
        
        # Get embedding dimension
        self.embed_dim = model.get_embedding_dim()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded features
        """
        return self.encoder(x)
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'VJEPAInference':
        """
        Create inference model from pretrained checkpoint.
        
        Args:
            path: Path to pretrained model
            
        Returns:
            Inference model
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create config from checkpoint
        encoder_config = EncoderConfig()
        if 'encoder_config' in checkpoint:
            # Update encoder config with saved values
            for k, v in checkpoint['encoder_config'].items():
                if hasattr(encoder_config, k):
                    setattr(encoder_config, k, v)
        
        # Create full model
        model_config = VJEPAConfig(encoder_config=encoder_config)
        model = VJEPA(model_config)
        
        # Load weights
        model.load_state_dict(checkpoint['model'], strict=False)
        
        # Create inference model
        return cls(model)


def create_vjepa_model(config: Optional[VJEPAConfig] = None) -> VJEPA:
    """
    Create VJEPA model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        VJEPA model
    """
    return VJEPA(config)