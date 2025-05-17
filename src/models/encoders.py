import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from einops import rearrange, repeat
from functools import partial

from ..utils.memory import MemoryOptimizer


@dataclass
class EncoderConfig:
    """Configuration for video encoder architectures."""
    # General parameters
    model_type: str = "vit"  # "vit", "resnet", or "efficient"
    img_size: int = 128
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    norm_layer: str = "layer_norm"  # "layer_norm" or "batch_norm"
    # Memory optimizations
    use_half_precision: bool = True
    use_gradient_checkpointing: bool = True
    # Temporal parameters
    use_temporal_attention: bool = True
    temporal_depth: int = 2
    # Performance optimizations
    fused_matmul: bool = True
    use_flash_attention: bool = False


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding with memory optimizations.
    """
    
    def __init__(self, 
                img_size: int = 128, 
                patch_size: int = 16, 
                in_channels: int = 3, 
                embed_dim: int = 384):
        """
        Initialize patch embedding.
        
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # Use conv2d for patch embedding - faster and more memory efficient
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Patch tokens [B, N, D]
        """
        B, C, H, W = x.shape
        
        # Handle non-standard input sizes
        if H != self.img_size or W != self.img_size:
            # Pad or interpolate to desired size
            x = F.interpolate(x, size=(self.img_size, self.img_size), 
                             mode='bilinear', align_corners=False)
        
        # Efficient patch embedding
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        
        return x


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient implementation of self-attention with optional flash attention.
    """
    
    def __init__(self, 
                dim: int, 
                num_heads: int = 8, 
                qkv_bias: bool = True,
                attn_drop: float = 0.0, 
                proj_drop: float = 0.0,
                use_flash_attn: bool = False):
        """
        Initialize attention module.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
            use_flash_attn: Whether to use flash attention
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.use_flash_attn = use_flash_attn and self._is_flash_attn_available()
    
    def _is_flash_attn_available(self) -> bool:
        """Check if flash attention is available."""
        try:
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            return False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, N, D]
            
        Returns:
            Output tensor [B, N, D]
        """
        B, N, D = x.shape
        
        # Get QKV projections
        qkv = self.qkv(x)  # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, D/H]
        
        # Use flash attention if available
        if self.use_flash_attn and self._is_flash_attn_available():
            from flash_attn import flash_attn_func
            
            # Reshape for flash attention
            q = q.transpose(1, 2)  # [B, N, H, D/H]
            k = k.transpose(1, 2)  # [B, N, H, D/H]
            v = v.transpose(1, 2)  # [B, N, H, D/H]
            
            # Apply flash attention
            output = flash_attn_func(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            
            # Reshape output
            output = output.reshape(B, N, D)
        else:
            # Standard scaled dot-product attention
            # Efficient implementation with proper reshaping
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            # Attention output
            output = (attn @ v).transpose(1, 2).reshape(B, N, D)
        
        # Output projection
        output = self.proj(output)
        output = self.proj_drop(output)
        
        return output


class MemoryEfficientMLP(nn.Module):
    """Memory-efficient MLP with fused operations."""
    
    def __init__(self, 
                in_features: int, 
                hidden_features: Optional[int] = None, 
                out_features: Optional[int] = None,
                activation: str = "gelu",
                drop: float = 0.0,
                fused: bool = True):
        """
        Initialize MLP.
        
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden feature dimension
            out_features: Output feature dimension
            activation: Activation function
            drop: Dropout rate
            fused: Whether to use fused operations
        """
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # Activation function
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "silu" or activation == "swish":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Option to use 1-step or 2-step MLP
        if fused and hasattr(nn.Linear, "reset_parameters_fused"):
            # Fused MLP implementation
            self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
            self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
            
            # Use fused reset for faster operation
            self.fc1.reset_parameters_fused()
            self.fc2.reset_parameters_fused()
        else:
            # Standard implementation
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MemoryEfficientBlock(nn.Module):
    """Memory-efficient transformer block with gradient checkpointing."""
    
    def __init__(self,
                dim: int,
                num_heads: int,
                mlp_ratio: float = 4.0,
                qkv_bias: bool = True,
                drop: float = 0.0,
                attn_drop: float = 0.0,
                norm_layer: Optional[nn.Module] = None,
                use_gradient_checkpointing: bool = False,
                use_flash_attention: bool = False,
                fused_mlp: bool = True):
        """
        Initialize transformer block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in QKV projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            norm_layer: Normalization layer
            use_gradient_checkpointing: Whether to use gradient checkpointing
            use_flash_attention: Whether to use flash attention
            fused_mlp: Whether to use fused MLP operations
        """
        super().__init__()
        
        # Normalization layer
        norm_layer = norm_layer or nn.LayerNorm
        
        # First normalization and attention
        self.norm1 = norm_layer(dim)
        self.attn = MemoryEfficientAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_flash_attn=use_flash_attention
        )
        
        # Second normalization and MLP
        self.norm2 = norm_layer(dim)
        self.mlp = MemoryEfficientMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
            fused=fused_mlp
        )
        
        # Gradient checkpointing flag
        self.use_gradient_checkpointing = use_gradient_checkpointing
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Use gradient checkpointing for memory efficiency
        if self.use_gradient_checkpointing and self.training:
            # Define custom forward functions for checkpointing
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            # Apply gradient checkpointing
            x = x + torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda current_x: self.attn(self.norm1(current_x))),
                x,
                use_reentrant=False # Explicitly set as recommended
            )
            
            x = x + torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda current_x: self.mlp(self.norm2(current_x))),
                x,
                use_reentrant=False # Explicitly set as recommended
            )
        else:
            # Standard forward pass
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        
        return x


class MemoryEfficientViT(nn.Module):
    """
    Memory-efficient Vision Transformer with M1-optimized implementation.
    """
    
    def __init__(self, config: Optional[EncoderConfig] = None):
        """
        Initialize ViT model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        # Use default config if not provided
        self.config = config or EncoderConfig()
        
        # Extract config parameters
        img_size = self.config.img_size
        patch_size = self.config.patch_size
        in_channels = self.config.in_channels
        embed_dim = self.config.embed_dim
        depth = self.config.depth
        num_heads = self.config.num_heads
        mlp_ratio = self.config.mlp_ratio
        qkv_bias = self.config.qkv_bias
        drop_rate = self.config.drop_rate
        attn_drop_rate = self.config.attn_drop_rate
        use_gradient_checkpointing = self.config.use_gradient_checkpointing
        
        # Determine normalization layer
        if self.config.norm_layer == "layer_norm":
            norm_layer = nn.LayerNorm
        elif self.config.norm_layer == "batch_norm":
            # BatchNorm1d wrapper for compatibility
            norm_layer = lambda dim: nn.BatchNorm1d(dim)
        else:
            norm_layer = nn.LayerNorm
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            MemoryEfficientBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_flash_attention=self.config.use_flash_attention,
                fused_mlp=self.config.fused_matmul
            ) for _ in range(depth)
        ])
        
        # Final normalization
        self.norm = norm_layer(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Apply weight initialization to all modules
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Feature tensor [B, N, D]
        """
        # Get patch embeddings
        x = self.patch_embed(x)  # [B, N, D]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Forward through transformer blocks
        x = self.blocks(x)
        
        # Apply final norm
        x = self.norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W] or [B, C, T, H, W]
            
        Returns:
            Output tensor [B, N, D] or [B, T, N, D]
        """
        # Handle video input [B, C, T, H, W]
        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
            
            # Process each frame independently
            features = []
            for t in range(T):
                # Extract frame
                frame = x[:, :, t]  # [B, C, H, W]
                
                # Process frame
                frame_features = self.forward_features(frame)  # [B, N, D]
                features.append(frame_features)
            
            # Stack along new time dimension
            x = torch.stack(features, dim=1)  # [B, T, N, D]
            
            # Apply temporal attention if configured
            if self.config.use_temporal_attention:
                x = self.forward_temporal_attention(x)
            
            return x
        
        # Standard image input [B, C, H, W]
        return self.forward_features(x)
    
    def forward_temporal_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention to video features.
        
        Args:
            x: Video features [B, T, N, D]
            
        Returns:
            Temporally processed features [B, T, N, D]
        """
        # Implement this if needed based on config
        return x


# Function to create encoder based on config
def create_encoder(config: Optional[EncoderConfig] = None) -> nn.Module:
    """
    Create encoder model based on configuration.
    
    Args:
        config: Encoder configuration
        
    Returns:
        Encoder model
    """
    # Use default config if not provided
    config = config or EncoderConfig()
    
    # Create appropriate model based on type
    if config.model_type == "vit":
        return MemoryEfficientViT(config)
    else:
        # Default to ViT
        return MemoryEfficientViT(config)


# Memory-efficient adapter for inference optimization
def optimize_encoder_for_inference(encoder: nn.Module) -> nn.Module:
    """
    Apply memory optimizations for inference.
    
    Args:
        encoder: Encoder model
        
    Returns:
        Optimized encoder model
    """
    # Apply memory optimizer
    encoder = MemoryOptimizer.optimize_model_for_inference(encoder)
    
    # Disable gradient checkpointing for inference
    for module in encoder.modules():
        if hasattr(module, 'use_gradient_checkpointing'):
            module.use_gradient_checkpointing = False
    
    return encoder