import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass


@dataclass
class MaskingConfig:
    """Configuration for video masking strategies."""
    strategy: str = "tube"  # 'tube', 'block', 'random'
    mask_ratio: float = 0.75  # Ratio of tokens to mask
    temporal_window: int = 2  # For temporal tube masking
    block_size: int = 2  # For block masking
    mask_on_token: bool = True  # If True, mask on token level, else on patch level
    shared_masking: bool = True  # If True, use same mask for all samples in batch


class VideoMasker:
    """
    Implements various masking strategies for video tokens in self-supervised learning.
    """
    
    def __init__(self, config: Optional[MaskingConfig] = None):
        """
        Initialize masking with configuration.
        
        Args:
            config: Masking configuration
        """
        self.config = config or MaskingConfig()
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking to video tensor.
        
        Args:
            x: Video tensor [B, C, T, H, W] or tokens [B, N, D]
            
        Returns:
            Tuple of (masked_x, mask) where mask is binary mask of 1s (keep) and 0s (mask)
        """
        # Choose masking strategy based on config
        if self.config.strategy == "tube":
            return self.apply_tube_masking(x)
        elif self.config.strategy == "block":
            return self.apply_block_masking(x)
        elif self.config.strategy == "random":
            return self.apply_random_masking(x)
        else:
            raise ValueError(f"Unknown masking strategy: {self.config.strategy}")
    
    def apply_random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random token masking.
        
        Args:
            x: Input tensor [B, N, D] for tokens or [B, C, T, H, W] for raw video
            
        Returns:
            Tuple of (masked_x, mask)
        """
        # Convert to tokens if needed
        is_video = len(x.shape) == 5
        if is_video:
            # Assume x is [B, C, T, H, W]
            # For simplicity, flatten spatial dims and handle as tokens
            B, C, T, H, W = x.shape
            x_flat = x.flatten(3).transpose(2, 3)  # [B, C, T*H*W/P^2, P^2]
            x_tokens = x_flat.flatten(2)  # [B, C, N]
            x_tokens = x_tokens.transpose(1, 2)  # [B, N, C]
        else:
            # Already token format [B, N, D]
            x_tokens = x
        
        B, N, D = x_tokens.shape
        
        # Determine number of tokens to keep
        keep_num = int(N * (1 - self.config.mask_ratio))
        
        # Generate masks - shared or per-sample
        if self.config.shared_masking:
            # Same mask for all samples in batch
            noise = torch.rand(1, N, device=x.device)
            noise = noise.expand(B, -1)  # [B, N]
        else:
            # Different mask for each sample
            noise = torch.rand(B, N, device=x.device)  # [B, N]
        
        # Get keep indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :keep_num]
        
        # Create binary mask (1=keep, 0=mask)
        mask = torch.zeros(B, N, device=x.device)
        mask.scatter_(1, ids_keep, 1)
        
        # Apply mask to tokens
        x_masked = x_tokens.clone()
        
        # For tokens we're masking, replace with zeros or mask token
        # Here using zeros as mask representation
        mask_token_value = torch.zeros(D, device=x.device)
        for i in range(B):
            x_masked[i, mask[i] == 0] = mask_token_value
        
        # Convert back to original format if needed
        if is_video:
            x_masked = x_masked.transpose(1, 2)  # [B, C, N]
            x_masked = x_masked.reshape(B, C, T, H, W)
        
        return x_masked, mask
    
    def apply_tube_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply tube masking - consistent across temporal dimension.
        
        Args:
            x: Input tensor [B, N, D] for tokens or [B, C, T, H, W] for raw video
            
        Returns:
            Tuple of (masked_x, mask)
        """
        # Convert to tokens if needed
        is_video = len(x.shape) == 5
        if is_video:
            B, C, T, H, W = x.shape
            
            # For tube masking, we need spatial dimensions
            spatial_tokens = H * W
            tokens_per_frame = spatial_tokens
            
            # Create mask for spatial tokens
            keep_num = int(tokens_per_frame * (1 - self.config.mask_ratio))
            
            # Generate spatial noise (same for all temporal frames)
            if self.config.shared_masking:
                # Same mask for all samples in batch
                noise = torch.rand(1, tokens_per_frame, device=x.device)
                noise = noise.expand(B, -1)  # [B, H*W]
            else:
                # Different mask for each sample
                noise = torch.rand(B, tokens_per_frame, device=x.device)  # [B, H*W]
                
            # Get keep indices for spatial dimension
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :keep_num]
            
            # Create spatial binary mask (1=keep, 0=mask)
            spatial_mask = torch.zeros(B, tokens_per_frame, device=x.device)
            spatial_mask.scatter_(1, ids_keep, 1)
            
            # Expand mask temporally
            mask = spatial_mask.unsqueeze(1).expand(-1, T, -1).reshape(B, T*tokens_per_frame)
            
            # Apply mask to video
            x_masked = x.clone()
            
            # Apply spatial masking for each frame
            for b in range(B):
                for t in range(T):
                    # Reshape mask for this frame
                    frame_mask = spatial_mask[b].reshape(H, W)
                    
                    # Zero out masked regions
                    x_masked[b, :, t, frame_mask == 0] = 0
            
            # Return masked video and flattened mask
            return x_masked, mask
        else:
            # For token representation, we need to know T, H, W to apply tube masking
            # For simplicity, assuming tokens are arranged as [B, T*H*W, D]
            B, N, D = x.shape
            
            # Need to infer temporal and spatial dimensions - using config or guessing
            T = self.config.temporal_window
            spatial_tokens = N // T
            
            # Create spatial mask
            keep_num = int(spatial_tokens * (1 - self.config.mask_ratio))
            
            # Generate spatial noise
            if self.config.shared_masking:
                noise = torch.rand(1, spatial_tokens, device=x.device).expand(B, -1)
            else:
                noise = torch.rand(B, spatial_tokens, device=x.device)
            
            # Get keep indices
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :keep_num]
            
            # Create spatial binary mask
            spatial_mask = torch.zeros(B, spatial_tokens, device=x.device)
            spatial_mask.scatter_(1, ids_keep, 1)
            
            # Repeat mask temporally
            mask = torch.repeat_interleave(spatial_mask, T, dim=1)
            
            # Apply mask to tokens
            x_masked = x.clone()
            
            # For tokens we're masking, replace with zeros or mask token
            mask_token_value = torch.zeros(D, device=x.device)
            
            for i in range(B):
                x_masked[i, mask[i] == 0] = mask_token_value
            
            return x_masked, mask
    
    def apply_block_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply block masking - mask rectangular spatial-temporal blocks.
        
        Args:
            x: Input tensor [B, N, D] for tokens or [B, C, T, H, W] for raw video
            
        Returns:
            Tuple of (masked_x, mask)
        """
        # This is more complex for arbitrary token arrangements
        # For now, focusing on video tensor format
        is_video = len(x.shape) == 5
        if not is_video:
            # Fall back to random masking for token format
            # In a full implementation, you'd need to know the T,H,W arrangement
            return self.apply_random_masking(x)
        
        B, C, T, H, W = x.shape
        block_size = self.config.block_size
        
        # Calculate number of blocks
        t_blocks = max(1, T // block_size)
        h_blocks = max(1, H // block_size)
        w_blocks = max(1, W // block_size)
        total_blocks = t_blocks * h_blocks * w_blocks
        
        # Determine number of blocks to keep
        keep_blocks = int(total_blocks * (1 - self.config.mask_ratio))
        
        # Initialize mask (1=keep, 0=mask)
        mask = torch.ones((B, T, H, W), device=x.device)
        
        # For each batch
        for b in range(B):
            # Generate random mask for blocks
            if self.config.shared_masking and b > 0:
                # Reuse mask from first sample
                mask[b] = mask[0]
                continue
                
            # Random permutation of blocks
            block_indices = torch.randperm(total_blocks, device=x.device)
            mask_indices = block_indices[keep_blocks:]  # Indices of blocks to mask
            
            # Apply block masking
            for idx in mask_indices:
                # Convert flat index to t,h,w block coordinates
                t_idx = (idx // (h_blocks * w_blocks)) * block_size
                h_idx = ((idx % (h_blocks * w_blocks)) // w_blocks) * block_size
                w_idx = ((idx % (h_blocks * w_blocks)) % w_blocks) * block_size
                
                # Set mask to 0 for this block
                t_end = min(t_idx + block_size, T)
                h_end = min(h_idx + block_size, H)
                w_end = min(w_idx + block_size, W)
                
                mask[b, t_idx:t_end, h_idx:h_end, w_idx:w_end] = 0
        
        # Apply mask to video
        x_masked = x.clone()
        for b in range(B):
            x_masked[b, :, mask[b] == 0] = 0
        
        # Flatten mask for return
        flat_mask = mask.reshape(B, -1)
        
        return x_masked, flat_mask


def apply_masking(x: torch.Tensor, config: Optional[MaskingConfig] = None) -> torch.Tensor:
    """
    Apply masking to input tensor - convenience function.
    
    Args:
        x: Input tensor
        config: Optional masking configuration
        
    Returns:
        Masked tensor
    """
    masker = VideoMasker(config)
    masked_x, _ = masker(x)
    return masked_x