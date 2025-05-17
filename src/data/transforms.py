import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union
import math
from dataclasses import dataclass
import threading


@dataclass
class TransformConfig:
    """Configuration for video transformations."""
    mode: str = "train"  # "train" or "val"
    size: int = 128
    backend: str = "torchvision"  # "torchvision" or "cv2"
    # Cropping
    random_crop: bool = True
    crop_scale: Tuple[float, float] = (0.8, 1.0)
    # Augmentations
    random_flip: bool = True
    color_jitter: bool = True
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.4
    # Normalization
    normalize: bool = True
    mean: List[float] = (0.485, 0.456, 0.406)
    std: List[float] = (0.229, 0.224, 0.225)
    # Cache computed random parameters for consistency in tube model
    cache_random_params: bool = True


class MemoryEfficientTransforms:
    """
    Memory-efficient video transformations using both CPU and GPU.
    Optimized for M1 Macs with limited memory.
    """
    
    def __init__(self, config: Optional[TransformConfig] = None):
        """
        Initialize transforms with configuration.
        
        Args:
            config: Transform configuration
        """
        self.config = config or TransformConfig()
        
        # Random state cache for consistent transforms across frames
        self.param_cache = {}
        self.cache_lock = threading.Lock()
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply transforms to video tensor.
        
        Args:
            video: Video tensor [T, C, H, W] or [C, T, H, W]
            
        Returns:
            Transformed video tensor
        """
        # Handle different input formats
        input_is_channel_first = video.shape[0] == 3
        
        if input_is_channel_first:  # [C, T, H, W]
            video = video.permute(1, 0, 2, 3)  # -> [T, C, H, W]
        
        # Get video parameters
        T, C, H, W = video.shape
        
        # Choose backend implementation
        if self.config.backend == "torchvision":
            output = self._transform_torchvision(video)
        elif self.config.backend == "cv2":
            output = self._transform_opencv(video)
        else:
            # Custom implementation optimized for memory
            output = self._transform_custom(video)
        
        # Return to original format if needed
        if input_is_channel_first:
            output = output.permute(1, 0, 2, 3)  # [C, T, H, W]
        
        return output
    
    def _transform_torchvision(self, video: torch.Tensor) -> torch.Tensor:
        """Transform using torchvision."""
        import torchvision.transforms as T
        import torchvision.transforms.functional as F
        
        # Check if we should create or reuse random params
        # This ensures consistent transforms across frames
        cache_id = id(video)
        reuse_params = self.config.cache_random_params
        
        with self.cache_lock:
            if reuse_params and cache_id in self.param_cache:
                params = self.param_cache[cache_id]
            else:
                # Determine random params
                params = {}
                
                # Random crop params
                if self.config.random_crop and self.config.mode == "train":
                    size = self.config.size
                    scale = self.config.crop_scale
                    ratio = (3/4, 4/3)  # Default from RandomResizedCrop
                    
                    height, width = video.shape[2], video.shape[3]
                    area = height * width
                    
                    for _ in range(10):  # Try 10 times as in torchvision
                        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
                        log_ratio = torch.log(torch.tensor(ratio))
                        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
                        
                        w = int(round(math.sqrt(target_area * aspect_ratio)))
                        h = int(round(math.sqrt(target_area / aspect_ratio)))
                        
                        if 0 < w <= width and 0 < h <= height:
                            top = torch.randint(0, height - h + 1, (1,)).item()
                            left = torch.randint(0, width - w + 1, (1,)).item()
                            params['crop'] = (top, left, h, w)
                            break
                
                # Random flip
                if self.config.random_flip and self.config.mode == "train":
                    params['flip'] = torch.rand(1).item() < 0.5
                
                # Color jitter
                if self.config.color_jitter and self.config.mode == "train":
                    brightness = self.config.brightness
                    contrast = self.config.contrast
                    saturation = self.config.saturation
                    
                    params['brightness'] = torch.empty(1).uniform_(max(0, 1 - brightness), 1 + brightness).item()
                    params['contrast'] = torch.empty(1).uniform_(max(0, 1 - contrast), 1 + contrast).item()
                    params['saturation'] = torch.empty(1).uniform_(max(0, 1 - saturation), 1 + saturation).item()
                
                if reuse_params:
                    self.param_cache[cache_id] = params
        
        # Process each frame with consistent params
        T, C, H, W = video.shape
        output = torch.zeros(T, C, self.config.size, self.config.size, 
                             dtype=video.dtype, device=video.device)
        
        for t in range(T):
            frame = video[t]  # [C, H, W]
            
            # Apply transforms
            if self.config.mode == "train":
                # Apply crop if params exist
                if 'crop' in params:
                    top, left, h, w = params['crop']
                    frame = F.crop(frame, top, left, h, w)
                
                # Resize if needed
                if frame.shape[1] != self.config.size or frame.shape[2] != self.config.size:
                    frame = F.resize(frame, [self.config.size, self.config.size])
                
                # Apply flip
                if params.get('flip', False):
                    frame = F.hflip(frame)
                
                # Apply color jitter
                if 'brightness' in params:
                    frame = F.adjust_brightness(frame, params['brightness'])
                if 'contrast' in params:
                    frame = F.adjust_contrast(frame, params['contrast'])
                if 'saturation' in params:
                    frame = F.adjust_saturation(frame, params['saturation'])
            else:
                # Validation mode - simple resize and center crop
                if H != self.config.size or W != self.config.size:
                    frame = F.resize(frame, [self.config.size, self.config.size])
            
            # Apply normalization
            if self.config.normalize:
                frame = F.normalize(frame, mean=self.config.mean, std=self.config.std)
            
            output[t] = frame
        
        # Clean up cache to prevent memory leaks
        if reuse_params and len(self.param_cache) > 100:
            with self.cache_lock:
                self.param_cache.clear()
        
        return output
    
    def _transform_opencv(self, video: torch.Tensor) -> torch.Tensor:
        """Transform using OpenCV for memory efficiency."""
        import cv2
        
        # Get video parameters
        T, C, H, W = video.shape
        size = self.config.size
        
        # Create output tensor
        output = torch.zeros(T, C, size, size, dtype=video.dtype, device=video.device)
        
        # Determine random params for consistency across frames
        cache_id = id(video)
        reuse_params = self.config.cache_random_params
        
        with self.cache_lock:
            if reuse_params and cache_id in self.param_cache:
                params = self.param_cache[cache_id]
            else:
                params = {}
                
                # Random crop params
                if self.config.random_crop and self.config.mode == "train":
                    scale = np.random.uniform(self.config.crop_scale[0], self.config.crop_scale[1])
                    crop_size = int(min(H, W) * scale)
                    top = np.random.randint(0, H - crop_size + 1)
                    left = np.random.randint(0, W - crop_size + 1)
                    params['crop'] = (top, left, crop_size)
                
                # Random flip
                if self.config.random_flip and self.config.mode == "train":
                    params['flip'] = np.random.random() > 0.5
                
                # Color jitter
                if self.config.color_jitter and self.config.mode == "train":
                    params['brightness'] = np.random.uniform(
                        1 - self.config.brightness, 1 + self.config.brightness)
                    params['contrast'] = np.random.uniform(
                        1 - self.config.contrast, 1 + self.config.contrast)
                    params['saturation'] = np.random.uniform(
                        1 - self.config.saturation, 1 + self.config.saturation)
                
                if reuse_params:
                    self.param_cache[cache_id] = params
        
        # Process each frame with CPU to save GPU memory
        for t in range(T):
            # Convert to numpy
            frame = video[t].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            
            # Apply transforms
            if self.config.mode == "train":
                # Crop
                if 'crop' in params:
                    top, left, crop_size = params['crop']
                    frame = frame[top:top+crop_size, left:left+crop_size]
                
                # Flip
                if params.get('flip', False):
                    frame = cv2.flip(frame, 1)  # horizontal flip
                
                # Color jitter
                if 'brightness' in params:
                    # Brightness
                    frame = cv2.convertScaleAbs(
                        frame, alpha=params['brightness'], beta=0)
                    
                    # Contrast
                    mean = np.mean(frame, axis=(0, 1), keepdims=True)
                    frame = (frame - mean) * params['contrast'] + mean
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                    
                    # Saturation (simplified)
                    if params['saturation'] != 1:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['saturation'], 0, 255)
                        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Resize to target size
            if frame.shape[0] != size or frame.shape[1] != size:
                frame = cv2.resize(frame, (size, size))
            
            # Normalize
            if self.config.normalize:
                frame = frame.astype(np.float32) / 255.0
                frame -= np.array(self.config.mean)
                frame /= np.array(self.config.std)
            
            # Back to tensor
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
            output[t] = frame_tensor.to(video.device)
        
        # Clean up cache periodically
        if reuse_params and len(self.param_cache) > 100:
            with self.cache_lock:
                self.param_cache.clear()
        
        return output
    
    def _transform_custom(self, video: torch.Tensor) -> torch.Tensor:
        """
        Custom memory-efficient transform implementation.
        Uses batch processing and device-agnostic code.
        """
        # Get video parameters
        T, C, H, W = video.shape
        device = video.device
        size = self.config.size
        
        # Create output tensor
        output = torch.zeros(T, C, size, size, dtype=video.dtype, device=device)
        
        # Determine if we can process on device or need CPU
        use_cpu = (device.type == "mps" and size > 64) or (size > 128)
        
        if use_cpu:
            # Process on CPU to save GPU memory
            video_cpu = video.cpu()
            
            # Process in batches
            batch_size = min(16, T)
            for i in range(0, T, batch_size):
                end = min(i + batch_size, T)
                batch = video_cpu[i:end]
                
                # Apply transforms
                # ... (similar to OpenCV version but with torch operations)
                
                # Transfer back to device
                output[i:end] = batch.to(device)
        else:
            # Process directly on device
            # ... (transform implementation)
            pass
        
        return output


# Helper function to create transforms
def create_transforms(config: Optional[TransformConfig] = None) -> Callable:
    """
    Create transform function based on configuration.
    
    Args:
        config: Transform configuration
        
    Returns:
        Transform function
    """
    transform = MemoryEfficientTransforms(config)
    return transform


class BatchTransform:
    """Apply transforms to batches, optimized for memory usage."""
    
    def __init__(self, transform: Callable):
        """
        Initialize batch transform.
        
        Args:
            transform: Transform function for individual samples
        """
        self.transform = transform
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Apply transform to batch.
        
        Args:
            batch: Batch tensor [B, C, T, H, W] or [B, T, C, H, W]
            
        Returns:
            Transformed batch
        """
        # Handle different input formats
        B = batch.shape[0]
        is_bcthw = batch.shape[1] == 3 or batch.shape[1] <= 16
        
        if is_bcthw:  # [B, C, T, H, W]
            # Process each sample
            for i in range(B):
                video = batch[i].permute(1, 0, 2, 3)  # [T, C, H, W]
                transformed = self.transform(video)
                batch[i] = transformed.permute(1, 0, 2, 3)  # [C, T, H, W]
        else:  # [B, T, C, H, W]
            # Process each sample
            for i in range(B):
                video = batch[i]  # [T, C, H, W]
                batch[i] = self.transform(video)
        
        return batch