import torch
import os
from typing import Optional, Tuple, Dict, Any


class DeviceManager:
    """
    Manages device-specific operations with automatic detection and fallbacks.
    Provides unified interface for CUDA, MPS, and CPU environments.
    """
    
    def __init__(self, device: str = "auto", precision: int = 16):
        """
        Initialize device manager with automatic detection.
        
        Args:
            device: Requested device ('auto', 'cuda', 'mps', or 'cpu')
            precision: Precision to use (16 or 32)
        """
        self.device_str = self._get_device(device)
        self.precision = precision
        self.supports_amp = self._check_amp_support()
        self.is_gpu = self.device_str != "cpu"
        
        # Set environment variables for optimizations
        if self.device_str == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        if precision == 16:
            # Enable faster matmul for fp16
            torch.set_float32_matmul_precision('medium')
    
    def _get_device(self, requested_device: str) -> str:
        """Determine the best available device."""
        if requested_device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        
        # Handle specific device requests
        if requested_device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back.")
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        
        if requested_device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available. Falling back to CPU.")
            return "cpu"
        
        return requested_device
    
    def _check_amp_support(self) -> bool:
        """Check if automatic mixed precision is supported on the device."""
        if self.device_str == "cuda":
            return True
        elif self.device_str == "mps":
            # MPS supports autocast since PyTorch 2.0
            return True
        return False
    
    @property
    def device(self) -> torch.device:
        """Get the torch device object."""
        return torch.device(self.device_str)
    
    def empty_cache(self) -> None:
        """Empty cached memory on the device."""
        if self.device_str == "cuda":
            torch.cuda.empty_cache()
        elif self.device_str == "mps":
            torch.mps.empty_cache()
    
    def get_mem_info(self) -> Dict[str, float]:
        """Get memory information for the device."""
        if self.device_str == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            return {"allocated_mb": allocated, "reserved_mb": reserved}
        elif self.device_str == "mps":
            allocated = torch.mps.current_allocated_memory() / 1024**2
            return {"allocated_mb": allocated}
        return {"allocated_mb": 0, "reserved_mb": 0}
    
    def pin_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pin memory for faster transfer to device if supported."""
        if self.is_gpu and hasattr(tensor, 'pin_memory'):
            return tensor.pin_memory(self.device_str)
        return tensor
    
    def get_amp_context(self, enabled: bool = True):
        """Get appropriate context manager for automatic mixed precision."""
        if not enabled or not self.supports_amp or self.precision != 16:
            # Return a dummy context manager when AMP not used
            from contextlib import nullcontext
            return nullcontext()
        
        return torch.autocast(device_type=self.device_str, dtype=torch.float16)
    
    def get_grad_scaler(self, enabled: bool = True):
        """Get gradient scaler for mixed precision training if supported."""
        if not enabled or not self.supports_amp or self.precision != 16:
            return None
        
        if self.device_str == "cuda":
            return torch.cuda.amp.GradScaler()
        return None
    
    def compile_model(self, model, mode="reduce-overhead"):
        """
        Compile model using torch.compile if available (PyTorch 2.0+)
        
        Args:
            model: PyTorch model to compile
            mode: Compilation mode
        """
        if hasattr(torch, 'compile'):
            return torch.compile(model, mode=mode)
        return model


# Singleton device manager for global access
_device_manager = None

def get_device_manager(device: str = "auto", precision: int = 16) -> DeviceManager:
    """Get or create the global device manager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(device, precision)
    return _device_manager