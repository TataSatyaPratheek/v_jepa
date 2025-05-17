import torch
import gc
import os
import psutil
import threading
import time
import logging # Add logging import
from typing import Optional, Dict, Any
from .device import get_device_manager

# Configure logger for this module
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Enhanced memory monitor for tracking both system and GPU memory.
    Supports background monitoring and detailed statistics.
    """
    
    def __init__(self, monitor_interval: float = 1.0, background: bool = False):
        """
        Initialize memory monitor.
        
        Args:
            monitor_interval: Interval for background monitoring in seconds
            background: Whether to run monitoring in background thread
        """
        self.device_manager = get_device_manager()
        self.peak_memory = 0.0
        self.peak_system_memory = 0.0
        self.history = []
        self.monitor_interval = monitor_interval
        
        # Background monitoring
        self._monitoring = False
        self._monitor_thread = None
        if background:
            self.start_background_monitoring()
    
    def update(self) -> Dict[str, float]:
        """
        Update memory statistics.
        
        Returns:
            Dict with current memory usage in MB
        """
        # Get device memory
        mem_info = self.device_manager.get_mem_info()
        current = mem_info.get("allocated_mb", 0.0)
        
        # Get system memory
        process = psutil.Process(os.getpid())
        system_memory = process.memory_info().rss / 1024**2  # MB
        
        # Update peaks
        self.peak_memory = max(self.peak_memory, current)
        self.peak_system_memory = max(self.peak_system_memory, system_memory)
        
        # Record history
        timestamp = time.time()
        record = {
            "timestamp": timestamp,
            "device_memory_mb": current,
            "system_memory_mb": system_memory
        }
        self.history.append(record)
        
        return {
            "device_memory_mb": current,
            "system_memory_mb": system_memory,
            "peak_device_memory_mb": self.peak_memory,
            "peak_system_memory_mb": self.peak_system_memory
        }
    
    def start_background_monitoring(self):
        """Start background memory monitoring thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_background_monitoring(self):
        """Stop background memory monitoring thread."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            self.update()
            time.sleep(self.monitor_interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of memory usage."""
        if not self.history:
            return {"error": "No monitoring data available"}
        
        device_values = [record["device_memory_mb"] for record in self.history]
        system_values = [record["system_memory_mb"] for record in self.history]
        
        return {
            "peak_device_memory_mb": self.peak_memory,
            "peak_system_memory_mb": self.peak_system_memory,
            "avg_device_memory_mb": sum(device_values) / len(device_values),
            "avg_system_memory_mb": sum(system_values) / len(system_values),
            "samples": len(self.history)
        }
    
    def print_report(self):
        """Print formatted memory usage report."""
        summary = self.get_summary()
        
        print("\n===== Memory Usage Report =====")
        print(f"Peak device memory: {summary['peak_device_memory_mb']:.2f} MB")
        print(f"Peak system memory: {summary['peak_system_memory_mb']:.2f} MB")
        print(f"Avg device memory: {summary['avg_device_memory_mb']:.2f} MB")
        print(f"Avg system memory: {summary['avg_system_memory_mb']:.2f} MB")
        print(f"Samples collected: {summary['samples']}")
        print("===============================\n")


def empty_cache(force_gc: bool = True):
    """
    Aggressively clean memory for both device and system.
    
    Args:
        force_gc: Whether to force garbage collection
    """
    # Empty device cache
    device_manager = get_device_manager()
    device_manager.empty_cache()
    
    # Force garbage collection if requested
    if force_gc:
        gc.collect()


class MemoryOptimizer:
    """
    Advanced memory optimization techniques for low-memory environments.
    """
    
    @staticmethod
    def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply memory optimizations for inference.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        # Get device manager
        device_manager = get_device_manager()

        # Check if M1-specific optimizations are implicitly active for inference
        # (e.g., running on MPS with half-precision, which is a common M1 opt setting)
        is_m1_optimized_inference = (
            device_manager.device_str == "mps" and
            device_manager.precision == 16
        )

        if is_m1_optimized_inference:
            logger.info(
                "MemoryOptimizer: M1/MPS detected with 16-bit precision. Applying inference optimizations (half precision, compilation if available)."
            )

        # Apply model optimizations
        if device_manager.device_str != "cpu":
            # Use 16-bit inference if supported
            if device_manager.precision == 16:
                logger.debug("MemoryOptimizer: Converting model to half precision for inference.")
                model = model.half()

        # Use model compilation if available (>= PyTorch 2.0)
        if hasattr(torch, 'compile'):
            logger.debug("MemoryOptimizer: Attempting to compile model with torch.compile for inference.")
            model = device_manager.compile_model(model)

        return model

    @staticmethod
    def optimize_state_dict_memory(state_dict: Dict) -> Dict:
        """
        Optimize memory usage of state dict for loading.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Optimized state dictionary
        """
        # Create new state dict with half precision
        optimized_dict = {}
        
        # Convert eligible tensors to half precision
        for key, tensor in state_dict.items():
            # Skip non-tensor items
            if not isinstance(tensor, torch.Tensor):
                optimized_dict[key] = tensor
                continue
                
            # Skip batch norm parameters to preserve numerical stability
            if any(x in key for x in ['bn', 'batch_norm', 'norm', 'running']):
                optimized_dict[key] = tensor
                continue
                
            # Convert eligible tensors to half precision
            optimized_dict[key] = tensor.half()
        
        return optimized_dict
    
    @staticmethod
    def apply_gradient_checkpointing(model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply gradient checkpointing to reduce memory usage during training.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model with gradient checkpointing enabled
        """
        # Enable gradient checkpointing for supported modules
        for module in model.modules():
            # Handle transformer modules (including BERT, ViT, etc)
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
            
            # Handle modules with gradient checkpointing function
            if hasattr(module, 'set_grad_checkpointing'):
                module.set_grad_checkpointing(True)
        
        return model