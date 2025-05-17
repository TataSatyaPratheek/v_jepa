import torch
import gc

def empty_cache():
    """Aggressive memory cleanup for M1"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

class MemoryMonitor:
    def __init__(self):
        self.peak_memory = 0
        
    def update(self):
        current = torch.mps.current_allocated_memory()
        self.peak_memory = max(self.peak_memory, current)
        return current / 1024**2  # Return in MB
