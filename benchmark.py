import os
import sys
import time
import logging
import argparse
from typing import Dict, Any, Optional, Tuple, List
import yaml
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import VideoDatasetConfig, MemoryEfficientVideoDataset, create_video_transforms, create_loader
from src.data.transforms import TransformConfig, create_transforms
from src.models.vjepa import VJEPAConfig, VJEPA, create_vjepa_model, VJEPAInference
from src.models.encoders import EncoderConfig, create_encoder
from src.utils.device import get_device_manager
from src.utils.memory import MemoryMonitor, empty_cache, MemoryOptimizer
from src.utils.logging_utils import configure_colorful_logger, BenchmarkProgressDisplay
from src.config.defaults import VJEPASystemConfig, get_default_config, apply_m1_optimizations

# Configure enhanced colorful logger
logger = configure_colorful_logger("benchmark")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark V-JEPA model")
    
    # Config arguments
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--optimize_for_m1', action='store_true', help='Apply M1-specific optimizations')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='vjepa', choices=['vjepa', 'encoder'], help='Model type to benchmark')
    
    # Runtime arguments
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--precision', type=int, default=16, choices=[16, 32], help='Precision (16 or 32)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for optimization')
    
    # Benchmark arguments
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--random_data', action='store_true', help='Use random data instead of real dataset')
    parser.add_argument('--input_size', type=int, nargs='+', default=[3, 8, 128, 128], help='Input size for random data [C, T, H, W]')
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, default=None, help='Path to dataset')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='benchmark_results', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def load_config(args) -> VJEPASystemConfig:
    """
    Load configuration from file and apply command line overrides.
    
    Args:
        args: Command line arguments
        
    Returns:
        System configuration
    """
    # Start with default configuration
    config = get_default_config()
    
    # Load configuration from file if provided
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        config = VJEPASystemConfig.from_yaml(args.config)
    
    # Apply M1 optimizations if requested
    if args.optimize_for_m1:
        logger.info("Applying M1-specific optimizations")
        config = apply_m1_optimizations(config)
    
    # Override with command line arguments
    if args.device:
        config.runtime.device = args.device
    
    if args.precision:
        config.runtime.precision = args.precision
    
    if args.compile is not None:
        config.runtime.compile = args.compile
    
    if args.data_path is not None:
        config.dataset.path = args.data_path
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    return config


def create_model(config: VJEPASystemConfig, 
               model_type: str, 
               model_path: Optional[str] = None,
               device: torch.device = torch.device('cpu')) -> nn.Module:
    """
    Create model for benchmarking.
    
    Args:
        config: System configuration
        model_type: Type of model to create ('vjepa' or 'encoder')
        model_path: Path to model checkpoint
        device: Device to use
        
    Returns:
        PyTorch model
    """
    if model_type == 'vjepa':
        # Create full VJEPA model
        model = create_vjepa_model(config.model)
        logger.info(f"Created VJEPA model with encoder dim {model.get_embedding_dim()}")
    else:
        # Create encoder only
        model = create_encoder(config.model.encoder_config)
        logger.info(f"Created encoder model with dim {config.model.encoder_config.embed_dim}")
    
    # Load weights if provided
    if model_path is not None and os.path.exists(model_path):
        logger.info(f"Loading weights from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model' in checkpoint:
            # Checkpoint contains model state dict
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            # Checkpoint is model state dict
            model.load_state_dict(checkpoint, strict=False)
    
    # Apply optimizations for inference
    model = MemoryOptimizer.optimize_model_for_inference(model)
    
    # Move to device
    model = model.to(device)
    
    # Set to eval mode
    model.eval()
    
    return model


def create_random_data(batch_size: int, 
                     input_size: List[int], 
                     device: torch.device,
                     precision: int = 32) -> torch.Tensor:
    """
    Create random data for benchmarking.
    
    Args:
        batch_size: Batch size
        input_size: Input size [C, T, H, W]
        device: Device to use
        precision: Precision (16 or 32)
        
    Returns:
        Random tensor
    """
    # Create input shape [B, C, T, H, W]
    shape = [batch_size] + input_size
    
    # Create random tensor
    if precision == 16:
        return torch.rand(shape, dtype=torch.float16, device=device)
    else:
        return torch.rand(shape, dtype=torch.float32, device=device)


def create_dataset_loader(config: VJEPASystemConfig, 
                        batch_size: int,
                        num_workers: int) -> Tuple[Dataset, DataLoader]:
    """
    Create dataset and loader for benchmarking.
    
    Args:
        config: System configuration
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        Tuple of (dataset, loader)
    """
    # Create transform config
    transform_config = TransformConfig(
        mode="val",
        size=config.model.encoder_config.img_size,
        backend="torchvision" if not config.training.optimize_for_m1 else "cv2",
        normalize=True
    )
    
    # Create transforms
    transform = create_transforms(transform_config)
    
    # Create dataset config
    dataset_config = config.dataset
    dataset_config.img_size = config.model.encoder_config.img_size
    
    # Create dataset
    logger.info(f"Creating dataset from {dataset_config.path}")
    dataset = MemoryEfficientVideoDataset(dataset_config, transform=transform)
    
    # Create data loader
    loader = create_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        optimize_for_m1=config.training.optimize_for_m1,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=dataset_config.prefetch_factor,
        drop_last=False
    )
    
    return dataset, loader


def benchmark_inference(model: nn.Module,
                      args,
                      config: VJEPASystemConfig,
                      device_manager,
                      memory_monitor: MemoryMonitor) -> Dict[str, Any]:
    """
    Benchmark model inference.
    
    Args:
        model: PyTorch model
        args: Command line arguments
        config: System configuration
        device_manager: Device manager
        memory_monitor: Memory monitor
        
    Returns:
        Dictionary of benchmark results
    """
    device = device_manager.device
    
    # Get AMP context
    amp_context = device_manager.get_amp_context(config.runtime.precision == 16)
    
    # Initialize progress display
    progress_display = BenchmarkProgressDisplay(
        total_iterations=args.iterations,
        warmup_iterations=args.warmup,
        use_rich=True  # Use Rich for fancy display
    )
    progress_display.start()
    
    # Prepare data
    if args.random_data:
        # Create random data
        logger.info(f"Creating random data with shape {args.input_size} and batch size {args.batch_size}")
        data = create_random_data(
            batch_size=args.batch_size,
            input_size=args.input_size,
            device=device,
            precision=config.runtime.precision
        )
        
        # Create data iterator
        data_iterator = range(args.iterations + args.warmup)
        iter_data = lambda _: data
    else:
        # Create dataset and loader
        _, loader = create_dataset_loader(
            config=config,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Create infinite iterator
        data_iterator = iter(loader)
        def iter_data(_):
            # Get next batch
            try:
                batch = next(data_iterator)
            except StopIteration:
                # Reset iterator
                nonlocal data_iterator
                data_iterator = iter(loader)
                batch = next(data_iterator)
            
            # Move to device
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device, non_blocking=True)
            
            return batch
    
    # Warm up
    logger.info(f"Warming up for {args.warmup} iterations")
    with torch.no_grad():
        for i in range(args.warmup):
            batch = iter_data(i)
            with amp_context:
                _ = model(batch)
            
            # Update progress display with dummy metrics
            progress_display.update_display(
                iteration=i+1,
                latency_ms=0.0,
                throughput=0.0,
                memory_mb=0.0
            )
            
            # Empty cache
            empty_cache()
    
    # Reset peak memory
    memory_monitor.peak_memory = 0
    memory_monitor.peak_system_memory = 0
    
    # Benchmark
    logger.info(f"Benchmarking for {args.iterations} iterations")
    latencies = []
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(args.iterations):
            # Get batch
            batch = iter_data(i)
            
            # Measure inference time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            
            # Forward pass
            with amp_context:
                _ = model(batch)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latency = time.perf_counter() - start
            latencies.append(latency)
            
            # Update memory monitor
            memory_info = memory_monitor.update()
            memory_mb = memory_info.get("device_memory_mb", 0.0)
            
            # Calculate iteration throughput
            throughput = 1.0 / latency if latency > 0 else 0.0
            
            # Update progress display
            progress_display.update_display(
                iteration=args.warmup + i + 1,
                latency_ms=latency * 1000,
                throughput=throughput,
                memory_mb=memory_mb
            )
    
    # Stop progress display
    progress_display.stop()
    
    # Calculate metrics
    total_time = time.time() - start_time
    latencies_ms = np.array(latencies) * 1000  # Convert to milliseconds
    
    results = {
        'batch_size': args.batch_size,
        'precision': config.runtime.precision,
        'device': device_manager.device_str,
        'iterations': args.iterations,
        'total_time_sec': total_time,
        'samples_per_sec': args.iterations * args.batch_size / total_time,
        'latency_mean_ms': float(np.mean(latencies_ms)),
        'latency_median_ms': float(np.median(latencies_ms)),
        'latency_min_ms': float(np.min(latencies_ms)),
        'latency_max_ms': float(np.max(latencies_ms)),
        'latency_p90_ms': float(np.percentile(latencies_ms, 90)),
        'latency_p95_ms': float(np.percentile(latencies_ms, 95)),
        'latency_p99_ms': float(np.percentile(latencies_ms, 99)),
        'latency_std_ms': float(np.std(latencies_ms)),
        'peak_device_memory_mb': float(memory_monitor.peak_memory),
        'peak_system_memory_mb': float(memory_monitor.peak_system_memory),
    }
    
    # Log results
    logger.info(f"Benchmark completed in {total_time:.2f}s")
    logger.info(f"Samples per second: {results['samples_per_sec']:.2f}")
    logger.info(f"Mean latency: {results['latency_mean_ms']:.2f}ms")
    logger.info(f"Median latency: {results['latency_median_ms']:.2f}ms")
    logger.info(f"Peak device memory: {results['peak_device_memory_mb']:.1f}MB")
    logger.info(f"Peak system memory: {results['peak_system_memory_mb']:.1f}MB")
    
    return results


def benchmark_model_sizes(model: nn.Module) -> Dict[str, Any]:
    """
    Benchmark model sizes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of model size metrics
    """
    # Calculate model size
    param_count = sum(p.numel() for p in model.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    # Calculate buffer size
    buffer_count = sum(b.numel() for b in model.buffers())
    buffer_size_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
    
    # Calculate state dict size
    state_dict = model.state_dict()
    state_dict_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024 * 1024)
    
    # Calculate half precision size
    half_precision_size_mb = sum(p.numel() * 2 for p in model.parameters()) / (1024 * 1024)
    
    # Log results
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Model parameters size: {param_size_mb:.2f}MB")
    logger.info(f"Model buffers: {buffer_count:,}")
    logger.info(f"Model buffers size: {buffer_size_mb:.2f}MB")
    logger.info(f"Model state dict size: {state_dict_size_mb:.2f}MB")
    logger.info(f"Model half precision size: {half_precision_size_mb:.2f}MB")
    
    return {
        'param_count': param_count,
        'param_size_mb': param_size_mb,
        'buffer_count': buffer_count,
        'buffer_size_mb': buffer_size_mb,
        'state_dict_size_mb': state_dict_size_mb,
        'half_precision_size_mb': half_precision_size_mb,
    }


def save_results(results: Dict[str, Any], args, config: VJEPASystemConfig):
    """
    Save benchmark results.
    
    Args:
        results: Benchmark results
        args: Command line arguments
        config: System configuration
    """
    # Create timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create results directory
    results_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(results_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save configuration
    config_path = os.path.join(results_dir, 'config.yaml')
    config.save(config_path)
    
    # Save command line arguments
    args_path = os.path.join(results_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Saved results to {results_dir}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.runtime.seed)
    np.random.seed(config.runtime.seed)
    
    # Set environment variables
    os.environ["TORCH_MPS_USE_SYSTEM_ALLOCATOR"] = "1"  # Better MPS memory management
    os.environ["OMP_NUM_THREADS"] = str(config.runtime.threads["omp"])
    
    # Setup device
    device_manager = get_device_manager(config.runtime.device, config.runtime.precision)
    device = device_manager.device
    
    logger.info(f"Using device: {device}")
    
    # Create memory monitor
    memory_monitor = MemoryMonitor()
    
    # Create model
    model = create_model(
        config=config,
        model_type=args.model_type,
        model_path=args.model_path,
        device=device
    )
    
    # Apply model compilation if configured
    if config.runtime.compile and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile")
        model = device_manager.compile_model(model)
    
    # Get model size metrics
    model_size_metrics = benchmark_model_sizes(model)
    
    # Benchmark inference
    inference_metrics = benchmark_inference(
        model=model,
        args=args,
        config=config,
        device_manager=device_manager,
        memory_monitor=memory_monitor
    )
    
    # Combine metrics
    results = {
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'device': device_manager.device_str,
        'precision': config.runtime.precision,
        'compile': config.runtime.compile,
        'model_metrics': model_size_metrics,
        'inference_metrics': inference_metrics,
    }
    
    # Print system info
    if device_manager.device_str == 'mps':
        import platform
        results['system_info'] = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__
        }
    
    # Save results
    save_results(results, args, config)


if __name__ == "__main__":
    main()