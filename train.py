import os
import sys
import time
import logging
import argparse
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import yaml
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import VideoDatasetConfig, MemoryEfficientVideoDataset, create_video_transforms, create_loader
from src.data.transforms import TransformConfig, create_transforms
from src.data.masking import MaskingConfig, VideoMasker, apply_masking
from src.models.vjepa import VJEPAConfig, VJEPA, create_vjepa_model, VJEPAInference
from src.models.encoders import EncoderConfig
from src.models.predictors import PredictorConfig
from src.utils.device import get_device_manager
from src.utils.memory import MemoryMonitor, empty_cache, MemoryOptimizer
from src.utils.optim import OptimizerConfig, create_optimizer, create_scheduler, GradientAccumulator
from src.utils.logging_utils import configure_colorful_logger, TrainingProgressDisplay
from src.config.defaults import VJEPASystemConfig, get_default_config, apply_m1_optimizations

# Configure enhanced colorful logger
logger = configure_colorful_logger("train")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train V-JEPA model")
    
    # Config arguments
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--optimize_for_m1', action='store_true', help='Apply M1-specific optimizations')
    
    # Runtime arguments
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--precision', type=int, default=16, choices=[16, 32], help='Precision (16 or 32)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for optimization')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=None, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, default=None, help='Path to dataset')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    
    # Debug arguments
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    
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
    
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    if args.accumulation_steps is not None:
        config.training.accumulation_steps = args.accumulation_steps
        config.optimizer.grad_accumulation_steps = args.accumulation_steps
    
    if args.epochs is not None:
        config.training.epochs = args.epochs
    
    if args.lr is not None:
        config.optimizer.lr = args.lr
    
    if args.resume is not None:
        config.training.resume = args.resume
    
    if args.data_path is not None:
        config.dataset.path = args.data_path
    
    if args.experiment_name is not None:
        config.training.experiment_name = args.experiment_name
    
    if args.profile is not None:
        config.runtime.profile = args.profile
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    config.training.log_dir = os.path.join(args.output_dir, 'logs', config.training.experiment_name)
    config.training.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints', config.training.experiment_name)
    
    os.makedirs(config.training.log_dir, exist_ok=True)
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.training.log_dir, 'config.yaml')
    config.save(config_path)
    
    return config


def create_datasets(config: VJEPASystemConfig):
    """
    Create training and validation datasets.
    
    Args:
        config: System configuration
        
    Returns:
        Tuple of (train_dataset, val_dataset, train_loader, val_loader)
    """
    # Create transform configurations
    train_transform_config = TransformConfig(
        mode="train",
        size=config.model.encoder_config.img_size,
        backend="torchvision" if not config.training.optimize_for_m1 else "cv2",
        random_crop=True,
        random_flip=True,
        color_jitter=True,
        normalize=True
    )
    
    val_transform_config = TransformConfig(
        mode="val",
        size=config.model.encoder_config.img_size,
        backend="torchvision" if not config.training.optimize_for_m1 else "cv2",
        normalize=True
    )
    
    # Create transforms
    train_transform = create_transforms(train_transform_config)
    val_transform = create_transforms(val_transform_config)
    
    # Create dataset config
    dataset_config = config.dataset
    dataset_config.img_size = config.model.encoder_config.img_size
    
    # Create datasets
    logger.info(f"Creating datasets from {dataset_config.path}")
    train_dataset = MemoryEfficientVideoDataset(dataset_config, transform=train_transform)
    
    # Create validation dataset with same config but different transform
    val_dataset = MemoryEfficientVideoDataset(dataset_config, transform=val_transform)
    
    # Create data loaders
    train_loader = create_loader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.runtime.threads["dataloader"],
        optimize_for_m1=config.training.optimize_for_m1,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=dataset_config.prefetch_factor,
        drop_last=True
    )
    
    val_loader = create_loader(
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.runtime.threads["dataloader"],
        optimize_for_m1=config.training.optimize_for_m1,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=dataset_config.prefetch_factor,
        drop_last=False
    )
    
    return train_dataset, val_dataset, train_loader, val_loader


def train_epoch(model: VJEPA,
               loader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               grad_accumulator: GradientAccumulator,
               device_manager,
               mask_generator: VideoMasker,
               epoch: int,
               config: VJEPASystemConfig,
               writer: SummaryWriter,
               memory_monitor: MemoryMonitor):
    """
    Train model for one epoch.
    
    Args:
        model: VJEPA model
        loader: Data loader
        optimizer: Optimizer
        grad_accumulator: Gradient accumulator
        device_manager: Device manager
        mask_generator: Mask generator
        epoch: Current epoch
        config: System configuration
        writer: TensorBoard writer
        memory_monitor: Memory monitor
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    device = device_manager.device
    
    # Get AMP context and scaler if configured
    amp_context = device_manager.get_amp_context(config.training.amp)
    scaler = device_manager.get_grad_scaler(config.training.amp)
    
    # Create progress display
    progress_display = TrainingProgressDisplay(
        total_epochs=config.training.epochs,
        steps_per_epoch=len(loader),
        metrics=['loss', 'lr', 'memory_mb'],
        use_rich=True  # Use Rich for fancy display
    )
    progress_display.start()
    
    # Training statistics
    train_loss = 0.0
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    
    # Start timer
    start = time.time()
    end = time.time()
    
    # Log initial message
    logger.info(f"Starting epoch {epoch}/{config.training.epochs} with {len(loader)} steps")
    logger.info(f"Batch size: {config.training.batch_size} × {config.optimizer.grad_accumulation_steps} (accumulation)")
    
    # Current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    # Iterate over batches
    for i, batch in enumerate(loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move to device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device, non_blocking=True)
        
        # Apply masking
        masked_batch = apply_masking(batch, config.masking)
        
        # Generate mask
        mask = torch.zeros_like(batch, device=device)
        mask[masked_batch > 0] = 1
        mask = mask.sum(dim=1).bool().float()  # [B, H, W]
        mask = mask.flatten(1)  # [B, H*W]
        
        # Forward pass
        with amp_context:
            outputs = model(batch, masked_batch, mask)
            loss = outputs["loss"]
        
        # Backward pass with gradient accumulation
        grad_accumulator.backward(loss)
        
        # Update weights if accumulation is complete
        if not grad_accumulator.is_accumulation_step:
            grad_accumulator.step()
            grad_accumulator.zero_grad(set_to_none=True)
        
        # Update metrics
        batch_time.update(time.time() - end)
        losses.update(loss.item())
        
        # Empty cache if configured
        if i % config.training.empty_cache_freq == 0:
            empty_cache()
        
        # Update memory monitor
        memory_info = memory_monitor.update()
        memory_mb = memory_info.get("device_memory_mb", 0.0)
        
        # Update progress display
        metrics = {
            'loss': loss.item(),
            'lr': current_lr,
            'memory_mb': memory_mb
        }
        progress_display.update_display(epoch, i+1, metrics)
        
        # Log to TensorBoard
        global_step = (epoch - 1) * len(loader) + i
        writer.add_scalar('train/loss', losses.val, global_step)
        writer.add_scalar('train/memory', memory_mb, global_step)
        writer.add_scalar('train/batch_time', batch_time.val, global_step)
        
        # Reset end time
        end = time.time()
    
    # Stop progress display
    progress_display.stop()
    
    # Calculate metrics
    metrics = {
        'loss': losses.avg,
        'time': time.time() - start,
        'memory': memory_monitor.peak_memory
    }
    
    logger.info(
        f"Epoch {epoch} completed in {metrics['time']:.2f}s. "
        f"Loss: {metrics['loss']:.4f}, "
        f"Peak memory: {metrics['memory']:.1f}MB"
    )
    
    return metrics


def validate(model: VJEPA,
            loader: torch.utils.data.DataLoader,
            device_manager,
            mask_generator: VideoMasker,
            config: VJEPASystemConfig,
            writer: SummaryWriter,
            epoch: int):
    """
    Validate model.
    
    Args:
        model: VJEPA model
        loader: Data loader
        device_manager: Device manager
        mask_generator: Mask generator
        config: System configuration
        writer: TensorBoard writer
        epoch: Current epoch
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    device = device_manager.device
    
    # Get AMP context
    amp_context = device_manager.get_amp_context(config.training.amp)
    
    # Create progress display
    progress_display = TrainingProgressDisplay(
        total_epochs=1,  # Just one validation pass
        steps_per_epoch=len(loader),
        metrics=['val_loss'],
        use_rich=True  # Use Rich for fancy display
    )
    progress_display.start()
    
    # Validation statistics
    val_loss = 0.0
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    
    # Start timer
    start = time.time()
    end = time.time()
    
    # Log initial message
    logger.info(f"Starting validation with {len(loader)} steps")
    
    # Iterate over batches
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Move to device
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device, non_blocking=True)
            
            # Apply masking
            masked_batch = apply_masking(batch, config.masking)
            
            # Generate mask
            mask = torch.zeros_like(batch, device=device)
            mask[masked_batch > 0] = 1
            mask = mask.sum(dim=1).bool().float()  # [B, H, W]
            mask = mask.flatten(1)  # [B, H*W]
            
            # Forward pass
            with amp_context:
                outputs = model(batch, masked_batch, mask)
                loss = outputs["loss"]
            
            # Update metrics
            batch_time.update(time.time() - end)
            losses.update(loss.item())
            
            # Update progress display
            metrics = {
                'val_loss': loss.item()
            }
            progress_display.update_display(1, i+1, metrics)
            
            # Reset end time
            end = time.time()
    
    # Stop progress display
    progress_display.stop()
    
    # Calculate metrics
    metrics = {
        'loss': losses.avg,
        'time': time.time() - start
    }
    
    # Log to TensorBoard
    writer.add_scalar('val/loss', metrics['loss'], epoch)
    
    logger.info(
        f"Validation completed in {metrics['time']:.2f}s. "
        f"Loss: {metrics['loss']:.4f}"
    )
    
    return metrics


def save_checkpoint(model: VJEPA,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int,
                   metrics: Dict[str, float],
                   config: VJEPASystemConfig,
                   is_best: bool = False):
    """
    Save model checkpoint.
    
    Args:
        model: VJEPA model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Training metrics
        config: System configuration
        is_best: Whether this is the best model so far
    """
    # Create checkpoint directory
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
        'encoder_config': asdict(config.model.encoder_config) if config.model.encoder_config else None,
        'config': config.to_dict()
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(config.training.checkpoint_dir, f'checkpoint_{epoch:04d}.pth')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model if applicable
    if is_best:
        best_path = os.path.join(config.training.checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")
    
    # Save latest model
    latest_path = os.path.join(config.training.checkpoint_dir, 'latest_model.pth')
    torch.save(checkpoint, latest_path)
    
    # Remove old checkpoints to save space if needed
    checkpoints = sorted([
        f for f in os.listdir(config.training.checkpoint_dir)
        if f.startswith('checkpoint_') and f.endswith('.pth')
    ])
    
    # Keep at most 5 checkpoints to save space
    max_keep = 5
    if len(checkpoints) > max_keep:
        for checkpoint_to_remove in checkpoints[:-max_keep]:
            os.remove(os.path.join(config.training.checkpoint_dir, checkpoint_to_remove))
            logger.info(f"Removed old checkpoint: {checkpoint_to_remove}")


def train(config: VJEPASystemConfig):
    """
    Train VJEPA model.
    
    Args:
        config: System configuration
    """
    # Setup device
    device_manager = get_device_manager(config.runtime.device, config.runtime.precision)
    device = device_manager.device
    
    logger.info(f"Using device: {device}")
    
    # Set OpenMP threads
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(config.runtime.threads["omp"])
    
    # Create datasets and loaders
    train_dataset, val_dataset, train_loader, val_loader = create_datasets(config)
    logger.info(f"Created datasets with {len(train_dataset)} training and {len(val_dataset)} validation samples")
    
    # Create mask generator
    mask_generator = VideoMasker(config.masking)
    
    # Create model
    model = create_vjepa_model(config.model)
    logger.info(f"Created VJEPA model with encoder dim {model.get_embedding_dim()}")
    
    # Apply memory optimizations
    if config.training.optimize_for_m1:
        logger.info("Applying memory optimizations for training")
        model.context_encoder = MemoryOptimizer.apply_gradient_checkpointing(model.context_encoder)
        if not config.model.share_parameters:
            model.target_encoder = MemoryOptimizer.apply_gradient_checkpointing(model.target_encoder)
    
    # Apply model compilation if configured
    if config.runtime.compile and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile")
        model.context_encoder = device_manager.compile_model(model.context_encoder)
        if not config.model.share_parameters:
            model.target_encoder = device_manager.compile_model(model.target_encoder)
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config.optimizer)
    scheduler = create_scheduler(optimizer, config.optimizer, config.training.epochs)
    logger.info(f"Created {config.optimizer.optimizer_type} optimizer with lr={config.optimizer.lr}")
    
    # Create gradient accumulator
    grad_accumulator = GradientAccumulator(
        optimizer,
        accumulation_steps=config.optimizer.grad_accumulation_steps,
        clip_grad_norm=config.optimizer.clip_grad_norm,
        scaler=device_manager.get_grad_scaler(config.training.amp)
    )
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=config.training.log_dir)
    
    # Create memory monitor
    memory_monitor = MemoryMonitor(background=True)
    memory_monitor.start_background_monitoring()
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if configured
    if config.training.resume:
        logger.info(f"Resuming from checkpoint: {config.training.resume}")
        checkpoint = torch.load(config.training.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        if 'metrics' in checkpoint and 'val_loss' in checkpoint['metrics']:
            best_val_loss = checkpoint['metrics']['val_loss']
    
    # Log model and dataset info
    logger.info(f"Training with batch size {config.training.batch_size} * {config.optimizer.grad_accumulation_steps} (accumulation)")
    logger.info(f"Training for {config.training.epochs} epochs")
    
    # Main training loop
    for epoch in range(start_epoch, config.training.epochs):
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            grad_accumulator=grad_accumulator,
            device_manager=device_manager,
            mask_generator=mask_generator,
            epoch=epoch,
            config=config,
            writer=writer,
            memory_monitor=memory_monitor
        )
        
        # Validate if configured
        if config.training.eval_every > 0 and (epoch + 1) % config.training.eval_every == 0:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                device_manager=device_manager,
                mask_generator=mask_generator,
                config=config,
                writer=writer,
                epoch=epoch
            )
            
            # Update best validation loss
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % config.training.save_every == 0:
                combined_metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=combined_metrics,
                    config=config,
                    is_best=is_best
                )
        else:
            # Save checkpoint without validation
            if (epoch + 1) % config.training.save_every == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=train_metrics,
                    config=config,
                    is_best=False
                )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('train/lr', current_lr, epoch)
            logger.info(f"Learning rate: {current_lr:.6f}")
    
    # Final validation
    final_val_metrics = validate(
        model=model,
        loader=val_loader,
        device_manager=device_manager,
        mask_generator=mask_generator,
        config=config,
        writer=writer,
        epoch=config.training.epochs
    )
    
    # Save final checkpoint
    combined_metrics = {**train_metrics, **{f'val_{k}': v for k, v in final_val_metrics.items()}}
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=config.training.epochs - 1,
        metrics=combined_metrics,
        config=config,
        is_best=final_val_metrics['loss'] < best_val_loss
    )
    
    # Create inference model
    inference_model = VJEPAInference(model)
    
    # Save inference model
    inference_path = os.path.join(config.training.checkpoint_dir, 'inference_model.pth')
    torch.save({
        'model': inference_model.state_dict(),
        'config': config.to_dict()
    }, inference_path)
    logger.info(f"Saved inference model to {inference_path}")
    
    # Stop memory monitoring
    memory_monitor.stop_background_monitoring()
    memory_monitor.print_report()
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info("Training completed successfully!")


class AverageMeter(object):
    """Computes and stores the average and current value."""
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Set random seed for reproducibility
    seed = config.runtime.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set environment variables
    os.environ["TORCH_MPS_USE_SYSTEM_ALLOCATOR"] = "1"  # Better MPS memory management
    
    # Start training
    train(config)


if __name__ == "__main__":
    main()