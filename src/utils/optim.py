import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR
import math
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    """Configuration for optimizer and scheduler."""
    # Optimizer parameters
    optimizer_type: str = "adamw"  # "adam", "adamw", "sgd", "lion"
    lr: float = 1.0e-4
    weight_decay: float = 0.05
    momentum: float = 0.9  # For SGD
    betas: Tuple[float, float] = (0.9, 0.95)  # For Adam, AdamW, Lion
    eps: float = 1.0e-8
    fused: bool = True  # Use fused implementations if available
    # Scheduler parameters
    scheduler_type: str = "cosine"  # "cosine", "warmup_cosine", "onecycle", "none"
    warmup_epochs: int = 10
    min_lr: float = 1.0e-6
    t_max: Optional[int] = None  # Max epochs (if None, use total_epochs)
    # Gradient clipping
    clip_grad_norm: Optional[float] = 1.0
    # Advanced optimization
    use_lookahead: bool = False  # Use Lookahead optimizer wrapper
    use_sam: bool = False  # Use Sharpness-Aware Minimization
    # Memory optimization
    grad_accumulation_steps: int = 1  # Gradient accumulation for larger effective batch size


def create_optimizer(model: torch.nn.Module, config: Optional[OptimizerConfig] = None) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Optimizer configuration
        
    Returns:
        PyTorch optimizer
    """
    # Use default config if not provided
    config = config or OptimizerConfig()
    
    # Extract parameters that require gradient
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Set up parameter groups with weight decay
    # Layer norm and bias parameters typically don't use weight decay
    if config.weight_decay > 0:
        decay_params = []
        no_decay_params = []
        
        for param in params:
            if len(param.shape) <= 1:  # Bias and LayerNorm parameters
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
    else:
        param_groups = params
    
    # Create optimizer based on type
    if config.optimizer_type == "adam":
        # Use fused implementation if available and requested
        use_fused = config.fused and 'fused' in inspect_optimizer_kwargs(optim.Adam)
        
        optimizer = optim.Adam(
            param_groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            fused=use_fused if use_fused else False
        )
    elif config.optimizer_type == "adamw":
        # Use fused implementation if available and requested
        use_fused = config.fused and 'fused' in inspect_optimizer_kwargs(optim.AdamW)
        
        optimizer = optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            fused=use_fused if use_fused else False
        )
    elif config.optimizer_type == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=config.lr,
            momentum=config.momentum,
            nesterov=True
        )
    elif config.optimizer_type == "lion":
        # Lion optimizer (if available)
        try:
            from lion_pytorch import Lion
            optimizer = Lion(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                weight_decay=config.weight_decay
            )
        except ImportError:
            print("Lion optimizer not available. Falling back to AdamW.")
            optimizer = optim.AdamW(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps
            )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")
    
    # Apply additional optimizer wrappers if configured
    if config.use_lookahead:
        try:
            from lookahead_pytorch import Lookahead
            optimizer = Lookahead(optimizer, k=5, alpha=0.5)
        except ImportError:
            print("Lookahead optimizer not available.")
    
    if config.use_sam:
        try:
            from sam import SAM
            optimizer = SAM(params, optimizer, rho=0.05)
        except ImportError:
            print("SAM optimizer not available.")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, 
                    config: Optional[OptimizerConfig] = None,
                    total_epochs: int = 100) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Optimizer configuration
        total_epochs: Total number of training epochs
        
    Returns:
        PyTorch learning rate scheduler
    """
    # Use default config if not provided
    config = config or OptimizerConfig()
    
    # Set t_max to total_epochs if not specified
    t_max = config.t_max or total_epochs
    
    # Create scheduler based on type
    if config.scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=config.min_lr
        )
    elif config.scheduler_type == "warmup_cosine":
        # Custom scheduler with linear warmup and cosine annealing
        def lr_lambda(epoch):
            if epoch < config.warmup_epochs:
                # Linear warmup
                return epoch / config.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - config.warmup_epochs) / (t_max - config.warmup_epochs)
                return config.min_lr + 0.5 * (1.0 - config.min_lr) * (1 + math.cos(math.pi * progress))
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif config.scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.lr,
            total_steps=t_max,
            pct_start=0.3,
            final_div_factor=config.lr / config.min_lr
        )
    elif config.scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
    
    return scheduler


def inspect_optimizer_kwargs(optimizer_class: type) -> List[str]:
    """
    Inspect available keyword arguments for an optimizer.
    
    Args:
        optimizer_class: Optimizer class to inspect
        
    Returns:
        List of available keyword arguments
    """
    import inspect
    
    # Get signature of optimizer constructor
    signature = inspect.signature(optimizer_class.__init__)
    
    # Return parameter names
    return [param for param in signature.parameters]


class GradientAccumulator:
    """
    Helper class for gradient accumulation to simulate larger batch sizes.
    """
    
    def __init__(self, 
                optimizer: torch.optim.Optimizer,
                accumulation_steps: int = 1,
                clip_grad_norm: Optional[float] = None,
                scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """
        Initialize gradient accumulator.
        
        Args:
            optimizer: PyTorch optimizer
            accumulation_steps: Number of steps to accumulate gradients
            clip_grad_norm: Maximum norm for gradient clipping
            scaler: Optional gradient scaler for mixed precision training
        """
        self.optimizer = optimizer
        self.accumulation_steps = max(1, accumulation_steps)
        self.clip_grad_norm = clip_grad_norm
        self.scaler = scaler
        
        # Track current accumulation step
        self.current_step = 0
        
        # Store optimization parameters
        self.optimizer_params = {
            "params": optimizer.param_groups[0]["params"]
        }
    
    def backward(self, loss: torch.Tensor) -> None:
        """
        Perform backward pass with scaling for gradient accumulation.
        
        Args:
            loss: Loss tensor
        """
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.accumulation_steps
        
        # Handle mixed precision if scaler is provided
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Increment step counter
        self.current_step += 1
    
    def step(self) -> None:
        """
        Perform optimization step if accumulation is complete.
        """
        # Only update if we've accumulated enough steps
        if self.current_step >= self.accumulation_steps:
            # Apply gradient clipping if configured
            if self.clip_grad_norm is not None:
                if self.scaler is not None:
                    # For mixed precision training
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.optimizer_params["params"],
                        self.clip_grad_norm
                    )
                else:
                    # Standard gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.optimizer_params["params"],
                        self.clip_grad_norm
                    )
            
            # Perform optimization step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Reset step counter
            self.current_step = 0
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Zero gradients after step.
        
        Args:
            set_to_none: Whether to set gradients to None (more memory efficient)
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    @property
    def is_accumulation_step(self) -> bool:
        """Whether current step is an accumulation step."""
        return self.current_step < self.accumulation_steps - 1