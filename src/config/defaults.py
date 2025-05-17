import os
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
import hydra
from omegaconf import OmegaConf, DictConfig
import logging

from ..models.vjepa import VJEPAConfig
from ..models.encoders import EncoderConfig
from ..models.predictors import PredictorConfig
from ..data.masking import MaskingConfig
from ..data.dataset import VideoDatasetConfig
from ..data.transforms import TransformConfig
from ..utils.optim import OptimizerConfig

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    # Basic training parameters
    batch_size: int = 2
    accumulation_steps: int = 4  # Effective batch size = batch_size * accumulation_steps
    epochs: int = 100
    save_every: int = 5
    eval_every: int = 1
    # Logging
    log_dir: str = "logs"
    experiment_name: str = "vjepa"
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume: Optional[str] = None
    # Distributed training
    distributed: bool = False
    dist_backend: str = "nccl"
    dist_url: str = "tcp://localhost:10001"
    # Mixed precision
    amp: bool = True
    # Memory optimizations
    empty_cache_freq: int = 5
    optimize_for_m1: bool = True


@dataclass
class RuntimeConfig:
    """Configuration for runtime environment."""
    # Hardware settings
    device: str = "auto"  # 'auto', 'cuda', 'mps', or 'cpu'
    precision: int = 16   # 16 or 32
    compile: bool = True  # Use torch.compile for optimization
    # Threading settings
    threads: Dict[str, int] = field(default_factory=lambda: {
        "omp": 2,         # OpenMP threads
        "dataloader": 2,  # DataLoader workers
        "io": 2           # IO threads
    })
    # Profiling
    profile: bool = False
    # Memory limits
    memory_limit: Optional[int] = None  # In MB
    # Reproducibility
    seed: int = 42


@dataclass
class VJEPASystemConfig:
    """Complete configuration for VJEPA system."""
    # Component configurations
    model: VJEPAConfig = field(default_factory=VJEPAConfig)
    dataset: VideoDatasetConfig = field(default_factory=VideoDatasetConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert to dictionary
        config_dict = self.to_dict()
        
        # Save to YAML
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VJEPASystemConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Configuration instance
        """
        # Create nested configurations
        model_config = VJEPAConfig(**config_dict.get('model', {}))
        dataset_config = VideoDatasetConfig(**config_dict.get('dataset', {}))
        transforms_config = TransformConfig(**config_dict.get('transforms', {}))
        masking_config = MaskingConfig(**config_dict.get('masking', {}))
        optimizer_config = OptimizerConfig(**config_dict.get('optimizer', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        runtime_config = RuntimeConfig(**config_dict.get('runtime', {}))
        
        # Create system config
        return cls(
            model=model_config,
            dataset=dataset_config,
            transforms=transforms_config,
            masking=masking_config,
            optimizer=optimizer_config,
            training=training_config,
            runtime=runtime_config
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> 'VJEPASystemConfig':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration instance
        """
        # Load YAML file
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config from dictionary
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_hydra(cls, config: DictConfig) -> 'VJEPASystemConfig':
        """
        Create configuration from Hydra config.
        
        Args:
            config: Hydra configuration
            
        Returns:
            Configuration instance
        """
        # Convert to dictionary
        config_dict = OmegaConf.to_container(config, resolve=True)
        
        # Create config from dictionary
        return cls.from_dict(config_dict)


def get_default_config() -> VJEPASystemConfig:
    """
    Get default configuration.
    
    Returns:
        Default configuration instance
    """
    return VJEPASystemConfig()


# Initialize Hydra
@hydra.main(config_path="../config", config_name="defaults")
def create_config(config: DictConfig) -> VJEPASystemConfig:
    """
    Create configuration from Hydra.
    
    Args:
        config: Hydra configuration
        
    Returns:
        Configuration instance
    """
    return VJEPASystemConfig.from_hydra(config)


def apply_m1_optimizations(config: VJEPASystemConfig) -> VJEPASystemConfig:
    """
    Apply optimizations for M1 Mac with 8GB RAM.
    
    Args:
        config: Original configuration
        
    Returns:
        Optimized configuration
    """
    # Create a copy of the config
    optimized_config = VJEPASystemConfig(**asdict(config))
    
    # Memory optimizations for M1
    # 1. Runtime settings
    optimized_config.runtime.precision = 16  # Use half precision
    optimized_config.runtime.threads["omp"] = 2  # Limit OpenMP threads
    optimized_config.runtime.threads["dataloader"] = 2  # Limit DataLoader workers
    optimized_config.runtime.threads["io"] = 2  # Limit IO threads
    
    # 2. Model settings
    if optimized_config.model.encoder_config is None:
        optimized_config.model.encoder_config = EncoderConfig()
    
    optimized_config.model.encoder_config.use_half_precision = True
    optimized_config.model.encoder_config.use_gradient_checkpointing = True
    optimized_config.model.share_parameters = True  # Share encoder parameters to save memory
    
    # 3. Training settings
    optimized_config.training.batch_size = 2
    optimized_config.training.accumulation_steps = 4  # Use gradient accumulation
    optimized_config.training.amp = True  # Use automatic mixed precision
    optimized_config.training.empty_cache_freq = 5  # Frequently clear cache
    optimized_config.training.optimize_for_m1 = True
    
    # 4. Dataset settings
    optimized_config.dataset.use_half_precision = True
    optimized_config.dataset.optimize_for_m1 = True
    
    # 5. Optimizer settings
    optimized_config.optimizer.fused = True  # Use fused implementations
    optimized_config.optimizer.grad_accumulation_steps = optimized_config.training.accumulation_steps
    
    return optimized_config


def create_defaults_py():
    """Create defaults.py file with default configuration."""
    # Get default config
    default_config = get_default_config()
    
    # Convert to dictionary
    config_dict = default_config.to_dict()
    
    # Convert to YAML
    yaml_str = yaml.dump(config_dict, default_flow_style=False)
    
    # Create Python file with default configuration
    content = f"""
# Default configuration for VJEPA system
import os
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass, field

# Root directory of project
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

# Default configuration as YAML
DEFAULT_CONFIG_YAML = \"\"\"
{yaml_str}
\"\"\"

# Load default configuration
CONFIG = OmegaConf.create(DEFAULT_CONFIG_YAML)

# Access configuration
def get_config():
    return CONFIG
"""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.join(ROOT_DIR, 'src', 'config'), exist_ok=True)
    
    # Write to file
    with open(os.path.join(ROOT_DIR, 'src', 'config', 'defaults.py'), 'w') as f:
        f.write(content)


# Global root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))