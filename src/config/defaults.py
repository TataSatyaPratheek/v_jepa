import os
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict, fields
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
        # Imports are already at the top of the file, no need to re-import here.

        # Helper function to filter dict to only include valid parameters for a dataclass
        def filter_params(param_dict, dataclass_type):
            valid_fields = {field.name for field in fields(dataclass_type)}
            return {k: v for k, v in param_dict.items() if k in valid_fields}

        # Instantiate leaf dataclasses first, then composite ones.
        # RuntimeConfig
        runtime_params = filter_params(config_dict.get('runtime', {}), RuntimeConfig)
        runtime_config = RuntimeConfig(**runtime_params)

        # TrainingConfig
        training_params = filter_params(config_dict.get('training', {}), TrainingConfig)
        training_config = TrainingConfig(**training_params)

        # OptimizerConfig
        optimizer_params = filter_params(config_dict.get('optimizer', {}), OptimizerConfig)
        optimizer_config = OptimizerConfig(**optimizer_params)

        # MaskingConfig
        masking_params = filter_params(config_dict.get('masking', {}), MaskingConfig)
        masking_config = MaskingConfig(**masking_params)

        # TransformConfig
        transforms_params = filter_params(config_dict.get('transforms', {}), TransformConfig)
        transforms_config = TransformConfig(**transforms_params)

        # VideoDatasetConfig
        dataset_params = filter_params(config_dict.get('dataset', {}), VideoDatasetConfig)
        dataset_config = VideoDatasetConfig(**dataset_params)

        # ModelConfig (VJEPAConfig with nested EncoderConfig and PredictorConfig)
        model_dict = config_dict.get('model', {})

        encoder_conf_dict = model_dict.get('encoder_config', {})
        encoder_config_obj = EncoderConfig(**filter_params(encoder_conf_dict, EncoderConfig))

        predictor_conf_dict = model_dict.get('predictor_config', {})
        predictor_config_obj = PredictorConfig(**filter_params(predictor_conf_dict, PredictorConfig))

        # Get direct VJEPAConfig parameters (excluding nested ones we just handled)
        vjepa_direct_params_dict = {
            k: v for k, v in model_dict.items()
            if k not in ['encoder_config', 'predictor_config']
        }
        vjepa_filtered_direct_params = filter_params(vjepa_direct_params_dict, VJEPAConfig)

        model_config = VJEPAConfig(
            **vjepa_filtered_direct_params,
            encoder_config=encoder_config_obj,
            predictor_config=predictor_config_obj
        )

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
@hydra.main(config_path="../config", config_name="defaults", version_base="1.3")
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
    # Memory optimizations for M1
    # 1. Runtime settings
    config.runtime.precision = 16  # Use half precision
    config.runtime.threads["omp"] = 2  # Limit OpenMP threads
    config.runtime.threads["dataloader"] = 2  # Limit DataLoader workers
    config.runtime.threads["io"] = 2  # Limit IO threads
    
    # 2. Model settings
    if config.model.encoder_config is None: # Ensure encoder_config exists if we are to modify it
        config.model.encoder_config = EncoderConfig()
    
    config.model.encoder_config.use_half_precision = True
    config.model.encoder_config.use_gradient_checkpointing = True
    config.model.share_parameters = True  # Share encoder parameters to save memory
    
    # 3. Training settings
    config.training.batch_size = 2
    config.training.accumulation_steps = 4  # Use gradient accumulation
    config.training.amp = True  # Use automatic mixed precision
    config.training.empty_cache_freq = 5  # Frequently clear cache
    config.training.optimize_for_m1 = True
    
    # 4. Dataset settings
    config.dataset.use_half_precision = True
    config.dataset.optimize_for_m1 = True
    
    # 5. Optimizer settings
    config.optimizer.fused = True  # Use fused implementations
    config.optimizer.grad_accumulation_steps = config.training.accumulation_steps # Sync with training
    
    return config


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