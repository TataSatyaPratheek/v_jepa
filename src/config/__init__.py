"""Configuration utilities for V-JEPA."""

from .defaults import (
    VJEPASystemConfig,
    get_default_config,
    apply_m1_optimizations,
    TrainingConfig,
    RuntimeConfig
)

__all__ = [
    'VJEPASystemConfig',
    'get_default_config',
    'apply_m1_optimizations',
    'TrainingConfig',
    'RuntimeConfig'
]