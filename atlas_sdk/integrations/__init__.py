"""
Atlas Framework Integrations

Seamless integration with popular ML frameworks:
- PyTorch
- PyTorch Lightning
- JAX/Flax
- Hugging Face Transformers
- TensorFlow/Keras
"""

from atlas_sdk.integrations.pytorch import (
    AtlasCallback,
    watch_model,
    log_gradients,
    log_parameters,
)
from atlas_sdk.integrations.lightning import AtlasLightningCallback
from atlas_sdk.integrations.huggingface import AtlasTrainerCallback
from atlas_sdk.integrations.jax import AtlasJAXLogger

__all__ = [
    # PyTorch
    "AtlasCallback",
    "watch_model",
    "log_gradients",
    "log_parameters",
    # Lightning
    "AtlasLightningCallback",
    # Hugging Face
    "AtlasTrainerCallback",
    # JAX
    "AtlasJAXLogger",
]
