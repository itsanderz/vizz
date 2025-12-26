"""
Atlas SDK - The Sovereign Experiment Engine

A local-first, high-performance instrumentation and analysis platform
for deep learning research. Enterprise-grade features with $0 cloud costs.

Example:
    import atlas_sdk as atlas

    with atlas.run("my_experiment", config={"lr": 0.001}) as run:
        for step in range(1000):
            loss = train_step()
            run.log({"loss": loss}, step=step)

Framework Integrations:
    # PyTorch
    from atlas_sdk.integrations import watch_model, AtlasCallback

    # PyTorch Lightning
    from atlas_sdk.integrations import AtlasLightningCallback

    # Hugging Face
    from atlas_sdk.integrations import AtlasTrainerCallback

    # JAX/Flax
    from atlas_sdk.integrations import AtlasJAXLogger

Enterprise Features:
    from atlas_sdk.config import configure, get_config
    from atlas_sdk.security import AuditLogger, DataEncryption
    from atlas_sdk.analysis import DeepDiggingAgent
"""

from atlas_sdk.run import Run, init, log, finish, get_current_run
from atlas_sdk.types import (
    LogData,
    RunConfig,
    Metric,
    Artifact,
    Insight,
    InsightVote,
    MetricType,
    ArtifactType,
    RunStatus,
)
from atlas_sdk.config import configure, get_config, AtlasConfig

__version__ = "0.1.0"
__all__ = [
    # Core API
    "Run",
    "init",
    "log",
    "finish",
    "get_current_run",
    # Types
    "LogData",
    "RunConfig",
    "Metric",
    "Artifact",
    "Insight",
    "InsightVote",
    "MetricType",
    "ArtifactType",
    "RunStatus",
    # Configuration
    "configure",
    "get_config",
    "AtlasConfig",
]


def __getattr__(name: str):
    """Lazy loading for optional modules."""
    if name == "integrations":
        from atlas_sdk import integrations
        return integrations

    if name == "analysis":
        from atlas_sdk import analysis
        return analysis

    if name == "security":
        from atlas_sdk import security
        return security

    raise AttributeError(f"module 'atlas_sdk' has no attribute '{name}'")
