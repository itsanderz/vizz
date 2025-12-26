"""
Atlas Enterprise Configuration

Centralized configuration management with environment variable support,
validation, and enterprise features.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class StorageBackend(str, Enum):
    """Storage backend options."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class CompressionLevel(str, Enum):
    """Compression levels for artifacts."""
    NONE = "none"
    FAST = "fast"
    BALANCED = "balanced"
    MAX = "max"


class TelemetryMode(str, Enum):
    """Telemetry collection modes."""
    DISABLED = "disabled"
    BASIC = "basic"
    FULL = "full"
    CUSTOM = "custom"


class EnterpriseConfig(BaseModel):
    """Enterprise-specific configuration."""

    # Organization
    organization_id: Optional[str] = None
    team_id: Optional[str] = None

    # Access control
    api_key: Optional[str] = Field(default=None, exclude=True)
    require_authentication: bool = False

    # Audit
    audit_logging_enabled: bool = True
    audit_log_path: Optional[Path] = None

    # Data governance
    data_retention_days: int = 365
    auto_cleanup_enabled: bool = False
    pii_detection_enabled: bool = False

    # Compliance
    gdpr_mode: bool = False
    hipaa_mode: bool = False


class StorageConfig(BaseModel):
    """Storage configuration."""

    backend: StorageBackend = StorageBackend.LOCAL
    base_path: Path = Field(default_factory=lambda: Path.cwd() / ".atlas")

    # SQLite
    sqlite_wal_mode: bool = True
    sqlite_cache_size_mb: int = 64
    sqlite_busy_timeout_ms: int = 30000

    # DuckDB
    duckdb_memory_limit_mb: int = 512
    duckdb_threads: int = 4

    # Parquet
    parquet_compression: str = "snappy"
    parquet_row_group_size: int = 100000

    # Artifacts
    artifact_compression: CompressionLevel = CompressionLevel.BALANCED
    max_artifact_size_mb: int = 1024

    # Cloud storage (if applicable)
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    gcs_bucket: Optional[str] = None
    azure_container: Optional[str] = None


class TelemetryConfig(BaseModel):
    """Hardware telemetry configuration."""

    mode: TelemetryMode = TelemetryMode.FULL
    interval_seconds: float = 1.0

    # What to collect
    collect_cpu: bool = True
    collect_memory: bool = True
    collect_gpu: bool = True
    collect_disk: bool = False
    collect_network: bool = False

    # GPU specifics
    gpu_metrics: List[str] = Field(default_factory=lambda: [
        "utilization", "memory", "temperature", "power"
    ])

    # Aggregation
    aggregation_window_seconds: int = 60
    store_raw_samples: bool = False


class VisualizationConfig(BaseModel):
    """Visualization configuration."""

    # Chart defaults
    default_chart_type: str = "line"
    max_points_per_chart: int = 10000
    downsampling_method: str = "lttb"  # Largest-Triangle-Three-Buckets

    # Colors
    color_scheme: str = "atlas"
    use_log_scale_for_loss: bool = True

    # Performance
    webgl_enabled: bool = True
    animation_enabled: bool = True

    # Export
    export_formats: List[str] = Field(default_factory=lambda: [
        "png", "svg", "pdf", "json"
    ])


class AIConfig(BaseModel):
    """AI and Deep Digging configuration."""

    enabled: bool = True

    # Anomaly detection
    anomaly_detection_enabled: bool = True
    anomaly_sensitivity: float = 0.8  # 0-1
    anomaly_min_samples: int = 100

    # Insight generation
    insight_generation_enabled: bool = True
    max_insights_per_run: int = 50
    insight_refresh_interval_seconds: int = 30

    # Model settings
    local_model_path: Optional[Path] = None
    use_vscode_lm_api: bool = True

    # Analysis
    gradient_analysis_enabled: bool = True
    activation_analysis_enabled: bool = True
    loss_landscape_analysis_enabled: bool = False


class PerformanceConfig(BaseModel):
    """Performance tuning configuration."""

    # Buffer
    ring_buffer_size_mb: int = 64
    ring_buffer_slots: int = 8192

    # Batching
    metric_batch_size: int = 1000
    flush_interval_ms: int = 100

    # Harvester
    harvester_workers: int = 2
    harvester_queue_size: int = 100000

    # Memory
    max_memory_mb: int = 2048
    gc_threshold_mb: int = 1024

    # Disk
    sync_mode: str = "normal"  # normal, safe, performance
    compression_threads: int = 2


class AtlasConfig(BaseModel):
    """Master Atlas configuration."""

    # Versioning
    config_version: str = "1.0.0"

    # General
    project_name: str = "default"
    environment: str = "development"  # development, staging, production
    log_level: LogLevel = LogLevel.INFO

    # Sub-configurations
    enterprise: EnterpriseConfig = Field(default_factory=EnterpriseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    # Feature flags
    features: Dict[str, bool] = Field(default_factory=lambda: {
        "deep_digging": True,
        "prompt_to_plot": True,
        "tensor_surgeon": True,
        "run_comparison": True,
        "report_generation": True,
        "jupyter_integration": True,
        "cli": True,
    })

    @classmethod
    def from_env(cls) -> "AtlasConfig":
        """Load configuration from environment variables."""
        config_dict: Dict[str, Any] = {}

        # Map environment variables to config
        env_mapping = {
            "ATLAS_PROJECT": "project_name",
            "ATLAS_ENV": "environment",
            "ATLAS_LOG_LEVEL": "log_level",
            "ATLAS_ORG_ID": ("enterprise", "organization_id"),
            "ATLAS_TEAM_ID": ("enterprise", "team_id"),
            "ATLAS_API_KEY": ("enterprise", "api_key"),
            "ATLAS_STORAGE_PATH": ("storage", "base_path"),
            "ATLAS_S3_BUCKET": ("storage", "s3_bucket"),
            "ATLAS_GPU_ENABLED": ("telemetry", "collect_gpu"),
            "ATLAS_AI_ENABLED": ("ai", "enabled"),
        }

        for env_var, config_path in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                if isinstance(config_path, tuple):
                    # Nested config
                    section, key = config_path
                    if section not in config_dict:
                        config_dict[section] = {}
                    config_dict[section][key] = value
                else:
                    config_dict[config_path] = value

        return cls(**config_dict)

    @classmethod
    def from_file(cls, path: Path) -> "AtlasConfig":
        """Load configuration from a YAML or JSON file."""
        import json

        if not path.exists():
            return cls()

        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                data = json.load(f)

        return cls(**data)

    def save(self, path: Path) -> None:
        """Save configuration to file."""
        import json

        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)


# Global configuration instance
_config: Optional[AtlasConfig] = None


def get_config() -> AtlasConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AtlasConfig.from_env()
    return _config


def set_config(config: AtlasConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def configure(**kwargs: Any) -> AtlasConfig:
    """Configure Atlas with the given options."""
    global _config
    _config = AtlasConfig(**kwargs)
    return _config
