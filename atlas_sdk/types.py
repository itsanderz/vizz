"""
Type definitions for Atlas SDK using Pydantic for runtime type checking.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field, field_validator


class MetricType(str, Enum):
    """Types of metrics that can be logged."""
    SCALAR = "scalar"
    IMAGE = "image"
    AUDIO = "audio"
    TENSOR = "tensor"
    HISTOGRAM = "histogram"
    TEXT = "text"


class ArtifactType(str, Enum):
    """Types of artifacts that can be stored."""
    MODEL = "model"
    CHECKPOINT = "checkpoint"
    TENSOR = "tensor"
    IMAGE = "image"
    AUDIO = "audio"
    CONFIG = "config"
    OTHER = "other"


class RunStatus(str, Enum):
    """Status of an experiment run."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class Metric(BaseModel):
    """A single metric data point."""
    name: str
    value: Union[float, int, str, List[float]]
    step: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metric_type: MetricType = MetricType.SCALAR
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class TensorMetric(BaseModel):
    """A tensor metric with shape and dtype info."""
    name: str
    shape: List[int]
    dtype: str
    step: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    artifact_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class Artifact(BaseModel):
    """An artifact (model, tensor, etc.) to be stored."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    artifact_type: ArtifactType
    path: str
    size_bytes: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    checksum: Optional[str] = None


class HardwareTelemetry(BaseModel):
    """Hardware telemetry data point."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    gpu_count: int = 0
    gpu_metrics: List[Dict[str, Any]] = Field(default_factory=list)


class RunConfig(BaseModel):
    """Configuration for an experiment run."""
    name: str
    project: str = "default"
    tags: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None
    resume: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Run name cannot be empty")
        return v.strip()


class LogData(BaseModel):
    """Data to be logged in a single log call."""
    metrics: Dict[str, Union[float, int, str, List[float]]] = Field(default_factory=dict)
    step: Optional[int] = None
    commit: bool = True

    model_config = {"arbitrary_types_allowed": True}


class InsightVote(BaseModel):
    """User vote on an AI-generated insight."""
    id: UUID = Field(default_factory=uuid4)
    insight_id: UUID
    run_id: UUID
    vote: int  # 1 for upvote, -1 for downvote
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    feedback: Optional[str] = None


class Insight(BaseModel):
    """An AI-generated insight about the run."""
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    insight_type: str
    title: str
    description: str
    severity: str = "info"  # info, warning, critical
    related_metrics: List[str] = Field(default_factory=list)
    visualization_spec: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    votes_up: int = 0
    votes_down: int = 0


class BufferMessage(BaseModel):
    """Message format for the lock-free buffer."""
    msg_type: str  # metric, artifact, telemetry, control
    payload: Dict[str, Any]
    timestamp: float  # Unix timestamp for ordering
    run_id: str

    model_config = {"arbitrary_types_allowed": True}


def numpy_to_list(arr: np.ndarray) -> List[Any]:
    """Convert numpy array to nested list for serialization."""
    return arr.tolist()


def validate_tensor_shape(shape: List[int]) -> bool:
    """Validate tensor shape dimensions."""
    return all(isinstance(d, int) and d > 0 for d in shape)
