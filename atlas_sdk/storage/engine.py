"""
Unified Storage Engine combining SQLite, DuckDB, and Safetensors.

Provides a single interface for all Atlas storage operations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from atlas_sdk.storage.artifacts import ArtifactStore
from atlas_sdk.storage.metadata import MetadataStore
from atlas_sdk.storage.timeseries import TimeSeriesStore
from atlas_sdk.types import (
    Artifact,
    ArtifactType,
    Insight,
    InsightVote,
    Metric,
    MetricType,
    RunConfig,
    RunStatus,
)


class StorageEngine:
    """
    Unified storage engine for Atlas.

    Coordinates between:
    - MetadataStore (SQLite) for run config, tags, votes
    - TimeSeriesStore (DuckDB/Parquet) for metrics
    - ArtifactStore (Safetensors) for tensors
    """

    ATLAS_DIR_NAME = ".atlas"

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        read_only: bool = False,
    ):
        """
        Initialize storage engine.

        Args:
            base_dir: Base directory for .atlas folder (default: cwd)
            run_id: Run ID for time series and artifacts (required for logging)
            read_only: If True, open in read-only mode
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.atlas_dir = self.base_dir / self.ATLAS_DIR_NAME
        self.run_id = run_id
        self.read_only = read_only

        # Create atlas directory structure
        if not read_only:
            self.atlas_dir.mkdir(parents=True, exist_ok=True)
            (self.atlas_dir / "metrics").mkdir(exist_ok=True)
            (self.atlas_dir / "artifacts").mkdir(exist_ok=True)

        # Initialize stores
        self.metadata = MetadataStore(self.atlas_dir / "atlas.db")

        self._timeseries: Optional[TimeSeriesStore] = None
        self._artifacts: Optional[ArtifactStore] = None

        if run_id:
            self._timeseries = TimeSeriesStore(
                self.atlas_dir,
                run_id,
                read_only=read_only,
            )
            self._artifacts = ArtifactStore(self.atlas_dir, run_id)

    @property
    def timeseries(self) -> TimeSeriesStore:
        """Get time series store (requires run_id)."""
        if self._timeseries is None:
            raise RuntimeError("TimeSeriesStore requires run_id to be set")
        return self._timeseries

    @property
    def artifacts(self) -> ArtifactStore:
        """Get artifact store (requires run_id)."""
        if self._artifacts is None:
            raise RuntimeError("ArtifactStore requires run_id to be set")
        return self._artifacts

    # ==================== Run Operations ====================

    def create_run(self, run_id: str, config: RunConfig) -> None:
        """Create a new run."""
        self.metadata.create_run(run_id, config)
        self.run_id = run_id
        self._timeseries = TimeSeriesStore(self.atlas_dir, run_id)
        self._artifacts = ArtifactStore(self.atlas_dir, run_id)

    def finish_run(
        self,
        status: RunStatus = RunStatus.COMPLETED,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Mark run as finished."""
        if not self.run_id:
            return
        self.metadata.update_run_status(self.run_id, status, duration_seconds)
        if self._timeseries:
            self._timeseries.close()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run by ID."""
        return self.metadata.get_run(run_id)

    def list_runs(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """List runs with optional filtering."""
        return self.metadata.list_runs(**kwargs)

    # ==================== Metric Operations ====================

    def log_metric(
        self,
        name: str,
        value: Union[float, int],
        step: int,
        metric_type: MetricType = MetricType.SCALAR,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single metric."""
        metric = Metric(
            name=name,
            value=value,
            step=step,
            metric_type=metric_type,
            metadata=metadata or {},
        )
        self.timeseries.log(metric)

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: int,
    ) -> None:
        """Log multiple metrics at once."""
        metric_list = [
            Metric(name=name, value=value, step=step)
            for name, value in metrics.items()
        ]
        self.timeseries.log_batch(metric_list)

    def get_metric(self, name: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Get metric values."""
        return self.timeseries.get_metric(name, **kwargs)

    def get_metric_statistics(self, name: str) -> Dict[str, Any]:
        """Get statistical summary for a metric."""
        return self.timeseries.get_statistics(name)

    def list_metrics(self) -> List[str]:
        """List all metric names."""
        return self.timeseries.list_metrics()

    def query_metrics(self, sql: str) -> List[Dict[str, Any]]:
        """Execute raw SQL query on metrics."""
        return self.timeseries.query(sql)

    # ==================== Tensor/Artifact Operations ====================

    def log_tensor(
        self,
        name: str,
        tensor: np.ndarray,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """Log a tensor artifact."""
        artifact = self.artifacts.save_tensor(name, tensor, step, metadata)
        self.metadata.create_artifact(self.run_id, artifact)
        return artifact

    def log_checkpoint(
        self,
        name: str,
        tensors: Dict[str, np.ndarray],
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """Log a checkpoint with multiple tensors."""
        artifact = self.artifacts.save_checkpoint(name, tensors, step, metadata)
        self.metadata.create_artifact(self.run_id, artifact)
        return artifact

    def load_tensor(self, **kwargs: Any) -> np.ndarray:
        """Load a tensor from storage."""
        return self.artifacts.load_tensor(**kwargs)

    def get_tensor_info(self, path: str) -> Dict[str, Dict[str, Any]]:
        """Get tensor metadata without loading."""
        return self.artifacts.get_tensor_info(path)

    # ==================== Insight Operations ====================

    def save_insight(self, insight: Insight) -> None:
        """Save an AI-generated insight."""
        self.metadata.create_insight(insight)

    def get_insights(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Get insights for current run."""
        if not self.run_id:
            return []
        return self.metadata.get_insights(self.run_id, **kwargs)

    def vote_insight(self, vote: InsightVote) -> None:
        """Record a vote on an insight."""
        self.metadata.record_vote(vote)

    def get_vote_stats(self) -> Dict[str, Any]:
        """Get vote statistics for recommendation training."""
        return self.metadata.get_vote_stats()

    # ==================== Comparison Operations ====================

    def compare_runs(
        self,
        run_ids: List[str],
        metric_name: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Compare a metric across multiple runs."""
        result = {}
        for run_id in run_ids:
            store = TimeSeriesStore(self.atlas_dir, run_id, read_only=True)
            result[run_id] = store.get_metric(metric_name)
            store.close()
        return result

    def diff_configs(
        self,
        run_id_a: str,
        run_id_b: str,
    ) -> Dict[str, Any]:
        """Diff configurations between two runs."""
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)

        if not run_a or not run_b:
            raise ValueError("One or both runs not found")

        config_a = run_a.get("config", {})
        config_b = run_b.get("config", {})

        # Find differences
        all_keys = set(config_a.keys()) | set(config_b.keys())
        diff = {}

        for key in all_keys:
            val_a = config_a.get(key)
            val_b = config_b.get(key)
            if val_a != val_b:
                diff[key] = {"run_a": val_a, "run_b": val_b}

        return {
            "run_a": run_id_a,
            "run_b": run_id_b,
            "differences": diff,
            "only_in_a": [k for k in config_a if k not in config_b],
            "only_in_b": [k for k in config_b if k not in config_a],
        }

    # ==================== Export Operations ====================

    def export_run(self, run_id: str, output_dir: Path) -> None:
        """Export a complete run to a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export metadata
        run = self.get_run(run_id)
        if run:
            with open(output_dir / "metadata.json", "w") as f:
                json.dump(run, f, indent=2, default=str)

        # Export metrics to Parquet
        store = TimeSeriesStore(self.atlas_dir, run_id, read_only=True)
        store.export_to_parquet(output_dir / "metrics.parquet")
        store.close()

        # Export insights
        insights = self.metadata.get_insights(run_id)
        if insights:
            with open(output_dir / "insights.json", "w") as f:
                json.dump(insights, f, indent=2, default=str)

    def get_atlas_info(self) -> Dict[str, Any]:
        """Get information about the .atlas directory."""
        total_runs = len(self.list_runs())

        # Calculate total size
        total_size = 0
        for root, dirs, files in os.walk(self.atlas_dir):
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))

        return {
            "path": str(self.atlas_dir),
            "total_runs": total_runs,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    def close(self) -> None:
        """Close all stores."""
        if self._timeseries:
            self._timeseries.close()
        self.metadata.close()


def open_atlas(
    base_dir: Optional[Path] = None,
    read_only: bool = True,
) -> StorageEngine:
    """
    Open an existing .atlas directory for reading.

    Args:
        base_dir: Directory containing .atlas folder
        read_only: Open in read-only mode (default True)

    Returns:
        StorageEngine instance
    """
    return StorageEngine(base_dir, read_only=read_only)
