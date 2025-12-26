"""
Run management for Atlas SDK.

Provides the main user-facing API for experiment tracking.
"""

from __future__ import annotations

import atexit
import os
import time
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from uuid import uuid4

import numpy as np

from atlas_sdk.harvester import HarvesterManager
from atlas_sdk.storage.engine import StorageEngine
from atlas_sdk.telemetry import TelemetryAggregator, TelemetryCollector
from atlas_sdk.types import (
    HardwareTelemetry,
    LogData,
    MetricType,
    RunConfig,
    RunStatus,
)


# Global state
_current_run: Optional["Run"] = None
_runs_lock = threading.Lock()


class Run:
    """
    An Atlas experiment run.

    Usage:
        with atlas.run("my_experiment") as run:
            for step in range(1000):
                loss = train_step()
                run.log({"loss": loss}, step=step)

    Or without context manager:
        run = atlas.init("my_experiment")
        for step in range(1000):
            run.log({"loss": loss}, step=step)
        run.finish()
    """

    def __init__(
        self,
        name: str,
        project: str = "default",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        base_dir: Optional[Path] = None,
        resume: bool = False,
        log_telemetry: bool = True,
        telemetry_interval: float = 1.0,
    ):
        """
        Initialize a new run.

        Args:
            name: Run name
            project: Project name for grouping
            config: Hyperparameters and configuration
            tags: Tags for filtering
            notes: Optional notes
            base_dir: Base directory for .atlas folder
            resume: Resume an existing run
            log_telemetry: Auto-log hardware telemetry
            telemetry_interval: Telemetry sampling interval in seconds
        """
        self.id = str(uuid4())
        self.name = name
        self.project = project
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

        self._config = RunConfig(
            name=name,
            project=project,
            config=config or {},
            tags=tags or [],
            notes=notes,
            resume=resume,
        )

        self._step = 0
        self._start_time = time.time()
        self._status = RunStatus.RUNNING
        self._finished = False

        # Initialize storage
        self._storage = StorageEngine(
            base_dir=self.base_dir,
            run_id=self.id,
        )
        self._storage.create_run(self.id, self._config)

        # Initialize harvester for background I/O
        self._harvester = HarvesterManager(
            run_id=self.id,
            atlas_dir=self._storage.atlas_dir,
        )
        self._harvester.start()

        # Initialize telemetry
        self._telemetry_collector: Optional[TelemetryCollector] = None
        self._telemetry_aggregator: Optional[TelemetryAggregator] = None

        if log_telemetry:
            self._telemetry_aggregator = TelemetryAggregator(window_size=60)
            self._telemetry_collector = TelemetryCollector(
                callback=self._handle_telemetry,
                interval=telemetry_interval,
            )
            self._telemetry_collector.start()

        # Register cleanup
        atexit.register(self._cleanup)

    def __enter__(self) -> "Run":
        """Context manager entry."""
        global _current_run
        with _runs_lock:
            _current_run = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        global _current_run

        if exc_type is not None:
            self.finish(status=RunStatus.FAILED)
        else:
            self.finish(status=RunStatus.COMPLETED)

        with _runs_lock:
            if _current_run is self:
                _current_run = None

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """
        Log metrics.

        This call is non-blocking and returns in < 5µs.

        Args:
            data: Dictionary of metric name -> value
            step: Step number (auto-increments if not provided)
            commit: Whether to commit immediately (for batching)
        """
        if self._finished:
            return

        if step is None:
            step = self._step
            self._step += 1
        else:
            self._step = max(self._step, step + 1)

        # Send to harvester
        metrics = []
        for name, value in data.items():
            if isinstance(value, (int, float)):
                metrics.append({
                    "name": name,
                    "value": float(value),
                    "step": step,
                    "metric_type": "scalar",
                })
            elif isinstance(value, np.ndarray):
                # For tensors, just log shape info
                # Full tensor saved separately
                metrics.append({
                    "name": f"{name}/shape",
                    "value": float(np.prod(value.shape)),
                    "step": step,
                    "metric_type": "scalar",
                    "metadata": {"shape": list(value.shape)},
                })

        if metrics:
            self._harvester.write({
                "msg_type": "metrics_batch",
                "metrics": metrics,
            })

    def log_tensor(
        self,
        name: str,
        tensor: np.ndarray,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a tensor artifact.

        Tensors are saved as safetensors files for zero-copy access.

        Args:
            name: Tensor name
            tensor: NumPy array
            step: Step number
            metadata: Optional metadata
        """
        if self._finished:
            return

        if step is None:
            step = self._step

        # Save directly via storage (bypasses harvester for large data)
        self._storage.log_tensor(name, tensor, step, metadata)

    def log_histogram(
        self,
        name: str,
        values: Union[np.ndarray, List[float]],
        step: Optional[int] = None,
        bins: int = 64,
    ) -> None:
        """
        Log a histogram of values.

        Args:
            name: Metric name
            values: Array of values
            step: Step number
            bins: Number of histogram bins
        """
        if self._finished:
            return

        if step is None:
            step = self._step

        arr = np.array(values)
        hist, bin_edges = np.histogram(arr, bins=bins)

        self._harvester.write({
            "msg_type": "metric",
            "name": name,
            "value": float(arr.mean()),
            "step": step,
            "metric_type": "histogram",
            "metadata": {
                "histogram": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "std": float(arr.std()),
            },
        })

    def log_image(
        self,
        name: str,
        image: np.ndarray,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """
        Log an image.

        Args:
            name: Image name
            image: Image array (HWC or HW format)
            step: Step number
            caption: Optional caption
        """
        if self._finished:
            return

        if step is None:
            step = self._step

        # Save as artifact
        self._storage.log_tensor(
            name,
            image,
            step,
            metadata={"type": "image", "caption": caption},
        )

    def log_attention(
        self,
        name: str,
        attention: np.ndarray,
        step: Optional[int] = None,
        layer: int = 0,
        head_names: Optional[List[str]] = None,
        token_labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log attention patterns for visualization.

        Attention tensors are stored efficiently and can be visualized
        as multi-head attention heatmaps in the Atlas VS Code extension.

        Args:
            name: Attention tensor name (e.g., "self_attention")
            attention: Attention weights [num_heads, seq_len, seq_len]
                      or [batch, num_heads, seq_len, seq_len]
            step: Step number
            layer: Layer index for multi-layer attention
            head_names: Optional names for each attention head
            token_labels: Optional labels for sequence positions
            metadata: Additional metadata

        Example:
            # Log attention from a transformer layer
            attn_weights = model.get_attention_weights()  # [12, 64, 64]
            run.log_attention(
                "layer_0_attention",
                attn_weights,
                step=step,
                layer=0,
                head_names=[f"Head {i}" for i in range(12)],
            )
        """
        if self._finished:
            return

        if step is None:
            step = self._step

        arr = np.array(attention)

        # Validate shape
        if arr.ndim == 3:
            num_heads, seq_len_q, seq_len_k = arr.shape
            batch_size = 1
        elif arr.ndim == 4:
            batch_size, num_heads, seq_len_q, seq_len_k = arr.shape
        else:
            raise ValueError(
                f"Attention tensor must be 3D [heads, seq, seq] or "
                f"4D [batch, heads, seq, seq], got shape {arr.shape}"
            )

        # Build metadata
        attn_metadata = {
            "type": "attention",
            "layer": layer,
            "num_heads": num_heads,
            "seq_len_query": seq_len_q,
            "seq_len_key": seq_len_k,
            "batch_size": batch_size,
        }

        if head_names:
            attn_metadata["head_names"] = head_names
        if token_labels:
            attn_metadata["token_labels"] = token_labels
        if metadata:
            attn_metadata.update(metadata)

        # Save tensor
        self._storage.log_tensor(
            f"attention/{name}/layer_{layer}",
            arr,
            step,
            metadata=attn_metadata,
        )

        # Log summary statistics as metrics
        self.log({
            f"{name}/layer_{layer}/mean": float(arr.mean()),
            f"{name}/layer_{layer}/max": float(arr.max()),
            f"{name}/layer_{layer}/entropy": float(self._compute_attention_entropy(arr)),
        }, step=step)

    def log_hidden_states(
        self,
        name: str,
        hidden_states: np.ndarray,
        step: Optional[int] = None,
        layer: int = 0,
        compute_similarity: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log hidden states for visualization.

        Hidden states can be visualized as similarity matrices or
        activation maps in the Atlas VS Code extension.

        Args:
            name: Hidden state tensor name
            hidden_states: Hidden states [batch, seq_len, hidden_dim]
                          or [seq_len, hidden_dim]
            step: Step number
            layer: Layer index
            compute_similarity: Compute and log similarity matrix
            metadata: Additional metadata

        Example:
            # Log hidden states from a transformer layer
            hidden = model.get_hidden_states()  # [batch, seq, dim]
            run.log_hidden_states(
                "encoder_hidden",
                hidden,
                step=step,
                layer=0,
            )
        """
        if self._finished:
            return

        if step is None:
            step = self._step

        arr = np.array(hidden_states)

        # Validate shape
        if arr.ndim == 2:
            seq_len, hidden_dim = arr.shape
            batch_size = 1
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            batch_size, seq_len, hidden_dim = arr.shape
        else:
            raise ValueError(
                f"Hidden states must be 2D [seq, dim] or "
                f"3D [batch, seq, dim], got shape {arr.shape}"
            )

        # Build metadata
        state_metadata = {
            "type": "hidden_state",
            "layer": layer,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
        }

        if metadata:
            state_metadata.update(metadata)

        # Save hidden states tensor
        self._storage.log_tensor(
            f"hidden_states/{name}/layer_{layer}",
            arr,
            step,
            metadata=state_metadata,
        )

        # Compute and save similarity matrix if requested
        if compute_similarity:
            # Use first batch for similarity
            h = arr[0]  # [seq_len, hidden_dim]

            # Normalize for cosine similarity
            norms = np.linalg.norm(h, axis=1, keepdims=True)
            h_normalized = h / (norms + 1e-8)

            # Compute similarity matrix
            similarity = np.dot(h_normalized, h_normalized.T)

            self._storage.log_tensor(
                f"similarity/{name}/layer_{layer}",
                similarity,
                step,
                metadata={
                    "type": "similarity_matrix",
                    "layer": layer,
                    "metric": "cosine",
                    "seq_len": seq_len,
                },
            )

            # Log summary statistics
            # Get off-diagonal elements
            mask = ~np.eye(seq_len, dtype=bool)
            off_diag = similarity[mask]

            self.log({
                f"{name}/layer_{layer}/similarity_mean": float(off_diag.mean()),
                f"{name}/layer_{layer}/similarity_std": float(off_diag.std()),
            }, step=step)

    def log_gradients(
        self,
        name: str,
        gradients: Dict[str, np.ndarray],
        step: Optional[int] = None,
        log_histograms: bool = True,
    ) -> None:
        """
        Log gradient statistics for visualization.

        Args:
            name: Name for this gradient snapshot
            gradients: Dict of parameter_name -> gradient array
            step: Step number
            log_histograms: Whether to log gradient histograms

        Example:
            # Log gradients from PyTorch model
            grads = {name: p.grad.cpu().numpy()
                    for name, p in model.named_parameters()
                    if p.grad is not None}
            run.log_gradients("training", grads, step=step)
        """
        if self._finished:
            return

        if step is None:
            step = self._step

        grad_norms = {}
        total_norm_sq = 0.0

        for param_name, grad in gradients.items():
            arr = np.array(grad)
            norm = float(np.linalg.norm(arr))
            grad_norms[param_name] = norm
            total_norm_sq += norm ** 2

            if log_histograms:
                self.log_histogram(
                    f"gradients/{name}/{param_name}",
                    arr.flatten(),
                    step=step,
                )

        total_norm = np.sqrt(total_norm_sq)

        # Log gradient flow metrics
        self.log({
            f"gradients/{name}/total_norm": total_norm,
            f"gradients/{name}/max_norm": max(grad_norms.values()) if grad_norms else 0.0,
            f"gradients/{name}/num_params": len(gradients),
        }, step=step)

    def _compute_attention_entropy(self, attention: np.ndarray) -> float:
        """Compute average entropy of attention distributions."""
        # Flatten to [num_distributions, seq_len]
        flat = attention.reshape(-1, attention.shape[-1])

        # Compute entropy for each distribution
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -np.sum(flat * np.log(flat + eps), axis=1)

        return float(np.mean(entropy))

    def set_config(self, config: Dict[str, Any]) -> None:
        """Update run configuration."""
        self._config.config.update(config)

    def add_tags(self, tags: List[str]) -> None:
        """Add tags to the run."""
        self._config.tags.extend(tags)

    def set_notes(self, notes: str) -> None:
        """Set run notes."""
        self._config.notes = notes

    def finish(
        self,
        status: RunStatus = RunStatus.COMPLETED,
        quiet: bool = False,
    ) -> None:
        """
        Finish the run.

        Args:
            status: Final status
            quiet: Suppress output
        """
        if self._finished:
            return

        self._finished = True
        self._status = status

        duration = time.time() - self._start_time

        # Stop telemetry
        if self._telemetry_collector:
            self._telemetry_collector.stop()

            # Flush remaining telemetry
            if self._telemetry_aggregator:
                stats = self._telemetry_aggregator.flush()
                if stats:
                    self._log_telemetry_stats(stats, self._step)

        # Stop harvester
        if self._harvester:
            self._harvester.write({
                "msg_type": "control",
                "command": "finish",
                "status": status.value,
                "duration": duration,
            })
            time.sleep(0.2)  # Let it process
            self._harvester.stop()

        # Update storage
        self._storage.finish_run(status, duration)
        self._storage.close()

        if not quiet:
            print(f"Atlas run finished: {self.name} ({status.value})")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Steps: {self._step}")
            print(f"  Data: {self._storage.atlas_dir}")

    def _handle_telemetry(self, telemetry: HardwareTelemetry) -> None:
        """Handle incoming telemetry data."""
        if self._telemetry_aggregator:
            stats = self._telemetry_aggregator.add(telemetry)
            if stats:
                self._log_telemetry_stats(stats, self._step)

    def _log_telemetry_stats(self, stats: Dict[str, Any], step: int) -> None:
        """Log aggregated telemetry stats."""
        self._harvester.write({
            "msg_type": "telemetry",
            "data": stats,
            "step": step,
        })

    def _cleanup(self) -> None:
        """Cleanup on exit."""
        if not self._finished:
            self.finish(status=RunStatus.INTERRUPTED, quiet=True)

    @property
    def config(self) -> Dict[str, Any]:
        """Get run configuration."""
        return self._config.config

    @property
    def step(self) -> int:
        """Get current step."""
        return self._step

    @property
    def status(self) -> RunStatus:
        """Get run status."""
        return self._status


def init(
    name: str,
    project: str = "default",
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    **kwargs: Any,
) -> Run:
    """
    Initialize a new Atlas run.

    This is the main entry point for the SDK.

    Args:
        name: Run name
        project: Project name
        config: Configuration dict
        tags: List of tags
        **kwargs: Additional Run arguments

    Returns:
        Run instance
    """
    global _current_run

    run = Run(
        name=name,
        project=project,
        config=config,
        tags=tags,
        **kwargs,
    )

    with _runs_lock:
        _current_run = run

    return run


@contextmanager
def run(
    name: str,
    project: str = "default",
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Iterator[Run]:
    """
    Context manager for creating a run.

    Usage:
        with atlas.run("my_experiment") as r:
            r.log({"loss": 0.5})
    """
    r = init(name, project, config, **kwargs)
    try:
        yield r
    except Exception:
        r.finish(status=RunStatus.FAILED)
        raise
    else:
        r.finish(status=RunStatus.COMPLETED)


def log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
) -> None:
    """
    Log metrics to the current run.

    Args:
        data: Dictionary of metric name -> value
        step: Step number
        commit: Whether to commit immediately
    """
    current = get_current_run()
    if current:
        current.log(data, step=step, commit=commit)
    else:
        raise RuntimeError("No active run. Call atlas.init() first.")


def finish(status: RunStatus = RunStatus.COMPLETED) -> None:
    """Finish the current run."""
    current = get_current_run()
    if current:
        current.finish(status=status)


def get_current_run() -> Optional[Run]:
    """Get the current active run."""
    with _runs_lock:
        return _current_run
