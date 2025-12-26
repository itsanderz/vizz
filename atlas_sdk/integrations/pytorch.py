"""
PyTorch Integration

Deep integration with PyTorch including:
- Automatic gradient logging
- Model parameter statistics
- Activation histograms
- Memory tracking
- Distributed training support
"""

from __future__ import annotations

import functools
import threading
import weakref
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

# Lazy imports for optional dependencies
torch = None
nn = None


def _ensure_torch():
    """Lazily import torch."""
    global torch, nn
    if torch is None:
        import torch as _torch
        import torch.nn as _nn
        torch = _torch
        nn = _nn


@dataclass
class GradientStats:
    """Statistics for a parameter's gradient."""
    name: str
    shape: Tuple[int, ...]
    norm: float
    mean: float
    std: float
    min: float
    max: float
    num_zeros: int
    has_nan: bool
    has_inf: bool


@dataclass
class ParameterStats:
    """Statistics for a model parameter."""
    name: str
    shape: Tuple[int, ...]
    norm: float
    mean: float
    std: float
    min: float
    max: float
    num_zeros: int
    sparsity: float


@dataclass
class WatchConfig:
    """Configuration for model watching."""
    log_gradients: bool = True
    log_parameters: bool = True
    log_activations: bool = False
    gradient_freq: int = 1  # Log every N steps
    parameter_freq: int = 100
    activation_freq: int = 100
    histogram_bins: int = 64
    log_histogram: bool = True
    log_layer_stats: bool = True
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)


class AtlasCallback:
    """
    PyTorch training callback for Atlas integration.

    Usage:
        callback = AtlasCallback(run, log_gradients=True)

        for batch in dataloader:
            loss = model(batch)
            loss.backward()

            callback.on_backward_end(step)

            optimizer.step()
            callback.on_step_end(step, {"loss": loss.item()})
    """

    def __init__(
        self,
        run: Any,  # atlas.Run
        config: Optional[WatchConfig] = None,
    ):
        _ensure_torch()
        self.run = run
        self.config = config or WatchConfig()
        self._step = 0
        self._model: Optional[Any] = None
        self._hooks: List[Any] = []

    def watch(self, model: Any) -> None:
        """
        Start watching a model for gradients and parameters.

        Args:
            model: PyTorch model (nn.Module)
        """
        _ensure_torch()
        self._model = model

        if self.config.log_activations:
            self._register_activation_hooks(model)

    def on_backward_end(self, step: Optional[int] = None) -> None:
        """Call after loss.backward() to log gradients."""
        if step is not None:
            self._step = step

        if not self.config.log_gradients:
            return

        if self._step % self.config.gradient_freq != 0:
            return

        if self._model is None:
            return

        gradient_stats = log_gradients(self._model, return_stats=True)
        self._log_gradient_summary(gradient_stats)

    def on_step_end(
        self,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Call after optimizer.step()."""
        if step is not None:
            self._step = step

        # Log metrics
        if metrics:
            self.run.log(metrics, step=self._step)

        # Log parameter stats periodically
        if self.config.log_parameters and self._step % self.config.parameter_freq == 0:
            if self._model is not None:
                param_stats = log_parameters(self._model, return_stats=True)
                self._log_parameter_summary(param_stats)

        self._step += 1

    def _register_activation_hooks(self, model: Any) -> None:
        """Register forward hooks to capture activations."""
        _ensure_torch()

        def hook_fn(name: str):
            def fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._log_activation(name, output)
            return fn

        for name, module in model.named_modules():
            if self._should_log_layer(name):
                hook = module.register_forward_hook(hook_fn(name))
                self._hooks.append(hook)

    def _should_log_layer(self, name: str) -> bool:
        """Check if layer should be logged based on patterns."""
        if self.config.exclude_patterns:
            for pattern in self.config.exclude_patterns:
                if pattern in name:
                    return False

        if self.config.include_patterns:
            for pattern in self.config.include_patterns:
                if pattern in name:
                    return True
            return False

        return True

    def _log_gradient_summary(self, stats: List[GradientStats]) -> None:
        """Log gradient summary metrics."""
        if not stats:
            return

        # Aggregate stats
        norms = [s.norm for s in stats if not (s.has_nan or s.has_inf)]
        total_norm = np.sqrt(sum(n**2 for n in norms)) if norms else 0

        self.run.log({
            "gradients/total_norm": total_norm,
            "gradients/max_norm": max(norms) if norms else 0,
            "gradients/mean_norm": np.mean(norms) if norms else 0,
            "gradients/num_layers": len(stats),
            "gradients/num_nan": sum(1 for s in stats if s.has_nan),
            "gradients/num_inf": sum(1 for s in stats if s.has_inf),
        }, step=self._step)

        # Log per-layer stats if enabled
        if self.config.log_layer_stats:
            for s in stats[:20]:  # Limit to first 20 layers
                self.run.log({
                    f"gradients/{s.name}/norm": s.norm,
                    f"gradients/{s.name}/mean": s.mean,
                }, step=self._step)

    def _log_parameter_summary(self, stats: List[ParameterStats]) -> None:
        """Log parameter summary metrics."""
        if not stats:
            return

        norms = [s.norm for s in stats]
        total_norm = np.sqrt(sum(n**2 for n in norms))

        self.run.log({
            "parameters/total_norm": total_norm,
            "parameters/max_norm": max(norms),
            "parameters/mean_sparsity": np.mean([s.sparsity for s in stats]),
            "parameters/num_layers": len(stats),
        }, step=self._step)

    def _log_activation(self, name: str, tensor: Any) -> None:
        """Log activation statistics."""
        if self._step % self.config.activation_freq != 0:
            return

        with torch.no_grad():
            self.run.log({
                f"activations/{name}/mean": tensor.mean().item(),
                f"activations/{name}/std": tensor.std().item(),
                f"activations/{name}/max": tensor.max().item(),
                f"activations/{name}/min": tensor.min().item(),
            }, step=self._step)

    def close(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def watch_model(
    model: Any,
    run: Any,
    config: Optional[WatchConfig] = None,
) -> AtlasCallback:
    """
    Convenience function to start watching a model.

    Args:
        model: PyTorch model
        run: Atlas run instance
        config: Optional watch configuration

    Returns:
        AtlasCallback instance
    """
    callback = AtlasCallback(run, config)
    callback.watch(model)
    return callback


def log_gradients(
    model: Any,
    run: Optional[Any] = None,
    step: Optional[int] = None,
    return_stats: bool = False,
) -> Optional[List[GradientStats]]:
    """
    Log gradient statistics for all model parameters.

    Args:
        model: PyTorch model
        run: Atlas run instance (optional if return_stats=True)
        step: Current step
        return_stats: If True, return stats instead of logging

    Returns:
        List of GradientStats if return_stats=True
    """
    _ensure_torch()
    stats = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach()

        with torch.no_grad():
            grad_stats = GradientStats(
                name=name,
                shape=tuple(grad.shape),
                norm=grad.norm().item(),
                mean=grad.mean().item(),
                std=grad.std().item(),
                min=grad.min().item(),
                max=grad.max().item(),
                num_zeros=int((grad == 0).sum().item()),
                has_nan=bool(torch.isnan(grad).any().item()),
                has_inf=bool(torch.isinf(grad).any().item()),
            )
            stats.append(grad_stats)

    if return_stats:
        return stats

    if run is not None:
        for s in stats:
            run.log({
                f"gradients/{s.name}/norm": s.norm,
                f"gradients/{s.name}/mean": s.mean,
                f"gradients/{s.name}/std": s.std,
            }, step=step)

    return None


def log_parameters(
    model: Any,
    run: Optional[Any] = None,
    step: Optional[int] = None,
    return_stats: bool = False,
) -> Optional[List[ParameterStats]]:
    """
    Log parameter statistics for all model parameters.

    Args:
        model: PyTorch model
        run: Atlas run instance (optional if return_stats=True)
        step: Current step
        return_stats: If True, return stats instead of logging

    Returns:
        List of ParameterStats if return_stats=True
    """
    _ensure_torch()
    stats = []

    for name, param in model.named_parameters():
        with torch.no_grad():
            data = param.detach()
            num_elements = data.numel()
            num_zeros = int((data == 0).sum().item())

            param_stats = ParameterStats(
                name=name,
                shape=tuple(data.shape),
                norm=data.norm().item(),
                mean=data.mean().item(),
                std=data.std().item(),
                min=data.min().item(),
                max=data.max().item(),
                num_zeros=num_zeros,
                sparsity=num_zeros / num_elements if num_elements > 0 else 0,
            )
            stats.append(param_stats)

    if return_stats:
        return stats

    if run is not None:
        for s in stats:
            run.log({
                f"parameters/{s.name}/norm": s.norm,
                f"parameters/{s.name}/sparsity": s.sparsity,
            }, step=step)

    return None


@contextmanager
def profile_forward(
    run: Any,
    model: Any,
    step: int,
) -> Iterator[None]:
    """
    Context manager to profile a forward pass.

    Usage:
        with profile_forward(run, model, step):
            output = model(input)
    """
    _ensure_torch()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        import time
        start_time = time.perf_counter()

    yield

    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

    run.log({"timing/forward_ms": elapsed_ms}, step=step)


def log_learning_rate(
    optimizer: Any,
    run: Any,
    step: int,
) -> None:
    """Log current learning rate from optimizer."""
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group.get("lr", 0)
        if len(optimizer.param_groups) == 1:
            run.log({"learning_rate": lr}, step=step)
        else:
            run.log({f"learning_rate/group_{i}": lr}, step=step)


def log_memory_stats(
    run: Any,
    step: int,
) -> None:
    """Log GPU memory statistics."""
    _ensure_torch()

    if not torch.cuda.is_available():
        return

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3

        prefix = f"gpu{i}/" if torch.cuda.device_count() > 1 else ""
        run.log({
            f"{prefix}memory/allocated_gb": allocated,
            f"{prefix}memory/reserved_gb": reserved,
            f"{prefix}memory/max_allocated_gb": max_allocated,
        }, step=step)
