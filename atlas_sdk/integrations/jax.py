"""
JAX/Flax Integration

Integration with JAX ecosystem for high-performance
ML experiment tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Lazy imports
jax = None
jnp = None


def _ensure_jax():
    """Lazily import JAX."""
    global jax, jnp
    if jax is None:
        import jax as _jax
        import jax.numpy as _jnp
        jax = _jax
        jnp = _jnp


@dataclass
class JAXLogConfig:
    """Configuration for JAX logging."""
    log_gradients: bool = True
    log_parameters: bool = True
    gradient_freq: int = 1
    parameter_freq: int = 100
    log_jit_compilation: bool = True
    log_device_stats: bool = True


class AtlasJAXLogger:
    """
    JAX/Flax logger for Atlas integration.

    Supports both pure JAX and Flax training loops.

    Usage with pure JAX:
        logger = AtlasJAXLogger(project="jax_training")

        @jax.jit
        def train_step(state, batch):
            def loss_fn(params):
                return compute_loss(params, batch)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss, grads

        for step, batch in enumerate(dataloader):
            state, loss, grads = train_step(state, batch)
            logger.log_step(step, {"loss": loss}, params=state.params, grads=grads)

    Usage with Flax:
        from flax.training import train_state

        logger = AtlasJAXLogger(project="flax_training")

        for step, batch in enumerate(dataloader):
            state, metrics = train_step(state, batch)
            logger.log_step(step, metrics, params=state.params)
    """

    def __init__(
        self,
        project: str = "default",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        log_config: Optional[JAXLogConfig] = None,
    ):
        _ensure_jax()

        self.project = project
        self.name = name
        self.config = config or {}
        self.tags = tags or []
        self.log_config = log_config or JAXLogConfig()

        self._run: Optional[Any] = None
        self._step = 0
        self._jit_times: Dict[str, float] = {}

    def start(self) -> "AtlasJAXLogger":
        """Start the Atlas run."""
        import atlas_sdk as atlas

        # Add JAX-specific config
        full_config = dict(self.config)
        full_config.update({
            "jax/version": jax.__version__,
            "jax/devices": len(jax.devices()),
            "jax/device_type": str(jax.devices()[0].device_kind),
            "jax/platform": jax.default_backend(),
        })

        self._run = atlas.init(
            name=self.name or "jax_training",
            project=self.project,
            config=full_config,
            tags=self.tags + ["jax"],
        )

        return self

    def finish(self) -> None:
        """Finish the Atlas run."""
        if self._run is not None:
            self._run.finish()
            self._run = None

    def log_step(
        self,
        step: int,
        metrics: Dict[str, Any],
        params: Optional[Any] = None,
        grads: Optional[Any] = None,
    ) -> None:
        """
        Log metrics for a training step.

        Args:
            step: Current step number
            metrics: Dictionary of metric values
            params: Optional pytree of model parameters
            grads: Optional pytree of gradients
        """
        if self._run is None:
            return

        self._step = step

        # Convert JAX arrays to Python floats
        clean_metrics = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                clean_metrics[key] = float(value.item())
            elif isinstance(value, (int, float)):
                clean_metrics[key] = value

        self._run.log(clean_metrics, step=step)

        # Log gradients
        if grads is not None and self.log_config.log_gradients:
            if step % self.log_config.gradient_freq == 0:
                self._log_pytree_stats(grads, "gradients", step)

        # Log parameters
        if params is not None and self.log_config.log_parameters:
            if step % self.log_config.parameter_freq == 0:
                self._log_pytree_stats(params, "parameters", step)

        # Log device stats
        if self.log_config.log_device_stats:
            self._log_device_stats(step)

    def _log_pytree_stats(
        self,
        pytree: Any,
        prefix: str,
        step: int,
    ) -> None:
        """Log statistics for a JAX pytree."""
        flat, _ = jax.tree_util.tree_flatten(pytree)

        norms = []
        total_params = 0

        for i, leaf in enumerate(flat):
            if hasattr(leaf, "shape"):
                arr = np.asarray(leaf)
                norm = float(np.linalg.norm(arr))
                norms.append(norm)
                total_params += arr.size

        if norms:
            total_norm = np.sqrt(sum(n**2 for n in norms))
            self._run.log({
                f"{prefix}/total_norm": total_norm,
                f"{prefix}/max_norm": max(norms),
                f"{prefix}/mean_norm": np.mean(norms),
                f"{prefix}/num_tensors": len(norms),
                f"{prefix}/total_elements": total_params,
            }, step=step)

    def _log_device_stats(self, step: int) -> None:
        """Log JAX device statistics."""
        try:
            # Get memory stats if available
            for i, device in enumerate(jax.devices()):
                prefix = f"device{i}/" if len(jax.devices()) > 1 else ""
                # Note: Memory stats availability depends on backend
                if hasattr(device, "memory_stats"):
                    stats = device.memory_stats()
                    if stats:
                        self._run.log({
                            f"{prefix}memory/bytes_in_use": stats.get("bytes_in_use", 0),
                        }, step=step)
        except Exception:
            pass

    def log_compilation_time(
        self,
        fn_name: str,
        time_ms: float,
    ) -> None:
        """Log JIT compilation time."""
        if not self.log_config.log_jit_compilation:
            return

        if self._run is None:
            return

        self._jit_times[fn_name] = time_ms
        self._run.log({
            f"jit/{fn_name}_compile_ms": time_ms,
        }, step=self._step)

    def __enter__(self) -> "AtlasJAXLogger":
        """Context manager entry."""
        return self.start()

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.finish()


def jit_with_logging(
    logger: AtlasJAXLogger,
    fn: Callable,
    **jit_kwargs: Any,
) -> Callable:
    """
    Wrapper for jax.jit that logs compilation time.

    Usage:
        @jit_with_logging(logger)
        def train_step(state, batch):
            ...
    """
    _ensure_jax()
    import time

    jitted_fn = jax.jit(fn, **jit_kwargs)
    fn_name = getattr(fn, "__name__", "unknown")
    first_call = [True]

    def wrapper(*args, **kwargs):
        if first_call[0]:
            start = time.perf_counter()
            result = jitted_fn(*args, **kwargs)
            # Block until computation is done
            jax.block_until_ready(result)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.log_compilation_time(fn_name, elapsed_ms)
            first_call[0] = False
            return result
        return jitted_fn(*args, **kwargs)

    return wrapper


def log_pytree_structure(
    pytree: Any,
    name: str = "model",
) -> Dict[str, Any]:
    """
    Analyze and return pytree structure info.

    Useful for logging model architecture.
    """
    _ensure_jax()

    flat, tree_def = jax.tree_util.tree_flatten(pytree)

    total_params = 0
    shapes = []

    for leaf in flat:
        if hasattr(leaf, "shape"):
            total_params += np.prod(leaf.shape)
            shapes.append(list(leaf.shape))

    return {
        f"{name}/total_parameters": int(total_params),
        f"{name}/num_tensors": len(flat),
        f"{name}/tree_structure": str(tree_def),
    }
