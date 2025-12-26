"""
PyTorch Lightning Integration

Seamless integration with PyTorch Lightning's callback system.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

# Type stubs for Lightning
Trainer = Any
LightningModule = Any
Callback = Any


class AtlasLightningCallback:
    """
    PyTorch Lightning callback for Atlas integration.

    Automatically logs:
    - Training/validation/test metrics
    - Learning rate schedules
    - Gradient norms
    - Model checkpoints
    - Hardware telemetry

    Usage:
        from atlas_sdk.integrations import AtlasLightningCallback

        callback = AtlasLightningCallback(
            project="my_project",
            name="experiment_1",
            log_gradients=True,
        )

        trainer = Trainer(callbacks=[callback])
        trainer.fit(model, dataloader)
    """

    def __init__(
        self,
        project: str = "default",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        log_gradients: bool = True,
        log_parameters: bool = False,
        gradient_freq: int = 100,
        parameter_freq: int = 500,
        log_model_checkpoints: bool = True,
    ):
        self.project = project
        self.name = name
        self.config = config or {}
        self.tags = tags or []
        self.log_gradients = log_gradients
        self.log_parameters = log_parameters
        self.gradient_freq = gradient_freq
        self.parameter_freq = parameter_freq
        self.log_model_checkpoints = log_model_checkpoints

        self._run: Optional[Any] = None
        self._step = 0

    @property
    def run(self) -> Any:
        """Get the current Atlas run."""
        return self._run

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        """Called when fit or test begins."""
        if self._run is not None:
            return

        import atlas_sdk as atlas

        # Build config from hyperparameters
        full_config = dict(self.config)
        if hasattr(pl_module, "hparams"):
            full_config.update(dict(pl_module.hparams))

        # Add trainer config
        full_config.update({
            "trainer/max_epochs": trainer.max_epochs,
            "trainer/accelerator": str(trainer.accelerator),
            "trainer/devices": trainer.num_devices,
            "trainer/precision": str(trainer.precision),
        })

        # Generate run name if not provided
        name = self.name or f"{pl_module.__class__.__name__}_{stage}"

        self._run = atlas.init(
            name=name,
            project=self.project,
            config=full_config,
            tags=self.tags + [stage],
        )

    def teardown(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        """Called when fit or test ends."""
        if self._run is not None:
            self._run.finish()
            self._run = None

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when a training batch ends."""
        if self._run is None:
            return

        self._step = trainer.global_step

        # Log training loss
        if outputs is not None:
            if isinstance(outputs, dict) and "loss" in outputs:
                self._run.log({"train/loss": outputs["loss"].item()}, step=self._step)
            elif hasattr(outputs, "loss"):
                self._run.log({"train/loss": outputs.loss.item()}, step=self._step)

        # Log gradients
        if self.log_gradients and self._step % self.gradient_freq == 0:
            self._log_gradients(pl_module)

        # Log parameters
        if self.log_parameters and self._step % self.parameter_freq == 0:
            self._log_parameters(pl_module)

        # Log learning rate
        self._log_learning_rate(trainer)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Called when a training epoch ends."""
        if self._run is None:
            return

        self._run.log({
            "epoch": trainer.current_epoch,
        }, step=self._step)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when a validation batch ends."""
        pass  # Metrics are logged via on_validation_epoch_end

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Called when validation epoch ends."""
        if self._run is None:
            return

        # Log all validation metrics
        for key, value in trainer.callback_metrics.items():
            if "val" in key.lower():
                if hasattr(value, "item"):
                    value = value.item()
                self._run.log({key: value}, step=self._step)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Called when test epoch ends."""
        if self._run is None:
            return

        for key, value in trainer.callback_metrics.items():
            if "test" in key.lower():
                if hasattr(value, "item"):
                    value = value.item()
                self._run.log({key: value}, step=self._step)

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Called when saving a checkpoint."""
        if not self.log_model_checkpoints:
            return

        if self._run is None:
            return

        # Log checkpoint info (actual saving handled by Atlas artifact system)
        self._run.log({
            "checkpoint/epoch": trainer.current_epoch,
            "checkpoint/global_step": trainer.global_step,
        }, step=self._step)

    def _log_gradients(self, pl_module: LightningModule) -> None:
        """Log gradient statistics."""
        from atlas_sdk.integrations.pytorch import log_gradients
        log_gradients(pl_module, self._run, self._step)

    def _log_parameters(self, pl_module: LightningModule) -> None:
        """Log parameter statistics."""
        from atlas_sdk.integrations.pytorch import log_parameters
        log_parameters(pl_module, self._run, self._step)

    def _log_learning_rate(self, trainer: Trainer) -> None:
        """Log current learning rate."""
        if trainer.lr_scheduler_configs:
            for i, config in enumerate(trainer.lr_scheduler_configs):
                scheduler = config.scheduler
                if hasattr(scheduler, "get_last_lr"):
                    lrs = scheduler.get_last_lr()
                    for j, lr in enumerate(lrs):
                        key = f"learning_rate" if len(lrs) == 1 else f"learning_rate/group_{j}"
                        self._run.log({key: lr}, step=self._step)
                    break

    # State dict methods for checkpointing
    def state_dict(self) -> Dict[str, Any]:
        """Return callback state."""
        return {
            "step": self._step,
            "run_id": self._run.id if self._run else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state."""
        self._step = state_dict.get("step", 0)


def auto_log(
    project: str = "default",
    name: Optional[str] = None,
    **kwargs: Any,
) -> AtlasLightningCallback:
    """
    Create an Atlas callback with automatic logging.

    Convenience function for quick setup.
    """
    return AtlasLightningCallback(
        project=project,
        name=name,
        log_gradients=True,
        log_parameters=True,
        **kwargs,
    )
