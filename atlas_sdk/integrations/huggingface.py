"""
Hugging Face Transformers Integration

Integration with the Hugging Face Trainer API for
seamless experiment tracking of LLM training.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

# Type stubs
TrainerCallback = Any
TrainingArguments = Any
TrainerState = Any
TrainerControl = Any


class AtlasTrainerCallback:
    """
    Hugging Face Trainer callback for Atlas integration.

    Automatically logs:
    - Training/evaluation metrics
    - Model configuration
    - Training arguments
    - Learning rate schedules
    - Token throughput
    - GPU memory usage

    Usage:
        from atlas_sdk.integrations import AtlasTrainerCallback
        from transformers import Trainer

        callback = AtlasTrainerCallback(
            project="llm_training",
            name="gpt2_finetuning",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            callbacks=[callback],
        )
        trainer.train()
    """

    def __init__(
        self,
        project: str = "default",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        log_model_config: bool = True,
        log_training_args: bool = True,
        log_token_throughput: bool = True,
        log_memory: bool = True,
    ):
        self.project = project
        self.name = name
        self.config = config or {}
        self.tags = tags or []
        self.log_model_config = log_model_config
        self.log_training_args = log_training_args
        self.log_token_throughput = log_token_throughput
        self.log_memory = log_memory

        self._run: Optional[Any] = None
        self._step = 0
        self._last_log_time = 0.0
        self._total_tokens = 0

    @property
    def run(self) -> Any:
        """Get the current Atlas run."""
        return self._run

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Called at the end of Trainer initialization."""
        pass

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Called at the beginning of training."""
        import atlas_sdk as atlas

        # Build config
        full_config = dict(self.config)

        # Add training arguments
        if self.log_training_args:
            full_config.update({
                "training/learning_rate": args.learning_rate,
                "training/batch_size": args.per_device_train_batch_size,
                "training/epochs": args.num_train_epochs,
                "training/warmup_steps": args.warmup_steps,
                "training/weight_decay": args.weight_decay,
                "training/gradient_accumulation": args.gradient_accumulation_steps,
                "training/fp16": args.fp16,
                "training/bf16": args.bf16,
                "training/max_grad_norm": args.max_grad_norm,
            })

        # Add model config
        if self.log_model_config and model is not None:
            if hasattr(model, "config"):
                model_config = model.config.to_dict() if hasattr(model.config, "to_dict") else {}
                full_config.update({
                    "model/name": getattr(model.config, "_name_or_path", "unknown"),
                    "model/hidden_size": model_config.get("hidden_size"),
                    "model/num_layers": model_config.get("num_hidden_layers"),
                    "model/num_heads": model_config.get("num_attention_heads"),
                    "model/vocab_size": model_config.get("vocab_size"),
                })

        # Generate name if not provided
        name = self.name
        if name is None and model is not None:
            name = f"{model.__class__.__name__}_training"

        self._run = atlas.init(
            name=name or "hf_training",
            project=self.project,
            config=full_config,
            tags=self.tags + ["huggingface"],
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Called at the end of training."""
        if self._run is not None:
            # Log final summary
            self._run.log({
                "summary/total_steps": state.global_step,
                "summary/total_tokens": self._total_tokens,
            }, step=state.global_step)

            self._run.finish()
            self._run = None

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Called at the end of each training step."""
        if self._run is None:
            return

        self._step = state.global_step

        # Log memory usage
        if self.log_memory:
            self._log_memory()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when logs are ready to be written."""
        if self._run is None or logs is None:
            return

        # Process and log metrics
        metrics = {}
        for key, value in logs.items():
            # Clean up key names
            clean_key = key.replace("train_", "train/").replace("eval_", "eval/")

            # Handle special metrics
            if "loss" in key:
                clean_key = clean_key.replace("loss", "loss")
            elif "learning_rate" in key:
                clean_key = "learning_rate"

            # Skip runtime metrics
            if key in ("train_runtime", "train_samples_per_second", "train_steps_per_second"):
                continue

            if isinstance(value, (int, float)):
                metrics[clean_key] = value

        if metrics:
            self._run.log(metrics, step=state.global_step)

        # Calculate token throughput
        if self.log_token_throughput and "train_samples_per_second" in (logs or {}):
            samples_per_sec = logs["train_samples_per_second"]
            # Estimate tokens (assuming average sequence length from args)
            seq_length = getattr(args, "max_seq_length", 512)
            tokens_per_sec = samples_per_sec * seq_length
            self._run.log({
                "throughput/tokens_per_second": tokens_per_sec,
                "throughput/samples_per_second": samples_per_sec,
            }, step=state.global_step)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Called after evaluation."""
        if self._run is None or metrics is None:
            return

        eval_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_key = f"eval/{key.replace('eval_', '')}"
                eval_metrics[clean_key] = value

        if eval_metrics:
            self._run.log(eval_metrics, step=state.global_step)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Called when saving a checkpoint."""
        if self._run is None:
            return

        self._run.log({
            "checkpoint/step": state.global_step,
            "checkpoint/best_metric": state.best_metric,
        }, step=state.global_step)

    def _log_memory(self) -> None:
        """Log GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3

                    prefix = f"gpu{i}/" if torch.cuda.device_count() > 1 else ""
                    self._run.log({
                        f"{prefix}memory/allocated_gb": allocated,
                        f"{prefix}memory/reserved_gb": reserved,
                    }, step=self._step)
        except ImportError:
            pass


def setup_atlas_logging(
    project: str = "default",
    name: Optional[str] = None,
    **kwargs: Any,
) -> AtlasTrainerCallback:
    """
    Convenience function to create Atlas callback for HF Trainer.

    Usage:
        from atlas_sdk.integrations.huggingface import setup_atlas_logging

        trainer = Trainer(
            ...
            callbacks=[setup_atlas_logging(project="my_project")],
        )
    """
    return AtlasTrainerCallback(project=project, name=name, **kwargs)
