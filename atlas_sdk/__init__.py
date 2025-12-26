"""
Atlas SDK - The Sovereign Experiment Engine

A local-first, high-performance instrumentation and analysis platform
for deep learning research.

Example:
    import atlas

    with atlas.run("my_experiment") as run:
        for step in range(1000):
            loss = train_step()
            run.log({"loss": loss, "step": step})
"""

from atlas_sdk.run import Run, init, log, finish, get_current_run
from atlas_sdk.types import LogData, RunConfig, Metric, Artifact

__version__ = "0.1.0"
__all__ = [
    "Run",
    "init",
    "log",
    "finish",
    "get_current_run",
    "LogData",
    "RunConfig",
    "Metric",
    "Artifact",
]
