"""
Hardware telemetry collection for GPU, CPU, and memory monitoring.

Automatically logs hardware metrics at 1Hz without user intervention.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import psutil

from atlas_sdk.types import HardwareTelemetry


@dataclass
class GPUMetrics:
    """Metrics for a single GPU."""
    index: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    utilization_percent: float
    temperature_celsius: float
    power_watts: float


class TelemetryCollector:
    """
    Collects hardware telemetry at regular intervals.

    Features:
    - CPU/Memory monitoring via psutil
    - GPU monitoring via pynvml (if available)
    - Non-blocking background collection
    - 1Hz default sampling rate
    """

    def __init__(
        self,
        callback: Callable[[HardwareTelemetry], None],
        interval: float = 1.0,  # 1Hz
    ):
        self.callback = callback
        self.interval = interval

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._gpu_available = False
        self._nvml_initialized = False

        # Try to initialize NVML for GPU monitoring
        self._init_gpu_monitoring()

    def _init_gpu_monitoring(self) -> None:
        """Initialize GPU monitoring if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._gpu_available = pynvml.nvmlDeviceGetCount() > 0
        except (ImportError, Exception):
            self._gpu_available = False

    def start(self) -> None:
        """Start the telemetry collection thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._collect_loop,
            daemon=True,
            name="atlas-telemetry",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the telemetry collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _collect_loop(self) -> None:
        """Main collection loop running in background thread."""
        while self._running:
            try:
                telemetry = self._collect_telemetry()
                self.callback(telemetry)
            except Exception:
                pass  # Don't let telemetry errors affect the main process

            time.sleep(self.interval)

    def _collect_telemetry(self) -> HardwareTelemetry:
        """Collect current hardware metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        telemetry = HardwareTelemetry(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024 ** 3),
        )

        # GPU metrics
        if self._gpu_available:
            gpu_metrics = self._collect_gpu_metrics()
            telemetry.gpu_count = len(gpu_metrics)
            telemetry.gpu_metrics = [
                {
                    "index": g.index,
                    "name": g.name,
                    "memory_used_mb": g.memory_used_mb,
                    "memory_total_mb": g.memory_total_mb,
                    "memory_percent": g.memory_percent,
                    "utilization_percent": g.utilization_percent,
                    "temperature_celsius": g.temperature_celsius,
                    "power_watts": g.power_watts,
                }
                for g in gpu_metrics
            ]

        return telemetry

    def _collect_gpu_metrics(self) -> List[GPUMetrics]:
        """Collect metrics for all GPUs."""
        metrics = []

        try:
            import pynvml

            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = mem_info.used / (1024 ** 2)
                memory_total_mb = mem_info.total / (1024 ** 2)
                memory_percent = (mem_info.used / mem_info.total) * 100

                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_percent = util.gpu

                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temp = 0.0

                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except Exception:
                    power = 0.0

                metrics.append(GPUMetrics(
                    index=i,
                    name=name,
                    memory_used_mb=memory_used_mb,
                    memory_total_mb=memory_total_mb,
                    memory_percent=memory_percent,
                    utilization_percent=utilization_percent,
                    temperature_celsius=temp,
                    power_watts=power,
                ))

        except Exception:
            pass

        return metrics

    def collect_once(self) -> HardwareTelemetry:
        """Collect telemetry once (for manual collection)."""
        return self._collect_telemetry()


class TelemetryAggregator:
    """
    Aggregates telemetry data for efficient storage.

    Instead of storing every 1Hz sample, aggregates into summary stats.
    """

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._buffer: List[HardwareTelemetry] = []

    def add(self, telemetry: HardwareTelemetry) -> Optional[Dict[str, Any]]:
        """
        Add a telemetry sample.

        Returns aggregated stats when window is full.
        """
        self._buffer.append(telemetry)

        if len(self._buffer) >= self.window_size:
            stats = self._aggregate()
            self._buffer.clear()
            return stats

        return None

    def _aggregate(self) -> Dict[str, Any]:
        """Aggregate buffered telemetry into summary stats."""
        if not self._buffer:
            return {}

        cpu_values = [t.cpu_percent for t in self._buffer]
        mem_values = [t.memory_percent for t in self._buffer]

        result = {
            "cpu_percent_avg": sum(cpu_values) / len(cpu_values),
            "cpu_percent_max": max(cpu_values),
            "memory_percent_avg": sum(mem_values) / len(mem_values),
            "memory_percent_max": max(mem_values),
            "memory_used_gb_avg": sum(t.memory_used_gb for t in self._buffer) / len(self._buffer),
            "sample_count": len(self._buffer),
        }

        # GPU aggregation
        if self._buffer[0].gpu_count > 0:
            for gpu_idx in range(self._buffer[0].gpu_count):
                gpu_util = [
                    t.gpu_metrics[gpu_idx]["utilization_percent"]
                    for t in self._buffer
                    if len(t.gpu_metrics) > gpu_idx
                ]
                gpu_mem = [
                    t.gpu_metrics[gpu_idx]["memory_percent"]
                    for t in self._buffer
                    if len(t.gpu_metrics) > gpu_idx
                ]
                gpu_temp = [
                    t.gpu_metrics[gpu_idx]["temperature_celsius"]
                    for t in self._buffer
                    if len(t.gpu_metrics) > gpu_idx
                ]

                if gpu_util:
                    result[f"gpu{gpu_idx}_util_avg"] = sum(gpu_util) / len(gpu_util)
                    result[f"gpu{gpu_idx}_util_max"] = max(gpu_util)
                if gpu_mem:
                    result[f"gpu{gpu_idx}_mem_avg"] = sum(gpu_mem) / len(gpu_mem)
                    result[f"gpu{gpu_idx}_mem_max"] = max(gpu_mem)
                if gpu_temp:
                    result[f"gpu{gpu_idx}_temp_avg"] = sum(gpu_temp) / len(gpu_temp)
                    result[f"gpu{gpu_idx}_temp_max"] = max(gpu_temp)

        return result

    def flush(self) -> Optional[Dict[str, Any]]:
        """Force aggregation of current buffer."""
        if self._buffer:
            stats = self._aggregate()
            self._buffer.clear()
            return stats
        return None
