"""
The Harvester - Background process for crash-safe disk I/O.

Runs in a separate process from the main training loop. If it crashes,
the training continues uninterrupted.
"""

from __future__ import annotations

import atexit
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional

from atlas_sdk.buffer import FallbackQueue, RingBufferReader
from atlas_sdk.storage.engine import StorageEngine
from atlas_sdk.types import Metric, MetricType, RunStatus


class Harvester:
    """
    Background process that drains the ring buffer and persists data.

    Features:
    - Runs in a separate process for crash isolation
    - Batches writes for efficiency
    - Handles graceful shutdown
    - Recovers from temporary failures
    """

    POLL_INTERVAL = 0.01  # 10ms
    BATCH_SIZE = 100
    MAX_RETRIES = 3

    def __init__(
        self,
        run_id: str,
        atlas_dir: Path,
        buffer_name: str,
        use_shared_memory: bool = True,
    ):
        self.run_id = run_id
        self.atlas_dir = atlas_dir
        self.buffer_name = buffer_name
        self.use_shared_memory = use_shared_memory

        self._process: Optional[Process] = None
        self._stop_flag = mp.Event()
        self._started = mp.Event()

        # Fallback queue if shared memory isn't available
        self._fallback_queue: Optional[Queue] = None

    def start(self) -> None:
        """Start the harvester process."""
        if self._process is not None and self._process.is_alive():
            return

        self._stop_flag.clear()
        self._started.clear()

        if not self.use_shared_memory:
            self._fallback_queue = mp.Queue(maxsize=100000)

        self._process = Process(
            target=self._run,
            args=(
                self.run_id,
                str(self.atlas_dir),
                self.buffer_name,
                self._stop_flag,
                self._started,
                self.use_shared_memory,
                self._fallback_queue,
            ),
            daemon=True,
            name=f"atlas-harvester-{self.run_id[:8]}",
        )
        self._process.start()

        # Wait for harvester to initialize
        self._started.wait(timeout=5.0)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the harvester gracefully."""
        if self._process is None or not self._process.is_alive():
            return

        self._stop_flag.set()
        self._process.join(timeout=timeout)

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)

    def is_alive(self) -> bool:
        """Check if harvester is running."""
        return self._process is not None and self._process.is_alive()

    def put(self, data: Dict[str, Any]) -> bool:
        """
        Put data into the harvester queue (fallback mode only).

        For shared memory mode, write directly to the ring buffer.
        """
        if self._fallback_queue is None:
            return False

        try:
            self._fallback_queue.put_nowait(data)
            return True
        except Exception:
            return False

    @staticmethod
    def _run(
        run_id: str,
        atlas_dir: str,
        buffer_name: str,
        stop_flag: mp.Event,
        started_flag: mp.Event,
        use_shared_memory: bool,
        fallback_queue: Optional[Queue],
    ) -> None:
        """Main harvester loop (runs in subprocess)."""
        # Setup signal handlers
        def signal_handler(signum, frame):
            stop_flag.set()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Initialize storage
        try:
            storage = StorageEngine(
                base_dir=Path(atlas_dir).parent,
                run_id=run_id,
            )
        except Exception as e:
            print(f"[Harvester] Failed to initialize storage: {e}", file=sys.stderr)
            started_flag.set()
            return

        # Initialize reader
        reader = None
        if use_shared_memory:
            try:
                reader = RingBufferReader(buffer_name)
            except FileNotFoundError:
                # Shared memory not created yet, wait a bit
                time.sleep(0.5)
                try:
                    reader = RingBufferReader(buffer_name)
                except Exception as e:
                    print(f"[Harvester] Could not open shared memory: {e}", file=sys.stderr)

        started_flag.set()

        # Main loop
        consecutive_errors = 0
        while not stop_flag.is_set():
            try:
                items = []

                # Read from shared memory or fallback queue
                if reader:
                    items = reader.read_batch(max_items=Harvester.BATCH_SIZE)
                elif fallback_queue:
                    while len(items) < Harvester.BATCH_SIZE:
                        try:
                            item = fallback_queue.get_nowait()
                            items.append(item)
                        except Empty:
                            break

                if not items:
                    time.sleep(Harvester.POLL_INTERVAL)
                    continue

                # Process items
                for item in items:
                    try:
                        Harvester._process_item(storage, item)
                    except Exception as e:
                        print(f"[Harvester] Error processing item: {e}", file=sys.stderr)

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                print(f"[Harvester] Error in main loop: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

                if consecutive_errors >= Harvester.MAX_RETRIES:
                    print("[Harvester] Too many errors, exiting", file=sys.stderr)
                    break

                time.sleep(0.1 * consecutive_errors)

        # Cleanup
        try:
            storage.finish_run(RunStatus.COMPLETED)
            storage.close()
        except Exception as e:
            print(f"[Harvester] Error during cleanup: {e}", file=sys.stderr)

        if reader:
            try:
                reader.close()
            except Exception:
                pass

    @staticmethod
    def _process_item(storage: StorageEngine, item: Dict[str, Any]) -> None:
        """Process a single item from the buffer."""
        msg_type = item.get("msg_type", "metric")

        if msg_type == "metric":
            metric = Metric(
                name=item["name"],
                value=item["value"],
                step=item["step"],
                metric_type=MetricType(item.get("metric_type", "scalar")),
                metadata=item.get("metadata", {}),
            )
            storage.timeseries.log(metric)

        elif msg_type == "metrics_batch":
            metrics = [
                Metric(
                    name=m["name"],
                    value=m["value"],
                    step=m["step"],
                    metric_type=MetricType(m.get("metric_type", "scalar")),
                )
                for m in item["metrics"]
            ]
            storage.timeseries.log_batch(metrics)

        elif msg_type == "tensor":
            import numpy as np
            tensor = np.array(item["data"])
            storage.log_tensor(
                name=item["name"],
                tensor=tensor,
                step=item["step"],
                metadata=item.get("metadata"),
            )

        elif msg_type == "telemetry":
            # Hardware telemetry
            for key, value in item.get("data", {}).items():
                storage.log_metric(
                    name=f"system/{key}",
                    value=value,
                    step=item.get("step", 0),
                    metric_type=MetricType.SCALAR,
                )

        elif msg_type == "control":
            cmd = item.get("command")
            if cmd == "flush":
                storage.timeseries._flush()
            elif cmd == "finish":
                storage.finish_run(
                    status=RunStatus(item.get("status", "completed")),
                    duration_seconds=item.get("duration"),
                )


class HarvesterManager:
    """
    Manages the harvester lifecycle for a run.

    Handles:
    - Starting/stopping harvesters
    - Fallback to queue-based mode if shared memory fails
    - Cleanup on process exit
    """

    _instances: Dict[str, "HarvesterManager"] = {}

    def __init__(self, run_id: str, atlas_dir: Path):
        self.run_id = run_id
        self.atlas_dir = atlas_dir
        self.buffer_name = f"atlas_{run_id[:16]}"

        self._harvester: Optional[Harvester] = None
        self._use_shared_memory = True

        # Try to use shared memory, fall back to queue
        try:
            from atlas_sdk.buffer import RingBufferWriter
            self._writer = RingBufferWriter(self.buffer_name, create=True)
        except Exception:
            self._use_shared_memory = False
            self._writer = None

        # Register cleanup
        atexit.register(self._cleanup)
        HarvesterManager._instances[run_id] = self

    def start(self) -> None:
        """Start the harvester."""
        self._harvester = Harvester(
            run_id=self.run_id,
            atlas_dir=self.atlas_dir,
            buffer_name=self.buffer_name,
            use_shared_memory=self._use_shared_memory,
        )
        self._harvester.start()

    def write(self, data: Dict[str, Any]) -> bool:
        """
        Write data to the buffer.

        Returns immediately (< 5µs target).
        """
        data["timestamp"] = time.time()
        data["run_id"] = self.run_id

        if self._writer:
            return self._writer.write(data)
        elif self._harvester:
            return self._harvester.put(data)
        return False

    def stop(self) -> None:
        """Stop the harvester and cleanup."""
        if self._harvester:
            # Send finish command
            self.write({"msg_type": "control", "command": "finish"})
            time.sleep(0.1)  # Let it process

            self._harvester.stop()

        if self._writer:
            self._writer.close()
            try:
                self._writer.unlink()
            except Exception:
                pass

    def _cleanup(self) -> None:
        """Cleanup on exit."""
        self.stop()
        HarvesterManager._instances.pop(self.run_id, None)

    @classmethod
    def get(cls, run_id: str) -> Optional["HarvesterManager"]:
        """Get harvester manager by run ID."""
        return cls._instances.get(run_id)
