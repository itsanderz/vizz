"""
Deep Digging Agent

Autonomous analysis agent that monitors training runs,
detects issues, and generates actionable insights.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID

import numpy as np

from atlas_sdk.analysis.anomaly import Anomaly, AnomalyDetector
from atlas_sdk.analysis.insights import Insight, InsightGenerator, TrainingMetrics


class AgentState(str, Enum):
    """Agent execution state."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    WAITING = "waiting"
    STOPPED = "stopped"


@dataclass
class AnalysisResult:
    """Result of a single analysis cycle."""
    run_id: str
    timestamp: datetime
    anomalies: List[Anomaly]
    insights: List[Insight]
    metrics_analyzed: int
    duration_ms: float


class DeepDiggingAgent:
    """
    Autonomous Deep Digging Agent.

    Runs in the background, continuously analyzing training runs
    and generating insights. Designed for enterprise-grade
    reliability and performance.

    Features:
    - Autonomous monitoring of active runs
    - Real-time anomaly detection
    - AI-powered insight generation
    - User feedback learning
    - Configurable analysis frequency
    """

    def __init__(
        self,
        storage_engine: Any,  # StorageEngine
        analysis_interval: float = 30.0,  # seconds
        sensitivity: float = 0.8,
        max_insights_per_run: int = 50,
        enable_ai: bool = True,
    ):
        self.storage = storage_engine
        self.analysis_interval = analysis_interval
        self.max_insights_per_run = max_insights_per_run

        # Analysis components
        self.anomaly_detector = AnomalyDetector(sensitivity=sensitivity)
        self.insight_generator = InsightGenerator(enable_ai=enable_ai)

        # State
        self._state = AgentState.IDLE
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Tracking
        self._analyzed_runs: Set[str] = set()
        self._last_analysis: Dict[str, datetime] = {}
        self._insight_counts: Dict[str, int] = {}

        # Callbacks
        self._on_anomaly: List[Callable[[Anomaly], None]] = []
        self._on_insight: List[Callable[[Insight], None]] = []

        # Results queue for async consumers
        self._results_queue: Queue[AnalysisResult] = Queue(maxsize=100)

    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state

    def start(self) -> None:
        """Start the agent in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._state = AgentState.WAITING
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="atlas-deep-digging-agent",
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the agent gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._state = AgentState.STOPPED

    def on_anomaly(self, callback: Callable[[Anomaly], None]) -> None:
        """Register callback for anomaly detection."""
        self._on_anomaly.append(callback)

    def on_insight(self, callback: Callable[[Insight], None]) -> None:
        """Register callback for insight generation."""
        self._on_insight.append(callback)

    def get_latest_result(self, timeout: float = 0.0) -> Optional[AnalysisResult]:
        """Get the latest analysis result (non-blocking by default)."""
        try:
            return self._results_queue.get(timeout=timeout)
        except Empty:
            return None

    def analyze_run_now(self, run_id: str) -> AnalysisResult:
        """Immediately analyze a specific run (synchronous)."""
        return self._analyze_run(run_id)

    def _run_loop(self) -> None:
        """Main agent loop."""
        while not self._stop_event.is_set():
            try:
                self._state = AgentState.ANALYZING

                # Get active runs
                runs = self.storage.list_runs(status="running")

                for run in runs:
                    if self._stop_event.is_set():
                        break

                    # Check if enough time has passed since last analysis
                    last = self._last_analysis.get(run["id"])
                    if last and (datetime.utcnow() - last).total_seconds() < self.analysis_interval:
                        continue

                    # Check insight limit
                    if self._insight_counts.get(run["id"], 0) >= self.max_insights_per_run:
                        continue

                    # Analyze
                    result = self._analyze_run(run["id"])

                    # Update tracking
                    self._last_analysis[run["id"]] = datetime.utcnow()
                    self._analyzed_runs.add(run["id"])

                    # Queue result
                    try:
                        self._results_queue.put_nowait(result)
                    except Exception:
                        pass  # Queue full, skip

                self._state = AgentState.WAITING

            except Exception as e:
                print(f"[DeepDiggingAgent] Error in analysis loop: {e}")

            # Wait for next cycle
            self._stop_event.wait(timeout=self.analysis_interval)

        self._state = AgentState.STOPPED

    def _analyze_run(self, run_id: str) -> AnalysisResult:
        """Perform analysis on a single run."""
        start_time = time.time()

        # Fetch metrics
        metric_names = self.storage.list_metrics(run_id)
        metrics_data: Dict[str, np.ndarray] = {}
        steps: Optional[np.ndarray] = None

        for name in metric_names[:20]:  # Limit to first 20 metrics
            try:
                data = self.storage.get_metric(run_id, name)
                if data:
                    values = np.array([d["value"] for d in data])
                    if steps is None:
                        steps = np.array([d["step"] for d in data])
                    metrics_data[name] = values
            except Exception:
                continue

        if not metrics_data or steps is None:
            return AnalysisResult(
                run_id=run_id,
                timestamp=datetime.utcnow(),
                anomalies=[],
                insights=[],
                metrics_analyzed=0,
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Detect anomalies
        anomalies = self.anomaly_detector.analyze(metrics_data, steps)

        # Notify callbacks
        for anomaly in anomalies:
            for callback in self._on_anomaly:
                try:
                    callback(anomaly)
                except Exception:
                    pass

        # Generate insights
        insights = self.insight_generator.generate_from_anomalies(anomalies)

        # Save insights to storage
        for insight in insights:
            try:
                self.storage.save_insight(insight)
                self._insight_counts[run_id] = self._insight_counts.get(run_id, 0) + 1
            except Exception:
                pass

            # Notify callbacks
            for callback in self._on_insight:
                try:
                    callback(insight)
                except Exception:
                    pass

        duration_ms = (time.time() - start_time) * 1000

        return AnalysisResult(
            run_id=run_id,
            timestamp=datetime.utcnow(),
            anomalies=anomalies,
            insights=insights,
            metrics_analyzed=len(metrics_data),
            duration_ms=duration_ms,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "state": self._state.value,
            "runs_analyzed": len(self._analyzed_runs),
            "total_insights_generated": sum(self._insight_counts.values()),
            "insights_by_run": dict(self._insight_counts),
            "pending_results": self._results_queue.qsize(),
        }


class DeepDiggingAgentPool:
    """
    Pool of Deep Digging Agents for enterprise scale.

    Distributes analysis across multiple agents for
    high-throughput environments.
    """

    def __init__(
        self,
        storage_engine: Any,
        num_agents: int = 2,
        **agent_kwargs: Any,
    ):
        self.storage = storage_engine
        self.agents: List[DeepDiggingAgent] = []

        for i in range(num_agents):
            agent = DeepDiggingAgent(
                storage_engine=storage_engine,
                **agent_kwargs,
            )
            self.agents.append(agent)

    def start(self) -> None:
        """Start all agents."""
        for agent in self.agents:
            agent.start()

    def stop(self) -> None:
        """Stop all agents."""
        for agent in self.agents:
            agent.stop()

    def get_all_results(self) -> List[AnalysisResult]:
        """Collect results from all agents."""
        results = []
        for agent in self.agents:
            while True:
                result = agent.get_latest_result()
                if result is None:
                    break
                results.append(result)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "num_agents": len(self.agents),
            "agents": [agent.get_statistics() for agent in self.agents],
        }
