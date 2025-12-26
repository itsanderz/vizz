"""
Anomaly Detection Engine

Statistical and ML-based anomaly detection for training metrics.
Detects loss spikes, gradient issues, and other training anomalies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    LOSS_SPIKE = "loss_spike"
    LOSS_PLATEAU = "loss_plateau"
    LOSS_DIVERGENCE = "loss_divergence"
    LOSS_NAN = "loss_nan"
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    GRADIENT_NAN = "gradient_nan"
    LEARNING_RATE_ISSUE = "learning_rate_issue"
    METRIC_DISCONTINUITY = "metric_discontinuity"
    UNUSUAL_PATTERN = "unusual_pattern"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class Severity(str, Enum):
    """Anomaly severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Detected anomaly with context."""
    id: UUID
    anomaly_type: AnomalyType
    severity: Severity
    metric_name: str
    step: int
    value: float
    expected_range: Tuple[float, float]
    confidence: float
    description: str
    suggested_actions: List[str]
    related_metrics: List[str]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "step": self.step,
            "value": self.value,
            "expected_range": self.expected_range,
            "confidence": self.confidence,
            "description": self.description,
            "suggested_actions": self.suggested_actions,
            "related_metrics": self.related_metrics,
            "timestamp": self.timestamp.isoformat(),
        }


class StatisticalDetector:
    """Statistical anomaly detection using z-scores and IQR."""

    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        min_samples: int = 30,
    ):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.min_samples = min_samples

    def detect(
        self,
        values: np.ndarray,
        metric_name: str,
        steps: np.ndarray,
    ) -> List[Anomaly]:
        """Detect anomalies using statistical methods."""
        if len(values) < self.min_samples:
            return []

        anomalies = []

        # Z-score detection
        mean = np.mean(values)
        std = np.std(values)

        if std > 0:
            z_scores = np.abs((values - mean) / std)
            z_anomaly_mask = z_scores > self.z_threshold

            for idx in np.where(z_anomaly_mask)[0]:
                anomalies.append(self._create_anomaly(
                    metric_name=metric_name,
                    step=int(steps[idx]),
                    value=float(values[idx]),
                    expected_range=(mean - self.z_threshold * std, mean + self.z_threshold * std),
                    confidence=min(1.0, z_scores[idx] / 5.0),
                    anomaly_type=self._classify_anomaly(values, idx, mean),
                ))

        # IQR-based detection
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr

        iqr_anomaly_mask = (values < lower_bound) | (values > upper_bound)

        for idx in np.where(iqr_anomaly_mask)[0]:
            if not z_anomaly_mask[idx]:  # Avoid duplicates
                anomalies.append(self._create_anomaly(
                    metric_name=metric_name,
                    step=int(steps[idx]),
                    value=float(values[idx]),
                    expected_range=(lower_bound, upper_bound),
                    confidence=0.7,
                    anomaly_type=self._classify_anomaly(values, idx, mean),
                ))

        return anomalies

    def _classify_anomaly(
        self,
        values: np.ndarray,
        idx: int,
        mean: float,
    ) -> AnomalyType:
        """Classify the type of anomaly."""
        value = values[idx]

        if np.isnan(value) or np.isinf(value):
            return AnomalyType.LOSS_NAN

        if value > mean * 10:
            return AnomalyType.LOSS_SPIKE

        if idx > 10:
            recent_trend = np.mean(values[max(0, idx-10):idx])
            if value > recent_trend * 5:
                return AnomalyType.LOSS_SPIKE

        return AnomalyType.UNUSUAL_PATTERN

    def _create_anomaly(
        self,
        metric_name: str,
        step: int,
        value: float,
        expected_range: Tuple[float, float],
        confidence: float,
        anomaly_type: AnomalyType,
    ) -> Anomaly:
        """Create an anomaly instance."""
        severity = Severity.WARNING
        if anomaly_type in (AnomalyType.LOSS_NAN, AnomalyType.GRADIENT_EXPLOSION):
            severity = Severity.CRITICAL
        elif confidence > 0.9:
            severity = Severity.CRITICAL

        return Anomaly(
            id=uuid4(),
            anomaly_type=anomaly_type,
            severity=severity,
            metric_name=metric_name,
            step=step,
            value=value,
            expected_range=expected_range,
            confidence=confidence,
            description=self._generate_description(anomaly_type, metric_name, value, step),
            suggested_actions=self._get_suggested_actions(anomaly_type),
            related_metrics=self._get_related_metrics(metric_name),
            timestamp=datetime.utcnow(),
        )

    def _generate_description(
        self,
        anomaly_type: AnomalyType,
        metric_name: str,
        value: float,
        step: int,
    ) -> str:
        """Generate human-readable description."""
        descriptions = {
            AnomalyType.LOSS_SPIKE: f"Sudden spike in {metric_name} to {value:.4f} at step {step}",
            AnomalyType.LOSS_PLATEAU: f"{metric_name} has plateaued around {value:.4f}",
            AnomalyType.LOSS_DIVERGENCE: f"{metric_name} is diverging ({value:.4f})",
            AnomalyType.LOSS_NAN: f"NaN/Inf detected in {metric_name} at step {step}",
            AnomalyType.GRADIENT_EXPLOSION: f"Gradient explosion detected at step {step}",
            AnomalyType.GRADIENT_VANISHING: f"Vanishing gradients detected at step {step}",
            AnomalyType.UNUSUAL_PATTERN: f"Unusual value in {metric_name}: {value:.4f}",
        }
        return descriptions.get(anomaly_type, f"Anomaly in {metric_name}")

    def _get_suggested_actions(self, anomaly_type: AnomalyType) -> List[str]:
        """Get suggested remediation actions."""
        actions = {
            AnomalyType.LOSS_SPIKE: [
                "Check for corrupted data batches",
                "Verify data augmentation parameters",
                "Consider gradient clipping",
                "Review recent model changes",
            ],
            AnomalyType.LOSS_PLATEAU: [
                "Try increasing learning rate",
                "Add learning rate warmup or decay",
                "Check for mode collapse (if GAN)",
                "Increase model capacity",
            ],
            AnomalyType.LOSS_DIVERGENCE: [
                "Reduce learning rate immediately",
                "Enable gradient clipping",
                "Check for numerical instability",
                "Verify loss function implementation",
            ],
            AnomalyType.LOSS_NAN: [
                "Check for division by zero",
                "Add epsilon to log operations",
                "Use mixed precision carefully",
                "Verify input data normalization",
            ],
            AnomalyType.GRADIENT_EXPLOSION: [
                "Enable gradient clipping (max_norm=1.0)",
                "Reduce learning rate",
                "Use gradient scaling with AMP",
                "Check model initialization",
            ],
            AnomalyType.GRADIENT_VANISHING: [
                "Use skip connections (ResNet style)",
                "Try different activation functions (GELU, SiLU)",
                "Use gradient checkpointing",
                "Check layer normalization",
            ],
        }
        return actions.get(anomaly_type, ["Investigate metric history"])

    def _get_related_metrics(self, metric_name: str) -> List[str]:
        """Get related metrics to investigate."""
        if "loss" in metric_name.lower():
            return ["learning_rate", "gradient_norm", "accuracy"]
        if "gradient" in metric_name.lower():
            return ["loss", "learning_rate", "weight_norm"]
        return []


class TrendDetector:
    """Detects trends and patterns in time series data."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

    def detect_plateau(
        self,
        values: np.ndarray,
        threshold: float = 0.001,
        min_duration: int = 50,
    ) -> Optional[Tuple[int, int]]:
        """Detect if values have plateaued."""
        if len(values) < min_duration:
            return None

        for start in range(len(values) - min_duration):
            window = values[start:start + min_duration]
            if np.std(window) < threshold * np.mean(np.abs(window)):
                return (start, start + min_duration)

        return None

    def detect_divergence(
        self,
        values: np.ndarray,
        threshold: float = 10.0,
    ) -> Optional[int]:
        """Detect if values are diverging (increasing without bound)."""
        if len(values) < 20:
            return None

        # Check if recent values are much larger than early values
        early_mean = np.mean(values[:10])
        late_mean = np.mean(values[-10:])

        if early_mean > 0 and late_mean / early_mean > threshold:
            # Find divergence point
            for i in range(10, len(values)):
                if values[i] > early_mean * threshold:
                    return i

        return None

    def compute_slope(self, values: np.ndarray) -> float:
        """Compute linear regression slope."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)


class AnomalyDetector:
    """
    Main anomaly detection engine combining multiple detection methods.

    Enterprise-grade anomaly detection with:
    - Statistical methods (z-score, IQR)
    - Trend analysis
    - Pattern matching
    - Cross-metric correlation
    """

    def __init__(
        self,
        sensitivity: float = 0.8,
        min_samples: int = 30,
    ):
        self.sensitivity = sensitivity
        self.min_samples = min_samples

        # Adjust thresholds based on sensitivity
        z_threshold = 4.0 - (sensitivity * 2)  # 2.0 to 4.0
        iqr_multiplier = 2.5 - (sensitivity * 1.5)  # 1.0 to 2.5

        self.statistical = StatisticalDetector(
            z_threshold=z_threshold,
            iqr_multiplier=iqr_multiplier,
            min_samples=min_samples,
        )
        self.trend = TrendDetector()

        self._history: Dict[str, List[Anomaly]] = {}

    def analyze(
        self,
        metrics: Dict[str, np.ndarray],
        steps: np.ndarray,
    ) -> List[Anomaly]:
        """
        Analyze metrics for anomalies.

        Args:
            metrics: Dict of metric_name -> values array
            steps: Array of step numbers

        Returns:
            List of detected anomalies
        """
        all_anomalies = []

        for metric_name, values in metrics.items():
            # Skip if not enough data
            if len(values) < self.min_samples:
                continue

            # Statistical detection
            anomalies = self.statistical.detect(values, metric_name, steps)
            all_anomalies.extend(anomalies)

            # NaN/Inf detection
            nan_mask = ~np.isfinite(values)
            if np.any(nan_mask):
                for idx in np.where(nan_mask)[0]:
                    all_anomalies.append(Anomaly(
                        id=uuid4(),
                        anomaly_type=AnomalyType.LOSS_NAN,
                        severity=Severity.CRITICAL,
                        metric_name=metric_name,
                        step=int(steps[idx]),
                        value=float(values[idx]) if not np.isnan(values[idx]) else 0.0,
                        expected_range=(0.0, 0.0),
                        confidence=1.0,
                        description=f"NaN/Inf detected in {metric_name}",
                        suggested_actions=["Check for numerical instability"],
                        related_metrics=[],
                        timestamp=datetime.utcnow(),
                    ))

            # Plateau detection (for loss metrics)
            if "loss" in metric_name.lower():
                plateau = self.trend.detect_plateau(values)
                if plateau:
                    start, end = plateau
                    all_anomalies.append(Anomaly(
                        id=uuid4(),
                        anomaly_type=AnomalyType.LOSS_PLATEAU,
                        severity=Severity.WARNING,
                        metric_name=metric_name,
                        step=int(steps[start]),
                        value=float(np.mean(values[start:end])),
                        expected_range=(0.0, 0.0),
                        confidence=0.8,
                        description=f"Loss plateau detected from step {steps[start]} to {steps[end]}",
                        suggested_actions=[
                            "Consider adjusting learning rate",
                            "Check for underfitting",
                        ],
                        related_metrics=["learning_rate"],
                        timestamp=datetime.utcnow(),
                    ))

                # Divergence detection
                div_step = self.trend.detect_divergence(values)
                if div_step:
                    all_anomalies.append(Anomaly(
                        id=uuid4(),
                        anomaly_type=AnomalyType.LOSS_DIVERGENCE,
                        severity=Severity.CRITICAL,
                        metric_name=metric_name,
                        step=int(steps[div_step]),
                        value=float(values[div_step]),
                        expected_range=(0.0, 0.0),
                        confidence=0.9,
                        description=f"Loss divergence detected at step {steps[div_step]}",
                        suggested_actions=[
                            "Immediately reduce learning rate",
                            "Enable gradient clipping",
                        ],
                        related_metrics=["gradient_norm", "learning_rate"],
                        timestamp=datetime.utcnow(),
                    ))

        # Deduplicate and sort by severity
        unique_anomalies = self._deduplicate(all_anomalies)
        return sorted(
            unique_anomalies,
            key=lambda a: (
                0 if a.severity == Severity.CRITICAL else
                1 if a.severity == Severity.WARNING else 2,
                -a.confidence
            ),
        )

    def _deduplicate(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Remove duplicate anomalies at same step/metric."""
        seen = set()
        unique = []

        for a in anomalies:
            key = (a.metric_name, a.step, a.anomaly_type)
            if key not in seen:
                seen.add(key)
                unique.append(a)

        return unique

    def get_summary(self, anomalies: List[Anomaly]) -> Dict[str, Any]:
        """Generate a summary of detected anomalies."""
        if not anomalies:
            return {"status": "healthy", "anomaly_count": 0}

        by_severity = {s.value: 0 for s in Severity}
        by_type = {}

        for a in anomalies:
            by_severity[a.severity.value] += 1
            by_type[a.anomaly_type.value] = by_type.get(a.anomaly_type.value, 0) + 1

        status = "critical" if by_severity["critical"] > 0 else \
                 "warning" if by_severity["warning"] > 0 else "info"

        return {
            "status": status,
            "anomaly_count": len(anomalies),
            "by_severity": by_severity,
            "by_type": by_type,
            "most_recent": anomalies[0].to_dict() if anomalies else None,
        }
