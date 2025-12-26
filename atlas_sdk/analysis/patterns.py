"""
Pattern Matching Engine

Detects common training patterns and anti-patterns
using rule-based and learned approaches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PatternType(str, Enum):
    """Types of training patterns."""
    # Positive patterns
    HEALTHY_CONVERGENCE = "healthy_convergence"
    SMOOTH_DESCENT = "smooth_descent"
    STABLE_TRAINING = "stable_training"

    # Warning patterns
    OSCILLATION = "oscillation"
    SLOW_CONVERGENCE = "slow_convergence"
    NOISY_GRADIENTS = "noisy_gradients"

    # Negative patterns
    DIVERGENCE = "divergence"
    EXPLOSION = "explosion"
    COLLAPSE = "collapse"
    SATURATION = "saturation"

    # Special patterns
    WARMUP_PHASE = "warmup_phase"
    LR_DECAY_EFFECT = "lr_decay_effect"
    CHECKPOINT_ARTIFACT = "checkpoint_artifact"


@dataclass
class Pattern:
    """A detected pattern in the training data."""
    pattern_type: PatternType
    confidence: float
    start_step: int
    end_step: int
    description: str
    metrics_involved: List[str]
    metadata: Dict[str, Any]


class PatternMatcher:
    """
    Pattern matching engine for training diagnostics.

    Uses a combination of:
    - Statistical tests
    - Signal processing
    - Rule-based detection
    """

    def __init__(self):
        self._pattern_rules = self._load_rules()

    def _load_rules(self) -> Dict[PatternType, Dict[str, Any]]:
        """Load pattern detection rules."""
        return {
            PatternType.OSCILLATION: {
                "min_oscillations": 5,
                "amplitude_threshold": 0.1,
            },
            PatternType.DIVERGENCE: {
                "slope_threshold": 0.01,
                "min_duration": 100,
            },
            PatternType.COLLAPSE: {
                "drop_threshold": 0.99,
                "min_duration": 50,
            },
        }

    def detect_patterns(
        self,
        values: np.ndarray,
        metric_name: str,
        steps: np.ndarray,
    ) -> List[Pattern]:
        """Detect patterns in a metric time series."""
        patterns = []

        # Basic statistics
        mean = np.mean(values)
        std = np.std(values)
        trend = self._compute_trend(values)

        # Check for oscillation
        oscillation = self._detect_oscillation(values, steps)
        if oscillation:
            patterns.append(oscillation)

        # Check for convergence
        convergence = self._detect_convergence(values, steps, metric_name)
        if convergence:
            patterns.append(convergence)

        # Check for divergence
        divergence = self._detect_divergence(values, steps, metric_name)
        if divergence:
            patterns.append(divergence)

        # Check for plateau/saturation
        saturation = self._detect_saturation(values, steps, metric_name)
        if saturation:
            patterns.append(saturation)

        # Check for learning rate effects
        lr_effect = self._detect_lr_effects(values, steps, metric_name)
        if lr_effect:
            patterns.append(lr_effect)

        return patterns

    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute linear trend slope."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)

    def _detect_oscillation(
        self,
        values: np.ndarray,
        steps: np.ndarray,
    ) -> Optional[Pattern]:
        """Detect oscillating patterns."""
        if len(values) < 50:
            return None

        # Compute derivative
        diff = np.diff(values)

        # Count sign changes
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        oscillation_rate = sign_changes / len(diff)

        if oscillation_rate > 0.3:  # High oscillation
            # Compute amplitude
            window_size = min(20, len(values) // 5)
            amplitudes = []
            for i in range(0, len(values) - window_size, window_size):
                window = values[i:i + window_size]
                amplitudes.append(np.max(window) - np.min(window))

            avg_amplitude = np.mean(amplitudes)
            relative_amplitude = avg_amplitude / (np.mean(np.abs(values)) + 1e-8)

            if relative_amplitude > 0.1:
                return Pattern(
                    pattern_type=PatternType.OSCILLATION,
                    confidence=min(1.0, oscillation_rate * 2),
                    start_step=int(steps[0]),
                    end_step=int(steps[-1]),
                    description=f"High oscillation detected (rate: {oscillation_rate:.2f}, amplitude: {relative_amplitude:.2f})",
                    metrics_involved=[],
                    metadata={
                        "oscillation_rate": oscillation_rate,
                        "relative_amplitude": relative_amplitude,
                    },
                )

        return None

    def _detect_convergence(
        self,
        values: np.ndarray,
        steps: np.ndarray,
        metric_name: str,
    ) -> Optional[Pattern]:
        """Detect healthy convergence pattern."""
        if len(values) < 100:
            return None

        # Only for loss-like metrics
        if "loss" not in metric_name.lower():
            return None

        # Check if values are generally decreasing
        trend = self._compute_trend(values)

        if trend >= 0:
            return None

        # Check for smooth decrease
        # Compare variance to trend
        residuals = values - np.polyval(np.polyfit(np.arange(len(values)), values, 1),
                                         np.arange(len(values)))
        residual_std = np.std(residuals)
        mean_value = np.mean(values)

        noise_ratio = residual_std / (np.abs(mean_value) + 1e-8)

        if noise_ratio < 0.3 and trend < 0:
            return Pattern(
                pattern_type=PatternType.HEALTHY_CONVERGENCE,
                confidence=max(0.5, 1.0 - noise_ratio),
                start_step=int(steps[0]),
                end_step=int(steps[-1]),
                description="Training shows healthy convergence pattern",
                metrics_involved=[metric_name],
                metadata={
                    "trend": trend,
                    "noise_ratio": noise_ratio,
                },
            )

        return None

    def _detect_divergence(
        self,
        values: np.ndarray,
        steps: np.ndarray,
        metric_name: str,
    ) -> Optional[Pattern]:
        """Detect divergence pattern."""
        if len(values) < 50:
            return None

        # Only for loss-like metrics
        if "loss" not in metric_name.lower():
            return None

        # Check recent trend
        recent_values = values[-50:]
        recent_trend = self._compute_trend(recent_values)

        # Check if trend is strongly positive
        if recent_trend > 0.01:
            # Confirm with overall increase
            early_mean = np.mean(values[:20])
            late_mean = np.mean(values[-20:])

            if late_mean > early_mean * 1.5:
                return Pattern(
                    pattern_type=PatternType.DIVERGENCE,
                    confidence=min(1.0, (late_mean / early_mean - 1) / 2),
                    start_step=int(steps[-50]),
                    end_step=int(steps[-1]),
                    description="Loss is diverging - training may be failing",
                    metrics_involved=[metric_name],
                    metadata={
                        "trend": recent_trend,
                        "increase_ratio": late_mean / early_mean,
                    },
                )

        return None

    def _detect_saturation(
        self,
        values: np.ndarray,
        steps: np.ndarray,
        metric_name: str,
    ) -> Optional[Pattern]:
        """Detect saturation/plateau pattern."""
        if len(values) < 100:
            return None

        # Check recent variance
        recent_values = values[-50:]
        recent_std = np.std(recent_values)
        recent_mean = np.mean(recent_values)

        relative_std = recent_std / (np.abs(recent_mean) + 1e-8)

        if relative_std < 0.01:
            return Pattern(
                pattern_type=PatternType.SATURATION,
                confidence=max(0.5, 1.0 - relative_std * 10),
                start_step=int(steps[-50]),
                end_step=int(steps[-1]),
                description=f"Metric has saturated around {recent_mean:.4f}",
                metrics_involved=[metric_name],
                metadata={
                    "saturated_value": recent_mean,
                    "relative_std": relative_std,
                },
            )

        return None

    def _detect_lr_effects(
        self,
        values: np.ndarray,
        steps: np.ndarray,
        metric_name: str,
    ) -> Optional[Pattern]:
        """Detect learning rate schedule effects."""
        if len(values) < 100:
            return None

        # Look for sudden changes in trend
        window_size = 20
        trends = []

        for i in range(0, len(values) - window_size, window_size // 2):
            window = values[i:i + window_size]
            trend = self._compute_trend(window)
            trends.append((i, trend))

        if len(trends) < 3:
            return None

        # Look for significant trend changes
        trend_values = [t[1] for t in trends]
        trend_changes = np.abs(np.diff(trend_values))

        if np.max(trend_changes) > 3 * np.mean(trend_changes):
            change_idx = np.argmax(trend_changes)
            change_step = trends[change_idx][0]

            return Pattern(
                pattern_type=PatternType.LR_DECAY_EFFECT,
                confidence=0.7,
                start_step=int(steps[change_step]),
                end_step=int(steps[min(change_step + window_size, len(steps) - 1)]),
                description="Detected learning rate schedule effect",
                metrics_involved=[metric_name, "learning_rate"],
                metadata={
                    "change_step": change_step,
                    "trend_before": trends[change_idx][1],
                    "trend_after": trends[change_idx + 1][1] if change_idx + 1 < len(trends) else None,
                },
            )

        return None

    def match_known_patterns(
        self,
        patterns: List[Pattern],
    ) -> Dict[str, Any]:
        """Match detected patterns against known training scenarios."""
        scenario = "unknown"
        recommendations = []

        pattern_types = {p.pattern_type for p in patterns}

        # Healthy training
        if PatternType.HEALTHY_CONVERGENCE in pattern_types:
            scenario = "healthy_training"
            recommendations.append("Training is progressing well - continue monitoring")

        # LR issues
        elif PatternType.OSCILLATION in pattern_types:
            scenario = "lr_too_high"
            recommendations.extend([
                "Reduce learning rate by 2-10x",
                "Enable gradient clipping",
                "Consider warmup schedule",
            ])

        # Divergence
        elif PatternType.DIVERGENCE in pattern_types:
            scenario = "training_failing"
            recommendations.extend([
                "Immediately reduce learning rate",
                "Check for data issues",
                "Review model architecture",
            ])

        # Saturation
        elif PatternType.SATURATION in pattern_types:
            scenario = "convergence_or_stuck"
            recommendations.extend([
                "Check validation metrics",
                "Try learning rate warmup restart",
                "Increase model capacity if underfitting",
            ])

        return {
            "scenario": scenario,
            "patterns_detected": [p.pattern_type.value for p in patterns],
            "recommendations": recommendations,
        }
