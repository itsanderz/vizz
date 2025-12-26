"""
Insight Generation Engine

Transforms anomalies and patterns into actionable insights
with AI-powered hypothesis generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np

from atlas_sdk.analysis.anomaly import Anomaly, AnomalyType, Severity


class InsightCategory(str, Enum):
    """Categories of insights."""
    TRAINING_PROGRESS = "training_progress"
    CONVERGENCE = "convergence"
    OPTIMIZATION = "optimization"
    DATA_QUALITY = "data_quality"
    HYPERPARAMETER = "hyperparameter"
    ARCHITECTURE = "architecture"
    RESOURCE_UTILIZATION = "resource_utilization"
    COMPARISON = "comparison"


@dataclass
class Insight:
    """An actionable insight with supporting evidence."""
    id: UUID
    category: InsightCategory
    title: str
    summary: str
    detailed_analysis: str
    evidence: List[Dict[str, Any]]
    recommendations: List[str]
    confidence: float
    severity: Severity
    related_metrics: List[str]
    visualization_spec: Optional[Dict[str, Any]]
    created_at: datetime
    votes_up: int = 0
    votes_down: int = 0
    user_feedback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "category": self.category.value,
            "title": self.title,
            "summary": self.summary,
            "detailed_analysis": self.detailed_analysis,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "related_metrics": self.related_metrics,
            "visualization_spec": self.visualization_spec,
            "created_at": self.created_at.isoformat(),
            "votes_up": self.votes_up,
            "votes_down": self.votes_down,
        }


@dataclass
class TrainingMetrics:
    """Aggregated training metrics for analysis."""
    total_steps: int
    elapsed_time_seconds: float
    loss_current: float
    loss_initial: float
    loss_best: float
    loss_trend: float  # Slope of recent loss
    learning_rate_current: float
    gradient_norm_mean: float
    gradient_norm_max: float
    throughput_samples_per_sec: float
    gpu_utilization_mean: float
    memory_utilization_mean: float


class InsightGenerator:
    """
    Enterprise insight generation engine.

    Combines:
    - Anomaly analysis
    - Statistical pattern recognition
    - Historical comparison
    - AI-powered hypothesis generation
    """

    def __init__(self, enable_ai: bool = True):
        self.enable_ai = enable_ai
        self._insight_templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load insight templates for common patterns."""
        return {
            "healthy_training": {
                "title": "Training Progressing Normally",
                "summary": "Loss is decreasing steadily with no anomalies detected.",
                "category": InsightCategory.TRAINING_PROGRESS,
                "severity": Severity.INFO,
            },
            "convergence_detected": {
                "title": "Model Approaching Convergence",
                "summary": "Loss reduction rate has slowed significantly, suggesting convergence.",
                "category": InsightCategory.CONVERGENCE,
                "severity": Severity.INFO,
            },
            "lr_too_high": {
                "title": "Learning Rate May Be Too High",
                "summary": "Loss oscillations suggest the learning rate should be reduced.",
                "category": InsightCategory.HYPERPARAMETER,
                "severity": Severity.WARNING,
            },
            "lr_too_low": {
                "title": "Learning Rate May Be Too Low",
                "summary": "Slow convergence suggests the learning rate could be increased.",
                "category": InsightCategory.HYPERPARAMETER,
                "severity": Severity.INFO,
            },
            "overfitting_detected": {
                "title": "Potential Overfitting Detected",
                "summary": "Training loss continues to decrease while validation loss increases.",
                "category": InsightCategory.TRAINING_PROGRESS,
                "severity": Severity.WARNING,
            },
            "gpu_underutilized": {
                "title": "GPU Underutilization",
                "summary": "GPU utilization is below optimal levels. Consider increasing batch size.",
                "category": InsightCategory.RESOURCE_UTILIZATION,
                "severity": Severity.INFO,
            },
            "gradient_issues": {
                "title": "Gradient Health Warning",
                "summary": "Gradient statistics indicate potential training instability.",
                "category": InsightCategory.OPTIMIZATION,
                "severity": Severity.WARNING,
            },
        }

    def generate_from_anomalies(
        self,
        anomalies: List[Anomaly],
    ) -> List[Insight]:
        """Generate insights from detected anomalies."""
        insights = []

        # Group anomalies by type
        by_type: Dict[AnomalyType, List[Anomaly]] = {}
        for a in anomalies:
            if a.anomaly_type not in by_type:
                by_type[a.anomaly_type] = []
            by_type[a.anomaly_type].append(a)

        # Generate insights for each type
        for anomaly_type, type_anomalies in by_type.items():
            insight = self._anomaly_to_insight(anomaly_type, type_anomalies)
            if insight:
                insights.append(insight)

        return insights

    def _anomaly_to_insight(
        self,
        anomaly_type: AnomalyType,
        anomalies: List[Anomaly],
    ) -> Optional[Insight]:
        """Convert anomalies of a type to an insight."""
        if not anomalies:
            return None

        # Most severe anomaly of this type
        primary = max(anomalies, key=lambda a: a.confidence)

        # Build evidence
        evidence = [a.to_dict() for a in anomalies[:5]]  # Top 5

        # Generate insight based on type
        if anomaly_type == AnomalyType.LOSS_SPIKE:
            return Insight(
                id=uuid4(),
                category=InsightCategory.TRAINING_PROGRESS,
                title="Loss Spike Detected",
                summary=f"Detected {len(anomalies)} loss spike(s) during training",
                detailed_analysis=self._analyze_loss_spikes(anomalies),
                evidence=evidence,
                recommendations=primary.suggested_actions,
                confidence=primary.confidence,
                severity=primary.severity,
                related_metrics=["loss", "learning_rate", "gradient_norm"],
                visualization_spec=self._spike_visualization_spec(anomalies),
                created_at=datetime.utcnow(),
            )

        elif anomaly_type == AnomalyType.LOSS_PLATEAU:
            return Insight(
                id=uuid4(),
                category=InsightCategory.CONVERGENCE,
                title="Training Plateau Detected",
                summary="Loss has stopped decreasing significantly",
                detailed_analysis=self._analyze_plateau(anomalies),
                evidence=evidence,
                recommendations=[
                    "Try cosine annealing or cyclical learning rate",
                    "Consider increasing model capacity",
                    "Add regularization if overfitting",
                    "Check if more data would help",
                ],
                confidence=primary.confidence,
                severity=Severity.WARNING,
                related_metrics=["loss", "learning_rate"],
                visualization_spec=None,
                created_at=datetime.utcnow(),
            )

        elif anomaly_type == AnomalyType.GRADIENT_EXPLOSION:
            return Insight(
                id=uuid4(),
                category=InsightCategory.OPTIMIZATION,
                title="Gradient Explosion Detected",
                summary="Gradients are growing unboundedly - immediate action required",
                detailed_analysis=self._analyze_gradient_issues(anomalies, "explosion"),
                evidence=evidence,
                recommendations=[
                    "Enable gradient clipping (recommended: max_norm=1.0)",
                    "Reduce learning rate by 10x",
                    "Check for numerical overflow in loss computation",
                    "Review model architecture for instability",
                ],
                confidence=1.0,
                severity=Severity.CRITICAL,
                related_metrics=["gradient_norm", "loss", "learning_rate"],
                visualization_spec=None,
                created_at=datetime.utcnow(),
            )

        elif anomaly_type == AnomalyType.LOSS_NAN:
            return Insight(
                id=uuid4(),
                category=InsightCategory.OPTIMIZATION,
                title="NaN Values Detected",
                summary="Training produced NaN values - training has failed",
                detailed_analysis="NaN values indicate numerical instability. Common causes:\n"
                    "1. Division by zero in loss computation\n"
                    "2. Log of zero or negative numbers\n"
                    "3. Gradient explosion followed by overflow\n"
                    "4. Incorrect loss function implementation",
                evidence=evidence,
                recommendations=[
                    "Add epsilon (1e-8) to log and division operations",
                    "Enable anomaly detection: torch.autograd.set_detect_anomaly(True)",
                    "Check input data for NaN/Inf values",
                    "Use torch.nan_to_num() for intermediate computations",
                ],
                confidence=1.0,
                severity=Severity.CRITICAL,
                related_metrics=["loss"],
                visualization_spec=None,
                created_at=datetime.utcnow(),
            )

        return None

    def _analyze_loss_spikes(self, anomalies: List[Anomaly]) -> str:
        """Generate detailed analysis for loss spikes."""
        steps = [a.step for a in anomalies]
        values = [a.value for a in anomalies]

        analysis = f"Detected {len(anomalies)} loss spike(s) at steps: {steps[:5]}...\n\n"

        # Check for patterns
        if len(steps) > 1:
            intervals = np.diff(steps)
            if np.std(intervals) < np.mean(intervals) * 0.1:
                analysis += f"Pattern: Spikes occur regularly every ~{np.mean(intervals):.0f} steps. "
                analysis += "This may indicate issues with specific data batches or a cyclic LR schedule.\n\n"

        # Severity assessment
        avg_spike = np.mean(values)
        analysis += f"Average spike magnitude: {avg_spike:.4f}\n"

        if avg_spike > 10:
            analysis += "SEVERE: Spikes are very large, indicating potential data issues or LR problems.\n"
        elif avg_spike > 2:
            analysis += "MODERATE: Spikes are notable but may be recoverable.\n"
        else:
            analysis += "MINOR: Spikes are small and may self-correct.\n"

        return analysis

    def _analyze_plateau(self, anomalies: List[Anomaly]) -> str:
        """Generate detailed analysis for plateau."""
        if not anomalies:
            return "Plateau detected but no details available."

        primary = anomalies[0]
        return (
            f"Loss has plateaued around {primary.value:.4f} starting at step {primary.step}.\n\n"
            "This could indicate:\n"
            "1. Model has converged (check validation metrics)\n"
            "2. Learning rate is too low to make further progress\n"
            "3. Model capacity is insufficient\n"
            "4. Local minimum - consider warm restarts\n"
        )

    def _analyze_gradient_issues(self, anomalies: List[Anomaly], issue_type: str) -> str:
        """Generate detailed analysis for gradient issues."""
        if issue_type == "explosion":
            return (
                "Gradient explosion occurs when gradient magnitudes grow exponentially.\n\n"
                "Root causes:\n"
                "1. Learning rate too high for this architecture\n"
                "2. Deep networks without proper normalization\n"
                "3. Recurrent networks with long sequences\n"
                "4. Unstable loss landscape\n\n"
                "Immediate mitigation: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)"
            )
        return "Gradient health issues detected."

    def _spike_visualization_spec(self, anomalies: List[Anomaly]) -> Dict[str, Any]:
        """Generate Vega-Lite spec for spike visualization."""
        spike_data = [
            {"step": a.step, "value": a.value, "type": "spike"}
            for a in anomalies
        ]

        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": "Loss Spikes",
            "data": {"values": spike_data},
            "mark": {"type": "point", "size": 100, "color": "red"},
            "encoding": {
                "x": {"field": "step", "type": "quantitative"},
                "y": {"field": "value", "type": "quantitative"},
            },
        }

    def generate_training_summary(
        self,
        metrics: TrainingMetrics,
        anomalies: List[Anomaly],
    ) -> Insight:
        """Generate overall training summary insight."""
        # Determine training health
        critical_count = sum(1 for a in anomalies if a.severity == Severity.CRITICAL)
        warning_count = sum(1 for a in anomalies if a.severity == Severity.WARNING)

        if critical_count > 0:
            severity = Severity.CRITICAL
            title = "Training Issues Detected - Action Required"
            summary = f"Found {critical_count} critical issues requiring immediate attention."
        elif warning_count > 0:
            severity = Severity.WARNING
            title = "Training Progressing with Warnings"
            summary = f"Training is progressing but {warning_count} warnings detected."
        else:
            severity = Severity.INFO
            title = "Training Progressing Normally"
            summary = "No significant issues detected."

        # Calculate progress metrics
        loss_reduction = (metrics.loss_initial - metrics.loss_current) / metrics.loss_initial
        eta_to_target = None

        detailed = f"""
## Training Progress Summary

**Steps Completed:** {metrics.total_steps:,}
**Time Elapsed:** {metrics.elapsed_time_seconds/3600:.1f} hours
**Throughput:** {metrics.throughput_samples_per_sec:.1f} samples/sec

### Loss Analysis
- Current Loss: {metrics.loss_current:.4f}
- Best Loss: {metrics.loss_best:.4f}
- Loss Reduction: {loss_reduction*100:.1f}%
- Trend: {"Decreasing" if metrics.loss_trend < 0 else "Increasing/Stable"}

### Resource Utilization
- GPU Utilization: {metrics.gpu_utilization_mean:.1f}%
- Memory Utilization: {metrics.memory_utilization_mean:.1f}%

### Gradient Health
- Mean Gradient Norm: {metrics.gradient_norm_mean:.4f}
- Max Gradient Norm: {metrics.gradient_norm_max:.4f}
"""

        recommendations = []
        if metrics.gpu_utilization_mean < 70:
            recommendations.append("Consider increasing batch size to improve GPU utilization")
        if metrics.loss_trend >= 0:
            recommendations.append("Loss is not decreasing - review learning rate and architecture")
        if metrics.gradient_norm_max > 10:
            recommendations.append("Enable gradient clipping to prevent instability")

        return Insight(
            id=uuid4(),
            category=InsightCategory.TRAINING_PROGRESS,
            title=title,
            summary=summary,
            detailed_analysis=detailed,
            evidence=[],
            recommendations=recommendations or ["Continue monitoring"],
            confidence=0.9,
            severity=severity,
            related_metrics=["loss", "learning_rate", "gradient_norm"],
            visualization_spec=None,
            created_at=datetime.utcnow(),
        )

    def generate_comparison_insight(
        self,
        run_a_id: str,
        run_b_id: str,
        metric_name: str,
        data_a: np.ndarray,
        data_b: np.ndarray,
    ) -> Insight:
        """Generate insight comparing two runs."""
        # Calculate comparison statistics
        final_a = float(data_a[-1]) if len(data_a) > 0 else 0
        final_b = float(data_b[-1]) if len(data_b) > 0 else 0
        best_a = float(np.min(data_a)) if len(data_a) > 0 else 0
        best_b = float(np.min(data_b)) if len(data_b) > 0 else 0

        # Determine winner
        if best_a < best_b:
            winner = run_a_id
            improvement = (best_b - best_a) / best_b * 100
            summary = f"Run A achieved {improvement:.1f}% better {metric_name}"
        else:
            winner = run_b_id
            improvement = (best_a - best_b) / best_a * 100
            summary = f"Run B achieved {improvement:.1f}% better {metric_name}"

        detailed = f"""
## Run Comparison: {metric_name}

| Metric | Run A | Run B |
|--------|-------|-------|
| Final Value | {final_a:.4f} | {final_b:.4f} |
| Best Value | {best_a:.4f} | {best_b:.4f} |
| Total Steps | {len(data_a)} | {len(data_b)} |

**Winner:** {winner}
"""

        return Insight(
            id=uuid4(),
            category=InsightCategory.COMPARISON,
            title=f"Run Comparison: {metric_name}",
            summary=summary,
            detailed_analysis=detailed,
            evidence=[],
            recommendations=[
                f"Review config differences to understand why {'A' if winner == run_a_id else 'B'} performed better",
            ],
            confidence=0.95,
            severity=Severity.INFO,
            related_metrics=[metric_name],
            visualization_spec=self._comparison_visualization_spec(run_a_id, run_b_id, metric_name),
            created_at=datetime.utcnow(),
        )

    def _comparison_visualization_spec(
        self,
        run_a_id: str,
        run_b_id: str,
        metric_name: str,
    ) -> Dict[str, Any]:
        """Generate Vega-Lite spec for run comparison."""
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": f"Comparison: {metric_name}",
            "width": "container",
            "height": 300,
            "data": {"name": "comparison"},
            "mark": "line",
            "encoding": {
                "x": {"field": "step", "type": "quantitative"},
                "y": {"field": "value", "type": "quantitative"},
                "color": {
                    "field": "run_id",
                    "type": "nominal",
                    "legend": {"title": "Run"},
                },
            },
        }
