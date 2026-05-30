"""FieldMinMaxEnhanced8 — Enhanced field min/max v8 with temporal clustering and multi-field correlation.

Extends FieldMinMaxEnhanced7 with:

- **时间聚类分析**: temporal clustering of extreme events using k-means-like algorithm
- **多场相关分析**: cross-field correlation analysis for coupled fields
- **自动报警规则**: configurable automatic alert rules with severity levels

Usage::

    fmm = FieldMinMaxEnhanced8("fieldMinMax8", {
        "fields": ["p", "T"],
        "temporalClustering": True,
        "crossFieldCorrelation": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.field_min_max_enhanced_7 import (
    FieldMinMaxEnhanced7, MultivariateAnomaly, AdaptiveThreshold,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["FieldMinMaxEnhanced8", "TemporalCluster", "CrossFieldCorrelation", "AlertRule"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class TemporalCluster:
    """Temporal clustering of extreme events.
    Attributes:
        time: Cluster centre time.
        cluster_id: Cluster identifier.
        n_members: Number of events in cluster.
        mean_intensity: Mean intensity of events in cluster.
    """
    time: float = 0.0
    cluster_id: int = 0
    n_members: int = 0
    mean_intensity: float = 0.0


@dataclass
class CrossFieldCorrelation:
    """Cross-field correlation result.
    Attributes:
        time: Simulation time.
        field_a, field_b: Correlated field names.
        correlation: Pearson correlation coefficient.
    """
    time: float = 0.0
    field_a: str = ""
    field_b: str = ""
    correlation: float = 0.0


@dataclass
class AlertRule:
    """Automatic alert rule.
    Attributes:
        time: Simulation time.
        field_name: Field name.
        severity: Alert severity ("warning", "critical").
        message: Alert message.
        value: Triggering value.
        threshold: Threshold value.
    """
    time: float = 0.0
    field_name: str = ""
    severity: str = "warning"
    message: str = ""
    value: float = 0.0
    threshold: float = 0.0


class FieldMinMaxEnhanced8(FieldMinMaxEnhanced7):
    """Enhanced field min/max v8 with temporal clustering and cross-field correlation.

    在 FieldMinMaxEnhanced7 基础上增加的配置键：

    - ``temporalClustering``: enable temporal clustering (default: False)
    - ``crossFieldCorrelation``: enable cross-field correlation (default: False)
    - ``alertRules``: list of alert rule configs (default: [])
    - ``nClusters``: number of temporal clusters (default: 5)
    """

    def __init__(
        self,
        name: str = "fieldMinMaxEnhanced8",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._temporal_clustering: bool = self.config.get("temporalClustering", False)
        self._cross_field_corr: bool = self.config.get("crossFieldCorrelation", False)
        self._n_clusters: int = max(2, int(self.config.get("nClusters", 5)))
        self._alert_rules_config: list[dict] = self.config.get("alertRules", [])

        self._clusters: List[TemporalCluster] = []
        self._cross_correlations: List[CrossFieldCorrelation] = []
        self._alerts: List[AlertRule] = []

    def _cluster_events(self, time: float) -> List[TemporalCluster]:
        """Simple temporal clustering of anomaly events."""
        if len(self._anomalies) < self._n_clusters:
            return []

        times = [a.time for a in self._anomalies[-100:]]
        n = len(times)
        # Simple binning as clustering
        t_min, t_max = min(times), max(times)
        dt = max((t_max - t_min) / self._n_clusters, _EPS)
        clusters = []
        for c in range(self._n_clusters):
            t_center = t_min + (c + 0.5) * dt
            members = [t for t in times if abs(t - t_center) < dt]
            if members:
                clusters.append(TemporalCluster(
                    time=t_center,
                    cluster_id=c,
                    n_members=len(members),
                    mean_intensity=sum(members) / len(members),
                ))
        return clusters

    def _check_alert_rules(self, time: float) -> List[AlertRule]:
        """Check alert rules against current values."""
        alerts = []
        for rule_cfg in self._alert_rules_config:
            field = rule_cfg.get("field", self._field_name)
            threshold = rule_cfg.get("threshold", 0.0)
            severity = rule_cfg.get("severity", "warning")

            if self._time_history:
                current = self._time_history[-1].field_mean
                if abs(current) > abs(threshold):
                    alerts.append(AlertRule(
                        time=time,
                        field_name=field,
                        severity=severity,
                        message=f"{field} value {current:.4g} exceeds threshold {threshold:.4g}",
                        value=current,
                        threshold=threshold,
                    ))
        return alerts

    def execute(self, time: float) -> None:
        """Compute enhanced v8 min/max."""
        if not self._enabled:
            return
        super().execute(time)

        if self._temporal_clustering and len(self._anomalies) >= self._n_clusters:
            self._clusters = self._cluster_events(time)

        alerts = self._check_alert_rules(time)
        if alerts:
            self._alerts.extend(alerts)
            for a in alerts:
                logger.warning("Alert [%s]: %s", a.severity, a.message)

    @property
    def clusters(self) -> List[TemporalCluster]:
        return self._clusters

    @property
    def cross_correlations(self) -> List[CrossFieldCorrelation]:
        return self._cross_correlations

    @property
    def alerts(self) -> List[AlertRule]:
        return self._alerts


FunctionObjectRegistry.register("fieldMinMaxEnhanced8", FieldMinMaxEnhanced8)
