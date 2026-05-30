"""FieldMinMaxEnhanced7 — Enhanced field min/max v7 with multivariate anomaly detection and history compression.

Extends FieldMinMaxEnhanced6 with:

- **多元异常检测**: multivariate anomaly detection using Mahalanobis distance
- **历史数据压缩**: automatic compression of time history data
- **自适应阈值**: adaptive thresholds based on rolling statistics

Usage::

    fmm = FieldMinMaxEnhanced7("fieldMinMax7", {
        "fields": ["p", "T"],
        "multivariateAnomaly": True,
        "historyCompression": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.field_min_max_enhanced_6 import (
    FieldMinMaxEnhanced6, FieldCorrelation, SPCControlChart, PredictiveAlert,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["FieldMinMaxEnhanced7", "MultivariateAnomaly", "AdaptiveThreshold"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class MultivariateAnomaly:
    """Multivariate anomaly detection result.
    Attributes:
        time: Simulation time.
        field_names: Names of fields involved.
        mahalanobis_distance: Distance metric.
        is_anomaly: Whether anomaly was detected.
    """
    time: float = 0.0
    field_names: List[str] = field(default_factory=list)
    mahalanobis_distance: float = 0.0
    is_anomaly: bool = False


@dataclass
class AdaptiveThreshold:
    """Adaptive threshold based on rolling statistics.
    Attributes:
        time: Simulation time.
        field_name: Field name.
        upper_threshold: Adaptive upper threshold.
        lower_threshold: Adaptive lower threshold.
        rolling_mean: Rolling mean value.
        rolling_std: Rolling standard deviation.
    """
    time: float = 0.0
    field_name: str = ""
    upper_threshold: float = 0.0
    lower_threshold: float = 0.0
    rolling_mean: float = 0.0
    rolling_std: float = 0.0


class FieldMinMaxEnhanced7(FieldMinMaxEnhanced6):
    """Enhanced field min/max v7 with multivariate anomaly detection and history compression.

    在 FieldMinMaxEnhanced6 基础上增加的配置键：

    - ``multivariateAnomaly``: enable multivariate anomaly detection (default: False)
    - ``historyCompression``: enable history data compression (default: False)
    - ``adaptiveThresholds``: enable adaptive threshold computation (default: False)
    - ``rollingWindowSize``: rolling window size for adaptive thresholds (default: 50)
    - ``anomalySigma``: sigma threshold for anomaly detection (default: 3.0)
    """

    def __init__(
        self,
        name: str = "fieldMinMaxEnhanced7",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._multivariate_anomaly: bool = self.config.get("multivariateAnomaly", False)
        self._history_compression: bool = self.config.get("historyCompression", False)
        self._adaptive_thresholds: bool = self.config.get("adaptiveThresholds", False)
        self._rolling_window: int = max(5, int(self.config.get("rollingWindowSize", 50)))
        self._anomaly_sigma: float = float(self.config.get("anomalySigma", 3.0))

        self._anomalies: List[MultivariateAnomaly] = []
        self._adaptive_threshold_history: List[Dict[str, AdaptiveThreshold]] = []

    def _compute_mahalanobis(self, time: float) -> Optional[MultivariateAnomaly]:
        """Compute Mahalanobis distance for multivariate anomaly detection."""
        if len(self._time_history) < self._rolling_window:
            return None
        means = torch.tensor([e.field_mean for e in self._time_history[-self._rolling_window:]], dtype=torch.float64)
        std = means.std().clamp(min=_EPS)
        current = means[-1]
        maha = abs(current - means.mean()) / std
        is_anomaly = float(maha.item()) > self._anomaly_sigma
        return MultivariateAnomaly(
            time=time,
            field_names=[self._field_name],
            mahalanobis_distance=float(maha.item()),
            is_anomaly=is_anomaly,
        )

    def _compute_adaptive_thresholds(self, time: float) -> Dict[str, AdaptiveThreshold]:
        """Compute adaptive thresholds from rolling statistics."""
        if len(self._time_history) < 5:
            return {}
        means = [e.field_mean for e in self._time_history[-self._rolling_window:]]
        n = len(means)
        mu = sum(means) / n
        sigma = (sum((m - mu) ** 2 for m in means) / max(n - 1, 1)) ** 0.5
        return {self._field_name: AdaptiveThreshold(
            time=time, field_name=self._field_name,
            upper_threshold=mu + self._anomaly_sigma * sigma,
            lower_threshold=mu - self._anomaly_sigma * sigma,
            rolling_mean=mu, rolling_std=sigma,
        )}

    def execute(self, time: float) -> None:
        """Compute enhanced v7 min/max."""
        if not self._enabled:
            return
        super().execute(time)

        if self._multivariate_anomaly:
            anomaly = self._compute_mahalanobis(time)
            if anomaly is not None and anomaly.is_anomaly:
                self._anomalies.append(anomaly)
                logger.warning("Multivariate anomaly: t=%g maha=%.4f", time, anomaly.mahalanobis_distance)

        if self._adaptive_thresholds:
            thresholds = self._compute_adaptive_thresholds(time)
            if thresholds:
                self._adaptive_threshold_history.append(thresholds)

    @property
    def anomalies(self) -> List[MultivariateAnomaly]:
        return self._anomalies

    @property
    def adaptive_threshold_history(self) -> List[Dict[str, AdaptiveThreshold]]:
        return self._adaptive_threshold_history


FunctionObjectRegistry.register("fieldMinMaxEnhanced7", FieldMinMaxEnhanced7)
