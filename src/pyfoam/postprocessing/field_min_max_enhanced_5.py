"""
FieldMinMaxEnhanced5 — Enhanced field min/max v5 with anomaly detection and alerts.

在 Enhanced v4 基础上增加：

- **异常检测**：基于统计阈值的场值异常检测
- **梯度警报**：当时间变化率超过阈值时触发警报
- **自适应采样**：根据场变化率自适应调整采样频率
- **趋势分析**：线性回归拟合场值趋势

Usage::

    fmm = FieldMinMaxEnhanced5("fieldMinMax5", {
        "fields": ["p", "T"],
        "anomalySigmaThreshold": 3.0,
        "alertRateOfChange": 1e3,
        "adaptiveSampling": True,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.field_min_max_enhanced_4 import (
    FieldMinMaxEnhanced4,
    RegionStats,
    TimeHistoryEntry,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["FieldMinMaxEnhanced5", "AnomalyEvent", "TrendAnalysis"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class AnomalyEvent:
    """Detected anomaly in field statistics.

    Attributes:
        time: Simulation time when anomaly was detected.
        field_name: Field name.
        metric: Which metric triggered (min, max, mean, rate).
        value: The anomalous value.
        threshold: The threshold that was exceeded.
        sigma: Number of standard deviations from mean.
    """

    time: float = 0.0
    field_name: str = ""
    metric: str = ""
    value: float = 0.0
    threshold: float = 0.0
    sigma: float = 0.0


@dataclass
class TrendAnalysis:
    """Linear trend fitted to a field statistic over time.

    Attributes:
        metric: Which metric (min, max, mean).
        slope: Fitted slope (value per second).
        intercept: Fitted intercept.
        r_squared: Coefficient of determination.
        n_points: Number of data points used.
    """

    metric: str = ""
    slope: float = 0.0
    intercept: float = 0.0
    r_squared: float = 0.0
    n_points: int = 0


class FieldMinMaxEnhanced5(FieldMinMaxEnhanced4):
    """Enhanced field min/max v5 with anomaly detection and trend analysis.

    在 FieldMinMaxEnhanced4 基础上增加的配置键：

    - ``anomalySigmaThreshold``: sigma threshold for anomaly detection (default: 3.0)
    - ``alertRateOfChange``: rate-of-change threshold for alerts (default: inf)
    - ``adaptiveSampling``: enable adaptive sampling rate (default: False)
    - ``trendAnalysis``: enable linear trend fitting (default: True)
    - ``maxAnomalies``: maximum stored anomaly events (default: 1000)
    """

    def __init__(
        self,
        name: str = "fieldMinMaxEnhanced5",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._anomaly_sigma: float = float(
            self.config.get("anomalySigmaThreshold", 3.0)
        )
        self._alert_rate: float = float(
            self.config.get("alertRateOfChange", float("inf"))
        )
        self._adaptive_sampling: bool = self.config.get("adaptiveSampling", False)
        self._trend_analysis: bool = self.config.get("trendAnalysis", True)
        self._max_anomalies: int = max(10, int(self.config.get("maxAnomalies", 1000)))

        # Storage
        self._anomalies: List[AnomalyEvent] = []
        self._trends: Dict[str, TrendAnalysis] = {}
        self._sample_interval: float = 1.0
        self._current_interval: float = 1.0

    # ------------------------------------------------------------------
    # 异常检测
    # ------------------------------------------------------------------

    def _detect_anomalies(
        self, data: torch.Tensor, time: float, field_name: str,
    ) -> List[AnomalyEvent]:
        """Detect anomalies in current field data based on historical statistics."""
        if len(self._time_history) < 3:
            return []

        # Historical statistics
        hist_means = [e.field_mean for e in self._time_history]
        hist_maxs = [e.field_max for e in self._time_history]
        hist_mins = [e.field_min for e in self._time_history]

        mean_of_means = sum(hist_means) / len(hist_means)
        std_of_means = (
            sum((m - mean_of_means) ** 2 for m in hist_means) / max(len(hist_means) - 1, 1)
        ) ** 0.5

        current_mean = float(data.mean().item())
        current_max = float(data.max().item())
        current_min = float(data.min().item())

        events: List[AnomalyEvent] = []

        # Mean anomaly
        if std_of_means > _EPS:
            sigma = abs(current_mean - mean_of_means) / std_of_means
            if sigma > self._anomaly_sigma:
                events.append(AnomalyEvent(
                    time=time, field_name=field_name,
                    metric="mean", value=current_mean,
                    threshold=mean_of_means + self._anomaly_sigma * std_of_means,
                    sigma=sigma,
                ))

        # Rate anomaly
        if self._time_history:
            rate = self._time_history[-1].rate_of_change
            if abs(rate) > self._alert_rate:
                events.append(AnomalyEvent(
                    time=time, field_name=field_name,
                    metric="rate", value=rate,
                    threshold=self._alert_rate,
                    sigma=0.0,
                ))

        # Prune anomalies
        if len(self._anomalies) + len(events) > self._max_anomalies:
            self._anomalies = self._anomalies[-(self._max_anomalies - len(events)):]

        return events

    # ------------------------------------------------------------------
    # 趋势分析
    # ------------------------------------------------------------------

    def _compute_trends(self) -> Dict[str, TrendAnalysis]:
        """Fit linear trends to min, max, mean time histories."""
        if len(self._time_history) < 3:
            return {}

        times = torch.tensor([e.time for e in self._time_history], dtype=torch.float64)
        trends: Dict[str, TrendAnalysis] = {}

        for metric_name, accessor in [
            ("mean", lambda e: e.field_mean),
            ("max", lambda e: e.field_max),
            ("min", lambda e: e.field_min),
        ]:
            values = torch.tensor(
                [accessor(e) for e in self._time_history], dtype=torch.float64,
            )
            n = times.numel()

            # Least-squares linear fit: y = a*x + b
            t_mean = times.mean()
            v_mean = values.mean()
            cov = ((times - t_mean) * (values - v_mean)).sum()
            var_t = ((times - t_mean) ** 2).sum().clamp(min=_EPS)

            slope = float((cov / var_t).item())
            intercept = float((v_mean - slope * t_mean).item())

            # R^2
            v_pred = slope * times + intercept
            ss_res = ((values - v_pred) ** 2).sum()
            ss_tot = ((values - v_mean) ** 2).sum().clamp(min=_EPS)
            r_sq = float((1.0 - ss_res / ss_tot).clamp(0.0, 1.0).item())

            trends[metric_name] = TrendAnalysis(
                metric=metric_name,
                slope=slope,
                intercept=intercept,
                r_squared=r_sq,
                n_points=n,
            )

        return trends

    # ------------------------------------------------------------------
    # 自适应采样
    # ------------------------------------------------------------------

    def _update_sampling_interval(self, data: torch.Tensor) -> None:
        """Adjust sampling interval based on field activity."""
        if not self._adaptive_sampling or not self._time_history:
            return

        rate = abs(self._time_history[-1].rate_of_change)
        data_range = float(data.max().item() - data.min().item())

        # Increase sampling when activity is high
        if rate > _EPS:
            self._current_interval = max(
                self._sample_interval * 0.1,
                min(self._sample_interval, data_range / rate * 0.1),
            )
        else:
            self._current_interval = self._sample_interval

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute enhanced v5 min/max with anomaly detection."""
        if not self._enabled:
            return

        # Run parent v4 execute
        super().execute(time)

        field = self._fields.get(self._field_name)
        if field is None:
            return

        data = self._extract_field_data(field)

        # Anomaly detection
        anomalies = self._detect_anomalies(data, time, self._field_name)
        self._anomalies.extend(anomalies)

        for a in anomalies:
            logger.warning(
                "Anomaly detected: t=%g field=%s metric=%s value=%.6g sigma=%.2f",
                a.time, a.field_name, a.metric, a.value, a.sigma,
            )

        # Adaptive sampling
        self._update_sampling_interval(data)

        # Trend analysis (periodically)
        if self._trend_analysis and len(self._time_history) % 10 == 0:
            self._trends = self._compute_trends()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def anomalies(self) -> List[AnomalyEvent]:
        """Detected anomaly events."""
        return self._anomalies

    @property
    def trends(self) -> Dict[str, TrendAnalysis]:
        """Computed trend analyses."""
        return self._trends

    @property
    def current_sample_interval(self) -> float:
        """Current adaptive sampling interval."""
        return self._current_interval

    def get_trend(self, metric: str = "mean") -> Optional[TrendAnalysis]:
        """Get trend analysis for a specific metric."""
        return self._trends.get(metric)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write v5 enhanced min/max data."""
        super().write()

        if self._output_path is None:
            return

        # Write anomaly events
        if self._anomalies:
            anomaly_file = self._output_path / "anomalies.dat"
            with open(anomaly_file, "w") as f:
                f.write("# time  field  metric  value  threshold  sigma\n")
                for a in self._anomalies:
                    f.write(
                        f"{a.time:.6e}  {a.field_name}  {a.metric}  "
                        f"{a.value:.10g}  {a.threshold:.10g}  {a.sigma:.2f}\n"
                    )

        # Write trend analysis
        if self._trends:
            trend_file = self._output_path / "trends.dat"
            with open(trend_file, "w") as f:
                f.write("# metric  slope  intercept  r_squared  n_points\n")
                for name, t in self._trends.items():
                    f.write(
                        f"{t.metric}  {t.slope:.6e}  {t.intercept:.6e}  "
                        f"{t.r_squared:.6f}  {t.n_points}\n"
                    )

        logger.info("Wrote FieldMinMaxEnhanced5 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("fieldMinMaxEnhanced5", FieldMinMaxEnhanced5)
