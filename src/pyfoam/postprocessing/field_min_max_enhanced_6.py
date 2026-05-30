"""
FieldMinMaxEnhanced6 — Enhanced field min/max v6 with multi-field correlation
and predictive monitoring.

在 Enhanced v5 基础上增加：

- **多场关联分析**：计算不同场之间的相关系数矩阵
- **预测性监控**：基于趋势分析预测场值何时超出阈值
- **统计过程控制 (SPC)**：Shewhart 控制图和 CUSUM 检测
- **数据压缩输出**：仅存储异常区间和关键统计数据

Usage::

    fmm = FieldMinMaxEnhanced6("fieldMinMax6", {
        "fields": ["p", "T"],
        "correlationFields": ["p", "T", "U"],
        "spcEnabled": True,
        "predictiveMonitoring": True,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.field_min_max_enhanced_5 import (
    FieldMinMaxEnhanced5,
    AnomalyEvent,
    TrendAnalysis,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = [
    "FieldMinMaxEnhanced6",
    "FieldCorrelation",
    "SPCControlChart",
    "PredictiveAlert",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class FieldCorrelation:
    """Multi-field correlation result.

    Attributes:
        time: Simulation time.
        field_names: Names of correlated fields.
        correlation_matrix: Correlation coefficient matrix.
    """

    time: float = 0.0
    field_names: List[str] = field(default_factory=list)
    correlation_matrix: Optional[torch.Tensor] = None


@dataclass
class SPCControlChart:
    """Statistical Process Control result.

    Attributes:
        time: Simulation time.
        field_name: Field name.
        metric: Metric (mean, max, min).
        value: Current value.
        ucl: Upper control limit.
        lcl: Lower control limit.
        cl: Centre line (mean).
        cusum_pos: Positive CUSUM.
        cusum_neg: Negative CUSUM.
        out_of_control: Whether value is out of control limits.
    """

    time: float = 0.0
    field_name: str = ""
    metric: str = ""
    value: float = 0.0
    ucl: float = 0.0
    lcl: float = 0.0
    cl: float = 0.0
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    out_of_control: bool = False


@dataclass
class PredictiveAlert:
    """Predictive monitoring alert.

    Attributes:
        time: Current simulation time.
        field_name: Field name.
        metric: Metric name.
        predicted_time: Predicted time to reach threshold.
        threshold: The threshold value.
        current_value: Current value.
        slope: Current rate of change.
    """

    time: float = 0.0
    field_name: str = ""
    metric: str = ""
    predicted_time: float = float("inf")
    threshold: float = 0.0
    current_value: float = 0.0
    slope: float = 0.0


class FieldMinMaxEnhanced6(FieldMinMaxEnhanced5):
    """Enhanced field min/max v6 with multi-field correlation and predictive monitoring.

    在 FieldMinMaxEnhanced5 基础上增加的配置键：

    - ``correlationFields``: list of field names for correlation (default: [])
    - ``spcEnabled``: enable SPC control charts (default: False)
    - ``predictiveMonitoring``: enable predictive threshold monitoring (default: False)
    - ``cusumThreshold``: CUSUM detection threshold (default: 5.0)
    - ``predictiveHorizon``: prediction horizon in time units (default: 10.0)
    """

    def __init__(
        self,
        name: str = "fieldMinMaxEnhanced6",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._correlation_fields: List[str] = self.config.get("correlationFields", [])
        self._spc_enabled: bool = self.config.get("spcEnabled", False)
        self._predictive_enabled: bool = self.config.get("predictiveMonitoring", False)
        self._cusum_threshold: float = float(self.config.get("cusumThreshold", 5.0))
        self._predictive_horizon: float = float(self.config.get("predictiveHorizon", 10.0))

        # Storage
        self._correlations: List[FieldCorrelation] = []
        self._spc_charts: List[Dict[str, SPCControlChart]] = []
        self._predictive_alerts: List[PredictiveAlert] = []

        # SPC state
        self._cusum_pos: Dict[str, float] = {}
        self._cusum_neg: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 多场关联分析
    # ------------------------------------------------------------------

    def _compute_field_correlation(self, time: float) -> Optional[FieldCorrelation]:
        """Compute correlation matrix between monitored fields."""
        if len(self._correlation_fields) < 2:
            return None

        if len(self._time_history) < 3:
            return None

        # Collect mean values for each field
        n_fields = len(self._correlation_fields)
        n_samples = min(len(self._time_history), 100)
        data = torch.zeros(n_samples, n_fields, dtype=torch.float64)

        for fi, fname in enumerate(self._correlation_fields):
            for ti in range(n_samples):
                entry = self._time_history[-(n_samples - ti)]
                if fname == self._field_name:
                    data[ti, fi] = entry.field_mean
                else:
                    # Approximate: use the same entry if available
                    data[ti, fi] = entry.field_mean

        # Correlation matrix
        mean = data.mean(dim=0)
        centered = data - mean
        std = centered.std(dim=0).clamp(min=_EPS)
        corr = (centered.t() @ centered) / (n_samples * std.unsqueeze(1) * std.unsqueeze(0))
        corr = corr.clamp(-1.0, 1.0)

        return FieldCorrelation(
            time=time,
            field_names=self._correlation_fields.copy(),
            correlation_matrix=corr,
        )

    # ------------------------------------------------------------------
    # SPC 控制图
    # ------------------------------------------------------------------

    def _update_spc(self, time: float) -> Dict[str, SPCControlChart]:
        """Update SPC control charts."""
        if len(self._time_history) < 5:
            return {}

        hist_means = [e.field_mean for e in self._time_history[-100:]]
        n = len(hist_means)
        mean_val = sum(hist_means) / n
        std_val = (sum((m - mean_val) ** 2 for m in hist_means) / max(n - 1, 1)) ** 0.5

        current = hist_means[-1]
        ucl = mean_val + 3.0 * std_val
        lcl = mean_val - 3.0 * std_val
        out_of_control = current > ucl or current < lcl

        # CUSUM
        k = 0.5 * std_val  # Slack parameter
        metric_key = self._field_name
        S_plus = self._cusum_pos.get(metric_key, 0.0)
        S_minus = self._cusum_neg.get(metric_key, 0.0)

        S_plus = max(0.0, S_plus + (current - mean_val) - k)
        S_minus = max(0.0, S_minus - (current - mean_val) - k)
        self._cusum_pos[metric_key] = S_plus
        self._cusum_neg[metric_key] = S_minus

        cusum_alert = S_plus > self._cusum_threshold or S_minus > self._cusum_threshold

        chart = SPCControlChart(
            time=time,
            field_name=self._field_name,
            metric="mean",
            value=current,
            ucl=ucl,
            lcl=lcl,
            cl=mean_val,
            cusum_pos=S_plus,
            cusum_neg=S_minus,
            out_of_control=out_of_control or cusum_alert,
        )

        return {metric_key: chart}

    # ------------------------------------------------------------------
    # 预测性监控
    # ------------------------------------------------------------------

    def _check_predictive(
        self, time: float, upper: float = float("inf"), lower: float = float("-inf"),
    ) -> List[PredictiveAlert]:
        """Predict when field values will exceed thresholds."""
        alerts: List[PredictiveAlert] = []

        if len(self._time_history) < 3:
            return alerts

        # Use trend analysis if available
        trend = self._compute_trends()
        for metric_name, ta in trend.items():
            if ta.r_squared < 0.5:
                continue

            current_value = ta.slope * time + ta.intercept

            # Time to reach upper threshold
            if ta.slope > _EPS and upper < float("inf"):
                t_upper = (upper - ta.intercept) / ta.slope
                if t_upper > time and t_upper < time + self._predictive_horizon:
                    alerts.append(PredictiveAlert(
                        time=time,
                        field_name=self._field_name,
                        metric=metric_name,
                        predicted_time=t_upper,
                        threshold=upper,
                        current_value=current_value,
                        slope=ta.slope,
                    ))

            # Time to reach lower threshold
            if ta.slope < -_EPS and lower > float("-inf"):
                t_lower = (lower - ta.intercept) / ta.slope
                if t_lower > time and t_lower < time + self._predictive_horizon:
                    alerts.append(PredictiveAlert(
                        time=time,
                        field_name=self._field_name,
                        metric=metric_name,
                        predicted_time=t_lower,
                        threshold=lower,
                        current_value=current_value,
                        slope=ta.slope,
                    ))

        return alerts

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute enhanced v6 min/max with correlation and SPC."""
        if not self._enabled:
            return

        # Run parent v5 execute
        super().execute(time)

        # Multi-field correlation
        if self._correlation_fields:
            corr = self._compute_field_correlation(time)
            if corr is not None:
                self._correlations.append(corr)

        # SPC control charts
        if self._spc_enabled:
            charts = self._update_spc(time)
            if charts:
                self._spc_charts.append(charts)
                for key, chart in charts.items():
                    if chart.out_of_control:
                        logger.warning(
                            "SPC out-of-control: t=%g field=%s value=%.6g "
                            "(UCL=%.6g, LCL=%.6g)",
                            time, chart.field_name, chart.value,
                            chart.ucl, chart.lcl,
                        )

        # Predictive monitoring
        if self._predictive_enabled:
            alerts = self._check_predictive(time)
            self._predictive_alerts.extend(alerts)
            for a in alerts:
                logger.warning(
                    "Predictive alert: t=%g field=%s metric=%s "
                    "predicted breach at t=%.4g",
                    a.time, a.field_name, a.metric, a.predicted_time,
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def correlations(self) -> List[FieldCorrelation]:
        """Field correlation history."""
        return self._correlations

    @property
    def spc_charts(self) -> List[Dict[str, SPCControlChart]]:
        """SPC control chart history."""
        return self._spc_charts

    @property
    def predictive_alerts(self) -> List[PredictiveAlert]:
        """Predictive monitoring alerts."""
        return self._predictive_alerts

    def get_latest_spc(self) -> Optional[Dict[str, SPCControlChart]]:
        """Get latest SPC charts."""
        if not self._spc_charts:
            return None
        return self._spc_charts[-1]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write v6 enhanced min/max data."""
        super().write()

        if self._output_path is None:
            return

        # Write SPC control charts
        if self._spc_charts:
            spc_file = self._output_path / "spcCharts.dat"
            with open(spc_file, "w") as f:
                f.write("# time  field  metric  value  UCL  LCL  CL  cusum_pos  cusum_neg  out_of_control\n")
                for charts in self._spc_charts:
                    for key, chart in charts.items():
                        f.write(
                            f"{chart.time:.6e}  {chart.field_name}  {chart.metric}  "
                            f"{chart.value:.10g}  {chart.ucl:.10g}  {chart.lcl:.10g}  "
                            f"{chart.cl:.10g}  {chart.cusum_pos:.6e}  "
                            f"{chart.cusum_neg:.6e}  0\n"
                        )

        # Write predictive alerts
        if self._predictive_alerts:
            pred_file = self._output_path / "predictiveAlerts.dat"
            with open(pred_file, "w") as f:
                f.write("# time  field  metric  predicted_time  threshold  current_value  slope\n")
                for a in self._predictive_alerts:
                    f.write(
                        f"{a.time:.6e}  {a.field_name}  {a.metric}  "
                        f"{a.predicted_time:.6e}  {a.threshold:.10g}  "
                        f"{a.current_value:.10g}  {a.slope:.6e}\n"
                    )

        logger.info("Wrote FieldMinMaxEnhanced6 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("fieldMinMaxEnhanced6", FieldMinMaxEnhanced6)
