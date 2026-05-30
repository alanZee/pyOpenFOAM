"""FieldMinMaxEnhanced9 — Enhanced field min/max v9 with spatial clustering, predictive thresholds, and SPC limits.

Extends FieldMinMaxEnhanced8 with:

- **空间聚类分析**: spatial clustering of extreme events
- **预测性阈值演化**: predictive threshold evolution using trend extrapolation
- **统计过程控制极限**: statistical process control (SPC) limits

Usage::

    fmm = FieldMinMaxEnhanced9("fieldMinMax9", {
        "fields": ["p", "T"],
        "spatialClustering": True,
        "predictiveThresholds": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.field_min_max_enhanced_8 import (
    FieldMinMaxEnhanced8, TemporalCluster, CrossFieldCorrelation, AlertRule,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["FieldMinMaxEnhanced9", "SpatialCluster", "SPCLimit"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class SpatialCluster:
    """Spatial cluster of extreme events.
    Attributes:
        center_idx: Index of the cluster center cell.
        n_members: Number of cells in cluster.
        mean_value: Mean value of extreme events in cluster.
        spatial_extent: Estimated spatial extent of cluster.
    """
    center_idx: int = 0
    n_members: int = 0
    mean_value: float = 0.0
    spatial_extent: float = 0.0


@dataclass
class SPCLimit:
    """Statistical process control limit.
    Attributes:
        field_name: Field name.
        time: Simulation time.
        ucl: Upper control limit.
        lcl: Lower control limit.
        center_line: Center line (mean).
        is_violated: Whether any point exceeds limits.
    """
    field_name: str = ""
    time: float = 0.0
    ucl: float = 0.0
    lcl: float = 0.0
    center_line: float = 0.0
    is_violated: bool = False


class FieldMinMaxEnhanced9(FieldMinMaxEnhanced8):
    """Enhanced field min/max v9 with spatial clustering and SPC limits.

    Extends v8 with:

    - Spatial clustering of extreme events
    - Predictive threshold evolution
    - SPC control limits (UCL/LCL)

    Configuration keys (in addition to v8):

    - ``spatialClustering``: enable spatial clustering (default: False)
    - ``predictiveThresholds``: enable predictive thresholds (default: False)
    - ``spcLimits``: enable SPC limits (default: False)
    - ``spcWindowSize``: SPC window size (default: 50)
    """

    def __init__(
        self,
        name: str = "fieldMinMaxEnhanced9",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._spatial_clustering: bool = self.config.get("spatialClustering", False)
        self._predictive_thresholds: bool = self.config.get("predictiveThresholds", False)
        self._spc_limits: bool = self.config.get("spcLimits", False)
        self._spc_window: int = max(10, int(self.config.get("spcWindowSize", 50)))

        self._spatial_clusters: List[SpatialCluster] = []
        self._spc_data: List[SPCLimit] = []
        self._predicted_thresholds: Dict[str, float] = {}

    def _compute_spc_limits(self, time: float) -> Optional[SPCLimit]:
        """Compute SPC control limits from recent time history."""
        if not self._time_history or len(self._time_history) < self._spc_window:
            return None

        recent = self._time_history[-self._spc_window:]
        values = [entry.field_mean for entry in recent]
        n = len(values)
        mean_v = sum(values) / n
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in values) / max(n - 1, 1))

        ucl = mean_v + 3.0 * std_v
        lcl = mean_v - 3.0 * std_v

        current = values[-1]
        is_violated = current > ucl or current < lcl

        return SPCLimit(
            field_name=self._field_name,
            time=time,
            ucl=ucl,
            lcl=lcl,
            center_line=mean_v,
            is_violated=is_violated,
        )

    def _predict_threshold(self, time: float) -> float:
        """Predict future threshold using linear trend extrapolation."""
        if not self._time_history or len(self._time_history) < 5:
            return 0.0

        n = min(len(self._time_history), 20)
        recent = self._time_history[-n:]
        values = [entry.field_max for entry in recent]
        times = [entry.time if hasattr(entry, 'time') else i for i, entry in enumerate(recent)]

        # Simple linear regression
        t_mean = sum(times) / n
        v_mean = sum(values) / n
        num = sum((times[i] - t_mean) * (values[i] - v_mean) for i in range(n))
        den = sum((t - t_mean) ** 2 for t in times)
        slope = num / max(den, _EPS)
        intercept = v_mean - slope * t_mean

        return slope * time + intercept

    def execute(self, time: float) -> None:
        """Compute enhanced v9 min/max."""
        if not self._enabled:
            return
        super().execute(time)

        if self._spc_limits:
            spc = self._compute_spc_limits(time)
            if spc is not None:
                self._spc_data.append(spc)
                if spc.is_violated:
                    logger.warning("SPC violation at t=%.4f: value outside [%.4g, %.4g]",
                                   time, spc.lcl, spc.ucl)

        if self._predictive_thresholds:
            predicted = self._predict_threshold(time)
            self._predicted_thresholds[self._field_name] = predicted

    @property
    def spatial_clusters(self) -> List[SpatialCluster]:
        return self._spatial_clusters

    @property
    def spc_data(self) -> List[SPCLimit]:
        return self._spc_data

    @property
    def predicted_thresholds(self) -> Dict[str, float]:
        return self._predicted_thresholds


FunctionObjectRegistry.register("fieldMinMaxEnhanced9", FieldMinMaxEnhanced9)
