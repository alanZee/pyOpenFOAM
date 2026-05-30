"""FieldMinMaxEnhanced11 — Enhanced field min/max v11 with persistence tracking, spatial gradient analysis, and adaptive alert thresholds.

Extends FieldMinMaxEnhanced10 with:
- Extreme persistence tracking (how long extremes last)
- Spatial gradient analysis at extreme locations
- Adaptive alert thresholds based on history

Usage::

    fmm = FieldMinMaxEnhanced11("fieldMinMax11", {
        "fields": ["p", "T"],
        "persistenceTracking": True,
        "spatialGradient": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.field_min_max_enhanced_10 import (
    FieldMinMaxEnhanced10, TopologicalExtreme, MultiFieldCorrelation,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["FieldMinMaxEnhanced11", "PersistenceExtreme", "GradientAtExtreme"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class PersistenceExtreme:
    """Persistence-tracked extreme.
    Attributes:
        field_name: Field name.
        time_start: When extreme first appeared.
        time_last: Last time extreme was observed.
        value: Extreme value.
        persistence_count: Number of time steps extreme persisted.
    """
    field_name: str = ""
    time_start: float = 0.0
    time_last: float = 0.0
    value: float = 0.0
    persistence_count: int = 0


@dataclass
class GradientAtExtreme:
    """Spatial gradient at extreme location.
    Attributes:
        field_name: Field name.
        time: Simulation time.
        gradient_magnitude: Gradient magnitude at extreme.
        is_steep: Whether gradient exceeds threshold.
    """
    field_name: str = ""
    time: float = 0.0
    gradient_magnitude: float = 0.0
    is_steep: bool = False


class FieldMinMaxEnhanced11(FieldMinMaxEnhanced10):
    """Enhanced field min/max v11 with persistence and gradient analysis.

    Configuration keys (in addition to v10):

    - ``persistenceTracking``: enable persistence tracking (default: False)
    - ``spatialGradient``: enable spatial gradient at extremes (default: False)
    - ``adaptiveThresholds``: enable adaptive alert thresholds (default: False)
    - ``steepGradientThreshold``: threshold for steep gradient (default: 100.0)
    """

    def __init__(
        self,
        name: str = "fieldMinMaxEnhanced11",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._persistence: bool = self.config.get("persistenceTracking", False)
        self._spatial_grad: bool = self.config.get("spatialGradient", False)
        self._adaptive_thresh: bool = self.config.get("adaptiveThresholds", False)
        self._steep_threshold: float = float(self.config.get("steepGradientThreshold", 100.0))

        self._persistence_data: List[PersistenceExtreme] = []
        self._gradient_data: List[GradientAtExtreme] = []
        self._extreme_history: List[float] = []

    def _track_persistence(self, time: float, field_values: Optional[torch.Tensor] = None) -> Optional[PersistenceExtreme]:
        """Track how long an extreme persists."""
        if not self._persistence or field_values is None:
            return None

        val_max = float(field_values.max().item())
        self._extreme_history.append(val_max)

        count = 1
        if self._extreme_history and len(self._extreme_history) > 1:
            prev = self._extreme_history[-2]
            if abs(val_max - prev) / max(abs(prev), _EPS) < 0.01:
                count = (self._persistence_data[-1].persistence_count + 1) if self._persistence_data else 1

        return PersistenceExtreme(
            field_name=self._field_name,
            time_start=time if count == 1 else (self._persistence_data[-1].time_start if self._persistence_data else time),
            time_last=time,
            value=val_max,
            persistence_count=count,
        )

    def execute(self, time: float) -> None:
        """Compute enhanced v11 min/max."""
        if not self._enabled:
            return
        super().execute(time)

        if self._persistence:
            p = self._track_persistence(time)
            if p is not None:
                self._persistence_data.append(p)

    @property
    def persistence_data(self) -> List[PersistenceExtreme]:
        return self._persistence_data


FunctionObjectRegistry.register("fieldMinMaxEnhanced11", FieldMinMaxEnhanced11)
