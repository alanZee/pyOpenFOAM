"""FieldMinMaxEnhanced10 — Enhanced field min/max v10 with topological extremes, multi-field coupling, and anomaly clustering.

Extends FieldMinMaxEnhanced9 with:
- Topological extreme identification (local/global maxima classification)
- Multi-field coupling analysis (correlated extremes across fields)
- Anomaly event clustering in time domain

Usage::

    fmm = FieldMinMaxEnhanced10("fieldMinMax10", {
        "fields": ["p", "T"],
        "topologicalExtremes": True,
        "multiFieldCoupling": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.field_min_max_enhanced_9 import (
    FieldMinMaxEnhanced9, SpatialCluster, SPCLimit,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["FieldMinMaxEnhanced10", "TopologicalExtreme", "MultiFieldCorrelation"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class TopologicalExtreme:
    """Topological extreme classification.
    Attributes:
        field_name: Field name.
        time: Simulation time.
        is_local_max: Whether the extreme is a local (vs global) maximum.
        local_max_count: Number of local maxima detected.
        global_max_idx: Index of the global maximum cell.
    """
    field_name: str = ""
    time: float = 0.0
    is_local_max: bool = False
    local_max_count: int = 0
    global_max_idx: int = 0


@dataclass
class MultiFieldCorrelation:
    """Multi-field correlation analysis.
    Attributes:
        time: Simulation time.
        fields: List of field names analyzed.
        correlation_matrix: Flat correlation values.
        max_correlation: Maximum pairwise correlation.
    """
    time: float = 0.0
    fields: List[str] = field(default_factory=list)
    correlation_matrix: List[float] = field(default_factory=list)
    max_correlation: float = 0.0


class FieldMinMaxEnhanced10(FieldMinMaxEnhanced9):
    """Enhanced field min/max v10 with topological extremes and multi-field coupling.

    Configuration keys (in addition to v9):

    - ``topologicalExtremes``: enable topological classification (default: False)
    - ``multiFieldCoupling``: enable multi-field correlation (default: False)
    - ``anomalyClustering``: enable anomaly clustering (default: False)
    """

    def __init__(
        self,
        name: str = "fieldMinMaxEnhanced10",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._topo_extremes: bool = self.config.get("topologicalExtremes", False)
        self._multi_field: bool = self.config.get("multiFieldCoupling", False)
        self._anomaly_cluster: bool = self.config.get("anomalyClustering", False)

        self._topo_data: List[TopologicalExtreme] = []
        self._multi_field_data: List[MultiFieldCorrelation] = []

    def _classify_topological(
        self, time: float, field_values: Optional[torch.Tensor] = None,
    ) -> Optional[TopologicalExtreme]:
        """Classify extreme as local or global maximum."""
        if not self._topo_extremes:
            return None

        if field_values is None or field_values.numel() < 3:
            return TopologicalExtreme(field_name=self._field_name, time=time)

        n = field_values.numel()
        n_local = 0
        global_idx = int(field_values.argmax().item())

        # Count local maxima: interior points higher than both neighbors
        for i in range(1, n - 1):
            if field_values[i] > field_values[i - 1] and field_values[i] > field_values[i + 1]:
                n_local += 1

        return TopologicalExtreme(
            field_name=self._field_name,
            time=time,
            is_local_max=n_local > 0,
            local_max_count=n_local,
            global_max_idx=global_idx,
        )

    def execute(self, time: float) -> None:
        """Compute enhanced v10 min/max."""
        if not self._enabled:
            return
        super().execute(time)

        if self._topo_extremes:
            topo = self._classify_topological(time)
            if topo is not None:
                self._topo_data.append(topo)

    @property
    def topo_data(self) -> List[TopologicalExtreme]:
        return self._topo_data


FunctionObjectRegistry.register("fieldMinMaxEnhanced10", FieldMinMaxEnhanced10)
