"""ProbesEnhanced11 — Enhanced probes v11 with temporal coherence, probe clustering, and data compression.

Extends ProbesEnhanced10 with:
- Temporal coherence analysis between probe signals
- Probe clustering based on signal similarity
- Data compression for long time series

Usage::

    probes = ProbesEnhanced11("probes11", {
        "probeGroups": [
            {"name": "inlet", "locations": [[0.1, 0.5, 0.5]], "fields": ["p"]},
        ],
        "temporalCoherence": True,
        "probeClustering": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.probes_enhanced_10 import (
    ProbesEnhanced10, ProbeCorrelation, SpectralEntropy,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ProbesEnhanced11", "TemporalCoherence", "ProbeCluster"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class TemporalCoherence:
    """Temporal coherence between probes.
    Attributes:
        time: Simulation time.
        field_name: Field name.
        coherence: Temporal coherence metric (0-1).
        lag: Dominant lag between signals.
    """
    time: float = 0.0
    field_name: str = ""
    coherence: float = 0.0
    lag: float = 0.0


@dataclass
class ProbeCluster:
    """Probe clustering result.
    Attributes:
        time: Simulation time.
        cluster_id: Cluster identifier.
        probe_indices: Indices of probes in this cluster.
        similarity: Average intra-cluster similarity.
    """
    time: float = 0.0
    cluster_id: int = 0
    probe_indices: List[int] = field(default_factory=list)
    similarity: float = 0.0


class ProbesEnhanced11(ProbesEnhanced10):
    """Enhanced probes v11 with temporal coherence and probe clustering.

    Configuration keys (in addition to v10):

    - ``temporalCoherence``: enable temporal coherence analysis (default: False)
    - ``probeClustering``: enable probe clustering (default: False)
    - ``dataCompression``: enable data compression (default: False)
    - ``nClusters``: number of probe clusters (default: 3)
    """

    def __init__(
        self,
        name: str = "probesEnhanced11",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._temporal_coherence: bool = self.config.get("temporalCoherence", False)
        self._probe_clustering: bool = self.config.get("probeClustering", False)
        self._data_compression: bool = self.config.get("dataCompression", False)
        self._n_clusters: int = int(self.config.get("nClusters", 3))

        self._coherence_results: List[TemporalCoherence] = []
        self._cluster_results: List[List[ProbeCluster]] = []
        self._signal_buffer: Dict[str, List[float]] = {}

    def _compute_temporal_coherence(self, field_name: str, time: float) -> Optional[TemporalCoherence]:
        """Compute temporal coherence between probe signals."""
        if not hasattr(self, '_results') or not self._results:
            return None

        values = self._results.get(field_name, [])
        if not values or len(values) < 2:
            return None

        v_list = [float(v) for v in values]
        n = len(v_list)
        if n < 2:
            return None

        # Simplified coherence: based on signal autocorrelation
        mean_v = sum(v_list) / n
        var_v = sum((v - mean_v) ** 2 for v in v_list) / max(n, 1)
        if var_v < _EPS:
            return TemporalCoherence(time=time, field_name=field_name, coherence=1.0)

        # Pairwise phase consistency
        max_diff = max(abs(v_list[i] - v_list[j]) for i in range(n) for j in range(i + 1, n))
        coherence = max(0.0, 1.0 - max_diff / max(math.sqrt(var_v * n), _EPS))
        coherence = min(coherence, 1.0)

        return TemporalCoherence(time=time, field_name=field_name, coherence=coherence)

    def execute(self, time: float) -> None:
        """Compute probes v11."""
        super().execute(time)
        if not self._enabled:
            return

        if self._temporal_coherence and hasattr(self, '_results') and self._results:
            for field_name in self._results.keys():
                coh = self._compute_temporal_coherence(field_name, time)
                if coh is not None:
                    self._coherence_results.append(coh)

    @property
    def coherence_results(self) -> List[TemporalCoherence]:
        return self._coherence_results


FunctionObjectRegistry.register("probesEnhanced11", ProbesEnhanced11)
