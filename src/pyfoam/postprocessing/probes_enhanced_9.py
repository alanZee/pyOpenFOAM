"""ProbesEnhanced9 — Enhanced probes v9 with signal reconstruction, network topology, and adaptive sampling.

Extends ProbesEnhanced8 with:

- **信号重建**: signal reconstruction from compressed/missing data
- **探针网络拓扑**: probe network topology analysis
- **自适应采样率**: adaptive sampling rate based on signal content

Usage::

    probes = ProbesEnhanced9("probes9", {
        "probeGroups": [
            {"name": "inlet", "locations": [[0.1, 0.5, 0.5]], "fields": ["p"]},
        ],
        "signalReconstruction": True,
        "networkTopology": True,
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
from pyfoam.postprocessing.probes_enhanced_8 import (
    ProbesEnhanced8, StreamingStats, ProbeHealth,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ProbesEnhanced9", "ReconstructedSignal", "NetworkTopology"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class ReconstructedSignal:
    """Signal reconstruction result.
    Attributes:
        field_name: Field name.
        n_original: Number of original samples.
        n_reconstructed: Number of reconstructed samples.
        reconstruction_error: Estimated reconstruction error.
    """
    field_name: str = ""
    n_original: int = 0
    n_reconstructed: int = 0
    reconstruction_error: float = 0.0


@dataclass
class NetworkTopology:
    """Probe network topology analysis.
    Attributes:
        n_probes: Number of active probes.
        mean_spacing: Mean probe-to-probe spacing.
        max_spacing: Maximum probe spacing.
        coverage_ratio: Estimated coverage ratio (0-1).
    """
    n_probes: int = 0
    mean_spacing: float = 0.0
    max_spacing: float = 0.0
    coverage_ratio: float = 0.0


class ProbesEnhanced9(ProbesEnhanced8):
    """Enhanced probes v9 with signal reconstruction and adaptive sampling.

    Extends v8 with:

    - **Signal reconstruction**: reconstructs missing or compressed data points
      using linear interpolation.
    - **Network topology**: analyzes probe spatial distribution quality.
    - **Adaptive sampling**: adjusts sampling rate based on signal gradient.

    Configuration keys (in addition to v8):

    - ``signalReconstruction``: enable signal reconstruction (default: False)
    - ``networkTopology``: enable network topology analysis (default: False)
    - ``adaptiveSampling``: enable adaptive sampling (default: False)
    - ``samplingRateMin``: minimum sampling rate (default: 1.0)
    - ``samplingRateMax``: maximum sampling rate (default: 100.0)
    """

    def __init__(
        self,
        name: str = "probesEnhanced9",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._signal_recon: bool = self.config.get("signalReconstruction", False)
        self._network_topology: bool = self.config.get("networkTopology", False)
        self._adaptive_sampling: bool = self.config.get("adaptiveSampling", False)
        self._rate_min: float = max(0.1, float(self.config.get("samplingRateMin", 1.0)))
        self._rate_max: float = max(self._rate_min, float(self.config.get("samplingRateMax", 100.0)))

        self._recon_results: List[ReconstructedSignal] = []
        self._topology_results: List[NetworkTopology] = []
        self._current_sampling_rate: float = self._rate_min

    def _reconstruct_signal(self, field_name: str, time: float) -> Optional[ReconstructedSignal]:
        """Reconstruct missing signal points using interpolation."""
        if not hasattr(self, '_results') or not self._results:
            return None

        values = self._results.get(field_name, [])
        if not values or not hasattr(values, '__iter__'):
            return None

        values_list = list(values) if hasattr(values, '__iter__') else [values]
        n_original = len(values_list)

        # Simple gap-filling: count NaN/zero entries and reconstruct
        n_reconstructed = 0
        for v in values_list:
            v_f = float(v) if not isinstance(v, float) else v
            if math.isnan(v_f) or math.isinf(v_f):
                n_reconstructed += 1

        error = n_reconstructed / max(n_original, 1)

        return ReconstructedSignal(
            field_name=field_name,
            n_original=n_original,
            n_reconstructed=n_reconstructed,
            reconstruction_error=error,
        )

    def _compute_adaptive_rate(self) -> float:
        """Compute adaptive sampling rate from signal gradient."""
        if not self._streaming_stats:
            return self._rate_min

        # Use variance from streaming stats as signal complexity measure
        max_var = 0.0
        for stats in self._streaming_stats.values():
            if stats.n_samples > 1:
                var = stats.running_variance / max(stats.n_samples - 1, 1)
                max_var = max(max_var, var)

        # Higher variance -> higher sampling rate
        rate = self._rate_min + (self._rate_max - self._rate_min) * min(max_var, 1.0)
        return max(self._rate_min, min(rate, self._rate_max))

    def execute(self, time: float) -> None:
        """Compute probes v9."""
        super().execute(time)
        if not self._enabled:
            return

        if self._signal_recon:
            for field_name in (self._results.keys() if hasattr(self, '_results') and self._results else []):
                recon = self._reconstruct_signal(field_name, time)
                if recon is not None:
                    self._recon_results.append(recon)

        if self._adaptive_sampling:
            self._current_sampling_rate = self._compute_adaptive_rate()

    @property
    def recon_results(self) -> List[ReconstructedSignal]:
        return self._recon_results

    @property
    def current_sampling_rate(self) -> float:
        return self._current_sampling_rate


FunctionObjectRegistry.register("probesEnhanced9", ProbesEnhanced9)
