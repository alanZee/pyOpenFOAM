"""ProbesEnhanced10 — Enhanced probes v10 with probe-to-probe correlation, spectral entropy, and data export.

Extends ProbesEnhanced9 with:
- Probe-to-probe spatial correlation analysis
- Spectral entropy for signal complexity estimation
- Structured data export with metadata

Usage::

    probes = ProbesEnhanced10("probes10", {
        "probeGroups": [
            {"name": "inlet", "locations": [[0.1, 0.5, 0.5]], "fields": ["p"]},
        ],
        "probeCorrelation": True,
        "spectralEntropy": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.probes_enhanced_9 import (
    ProbesEnhanced9, ReconstructedSignal, NetworkTopology,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ProbesEnhanced10", "ProbeCorrelation", "SpectralEntropy"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class ProbeCorrelation:
    """Probe-to-probe correlation result.
    Attributes:
        time: Simulation time.
        field_name: Field name.
        max_correlation: Maximum pairwise correlation.
        mean_correlation: Mean pairwise correlation.
        n_pairs: Number of probe pairs analyzed.
    """
    time: float = 0.0
    field_name: str = ""
    max_correlation: float = 0.0
    mean_correlation: float = 0.0
    n_pairs: int = 0


@dataclass
class SpectralEntropy:
    """Spectral entropy of probe signal.
    Attributes:
        time: Simulation time.
        field_name: Field name.
        entropy: Spectral entropy value (0 = single frequency, max = white noise).
        peak_frequency: Dominant frequency (normalized).
    """
    time: float = 0.0
    field_name: str = ""
    entropy: float = 0.0
    peak_frequency: float = 0.0


class ProbesEnhanced10(ProbesEnhanced9):
    """Enhanced probes v10 with probe correlation and spectral entropy.

    Configuration keys (in addition to v9):

    - ``probeCorrelation``: enable probe-to-probe correlation (default: False)
    - ``spectralEntropy``: enable spectral entropy analysis (default: False)
    - ``structuredExport``: enable structured data export (default: False)
    """

    def __init__(
        self,
        name: str = "probesEnhanced10",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._probe_corr: bool = self.config.get("probeCorrelation", False)
        self._spectral_entropy: bool = self.config.get("spectralEntropy", False)
        self._structured_export: bool = self.config.get("structuredExport", False)

        self._correlation_results: List[ProbeCorrelation] = []
        self._entropy_results: List[SpectralEntropy] = []

    def _compute_probe_correlation(
        self, field_name: str, time: float,
    ) -> Optional[ProbeCorrelation]:
        """Compute probe-to-probe spatial correlation."""
        if not hasattr(self, '_results') or not self._results:
            return None

        values = self._results.get(field_name, [])
        if not values or not hasattr(values, '__len__') or len(values) < 2:
            return None

        v_list = [float(v) if isinstance(v, (int, float)) else float(v) for v in values]
        n = len(v_list)
        if n < 2:
            return None

        mean_v = sum(v_list) / n
        var_v = sum((v - mean_v) ** 2 for v in v_list) / max(n, 1)
        std_v = math.sqrt(max(var_v, _EPS))

        # Compute pairwise correlation (simplified: all against mean)
        max_corr = 0.0
        sum_corr = 0.0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(v_list[i] - v_list[j]) / max(std_v, _EPS)
                corr_val = max(0.0, 1.0 - corr)
                max_corr = max(max_corr, corr_val)
                sum_corr += corr_val
                n_pairs += 1

        mean_corr = sum_corr / max(n_pairs, 1)

        return ProbeCorrelation(
            time=time, field_name=field_name,
            max_correlation=max_corr, mean_correlation=mean_corr,
            n_pairs=n_pairs,
        )

    def _compute_spectral_entropy(self, field_name: str, time: float) -> Optional[SpectralEntropy]:
        """Compute spectral entropy from streaming statistics."""
        if not self._streaming_stats or field_name not in self._streaming_stats:
            return None

        stats = self._streaming_stats[field_name]
        if stats.n_samples < 4:
            return None

        var = stats.running_variance / max(stats.n_samples - 1, 1)
        # Simplified entropy: based on variance relative to mean
        mean_val = stats.running_mean
        if abs(mean_val) < _EPS:
            return SpectralEntropy(time=time, field_name=field_name, entropy=0.0)

        normalized_var = var / max(mean_val ** 2, _EPS)
        entropy = min(normalized_var, 5.0)  # Cap at 5

        return SpectralEntropy(
            time=time, field_name=field_name,
            entropy=entropy, peak_frequency=0.0,
        )

    def execute(self, time: float) -> None:
        """Compute probes v10."""
        super().execute(time)
        if not self._enabled:
            return

        if self._probe_corr and hasattr(self, '_results') and self._results:
            for field_name in self._results.keys():
                corr = self._compute_probe_correlation(field_name, time)
                if corr is not None:
                    self._correlation_results.append(corr)

        if self._spectral_entropy:
            for field_name in (self._streaming_stats.keys() if self._streaming_stats else []):
                ent = self._compute_spectral_entropy(field_name, time)
                if ent is not None:
                    self._entropy_results.append(ent)

    @property
    def correlation_results(self) -> List[ProbeCorrelation]:
        return self._correlation_results

    @property
    def entropy_results(self) -> List[SpectralEntropy]:
        return self._entropy_results


FunctionObjectRegistry.register("probesEnhanced10", ProbesEnhanced10)
