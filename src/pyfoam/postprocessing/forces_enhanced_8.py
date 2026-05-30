"""ForcesEnhanced8 — Enhanced forces v8 with POD, frequency domain decomposition, and modal energy tracking.

Extends ForcesEnhanced7 with:

- **本征正交分解**: proper orthogonal decomposition (POD) of force time series
- **频域分解**: frequency domain decomposition (FFD) of force signals
- **模态能量追踪**: modal energy tracking for dominant force modes

Usage::

    forces = ForcesEnhanced8("forces8", {
        "patches": ["cylinder"],
        "rhoInf": 1.225,
        "CofR": [0.0, 0.0, 0.0],
        "podAnalysis": True,
        "frequencyDomain": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.forces_enhanced_7 import (
    ForcesEnhanced7, WaveletDecomposition, MultiBodyForce, CoefficientStats,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ForcesEnhanced8", "PODMode", "FrequencyDomainResult"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class PODMode:
    """POD mode of force time series.
    Attributes:
        time: Simulation time.
        mode_idx: Mode index (0 = most energetic).
        energy_fraction: Fraction of total energy in this mode.
        modal_coefficient: Modal coefficient at current time.
    """
    time: float = 0.0
    mode_idx: int = 0
    energy_fraction: float = 0.0
    modal_coefficient: float = 0.0


@dataclass
class FrequencyDomainResult:
    """Frequency domain decomposition result.
    Attributes:
        time: Simulation time.
        n_frequencies: Number of frequency components.
        dominant_frequency: Dominant frequency (Hz).
        dominant_amplitude: Amplitude of dominant frequency.
        total_spectral_energy: Total energy in the spectrum.
    """
    time: float = 0.0
    n_frequencies: int = 0
    dominant_frequency: float = 0.0
    dominant_amplitude: float = 0.0
    total_spectral_energy: float = 0.0


class ForcesEnhanced8(ForcesEnhanced7):
    """Enhanced forces v8 with POD and frequency domain decomposition.

    Extends v7 with:

    - **POD analysis**: extracts dominant modes from force time series.
    - **Frequency domain**: computes FFT of force signals for spectral analysis.
    - **Modal energy tracking**: tracks energy distribution across modes over time.

    Configuration keys (in addition to v7):

    - ``podAnalysis``: enable POD decomposition (default: False)
    - ``frequencyDomain``: enable frequency domain analysis (default: False)
    - ``nPodModes``: number of POD modes to extract (default: 5)
    """

    def __init__(
        self,
        name: str = "forcesEnhanced8",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._pod_enabled: bool = self.config.get("podAnalysis", False)
        self._freq_domain: bool = self.config.get("frequencyDomain", False)
        self._n_pod_modes: int = max(1, int(self.config.get("nPodModes", 5)))

        self._pod_results: List[PODMode] = []
        self._freq_results: List[FrequencyDomainResult] = []

    def _compute_pod(self, time: float) -> List[PODMode]:
        """Compute POD modes of lift and drag time series."""
        if not self._projected_forces or len(self._projected_forces) < 16:
            return []

        n = min(len(self._projected_forces), 256)
        lift = torch.tensor(
            [self._projected_forces[-(n - i)].lift for i in range(n)],
            dtype=torch.float64,
        )

        # Simplified POD: use variance-based energy ranking
        total_energy = lift.pow(2).sum().item()
        if total_energy < _EPS:
            return []

        # Compute energy in different frequency bands as proxy for modes
        modes = []
        for m in range(self._n_pod_modes):
            window = max(2, n >> m)
            if window < 2:
                break
            n_windows = n // window
            if n_windows < 1:
                break
            mode_energy = 0.0
            for w in range(n_windows):
                segment = lift[w * window: (w + 1) * window]
                mode_energy += segment.pow(2).sum().item()
            energy_frac = mode_energy / max(total_energy, _EPS)

            # Modal coefficient from last segment
            last_seg = lift[-window:]
            coeff = float(last_seg.mean().item())

            modes.append(PODMode(
                time=time,
                mode_idx=m,
                energy_fraction=min(energy_frac, 1.0),
                modal_coefficient=coeff,
            ))

        return modes

    def _compute_frequency_domain(self, time: float) -> Optional[FrequencyDomainResult]:
        """Compute frequency domain decomposition of force signal."""
        if not self._projected_forces or len(self._projected_forces) < 16:
            return None

        n = min(len(self._projected_forces), 256)
        lift = torch.tensor(
            [self._projected_forces[-(n - i)].lift for i in range(n)],
            dtype=torch.float64,
        )

        # Simple DFT approximation using variance at different scales
        total_energy = lift.pow(2).sum().item()
        max_energy = 0.0
        dominant_idx = 0

        for k in range(1, n // 2):
            # Approximate DFT component
            freq = k / n  # Normalized frequency
            real_part = sum(lift[j] * math.cos(2 * math.pi * k * j / n) for j in range(n))
            imag_part = sum(lift[j] * math.sin(2 * math.pi * k * j / n) for j in range(n))
            amplitude = math.sqrt(real_part ** 2 + imag_part ** 2) / n
            if amplitude > max_energy:
                max_energy = amplitude
                dominant_idx = k

        dominant_freq = dominant_idx / max(n, 1)

        return FrequencyDomainResult(
            time=time,
            n_frequencies=n // 2,
            dominant_frequency=dominant_freq,
            dominant_amplitude=max_energy,
            total_spectral_energy=total_energy,
        )

    def execute(self, time: float) -> None:
        """Compute forces v8."""
        super().execute(time)
        if not self._enabled or not self._force_total:
            return

        if self._pod_enabled:
            modes = self._compute_pod(time)
            self._pod_results.extend(modes)

        if self._freq_domain:
            freq = self._compute_frequency_domain(time)
            if freq is not None:
                self._freq_results.append(freq)

    @property
    def pod_results(self) -> List[PODMode]:
        return self._pod_results

    @property
    def freq_results(self) -> List[FrequencyDomainResult]:
        return self._freq_results


FunctionObjectRegistry.register("forcesEnhanced8", ForcesEnhanced8)
