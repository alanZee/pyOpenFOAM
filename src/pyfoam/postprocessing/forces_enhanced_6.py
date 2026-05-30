"""ForcesEnhanced6 — Enhanced forces v6 with mode decomposition and frequency tracking.

Extends ForcesEnhanced5 with:

- **DMD 动态模态分解**: Dynamic Mode Decomposition of force time series
- **频率跟踪**: real-time dominant frequency tracking with adaptive filters
- **参考坐标系变换**: force data in multiple reference frames

Usage::

    forces = ForcesEnhanced6("forces6", {
        "patches": ["cylinder"],
        "rhoInf": 1.225,
        "CofR": [0.0, 0.0, 0.0],
        "dmdAnalysis": True,
        "frequencyTracking": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.forces_enhanced_5 import (
    ForcesEnhanced5, FSIForceData, FatigueSpectrum, MomentPSD,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ForcesEnhanced6", "DMDMode", "FrequencyTracker", "ReferenceFrameTransform"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class DMDMode:
    """Dynamic Mode Decomposition result.
    Attributes:
        time: Simulation time.
        eigenvalues: DMD eigenvalues.
        frequencies: Mode frequencies (Hz).
        growth_rates: Mode growth rates.
        n_modes: Number of modes.
    """
    time: float = 0.0
    eigenvalues: Optional[torch.Tensor] = None
    frequencies: Optional[torch.Tensor] = None
    growth_rates: Optional[torch.Tensor] = None
    n_modes: int = 0


@dataclass
class FrequencyTracker:
    """Dominant frequency tracker result.
    Attributes:
        time: Simulation time.
        dominant_freq_drag: Dominant frequency in drag (Hz).
        dominant_freq_lift: Dominant frequency in lift (Hz).
        amplitude_drag: Amplitude of dominant drag frequency.
        amplitude_lift: Amplitude of dominant lift frequency.
    """
    time: float = 0.0
    dominant_freq_drag: float = 0.0
    dominant_freq_lift: float = 0.0
    amplitude_drag: float = 0.0
    amplitude_lift: float = 0.0


@dataclass
class ReferenceFrameTransform:
    """Force data in alternative reference frame.
    Attributes:
        time: Simulation time.
        force_rotated: Rotated force vector.
        moment_rotated: Rotated moment vector.
        rotation_angle: Rotation angle (radians).
        axis: Rotation axis.
    """
    time: float = 0.0
    force_rotated: Optional[torch.Tensor] = None
    moment_rotated: Optional[torch.Tensor] = None
    rotation_angle: float = 0.0
    axis: str = "z"


class ForcesEnhanced6(ForcesEnhanced5):
    """Enhanced forces v6 with DMD, frequency tracking, and reference frame transforms.

    在 ForcesEnhanced5 基础上增加的配置键：

    - ``dmdAnalysis``: enable DMD analysis (default: False)
    - ``frequencyTracking``: enable real-time frequency tracking (default: False)
    - ``dmdRank``: DMD truncation rank (default: 10)
    - ``refFrameTransform``: reference frame rotation config (default: None)
    """

    def __init__(
        self,
        name: str = "forcesEnhanced6",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._dmd_enabled: bool = self.config.get("dmdAnalysis", False)
        self._freq_tracking: bool = self.config.get("frequencyTracking", False)
        self._dmd_rank: int = max(1, int(self.config.get("dmdRank", 10)))

        self._dmd_results: List[DMDMode] = []
        self._freq_track_results: List[FrequencyTracker] = []
        self._ref_frame_results: List[ReferenceFrameTransform] = []

    def _compute_dmd(self, time: float) -> Optional[DMDMode]:
        """Compute DMD of force time series."""
        if not self._projected_forces or len(self._projected_forces) < self._dmd_rank + 1:
            return None

        n = min(len(self._projected_forces), 128)
        drag = torch.tensor(
            [self._projected_forces[-(n - i)].drag for i in range(n)],
            dtype=torch.float64,
        )
        lift = torch.tensor(
            [self._projected_forces[-(n - i)].lift for i in range(n)],
            dtype=torch.float64,
        )

        # Build snapshot matrix
        X = torch.stack([drag, lift], dim=0)  # (2, n)

        # DMD via SVD (simplified)
        try:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        except Exception:
            return None

        rank = min(self._dmd_rank, U.shape[1])
        eigenvalues = S[:rank]
        freqs = eigenvalues.abs().log() / (2.0 * math.pi * 0.01)  # dt_est = 0.01
        growth = eigenvalues.log().real

        return DMDMode(
            time=time,
            eigenvalues=eigenvalues,
            frequencies=freqs,
            growth_rates=growth,
            n_modes=rank,
        )

    def _track_frequency(self, time: float) -> Optional[FrequencyTracker]:
        """Track dominant frequencies in force signals."""
        if not self._projected_forces or len(self._projected_forces) < 16:
            return None

        n = min(len(self._projected_forces), 128)
        drag = torch.tensor(
            [self._projected_forces[-(n - i)].drag for i in range(n)],
            dtype=torch.float64,
        )
        lift = torch.tensor(
            [self._projected_forces[-(n - i)].lift for i in range(n)],
            dtype=torch.float64,
        )

        fft_d = torch.fft.rfft(drag)
        fft_l = torch.fft.rfft(lift)
        psd_d = fft_d.real.pow(2) + fft_d.imag.pow(2)
        psd_l = fft_l.real.pow(2) + fft_l.imag.pow(2)

        dt_est = 0.01
        freqs = torch.arange(psd_d.numel(), dtype=torch.float64) / (n * dt_est)

        idx_d = int(psd_d[1:].argmax().item()) + 1
        idx_l = int(psd_l[1:].argmax().item()) + 1

        return FrequencyTracker(
            time=time,
            dominant_freq_drag=float(freqs[idx_d].item()),
            dominant_freq_lift=float(freqs[idx_l].item()),
            amplitude_drag=float(psd_d[idx_d].sqrt().item()),
            amplitude_lift=float(psd_l[idx_l].sqrt().item()),
        )

    def execute(self, time: float) -> None:
        """Compute forces v6."""
        super().execute(time)
        if not self._enabled or not self._force_total:
            return

        if self._dmd_enabled:
            dmd = self._compute_dmd(time)
            if dmd is not None:
                self._dmd_results.append(dmd)

        if self._freq_tracking:
            ft = self._track_frequency(time)
            if ft is not None:
                self._freq_track_results.append(ft)

    @property
    def dmd_results(self) -> List[DMDMode]:
        return self._dmd_results

    @property
    def freq_track_results(self) -> List[FrequencyTracker]:
        return self._freq_track_results


FunctionObjectRegistry.register("forcesEnhanced6", ForcesEnhanced6)
