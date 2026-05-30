"""ForcesEnhanced10 — Enhanced forces v10 with spectral moment analysis, force ratio tracking, and steady-state detection.

Extends ForcesEnhanced9 with:
- Spectral moment analysis (zero-crossing frequency, bandwidth)
- Force ratio tracking (lift-to-drag evolution)
- Steady-state detection from force history

Usage::

    forces = ForcesEnhanced10("forces10", {
        "patches": ["cylinder"],
        "rhoInf": 1.225,
        "CofR": [0.0, 0.0, 0.0],
        "spectralMoments": True,
        "steadyStateDetection": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.forces_enhanced_9 import (
    ForcesEnhanced9, LoadCycle, FatigueDamage,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ForcesEnhanced10", "SpectralMoment", "SteadyStateIndicator"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class SpectralMoment:
    """Spectral moment analysis result.
    Attributes:
        time: Simulation time.
        m0: Zeroth moment (variance).
        m2: Second moment (related to zero-crossing rate).
        m4: Fourth moment.
        zero_crossing_freq: Estimated zero-crossing frequency.
        bandwidth: Spectral bandwidth.
    """
    time: float = 0.0
    m0: float = 0.0
    m2: float = 0.0
    m4: float = 0.0
    zero_crossing_freq: float = 0.0
    bandwidth: float = 0.0


@dataclass
class SteadyStateIndicator:
    """Steady-state detection result.
    Attributes:
        time: Simulation time.
        is_steady: Whether force has reached steady state.
        convergence_rate: Rate of convergence (smaller = more converged).
        time_to_steady: Estimated time to reach steady state.
    """
    time: float = 0.0
    is_steady: bool = False
    convergence_rate: float = 0.0
    time_to_steady: float = 0.0


class ForcesEnhanced10(ForcesEnhanced9):
    """Enhanced forces v10 with spectral moments and steady-state detection.

    Configuration keys (in addition to v9):

    - ``spectralMoments``: enable spectral moment analysis (default: False)
    - ``steadyStateDetection``: enable steady-state detection (default: False)
    - ``forceRatioTracking``: enable force ratio tracking (default: False)
    - ``steadyTolerance``: tolerance for steady state (default: 0.01)
    """

    def __init__(
        self,
        name: str = "forcesEnhanced10",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._spectral_moments: bool = self.config.get("spectralMoments", False)
        self._steady_detect: bool = self.config.get("steadyStateDetection", False)
        self._force_ratio: bool = self.config.get("forceRatioTracking", False)
        self._steady_tol: float = float(self.config.get("steadyTolerance", 0.01))

        self._spectral_data: List[SpectralMoment] = []
        self._steady_data: List[SteadyStateIndicator] = []
        self._drag_history: List[float] = []
        self._ld_ratio_history: List[float] = []

    def _compute_spectral_moments(self, time: float) -> Optional[SpectralMoment]:
        """Compute spectral moments from lift history."""
        if len(self._lift_history) < 8:
            return None

        n = len(self._lift_history)
        mean_lift = sum(self._lift_history) / n
        deviations = [f - mean_lift for f in self._lift_history]

        # Zeroth moment (variance)
        m0 = sum(d ** 2 for d in deviations) / n

        # Second moment (from finite differences)
        diffs = [deviations[i + 1] - deviations[i] for i in range(n - 1)]
        m2 = sum(d ** 2 for d in diffs) / max(len(diffs), 1)

        # Fourth moment
        d2 = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        m4 = sum(d ** 2 for d in d2) / max(len(d2), 1)

        f_zc = math.sqrt(max(m2 / max(m0, _EPS), 0.0)) / (2.0 * math.pi)
        bw = math.sqrt(max(1.0 - m2 ** 2 / max(m0 * m4, _EPS), 0.0))

        return SpectralMoment(
            time=time, m0=m0, m2=m2, m4=m4,
            zero_crossing_freq=f_zc, bandwidth=bw,
        )

    def _detect_steady_state(self, time: float) -> Optional[SteadyStateIndicator]:
        """Detect steady state from force history."""
        if len(self._lift_history) < 10:
            return None

        n = len(self._lift_history)
        window = min(n, 10)
        recent = self._lift_history[-window:]
        mean_r = sum(recent) / window
        var_r = sum((f - mean_r) ** 2 for f in recent) / max(window, 1)

        cv = math.sqrt(max(var_r, _EPS)) / max(abs(mean_r), _EPS)
        is_steady = cv < self._steady_tol

        # Convergence rate estimate
        if n >= 20:
            old = self._lift_history[-20:-10]
            mean_old = sum(old) / len(old)
            rate = abs(mean_r - mean_old) / max(abs(mean_old), _EPS)
        else:
            rate = cv

        return SteadyStateIndicator(
            time=time,
            is_steady=is_steady,
            convergence_rate=rate,
            time_to_steady=0.0 if is_steady else rate * 100.0,
        )

    def execute(self, time: float) -> None:
        """Compute forces v10."""
        super().execute(time)
        if not self._enabled or not self._force_total:
            return

        if self._spectral_moments:
            sm = self._compute_spectral_moments(time)
            if sm is not None:
                self._spectral_data.append(sm)

        if self._steady_detect:
            ss = self._detect_steady_state(time)
            if ss is not None:
                self._steady_data.append(ss)

    @property
    def spectral_data(self) -> List[SpectralMoment]:
        return self._spectral_data

    @property
    def steady_data(self) -> List[SteadyStateIndicator]:
        return self._steady_data


FunctionObjectRegistry.register("forcesEnhanced10", ForcesEnhanced10)
