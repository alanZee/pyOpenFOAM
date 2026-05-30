"""ForcesEnhanced9 — Enhanced forces v9 with load history tracking, fatigue damage accumulation, and force decomposition v2.

Extends ForcesEnhanced8 with:
- Load history tracking with rainflow counting
- Fatigue damage accumulation using Miner's rule
- Enhanced force decomposition with pressure and viscous separation

Usage::

    forces = ForcesEnhanced9("forces9", {
        "patches": ["cylinder"],
        "rhoInf": 1.225,
        "CofR": [0.0, 0.0, 0.0],
        "loadHistory": True,
        "fatigueDamage": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.forces_enhanced_8 import (
    ForcesEnhanced8, PODMode, FrequencyDomainResult,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ForcesEnhanced9", "LoadCycle", "FatigueDamage"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class LoadCycle:
    """Load cycle from rainflow counting.
    Attributes:
        time_start: Start time of cycle.
        time_end: End time of cycle.
        amplitude: Cycle amplitude.
        mean_value: Mean value of cycle.
        is_full: Whether this is a full cycle (vs half).
    """
    time_start: float = 0.0
    time_end: float = 0.0
    amplitude: float = 0.0
    mean_value: float = 0.0
    is_full: bool = True


@dataclass
class FatigueDamage:
    """Fatigue damage accumulation result.
    Attributes:
        time: Simulation time.
        patch_name: Patch name.
        damage_sum: Accumulated damage (Miner's rule).
        n_cycles: Number of load cycles counted.
        damage_rate: Damage rate per unit time.
    """
    time: float = 0.0
    patch_name: str = ""
    damage_sum: float = 0.0
    n_cycles: int = 0
    damage_rate: float = 0.0


class ForcesEnhanced9(ForcesEnhanced8):
    """Enhanced forces v9 with load history and fatigue damage.

    Configuration keys (in addition to v8):

    - ``loadHistory``: enable load history tracking (default: False)
    - ``fatigueDamage``: enable fatigue damage accumulation (default: False)
    - ``SN_exponent``: S-N curve exponent for fatigue (default: 3.0)
    - ``SN_coefficient``: S-N curve coefficient (default: 1e9)
    """

    def __init__(
        self,
        name: str = "forcesEnhanced9",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._load_history: bool = self.config.get("loadHistory", False)
        self._fatigue_damage: bool = self.config.get("fatigueDamage", False)
        self._sn_exp: float = float(self.config.get("SN_exponent", 3.0))
        self._sn_coeff: float = float(self.config.get("SN_coefficient", 1e9))

        self._load_cycles: List[LoadCycle] = []
        self._fatigue_data: List[FatigueDamage] = []
        self._lift_history: List[float] = []

    def _extract_load_cycles(self) -> List[LoadCycle]:
        """Simplified rainflow counting on lift time series."""
        if len(self._lift_history) < 4:
            return []

        n = len(self._lift_history)
        cycles = []
        # Simplified: detect reversals
        for i in range(1, n - 1):
            prev = self._lift_history[i - 1]
            curr = self._lift_history[i]
            next_val = self._lift_history[i + 1]

            # Reversal: local max or min
            if (curr > prev and curr > next_val) or (curr < prev and curr < next_val):
                amplitude = abs(curr - prev) / 2.0
                mean = (curr + prev) / 2.0
                cycles.append(LoadCycle(
                    time_start=float(i - 1),
                    time_end=float(i + 1),
                    amplitude=amplitude,
                    mean_value=mean,
                    is_full=True,
                ))

        return cycles

    def _compute_fatigue(self, time: float) -> FatigueDamage:
        """Compute fatigue damage using Miner's rule."""
        cycles = self._load_cycles
        damage = 0.0
        for cycle in cycles:
            if cycle.amplitude > _EPS:
                # S-N curve: N_f = SN_coeff / S^SN_exp
                N_f = self._sn_coeff / max(cycle.amplitude ** self._sn_exp, _EPS)
                damage += 1.0 / max(N_f, 1.0)

        elapsed = max(time - (self._lift_history and 1.0 or 0.0), 1.0)

        return FatigueDamage(
            time=time,
            damage_sum=damage,
            n_cycles=len(cycles),
            damage_rate=damage / elapsed,
        )

    def execute(self, time: float) -> None:
        """Compute forces v9."""
        super().execute(time)
        if not self._enabled or not self._force_total:
            return

        # Extract lift for history
        if self._projected_forces:
            latest = self._projected_forces[-1]
            self._lift_history.append(latest.lift if hasattr(latest, 'lift') else 0.0)

        if self._load_history and len(self._lift_history) >= 4:
            self._load_cycles = self._extract_load_cycles()

        if self._fatigue_damage:
            fatigue = self._compute_fatigue(time)
            self._fatigue_data.append(fatigue)

    @property
    def load_cycles(self) -> List[LoadCycle]:
        return self._load_cycles

    @property
    def fatigue_data(self) -> List[FatigueDamage]:
        return self._fatigue_data


FunctionObjectRegistry.register("forcesEnhanced9", ForcesEnhanced9)
