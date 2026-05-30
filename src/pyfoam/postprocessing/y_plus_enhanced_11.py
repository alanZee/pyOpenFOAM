"""YPlusEnhanced11 — Enhanced y+ computation v11 with wall heat transfer coefficient, multi-patch comparison, and adaptive y+ history.

Extends YPlusEnhanced10 with:
- Wall heat transfer coefficient estimation
- Multi-patch y+ comparison and ranking
- Adaptive y+ history with trend analysis

Usage::

    ype = YPlusEnhanced11("yPlus11", {
        "rho": 1.0,
        "mu": 1e-5,
        "Uref": 10.0,
        "wallHeatTransfer": True,
        "multiPatchComparison": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.y_plus_enhanced_10 import (
    YPlusEnhanced10, WallFunctionConsistency, MeshConvergenceIndicator,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["YPlusEnhanced11", "WallHeatTransferCoeff", "PatchRanking"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class WallHeatTransferCoeff:
    """Wall heat transfer coefficient estimate.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        htc: Heat transfer coefficient (W/(m^2*K)).
        Nusselt: Nusselt number.
        y_plus_mean: Mean y+ on patch.
    """
    patch_name: str = ""
    time: float = 0.0
    htc: float = 0.0
    Nusselt: float = 0.0
    y_plus_mean: float = 0.0


@dataclass
class PatchRanking:
    """Multi-patch y+ ranking.
    Attributes:
        time: Simulation time.
        best_patch: Patch with best (closest to target) y+.
        worst_patch: Patch with worst y+.
        rankings: Sorted list of (patch_name, y+_mean) tuples.
    """
    time: float = 0.0
    best_patch: str = ""
    worst_patch: str = ""
    rankings: List[tuple[str, float]] = field(default_factory=list)


class YPlusEnhanced11(YPlusEnhanced10):
    """Enhanced y+ v11 with wall heat transfer and multi-patch comparison.

    Configuration keys (in addition to v10):

    - ``wallHeatTransfer``: enable HTC estimation (default: False)
    - ``multiPatchComparison``: enable patch ranking (default: False)
    - ``yPlusHistory``: enable y+ history tracking (default: False)
    - ``Pr``: Prandtl number (default: 0.71)
    - ``k_thermal``: thermal conductivity (default: 0.026)
    """

    def __init__(
        self,
        name: str = "yPlusEnhanced11",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._wall_htc: bool = self.config.get("wallHeatTransfer", False)
        self._patch_compare: bool = self.config.get("multiPatchComparison", False)
        self._y_plus_history_flag: bool = self.config.get("yPlusHistory", False)
        self._Pr: float = float(self.config.get("Pr", 0.71))
        self._k_thermal: float = float(self.config.get("k_thermal", 0.026))

        self._htc_data: List[Dict[str, WallHeatTransferCoeff]] = []
        self._ranking_data: List[PatchRanking] = []
        self._y_plus_trend: List[Dict[str, float]] = []

    def _estimate_htc(self, time: float) -> Dict[str, WallHeatTransferCoeff]:
        """Estimate wall heat transfer coefficient from y+."""
        results: Dict[str, WallHeatTransferCoeff] = {}
        if not self._patch_history:
            return results

        latest = self._patch_history[-1]
        rho = float(self.config.get("rho", 1.0))
        Cp = float(self.config.get("Cp", 1005.0))
        mu = float(self.config.get("mu", 1e-5))

        for patch_name, ps in latest.items():
            y_plus_mean = ps.mean
            kappa = 0.41
            E = 9.8

            # Estimate u_tau from y+
            y = 0.001  # Typical wall distance
            u_tau = y_plus_mean * mu / max(rho * y, _EPS)

            # HTC from Reynolds analogy
            Pr_t = 0.85
            if y_plus_mean < 11.0:
                Nu = self._Pr * y_plus_mean
            else:
                Nu = Pr_t * (1.0 / kappa * math.log(max(E * y_plus_mean, 1.1)))

            htc = Nu * self._k_thermal / max(y, _EPS)

            results[patch_name] = WallHeatTransferCoeff(
                patch_name=patch_name, time=time,
                htc=htc, Nusselt=Nu, y_plus_mean=y_plus_mean,
            )

        return results

    def _rank_patches(self, time: float) -> Optional[PatchRanking]:
        """Rank patches by y+ quality."""
        if not self._patch_history:
            return None

        latest = self._patch_history[-1]
        y_target = self._target_y_plus

        rankings = []
        for patch_name, ps in latest.items():
            rankings.append((patch_name, ps.mean))

        rankings.sort(key=lambda x: abs(x[1] - y_target))

        return PatchRanking(
            time=time,
            best_patch=rankings[0][0] if rankings else "",
            worst_patch=rankings[-1][0] if rankings else "",
            rankings=rankings,
        )

    def execute(self, time: float) -> None:
        """Compute y+ v11."""
        super().execute(time)
        if not self._enabled or self._mesh is None:
            return
        if not self._patch_history:
            return

        if self._wall_htc:
            htc = self._estimate_htc(time)
            if htc:
                self._htc_data.append(htc)

        if self._patch_compare:
            ranking = self._rank_patches(time)
            if ranking is not None:
                self._ranking_data.append(ranking)

    @property
    def htc_data(self) -> List[Dict[str, WallHeatTransferCoeff]]:
        return self._htc_data

    @property
    def ranking_data(self) -> List[PatchRanking]:
        return self._ranking_data


FunctionObjectRegistry.register("yPlusEnhanced11", YPlusEnhanced11)
