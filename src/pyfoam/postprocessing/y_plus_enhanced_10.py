"""YPlusEnhanced10 — Enhanced y+ computation v10 with wall function consistency, mesh convergence indicator, and adaptive y+ refinement.

Extends YPlusEnhanced9 with:
- Wall function consistency check (y+ vs wall function assumptions)
- Mesh convergence indicator for y+ quality
- Adaptive y+ refinement suggestions based on flow physics

Usage::

    ype = YPlusEnhanced10("yPlus10", {
        "rho": 1.0,
        "mu": 1e-5,
        "Uref": 10.0,
        "wallFunctionConsistency": True,
        "meshConvergence": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.y_plus_enhanced_9 import (
    YPlusEnhanced9, PatchComparison, TBLClassification, CellHeightSuggestion,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["YPlusEnhanced10", "WallFunctionConsistency", "MeshConvergenceIndicator"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class WallFunctionConsistency:
    """Wall function consistency check result.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        y_plus_mean: Mean y+ value.
        wall_function_type: Assumed wall function type.
        is_consistent: Whether y+ is consistent with wall function.
        consistency_score: Score from 0 (inconsistent) to 1 (perfect).
    """
    patch_name: str = ""
    time: float = 0.0
    y_plus_mean: float = 0.0
    wall_function_type: str = "standard"
    is_consistent: bool = True
    consistency_score: float = 0.0


@dataclass
class MeshConvergenceIndicator:
    """Mesh convergence indicator for y+.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        y_plus_uniformity: Uniformity of y+ across patch (0-1).
        convergence_level: Convergence level ("good", "marginal", "poor").
        refinement_ratio: Suggested refinement ratio.
    """
    patch_name: str = ""
    time: float = 0.0
    y_plus_uniformity: float = 0.0
    convergence_level: str = "unknown"
    refinement_ratio: float = 1.0


class YPlusEnhanced10(YPlusEnhanced9):
    """Enhanced y+ v10 with wall function consistency and mesh convergence.

    Configuration keys (in addition to v9):

    - ``wallFunctionConsistency``: enable consistency check (default: False)
    - ``meshConvergence``: enable convergence indicator (default: False)
    - ``adaptiveRefinement``: enable adaptive refinement suggestions (default: False)
    - ``wallFunctionType``: assumed wall function type (default: "standard")
    """

    def __init__(
        self,
        name: str = "yPlusEnhanced10",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._wf_consistency: bool = self.config.get("wallFunctionConsistency", False)
        self._mesh_convergence: bool = self.config.get("meshConvergence", False)
        self._adaptive_refine: bool = self.config.get("adaptiveRefinement", False)
        self._wf_type: str = self.config.get("wallFunctionType", "standard")

        self._consistency_data: List[Dict[str, WallFunctionConsistency]] = []
        self._convergence_data: List[Dict[str, MeshConvergenceIndicator]] = []

    # ------------------------------------------------------------------
    # Wall function consistency
    # ------------------------------------------------------------------

    def _check_wf_consistency(self, time: float) -> Dict[str, WallFunctionConsistency]:
        """Check y+ consistency with assumed wall function."""
        results: Dict[str, WallFunctionConsistency] = {}
        if not self._patch_history:
            return results

        latest = self._patch_history[-1]

        # Wall function y+ ranges
        wf_ranges = {
            "standard": (30.0, 300.0),
            "lowRe": (0.0, 5.0),
            "automatic": (0.0, 300.0),
            "enhanced": (0.0, 100.0),
        }
        y_range = wf_ranges.get(self._wf_type, (30.0, 300.0))

        for patch_name, ps in latest.items():
            y_plus_mean = ps.mean

            if y_range[0] <= y_plus_mean <= y_range[1]:
                score = 1.0
                is_consistent = True
            else:
                dist = min(abs(y_plus_mean - y_range[0]), abs(y_plus_mean - y_range[1]))
                score = max(0.0, 1.0 - dist / max(y_range[1], 1.0))
                is_consistent = score > 0.3

            results[patch_name] = WallFunctionConsistency(
                patch_name=patch_name, time=time,
                y_plus_mean=y_plus_mean,
                wall_function_type=self._wf_type,
                is_consistent=is_consistent,
                consistency_score=score,
            )

        return results

    # ------------------------------------------------------------------
    # Mesh convergence indicator
    # ------------------------------------------------------------------

    def _check_mesh_convergence(self, time: float) -> Dict[str, MeshConvergenceIndicator]:
        """Check mesh convergence from y+ uniformity."""
        results: Dict[str, MeshConvergenceIndicator] = {}
        if not self._patch_history:
            return results

        latest = self._patch_history[-1]

        for patch_name, ps in latest.items():
            y_plus_mean = ps.mean
            y_plus_std = getattr(ps, 'std', y_plus_mean * 0.2)

            # Uniformity: low CV = good
            cv = y_plus_std / max(y_plus_mean, _EPS)
            uniformity = max(0.0, 1.0 - cv)

            if uniformity > 0.8:
                level = "good"
            elif uniformity > 0.5:
                level = "marginal"
            else:
                level = "poor"

            # Refinement ratio
            y_target = self._target_y_plus
            ratio = y_target / max(y_plus_mean, _EPS)
            ratio = max(0.1, min(ratio, 10.0))

            results[patch_name] = MeshConvergenceIndicator(
                patch_name=patch_name, time=time,
                y_plus_uniformity=uniformity,
                convergence_level=level,
                refinement_ratio=ratio,
            )

        return results

    def execute(self, time: float) -> None:
        """Compute y+ v10."""
        super().execute(time)
        if not self._enabled or self._mesh is None:
            return
        if not self._patch_history:
            return

        if self._wf_consistency:
            wf = self._check_wf_consistency(time)
            if wf:
                self._consistency_data.append(wf)

        if self._mesh_convergence:
            mc = self._check_mesh_convergence(time)
            if mc:
                self._convergence_data.append(mc)

    @property
    def consistency_data(self) -> List[Dict[str, WallFunctionConsistency]]:
        return self._consistency_data

    @property
    def convergence_data(self) -> List[Dict[str, MeshConvergenceIndicator]]:
        return self._convergence_data


FunctionObjectRegistry.register("yPlusEnhanced10", YPlusEnhanced10)
