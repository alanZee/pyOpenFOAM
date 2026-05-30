"""YPlusEnhanced9 — Enhanced y+ computation v9 with multi-patch comparison, TBL classification, and first cell height suggestion.

Extends YPlusEnhanced8 with:

- **多 patch y+ 对比**: multi-patch y+ comparison and ranking
- **湍流边界层分类**: turbulent boundary layer regime classification
- **第一层网格高度建议**: first cell height suggestion for target y+

Usage::

    ype = YPlusEnhanced9("yPlus9", {
        "rho": 1.0,
        "mu": 1e-5,
        "Uref": 10.0,
        "multiPatchComparison": True,
        "tblClassification": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.y_plus_enhanced_8 import (
    YPlusEnhanced8, YPlusSpectrum, MeshAdaptationCriterion,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["YPlusEnhanced9", "PatchComparison", "TBLClassification", "CellHeightSuggestion"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class PatchComparison:
    """Multi-patch y+ comparison.
    Attributes:
        time: Simulation time.
        patch_means: Dict of {patch_name: mean y+}.
        best_patch: Patch with y+ closest to optimal.
        worst_patch: Patch with y+ furthest from optimal.
    """
    time: float = 0.0
    patch_means: Dict[str, float] = field(default_factory=dict)
    best_patch: str = ""
    worst_patch: str = ""


@dataclass
class TBLClassification:
    """Turbulent boundary layer classification.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        regime: Classification ("laminar", "transitional", "turbulent").
        Re_theta: Estimated momentum thickness Reynolds number.
    """
    patch_name: str = ""
    time: float = 0.0
    regime: str = "unknown"
    Re_theta: float = 0.0


@dataclass
class CellHeightSuggestion:
    """First cell height suggestion for target y+.
    Attributes:
        patch_name: Patch name.
        current_y_plus: Current mean y+.
        target_y_plus: Target y+.
        suggested_height: Suggested first cell height (m).
        height_ratio: Ratio of suggested to current height.
    """
    patch_name: str = ""
    current_y_plus: float = 0.0
    target_y_plus: float = 1.0
    suggested_height: float = 0.0
    height_ratio: float = 1.0


class YPlusEnhanced9(YPlusEnhanced8):
    """Enhanced y+ v9 with multi-patch comparison and cell height suggestion.

    Extends v8 with:

    - **Multi-patch comparison**: ranks patches by y+ quality.
    - **TBL classification**: classifies boundary layer regime.
    - **Cell height suggestion**: computes required first cell height for target y+.

    Configuration keys (in addition to v8):

    - ``multiPatchComparison``: enable multi-patch comparison (default: False)
    - ``tblClassification``: enable TBL classification (default: False)
    - ``cellHeightSuggestion``: enable cell height suggestion (default: False)
    - ``targetYPlus``: target y+ for cell height suggestion (default: 1.0)
    """

    def __init__(
        self,
        name: str = "yPlusEnhanced9",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._multi_patch: bool = self.config.get("multiPatchComparison", False)
        self._tbl_classify: bool = self.config.get("tblClassification", False)
        self._cell_height: bool = self.config.get("cellHeightSuggestion", False)
        self._target_y_plus: float = float(self.config.get("targetYPlus", 1.0))

        self._patch_comparisons: List[PatchComparison] = []
        self._tbl_data: List[Dict[str, TBLClassification]] = []
        self._cell_height_suggestions: List[Dict[str, CellHeightSuggestion]] = []

    def _compare_patches(self, time: float) -> Optional[PatchComparison]:
        """Compare y+ across patches."""
        if not self._patch_history:
            return None

        latest = self._patch_history[-1]
        patch_means: Dict[str, float] = {}
        for patch_name, ps in latest.items():
            patch_means[patch_name] = ps.mean

        if not patch_means:
            return None

        y_opt = self._y_plus_optimal
        best = min(patch_means, key=lambda k: abs(patch_means[k] - y_opt))
        worst = max(patch_means, key=lambda k: abs(patch_means[k] - y_opt))

        return PatchComparison(
            time=time,
            patch_means=patch_means,
            best_patch=best,
            worst_patch=worst,
        )

    def _classify_tbl(self, time: float) -> Dict[str, TBLClassification]:
        """Classify turbulent boundary layer regime from y+."""
        if not self._patch_history:
            return {}

        latest = self._patch_history[-1]
        results: Dict[str, TBLClassification] = {}

        for patch_name, ps in latest.items():
            y_plus_mean = ps.mean
            U_ref = self.config.get("Uref", 10.0)
            nu = self.config.get("mu", 1e-5)
            rho = self.config.get("rho", 1.0)

            # Estimate Re_theta from y+ (simplified)
            Re_theta = y_plus_mean * U_ref / max(nu, _EPS) * 0.01

            if Re_theta < 100:
                regime = "laminar"
            elif Re_theta < 500:
                regime = "transitional"
            else:
                regime = "turbulent"

            results[patch_name] = TBLClassification(
                patch_name=patch_name, time=time,
                regime=regime, Re_theta=Re_theta,
            )

        return results

    def _suggest_cell_height(self, time: float) -> Dict[str, CellHeightSuggestion]:
        """Suggest first cell height for target y+."""
        if not self._patch_history:
            return {}

        latest = self._patch_history[-1]
        results: Dict[str, CellHeightSuggestion] = {}
        y_target = self._target_y_plus
        nu = self.config.get("mu", 1e-5)
        U_ref = self.config.get("Uref", 10.0)

        for patch_name, ps in latest.items():
            y_plus_current = ps.mean
            if y_plus_current < _EPS:
                continue

            # Estimate current cell height: y = y+ * nu / u_tau
            # u_tau ~ U_ref * sqrt(Cf/2) with Cf ~ 0.026 * Re^(-1/7)
            u_tau_est = U_ref * 0.05  # Simplified estimate
            y_current = y_plus_current * nu / max(u_tau_est, _EPS)
            y_suggested = y_target * nu / max(u_tau_est, _EPS)

            results[patch_name] = CellHeightSuggestion(
                patch_name=patch_name,
                current_y_plus=y_plus_current,
                target_y_plus=y_target,
                suggested_height=y_suggested,
                height_ratio=y_suggested / max(y_current, _EPS),
            )

        return results

    def execute(self, time: float) -> None:
        """Compute y+ v9."""
        super().execute(time)
        if not self._enabled or self._mesh is None:
            return
        if not self._patch_history:
            return

        if self._multi_patch:
            comp = self._compare_patches(time)
            if comp is not None:
                self._patch_comparisons.append(comp)

        if self._tbl_classify:
            tbl = self._classify_tbl(time)
            if tbl:
                self._tbl_data.append(tbl)

        if self._cell_height:
            ch = self._suggest_cell_height(time)
            if ch:
                self._cell_height_suggestions.append(ch)

    @property
    def patch_comparisons(self) -> List[PatchComparison]:
        return self._patch_comparisons

    @property
    def tbl_data(self) -> List[Dict[str, TBLClassification]]:
        return self._tbl_data

    @property
    def cell_height_suggestions(self) -> List[Dict[str, CellHeightSuggestion]]:
        return self._cell_height_suggestions


FunctionObjectRegistry.register("yPlusEnhanced9", YPlusEnhanced9)
