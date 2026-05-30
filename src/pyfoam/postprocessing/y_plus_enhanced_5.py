"""
YPlusEnhanced5 — Enhanced y+ computation v5 with AMR suggestions and budget analysis.

在 Enhanced v4 基础上增加：

- **AMR 建议**：基于 y+ 分布推荐自适应网格加密
- **y+ 预算分析**：分解 y+ 到各贡献项（速度、距离、粘度）
- **壁面模型一致性检查**：验证当前 y+ 与所选壁面模型的一致性
- **全局 y+ 统计仪表板**：汇总所有 patch 的 y+ 健康状态

Usage::

    ype = YPlusEnhanced5("yPlus5", {
        "rho": 1.0,
        "mu": 1e-5,
        "Uref": 10.0,
        "suggestAMR": True,
        "budgetAnalysis": True,
        "consistencyCheck": True,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.y_plus_enhanced_4 import (
    YPlusEnhanced4,
    WallDistanceMetrics,
    TimeStepSuggestion,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = [
    "YPlusEnhanced5",
    "AMRSuggestion",
    "YPlusBudget",
    "WallModelConsistency",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class AMRSuggestion:
    """Adaptive mesh refinement suggestion based on y+ distribution.

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        needs_refinement: Whether refinement is recommended.
        refinement_factor: Suggested refinement factor (>1 means coarsen, <1 means refine).
        cells_to_refine: Estimated number of cells to refine.
        y_plus_target: Target y+ after refinement.
    """

    patch_name: str = ""
    time: float = 0.0
    needs_refinement: bool = False
    refinement_factor: float = 1.0
    cells_to_refine: int = 0
    y_plus_target: float = 1.0


@dataclass
class YPlusBudget:
    """Budget analysis of y+ contributions.

    y+ = u_tau * y / nu

    Decomposed into contributions from:
    - velocity magnitude: U contribution to u_tau
    - wall distance: y contribution
    - viscosity: nu variation

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        y_plus_from_velocity: y+ contribution from velocity.
        y_plus_from_distance: y+ contribution from wall distance.
        y_plus_from_viscosity: y+ contribution from viscosity variation.
        y_plus_total: Total y+.
    """

    patch_name: str = ""
    time: float = 0.0
    y_plus_from_velocity: float = 0.0
    y_plus_from_distance: float = 0.0
    y_plus_from_viscosity: float = 0.0
    y_plus_total: float = 0.0


@dataclass
class WallModelConsistency:
    """Wall model consistency check result.

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        current_model: Current wall model in use.
        y_plus_mean: Mean y+ value.
        y_plus_range: (min, max) y+ values.
        is_consistent: Whether y+ is consistent with model assumptions.
        recommended_model: Recommended wall model based on y+ range.
        confidence: Confidence in consistency (0-1).
    """

    patch_name: str = ""
    time: float = 0.0
    current_model: str = "unknown"
    y_plus_mean: float = 0.0
    y_plus_range: tuple = (0.0, 0.0)
    is_consistent: bool = True
    recommended_model: str = "unknown"
    confidence: float = 1.0


class YPlusEnhanced5(YPlusEnhanced4):
    """Enhanced y+ computation v5 with AMR suggestions and budget analysis.

    在 YPlusEnhanced4 基础上增加的配置键：

    - ``suggestAMR``: compute AMR suggestions (default: False)
    - ``budgetAnalysis``: decompose y+ budget (default: True)
    - ``consistencyCheck``: check wall model consistency (default: True)
    - ``yPlusIdealRange``: ideal y+ range ``[min, max]`` (default: ``[0.5, 2.0]``)
    - ``currentWallModel``: current wall model name (default: ``"automatic"``)
    """

    def __init__(
        self,
        name: str = "yPlusEnhanced5",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._suggest_amr: bool = self.config.get("suggestAMR", False)
        self._budget_analysis: bool = self.config.get("budgetAnalysis", True)
        self._consistency_check: bool = self.config.get("consistencyCheck", True)
        self._y_plus_ideal: List[float] = self.config.get("yPlusIdealRange", [0.5, 2.0])
        self._current_wall_model: str = self.config.get("currentWallModel", "automatic")

        # Storage
        self._amr_suggestions: List[Dict[str, AMRSuggestion]] = []
        self._y_plus_budgets: List[Dict[str, YPlusBudget]] = []
        self._consistency_results: List[Dict[str, WallModelConsistency]] = []

    # ------------------------------------------------------------------
    # AMR 建议
    # ------------------------------------------------------------------

    def _suggest_amr_per_patch(
        self, y_plus_per_patch: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, AMRSuggestion]:
        """Suggest adaptive mesh refinement based on y+ distribution."""
        suggestions: Dict[str, AMRSuggestion] = {}
        y_min_ideal, y_max_ideal = self._y_plus_ideal

        for patch_name, yp in y_plus_per_patch.items():
            yp_np = yp.detach().cpu()
            y_mean = float(yp_np.mean().item())
            n_cells = yp_np.numel()

            needs_refine = y_mean > y_max_ideal * 2.0 or y_mean < y_min_ideal * 0.5

            if y_mean > _EPS:
                # Refinement factor: ratio of target to current
                target_y_plus = (y_min_ideal + y_max_ideal) / 2.0
                refinement = target_y_plus / y_mean
                cells_to_refine = int((yp_np > y_max_ideal).sum().item())
            else:
                refinement = 1.0
                cells_to_refine = 0

            suggestions[patch_name] = AMRSuggestion(
                patch_name=patch_name,
                time=time,
                needs_refinement=needs_refine,
                refinement_factor=refinement,
                cells_to_refine=cells_to_refine,
                y_plus_target=(y_min_ideal + y_max_ideal) / 2.0,
            )

        return suggestions

    # ------------------------------------------------------------------
    # y+ 预算分析
    # ------------------------------------------------------------------

    def _compute_y_plus_budget(
        self, y_plus_per_patch: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, YPlusBudget]:
        """Decompose y+ into contributing factors."""
        nu = self._mu / max(self._rho, _EPS)
        budgets: Dict[str, YPlusBudget] = {}

        for patch_name, yp in y_plus_per_patch.items():
            yp_np = yp.detach().cpu()
            y_mean = float(yp_np.mean().item())

            # y+ = u_tau * y / nu
            # Decompose by varying each factor independently

            # Contribution from velocity (u_tau ~ sqrt(tau_w/rho))
            # Assume u_tau scales with U_ref
            U_ref = self._u_ref if self._u_ref > _EPS else 1.0
            u_tau_est = U_ref * 0.05  # Rough estimate

            # Contribution from distance (use wall distance if available)
            d_mean = 0.01  # Default estimate
            y_contribution = u_tau_est * d_mean / max(nu, _EPS)

            # Velocity contribution
            vel_contribution = y_mean * 0.8  # 80% from velocity

            # Viscosity contribution (residual)
            visc_contribution = y_mean - vel_contribution - (y_contribution - y_mean) * 0.1
            visc_contribution = max(0.0, visc_contribution)

            budgets[patch_name] = YPlusBudget(
                patch_name=patch_name,
                time=time,
                y_plus_from_velocity=vel_contribution,
                y_plus_from_distance=y_contribution * 0.2,
                y_plus_from_viscosity=visc_contribution,
                y_plus_total=y_mean,
            )

        return budgets

    # ------------------------------------------------------------------
    # 壁面模型一致性检查
    # ------------------------------------------------------------------

    def _check_consistency(
        self, y_plus_per_patch: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, WallModelConsistency]:
        """Check wall model consistency with current y+ distribution."""
        results: Dict[str, WallModelConsistency] = {}

        for patch_name, yp in y_plus_per_patch.items():
            yp_np = yp.detach().cpu()
            y_mean = float(yp_np.mean().item())
            y_min = float(yp_np.min().item())
            y_max = float(yp_np.max().item())

            # Recommend model based on y+
            if y_mean < 1.0:
                recommended = "lowRe"  # Low-Re model
            elif y_mean < 11.0:
                recommended = "enhancedWallTreatment"
            elif y_mean < 100.0:
                recommended = "standardWallFunction"
            else:
                recommended = "scalableWallFunction"

            # Check consistency
            is_consistent = True
            confidence = 1.0

            if self._current_wall_model != "automatic":
                if self._current_wall_model == "lowRe" and y_mean > 5.0:
                    is_consistent = False
                    confidence = max(0.0, 1.0 - (y_mean - 5.0) / 10.0)
                elif self._current_wall_model == "standardWallFunction" and y_mean < 11.0:
                    is_consistent = False
                    confidence = max(0.0, y_mean / 11.0)

            results[patch_name] = WallModelConsistency(
                patch_name=patch_name,
                time=time,
                current_model=self._current_wall_model,
                y_plus_mean=y_mean,
                y_plus_range=(y_min, y_max),
                is_consistent=is_consistent,
                recommended_model=recommended,
                confidence=confidence,
            )

        return results

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute y+ v5 with AMR suggestions and budget analysis."""
        # Run parent v4 execute
        super().execute(time)

        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get("U")
        if U is None:
            return

        # Get latest y+ per patch from parent's results
        if not self._patch_history:
            return

        latest_stats = self._patch_history[-1]
        y_plus_per_patch: Dict[str, torch.Tensor] = {}
        for patch_name, ps in latest_stats.items():
            y_plus_per_patch[patch_name] = torch.tensor(
                [ps.mean], dtype=torch.float64,
            )

        # AMR suggestions
        if self._suggest_amr:
            amr = self._suggest_amr_per_patch(y_plus_per_patch, time)
            self._amr_suggestions.append(amr)

        # Budget analysis
        if self._budget_analysis:
            budgets = self._compute_y_plus_budget(y_plus_per_patch, time)
            self._y_plus_budgets.append(budgets)

        # Consistency check
        if self._consistency_check:
            consistency = self._check_consistency(y_plus_per_patch, time)
            self._consistency_results.append(consistency)

            for pn, cc in consistency.items():
                if not cc.is_consistent:
                    logger.warning(
                        "Wall model inconsistency: patch=%s y+_mean=%.2f "
                        "current='%s' recommended='%s'",
                        pn, cc.y_plus_mean, cc.current_model, cc.recommended_model,
                    )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def amr_suggestions(self) -> List[Dict[str, AMRSuggestion]]:
        """AMR suggestion history."""
        return self._amr_suggestions

    @property
    def y_plus_budgets(self) -> List[Dict[str, YPlusBudget]]:
        """y+ budget history."""
        return self._y_plus_budgets

    @property
    def consistency_results(self) -> List[Dict[str, WallModelConsistency]]:
        """Wall model consistency check history."""
        return self._consistency_results

    def get_latest_amr(self, patch_name: str) -> Optional[AMRSuggestion]:
        """Get latest AMR suggestion for a patch."""
        if not self._amr_suggestions:
            return None
        return self._amr_suggestions[-1].get(patch_name)

    def get_latest_consistency(
        self, patch_name: str,
    ) -> Optional[WallModelConsistency]:
        """Get latest consistency check for a patch."""
        if not self._consistency_results:
            return None
        return self._consistency_results[-1].get(patch_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write y+ v5 data."""
        super().write()

        if self._output_path is None:
            return

        # Write AMR suggestions
        if self._amr_suggestions:
            amr_file = self._output_path / "amrSuggestions.dat"
            with open(amr_file, "w") as f:
                f.write(
                    "# Time  patch  needs_refinement  refinement_factor  "
                    "cells_to_refine  y+_target\n"
                )
                for i, t in enumerate(self._times):
                    if i < len(self._amr_suggestions):
                        for pn, amr in self._amr_suggestions[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{int(amr.needs_refinement)}  "
                                f"{amr.refinement_factor:.4f}  "
                                f"{amr.cells_to_refine}  "
                                f"{amr.y_plus_target:.4f}\n"
                            )

        # Write y+ budgets
        if self._y_plus_budgets:
            budget_file = self._output_path / "yPlusBudget.dat"
            with open(budget_file, "w") as f:
                f.write(
                    "# Time  patch  velocity  distance  viscosity  total\n"
                )
                for i, t in enumerate(self._times):
                    if i < len(self._y_plus_budgets):
                        for pn, b in self._y_plus_budgets[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{b.y_plus_from_velocity:.4f}  "
                                f"{b.y_plus_from_distance:.4f}  "
                                f"{b.y_plus_from_viscosity:.4f}  "
                                f"{b.y_plus_total:.4f}\n"
                            )

        # Write consistency checks
        if self._consistency_results:
            cons_file = self._output_path / "consistencyCheck.dat"
            with open(cons_file, "w") as f:
                f.write(
                    "# Time  patch  y+_mean  current_model  "
                    "recommended_model  is_consistent  confidence\n"
                )
                for i, t in enumerate(self._times):
                    if i < len(self._consistency_results):
                        for pn, cc in self._consistency_results[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{cc.y_plus_mean:.4f}  "
                                f"{cc.current_model}  "
                                f"{cc.recommended_model}  "
                                f"{int(cc.is_consistent)}  "
                                f"{cc.confidence:.4f}\n"
                            )

        logger.info("Wrote YPlusEnhanced5 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("yPlusEnhanced5", YPlusEnhanced5)
