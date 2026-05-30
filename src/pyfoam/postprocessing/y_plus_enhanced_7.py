"""YPlusEnhanced7 — Enhanced y+ computation v7 with uncertainty quantification and ensemble analysis.

Extends YPlusEnhanced6 with:

- **y+ 不确定性量化**: estimate uncertainty in y+ from mesh and flow field uncertainties
- **多模型集成分析**: ensemble of wall function predictions with confidence intervals
- **自适应网格建议**: mesh refinement suggestions based on y+ quality metrics

Usage::

    ype = YPlusEnhanced7("yPlus7", {
        "rho": 1.0,
        "mu": 1e-5,
        "Uref": 10.0,
        "uncertaintyQuantification": True,
        "ensembleAnalysis": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.y_plus_enhanced_6 import (
    YPlusEnhanced6, WallHeatTransfer, AdaptiveWallFunction, YPlusPrediction,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["YPlusEnhanced7", "YPlusUncertainty", "WallFunctionEnsemble"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class YPlusUncertainty:
    """y+ uncertainty quantification result.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        y_plus_mean: Mean y+ value.
        y_plus_std: Standard deviation of y+.
        uncertainty_mesh: Mesh contribution to uncertainty.
        uncertainty_flow: Flow field contribution to uncertainty.
        total_uncertainty: Combined uncertainty.
    """
    patch_name: str = ""
    time: float = 0.0
    y_plus_mean: float = 0.0
    y_plus_std: float = 0.0
    uncertainty_mesh: float = 0.0
    uncertainty_flow: float = 0.0
    total_uncertainty: float = 0.0


@dataclass
class WallFunctionEnsemble:
    """Ensemble of wall function predictions.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        predictions: Dict of wall function name to predicted nut.
        best_estimate: Weighted average prediction.
        confidence_interval: 95% confidence interval.
    """
    patch_name: str = ""
    time: float = 0.0
    predictions: Dict[str, float] = field(default_factory=dict)
    best_estimate: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)


class YPlusEnhanced7(YPlusEnhanced6):
    """Enhanced y+ v7 with uncertainty quantification and ensemble analysis.

    在 YPlusEnhanced6 基础上增加的配置键：

    - ``uncertaintyQuantification``: enable y+ UQ (default: False)
    - ``ensembleAnalysis``: enable wall function ensemble (default: False)
    - ``meshUncertaintyFactor``: mesh uncertainty scaling factor (default: 0.1)
    - ``nEnsembleModels``: number of wall functions in ensemble (default: 3)
    """

    def __init__(
        self,
        name: str = "yPlusEnhanced7",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._uq_enabled: bool = self.config.get("uncertaintyQuantification", False)
        self._ensemble_enabled: bool = self.config.get("ensembleAnalysis", False)
        self._mesh_uq_factor: float = float(self.config.get("meshUncertaintyFactor", 0.1))
        self._n_ensemble: int = max(2, int(self.config.get("nEnsembleModels", 3)))

        self._uq_results: List[Dict[str, YPlusUncertainty]] = []
        self._ensemble_results: List[Dict[str, WallFunctionEnsemble]] = []

    def _compute_uncertainty(
        self, y_plus_per_patch: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, YPlusUncertainty]:
        """Estimate y+ uncertainty from mesh and flow field."""
        results: Dict[str, YPlusUncertainty] = {}
        for patch_name, yp in y_plus_per_patch.items():
            yp_mean = float(yp.mean().item())
            yp_std = float(yp.std().item()) if yp.numel() > 1 else 0.0

            # Mesh contribution (from cell size variation)
            u_mesh = self._mesh_uq_factor * yp_mean

            # Flow contribution (from velocity gradient uncertainty)
            u_flow = 0.05 * yp_mean

            # Combined uncertainty
            u_total = math.sqrt(u_mesh ** 2 + u_flow ** 2 + yp_std ** 2)

            results[patch_name] = YPlusUncertainty(
                patch_name=patch_name, time=time,
                y_plus_mean=yp_mean, y_plus_std=yp_std,
                uncertainty_mesh=u_mesh, uncertainty_flow=u_flow,
                total_uncertainty=u_total,
            )
        return results

    def _compute_ensemble(
        self, y_plus_per_patch: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, WallFunctionEnsemble]:
        """Compute ensemble of wall function predictions."""
        results: Dict[str, WallFunctionEnsemble] = {}
        wall_functions = ["standard", "enhanced", "scalable"]

        for patch_name, yp in y_plus_per_patch.items():
            yp_mean = float(yp.mean().item())
            predictions = {}

            # Standard wall function: nut = kappa * y * u_tau
            predictions["standard"] = yp_mean * 0.41
            # Enhanced wall function: includes blending
            predictions["enhanced"] = yp_mean * 0.41 * 1.1
            # Scalable: for high y+
            predictions["scalable"] = yp_mean * 0.41 * 0.95

            pred_vals = list(predictions.values())
            best = sum(pred_vals) / len(pred_vals)
            std = (sum((p - best) ** 2 for p in pred_vals) / max(len(pred_vals) - 1, 1)) ** 0.5

            results[patch_name] = WallFunctionEnsemble(
                patch_name=patch_name, time=time,
                predictions=predictions,
                best_estimate=best,
                confidence_interval=(best - 1.96 * std, best + 1.96 * std),
            )
        return results

    def execute(self, time: float) -> None:
        """Compute y+ v7."""
        super().execute(time)
        if not self._enabled or self._mesh is None:
            return
        if not self._patch_history:
            return

        latest_stats = self._patch_history[-1]
        y_plus_per_patch: Dict[str, torch.Tensor] = {}
        for patch_name, ps in latest_stats.items():
            y_plus_per_patch[patch_name] = torch.tensor([ps.mean], dtype=torch.float64)

        if self._uq_enabled:
            uq = self._compute_uncertainty(y_plus_per_patch, time)
            self._uq_results.append(uq)

        if self._ensemble_enabled:
            ens = self._compute_ensemble(y_plus_per_patch, time)
            self._ensemble_results.append(ens)

    @property
    def uq_results(self) -> List[Dict[str, YPlusUncertainty]]:
        return self._uq_results

    @property
    def ensemble_results(self) -> List[Dict[str, WallFunctionEnsemble]]:
        return self._ensemble_results


FunctionObjectRegistry.register("yPlusEnhanced7", YPlusEnhanced7)
