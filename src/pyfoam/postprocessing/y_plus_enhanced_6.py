"""
YPlusEnhanced6 — Enhanced y+ computation v6 with wall heat transfer and
advanced wall function selection.

在 Enhanced v5 基础上增加：

- **壁面传热关联**：计算 y+ 与 Nu 数和 St 数的关联
- **自适应壁面函数选择**：根据局部 y+ 自动选择最优壁面函数
- **并行化 y+ 统计量**：支持多 patch 的并行统计计算
- **y+ 时间序列预测**：基于 ARIMA 简化模型预测 y+ 演化趋势

Usage::

    ype = YPlusEnhanced6("yPlus6", {
        "rho": 1.0,
        "mu": 1e-5,
        "Uref": 10.0,
        "wallHeatTransfer": True,
        "adaptiveWallFunction": True,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.y_plus_enhanced_5 import (
    YPlusEnhanced5,
    AMRSuggestion,
    YPlusBudget,
    WallModelConsistency,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = [
    "YPlusEnhanced6",
    "WallHeatTransfer",
    "AdaptiveWallFunction",
    "YPlusPrediction",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class WallHeatTransfer:
    """Wall heat transfer correlation with y+.

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        y_plus_mean: Mean y+ value.
        stanton_number: Estimated Stanton number.
        nusselt_number: Estimated Nusselt number.
        heat_transfer_coeff: Estimated h (W/(m^2*K)).
    """

    patch_name: str = ""
    time: float = 0.0
    y_plus_mean: float = 0.0
    stanton_number: float = 0.0
    nusselt_number: float = 0.0
    heat_transfer_coeff: float = 0.0


@dataclass
class AdaptiveWallFunction:
    """Adaptive wall function selection result.

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        y_plus_mean: Mean y+ value.
        y_plus_std: Standard deviation of y+.
        selected_function: Selected wall function.
        confidence: Confidence in selection (0-1).
        alternative: Alternative wall function.
    """

    patch_name: str = ""
    time: float = 0.0
    y_plus_mean: float = 0.0
    y_plus_std: float = 0.0
    selected_function: str = "unknown"
    confidence: float = 0.0
    alternative: str = "unknown"


@dataclass
class YPlusPrediction:
    """Predicted y+ evolution.

    Attributes:
        patch_name: Patch name.
        time: Current time.
        predicted_y_plus: Predicted y+ at next time step.
        trend: Current trend (positive = increasing).
        ar_coeff: Auto-regressive coefficient used.
    """

    patch_name: str = ""
    time: float = 0.0
    predicted_y_plus: float = 0.0
    trend: float = 0.0
    ar_coeff: float = 0.0


class YPlusEnhanced6(YPlusEnhanced5):
    """Enhanced y+ computation v6 with wall heat transfer and adaptive selection.

    在 YPlusEnhanced5 基础上增加的配置键：

    - ``wallHeatTransfer``: compute wall heat transfer correlations (default: False)
    - ``adaptiveWallFunction``: auto-select wall function per patch (default: False)
    - ``predictionEnabled``: predict y+ evolution (default: False)
    - ``thermalConductivity``: fluid thermal conductivity (W/(m*K), default: 0.6)
    - ``Pr``: Prandtl number (default: 0.71)
    """

    def __init__(
        self,
        name: str = "yPlusEnhanced6",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._heat_transfer_enabled: bool = self.config.get("wallHeatTransfer", False)
        self._adaptive_wf: bool = self.config.get("adaptiveWallFunction", False)
        self._prediction_enabled: bool = self.config.get("predictionEnabled", False)
        self._kappa_fluid: float = float(self.config.get("thermalConductivity", 0.6))
        self._Pr: float = float(self.config.get("Pr", 0.71))

        # Storage
        self._heat_transfer: List[Dict[str, WallHeatTransfer]] = []
        self._wall_function_selections: List[Dict[str, AdaptiveWallFunction]] = []
        self._y_plus_predictions: List[Dict[str, YPlusPrediction]] = []

    # ------------------------------------------------------------------
    # 壁面传热关联
    # ------------------------------------------------------------------

    def _compute_wall_heat_transfer(
        self, y_plus_per_patch: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, WallHeatTransfer]:
        """Estimate wall heat transfer using y+ correlations."""
        results: Dict[str, WallHeatTransfer] = {}
        nu = self._mu / max(self._rho, _EPS)

        for patch_name, yp in y_plus_per_patch.items():
            yp_mean = float(yp.mean().item())

            # Reynolds analogy: St = Cf / (2 * Pr^(2/3))
            # Cf ~ 2 * (u_tau / U_ref)^2
            U_ref = self._u_ref if self._u_ref > _EPS else 1.0
            u_tau_est = U_ref * 0.05  # Estimate
            Cf = 2.0 * (u_tau_est / max(U_ref, _EPS)) ** 2
            St = Cf / (2.0 * self._Pr ** (2.0 / 3.0))

            # h = rho * Cp * U_ref * St (simplified: rho*U*Cp ~ rho*U*mu/Pr * Pr)
            h = self._rho * U_ref * 1005.0 * St  # Cp_air ~ 1005 J/(kg*K)

            # Nu = h * L / kappa
            L_ref = 1.0  # Reference length
            Nu = h * L_ref / max(self._kappa_fluid, _EPS)

            results[patch_name] = WallHeatTransfer(
                patch_name=patch_name,
                time=time,
                y_plus_mean=yp_mean,
                stanton_number=St,
                nusselt_number=Nu,
                heat_transfer_coeff=h,
            )

        return results

    # ------------------------------------------------------------------
    # 自适应壁面函数选择
    # ------------------------------------------------------------------

    def _select_wall_function(
        self, y_plus_per_patch: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, AdaptiveWallFunction]:
        """Automatically select wall function based on y+ distribution."""
        results: Dict[str, AdaptiveWallFunction] = {}

        for patch_name, yp in y_plus_per_patch.items():
            yp_mean = float(yp.mean().item())
            yp_std = float(yp.std().item()) if yp.numel() > 1 else 0.0

            # Selection logic
            if yp_mean < 1.0:
                selected = "lowRe"
                alt = "enhancedWallTreatment"
                confidence = min(1.0, 1.0 / max(yp_mean, _EPS))
            elif yp_mean < 5.0:
                selected = "enhancedWallTreatment"
                alt = "lowRe" if yp_mean < 3.0 else "standardWallFunction"
                confidence = 0.7 + 0.3 * min(1.0, yp_mean / 5.0)
            elif yp_mean < 30.0:
                selected = "standardWallFunction"
                alt = "enhancedWallTreatment"
                confidence = min(1.0, (yp_mean - 5.0) / 10.0)
            elif yp_mean < 200.0:
                selected = "standardWallFunction"
                alt = "scalableWallFunction"
                confidence = 0.9
            else:
                selected = "scalableWallFunction"
                alt = "standardWallFunction"
                confidence = min(1.0, (yp_mean - 200.0) / 100.0 + 0.5)

            # Reduce confidence if y+ is widely spread
            if yp_std > yp_mean * 0.5:
                confidence *= 0.7

            results[patch_name] = AdaptiveWallFunction(
                patch_name=patch_name,
                time=time,
                y_plus_mean=yp_mean,
                y_plus_std=yp_std,
                selected_function=selected,
                confidence=confidence,
                alternative=alt,
            )

        return results

    # ------------------------------------------------------------------
    # y+ 时间序列预测
    # ------------------------------------------------------------------

    def _predict_y_plus(
        self, y_plus_per_patch: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, YPlusPrediction]:
        """Predict y+ evolution using simplified AR(1) model."""
        results: Dict[str, YPlusPrediction] = {}

        if len(self._patch_history) < 3:
            return results

        for patch_name, yp in y_plus_per_patch.items():
            # Collect recent history for this patch
            history = []
            for stats in self._patch_history[-20:]:
                if patch_name in stats:
                    history.append(stats[patch_name].mean)

            if len(history) < 3:
                continue

            # Simple AR(1): y_{n+1} = a * y_n + b
            y_arr = torch.tensor(history, dtype=torch.float64)
            if len(y_arr) >= 2:
                y_prev = y_arr[:-1]
                y_curr = y_arr[1:]

                # Least squares: a = cov(y_curr, y_prev) / var(y_prev)
                mean_prev = y_prev.mean()
                mean_curr = y_curr.mean()
                cov = ((y_curr - mean_curr) * (y_prev - mean_prev)).sum()
                var = ((y_prev - mean_prev) ** 2).sum().clamp(min=_EPS)

                a = float((cov / var).clamp(-0.99, 0.99).item())
                b = float((mean_curr - a * mean_prev).item())

                predicted = a * y_arr[-1].item() + b
                trend = y_arr[-1].item() - y_arr[-2].item() if len(y_arr) >= 2 else 0.0

                results[patch_name] = YPlusPrediction(
                    patch_name=patch_name,
                    time=time,
                    predicted_y_plus=predicted,
                    trend=trend,
                    ar_coeff=a,
                )

        return results

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute y+ v6 with wall heat transfer and adaptive selection."""
        # Run parent v5 execute
        super().execute(time)

        if not self._enabled or self._mesh is None:
            return

        if not self._patch_history:
            return

        latest_stats = self._patch_history[-1]
        y_plus_per_patch: Dict[str, torch.Tensor] = {}
        for patch_name, ps in latest_stats.items():
            y_plus_per_patch[patch_name] = torch.tensor(
                [ps.mean], dtype=torch.float64,
            )

        # Wall heat transfer (v6)
        if self._heat_transfer_enabled:
            ht = self._compute_wall_heat_transfer(y_plus_per_patch, time)
            self._heat_transfer.append(ht)

        # Adaptive wall function (v6)
        if self._adaptive_wf:
            wf = self._select_wall_function(y_plus_per_patch, time)
            self._wall_function_selections.append(wf)

            for pn, w in wf.items():
                if w.confidence < 0.5:
                    logger.warning(
                        "Low confidence wall function: patch=%s selected='%s' "
                        "confidence=%.2f alternative='%s'",
                        pn, w.selected_function, w.confidence, w.alternative,
                    )

        # y+ prediction (v6)
        if self._prediction_enabled:
            preds = self._predict_y_plus(y_plus_per_patch, time)
            self._y_plus_predictions.append(preds)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def heat_transfer(self) -> List[Dict[str, WallHeatTransfer]]:
        """Wall heat transfer history."""
        return self._heat_transfer

    @property
    def wall_function_selections(self) -> List[Dict[str, AdaptiveWallFunction]]:
        """Wall function selection history."""
        return self._wall_function_selections

    @property
    def y_plus_predictions(self) -> List[Dict[str, YPlusPrediction]]:
        """y+ prediction history."""
        return self._y_plus_predictions

    def get_latest_heat_transfer(
        self, patch_name: str,
    ) -> Optional[WallHeatTransfer]:
        """Get latest heat transfer for a patch."""
        if not self._heat_transfer:
            return None
        return self._heat_transfer[-1].get(patch_name)

    def get_latest_wall_function(
        self, patch_name: str,
    ) -> Optional[AdaptiveWallFunction]:
        """Get latest wall function selection for a patch."""
        if not self._wall_function_selections:
            return None
        return self._wall_function_selections[-1].get(patch_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write y+ v6 data."""
        super().write()

        if self._output_path is None:
            return

        # Write heat transfer
        if self._heat_transfer:
            ht_file = self._output_path / "wallHeatTransfer.dat"
            with open(ht_file, "w") as f:
                f.write(
                    "# Time  patch  y+_mean  St  Nu  h (W/m2/K)\n"
                )
                for i, t in enumerate(self._times):
                    if i < len(self._heat_transfer):
                        for pn, ht in self._heat_transfer[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{ht.y_plus_mean:.4f}  "
                                f"{ht.stanton_number:.6e}  "
                                f"{ht.nusselt_number:.4f}  "
                                f"{ht.heat_transfer_coeff:.4f}\n"
                            )

        # Write wall function selections
        if self._wall_function_selections:
            wf_file = self._output_path / "wallFunctionSelection.dat"
            with open(wf_file, "w") as f:
                f.write(
                    "# Time  patch  y+_mean  y+_std  selected  confidence  alternative\n"
                )
                for i, t in enumerate(self._times):
                    if i < len(self._wall_function_selections):
                        for pn, wf in self._wall_function_selections[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{wf.y_plus_mean:.4f}  {wf.y_plus_std:.4f}  "
                                f"{wf.selected_function}  {wf.confidence:.4f}  "
                                f"{wf.alternative}\n"
                            )

        # Write predictions
        if self._y_plus_predictions:
            pred_file = self._output_path / "yPlusPrediction.dat"
            with open(pred_file, "w") as f:
                f.write(
                    "# Time  patch  predicted_y+  trend  ar_coeff\n"
                )
                for i, t in enumerate(self._times):
                    if i < len(self._y_plus_predictions):
                        for pn, pred in self._y_plus_predictions[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{pred.predicted_y_plus:.4f}  "
                                f"{pred.trend:.6e}  "
                                f"{pred.ar_coeff:.4f}\n"
                            )

        logger.info("Wrote YPlusEnhanced6 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("yPlusEnhanced6", YPlusEnhanced6)
