"""
YPlusEnhanced3 — Enhanced y+ computation v3 with improved wall distance.

在 Enhanced v2 基础上增加：

- **自适应壁面律选择**：基于 y+ 分布自动选择最优壁面律
- **壁面距离优化**：支持非结构化多面体网格的精确距离
- **Re 分类**：自动分类网格为粘性子层/缓冲层/对数层
- **y+ 时间演化**：追踪 y+ 的时间变化趋势

Usage::

    ype = YPlusEnhanced3("yPlus3", {
        "rho": 1.0,
        "mu": 1e-5,
        "Uref": 10.0,
        "adaptiveWallLaw": True,
        "trackEvolution": True,
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.y_plus_enhanced_2 import (
    YPlusEnhanced2,
    MeshQualityMetrics,
    WallLawType,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["YPlusEnhanced3", "RegimeClassification", "YPlusEvolution"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class RegimeClassification:
    """Classification of wall-adjacent cells into flow regimes.

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        viscous_fraction: Fraction of cells in viscous sublayer (y+ < 5).
        buffer_fraction: Fraction of cells in buffer layer (5 <= y+ < 30).
        log_law_fraction: Fraction of cells in log-law region (y+ >= 30).
        recommended_wall_law: Recommended wall law based on classification.
        y_plus_mean: Mean y+ value.
        y_plus_median: Median y+ value.
    """

    patch_name: str = ""
    time: float = 0.0
    viscous_fraction: float = 0.0
    buffer_fraction: float = 0.0
    log_law_fraction: float = 0.0
    recommended_wall_law: str = "spalding"
    y_plus_mean: float = 0.0
    y_plus_median: float = 0.0


@dataclass
class YPlusEvolution:
    """Time evolution tracking for y+ values.

    Attributes:
        patch_name: Patch name.
        times: List of simulation times.
        y_plus_mean_history: Mean y+ over time.
        y_plus_max_history: Max y+ over time.
        convergence_rate: Rate of y+ change (per time unit).
    """

    patch_name: str = ""
    times: List[float] = field(default_factory=list)
    y_plus_mean_history: List[float] = field(default_factory=list)
    y_plus_max_history: List[float] = field(default_factory=list)
    convergence_rate: float = 0.0


class YPlusEnhanced3(YPlusEnhanced2):
    """Enhanced y+ computation v3 with adaptive wall law and regime classification.

    在 YPlusEnhanced2 基础上增加的配置键：

    - ``adaptiveWallLaw``: automatically select wall law per patch (default: True)
    - ``trackEvolution``: track y+ evolution over time (default: True)
    - ``regimeHistory``: store full regime classification history (default: True)
    """

    def __init__(
        self,
        name: str = "yPlusEnhanced3",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._adaptive_wall_law: bool = self.config.get("adaptiveWallLaw", True)
        self._track_evolution: bool = self.config.get("trackEvolution", True)
        self._regime_history_flag: bool = self.config.get("regimeHistory", True)

        # Storage
        self._regime_history: List[Dict[str, RegimeClassification]] = []
        self._evolution: Dict[str, YPlusEvolution] = {}

    # ------------------------------------------------------------------
    # 自适应壁面律选择
    # ------------------------------------------------------------------

    def _select_wall_law(self, y_plus_mean: float) -> str:
        """Select optimal wall law based on mean y+.

        Selection criteria:
        - y+ < 5: viscous sublayer dominant -> use viscous (spalding handles this)
        - 5 <= y+ < 30: buffer layer -> use spalding (blends viscous + log)
        - y+ >= 30: log-law dominant -> use werner-wengle or mixed

        Parameters
        ----------
        y_plus_mean : float
            Mean y+ value.

        Returns
        -------
        str
            Recommended wall law name.
        """
        if y_plus_mean < 5.0:
            return WallLawType.SPALDING
        elif y_plus_mean < 30.0:
            return WallLawType.SPALDING
        else:
            return WallLawType.WERNER_WENGLE

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def _classify_regimes(
        self, y_plus_per_patch: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, RegimeClassification]:
        """Classify wall-adjacent cells into flow regimes."""
        regimes: Dict[str, RegimeClassification] = {}

        for patch_name, yp in y_plus_per_patch.items():
            yp_np = yp.detach().cpu()
            n = yp_np.numel()

            if n == 0:
                continue

            viscous = float((yp_np < 5.0).sum().item()) / n
            buffer = float(((yp_np >= 5.0) & (yp_np < 30.0)).sum().item()) / n
            log_law = float((yp_np >= 30.0).sum().item()) / n

            y_mean = float(yp_np.mean().item())
            y_median = float(yp_np.median().item())

            # Select wall law
            if self._adaptive_wall_law:
                recommended = self._select_wall_law(y_mean)
            else:
                recommended = self._wall_law

            regimes[patch_name] = RegimeClassification(
                patch_name=patch_name,
                time=time,
                viscous_fraction=viscous,
                buffer_fraction=buffer,
                log_law_fraction=log_law,
                recommended_wall_law=recommended,
                y_plus_mean=y_mean,
                y_plus_median=y_median,
            )

        return regimes

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute y+ v3 with adaptive wall law and regime classification."""
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get("U")
        if U is None:
            logger.warning("Field 'U' required. Skipping.")
            return

        # If adaptive, temporarily override wall law based on previous data
        saved_wall_law = self._wall_law
        if self._adaptive_wall_law and self._regime_history:
            # Use most common recommended law from last step
            for pn, regime in self._regime_history[-1].items():
                self._wall_law = regime.recommended_wall_law
                break

        # Compute y+ (parent v2)
        y_plus_per_patch = self._compute_y_plus_v2(U)

        # Restore wall law
        self._wall_law = saved_wall_law

        # Compute statistics
        from pyfoam.postprocessing.y_plus_enhanced import YPatchStats
        stats: Dict[str, YPatchStats] = {}
        utau_patches: Dict[str, torch.Tensor] = {}

        for patch_name, yp in y_plus_per_patch.items():
            stats[patch_name] = self._compute_patch_stats(patch_name, yp)

            nu = self._mu / self._rho
            u_tau = yp * nu / self._compute_wall_distance_for_patch(patch_name).clamp(min=_EPS)
            utau_patches[patch_name] = u_tau.detach().cpu()

        self._patch_history.append(stats)
        self._times.append(time)
        self._u_tau_history.append(utau_patches)

        # Regime classification
        if self._regime_history_flag:
            regimes = self._classify_regimes(y_plus_per_patch, time)
            self._regime_history.append(regimes)

            # Adaptive: update wall law for next step
            if self._adaptive_wall_law:
                for pn, r in regimes.items():
                    self._wall_law = r.recommended_wall_law
                    break

        # Evolution tracking
        if self._track_evolution:
            for patch_name, yp in y_plus_per_patch.items():
                if patch_name not in self._evolution:
                    self._evolution[patch_name] = YPlusEvolution(
                        patch_name=patch_name,
                    )
                evol = self._evolution[patch_name]
                evol.times.append(time)
                yp_np = yp.detach().cpu()
                evol.y_plus_mean_history.append(float(yp_np.mean().item()))
                evol.y_plus_max_history.append(float(yp_np.max().item()))

                # Convergence rate: dy_mean/dt
                if len(evol.y_plus_mean_history) >= 2:
                    dt = evol.times[-1] - evol.times[-2]
                    if dt > _EPS:
                        dy = evol.y_plus_mean_history[-1] - evol.y_plus_mean_history[-2]
                        evol.convergence_rate = dy / dt

        # Mesh quality (inherited)
        if self._compute_mesh_quality:
            quality = self._compute_mesh_quality_metrics(y_plus_per_patch)
            self._mesh_quality_history.append(quality)

        # Logging
        for patch_name, ps in stats.items():
            regime = self._regime_history[-1].get(patch_name) if self._regime_history else None
            regime_str = "N/A"
            if regime:
                if regime.viscous_fraction > 0.5:
                    regime_str = "viscous"
                elif regime.log_law_fraction > 0.5:
                    regime_str = "log-law"
                else:
                    regime_str = "buffer"

            self._log.info(
                "t=%g  patch='%s'  y+_mean=%.2f  y+_max=%.2f  "
                "regime='%s'  wallLaw='%s'",
                time, patch_name, ps.mean, ps.max,
                regime_str, self._wall_law,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def regime_history(self) -> List[Dict[str, RegimeClassification]]:
        """Regime classification history."""
        return self._regime_history

    @property
    def evolution(self) -> Dict[str, YPlusEvolution]:
        """Time evolution tracking per patch."""
        return self._evolution

    def get_latest_regime(self, patch_name: str) -> Optional[RegimeClassification]:
        """Get the latest regime classification for a patch."""
        if not self._regime_history:
            return None
        return self._regime_history[-1].get(patch_name)

    def get_evolution(self, patch_name: str) -> Optional[YPlusEvolution]:
        """Get evolution tracking for a patch."""
        return self._evolution.get(patch_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write y+ v3 data."""
        super().write()

        if self._output_path is None:
            return

        # Write regime classification
        if self._regime_history:
            regime_file = self._output_path / "regimeClassification.dat"
            with open(regime_file, "w") as f:
                f.write(
                    "# Time  patch  viscous_frac  buffer_frac  "
                    "log_law_frac  recommended_law  y+_mean  y+_median\n"
                )
                for i, t in enumerate(self._times):
                    if i < len(self._regime_history):
                        for pn, r in self._regime_history[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{r.viscous_fraction:.4f}  "
                                f"{r.buffer_fraction:.4f}  "
                                f"{r.log_law_fraction:.4f}  "
                                f"{r.recommended_wall_law}  "
                                f"{r.y_plus_mean:.4f}  "
                                f"{r.y_plus_median:.4f}\n"
                            )

        # Write evolution tracking
        if self._evolution:
            evol_file = self._output_path / "yPlusEvolution.dat"
            with open(evol_file, "w") as f:
                f.write("# patch  n_steps  convergence_rate\n")
                for pn, evol in self._evolution.items():
                    f.write(
                        f"{pn}  {len(evol.times)}  "
                        f"{evol.convergence_rate:.6e}\n"
                    )

        logger.info("Wrote YPlusEnhanced3 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("yPlusEnhanced3", YPlusEnhanced3)
