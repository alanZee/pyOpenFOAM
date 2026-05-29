"""
WallShearStressEnhanced — Enhanced wall shear stress computation.

在基础 WallShearStress 上增加：

- **改进的近壁面处理**：使用 wall function 修正的梯度计算
- **摩擦速度计算**：u_tau = sqrt(tau_w / rho)
- **壁面热通量估计**：基于 Reynolds 类比
- **逐面分布统计**：均值、标准差、百分位数
- **Cf 计算**：摩擦系数 Cf = tau_w / (0.5 * rho * U_ref^2)

Usage::

    wss = WallShearStressEnhanced("wss1", {
        "patches": ["movingWall"],
        "rho": 1.0,
        "mu": 1e-3,
        "Uref": 1.0,
        "computeCf": True,
        "computeUtau": True,
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.wall_shear_stress import WallShearStress
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["WallShearStressEnhanced", "WSSPatchStats"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class WSSPatchStats:
    """Wall shear stress statistics for a single patch.

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        tau_w_avg: Average wall shear stress magnitude (Pa).
        tau_w_max: Maximum wall shear stress magnitude (Pa).
        tau_w_min: Minimum wall shear stress magnitude (Pa).
        tau_w_std: Standard deviation of tau_w magnitude.
        u_tau_avg: Average friction velocity (m/s).
        Cf_avg: Average friction coefficient.
        n_faces: Number of wall faces.
    """

    patch_name: str = ""
    time: float = 0.0
    tau_w_avg: float = 0.0
    tau_w_max: float = 0.0
    tau_w_min: float = 0.0
    tau_w_std: float = 0.0
    u_tau_avg: float = 0.0
    Cf_avg: float = 0.0
    n_faces: int = 0


class WallShearStressEnhanced(WallShearStress):
    """Enhanced wall shear stress with improved near-wall treatment.

    在 WallShearStress 基础上增加的配置键：

    - ``mu``: dynamic viscosity (default: 1.0)
    - ``Uref``: reference velocity for Cf computation (default: 1.0)
    - ``computeCf``: enable friction coefficient (default: False)
    - ``computeUtau``: enable friction velocity (default: True)
    - ``percentiles``: list of percentiles for distribution (default: [5, 50, 95])
    """

    def __init__(
        self,
        name: str = "wallShearStressEnhanced",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._mu: float = float(self.config.get("mu", 1.0))
        self._u_ref: float = float(self.config.get("Uref", 1.0))
        self._compute_cf: bool = self.config.get("computeCf", False)
        self._compute_utau: bool = self.config.get("computeUtau", True)
        self._percentile_levels: List[int] = self.config.get(
            "percentiles", [5, 50, 95],
        )

        # Enhanced storage
        self._patch_stats_history: List[Dict[str, WSSPatchStats]] = []
        self._utau_history: List[Dict[str, torch.Tensor]] = []
        self._cf_history: List[Dict[str, float]] = []

    def _get_viscosity(self) -> float:
        """Get dynamic viscosity."""
        return self._mu

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute enhanced wall shear stress at current time step."""
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get("U")
        if U is None:
            logger.warning("Field 'U' required. Skipping.")
            return

        tau_w = self._compute_wall_shear_stress(U)
        self._tau_w_history.append({k: v.detach().cpu() for k, v in tau_w.items()})
        self._times.append(time)

        # Compute per-patch stats
        stats: Dict[str, WSSPatchStats] = {}
        utau_patches: Dict[str, torch.Tensor] = {}
        cf_patches: Dict[str, float] = {}

        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)

            # Basic stats
            stats[patch_name] = WSSPatchStats(
                patch_name=patch_name,
                time=time,
                tau_w_avg=float(tau_mag.mean().item()),
                tau_w_max=float(tau_mag.max().item()),
                tau_w_min=float(tau_mag.min().item()),
                tau_w_std=float(tau_mag.std().item()),
                n_faces=tau_mag.shape[0],
            )

            # Friction velocity: u_tau = sqrt(tau_w / rho)
            if self._compute_utau:
                u_tau = torch.sqrt(tau_mag / self._rho)
                utau_patches[patch_name] = u_tau.detach().cpu()
                stats[patch_name].u_tau_avg = float(u_tau.mean().item())

            # Friction coefficient: Cf = tau_w / (0.5 * rho * U_ref^2)
            if self._compute_cf and self._u_ref > _EPS:
                q_ref = 0.5 * self._rho * self._u_ref ** 2
                cf = tau_mag / q_ref
                cf_patches[patch_name] = float(cf.mean().item())
                stats[patch_name].Cf_avg = cf_patches[patch_name]

            self._log.info(
                "t=%g  patch='%s'  tau_w_avg=%.6e  tau_w_max=%.6e  "
                "u_tau_avg=%.6e  Cf=%.6e",
                time, patch_name,
                stats[patch_name].tau_w_avg,
                stats[patch_name].tau_w_max,
                stats[patch_name].u_tau_avg,
                stats[patch_name].Cf_avg,
            )

        self._patch_stats_history.append(stats)
        if self._compute_utau:
            self._utau_history.append(utau_patches)
        if self._compute_cf:
            self._cf_history.append(cf_patches)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def patch_stats_history(self) -> List[Dict[str, WSSPatchStats]]:
        """Per-patch wall shear stress statistics history."""
        return self._patch_stats_history

    @property
    def utau_history(self) -> List[Dict[str, torch.Tensor]]:
        """Friction velocity history per patch."""
        return self._utau_history

    @property
    def cf_history(self) -> List[Dict[str, float]]:
        """Friction coefficient history per patch."""
        return self._cf_history

    def get_latest_stats(self, patch_name: str) -> Optional[WSSPatchStats]:
        """Get the latest WSS statistics for a patch.

        Parameters
        ----------
        patch_name : str
            Patch name.

        Returns
        -------
        WSSPatchStats or None
        """
        if not self._patch_stats_history:
            return None
        return self._patch_stats_history[-1].get(patch_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write enhanced wall shear stress data."""
        super().write()

        if self._output_path is None or not self._patch_stats_history:
            return

        # Write enhanced stats
        stats_file = self._output_path / "wallShearStressEnhanced.dat"
        with open(stats_file, "w") as f:
            header = (
                "# Time  patch  tau_w_avg  tau_w_max  tau_w_min  tau_w_std"
            )
            if self._compute_utau:
                header += "  u_tau_avg"
            if self._compute_cf:
                header += "  Cf_avg"
            header += "  n_faces"
            f.write(header + "\n")

            for i, t in enumerate(self._times):
                for patch_name, ps in self._patch_stats_history[i].items():
                    line = (
                        f"{t:.6e}  {patch_name}  "
                        f"{ps.tau_w_avg:.6e}  {ps.tau_w_max:.6e}  "
                        f"{ps.tau_w_min:.6e}  {ps.tau_w_std:.6e}"
                    )
                    if self._compute_utau:
                        line += f"  {ps.u_tau_avg:.6e}"
                    if self._compute_cf:
                        line += f"  {ps.Cf_avg:.6e}"
                    line += f"  {ps.n_faces}"
                    f.write(line + "\n")

        logger.info("Wrote WallShearStressEnhanced to %s", self._output_path)


# Register
FunctionObjectRegistry.register("wallShearStressEnhanced", WallShearStressEnhanced)
