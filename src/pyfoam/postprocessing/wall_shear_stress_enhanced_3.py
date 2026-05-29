"""
WallShearStressEnhanced3 — Enhanced wall shear stress v3 with improved near-wall treatment.

在 Enhanced v2 基础上增加：

- **自适应近壁面模型**：根据 y+ 自动选择壁面剪切应力计算方式
- **摩擦系数分布**：逐面 Cf 分布统计
- **时间演化追踪**：tau_w 和 u_tau 的时间变化监测

Usage::

    wss = WallShearStressEnhanced3("wss3", {
        "patches": ["movingWall"],
        "rho": 1.0,
        "mu": 1e-3,
        "Uref": 1.0,
        "adaptiveNearWall": True,
        "trackEvolution": True,
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.wall_shear_stress_enhanced_2 import (
    WallShearStressEnhanced2,
    WSSDistribution,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["WallShearStressEnhanced3", "CfDistribution", "WSSEvolution"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class CfDistribution:
    """Per-face friction coefficient distribution.

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        cf_mean: Mean friction coefficient.
        cf_min: Minimum friction coefficient.
        cf_max: Maximum friction coefficient.
        cf_std: Standard deviation of Cf.
        n_faces: Number of faces.
    """

    patch_name: str = ""
    time: float = 0.0
    cf_mean: float = 0.0
    cf_min: float = 0.0
    cf_max: float = 0.0
    cf_std: float = 0.0
    n_faces: int = 0


@dataclass
class WSSEvolution:
    """Time evolution tracking for wall shear stress.

    Attributes:
        patch_name: Patch name.
        times: List of times.
        tau_w_mean_history: Mean tau_w over time.
        tau_w_max_history: Max tau_w over time.
        u_tau_mean_history: Mean u_tau over time.
        convergence_rate: Rate of tau_w change per time unit.
    """

    patch_name: str = ""
    times: List[float] = field(default_factory=list)
    tau_w_mean_history: List[float] = field(default_factory=list)
    tau_w_max_history: List[float] = field(default_factory=list)
    u_tau_mean_history: List[float] = field(default_factory=list)
    convergence_rate: float = 0.0


class WallShearStressEnhanced3(WallShearStressEnhanced2):
    """Enhanced wall shear stress v3 with adaptive near-wall and evolution tracking.

    在 WallShearStressEnhanced2 基础上增加的配置键：

    - ``adaptiveNearWall``: adapt tau_w calculation based on y+ (default: True)
    - ``trackEvolution``: track tau_w evolution over time (default: True)
    - ``computeCfDistribution``: compute per-face Cf (default: True)
    - ``y_plus_threshold``: y+ threshold for switching model (default: 11.0)
    """

    def __init__(
        self,
        name: str = "wallShearStressEnhanced3",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._adaptive_near_wall: bool = self.config.get("adaptiveNearWall", True)
        self._track_evolution: bool = self.config.get("trackEvolution", True)
        self._compute_cf_dist: bool = self.config.get("computeCfDistribution", True)
        self._y_plus_threshold: float = float(self.config.get("y_plus_threshold", 11.0))

        # Storage
        self._cf_distribution_history: List[Dict[str, CfDistribution]] = []
        self._evolution: Dict[str, WSSEvolution] = {}

    # ------------------------------------------------------------------
    # 自适应近壁面处理
    # ------------------------------------------------------------------

    def _adaptive_wall_shear_stress(
        self, U_field, y_plus_est: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute wall shear stress with adaptive near-wall treatment.

        For y+ < threshold: use direct gradient (viscous sublayer)
        For y+ >= threshold: use log-law estimate

        Parameters
        ----------
        U_field : field
            Velocity field.
        y_plus_est : dict, optional
            Pre-computed y+ estimates per patch.

        Returns
        -------
        dict
            Per-patch wall shear stress tensors.
        """
        device = get_device()
        dtype = get_default_dtype()
        mesh = self._mesh

        if hasattr(U_field, "internal_field"):
            U_data = U_field.internal_field.to(device=device, dtype=dtype)
        else:
            U_data = U_field.to(device=device, dtype=dtype)

        tau_w_patches: Dict[str, torch.Tensor] = {}

        for patch_name in self._patches:
            patch_info = self._get_patch_info(patch_name)
            if patch_info is None:
                continue

            start_face = patch_info["startFace"]
            n_faces = patch_info["nFaces"]
            face_indices = torch.arange(
                start_face, start_face + n_faces, device=device, dtype=torch.long,
            )

            # Face geometry
            S = mesh.face_areas[face_indices].to(device=device, dtype=dtype)
            S_mag = S.norm(dim=1, keepdim=True)
            n = S / S_mag.clamp(min=_EPS)

            owner = mesh.owner[face_indices]
            x_P = mesh.cell_centres[owner].to(device=device, dtype=dtype)
            x_f = mesh.face_centres[face_indices].to(device=device, dtype=dtype)

            d_vec = x_f - x_P
            d_n = torch.abs(torch.sum(d_vec * n, dim=1, keepdim=True)).clamp(min=_EPS)

            # Velocity at owner cells
            U_P = U_data[owner]
            U_n = torch.sum(U_P * n, dim=1, keepdim=True)
            U_t = U_P - U_n * n

            if self._adaptive_near_wall and y_plus_est is not None:
                # Adaptive: switch between direct and log-law based on y+
                yp = y_plus_est.get(patch_name)
                if yp is not None:
                    yp = yp.to(device=device, dtype=dtype)
                    nu = self._mu / self._rho
                    u_tau_guess = yp * nu / d_n.squeeze().clamp(min=_EPS)

                    # Direct gradient for viscous sublayer (y+ < threshold)
                    tau_direct = self._mu * U_t / d_n

                    # Log-law estimate for log layer (y+ >= threshold)
                    kappa = self._kappa
                    B = self._B_plus
                    U_mag_t = U_t.norm(dim=1, keepdim=True).clamp(min=_EPS)
                    # Invert log-law: u+ = ln(E * y+)/kappa
                    yp_clamped = yp.unsqueeze(-1).clamp(min=_EPS)
                    u_plus_law = torch.log(yp_clamped * 9.8) / kappa  # E=9.8
                    tau_log = self._rho * (U_mag_t / u_plus_law.clamp(min=_EPS)).pow(2)

                    # Blend based on y+
                    blend = (yp.unsqueeze(-1) / self._y_plus_threshold).clamp(0.0, 1.0)
                    tau_w = (1.0 - blend) * tau_direct + blend * tau_log
                else:
                    tau_w = self._mu * U_t / d_n
            else:
                # Standard: direct gradient
                tau_w = self._mu * U_t / d_n

            tau_w_patches[patch_name] = tau_w

        return tau_w_patches

    # ------------------------------------------------------------------
    # Cf 分布
    # ------------------------------------------------------------------

    def _compute_cf_distribution(
        self, tau_w: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, CfDistribution]:
        """Compute friction coefficient distribution per patch."""
        q_ref = 0.5 * self._rho * self._u_ref ** 2 if self._u_ref > _EPS else 0.5 * self._rho
        cf_dist: Dict[str, CfDistribution] = {}

        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            cf = tau_mag / max(q_ref, _EPS)

            cf_np = cf.detach().cpu()
            n = cf_np.numel()

            cf_dist[patch_name] = CfDistribution(
                patch_name=patch_name,
                time=time,
                cf_mean=float(cf_np.mean().item()),
                cf_min=float(cf_np.min().item()),
                cf_max=float(cf_np.max().item()),
                cf_std=float(cf_np.std().item()) if n > 1 else 0.0,
                n_faces=n,
            )

        return cf_dist

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute wall shear stress v3 at current time step."""
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get("U")
        if U is None:
            logger.warning("Field 'U' required. Skipping.")
            return

        # Compute y+ estimate for adaptive model
        y_plus_est = None
        if self._adaptive_near_wall:
            y_plus_est = {}
            nu = self._mu / self._rho
            for patch_name in self._patches:
                # Simple y+ estimate: y+ = U * d / nu
                # Use approximate wall distance
                d_est = 0.01  # Default estimate
                U_ref = self._u_ref if self._u_ref > _EPS else 1.0
                y_plus_est[patch_name] = torch.full(
                    (1,), U_ref * d_est / max(nu, _EPS), dtype=torch.float64,
                )

        # Compute with adaptive near-wall
        if self._adaptive_near_wall:
            tau_w = self._adaptive_wall_shear_stress(U, y_plus_est)
        else:
            tau_w = self._compute_wall_shear_stress_corrected(U)

        self._tau_w_history.append({k: v.detach().cpu() for k, v in tau_w.items()})
        self._times.append(time)

        # Per-patch stats (reuse parent pattern)
        from pyfoam.postprocessing.wall_shear_stress_enhanced import WSSPatchStats
        stats: Dict[str, WSSPatchStats] = {}
        utau_patches: Dict[str, torch.Tensor] = {}
        cf_patches: Dict[str, float] = {}

        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)

            stats[patch_name] = WSSPatchStats(
                patch_name=patch_name,
                time=time,
                tau_w_avg=float(tau_mag.mean().item()),
                tau_w_max=float(tau_mag.max().item()),
                tau_w_min=float(tau_mag.min().item()),
                tau_w_std=float(tau_mag.std().item()),
                n_faces=tau_mag.shape[0],
            )

            if self._compute_utau:
                u_tau = torch.sqrt(tau_mag / self._rho)
                utau_patches[patch_name] = u_tau.detach().cpu()
                stats[patch_name].u_tau_avg = float(u_tau.mean().item())

            if self._compute_cf and self._u_ref > _EPS:
                q_ref = 0.5 * self._rho * self._u_ref ** 2
                cf = tau_mag / q_ref
                cf_patches[patch_name] = float(cf.mean().item())
                stats[patch_name].Cf_avg = cf_patches[patch_name]

        self._patch_stats_history.append(stats)
        if self._compute_utau:
            self._utau_history.append(utau_patches)
        if self._compute_cf:
            self._cf_history.append(cf_patches)

        # Distribution statistics (inherited)
        if self._compute_distribution_enabled:
            dist = self._compute_distribution_stats(tau_w, time)
            self._distribution_history.append(dist)

        # Cf distribution (v3)
        if self._compute_cf_dist:
            cf_dist = self._compute_cf_distribution(tau_w, time)
            self._cf_distribution_history.append(cf_dist)

        # Evolution tracking (v3)
        if self._track_evolution:
            for patch_name, tau in tau_w.items():
                if patch_name not in self._evolution:
                    self._evolution[patch_name] = WSSEvolution(patch_name=patch_name)

                evol = self._evolution[patch_name]
                evol.times.append(time)

                tau_mag = tau.norm(dim=1)
                evol.tau_w_mean_history.append(float(tau_mag.mean().item()))
                evol.tau_w_max_history.append(float(tau_mag.max().item()))

                if self._compute_utau:
                    u_tau = torch.sqrt(tau_mag / self._rho)
                    evol.u_tau_mean_history.append(float(u_tau.mean().item()))

                # Convergence rate
                if len(evol.tau_w_mean_history) >= 2:
                    dt = evol.times[-1] - evol.times[-2]
                    if dt > _EPS:
                        dtau = evol.tau_w_mean_history[-1] - evol.tau_w_mean_history[-2]
                        evol.convergence_rate = dtau / dt

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cf_distribution_history(self) -> List[Dict[str, CfDistribution]]:
        """Cf distribution history."""
        return self._cf_distribution_history

    @property
    def evolution(self) -> Dict[str, WSSEvolution]:
        """Time evolution tracking per patch."""
        return self._evolution

    def get_cf_distribution(self, patch_name: str) -> Optional[CfDistribution]:
        """Get latest Cf distribution for a patch."""
        if not self._cf_distribution_history:
            return None
        return self._cf_distribution_history[-1].get(patch_name)

    def get_evolution(self, patch_name: str) -> Optional[WSSEvolution]:
        """Get evolution tracking for a patch."""
        return self._evolution.get(patch_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write wall shear stress v3 data."""
        super().write()

        if self._output_path is None:
            return

        # Write Cf distribution
        if self._cf_distribution_history:
            cf_file = self._output_path / "cfDistribution.dat"
            with open(cf_file, "w") as f:
                f.write("# Time  patch  cf_mean  cf_min  cf_max  cf_std  n_faces\n")
                for i, t in enumerate(self._times):
                    if i < len(self._cf_distribution_history):
                        for pn, cfd in self._cf_distribution_history[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{cfd.cf_mean:.6e}  {cfd.cf_min:.6e}  "
                                f"{cfd.cf_max:.6e}  {cfd.cf_std:.6e}  "
                                f"{cfd.n_faces}\n"
                            )

        # Write evolution
        if self._evolution:
            evol_file = self._output_path / "wssEvolution.dat"
            with open(evol_file, "w") as f:
                f.write("# patch  n_steps  convergence_rate\n")
                for pn, evol in self._evolution.items():
                    f.write(
                        f"{pn}  {len(evol.times)}  "
                        f"{evol.convergence_rate:.6e}\n"
                    )

        logger.info("Wrote WallShearStressEnhanced3 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("wallShearStressEnhanced3", WallShearStressEnhanced3)
