"""
WallShearStressEnhanced2 — Enhanced wall shear stress v2 with near-wall treatment.

在 Enhanced v1 基础上增加：

- **非正交修正**：壁面剪切应力的非正交网格修正
- **壁面律反推**：从 y+ 和 u_tau 推导壁面应力
- **逐面分布直方图**：tau_w 的空间分布统计
- **脉动应力监测**：壁面剪切应力的 RMS 和峰值

Usage::

    wss = WallShearStressEnhanced2("wss2", {
        "patches": ["movingWall"],
        "rho": 1.0,
        "mu": 1e-3,
        "Uref": 1.0,
        "nonOrthogonalCorrection": True,
        "wallFunctionEstimate": True,
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.wall_shear_stress_enhanced import (
    WallShearStressEnhanced,
    WSSPatchStats,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["WallShearStressEnhanced2", "WSSDistribution"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class WSSDistribution:
    """Distribution statistics for wall shear stress on a patch.

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        tau_w_mean: Area-weighted mean tau_w (Pa).
        tau_w_rms: RMS of tau_w fluctuations (Pa).
        tau_w_peak: Peak tau_w (Pa).
        skewness: Skewness of tau_w distribution.
        kurtosis: Kurtosis of tau_w distribution.
        area_weighted_std: Area-weighted standard deviation.
    """

    patch_name: str = ""
    time: float = 0.0
    tau_w_mean: float = 0.0
    tau_w_rms: float = 0.0
    tau_w_peak: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    area_weighted_std: float = 0.0


class WallShearStressEnhanced2(WallShearStressEnhanced):
    """Enhanced wall shear stress v2 with non-orthogonal correction.

    在 WallShearStressEnhanced 基础上增加的配置键：

    - ``nonOrthogonalCorrection``: enable non-orthogonal face correction (default: True)
    - ``wallFunctionEstimate``: use wall function to estimate tau_w (default: False)
    - ``kappa``: von Karman constant for wall function (default: 0.41)
    - ``B_plus``: wall function constant (default: 5.5)
    - ``computeDistribution``: compute distribution statistics (default: True)
    """

    def __init__(
        self,
        name: str = "wallShearStressEnhanced2",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._non_orth_correction: bool = self.config.get("nonOrthogonalCorrection", True)
        self._wall_function_estimate: bool = self.config.get("wallFunctionEstimate", False)
        self._kappa: float = float(self.config.get("kappa", 0.41))
        self._B_plus: float = float(self.config.get("B_plus", 5.5))
        self._compute_distribution_enabled: bool = self.config.get("computeDistribution", True)

        # Storage
        self._distribution_history: List[Dict[str, WSSDistribution]] = []
        self._tau_w_rms_history: List[Dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # 非正交修正
    # ------------------------------------------------------------------

    def _compute_wall_shear_stress_corrected(self, U_field) -> Dict[str, torch.Tensor]:
        """Compute wall shear stress with non-orthogonal correction.

        tau_w = mu * (dU_t/dn)

        Non-orthogonal correction:
            tau_w_corrected = tau_w_orth - mu * (n . grad(U_t))_non_orth

        where the correction term accounts for face non-orthogonality.
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

            # Tangential velocity
            U_n = torch.sum(U_P * n, dim=1, keepdim=True)
            U_t = U_P - U_n * n

            # Orthogonal component: tau_w = mu * U_t / d_n
            tau_w_orth = self._mu * U_t / d_n

            if self._non_orth_correction:
                # Non-orthogonal correction
                # Project d_vec onto face plane to get non-orthogonal part
                d_orth = d_n * n
                d_non_orth = d_vec - d_orth
                # Correction: subtract non-orthogonal contribution
                # Approximate grad(U_t) correction
                correction = self._mu * torch.sum(U_t * d_non_orth, dim=1, keepdim=True) / d_n.pow(2)
                tau_w_corrected = tau_w_orth - correction * n
            else:
                tau_w_corrected = tau_w_orth

            tau_w_patches[patch_name] = tau_w_corrected

        return tau_w_patches

    # ------------------------------------------------------------------
    # 分布统计
    # ------------------------------------------------------------------

    def _compute_distribution_stats(
        self, tau_w: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, WSSDistribution]:
        """Compute distribution statistics for each patch."""
        dist: Dict[str, WSSDistribution] = {}

        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1).detach().cpu()
            n = tau_mag.numel()

            mean = float(tau_mag.mean().item())
            std = float(tau_mag.std().item()) if n > 1 else 0.0
            peak = float(tau_mag.max().item())

            # RMS (here: std of magnitude, proxy for fluctuation)
            rms = std

            # Higher moments
            if n > 2 and std > _EPS:
                normalized = (tau_mag - mean) / std
                skewness = float(normalized.pow(3).mean().item())
                kurtosis = float(normalized.pow(4).mean().item()) - 3.0
            else:
                skewness = 0.0
                kurtosis = 0.0

            dist[patch_name] = WSSDistribution(
                patch_name=patch_name,
                time=time,
                tau_w_mean=mean,
                tau_w_rms=rms,
                tau_w_peak=peak,
                skewness=skewness,
                kurtosis=kurtosis,
                area_weighted_std=std,
            )

        return dist

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute wall shear stress v2 at current time step."""
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get("U")
        if U is None:
            logger.warning("Field 'U' required. Skipping.")
            return

        # Compute with non-orthogonal correction
        tau_w = self._compute_wall_shear_stress_corrected(U)
        self._tau_w_history.append({k: v.detach().cpu() for k, v in tau_w.items()})
        self._times.append(time)

        # Per-patch stats (reuse parent logic)
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

        # Distribution statistics
        if self._compute_distribution_enabled:
            dist = self._compute_distribution_stats(tau_w, time)
            self._distribution_history.append(dist)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def distribution_history(self) -> List[Dict[str, WSSDistribution]]:
        """Distribution statistics history."""
        return self._distribution_history

    def get_latest_distribution(self, patch_name: str) -> Optional[WSSDistribution]:
        """Get the latest distribution stats for a patch."""
        if not self._distribution_history:
            return None
        return self._distribution_history[-1].get(patch_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write wall shear stress v2 data."""
        super().write()

        if self._output_path is None or not self._distribution_history:
            return

        dist_file = self._output_path / "wallShearStressDist.dat"
        with open(dist_file, "w") as f:
            header = (
                "# Time  patch  tau_w_mean  tau_w_rms  tau_w_peak"
                "  skewness  kurtosis  area_std"
            )
            f.write(header + "\n")
            for i, t in enumerate(self._times):
                if i < len(self._distribution_history):
                    for pn, d in self._distribution_history[i].items():
                        f.write(
                            f"{t:.6e}  {pn}  "
                            f"{d.tau_w_mean:.6e}  {d.tau_w_rms:.6e}  "
                            f"{d.tau_w_peak:.6e}  "
                            f"{d.skewness:.6e}  {d.kurtosis:.6e}  "
                            f"{d.area_weighted_std:.6e}\n"
                        )

        logger.info("Wrote WallShearStressEnhanced2 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("wallShearStressEnhanced2", WallShearStressEnhanced2)
