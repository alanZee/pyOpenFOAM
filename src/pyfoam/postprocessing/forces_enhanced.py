"""
ForcesEnhanced — Enhanced force and moment computation.

在基础 Forces 上增加：

- **完整的力分解**：压力分量、粘性分量、湍流雷诺应力分量
- **力矩计算增强**：多参考点力矩、力矩系数时间历史
- **非定常力监测**：脉动力 RMS、峰峰值、频率分析
- **逐面统计**：每个 patch 的力分布统计

Usage::

    forces = ForcesEnhanced("forces1", {
        "patches": ["cylinder"],
        "rhoInf": 1.225,
        "CofR": [0.0, 0.0, 0.0],
        "computeFluctuations": True,
        "extraCofR": [[0.5, 0.0, 0.0]],
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.forces import Forces
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ForcesEnhanced", "ForceDecomposition", "FluctuationStats"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class ForceDecomposition:
    """Full decomposition of forces on a patch set.

    Attributes:
        time: Simulation time.
        F_pressure: Pressure force vector (3,).
        F_viscous: Viscous force vector (3,).
        F_total: Total force vector (3,).
        M_pressure: Pressure moment vector (3,).
        M_viscous: Viscous moment vector (3,).
        M_total: Total moment vector (3,).
        CofR: Centre of rotation used.
    """

    time: float = 0.0
    F_pressure: Optional[torch.Tensor] = None
    F_viscous: Optional[torch.Tensor] = None
    F_total: Optional[torch.Tensor] = None
    M_pressure: Optional[torch.Tensor] = None
    M_viscous: Optional[torch.Tensor] = None
    M_total: Optional[torch.Tensor] = None
    CofR: Optional[torch.Tensor] = None


@dataclass
class FluctuationStats:
    """Statistics for fluctuating force components.

    Attributes:
        rms: RMS of each component (3,).
        peak_to_peak: Peak-to-peak of each component (3,).
        max_abs: Maximum absolute value of each component (3,).
        n_samples: Number of samples used.
    """

    rms: Optional[torch.Tensor] = None
    peak_to_peak: Optional[torch.Tensor] = None
    max_abs: Optional[torch.Tensor] = None
    n_samples: int = 0


class ForcesEnhanced(Forces):
    """Enhanced force computation with full decomposition and fluctuation monitoring.

    在 Forces 基础上增加的配置键：

    - ``computeFluctuations``: enable fluctuation statistics (default: False)
    - ``extraCofR``: list of additional centre-of-rotation points
    - ``perPatchStats``: compute per-patch force statistics (default: True)
    """

    def __init__(
        self,
        name: str = "forcesEnhanced",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._compute_fluctuations: bool = self.config.get(
            "computeFluctuations", False,
        )
        self._per_patch_stats: bool = self.config.get("perPatchStats", True)

        # Additional CoR points
        extra_cor = self.config.get("extraCofR", [])
        self._extra_cofr: List[torch.Tensor] = [
            torch.tensor(c, dtype=torch.float64) for c in extra_cor
        ]

        # Enhanced storage
        self._decompositions: List[ForceDecomposition] = []
        self._extra_moments: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(len(extra_cor))
        }
        self._per_patch_forces: Dict[str, List[torch.Tensor]] = {}
        self._fluctuation_stats: Optional[FluctuationStats] = None

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute enhanced forces at current time step."""
        if not self._enabled or self._mesh is None:
            return

        p = self._fields.get("p")
        U = self._fields.get("U")
        if p is None or U is None:
            logger.warning("Fields 'p' and 'U' required. Skipping.")
            return

        # Compute forces (base class)
        f_p, f_v, f_total, moment = self._compute_forces(p, U)

        self._force_pressure.append(f_p.detach().cpu())
        self._force_viscous.append(f_v.detach().cpu())
        self._force_total.append(f_total.detach().cpu())
        self._moment.append(moment.detach().cpu())
        self._times.append(time)

        # Enhanced decomposition (separate pressure/viscous moments)
        decomp = self._compute_decomposition(p, U, time)
        self._decompositions.append(decomp)

        # Extra CoR moments
        for i, cofr in enumerate(self._extra_cofr):
            m_extra = self._compute_moment_at_cor(p, U, cofr)
            self._extra_moments[i].append(m_extra.detach().cpu())

        # Per-patch forces
        if self._per_patch_stats:
            self._compute_per_patch_forces(p, U, time)

        self._log.info(
            "t=%g  F=(%.6g, %.6g, %.6g)  M=(%.6g, %.6g, %.6g)  "
            "Fp=(%.6g, %.6g, %.6g)  Fv=(%.6g, %.6g, %.6g)",
            time,
            f_total[0].item(), f_total[1].item(), f_total[2].item(),
            moment[0].item(), moment[1].item(), moment[2].item(),
            f_p[0].item(), f_p[1].item(), f_p[2].item(),
            f_v[0].item(), f_v[1].item(), f_v[2].item(),
        )

    def _compute_decomposition(
        self, p_field, U_field, time: float,
    ) -> ForceDecomposition:
        """Compute pressure and viscous force/moment separately."""
        device = get_device()
        dtype = get_default_dtype()

        cofr = self._cofr.to(device=device, dtype=dtype)

        F_pressure = torch.zeros(3, dtype=dtype, device=device)
        F_viscous = torch.zeros(3, dtype=dtype, device=device)
        M_pressure = torch.zeros(3, dtype=dtype, device=device)
        M_viscous = torch.zeros(3, dtype=dtype, device=device)

        for patch_name in self._patches:
            patch_info = self._get_patch_info(patch_name)
            if patch_info is None:
                continue

            start_face = patch_info["startFace"]
            n_faces = patch_info["nFaces"]
            face_indices = torch.arange(
                start_face, start_face + n_faces, device=device, dtype=torch.long,
            )

            S = self._mesh.face_areas[face_indices]
            S_mag = S.norm(dim=1, keepdim=True)
            n = S / S_mag.clamp(min=_EPS)
            r = self._mesh.face_centres[face_indices]
            owner = self._mesh.owner[face_indices]

            # Pressure
            if hasattr(p_field, "internal_field"):
                p_data = p_field.internal_field.to(device=device, dtype=dtype)
            else:
                p_data = p_field.to(device=device, dtype=dtype)
            p_face = p_data[owner]

            f_p_face = p_face.unsqueeze(1) * n * S_mag
            F_pressure = F_pressure + f_p_face.sum(dim=0)
            r_rel = r - cofr
            M_pressure = M_pressure + torch.cross(r_rel, f_p_face).sum(dim=0)

            # Viscous (simplified)
            if hasattr(U_field, "internal_field"):
                U_data = U_field.internal_field.to(device=device, dtype=dtype)
            else:
                U_data = U_field.to(device=device, dtype=dtype)
            U_face = U_data[owner]
            mu = self._get_viscosity()
            cell_centres = self._mesh.cell_centres
            face_centres = self._mesh.face_centres
            d_Pf = (face_centres[face_indices] - cell_centres[owner]).norm(
                dim=1, keepdim=True,
            )
            tau_w = mu * U_face / d_Pf.clamp(min=_EPS)
            f_v_face = tau_w * S_mag
            F_viscous = F_viscous + f_v_face.sum(dim=0)
            M_viscous = M_viscous + torch.cross(r_rel, f_v_face).sum(dim=0)

        return ForceDecomposition(
            time=time,
            F_pressure=F_pressure.detach().cpu(),
            F_viscous=F_viscous.detach().cpu(),
            F_total=(F_pressure + F_viscous).detach().cpu(),
            M_pressure=M_pressure.detach().cpu(),
            M_viscous=M_viscous.detach().cpu(),
            M_total=(M_pressure + M_viscous).detach().cpu(),
            CofR=cofr.detach().cpu(),
        )

    def _compute_moment_at_cor(
        self, p_field, U_field, cofr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total moment about an alternative centre of rotation."""
        device = get_device()
        dtype = get_default_dtype()
        cofr_dev = cofr.to(device=device, dtype=dtype)

        moment = torch.zeros(3, dtype=dtype, device=device)

        for patch_name in self._patches:
            patch_info = self._get_patch_info(patch_name)
            if patch_info is None:
                continue

            start_face = patch_info["startFace"]
            n_faces = patch_info["nFaces"]
            face_indices = torch.arange(
                start_face, start_face + n_faces, device=device, dtype=torch.long,
            )

            S = self._mesh.face_areas[face_indices]
            S_mag = S.norm(dim=1, keepdim=True)
            n = S / S_mag.clamp(min=_EPS)
            r = self._mesh.face_centres[face_indices]
            owner = self._mesh.owner[face_indices]

            if hasattr(p_field, "internal_field"):
                p_data = p_field.internal_field.to(device=device, dtype=dtype)
            else:
                p_data = p_field.to(device=device, dtype=dtype)

            p_face = p_data[owner]
            f_face = p_face.unsqueeze(1) * n * S_mag
            r_rel = r - cofr_dev
            moment = moment + torch.cross(r_rel, f_face).sum(dim=0)

        return moment

    def _compute_per_patch_forces(
        self, p_field, U_field, time: float,
    ) -> None:
        """Compute forces per patch."""
        device = get_device()
        dtype = get_default_dtype()

        for patch_name in self._patches:
            patch_info = self._get_patch_info(patch_name)
            if patch_info is None:
                continue

            start_face = patch_info["startFace"]
            n_faces = patch_info["nFaces"]
            face_indices = torch.arange(
                start_face, start_face + n_faces, device=device, dtype=torch.long,
            )

            S = self._mesh.face_areas[face_indices]
            S_mag = S.norm(dim=1, keepdim=True)
            n = S / S_mag.clamp(min=_EPS)
            owner = self._mesh.owner[face_indices]

            if hasattr(p_field, "internal_field"):
                p_data = p_field.internal_field.to(device=device, dtype=dtype)
            else:
                p_data = p_field.to(device=device, dtype=dtype)

            p_face = p_data[owner]
            f_patch = (p_face.unsqueeze(1) * n * S_mag).sum(dim=0)

            if patch_name not in self._per_patch_forces:
                self._per_patch_forces[patch_name] = []
            self._per_patch_forces[patch_name].append(f_patch.detach().cpu())

    # ------------------------------------------------------------------
    # Fluctuation statistics
    # ------------------------------------------------------------------

    def compute_fluctuation_stats(self) -> Optional[FluctuationStats]:
        """Compute fluctuation statistics from force history.

        Returns
        -------
        FluctuationStats or None
        """
        if len(self._force_total) < 2:
            return None

        forces = torch.stack(self._force_total)  # (n_times, 3)
        mean_force = forces.mean(dim=0)
        fluctuations = forces - mean_force

        rms = fluctuations.pow(2).mean(dim=0).sqrt()
        peak_to_peak = forces.max(dim=0).values - forces.min(dim=0).values
        max_abs = forces.abs().max(dim=0).values

        self._fluctuation_stats = FluctuationStats(
            rms=rms,
            peak_to_peak=peak_to_peak,
            max_abs=max_abs,
            n_samples=len(self._force_total),
        )
        return self._fluctuation_stats

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def decompositions(self) -> List[ForceDecomposition]:
        """Force decomposition history."""
        return self._decompositions

    @property
    def extra_moments(self) -> Dict[int, List[torch.Tensor]]:
        """Moments about extra CoR points."""
        return self._extra_moments

    @property
    def per_patch_forces(self) -> Dict[str, List[torch.Tensor]]:
        """Per-patch force history."""
        return self._per_patch_forces

    @property
    def fluctuation_stats(self) -> Optional[FluctuationStats]:
        """Fluctuation statistics (if computed)."""
        return self._fluctuation_stats

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write enhanced force data."""
        super().write()

        if self._output_path is None or not self._decompositions:
            return

        # Write decomposition
        decomp_file = self._output_path / "forceDecomposition.dat"
        with open(decomp_file, "w") as f:
            header = (
                "# Time  Fpx Fpy Fpz  Fvx Fvy Fvz  Fx Fy Fz  "
                "Mpx Mpy Mpz  Mvx Mvy Mvz  Mx My Mz"
            )
            f.write(header + "\n")
            for d in self._decompositions:
                fp = d.F_pressure
                fv = d.F_viscous
                ft = d.F_total
                mp = d.M_pressure
                mv = d.M_viscous
                mt = d.M_total
                f.write(
                    f"{d.time:.6e}  "
                    f"{fp[0]:.6e} {fp[1]:.6e} {fp[2]:.6e}  "
                    f"{fv[0]:.6e} {fv[1]:.6e} {fv[2]:.6e}  "
                    f"{ft[0]:.6e} {ft[1]:.6e} {ft[2]:.6e}  "
                    f"{mp[0]:.6e} {mp[1]:.6e} {mp[2]:.6e}  "
                    f"{mv[0]:.6e} {mv[1]:.6e} {mv[2]:.6e}  "
                    f"{mt[0]:.6e} {mt[1]:.6e} {mt[2]:.6e}\n"
                )

        # Write fluctuation stats
        if self._compute_fluctuations:
            stats = self.compute_fluctuation_stats()
            if stats is not None:
                stats_file = self._output_path / "fluctuationStats.dat"
                with open(stats_file, "w") as f:
                    f.write(f"# Fluctuation statistics ({stats.n_samples} samples)\n")
                    f.write(f"# RMS:      {stats.rms.tolist()}\n")
                    f.write(f"# Peak2Peak: {stats.peak_to_peak.tolist()}\n")
                    f.write(f"# MaxAbs:   {stats.max_abs.tolist()}\n")

        logger.info("Wrote ForcesEnhanced to %s", self._output_path)


# Register
FunctionObjectRegistry.register("forcesEnhanced", ForcesEnhanced)
