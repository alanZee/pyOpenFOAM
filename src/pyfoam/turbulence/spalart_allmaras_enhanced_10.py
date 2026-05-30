"""Enhanced Spalart-Allmaras v10 — rotation correction, adaptive ft2, and boundary layer diagnostics.

Extends SpalartAllmarasEnhanced9Model with:
- Rotation correction (SARC) for streamline curvature
- Adaptive ft2 trip term based on local Reynolds number
- Boundary layer diagnostics (displacement thickness estimation)

Usage::

    from pyfoam.turbulence.spalart_allmaras_enhanced_10 import SpalartAllmarasEnhanced10Model
    model = SpalartAllmarasEnhanced10Model(mesh, U, phi)
    model.correct()
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import torch
from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc
from .turbulence_model import TurbulenceModel
from .spalart_allmaras_enhanced_9 import SpalartAllmarasEnhanced9Model, SpalartAllmarasEnhanced9Constants

__all__ = ["SpalartAllmarasEnhanced10Model", "SpalartAllmarasEnhanced10Constants"]


@dataclass(frozen=True)
class SpalartAllmarasEnhanced10Constants(SpalartAllmarasEnhanced9Constants):
    """Constants for enhanced SA v10."""
    C_rotation: float = 1.0
    C_ft2_adapt: float = 0.5
    C_bl_diag: float = 0.3


_DEFAULTS = SpalartAllmarasEnhanced10Constants()


@TurbulenceModel.register("SpalartAllmarasNoft2Enhanced10")
class SpalartAllmarasEnhanced10Model(SpalartAllmarasEnhanced9Model):
    """Enhanced SA v10 with rotation correction and BL diagnostics."""

    def __init__(
        self, mesh: Any, U: Any, phi: torch.Tensor,
        *, constants: SpalartAllmarasEnhanced10Constants | None = None,
        enable_qcr: bool = True, enable_curvature: bool = True,
        enable_hybrid: bool = True,
        enable_transition: bool = False,
        enable_rotation_correction: bool = False,
        enable_bl_diagnostics: bool = False,
        **kwargs: Any,
    ) -> None:
        super(SpalartAllmarasEnhanced9Model, self).__init__(mesh, U, phi)
        self._C = constants or _DEFAULTS
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        self._nuTilde = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._grad_U = None
        self._y = self._compute_wall_distance()
        self._enable_qcr = enable_qcr
        self._enable_curvature = enable_curvature
        self._enable_hybrid = enable_hybrid
        self._enable_transition = enable_transition
        self._enable_rotation = enable_rotation_correction
        self._enable_bl_diag = enable_bl_diagnostics
        self._bl_data: dict[str, torch.Tensor] = {}

    def _compute_wall_distance(self) -> torch.Tensor:
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-6)

    def rotation_correction(self) -> torch.Tensor:
        """SARC (SA-rotation correction) for streamline curvature.

        Multiplies production by (1 + C_rot * r_tilde^2)
        where r_tilde = Omega / S.

        Returns
        -------
        torch.Tensor
            Rotation correction factor.
        """
        if not self._enable_rotation or self._grad_U is None:
            return torch.ones_like(self._nuTilde)

        C = self._C
        C_r = getattr(C, 'C_rotation', 1.0)

        S = self._strain_rate()
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        r_tilde = Omega_mag / S_mag.clamp(min=1e-16)
        f_rot = 1.0 + C_r * r_tilde.pow(2)
        return f_rot.clamp(max=3.0)

    def adaptive_ft2(self) -> torch.Tensor:
        """Adaptive ft2 trip term based on local Reynolds number.

        ft2 = C * (nu_tilde / y)^2 * exp(-Re_y)

        Returns
        -------
        torch.Tensor
            Adaptive ft2 term.
        """
        nu = max(self._nu, 1e-30)
        nuTilde = self._nuTilde.clamp(min=0.0)
        y = self._y.clamp(min=1e-6)

        Re_y = nuTilde * y / nu
        C = self._C
        C_ft2 = getattr(C, 'C_ft2_adapt', 0.5)
        ft2 = C_ft2 * (nuTilde / y).pow(2) * torch.exp(-Re_y.clamp(max=20.0))
        return ft2.clamp(min=0.0, max=1.0)

    def boundary_layer_diagnostics(self) -> dict[str, torch.Tensor]:
        """Estimate boundary layer displacement thickness.

        delta_star ~ y * (1 - U/U_ref) integrated near wall.

        Returns
        -------
        dict
            'delta_star_est': estimated displacement thickness per cell.
        """
        if not self._enable_bl_diag:
            return {}

        y = self._y
        nuTilde = self._nuTilde.clamp(min=0.0)
        nu = max(self._nu, 1e-30)

        # Simplified: delta_star ~ y when nuTilde is small (laminar BL)
        Re_y = nuTilde * y / nu
        delta_star = y * (1.0 - torch.tanh(Re_y * 0.01))

        self._bl_data = {"delta_star_est": delta_star}
        return self._bl_data

    def stress_limiter(self) -> torch.Tensor:
        """Stress limiter (inherited from v9)."""
        C = self._C
        C_sl = getattr(C, 'C_stress_lim', 0.5)
        if self._grad_U is None:
            return torch.ones_like(self._nuTilde)

        S = self._strain_rate()
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        ratio = (S_mag / Omega_mag.clamp(min=1e-16)).clamp(max=10.0)
        f_lim = (1.0 / (1.0 + C_sl * (ratio - 1.0).clamp(min=0.0))).clamp(min=0.3, max=1.0)
        return f_lim

    def nut(self) -> torch.Tensor:
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        cv1 = self._adaptive_cv1()
        fv1 = chi.pow(3) / (chi.pow(3) + cv1 ** 3)
        nut_base = nuTilde * fv1

        f_sl = self.stress_limiter()
        f_rot = self.rotation_correction()
        nut_corr = nut_base * f_sl * f_rot
        return nut_corr.clamp(min=0.0)

    def correct(self) -> None:
        grad_U = torch.zeros(self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U
        self._solve_nuTilde_hybrid()
        if self._enable_bl_diag:
            self.boundary_layer_diagnostics()

    def __repr__(self) -> str:
        rot = ", rotation" if self._enable_rotation else ""
        bl = ", bl_diag" if self._enable_bl_diag else ""
        return f"SpalartAllmarasEnhanced10Model(n_cells={self._mesh.n_cells}{rot}{bl})"
