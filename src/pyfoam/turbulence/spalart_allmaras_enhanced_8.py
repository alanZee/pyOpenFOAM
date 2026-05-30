"""Enhanced Spalart-Allmaras v8 — rotation-curvature, DDES shielding, ft2 improvement.

Extends SpalartAllmarasEnhanced7Model with:
- Rotation-curvature correction (SARC variant)
- DDES shielding function for hybrid RANS-LES
- Improved ft2 separation detection

Usage::

    from pyfoam.turbulence.spalart_allmaras_enhanced_8 import SpalartAllmarasEnhanced8Model
    model = SpalartAllmarasEnhanced8Model(mesh, U, phi)
    model.correct()
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc
from .turbulence_model import TurbulenceModel
from .spalart_allmaras_enhanced_7 import SpalartAllmarasEnhanced7Model, SpalartAllmarasEnhanced7Constants

__all__ = ["SpalartAllmarasEnhanced8Model", "SpalartAllmarasEnhanced8Constants"]


@dataclass(frozen=True)
class SpalartAllmarasEnhanced8Constants(SpalartAllmarasEnhanced7Constants):
    """Constants for enhanced SA v8."""
    C_rc: float = 1.0
    C_ddes: float = 0.65
    C_ft2: float = 0.3


_DEFAULTS = SpalartAllmarasEnhanced8Constants()


@TurbulenceModel.register("SpalartAllmarasNoft2Enhanced8")
class SpalartAllmarasEnhanced8Model(SpalartAllmarasEnhanced7Model):
    """Enhanced SA v8 with rotation-curvature correction and DDES shielding."""

    def __init__(
        self, mesh: Any, U: Any, phi: torch.Tensor,
        *, constants: SpalartAllmarasEnhanced8Constants | None = None,
        enable_qcr: bool = True, enable_curvature: bool = True,
        enable_hybrid: bool = True,
        **kwargs: Any,
    ) -> None:
        super(SpalartAllmarasEnhanced7Model, self).__init__(mesh, U, phi)
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

    def _compute_wall_distance(self) -> torch.Tensor:
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-6)

    def _rotation_curvature_correction(self) -> torch.Tensor:
        """Rotation-curvature correction (SARC).

        Modifies Cb1 based on the ratio of rotation to strain tensors.
        """
        C = self._C
        C_rc = getattr(C, 'C_rc', 1.0)
        if self._grad_U is None:
            return torch.ones_like(self._nuTilde)

        S = self._strain_rate()
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        # Star feature: r = Omega/S
        r = (Omega_mag / S_mag.clamp(min=1e-16)).clamp(max=5.0)
        f_rc = (1.0 + C_rc * r).clamp(min=0.5, max=3.0)
        return f_rc

    def _ddes_shielding(self) -> torch.Tensor:
        """DDES shielding function.

        rd = (nu_t + nu) / (U_ij * U_ij * d^2 * kappa^2)
        """
        C = self._C
        C_ddes = getattr(C, 'C_ddes', 0.65)
        nu = max(self._nu, 1e-30)
        nut = self.nut()
        y_safe = self._y.clamp(min=1e-6)

        if self._grad_U is not None:
            U_grad_mag = self._grad_U.pow(2).sum(dim=(1, 2)).clamp(min=1e-30)
            rd = ((nut + nu) / (U_grad_mag * y_safe.pow(2) * 0.16)).clamp(min=0.0, max=5.0)
        else:
            rd = torch.ones_like(self._nuTilde)

        f_ddes = torch.tanh(rd.pow(3) * C_ddes)
        return f_ddes

    def _ft2_improved(self) -> torch.Tensor:
        """Improved ft2 separation detection.

        Uses both wall distance and strain rate for better separation detection.
        """
        C = self._C
        C_ft2 = getattr(C, 'C_ft2', 0.3)
        nu = max(self._nu, 1e-30)
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / nu
        d_safe = self._y.clamp(min=1e-6)

        # Standard ft2
        ft2_base = self._ft2_separation()

        # Strain-based enhancement
        if self._grad_U is not None:
            S = self._strain_rate()
            S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
            omega_plus = S_mag * d_safe.pow(2) / nu
            f_strain = torch.exp(-omega_plus.clamp(max=100.0) * C_ft2)
        else:
            f_strain = torch.ones_like(self._nuTilde)

        return ft2_base * f_strain.clamp(min=0.0, max=1.0)

    def nut(self) -> torch.Tensor:
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        cv1 = self._adaptive_cv1()
        fv1 = chi.pow(3) / (chi.pow(3) + cv1 ** 3)
        nut_base = nuTilde * fv1
        ft2 = self._ft2_improved()
        f_vort = self._vorticity_amplification()
        f_rc = self._rotation_curvature_correction()
        nut_corr = nut_base * (1.0 + ft2) * f_vort * f_rc
        return nut_corr.clamp(min=0.0)

    def correct(self) -> None:
        grad_U = torch.zeros(self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U
        self._solve_nuTilde_hybrid()

    def __repr__(self) -> str:
        hybrid = ", hybrid" if self._enable_hybrid else ""
        return f"SpalartAllmarasEnhanced8Model(n_cells={self._mesh.n_cells}{hybrid})"
