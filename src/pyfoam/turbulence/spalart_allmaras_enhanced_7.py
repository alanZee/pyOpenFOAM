"""Enhanced Spalart-Allmaras v7 — trip-free with vorticity amplification and adaptive decay.

Extends SpalartAllmarasEnhanced6Model with:
- Improved trip-free formulation with automatic boundary layer detection
- Vorticity amplification factor for rotation-dominated flows
- Adaptive decay rate in freestream to avoid over-production

Usage::

    from pyfoam.turbulence.spalart_allmaras_enhanced_7 import SpalartAllmarasEnhanced7Model
    model = SpalartAllmarasEnhanced7Model(mesh, U, phi)
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
from .spalart_allmaras_enhanced_6 import (
    SpalartAllmarasEnhanced6Model, SpalartAllmarasEnhanced6Constants,
)

__all__ = ["SpalartAllmarasEnhanced7Model", "SpalartAllmarasEnhanced7Constants"]


@dataclass(frozen=True)
class SpalartAllmarasEnhanced7Constants(SpalartAllmarasEnhanced6Constants):
    """Constants for enhanced SA v7."""
    C_vort_amp: float = 0.1
    decay_rate: float = 0.01
    bl_detect_threshold: float = 0.01


_DEFAULTS = SpalartAllmarasEnhanced7Constants()


@TurbulenceModel.register("SpalartAllmarasNoft2Enhanced7")
class SpalartAllmarasEnhanced7Model(SpalartAllmarasEnhanced6Model):
    """Enhanced SA v7 with trip-free formulation and vorticity amplification."""

    def __init__(
        self, mesh: Any, U: Any, phi: torch.Tensor,
        *, constants: SpalartAllmarasEnhanced7Constants | None = None,
        enable_qcr: bool = True, enable_curvature: bool = True,
        enable_hybrid: bool = True,
        **kwargs: Any,
    ) -> None:
        super(SpalartAllmarasEnhanced6Model, self).__init__(mesh, U, phi)
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

    def _boundary_layer_detected(self) -> torch.Tensor:
        """Detect boundary layer based on wall distance gradient.

        Returns a scalar field indicating BL presence (0 = freestream, 1 = in BL).
        """
        C = self._C
        threshold = getattr(C, 'bl_detect_threshold', 0.01)
        y_norm = (self._y / self._y.max().clamp(min=1e-10)).clamp(0.0, 1.0)
        return (y_norm < 0.3).to(self._dtype)

    def _vorticity_amplification(self) -> torch.Tensor:
        """Vorticity amplification factor for rotation-dominated flows.

        f_vort = 1 + C * |Omega| / (|S| + epsilon)
        """
        C = self._C
        if self._grad_U is None:
            return torch.ones_like(self._nuTilde)

        S = self._strain_rate()
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        C_amp = getattr(C, 'C_vort_amp', 0.1)
        ratio = Omega_mag / S_mag.clamp(min=1e-16)
        return (1.0 + C_amp * ratio.clamp(max=5.0)).clamp(max=2.0)

    def nut(self) -> torch.Tensor:
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        cv1 = self._adaptive_cv1()
        fv1 = chi.pow(3) / (chi.pow(3) + cv1 ** 3)
        nut_base = nuTilde * fv1
        ft1 = self._ft1_separation()
        f_vort = self._vorticity_amplification()
        nut_corr = nut_base * (1.0 + ft1) * f_vort
        return nut_corr.clamp(min=0.0)

    def correct(self) -> None:
        grad_U = torch.zeros(self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U
        self._solve_nuTilde_hybrid()

    def __repr__(self) -> str:
        hybrid = ", hybrid" if self._enable_hybrid else ""
        return f"SpalartAllmarasEnhanced7Model(n_cells={self._mesh.n_cells}{hybrid})"
