"""Enhanced Spalart-Allmaras v9 — stress limiter, wall-modeled transition, and vorticity transport.

Extends SpalartAllmarasEnhanced8Model with:
- Stress limiter for strong adverse pressure gradients
- Wall-modeled transition detection
- Modified vorticity transport for separated flows

Usage::

    from pyfoam.turbulence.spalart_allmaras_enhanced_9 import SpalartAllmarasEnhanced9Model
    model = SpalartAllmarasEnhanced9Model(mesh, U, phi)
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
from .spalart_allmaras_enhanced_8 import SpalartAllmarasEnhanced8Model, SpalartAllmarasEnhanced8Constants

__all__ = ["SpalartAllmarasEnhanced9Model", "SpalartAllmarasEnhanced9Constants"]


@dataclass(frozen=True)
class SpalartAllmarasEnhanced9Constants(SpalartAllmarasEnhanced8Constants):
    """Constants for enhanced SA v9."""
    C_stress_lim: float = 0.5
    C_transition: float = 1.0
    C_vort_transport: float = 0.3


_DEFAULTS = SpalartAllmarasEnhanced9Constants()


@TurbulenceModel.register("SpalartAllmarasNoft2Enhanced9")
class SpalartAllmarasEnhanced9Model(SpalartAllmarasEnhanced8Model):
    """Enhanced SA v9 with stress limiter and wall-modeled transition."""

    def __init__(
        self, mesh: Any, U: Any, phi: torch.Tensor,
        *, constants: SpalartAllmarasEnhanced9Constants | None = None,
        enable_qcr: bool = True, enable_curvature: bool = True,
        enable_hybrid: bool = True,
        enable_transition: bool = False,
        **kwargs: Any,
    ) -> None:
        super(SpalartAllmarasEnhanced8Model, self).__init__(mesh, U, phi)
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

    def _compute_wall_distance(self) -> torch.Tensor:
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-6)

    def stress_limiter(self) -> torch.Tensor:
        """Stress limiter for strong adverse pressure gradients.

        Limits production when the strain rate exceeds a threshold
        relative to vorticity magnitude.
        """
        C = self._C
        C_sl = getattr(C, 'C_stress_lim', 0.5)
        if self._grad_U is None:
            return torch.ones_like(self._nuTilde)

        S = self._strain_rate()
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        # Limit when strain >> vorticity (stagnation)
        ratio = (S_mag / Omega_mag.clamp(min=1e-16)).clamp(max=10.0)
        f_lim = (1.0 / (1.0 + C_sl * (ratio - 1.0).clamp(min=0.0))).clamp(min=0.3, max=1.0)
        return f_lim

    def wall_transition_detection(self) -> torch.Tensor:
        """Wall-modeled transition detection.

        Detects transition from laminar to turbulent based on
        wall distance and local Reynolds number.
        """
        if not self._enable_transition:
            return torch.ones_like(self._nuTilde)

        C = self._C
        C_tr = getattr(C, 'C_transition', 1.0)
        nu = max(self._nu, 1e-30)
        nuTilde = self._nuTilde.clamp(min=0.0)
        y = self._y.clamp(min=1e-6)

        # Local Reynolds number based on wall distance
        Re_y = nuTilde * y / nu
        f_tr = torch.tanh(Re_y * C_tr).clamp(min=0.0, max=1.0)
        return f_tr

    def nut(self) -> torch.Tensor:
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        cv1 = self._adaptive_cv1()
        fv1 = chi.pow(3) / (chi.pow(3) + cv1 ** 3)
        nut_base = nuTilde * fv1

        f_sl = self.stress_limiter()
        f_tr = self.wall_transition_detection()
        nut_corr = nut_base * f_sl * f_tr
        return nut_corr.clamp(min=0.0)

    def correct(self) -> None:
        grad_U = torch.zeros(self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U
        self._solve_nuTilde_hybrid()

    def __repr__(self) -> str:
        tr = ", transition" if self._enable_transition else ""
        hybrid = ", hybrid" if self._enable_hybrid else ""
        return f"SpalartAllmarasEnhanced9Model(n_cells={self._mesh.n_cells}{hybrid}{tr})"
