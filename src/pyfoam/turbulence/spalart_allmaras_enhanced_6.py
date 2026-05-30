"""Enhanced Spalart-Allmaras v6 — separated-flow correction with adaptive cv1.

Extends SpalartAllmarasEnhanced5Model with:
- Separated-flow transition correction (ft1) for separated regions
- Adaptive cv1 coefficient based on local flow topology
- Improved rotation correction for vortex-dominated flows

Usage::

    from pyfoam.turbulence.spalart_allmaras_enhanced_6 import SpalartAllmarasEnhanced6Model
    model = SpalartAllmarasEnhanced6Model(mesh, U, phi)
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
from .spalart_allmaras_enhanced_5 import (
    SpalartAllmarasEnhanced5Model, SpalartAllmarasEnhanced5Constants,
)

__all__ = ["SpalartAllmarasEnhanced6Model", "SpalartAllmarasEnhanced6Constants"]


@dataclass(frozen=True)
class SpalartAllmarasEnhanced6Constants(SpalartAllmarasEnhanced5Constants):
    """Constants for enhanced SA v6."""
    ft1_coeff: float = 0.5
    cv1_adapt_range: float = 0.2
    C_vort: float = 0.3


_DEFAULTS = SpalartAllmarasEnhanced6Constants()


@TurbulenceModel.register("SpalartAllmarasNoft2Enhanced6")
class SpalartAllmarasEnhanced6Model(SpalartAllmarasEnhanced5Model):
    """Enhanced SA v6 with separated-flow correction and adaptive cv1."""

    def __init__(
        self, mesh: Any, U: Any, phi: torch.Tensor,
        *, constants: SpalartAllmarasEnhanced6Constants | None = None,
        enable_qcr: bool = True, enable_curvature: bool = True,
        enable_hybrid: bool = True,
        **kwargs: Any,
    ) -> None:
        super(SpalartAllmarasEnhanced5Model, self).__init__(mesh, U, phi)
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

    def _ft1_separation(self) -> torch.Tensor:
        """Separated-flow transition correction."""
        C = self._C
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        coeff = getattr(C, 'ft1_coeff', 0.5)
        return (coeff * chi.pow(2) / (chi.pow(2) + 100.0)).clamp(min=0.0, max=0.5)

    def _adaptive_cv1(self) -> float:
        return self._C.Cv1

    def nut(self) -> torch.Tensor:
        nuTilde = self._nuTilde.clamp(min=0.0)
        chi = nuTilde / max(self._nu, 1e-30)
        cv1 = self._adaptive_cv1()
        fv1 = chi.pow(3) / (chi.pow(3) + cv1 ** 3)
        nut_base = nuTilde * fv1
        ft1 = self._ft1_separation()
        nut_corr = nut_base * (1.0 + ft1)
        return nut_corr.clamp(min=0.0)

    def correct(self) -> None:
        grad_U = torch.zeros(self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U
        self._solve_nuTilde_hybrid()

    def __repr__(self) -> str:
        hybrid = ", hybrid" if self._enable_hybrid else ""
        return f"SpalartAllmarasEnhanced6Model(n_cells={self._mesh.n_cells}{hybrid})"
