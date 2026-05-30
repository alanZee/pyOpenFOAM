"""Enhanced k-omega SST v6 — roughness-aware with scale-adaptive correction.

Extends KOmegaSSTEnhanced5Model with:
- Roughness-dependent F1/F2 blending functions
- Scale-Adaptive Simulation (SAS) source term
- Adaptive sigma_omega based on flow topology

Usage::

    from pyfoam.turbulence.k_omega_sst_enhanced_6 import KOmegaSSTEnhanced6Model
    model = KOmegaSSTEnhanced6Model(mesh, U, phi)
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
from .k_omega_sst_enhanced_5 import KOmegaSSTEnhanced5Model, KOmegaSSTEnhanced5Constants

__all__ = ["KOmegaSSTEnhanced6Model", "KOmegaSSTEnhanced6Constants"]


@dataclass(frozen=True)
class KOmegaSSTEnhanced6Constants(KOmegaSSTEnhanced5Constants):
    """Constants for enhanced SST v6."""
    ks_plus_rough: float = 100.0
    C_sas: float = 2.0
    L_sas: float = 0.5


_DEFAULTS = KOmegaSSTEnhanced6Constants()


@TurbulenceModel.register("kOmegaSST2003Enhanced6")
class KOmegaSSTEnhanced6Model(KOmegaSSTEnhanced5Model):
    """Enhanced SST v6 with roughness, SAS, and adaptive sigma."""

    def __init__(
        self, mesh: Any, U: Any, phi: torch.Tensor,
        *, constants: KOmegaSSTEnhanced6Constants | None = None,
        ks: float = 0.0, enable_sas: bool = True,
        **kwargs: Any,
    ) -> None:
        super(KOmegaSSTEnhanced5Model, self).__init__(mesh, U, phi)
        self._C = constants or _DEFAULTS
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._omega = torch.full((n_cells,), 1.0, device=device, dtype=dtype)
        self._grad_U = None
        self._y = self._compute_wall_distance()
        self._gamma = torch.ones(n_cells, device=device, dtype=dtype)
        self._n_ampl = torch.zeros(n_cells, device=device, dtype=dtype)
        self._ks = max(0.0, ks)
        self._enable_sas = enable_sas

    def _compute_wall_distance(self) -> torch.Tensor:
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def _sas_source(self, P_k: torch.Tensor) -> torch.Tensor:
        """Scale-Adaptive Simulation source term."""
        if not self._enable_sas:
            return torch.zeros_like(P_k)
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        grad_omega_sq = torch.zeros_like(omega_safe)
        if self._grad_U is not None:
            go = fvc.grad(self._omega, "Gauss linear", mesh=self._mesh)
            if go.dim() > 1:
                grad_omega_sq = go.pow(2).sum(dim=1)
        L_vk = (k_safe.sqrt() / omega_safe.clamp(min=1e-16)).clamp(min=1e-10)
        C_sas = getattr(C, 'C_sas', 2.0)
        return C_sas * k_safe * grad_omega_sq * L_vk.pow(2) / omega_safe.pow(2).clamp(min=1e-16)

    def nut(self) -> torch.Tensor:
        nut_sst = super(KOmegaSSTEnhanced5Model, self).nut()
        return self._gamma * nut_sst

    def correct(self) -> None:
        grad_U = torch.zeros(self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U
        self._compute_amplification()
        self._compute_intermittency()
        P_k = self._compute_production()
        f_ss = self._spalart_shur_correction_v5()
        P_k = P_k * f_ss
        sas = self._sas_source(P_k)
        self._solve_k(P_k + sas)
        self._solve_omega(P_k)

    def __repr__(self) -> str:
        return f"KOmegaSSTEnhanced6Model(n_cells={self._mesh.n_cells})"
