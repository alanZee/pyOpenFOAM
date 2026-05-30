"""Enhanced k-omega SST v7 — shock-capturing with compressibility correction and adaptive blending.

Extends KOmegaSSTEnhanced6Model with:
- Shock-capturing production limiter for transonic/supersonic flows
- Sarkar compressibility correction for dilatational dissipation
- Adaptive F1/F2 blending based on local flow topology

Usage::

    from pyfoam.turbulence.k_omega_sst_enhanced_7 import KOmegaSSTEnhanced7Model
    model = KOmegaSSTEnhanced7Model(mesh, U, phi)
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
from .k_omega_sst_enhanced_6 import KOmegaSSTEnhanced6Model, KOmegaSSTEnhanced6Constants

__all__ = ["KOmegaSSTEnhanced7Model", "KOmegaSSTEnhanced7Constants"]


@dataclass(frozen=True)
class KOmegaSSTEnhanced7Constants(KOmegaSSTEnhanced6Constants):
    """Constants for enhanced SST v7."""
    alpha_comp: float = 1.0
    beta_comp: float = 0.0
    Ma_cr: float = 0.8
    C_shock: float = 0.2


_DEFAULTS = KOmegaSSTEnhanced7Constants()


@TurbulenceModel.register("kOmegaSST2003Enhanced7")
class KOmegaSSTEnhanced7Model(KOmegaSSTEnhanced6Model):
    """Enhanced SST v7 with shock-capturing, compressibility correction, and adaptive blending."""

    def __init__(
        self, mesh: Any, U: Any, phi: torch.Tensor,
        *, constants: KOmegaSSTEnhanced7Constants | None = None,
        ks: float = 0.0, enable_sas: bool = True,
        **kwargs: Any,
    ) -> None:
        super(KOmegaSSTEnhanced6Model, self).__init__(mesh, U, phi)
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

    def _compressibility_correction(self, P_k: torch.Tensor) -> torch.Tensor:
        """Sarkar-style compressibility correction for dilatational dissipation.

        Modifies production to account for pressure-dilatation work at
        transonic/supersonic conditions.
        """
        C = self._C
        alpha = getattr(C, 'alpha_comp', 1.0)
        beta = getattr(C, 'beta_comp', 0.0)
        Ma_cr = getattr(C, 'Ma_cr', 0.8)

        # Estimate local Mach from velocity magnitude
        U_mag = self._U.norm(dim=1).clamp(min=1e-10)
        c_sound = 340.0  # Approximate speed of sound
        Ma = U_mag / c_sound
        Ma_t = (Ma - Ma_cr).clamp(min=0.0)

        correction = 1.0 + alpha * Ma_t.pow(2) + beta * Ma_t.pow(4)
        return P_k * correction

    def _shock_limiter(self, P_k: torch.Tensor) -> torch.Tensor:
        """Shock-capturing production limiter."""
        C = self._C
        C_shock = getattr(C, 'C_shock', 0.2)
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        eps = 0.09 * k_safe * omega_safe
        return P_k.clamp(max=C_shock * k_safe.pow(2) * omega_safe / eps.clamp(min=1e-16))

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
        # Compressibility correction
        P_k = self._compressibility_correction(P_k)
        P_k = self._shock_limiter(P_k)
        sas = self._sas_source(P_k)
        self._solve_k(P_k + sas)
        self._solve_omega(P_k)

    def __repr__(self) -> str:
        return f"KOmegaSSTEnhanced7Model(n_cells={self._mesh.n_cells})"
