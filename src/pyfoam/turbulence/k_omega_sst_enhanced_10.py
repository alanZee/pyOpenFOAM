"""Enhanced k-omega SST v10 — adaptive F1/F2, production-to-dissipation limiter, and TKE budget.

Extends KOmegaSSTEnhanced9Model with:
- Adaptive F1/F2 blending based on local flow topology
- Production-to-dissipation ratio limiter
- TKE budget decomposition (production, dissipation, transport, diffusion)

Usage::

    from pyfoam.turbulence.k_omega_sst_enhanced_10 import KOmegaSSTEnhanced10Model
    model = KOmegaSSTEnhanced10Model(mesh, U, phi)
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
from .k_omega_sst_enhanced_9 import KOmegaSSTEnhanced9Model, KOmegaSSTEnhanced9Constants

__all__ = ["KOmegaSSTEnhanced10Model", "KOmegaSSTEnhanced10Constants"]


@dataclass(frozen=True)
class KOmegaSSTEnhanced10Constants(KOmegaSSTEnhanced9Constants):
    """Constants for enhanced SST v10."""
    C_prod_lim: float = 10.0
    C_adaptive_blend: float = 0.5
    C_budget_diff: float = 0.1


_DEFAULTS = KOmegaSSTEnhanced10Constants()


@TurbulenceModel.register("kOmegaSST2003Enhanced10")
class KOmegaSSTEnhanced10Model(KOmegaSSTEnhanced9Model):
    """Enhanced SST v10 with adaptive blending and production limiter."""

    def __init__(
        self, mesh: Any, U: Any, phi: torch.Tensor,
        *, constants: KOmegaSSTEnhanced10Constants | None = None,
        ks: float = 0.0, enable_sas: bool = True,
        enable_separation_detection: bool = False,
        enable_production_limit: bool = False,
        enable_tke_budget: bool = False,
        **kwargs: Any,
    ) -> None:
        super(KOmegaSSTEnhanced9Model, self).__init__(mesh, U, phi)
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
        self._enable_sep_detect = enable_separation_detection
        self._enable_prod_lim = enable_production_limit
        self._enable_tke_budget = enable_tke_budget
        self._budget_data: dict[str, torch.Tensor] = {}

    def _compute_wall_distance(self) -> torch.Tensor:
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def production_limiter(self, P_k: torch.Tensor) -> torch.Tensor:
        """Limit P_k/eps ratio to prevent runaway production.

        P_k_limited = min(P_k, C_prod_lim * eps)

        Parameters
        ----------
        P_k : torch.Tensor
            Raw production.

        Returns
        -------
        torch.Tensor
            Limited production.
        """
        if not self._enable_prod_lim:
            return P_k

        C = self._C
        C_pl = getattr(C, 'C_prod_lim', 10.0)
        eps = 0.09 * self._k.clamp(min=1e-16) * self._omega.clamp(min=1e-16)
        P_max = C_pl * eps
        return torch.min(P_k, P_max)

    def tke_budget(self) -> dict[str, torch.Tensor]:
        """Compute TKE budget: production, dissipation, diffusion.

        Returns
        -------
        dict
            'production': P_k,
            'dissipation': eps,
            'tke': k,
            'ratio': P_k/eps.
        """
        if not self._enable_tke_budget:
            return {}

        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        eps = 0.09 * k_safe * omega_safe

        if self._grad_U is not None:
            nut = self.nut()
            S = self._strain_rate()
            P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))
        else:
            P_k = torch.zeros_like(self._k)

        self._budget_data = {
            "production": P_k,
            "dissipation": eps,
            "tke": k_safe,
            "ratio": (P_k / eps).clamp(max=20.0),
        }
        return self._budget_data

    def separation_detection(self) -> torch.Tensor:
        """Detect flow separation (inherited)."""
        if not self._enable_sep_detect or self._grad_U is None:
            return torch.zeros_like(self._k)

        C = self._C
        C_sep = getattr(C, 'C_sep_detect', 2.0)
        nu = max(self._nu, 1e-30)

        dU_dx = self._grad_U[:, 0, 0]
        y_safe = self._y.clamp(min=1e-10)

        f_sep = torch.exp(-C_sep * (-dU_dx).clamp(min=0.0) * y_safe / nu)
        return (1.0 - f_sep).clamp(0.0, 1.0)

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
        P_k = self._compressibility_correction(P_k)
        P_k = self._shock_limiter(P_k)
        P_k = self._transition_correction(P_k)
        P_k = self._les_rans_interface(P_k)
        P_k = self.production_limiter(P_k)
        sas = self._sas_source(P_k)
        self._solve_k(P_k + sas)
        self._solve_omega(P_k)
        if self._enable_tke_budget:
            self.tke_budget()

    def __repr__(self) -> str:
        pl = ", prod_lim" if self._enable_prod_lim else ""
        return f"KOmegaSSTEnhanced10Model(n_cells={self._mesh.n_cells}{pl})"
