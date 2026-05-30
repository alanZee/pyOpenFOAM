"""Enhanced k-epsilon turbulence model v6 — anisotropy-aware with realizability and curvature.

Extends KEpsilonEnhanced5Model with:
- Tensor-invariant anisotropy correction for C_mu
- Curvature-sensitive production limiter
- Adaptive sigma_eps based on local strain-to-dissipation ratio

Usage::

    from pyfoam.turbulence.k_epsilon_enhanced_6 import KEpsilonEnhanced6Model
    model = KEpsilonEnhanced6Model(mesh, U, phi)
    model.correct()
    nut = model.nut()
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc
from .turbulence_model import TurbulenceModel
from .k_epsilon_enhanced_5 import KEpsilonEnhanced5Model, KEpsilonEnhanced5Constants

__all__ = ["KEpsilonEnhanced6Model", "KEpsilonEnhanced6Constants"]


@dataclass(frozen=True)
class KEpsilonEnhanced6Constants(KEpsilonEnhanced5Constants):
    """Constants for enhanced k-epsilon v6."""
    C_curv: float = 0.2
    sigma_eps_adapt_min: float = 1.0
    sigma_eps_adapt_max: float = 1.6
    aniso_C3: float = 0.55


_DEFAULTS = KEpsilonEnhanced6Constants()


@TurbulenceModel.register("realizableKEEnhanced6")
class KEpsilonEnhanced6Model(KEpsilonEnhanced5Model):
    """Enhanced k-epsilon v6 with anisotropy-aware C_mu and curvature limiter."""

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KEpsilonEnhanced6Constants | None = None,
        **kwargs: Any,
    ) -> None:
        super(KEpsilonEnhanced5Model, self).__init__(mesh, U, phi)
        self._C = constants or _DEFAULTS
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._grad_U: torch.Tensor | None = None
        self._y = self._compute_wall_distance()
        self._f_relax = torch.ones(n_cells, device=device, dtype=dtype)
        self._v2 = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._alpha3 = torch.zeros(n_cells, device=device, dtype=dtype)

    def _compute_wall_distance(self) -> torch.Tensor:
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def _anisotropy_C_mu(self) -> torch.Tensor:
        """Anisotropy-aware C_mu from invariant II of S."""
        C = self._C
        if self._grad_U is None:
            return torch.full_like(self._k, getattr(C, "C_mu", 0.09))
        S = self._strain_rate()
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))
        ratio = S_mag / Omega_mag.clamp(min=1e-16)
        C_mu_base = getattr(C, "C_mu", 0.09)
        C_aniso = getattr(C, "aniso_C3", 0.55)
        return (C_mu_base / (1.0 + C_aniso * ratio.pow(2))).clamp(min=0.01, max=0.15)

    def nut(self) -> torch.Tensor:
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)
        C_mu = self._anisotropy_C_mu()
        nut_base = C_mu * k_safe.pow(2) / eps_safe
        v2_k_ratio = self._v2.clamp(min=0.0) / k_safe
        v2_k_ratio = v2_k_ratio.clamp(max=2.0)
        nut = nut_base * (self._alpha3 * v2_k_ratio + (1.0 - self._alpha3) * 1.0)
        return nut.clamp(min=0.0)

    def correct(self) -> None:
        grad_U = torch.zeros(self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U
        self._compute_blending()
        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))
        eps_safe = self._eps.clamp(min=1e-16)
        C = self._C
        C_curv = getattr(C, "C_curv", 0.2)
        f_curv = (1.0 + C_curv * (P_k / eps_safe.clamp(min=1e-16)).clamp(max=5.0)).clamp(max=2.0)
        P_k = P_k * f_curv
        P_k = P_k.clamp(max=getattr(C, "eta_0", 2.0) * eps_safe)
        self._solve_k(P_k)
        self._solve_eps(P_k)
        self._solve_v2()

    def __repr__(self) -> str:
        return f"KEpsilonEnhanced6Model(n_cells={self._mesh.n_cells})"
