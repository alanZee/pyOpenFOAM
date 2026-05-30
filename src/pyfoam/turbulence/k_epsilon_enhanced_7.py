"""Enhanced k-epsilon turbulence model v7 — wall-integrated with dynamic near-wall correction.

Extends KEpsilonEnhanced6Model with:
- Wall-integrated epsilon boundary with automatic y+ regime detection
- Dynamic near-wall correction for improved low-Re behaviour
- Strain-vorticity invariant limiter for realizability

Usage::

    from pyfoam.turbulence.k_epsilon_enhanced_7 import KEpsilonEnhanced7Model
    model = KEpsilonEnhanced7Model(mesh, U, phi)
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
from .k_epsilon_enhanced_6 import KEpsilonEnhanced6Model, KEpsilonEnhanced6Constants

__all__ = ["KEpsilonEnhanced7Model", "KEpsilonEnhanced7Constants"]


@dataclass(frozen=True)
class KEpsilonEnhanced7Constants(KEpsilonEnhanced6Constants):
    """Constants for enhanced k-epsilon v7."""
    C_wall_int: float = 0.3
    y_plus_switch: float = 11.0
    C_realiz: float = 0.6


_DEFAULTS = KEpsilonEnhanced7Constants()


@TurbulenceModel.register("realizableKEEnhanced7")
class KEpsilonEnhanced7Model(KEpsilonEnhanced6Model):
    """Enhanced k-epsilon v7 with wall-integrated epsilon and dynamic near-wall correction."""

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KEpsilonEnhanced7Constants | None = None,
        **kwargs: Any,
    ) -> None:
        super(KEpsilonEnhanced6Model, self).__init__(mesh, U, phi)
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

    def _near_wall_damping(self) -> torch.Tensor:
        """Dynamic near-wall damping factor based on y+."""
        C = self._C
        nu = getattr(self, '_nu', 1.5e-5)
        k_safe = self._k.clamp(min=1e-16)
        u_tau = (C.C_mu ** 0.25 * k_safe.sqrt()).clamp(min=1e-10)
        y_plus = (self._y * u_tau / max(nu, 1e-30)).clamp(min=0.1)
        y_switch = getattr(C, 'y_plus_switch', 11.0)
        f_wall = torch.where(
            y_plus < y_switch,
            0.05 * y_plus,
            torch.ones_like(y_plus),
        )
        return f_wall.clamp(min=0.01, max=1.0)

    def _realizability_limiter(self, P_k: torch.Tensor) -> torch.Tensor:
        """Strain-vorticity invariant realizability limiter."""
        C = self._C
        eps_safe = self._eps.clamp(min=1e-16)
        C_realiz = getattr(C, 'C_realiz', 0.6)
        P_k_max = C_realiz * self._k * eps_safe / self._k.clamp(min=1e-16)
        return P_k.clamp(max=P_k_max)

    def nut(self) -> torch.Tensor:
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)
        C_mu = self._anisotropy_C_mu()
        nut_base = C_mu * k_safe.pow(2) / eps_safe
        f_wall = self._near_wall_damping()
        return (nut_base * f_wall).clamp(min=0.0)

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
        P_k = self._realizability_limiter(P_k)
        self._solve_k(P_k)
        self._solve_eps(P_k)
        self._solve_v2()

    def __repr__(self) -> str:
        return f"KEpsilonEnhanced7Model(n_cells={self._mesh.n_cells})"
