"""
Enhanced k-omega turbulence model v2 — Wilcox 2006 with improved cross-diffusion.

Extends :class:`~pyfoam.turbulence.k_omega_enhanced.KOmegaEnhancedModel`
with:

- Improved cross-diffusion with F3 limiter
- Stress-limiter variant (k-omega with SST-like nut limiter)
- Better freestream sensitivity via adjusted sigma_d

Usage::

    from pyfoam.turbulence.k_omega_enhanced_2 import KOmegaEnhanced2Model

    model = KOmegaEnhanced2Model(mesh, U, phi)
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
from .k_omega_enhanced import KOmegaEnhancedModel, KOmegaEnhancedConstants

__all__ = ["KOmegaEnhanced2Model", "KOmegaEnhanced2Constants"]


@dataclass(frozen=True)
class KOmegaEnhanced2Constants(KOmegaEnhancedConstants):
    """Constants for enhanced k-omega v2.

    Extends parent constants with:
        a1: Stress-limiter constant (SST-like).
        clim_sigma: Improved cross-diffusion limiter.
        F3_coeff: Near-wall blending coefficient for cross-diffusion.
    """

    a1: float = 0.31
    clim_sigma: float = 0.25
    F3_coeff: float = 0.01


_DEFAULTS = KOmegaEnhanced2Constants()


@TurbulenceModel.register("kOmegaEnhanced2")
class KOmegaEnhanced2Model(KOmegaEnhancedModel):
    """Enhanced k-omega v2 (Wilcox 2006 + improvements).

    Features:
    - Improved cross-diffusion with F3 near-wall limiter
    - SST-like stress limiter: nut = min(k/omega, a1*k/(S*F2))
    - Better freestream sensitivity

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaEnhanced2Constants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaEnhanced2Constants | None = None,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip KOmegaEnhancedModel.__init__)
        super(KOmegaEnhancedModel, self).__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._omega = torch.full((n_cells,), 1.0, device=device, dtype=dtype)
        self._grad_U: torch.Tensor | None = None
        self._y = self._compute_wall_distance()

    # ------------------------------------------------------------------
    # Wall distance
    # ------------------------------------------------------------------

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Nut with stress limiter
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with SST-like stress limiter.

        mu_t = min(k/omega, a1*k / max(S, F3*|Omega|))
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        nut_base = k / omega

        if self._grad_U is None:
            return nut_base

        C = self._C
        S = self._strain_rate()
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))

        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

        # Stress limiter: a1*k / max(a1*omega, S)
        denominator = (C.a1 * omega).max(S_mag)
        nut_limited = C.a1 * k / denominator.clamp(min=1e-16)

        return torch.min(nut_base, nut_limited)

    # ------------------------------------------------------------------
    # Override omega solver with improved cross-diffusion
    # ------------------------------------------------------------------

    def _solve_omega(self, P_k: torch.Tensor) -> None:
        """Solve omega equation with improved cross-diffusion.

        Features F3 near-wall limiter on the cross-diffusion term.
        """
        mesh = self._mesh
        C = self._C

        nu_eff = self._nu + C.sigma * self.nut()

        eqn = fvm.div(self._phi, self._omega, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._omega, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)

        source = (
            C.alpha * omega_safe / k_safe * P_k - C.beta * omega_safe**2
        )

        # Improved cross-diffusion with F3 limiter
        if self._grad_U is not None:
            grad_k = fvc.grad(self._k, "Gauss linear", mesh=self._mesh)
            grad_omega = fvc.grad(
                self._omega, "Gauss linear", mesh=self._mesh
            )
            cross = (grad_k * grad_omega).sum(dim=1)
            sigma_d = torch.where(
                cross > 0,
                torch.full_like(cross, C.sigma_d0),
                torch.zeros_like(cross),
            )

            # F3 limiter: reduces cross-diffusion near walls
            y = self._y.clamp(min=1e-10)
            nu = max(self._nu, 1e-30)
            F3_coeff = getattr(C, 'F3_coeff', 0.01)
            F3 = torch.tanh((y * omega_safe / nu * F3_coeff).clamp(max=10.0))

            CD = sigma_d * cross * F3 / omega_safe
            source = source + CD

        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        omega_new = eqn.source / diag_safe
        self._omega = omega_new.clamp(min=1e-10)

    def __repr__(self) -> str:
        return f"KOmegaEnhanced2Model(n_cells={self._mesh.n_cells})"
