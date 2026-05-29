"""
Enhanced k-omega turbulence model v3 — Wilcox 2006 with improved cross-diffusion.

Extends :class:`~pyfoam.turbulence.k_omega_enhanced_2.KOmegaEnhanced2Model`
with:

- Strain-rate-dependent beta coefficient
- Improved freestream sensitivity via omega source clipping
- Limiting strain-to-vorticity ratio for nut

Usage::

    from pyfoam.turbulence.k_omega_enhanced_3 import KOmegaEnhanced3Model

    model = KOmegaEnhanced3Model(mesh, U, phi)
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
from .k_omega_enhanced_2 import KOmegaEnhanced2Model, KOmegaEnhanced2Constants

__all__ = ["KOmegaEnhanced3Model", "KOmegaEnhanced3Constants"]


@dataclass(frozen=True)
class KOmegaEnhanced3Constants(KOmegaEnhanced2Constants):
    """Constants for enhanced k-omega v3.

    Extends parent constants with:
        beta_star_ratio: Ratio of strain-dependent beta_star correction.
        omega_clip_ratio: Maximum omega production-to-dissipation ratio.
        nut_strain_limit: Maximum strain-to-vorticity ratio for nut limiter.
    """

    beta_star_ratio: float = 0.09
    omega_clip_ratio: float = 10.0
    nut_strain_limit: float = 2.0


_DEFAULTS = KOmegaEnhanced3Constants()


@TurbulenceModel.register("kOmegaEnhanced3")
class KOmegaEnhanced3Model(KOmegaEnhanced2Model):
    """Enhanced k-omega v3 (Wilcox 2006 + improvements).

    Features:
    - Strain-dependent beta coefficient: beta_eff = beta + beta_star_ratio * (S/omega)^2
    - Omega production clipping to prevent runaway growth
    - Strain-rate-limited nut

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaEnhanced3Constants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaEnhanced3Constants | None = None,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip parent chain)
        super(KOmegaEnhanced2Model, self).__init__(mesh, U, phi)

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
    # Enhanced nut with strain-rate limiting
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with strain-rate limiter.

        mu_t = min(k/omega, a1*k / max(a1*omega, S), nut_strain_limit * k / S)
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        nut_base = k / omega

        if self._grad_U is None:
            return nut_base

        C = self._C
        S = self._strain_rate()
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

        # SST-like stress limiter
        denominator = (C.a1 * omega).max(S_mag)
        nut_limited = C.a1 * k / denominator.clamp(min=1e-16)

        # Additional strain-rate limiter
        nut_strain = C.nut_strain_limit * k / S_mag.clamp(min=1e-16)

        return torch.min(nut_base, torch.min(nut_limited, nut_strain))

    # ------------------------------------------------------------------
    # Override omega solver with strain-dependent beta and clipping
    # ------------------------------------------------------------------

    def _solve_omega(self, P_k: torch.Tensor) -> None:
        """Solve omega equation with strain-dependent beta and clipping.

        Features:
        - beta_eff = beta + beta_star_ratio * chi_omega^2
          where chi_omega = |Omega| / omega
        - Production clipping to prevent runaway omega growth
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

        # Strain-dependent beta
        if self._grad_U is not None:
            Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
            Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))
            chi_omega = Omega_mag / omega_safe
            beta_eff = C.beta + getattr(C, 'beta_star_ratio', 0.09) * chi_omega.pow(2)
            beta_eff = beta_eff.clamp(max=2.0 * C.beta)
        else:
            beta_eff = C.beta

        # Production with clipping
        prod = C.alpha * omega_safe / k_safe * P_k
        prod_max = getattr(C, 'omega_clip_ratio', 10.0) * beta_eff * omega_safe.pow(2)
        prod = prod.clamp(max=prod_max)

        source = prod - beta_eff * omega_safe.pow(2)

        # Cross-diffusion with F3 limiter (from parent)
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
        return f"KOmegaEnhanced3Model(n_cells={self._mesh.n_cells})"
