"""
Enhanced k-omega turbulence model v4 — low-Re with cross-diffusion limiter.

Extends :class:`~pyfoam.turbulence.k_omega_enhanced_3.KOmegaEnhanced3Model`
with:

- Low-Reynolds-number damping functions (Wilcox 2008)
- Improved cross-diffusion with viscosity-dependent limiter
- Blended freestream/inside boundary layer omega specification

Usage::

    from pyfoam.turbulence.k_omega_enhanced_4 import KOmegaEnhanced4Model

    model = KOmegaEnhanced4Model(mesh, U, phi)
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
from .k_omega_enhanced_3 import KOmegaEnhanced3Model, KOmegaEnhanced3Constants

__all__ = ["KOmegaEnhanced4Model", "KOmegaEnhanced4Constants"]


@dataclass(frozen=True)
class KOmegaEnhanced4Constants(KOmegaEnhanced3Constants):
    """Constants for enhanced k-omega v4.

    Extends parent constants with:
        R_beta: Low-Re damping coefficient for beta.
        R_k: Low-Re damping coefficient for k equation.
        alpha_low_re: Low-Re alpha coefficient.
        cross_diff_visc_coeff: Viscosity-dependent cross-diffusion limiter coefficient.
    """

    R_beta: float = 8.0
    R_k: float = 6.0
    alpha_low_re: float = 0.5
    cross_diff_visc_coeff: float = 0.5


_DEFAULTS = KOmegaEnhanced4Constants()


@TurbulenceModel.register("kOmegaEnhanced4")
class KOmegaEnhanced4Model(KOmegaEnhanced3Model):
    """Enhanced k-omega v4 (Wilcox 2008 + low-Re improvements).

    Features:
    - Low-Re damping: f_beta = (1 + R_beta * Ret^(-1)) * tanh(alpha_omega * y+)
    - Viscosity-dependent cross-diffusion limiter
    - Improved freestream sensitivity

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaEnhanced4Constants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaEnhanced4Constants | None = None,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip parent chain)
        super(KOmegaEnhanced3Model, self).__init__(mesh, U, phi)

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
    # Low-Re damping functions
    # ------------------------------------------------------------------

    def _f_beta(self) -> torch.Tensor:
        """Low-Re damping for beta coefficient.

        f_beta = (1 + R_beta / Ret) * tanh(alpha_omega * y_plus)

        where Ret = k / (nu * omega).
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        nu = max(self._nu, 1e-30)

        Ret = k_safe / (nu * omega_safe)
        R_beta = getattr(C, 'R_beta', 8.0)

        y = self._y.clamp(min=1e-10)
        y_plus = y * omega_safe.sqrt() / nu
        alpha_om = getattr(C, 'alpha_low_re', 0.5)

        f_beta = (1.0 + R_beta / Ret.clamp(min=0.01)) * torch.tanh(
            (alpha_om * y_plus).clamp(max=10.0)
        )

        return f_beta.clamp(min=0.1, max=5.0)

    def _f_k(self) -> torch.Tensor:
        """Low-Re damping for k production/destruction.

        f_k = max(1, (1 + R_k / Ret)^(-1))
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        nu = max(self._nu, 1e-30)

        Ret = k_safe / (nu * omega_safe)
        R_k = getattr(C, 'R_k', 6.0)

        return (1.0 + R_k / Ret.clamp(min=0.01)).clamp(min=1.0)

    # ------------------------------------------------------------------
    # Enhanced nut with low-Re correction
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with low-Re and strain-rate limiter.

        mu_t = f_beta * min(k/omega, a1*k / max(a1*omega, S))
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        nut_base = k / omega

        f_beta = self._f_beta()

        if self._grad_U is not None:
            C = self._C
            S = self._strain_rate()
            S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
            denominator = (C.a1 * omega).max(S_mag)
            nut_limited = C.a1 * k / denominator.clamp(min=1e-16)
            nut_base = torch.min(nut_base, nut_limited)

        return (f_beta * nut_base).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Override omega solver with low-Re cross-diffusion
    # ------------------------------------------------------------------

    def _solve_omega(self, P_k: torch.Tensor) -> None:
        """Solve omega with low-Re damping and improved cross-diffusion."""
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

        # Low-Re damping
        f_beta = self._f_beta()
        f_k = self._f_k()

        # Strain-dependent beta with low-Re damping
        if self._grad_U is not None:
            Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
            Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))
            chi_omega = Omega_mag / omega_safe
            beta_eff = C.beta + getattr(C, 'beta_star_ratio', 0.09) * chi_omega.pow(2)
            beta_eff = (beta_eff * f_beta).clamp(max=2.0 * C.beta)
        else:
            beta_eff = C.beta * f_beta

        # Production with low-Re correction and clipping
        prod = C.alpha * omega_safe / k_safe * P_k / f_k
        prod_max = getattr(C, 'omega_clip_ratio', 10.0) * beta_eff * omega_safe.pow(2)
        prod = prod.clamp(max=prod_max)

        source = prod - beta_eff * omega_safe.pow(2)

        # Cross-diffusion with viscosity-dependent limiter
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

            # Viscosity-dependent limiter
            visc_coeff = getattr(C, 'cross_diff_visc_coeff', 0.5)
            nu_ratio = (self._nu / (nu_eff).clamp(min=1e-30)).clamp(max=1.0)
            sigma_d = sigma_d * (1.0 - visc_coeff * nu_ratio)

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
        return f"KOmegaEnhanced4Model(n_cells={self._mesh.n_cells})"
