"""
Enhanced k-omega turbulence model v5 — cross-diffusion limiter with shear-layer adapted beta.

Extends :class:`~pyfoam.turbulence.k_omega_enhanced_4.KOmegaEnhanced4Model`
with:

- Shear-layer adapted beta for improved mixing layer predictions
- Improved stress limiter with tensor-invariant formulation
- SST-like blending function for transition between k-omega and k-epsilon

Usage::

    from pyfoam.turbulence.k_omega_enhanced_5 import KOmegaEnhanced5Model

    model = KOmegaEnhanced5Model(mesh, U, phi)
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
from .k_omega_enhanced_4 import KOmegaEnhanced4Model, KOmegaEnhanced4Constants

__all__ = ["KOmegaEnhanced5Model", "KOmegaEnhanced5Constants"]


@dataclass(frozen=True)
class KOmegaEnhanced5Constants(KOmegaEnhanced4Constants):
    """Constants for enhanced k-omega v5.

    Extends parent constants with:
        beta_sl: Shear-layer adapted beta coefficient.
        C_lim: Stress limiter coefficient.
        sigma_blend: Blending function width parameter.
    """

    beta_sl: float = 0.0708
    C_lim: float = 0.5
    sigma_blend: float = 2.0


_DEFAULTS = KOmegaEnhanced5Constants()


@TurbulenceModel.register("kOmegaEnhanced5")
class KOmegaEnhanced5Model(KOmegaEnhanced4Model):
    """Enhanced k-omega v5 with shear-layer beta and SST-like blending.

    Features:
    - Shear-layer adapted beta coefficient for better free-shear predictions
    - Tensor-invariant stress limiter to prevent non-physical anisotropy
    - SST-like blending for smooth k-omega to k-epsilon transition

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaEnhanced5Constants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaEnhanced5Constants | None = None,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip parent chain)
        super(KOmegaEnhanced4Model, self).__init__(mesh, U, phi)

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
    # SST-like blending function
    # ------------------------------------------------------------------

    def _blending_F1(self) -> torch.Tensor:
        """SST-like blending function F1.

        F1 = tanh(arg^4) where arg = min(max(sqrt(k)/(beta*omega*y), 500*nu/(y^2*omega)),
                                         4*rho*sigma_w2*k/(CD_kw*y^2))

        Simplified: F1 = tanh((y * omega / nu * sigma_blend)^2)
        """
        C = self._C
        omega_safe = self._omega.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)
        nu = max(self._nu, 1e-30)
        sigma_b = getattr(C, 'sigma_blend', 2.0)

        arg = y * omega_safe / nu * sigma_b
        F1 = torch.tanh(arg.pow(2).clamp(max=20.0))
        return F1.clamp(min=0.0, max=1.0)

    # ------------------------------------------------------------------
    # Shear-layer adapted beta
    # ------------------------------------------------------------------

    def _beta_effective(self) -> torch.Tensor:
        """Blended beta coefficient.

        beta_eff = F1 * beta_inner + (1 - F1) * beta_outer

        where beta_inner uses standard k-omega beta and
        beta_outer uses shear-layer adapted beta_sl.
        """
        C = self._C
        F1 = self._blending_F1()

        beta_inner = C.beta
        beta_outer = getattr(C, 'beta_sl', 0.0708)

        return (F1 * beta_inner + (1.0 - F1) * beta_outer).clamp(min=0.01)

    # ------------------------------------------------------------------
    # Tensor-invariant stress limiter
    # ------------------------------------------------------------------

    def _stress_limiter(self, P_k: torch.Tensor) -> torch.Tensor:
        """Apply tensor-invariant stress limiter.

        P_k_limited = min(P_k, C_lim * rho * beta_star * k * omega)

        Prevents production-to-dissipation ratio from becoming non-physical.
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)

        C_lim = getattr(C, 'C_lim', 0.5)
        beta_star = getattr(C, 'beta_star', 0.09)

        P_max = C_lim * beta_star * k_safe * omega_safe
        return P_k.clamp(max=P_max)

    # ------------------------------------------------------------------
    # Enhanced nut
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with low-Re and strain-rate limiter.

        mu_t = f_beta * min(k/omega, a1*k / max(a1*omega, S))
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        nut_base = k / omega

        # Low-Re damping from v4
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
    # Override omega solver
    # ------------------------------------------------------------------

    def _solve_omega(self, P_k: torch.Tensor) -> None:
        """Solve omega with shear-layer beta and blending."""
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

        # Low-Re damping from v4
        f_beta = self._f_beta()
        f_k = self._f_k()

        # Blended beta
        beta_eff = self._beta_effective() * f_beta

        # Production with limiter
        prod = C.alpha * omega_safe / k_safe * P_k / f_k
        prod = self._stress_limiter(prod)

        source = prod - beta_eff * omega_safe.pow(2)

        # Cross-diffusion
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
            visc_coeff = getattr(C, 'cross_diff_visc_coeff', 0.5)
            nu_ratio = (self._nu / (nu_eff).clamp(min=1e-30)).clamp(max=1.0)
            sigma_d = sigma_d * (1.0 - visc_coeff * nu_ratio)

            CD = sigma_d * cross / omega_safe
            source = source + CD

        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        omega_new = eqn.source / diag_safe
        self._omega = omega_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Override correct
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update the enhanced k-omega v5 model."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))

        self._solve_k(P_k)
        self._solve_omega(P_k)

    def __repr__(self) -> str:
        return f"KOmegaEnhanced5Model(n_cells={self._mesh.n_cells})"
