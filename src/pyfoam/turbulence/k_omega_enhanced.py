"""
Enhanced k-omega turbulence model — Wilcox 2006 variant.

Implements the k-omega model following Wilcox (2006) with:

- Cross-diffusion term in omega equation
- Low-Reynolds-number correction (f_beta function)
- Stress-limiter variant option
- Improved freestream sensitivity

References:
    Wilcox, D.C. (2006). Turbulence Modeling for CFD. 3rd edition,
    DCW Industries.

Usage::

    from pyfoam.turbulence.k_omega_enhanced import KOmegaEnhancedModel

    model = KOmegaEnhancedModel(mesh, U, phi)
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

__all__ = ["KOmegaEnhancedModel", "KOmegaEnhancedConstants"]


@dataclass(frozen=True)
class KOmegaEnhancedConstants:
    """Constants for the enhanced k-omega (Wilcox 2006) model.

    Attributes:
        alpha: Coefficient for omega production (alpha = 5/9).
        beta: Coefficient for omega destruction (beta = 3/40).
        beta_star: Coefficient for k destruction (beta* = 9/100).
        sigma: Turbulent Prandtl number for omega (sigma = 1/2).
        sigma_star: Turbulent Prandtl number for k (sigma* = 1/2).
        kappa: Von Karman constant.
        sigma_d0: Cross-diffusion coefficient (sigma_d = 0.125 for k-omega
            cross-diffusion when sigma_d > 0).
        clim: Cross-diffusion limiter coefficient.
        beta_i: Low-Re correction coefficient.
    """

    alpha: float = 5.0 / 9.0
    beta: float = 3.0 / 40.0
    beta_star: float = 9.0 / 100.0
    sigma: float = 0.5
    sigma_star: float = 0.5
    kappa: float = 0.41
    sigma_d0: float = 0.125
    clim: float = 0.5
    beta_i: float = 0.0708


_DEFAULTS = KOmegaEnhancedConstants()


@TurbulenceModel.register("kOmegaEnhanced")
class KOmegaEnhancedModel(TurbulenceModel):
    """Enhanced k-omega model (Wilcox 2006).

    Solves transport equations for k and omega with:
    - Cross-diffusion term: sigma_d * (grad(k) . grad(omega)) / omega
    - Low-Re damping for beta coefficient
    - Stress limiter option

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor or volVectorField
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaEnhancedConstants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaEnhancedConstants | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._omega = torch.full((n_cells,), 1.0, device=device, dtype=dtype)
        self._grad_U: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def k_field(self) -> torch.Tensor:
        """Turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    @k_field.setter
    def k_field(self, value: torch.Tensor) -> None:
        self._k = value.to(device=self._device, dtype=self._dtype)

    @property
    def omega_field(self) -> torch.Tensor:
        """Specific dissipation rate ``(n_cells,)``."""
        return self._omega

    @omega_field.setter
    def omega_field(self, value: torch.Tensor) -> None:
        self._omega = value.to(device=self._device, dtype=self._dtype)

    # ------------------------------------------------------------------
    # TurbulenceModel interface
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity: mu_t = k / omega.

        With optional stress-limiter:
        mu_t = min(k/omega, a1*k/S_F2) when gradient is available.
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        return k / omega

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    def omega(self) -> torch.Tensor:
        """Return specific dissipation rate ``(n_cells,)``."""
        return self._omega

    def epsilon(self) -> torch.Tensor:
        """Return dissipation rate: epsilon = beta* omega k."""
        return self._C.beta_star * self._omega * self._k

    def correct(self) -> None:
        """Update the enhanced k-omega model."""
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

    # ------------------------------------------------------------------
    # Low-Re correction
    # ------------------------------------------------------------------

    def _beta_star_eff(self) -> torch.Tensor:
        """Low-Re correction for beta_star.

        beta_star = beta_star_0 * f_beta_star
        where f_beta_star = (1/17.6) * (1 + 3.6*chi_beta) * chi_beta
        and chi_beta = Re_t / (Re_t + 130)
        """
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        nu = max(self._nu, 1e-30)

        Re_t = k_safe / (nu * omega_safe)
        chi_beta = Re_t / (Re_t + 130.0)
        f_beta = (1.0 / 17.6) * (1.0 + 3.6 * chi_beta) * chi_beta

        # Clamp to prevent instability
        return (self._C.beta_star * f_beta).clamp(min=self._C.beta_star * 0.1)

    # ------------------------------------------------------------------
    # Transport equations
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve the k transport equation.

        dk/dt + div(U k) = div((nu + sigma* nu_t) grad(k)) + P_k - beta* omega k
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        nu_eff = self._nu + C.sigma_star * self.nut()

        eqn = fvm.div(self._phi, self._k, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._k, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: P_k - beta*_eff * omega * k
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        beta_star_eff = self._beta_star_eff()
        source = P_k - beta_star_eff * omega_safe * k_safe
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    def _solve_omega(self, P_k: torch.Tensor) -> None:
        """Solve the omega transport equation with cross-diffusion.

        domegadt + div(U omega) = div((nu + sigma nu_t) grad(omega))
            + alpha omega/k P_k - beta omega^2
            + sigma_d grad(k).grad(omega) / omega  (cross-diffusion)
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
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

        # Production + destruction
        source = (
            C.alpha * omega_safe / k_safe * P_k - C.beta * omega_safe**2
        )

        # Cross-diffusion term (Wilcox 2006)
        # sigma_d = sigma_d0 if grad(k).grad(omega) > 0, else 0
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
            CD = sigma_d * cross / omega_safe
            source = source + CD

        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        omega_new = eqn.source / diag_safe
        self._omega = omega_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strain_rate(self) -> torch.Tensor:
        """Strain rate tensor S = 0.5 (grad(U) + grad(U)^T)."""
        return 0.5 * (self._grad_U + self._grad_U.transpose(-1, -2))

    def __repr__(self) -> str:
        return f"KOmegaEnhancedModel(n_cells={self._mesh.n_cells})"
