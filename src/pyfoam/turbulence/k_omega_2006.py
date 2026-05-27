"""
Wilcox 2006 k-ω turbulence model with cross-diffusion and low-Re corrections.

Extends the standard k-ω model with two key improvements from
Wilcox (2006):

1. **Cross-diffusion term** in the ω equation:
   σ_d / (k ω) max(∇k · ∇ω, 0)

2. **Low-Reynolds-number correction** for β*:
   β* = β*_0 f_β*(Re_t, χ_ω)

Transport equations:

    ∂k/∂t + ∇·(U k) = ∇·((ν + σ* ν_t) ∇k) + P_k - β* ω k

    ∂ω/∂t + ∇·(U ω) = ∇·((ν + σ ν_t) ∇ω) + α ω/k P_k - β ω²
                      + σ_d / (k ω) max(∇k · ∇ω, 0)

Turbulent viscosity:
    μ_t = ρ k / ω

Constants (Wilcox 2006):
    α = 5/9, β = 3/40, β*₀ = 9/100, σ = 1/2, σ* = 1/2, σ_d = 1/8

References
----------
Wilcox, D.C. (2006). Turbulence Modeling for CFD. 3rd edition,
DCW Industries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel

__all__ = ["KOmega2006Model", "KOmega2006Constants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KOmega2006Constants:
    """Constants for the Wilcox 2006 k-ω turbulence model.

    Attributes:
        alpha: Coefficient for ω production (α = 5/9).
        beta: Coefficient for ω destruction (β = 3/40).
        beta_star_0: Base coefficient for k destruction (β*₀ = 9/100).
        sigma: Turbulent Prandtl number for ω (σ = 1/2).
        sigma_star: Turbulent Prandtl number for k (σ* = 1/2).
        sigma_d: Cross-diffusion coefficient (σ_d = 1/8).
        kappa: Von Karman constant.
    """

    alpha: float = 5.0 / 9.0
    beta: float = 3.0 / 40.0
    beta_star_0: float = 9.0 / 100.0
    sigma: float = 0.5
    sigma_star: float = 0.5
    sigma_d: float = 1.0 / 8.0
    kappa: float = 0.41


_DEFAULT_CONSTANTS = KOmega2006Constants()


# ---------------------------------------------------------------------------
# Wilcox 2006 k-ω model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("kOmega2006")
class KOmega2006Model(TurbulenceModel):
    """Wilcox 2006 k-ω turbulence model with cross-diffusion and low-Re correction.

    Extends the standard k-ω model with:

    - Cross-diffusion source term: σ_d / (k ω) max(∇k · ∇ω, 0)
    - Low-Re β* correction: β* = β*₀ f_β*(Re_t, χ_ω)

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmega2006Constants, optional
        Model constants.  Defaults to Wilcox (2006) values.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmega2006Constants | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)

        self._C = constants or _DEFAULT_CONSTANTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        # Turbulence fields
        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._omega = torch.full((n_cells,), 1.0, device=device, dtype=dtype)

        # Velocity gradient tensor
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
        """Turbulent viscosity: μ_t = k / ω.

        Returns:
            ``(n_cells,)`` turbulent viscosity field.
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
        """Return dissipation rate: ε = β* ω k.

        Uses the low-Re corrected β* value.

        Returns:
            ``(n_cells,)`` dissipation rate.
        """
        beta_star = self._beta_star_correction()
        return beta_star * self._omega * self._k

    def correct(self) -> None:
        """Update the k-ω model: compute nut, solve k and ω equations."""
        # Compute velocity gradient tensor
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U

        # Production rate: P_k = 2 ν_t S:S
        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))

        # Solve k equation
        self._solve_k(P_k)

        # Solve ω equation (with cross-diffusion)
        self._solve_omega(P_k)

    # ------------------------------------------------------------------
    # Internal: low-Re β* correction
    # ------------------------------------------------------------------

    def _beta_star_correction(self) -> torch.Tensor:
        """Compute low-Re corrected β* field.

        Wilcox 2006 Eq. (4.44):
            β* = β*₀ f_β*

        where f_β* is a piecewise function of Re_t and χ_ω:

            Re_t = k / (ν ω)
            χ_ω = |Ω² / (β*₀ ω²)|   (Ω = vorticity magnitude)

            If Re_t ≤ 200 or χ_ω ≤ 1/β*₀:
                f_β* = (2/9 + (Re_t/8) (1 + (Re_t/8)²)) / (1 + (Re_t/8)³)
            Else:
                f_β* = 1

        Returns:
            ``(n_cells,)`` corrected β* field.
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)

        # Turbulent Reynolds number
        Re_t = k_safe / (self._nu * omega_safe)

        # Vorticity magnitude squared (Ω² = 2 W:W where W is rotation tensor)
        if self._grad_U is not None:
            W = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
            Omega2 = 2.0 * (W * W).sum(dim=(1, 2))
        else:
            Omega2 = torch.zeros_like(k_safe)

        # χ_ω = |Ω²| / (β*₀ ω²)
        chi_omega = Omega2 / (C.beta_star_0 * omega_safe**2)

        # Low-Re correction factor
        # Condition: Re_t > 200 AND χ_ω > 1/β*₀  =>  f_beta_star = 1
        Re_ratio = Re_t / 8.0
        f_beta_star = (2.0 / 9.0 + Re_ratio * (1.0 + Re_ratio**2)) / (
            1.0 + Re_ratio**3
        )

        # Apply correction only when low-Re conditions are met
        high_re = (Re_t > 200.0) & (chi_omega > 1.0 / C.beta_star_0)
        f_beta_star = torch.where(high_re, torch.ones_like(f_beta_star), f_beta_star)

        return C.beta_star_0 * f_beta_star

    # ------------------------------------------------------------------
    # Internal: transport equations
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve the k transport equation.

        ∂k/∂t + ∇·(U k) = ∇·((ν + σ* ν_t) ∇k) + P_k - β* ω k
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        # Effective diffusivity
        nu_eff = self._nu + C.sigma_star * self.nut()

        # Build equation
        eqn = fvm.div(self._phi, self._k, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._k, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: P_k - β* ω k  (β* includes low-Re correction)
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        beta_star = self._beta_star_correction()
        source = P_k - beta_star * omega_safe * k_safe
        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    def _solve_omega(self, P_k: torch.Tensor) -> None:
        """Solve the ω transport equation with cross-diffusion.

        ∂ω/∂t + ∇·(U ω) = ∇·((ν + σ ν_t) ∇ω)
                         + α ω/k P_k - β ω²
                         + σ_d / (k ω) max(∇k · ∇ω, 0)
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        # Effective diffusivity
        nu_eff = self._nu + C.sigma * self.nut()

        # Build equation
        eqn = fvm.div(self._phi, self._omega, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._omega, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: α ω/k P_k - β ω² + cross-diffusion
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)

        source = C.alpha * omega_safe / k_safe * P_k - C.beta * omega_safe**2

        # Cross-diffusion term: σ_d / (k ω) max(∇k · ∇ω, 0)
        cross_diff = self._cross_diffusion(k_safe, omega_safe)
        source = source + cross_diff

        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        omega_new = eqn.source / diag_safe
        self._omega = omega_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Internal: helper computations
    # ------------------------------------------------------------------

    def _strain_rate(self) -> torch.Tensor:
        """Compute strain rate tensor S = 0.5 (∇U + ∇U^T).

        Returns:
            ``(n_cells, 3, 3)`` strain rate tensor.
        """
        grad_U = self._grad_U
        return 0.5 * (grad_U + grad_U.transpose(-1, -2))

    def _cross_diffusion(
        self, k_safe: torch.Tensor, omega_safe: torch.Tensor
    ) -> torch.Tensor:
        """Compute the cross-diffusion source term for ω.

        CD_kw = σ_d / (k ω) max(∇k · ∇ω, 0)

        Returns:
            ``(n_cells,)`` cross-diffusion source contribution.
        """
        C = self._C

        # Compute cell-centred gradients of k and ω
        grad_k = fvc.grad(self._k, "Gauss linear", mesh=self._mesh)   # (n_cells, 3)
        grad_omega = fvc.grad(self._omega, "Gauss linear", mesh=self._mesh)

        # Dot product ∇k · ∇ω
        dot_product = (grad_k * grad_omega).sum(dim=1)

        # σ_d / (k ω) max(∇k · ∇ω, 0)
        cross_diff = C.sigma_d / (k_safe * omega_safe) * dot_product.clamp(min=0.0)

        return cross_diff
