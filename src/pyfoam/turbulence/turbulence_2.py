"""
Enhanced turbulence models — k-ω SST 2003 and realizable k-ε v2.

Provides improved variants of standard RANS turbulence models:

- :class:`KOmegaSST2003Model` — Menter 2003 k-ω SST with updated
  constants and improved blending.
- :class:`RealizableKE2Model` — Enhanced realizable k-ε with improved
  production limiter and realizability constraints.

Both models register via :class:`TurbulenceModel.register` RTS.

Usage::

    from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model

    model = TurbulenceModel.create("kOmegaSST2003", mesh, U, phi)
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

__all__ = [
    "KOmegaSST2003Model",
    "KOmegaSST2003Constants",
    "RealizableKE2Model",
    "RealizableKE2Constants",
]


# ======================================================================
# k-ω SST 2003 (Menter 2003)
# ======================================================================


@dataclass(frozen=True)
class KOmegaSST2003Constants:
    """Constants for the k-ω SST 2003 model.

    Updated constants from Menter et al. (2003) with improved
    behaviour for separated flows and transition prediction.

    Attributes:
        sigma_k1: Turbulent Prandtl number for k (inner).
        sigma_k2: Turbulent Prandtl number for k (outer).
        sigma_omega1: Turbulent Prandtl number for ω (inner).
        sigma_omega2: Turbulent Prandtl number for ω (outer).
        beta1: Destruction coefficient for ω (inner).
        beta2: Destruction coefficient for ω (outer).
        gamma1: Production coefficient for ω (inner).
        gamma2: Production coefficient for ω (outer).
        a1: SST blending constant.
        beta_star: k destruction coefficient.
        kappa: von Karman constant.
        c1: Cross-diffusion limiter constant (2003 addition).
    """

    sigma_k1: float = 0.85
    sigma_k2: float = 1.0
    sigma_omega1: float = 0.5
    sigma_omega2: float = 0.856
    beta1: float = 0.075
    beta2: float = 0.0828
    gamma1: float = 5.0 / 9.0
    gamma2: float = 0.44
    a1: float = 0.31
    beta_star: float = 0.09
    kappa: float = 0.41
    c1: float = 10.0


_SST2003_DEFAULTS = KOmegaSST2003Constants()


@TurbulenceModel.register("kOmegaSST2003")
class KOmegaSST2003Model(TurbulenceModel):
    """k-ω SST Menter 2003 variant.

    Improvements over the 1994 SST:
    - Updated blending with cross-diffusion limiter (c1 constant)
    - Improved wall-distance dependency for separated flows
    - Better freestream sensitivity

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor or volVectorField
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaSST2003Constants, optional
        Model constants. Defaults to Menter (2003) values.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaSST2003Constants | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)

        self._C = constants or _SST2003_DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._omega = torch.full((n_cells,), 1.0, device=device, dtype=dtype)
        self._grad_U: torch.Tensor | None = None
        self._y = self._compute_wall_distance()

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
        """Turbulent viscosity with SST limiter.

        μ_t = ρ a₁ k / max(a₁ ω, S F₂)
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)

        if self._grad_U is None:
            return k / omega

        S = self._strain_magnitude()
        F2 = self._F2()
        denominator = (self._C.a1 * omega).max(S * F2)
        return self._C.a1 * k / denominator.clamp(min=1e-16)

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    def omega(self) -> torch.Tensor:
        """Return specific dissipation rate ``(n_cells,)``."""
        return self._omega

    def epsilon(self) -> torch.Tensor:
        """Return dissipation rate: ε = β* ω k."""
        return self._C.beta_star * self._omega * self._k

    def correct(self) -> None:
        """Update the k-ω SST 2003 model."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U

        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))

        self._solve_k(P_k)
        self._solve_omega(P_k)

    # ------------------------------------------------------------------
    # Blending functions
    # ------------------------------------------------------------------

    def _F1(self) -> torch.Tensor:
        """First blending function F1 (inner/outer transition)."""
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)
        C = self._C

        arg1 = torch.sqrt(k) / (C.beta_star * omega * y)
        arg2 = 500.0 * self._nu / (y**2 * omega)
        CD_kω = (2.0 * C.sigma_omega2 / omega).clamp(min=1e-10)
        arg3 = 4.0 * C.sigma_omega2 * k / (CD_kω * y**2)

        arg = torch.min(torch.max(arg1, arg2), arg3)
        return torch.tanh(arg**4)

    def _F2(self) -> torch.Tensor:
        """Second blending function F2 (shear stress limiter)."""
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)
        C = self._C

        arg1 = 2.0 * torch.sqrt(k) / (C.beta_star * omega * y)
        arg2 = 500.0 * self._nu / (y**2 * omega)

        arg = torch.max(arg1, arg2)
        return torch.tanh(arg**2)

    # ------------------------------------------------------------------
    # Transport equations
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve the k transport equation."""
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        F1 = self._F1()
        sigma_k = F1 * C.sigma_k1 + (1.0 - F1) * C.sigma_k2
        nu_eff = self._nu + sigma_k * self.nut()

        eqn = fvm.div(self._phi, self._k, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._k, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        source = P_k - C.beta_star * omega_safe * k_safe
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    def _solve_omega(self, P_k: torch.Tensor) -> None:
        """Solve the ω transport equation with 2003 cross-diffusion limiter."""
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        F1 = self._F1()
        sigma_w = F1 * C.sigma_omega1 + (1.0 - F1) * C.sigma_omega2
        beta = F1 * C.beta1 + (1.0 - F1) * C.beta2
        gamma = F1 * C.gamma1 + (1.0 - F1) * C.gamma2

        nu_eff = self._nu + sigma_w * self.nut()

        eqn = fvm.div(self._phi, self._omega, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._omega, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        nut = self.nut()
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)

        # Production + destruction + cross-diffusion
        source = (
            gamma * P_k / nut.clamp(min=1e-16)
            - beta * omega_safe**2
        )

        # 2003 cross-diffusion with limiter
        if self._grad_U is not None:
            CD_kw = (
                2.0 * (1.0 - F1) * C.sigma_omega2
                * self._cross_diffusion()
                / omega_safe
            ).clamp(min=0.0)
            # c1 limiter (2003 addition)
            F3 = torch.tanh(self._y.clamp(min=1e-10) * omega_safe / self._nu * 0.01)
            CD_limited = CD_kw * (1.0 + C.c1 * F3)
            source = source + CD_limited

        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        omega_new = eqn.source / diag_safe
        self._omega = omega_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strain_rate(self) -> torch.Tensor:
        """Compute strain rate tensor S = 0.5 (∇U + ∇U^T)."""
        grad_U = self._grad_U
        return 0.5 * (grad_U + grad_U.transpose(-1, -2))

    def _strain_magnitude(self) -> torch.Tensor:
        """Compute magnitude of strain rate |S| = sqrt(2 S:S)."""
        S = self._strain_rate()
        return torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

    def _cross_diffusion(self) -> torch.Tensor:
        """Compute ∇k · ∇ω for cross-diffusion term."""
        grad_k = fvc.grad(self._k, "Gauss linear", mesh=self._mesh)
        grad_omega = fvc.grad(self._omega, "Gauss linear", mesh=self._mesh)
        return (grad_k * grad_omega).sum(dim=1)

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance from cell centres."""
        cc = self._mesh.cell_centres
        y = cc.norm(dim=1).clamp(min=1e-10)
        return y

    def __repr__(self) -> str:
        return f"KOmegaSST2003Model(n_cells={self._mesh.n_cells})"


# ======================================================================
# Enhanced Realizable k-ε
# ======================================================================


@dataclass(frozen=True)
class RealizableKE2Constants:
    """Constants for enhanced realizable k-ε model.

    Attributes:
        C_mu_base: Base eddy-viscosity constant.
        C1: Production coefficient for ε.
        C2: Destruction coefficient for ε.
        sigma_k: Turbulent Prandtl number for k.
        sigma_eps: Turbulent Prandtl number for ε.
        A0: Realizability constant (default 4.0).
        A_s: Realizability strain-rate constant.
        C3: Buoyancy coefficient (default 0.0, no buoyancy).
        eta_0: Production limiter threshold.
        C_lim: Production limiter constant.
    """

    C_mu_base: float = 0.09
    C1: float = 1.44
    C2: float = 1.9
    sigma_k: float = 1.0
    sigma_eps: float = 1.2
    A0: float = 4.0
    A_s: float = 0.0  # computed dynamically if 0
    C3: float = 0.0
    eta_0: float = 4.38
    C_lim: float = 0.43


_RKE2_DEFAULTS = RealizableKE2Constants()

# Precompute sqrt(6) * cos(pi/3) for default A_s
_SQRT6_COS_PI3 = 6.0**0.5 * 0.5


@TurbulenceModel.register("realizableKE2")
class RealizableKE2Model(TurbulenceModel):
    """Enhanced realizable k-ε turbulence model.

    Improvements over standard realizable k-ε:
    - Production limiter: P_k / ε bounded by C_lim
    - Improved C_mu computation with better strain-vorticity coupling
    - Enhanced realizability enforcement
    - Backscatter prevention via explicit production clipping

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor or volVectorField
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : RealizableKE2Constants, optional
        Model constants. Defaults to Shih et al. (1995) values.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: RealizableKE2Constants | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)

        self._C = constants or _RKE2_DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
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
    def epsilon_field(self) -> torch.Tensor:
        """Dissipation rate ``(n_cells,)``."""
        return self._eps

    @epsilon_field.setter
    def epsilon_field(self, value: torch.Tensor) -> None:
        self._eps = value.to(device=self._device, dtype=self._dtype)

    # ------------------------------------------------------------------
    # TurbulenceModel interface
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with dynamic C_mu.

        μ_t = C_mu * k² / ε
        where C_mu is computed from strain and rotation rates.
        """
        k = self._k.clamp(min=1e-16)
        eps = self._eps.clamp(min=1e-16)
        C_mu = self._compute_C_mu()
        return C_mu * k**2 / eps

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    def epsilon(self) -> torch.Tensor:
        """Return dissipation rate ``(n_cells,)``."""
        return self._eps

    def correct(self) -> None:
        """Update the enhanced realizable k-ε model."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U

        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))

        # Apply production limiter
        eps_safe = self._eps.clamp(min=1e-16)
        P_k = P_k.clamp(max=self._C.C_lim * eps_safe)

        self._solve_k(P_k)
        self._solve_eps(P_k)

    # ------------------------------------------------------------------
    # C_mu computation
    # ------------------------------------------------------------------

    def _compute_C_mu(self) -> torch.Tensor:
        """Compute dynamic C_mu for realizable k-ε.

        C_mu = 1 / (A0 + A_s * U* * k / ε)

        where U* = sqrt(S:S + Ω:Ω), A0 = 4.0, A_s = sqrt(6) cos(φ).
        """
        if self._grad_U is None:
            return torch.full_like(self._k, self._C.C_mu_base)

        S = self._strain_rate()
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))

        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        U_star = torch.sqrt(S_mag**2 + Omega_mag**2)

        A_s = self._C.A_s if self._C.A_s > 0 else _SQRT6_COS_PI3

        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        denominator = (
            self._C.A0 + A_s * U_star * k_safe / eps_safe
        ).clamp(min=1e-10)

        return (1.0 / denominator).clamp(min=0.001, max=0.5)

    # ------------------------------------------------------------------
    # Transport equations
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve the k transport equation.

        ∂k/∂t + ∇·(U k) = ∇·((ν + ν_t/σ_k) ∇k) + P_k - ε
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        nu_eff = self._nu + self.nut() / self._C.sigma_k

        eqn = fvm.div(self._phi, self._k, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._k, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: P_k - ε (with backscatter prevention)
        source = P_k - self._eps.clamp(min=0.0)
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    def _solve_eps(self, P_k: torch.Tensor) -> None:
        """Solve the ε transport equation.

        ∂ε/∂t + ∇·(U ε) = ∇·((ν + ν_t/σ_ε) ∇ε)
                         + C1 S ε - C2 ε² / (k + √(ν ε))
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        nu_eff = self._nu + self.nut() / C.sigma_eps

        eqn = fvm.div(self._phi, self._eps, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._eps, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        # Strain magnitude for production term
        S = self._strain_rate()
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

        # Enhanced realizable source
        # C1 * S * ε (production)
        # C2 * ε² / (k + √(ν ε)) (destruction with improved denominator)
        nu_eps = (self._nu * eps_safe).sqrt()
        source = C.C1 * S_mag * eps_safe - C.C2 * eps_safe**2 / (k_safe + nu_eps)

        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        eps_new = eqn.source / diag_safe
        self._eps = eps_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strain_rate(self) -> torch.Tensor:
        """Compute strain rate tensor S = 0.5 (∇U + ∇U^T)."""
        grad_U = self._grad_U
        return 0.5 * (grad_U + grad_U.transpose(-1, -2))

    def _vorticity_magnitude(self) -> torch.Tensor:
        """Compute magnitude of vorticity |Ω|."""
        grad_U = self._grad_U
        Omega = 0.5 * (grad_U - grad_U.transpose(-1, -2))
        return torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

    def __repr__(self) -> str:
        return f"RealizableKE2Model(n_cells={self._mesh.n_cells})"
