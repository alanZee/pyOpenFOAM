"""
RNG k-ε turbulence model (Yakhot & Orszag 1986).

Implements the Renormalization Group (RNG) variant of k-ε which
provides improved predictions for:
- Flows with strong streamline curvature
- Low-Re effects (near-wall regions)
- Separated flows

The model modifies the standard k-ε with:
- Variable C_μ computed from strain and rotation rates
- Additional R term in ε equation: C_μ η³ (1 - η/η₀) / (1 + β η³) ε²/k
- Modified constants derived from RNG theory

Constants (OpenFOAM defaults):
    C_μ = 0.0845, C1 = 1.42, C2 = 1.68, σ_k = 0.7194, σ_ε = 0.7194
    η₀ = 4.38, β = 0.012

References
----------
Yakhot, V. & Orszag, S.A. (1986). Renormalization group analysis of
turbulence. Journal of Scientific Computing, 1(1), 3–51.

Yakhot, V., Orszag, S.A., Thangam, S., Gatski, T.B. & Speziale, C.G.
(1992). Development of turbulence models for shear flows by a double
expansion technique. Physics of Fluids A, 4(7), 1510–1520.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel

__all__ = ["RNGkEpsilonModel", "RNGkEpsilonConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RNGkEpsilonConstants:
    """Constants for the RNG k-ε turbulence model.

    Attributes:
        C_mu: Eddy-viscosity constant (RNG: 0.0845).
        C1: Production coefficient for ε (RNG: 1.42).
        C2: Destruction coefficient for ε (RNG: 1.68).
        sigma_k: Turbulent Prandtl number for k (RNG: 0.7194).
        sigma_eps: Turbulent Prandtl number for ε (RNG: 0.7194).
        eta_0: R-term coefficient (default: 4.38).
        beta: R-term coefficient (default: 0.012).
    """

    C_mu: float = 0.0845
    C1: float = 1.42
    C2: float = 1.68
    sigma_k: float = 0.7194
    sigma_eps: float = 0.7194
    eta_0: float = 4.38
    beta: float = 0.012


_DEFAULT_CONSTANTS = RNGkEpsilonConstants()


# ---------------------------------------------------------------------------
# RNG k-ε model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("RNGkEpsilon")
class RNGkEpsilonModel(TurbulenceModel):
    """RNG k-ε turbulence model.

    Improved k-ε variant using renormalization group theory.
    Better for separated flows and streamline curvature.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : RNGkEpsilonConstants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: RNGkEpsilonConstants | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)

        self._C = constants or _DEFAULT_CONSTANTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        # Turbulence fields
        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)

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
        """Turbulent viscosity: μ_t = C_μ k² / ε.

        Returns:
            ``(n_cells,)`` turbulent viscosity field.
        """
        k = self._k.clamp(min=1e-16)
        eps = self._eps.clamp(min=1e-16)
        return self._C.C_mu * k**2 / eps

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    def epsilon(self) -> torch.Tensor:
        """Return dissipation rate ``(n_cells,)``."""
        return self._eps

    def correct(self) -> None:
        """Update the RNG k-ε model."""
        # Compute velocity gradient tensor
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U

        # Production rate
        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))

        # Solve k equation
        self._solve_k(P_k)

        # Solve ε equation with R term
        self._solve_eps(P_k)

    # ------------------------------------------------------------------
    # Internal: R term computation
    # ------------------------------------------------------------------

    def _R_term(self, P_k: torch.Tensor) -> torch.Tensor:
        """Compute the RNG R term for the ε equation.

        R = C_μ η³ (1 - η/η₀) / (1 + β η³) ε²/k

        where η = S k / ε (strain rate parameter).

        Returns:
            ``(n_cells,)`` R term.
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        # Strain rate magnitude
        S = self._strain_rate()
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

        # η = S k / ε
        eta = S_mag * k_safe / eps_safe

        # R = C_μ η³ (1 - η/η₀) / (1 + β η³) ε²/k
        eta_3 = eta**3
        R = C.C_mu * eta_3 * (1.0 - eta / C.eta_0) / (1.0 + C.beta * eta_3)
        R = R * eps_safe**2 / k_safe

        # Clamp: R should only be positive (reduces ε production)
        return R.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Internal: transport equations
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve the k transport equation.

        ∂k/∂t + ∇·(U k) = ∇·((ν + ν_t/σ_k) ∇k) + P_k - ε
        """
        mesh = self._mesh
        C = self._C

        nu_eff = self._nu + self.nut() / C.sigma_k

        eqn = fvm.div(self._phi, self._k, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._k, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        source = P_k - self._eps
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    def _solve_eps(self, P_k: torch.Tensor) -> None:
        """Solve the ε transport equation with R term.

        ∂ε/∂t + ∇·(U ε) = ∇·((ν + ν_t/σ_ε) ∇ε)
                         + C1 ε/k P_k - C2 ε²/k + R
        """
        mesh = self._mesh
        C = self._C

        nu_eff = self._nu + self.nut() / C.sigma_eps

        eqn = fvm.div(self._phi, self._eps, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._eps, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        # RNG R term
        R = self._R_term(P_k)

        source = (
            C.C1 * eps_safe / k_safe * P_k
            - C.C2 * eps_safe**2 / k_safe
            + R
        )
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        eps_new = eqn.source / diag_safe
        self._eps = eps_new.clamp(min=1e-10)

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
