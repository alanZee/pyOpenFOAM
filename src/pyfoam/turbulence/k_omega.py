"""
Standard k-ω turbulence model (Wilcox 1988, 2006).

Implements the two-equation k-ω model solving for turbulent kinetic
energy *k* and specific dissipation rate *ω*.  This is the baseline
k-ω model without the SST blending used in kOmegaSST.

Transport equations:

    ∂k/∂t + ∇·(U k) = ∇·((ν + σ* ν_t) ∇k) + P_k - β* ω k

    ∂ω/∂t + ∇·(U ω) = ∇·((ν + σ ν_t) ∇ω) + α ω/k P_k - β ω²

Turbulent viscosity:
    μ_t = ρ k / ω

Constants (Wilcox 2006):
    α = 5/9, β = 3/40, β* = 9/100, σ = 1/2, σ* = 1/2

References
----------
Wilcox, D.C. (1988). Reassessment of the scale-determining equation
for advanced turbulence models. AIAA Journal, 26(11), 1299–1310.

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

__all__ = ["KOmegaModel", "KOmegaConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KOmegaConstants:
    """Constants for the standard k-ω turbulence model.

    Attributes:
        alpha: Coefficient for ω production (α = 5/9).
        beta: Coefficient for ω destruction (β = 3/40).
        beta_star: Coefficient for k destruction (β* = 9/100).
        sigma: Turbulent Prandtl number for ω (σ = 1/2).
        sigma_star: Turbulent Prandtl number for k (σ* = 1/2).
        kappa: Von Karman constant.
    """

    alpha: float = 5.0 / 9.0
    beta: float = 3.0 / 40.0
    beta_star: float = 9.0 / 100.0
    sigma: float = 0.5
    sigma_star: float = 0.5
    kappa: float = 0.41


_DEFAULT_CONSTANTS = KOmegaConstants()


# ---------------------------------------------------------------------------
# Standard k-ω model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("kOmega")
class KOmegaModel(TurbulenceModel):
    """Standard k-ω turbulence model (Wilcox 2006).

    Solves transport equations for k (turbulent kinetic energy) and
    ω (specific dissipation rate).  The turbulent viscosity is
    computed as μ_t = k / ω.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaConstants, optional
        Model constants.  Defaults to Wilcox (2006) values.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaConstants | None = None,
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

        Returns:
            ``(n_cells,)`` dissipation rate.
        """
        return self._C.beta_star * self._omega * self._k

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

        # Solve ω equation
        self._solve_omega(P_k)

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

        # Source: P_k - β* ω k
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        source = P_k - C.beta_star * omega_safe * k_safe
        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    def _solve_omega(self, P_k: torch.Tensor) -> None:
        """Solve the ω transport equation.

        ∂ω/∂t + ∇·(U ω) = ∇·((ν + σ ν_t) ∇ω)
                         + α ω/k P_k - β ω²
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

        # Source: α ω/k P_k - β ω²
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)

        source = C.alpha * omega_safe / k_safe * P_k - C.beta * omega_safe**2
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
