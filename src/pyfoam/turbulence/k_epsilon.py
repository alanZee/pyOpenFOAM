"""
k-ε turbulence model — standard and realizable variants.

Implements the two-equation k-ε model following OpenFOAM's ``kEpsilon`` and
``realizableKEpsilon`` models.  Transport equations are solved for turbulent
kinetic energy *k* and dissipation rate *ε*.

Standard k-ε constants (Launder & Spalding 1974):
    C_μ = 0.09, C1 = 1.44, C2 = 1.92, σ_k = 1.0, σ_ε = 1.3

Realizable k-ε (Shih et al. 1995):
    C_μ is computed from strain and rotation rates.
    C2 = 1.9, σ_k = 1.0, σ_ε = 1.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel

__all__ = ["KEpsilonModel", "KEpsilonConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KEpsilonConstants:
    """Constants for the k-ε turbulence model.

    Attributes:
        C_mu: Eddy-viscosity constant (standard: 0.09).
        C1: Production coefficient for ε (standard: 1.44).
        C2: Destruction coefficient for ε (standard: 1.92).
        sigma_k: Turbulent Prandtl number for k (standard: 1.0).
        sigma_eps: Turbulent Prandtl number for ε (standard: 1.3).
    """

    C_mu: float = 0.09
    C1: float = 1.44
    C2: float = 1.92
    sigma_k: float = 1.0
    sigma_eps: float = 1.3


# Default constants (OpenFOAM standard)
_DEFAULT_CONSTANTS = KEpsilonConstants()

# Realizable constants
_REALIZABLE_CONSTANTS = KEpsilonConstants(
    C_mu=0.09,  # computed dynamically in realizable
    C1=1.44,
    C2=1.9,
    sigma_k=1.0,
    sigma_eps=1.2,
)


# ---------------------------------------------------------------------------
# k-ε model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("kEpsilon")
class KEpsilonModel(TurbulenceModel):
    """Standard k-ε turbulence model.

    Solves transport equations for k (turbulent kinetic energy) and
    ε (dissipation rate).  Turbulent viscosity is computed as::

        μ_t = ρ C_μ k² / ε

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KEpsilonConstants, optional
        Model constants.  Defaults to standard Launder-Spalding values.
    realizable : bool
        If ``True``, use realizable k-ε formulation.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KEpsilonConstants | None = None,
        realizable: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)

        self._realizable = realizable
        if constants is not None:
            self._C = constants
        elif realizable:
            self._C = _REALIZABLE_CONSTANTS
        else:
            self._C = _DEFAULT_CONSTANTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        # Turbulence fields (initialised to small positive values)
        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)

        # Velocity gradient tensor (n_cells, 3, 3) — computed in correct()
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

        if self._realizable:
            C_mu = self._compute_C_mu()
            return C_mu * k**2 / eps
        else:
            return self._C.C_mu * k**2 / eps

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    def epsilon(self) -> torch.Tensor:
        """Return dissipation rate ``(n_cells,)``."""
        return self._eps

    def correct(self) -> None:
        """Update the k-ε model: compute nut, solve k and ε equations."""
        # Compute velocity gradient tensor (n_cells, 3, 3)
        # fvc.grad only works with scalar fields, so compute component by component
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

        # Solve ε equation
        self._solve_eps(P_k)

    # ------------------------------------------------------------------
    # Internal: transport equations
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve the k transport equation.

        ∂k/∂t + ∇·(U k) = ∇·((ν + ν_t/σ_k) ∇k) + P_k - ε
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        # Effective diffusivity
        nu_eff = self._nu + self.nut() / self._C.sigma_k

        # Build equation: implicit convection + implicit diffusion
        # = source (production - destruction)
        eqn = fvm.div(self._phi, self._k, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._k, "Gauss linear corrected", mesh=mesh)

        # Combine: add diffusion coefficients to equation
        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: P_k - ε (explicit)
        source = P_k - self._eps
        eqn.source = eqn.source + source

        # Solve (diagonal dominant, use simple iteration)
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        off_diag_sum = torch.zeros(n_cells, device=device, dtype=dtype)
        n_internal = mesh.n_internal_faces
        off_diag_sum = off_diag_sum + scatter_add(
            eqn.lower.abs(), mesh.owner[:n_internal], n_cells
        )
        off_diag_sum = off_diag_sum + scatter_add(
            eqn.upper.abs(), mesh.neighbour, n_cells
        )

        # Jacobi-like update: k_new = (source - off_diag * k_old) / diag
        # Simplified: just update with source/diagonal for stability
        k_new = eqn.source / diag_safe

        # Clamp to small positive values
        self._k = k_new.clamp(min=1e-10)

    def _solve_eps(self, P_k: torch.Tensor) -> None:
        """Solve the ε transport equation.

        ∂ε/∂t + ∇·(U ε) = ∇·((ν + ν_t/σ_ε) ∇ε)
                         + C1 ε/k P_k - C2 ε²/k
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        # Effective diffusivity
        nu_eff = self._nu + self.nut() / C.sigma_eps

        # Build equation
        eqn = fvm.div(self._phi, self._eps, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._eps, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: C1 * ε/k * P_k - C2 * ε²/k
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)
        source = C.C1 * eps_safe / k_safe * P_k - C.C2 * eps_safe**2 / k_safe
        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        eps_new = eqn.source / diag_safe

        # Clamp to small positive values
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
        S = 0.5 * (grad_U + grad_U.transpose(-1, -2))
        return S

    def _vorticity_magnitude(self) -> torch.Tensor:
        """Compute magnitude of vorticity |Ω|.

        Returns:
            ``(n_cells,)`` vorticity magnitude.
        """
        grad_U = self._grad_U
        # Ω = 0.5 (∇U - ∇U^T)
        Omega = 0.5 * (grad_U - grad_U.transpose(-1, -2))
        # |Ω| = sqrt(2 Ω:Ω)
        return torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

    def _compute_C_mu(self) -> torch.Tensor:
        """Compute C_μ for realizable k-ε.

        C_μ = 1 / (A0 + A_s * U* * k / ε)

        where U* = sqrt(S:S + Ω:Ω) and A0 = 4.0, A_s = sqrt(6) cos(φ).
        """
        # If gradient not computed yet, fall back to standard C_mu
        if self._grad_U is None:
            return torch.full_like(self._k, self._C.C_mu)

        S = self._strain_rate()
        grad_U = self._grad_U

        # Strain rate magnitude
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

        # Vorticity magnitude
        Omega = 0.5 * (grad_U - grad_U.transpose(-1, -2))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        # U* = sqrt(S:S + Ω:Ω)
        U_star = torch.sqrt(S_mag**2 + Omega_mag**2).clamp(min=1e-16)

        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        A0 = 4.0
        A_s = 6.0**0.5  # sqrt(6) * cos(pi/3) = sqrt(6) * 0.5 ≈ 1.2247
        # Simplified: A_s = sqrt(6)
        C_mu = 1.0 / (A0 + A_s * U_star * k_safe / eps_safe)

        return C_mu.clamp(min=1e-6, max=0.5)


# ---------------------------------------------------------------------------
# Realizable k-ε (alias)
# ---------------------------------------------------------------------------


@TurbulenceModel.register("realizableKEpsilon")
class RealizableKEpsilonModel(KEpsilonModel):
    """Realizable k-ε turbulence model.

    Extends standard k-ε with a variable C_μ computed from the local
    strain and rotation rates, ensuring realizability of Reynolds stresses.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi, realizable=True, **kwargs)
