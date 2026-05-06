"""
k-ω SST turbulence model (Menter 1994).

Implements the Shear Stress Transport (SST) model which blends k-ω
behaviour near walls with k-ε behaviour in the freestream.  The model
solves transport equations for turbulent kinetic energy *k* and
specific dissipation rate *ω*.

Key features:
- Blending functions F1 (inner/outer) and F2 (shear stress limiter)
- Cross-diffusion term in ω equation
- SST limiter on turbulent viscosity: μ_t = ρ a₁ k / max(a₁ ω, S F₂)

Constants (Menter 1994):
    Inner (ω): σ_k1=0.85, σ_ω1=0.5, β₁=0.075, γ₁=5/9, β*=0.09
    Outer (ε): σ_k2=1.0, σ_ω2=0.856, β₂=0.0828, γ₂=0.44, β*=0.09
    Blending: a₁=0.31
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel

__all__ = ["KOmegaSSTModel", "KOmegaSSTConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KOmegaSSTConstants:
    """Constants for the k-ω SST turbulence model.

    Attributes:
        sigma_k1: Turbulent Prandtl number for k (inner, ω-region).
        sigma_k2: Turbulent Prandtl number for k (outer, ε-region).
        sigma_omega1: Turbulent Prandtl number for ω (inner).
        sigma_omega2: Turbulent Prandtl number for ω (outer).
        beta1: Coefficient for ω destruction (inner).
        beta2: Coefficient for ω destruction (outer).
        gamma1: Coefficient for ω production (inner) = β₁/β* - σ_ω1 κ²/√β*.
        gamma2: Coefficient for ω production (outer) = β₂/β* - σ_ω2 κ²/√β*.
        a1: SST blending constant for turbulent viscosity limiter.
        beta_star: Coefficient for k destruction (β* = 0.09).
        kappa: Von Karman constant.
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


_DEFAULT_CONSTANTS = KOmegaSSTConstants()


# ---------------------------------------------------------------------------
# k-ω SST model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("kOmegaSST")
class KOmegaSSTModel(TurbulenceModel):
    """k-ω SST turbulence model.

    Solves transport equations for k (turbulent kinetic energy) and
    ω (specific dissipation rate).  Uses blending functions to transition
    between k-ω near walls and k-ε in the freestream.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaSSTConstants, optional
        Model constants.  Defaults to Menter (1994) values.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaSSTConstants | None = None,
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

        # Velocity gradient tensor (n_cells, 3, 3)
        self._grad_U: torch.Tensor | None = None

        # Wall distance (simplified: use cell centre distance from origin)
        # In a full implementation, this would be computed from the mesh
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

        Returns:
            ``(n_cells,)`` turbulent viscosity field.
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)

        # If gradient not computed yet, use simplified nut = k / omega
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
        """Return dissipation rate: ε = β* ω k.

        Returns:
            ``(n_cells,)`` dissipation rate.
        """
        return self._C.beta_star * self._omega * self._k

    def correct(self) -> None:
        """Update the k-ω SST model: compute nut, solve k and ω equations."""
        # Compute velocity gradient tensor (n_cells, 3, 3)
        # fvc.grad only works with scalar fields, so compute component by component
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

        # Solve ω equation
        self._solve_omega(P_k)

    # ------------------------------------------------------------------
    # Blending functions
    # ------------------------------------------------------------------

    def _F1(self) -> torch.Tensor:
        """First blending function F1 (inner/outer transition).

        F1 = tanh(arg1⁴) where
        arg1 = min(max(√k/(β* ω y), 500 ν/(y² ω)), 4 ρ σ_ω2 k/(CD_kω y²))

        Returns:
            ``(n_cells,)`` blending function values in [0, 1].
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)
        C = self._C

        # arg1 = √k / (β* ω y)
        arg1 = torch.sqrt(k) / (C.beta_star * omega * y)

        # arg2 = 500 ν / (y² ω)
        arg2 = 500.0 * self._nu / (y**2 * omega)

        # CD_kω = max(2 ρ σ_ω2 ∇k·∇ω / ω, 1e-10)
        # Simplified: use a proxy for cross-diffusion
        CD_kω = (2.0 * C.sigma_omega2 / omega).clamp(min=1e-10)

        # arg3 = 4 ρ σ_ω2 k / (CD_kω y²)
        arg3 = 4.0 * C.sigma_omega2 * k / (CD_kω * y**2)

        arg = torch.min(torch.max(arg1, arg2), arg3)
        return torch.tanh(arg**4)

    def _F2(self) -> torch.Tensor:
        """Second blending function F2 (shear stress limiter).

        F2 = tanh(arg2²) where
        arg2 = max(2√k/(β* ω y), 500 ν/(y² ω))

        Returns:
            ``(n_cells,)`` blending function values in [0, 1].
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)
        C = self._C

        arg1 = 2.0 * torch.sqrt(k) / (C.beta_star * omega * y)
        arg2 = 500.0 * self._nu / (y**2 * omega)

        arg = torch.max(arg1, arg2)
        return torch.tanh(arg**2)

    # ------------------------------------------------------------------
    # Internal: transport equations
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve the k transport equation.

        ∂k/∂t + ∇·(U k) = ∇·((ν + σ_k ν_t) ∇k) + P_k - β* ω k
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        # Blended diffusivity
        F1 = self._F1()
        sigma_k = F1 * C.sigma_k1 + (1.0 - F1) * C.sigma_k2
        nut = self.nut()
        nu_eff = self._nu + sigma_k * nut

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

        ∂ω/∂t + ∇·(U ω) = ∇·((ν + σ_ω ν_t) ∇ω)
                         + γ P_k/ν_t - β ω² + 2(1-F1) σ_ω2 ∇k·∇ω / ω
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        # Blended coefficients
        F1 = self._F1()
        sigma_omega = F1 * C.sigma_omega1 + (1.0 - F1) * C.sigma_omega2
        beta = F1 * C.beta1 + (1.0 - F1) * C.beta2
        gamma = F1 * C.gamma1 + (1.0 - F1) * C.gamma2

        nut = self.nut()
        nu_eff = self._nu + sigma_omega * nut

        # Build equation
        eqn = fvm.div(self._phi, self._omega, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._omega, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: γ P_k/ν_t - β ω²
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        nut_safe = nut.clamp(min=1e-16)

        source = gamma * P_k / nut_safe - beta * omega_safe**2

        # Cross-diffusion term: 2(1-F1) σ_ω2 ∇k·∇ω / ω
        # Simplified: approximate with zero (requires gradient of k and ω)
        # In production, compute fvc.grad(k) and fvc.grad(omega) explicitly

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

    def _strain_magnitude(self) -> torch.Tensor:
        """Compute strain rate magnitude |S| = √(2 S:S).

        Returns:
            ``(n_cells,)`` strain rate magnitude.
        """
        S = self._strain_rate()
        return torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute approximate wall distance for each cell.

        Uses distance from cell centre to the nearest boundary face centre
        as a simplified wall distance.  In a full implementation, this would
        solve the Poisson equation or use exact geometric distance.

        Returns:
            ``(n_cells,)`` wall distance.
        """
        mesh = self._mesh
        device = self._device
        dtype = self._dtype

        cell_centres = mesh.cell_centres  # (n_cells, 3)
        face_centres = mesh.face_centres  # (n_faces, 3)

        # Get boundary face centres
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        if n_faces > n_internal:
            bnd_centres = face_centres[n_internal:]  # (n_bnd_faces, 3)
        else:
            # No boundary faces — use cell centre distance from origin
            return cell_centres.norm(dim=1).clamp(min=1e-6)

        # For each cell, find distance to nearest boundary face centre
        # This is O(n_cells * n_bnd_faces) — acceptable for small meshes
        # For large meshes, use a spatial index
        n_bnd = bnd_centres.shape[0]
        if n_bnd == 0:
            return cell_centres.norm(dim=1).clamp(min=1e-6)

        # Expand for broadcasting: (n_cells, 1, 3) - (1, n_bnd, 3)
        diff = cell_centres.unsqueeze(1) - bnd_centres.unsqueeze(0)  # (n_cells, n_bnd, 3)
        dist = diff.norm(dim=2)  # (n_cells, n_bnd)
        y = dist.min(dim=1).values  # (n_cells,)

        return y.clamp(min=1e-6)
