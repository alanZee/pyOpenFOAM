"""
Launder-Sharma low-Re k-ε turbulence model (Launder & Sharma 1974).

Implements the low-Reynolds-number k-ε model with wall-damping
functions suitable for resolving the viscous sublayer directly.

The model modifies the standard k-ε with:
- Additional destruction term in ε equation: -2 ν (∂√k/∂x_j)²
- Wall-damping functions f_μ, f_1, f_2
- Modified ε̃ = ε - 2 ν (∂√k/∂x_j)²

Turbulent viscosity:
    μ_t = ρ C_μ f_μ k² / ε̃

where:
    f_μ = exp(-3.4 / (1 + Re_t/50)²)
    f_1 = 1.0
    f_2 = 1 - 0.3 exp(-Re_t²)

References
----------
Launder, B.E. & Sharma, B.I. (1974). Application of the energy-
dissipation model of turbulence to the calculation of flow near a
spinning disc. Letters in Heat and Mass Transfer, 1(2), 131–138.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel

__all__ = ["LaunderSharmaKEModel", "LaunderSharmaKEConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunderSharmaKEConstants:
    """Constants for the Launder-Sharma low-Re k-ε model.

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


_DEFAULT_CONSTANTS = LaunderSharmaKEConstants()


# ---------------------------------------------------------------------------
# Launder-Sharma model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("LaunderSharmaKE")
class LaunderSharmaKEModel(TurbulenceModel):
    """Launder-Sharma low-Re k-ε turbulence model.

    Low-Reynolds-number variant of k-ε with wall-damping functions.
    Suitable for resolving the viscous sublayer without wall functions.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : LaunderSharmaKEConstants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: LaunderSharmaKEConstants | None = None,
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

        # Wall distance
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
        """Turbulent viscosity: μ_t = C_μ f_μ k² / ε̃.

        Returns:
            ``(n_cells,)`` turbulent viscosity field.
        """
        k = self._k.clamp(min=1e-16)
        eps = self._eps_tilde().clamp(min=1e-16)

        f_mu = self._f_mu()
        return self._C.C_mu * f_mu * k**2 / eps

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    def epsilon(self) -> torch.Tensor:
        """Return dissipation rate ``(n_cells,)``."""
        return self._eps

    def correct(self) -> None:
        """Update the Launder-Sharma model."""
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

        # Solve ε equation
        self._solve_eps(P_k)

    # ------------------------------------------------------------------
    # Wall-damping functions
    # ------------------------------------------------------------------

    def _Re_t(self) -> torch.Tensor:
        """Turbulent Reynolds number: Re_t = k² / (ν ε).

        Returns:
            ``(n_cells,)`` turbulent Reynolds number.
        """
        k = self._k.clamp(min=1e-16)
        eps = self._eps.clamp(min=1e-16)
        return k**2 / (self._nu * eps)

    def _f_mu(self) -> torch.Tensor:
        """Wall-damping function f_μ.

        f_μ = exp(-3.4 / (1 + Re_t/50)²)

        Returns:
            ``(n_cells,)`` damping function values.
        """
        Re_t = self._Re_t()
        return torch.exp(-3.4 / (1.0 + Re_t / 50.0) ** 2)

    def _f_1(self) -> torch.Tensor:
        """Wall-damping function f_1 (= 1.0 for Launder-Sharma).

        Returns:
            ``(n_cells,)`` ones.
        """
        return torch.ones_like(self._k)

    def _f_2(self) -> torch.Tensor:
        """Wall-damping function f_2.

        f_2 = 1 - 0.3 exp(-Re_t²)

        Returns:
            ``(n_cells,)`` damping function values.
        """
        Re_t = self._Re_t()
        return 1.0 - 0.3 * torch.exp(-Re_t**2)

    def _eps_tilde(self) -> torch.Tensor:
        """Modified dissipation rate: ε̃ = ε - 2ν(∂√k/∂x_j)².

        Returns:
            ``(n_cells,)`` modified dissipation rate.
        """
        k_safe = self._k.clamp(min=1e-16)
        sqrt_k = torch.sqrt(k_safe)

        # Compute gradient of √k
        grad_sqrt_k = fvc.grad(sqrt_k, "Gauss linear", mesh=self._mesh)

        # (∂√k/∂x_j)² = sum of squares
        grad_sqrt_k_sq = (grad_sqrt_k * grad_sqrt_k).sum(dim=1)

        eps_tilde = self._eps - 2.0 * self._nu * grad_sqrt_k_sq
        return eps_tilde.clamp(min=1e-16)

    # ------------------------------------------------------------------
    # Internal: transport equations
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve the k transport equation.

        ∂k/∂t + ∇·(U k) = ∇·((ν + ν_t/σ_k) ∇k) + P_k - ε̃
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        # Effective diffusivity
        nu_eff = self._nu + self.nut() / C.sigma_k

        # Build equation
        eqn = fvm.div(self._phi, self._k, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._k, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: P_k - ε̃
        source = P_k - self._eps_tilde()
        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    def _solve_eps(self, P_k: torch.Tensor) -> None:
        """Solve the ε transport equation.

        ∂ε/∂t + ∇·(U ε) = ∇·((ν + ν_t/σ_ε) ∇ε)
                         + f_1 C1 ε/k P_k - f_2 C2 ε²/k
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

        # Source: f_1 C1 ε/k P_k - f_2 C2 ε²/k
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)
        f1 = self._f_1()
        f2 = self._f_2()

        source = (
            f1 * C.C1 * eps_safe / k_safe * P_k
            - f2 * C.C2 * eps_safe**2 / k_safe
        )
        eqn.source = eqn.source + source

        # Solve
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

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute approximate wall distance for each cell.

        Returns:
            ``(n_cells,)`` wall distance.
        """
        mesh = self._mesh
        device = self._device
        dtype = self._dtype

        cell_centres = mesh.cell_centres
        face_centres = mesh.face_centres

        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        if n_faces > n_internal:
            bnd_centres = face_centres[n_internal:]
        else:
            return cell_centres.norm(dim=1).clamp(min=1e-6)

        n_bnd = bnd_centres.shape[0]
        if n_bnd == 0:
            return cell_centres.norm(dim=1).clamp(min=1e-6)

        diff = cell_centres.unsqueeze(1) - bnd_centres.unsqueeze(0)
        dist = diff.norm(dim=2)
        y = dist.min(dim=1).values

        return y.clamp(min=1e-6)
