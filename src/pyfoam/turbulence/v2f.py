"""
v²-f turbulence model (Durbin 1995, 2011).

Implements the v²-f (vortex-drag / elliptic relaxation) model which
extends the k-ε framework with a velocity scale variable v² and an
elliptic relaxation function f.

Transport equations:
    ∂k/∂t + ∇·(U k) = ∇·((ν + ν_t/σ_k) ∇k) + P_k - ε

    ∂ε/∂t + ∇·(U ε) = ∇·((ν + ν_t/σ_ε) ∇ε)
                     + C1 (P_k - ε) / T + C2 ε² / T

    ∂v²/∂t + ∇·(U v²) = ∇·((ν + ν_t/σ_k) ∇v²)
                       + k f - 6 v² ε / k

    L² ∇²f - f = (C1 - 1) (2/3 - v²/k) / T - C2 P_k / k

Turbulent viscosity:
    μ_t = ρ C_μ v² T

where T = max(k/ε, 6√(ν/ε)) is the turbulent time scale.

References
----------
Durbin, P.A. (1995). Near-wall turbulence closure modeling without
"damping functions". Theoretical and Computational Fluid Dynamics,
8, 1–13.

Laurence, D.R., Uribe, J.C. & Utyuzhnikov, S.V. (2005). A robust
formulation of the v²-f model. Flow, Turbulence and Combustion, 73,
169–185.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel

__all__ = ["V2FModel", "V2FConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V2FConstants:
    """Constants for the v²-f turbulence model.

    Attributes:
        C_mu: Eddy-viscosity constant (default: 0.22).
        C1: Time-scale coefficient (default: 1.4).
        C2: Time-scale coefficient (default: 0.3).
        C_L: Length-scale coefficient (default: 0.23).
        C_eta: Length-scale coefficient (default: 70.0).
        sigma_k: Turbulent Prandtl number for k (default: 1.0).
        sigma_eps: Turbulent Prandtl number for ε (default: 1.3).
    """

    C_mu: float = 0.22
    C1: float = 1.4
    C2: float = 0.3
    C_L: float = 0.23
    C_eta: float = 70.0
    sigma_k: float = 1.0
    sigma_eps: float = 1.3


_DEFAULT_CONSTANTS = V2FConstants()


# ---------------------------------------------------------------------------
# v²-f model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("v2f")
class V2FModel(TurbulenceModel):
    """v²-f turbulence model (Durbin 1995).

    Extends the k-ε framework with a velocity scale v² and an elliptic
    relaxation function f.  Provides correct near-wall behaviour
    without wall-damping functions.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : V2FConstants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: V2FConstants | None = None,
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
        self._v2 = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._f = torch.zeros(n_cells, device=device, dtype=dtype)

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

    @property
    def v2_field(self) -> torch.Tensor:
        """Velocity scale v² ``(n_cells,)``."""
        return self._v2

    @v2_field.setter
    def v2_field(self, value: torch.Tensor) -> None:
        self._v2 = value.to(device=self._device, dtype=self._dtype)

    @property
    def f_field(self) -> torch.Tensor:
        """Elliptic relaxation function f ``(n_cells,)``."""
        return self._f

    @f_field.setter
    def f_field(self, value: torch.Tensor) -> None:
        self._f = value.to(device=self._device, dtype=self._dtype)

    # ------------------------------------------------------------------
    # TurbulenceModel interface
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity: μ_t = C_μ v² T.

        Returns:
            ``(n_cells,)`` turbulent viscosity field.
        """
        T = self._time_scale()
        v2 = self._v2.clamp(min=0.0)
        return self._C.C_mu * v2 * T

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    def epsilon(self) -> torch.Tensor:
        """Return dissipation rate ``(n_cells,)``."""
        return self._eps

    def correct(self) -> None:
        """Update the v²-f model."""
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

        # Solve equations in order: k, ε, f (implicit), v²
        self._solve_k(P_k)
        self._solve_eps(P_k)
        self._solve_f()
        self._solve_v2()

    # ------------------------------------------------------------------
    # Turbulent scales
    # ------------------------------------------------------------------

    def _time_scale(self) -> torch.Tensor:
        """Turbulent time scale: T = max(k/ε, 6√(ν/ε)).

        Returns:
            ``(n_cells,)`` time scale.
        """
        k = self._k.clamp(min=1e-16)
        eps = self._eps.clamp(min=1e-16)

        T_k = k / eps
        T_eta = 6.0 * torch.sqrt(self._nu / eps)
        return torch.max(T_k, T_eta)

    def _length_scale(self) -> torch.Tensor:
        """Turbulent length scale: L = C_L max(k^{3/2}/ε, C_η ν^{3/4} / ε^{1/4}).

        Returns:
            ``(n_cells,)`` length scale.
        """
        C = self._C
        k = self._k.clamp(min=1e-16)
        eps = self._eps.clamp(min=1e-16)

        L_turb = k**1.5 / eps
        L_kolm = C.C_eta * (self._nu**0.75) / (eps**0.25)
        return C.C_L * torch.max(L_turb, L_kolm)

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
        """Solve the ε transport equation.

        ∂ε/∂t + ∇·(U ε) = ∇·((ν + ν_t/σ_ε) ∇ε)
                         + C1 (P_k - ε) / T + C2 ε² / T
        """
        mesh = self._mesh
        C = self._C

        nu_eff = self._nu + self.nut() / C.sigma_eps
        T = self._time_scale()

        eqn = fvm.div(self._phi, self._eps, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._eps, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)
        T_safe = T.clamp(min=1e-16)

        source = (
            C.C1 * (P_k - eps_safe) / T_safe
            - C.C2 * eps_safe**2 / T_safe
        )
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        eps_new = eqn.source / diag_safe
        self._eps = eps_new.clamp(min=1e-10)

    def _solve_f(self) -> None:
        """Solve the elliptic relaxation equation for f.

        L² ∇²f - f = (C1 - 1) (2/3 - v²/k) / T - C2 P_k / k

        Simplified: algebraic approximation (not solving Poisson equation).
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)
        v2_safe = self._v2.clamp(min=1e-16)
        T = self._time_scale().clamp(min=1e-16)
        L = self._length_scale().clamp(min=1e-16)

        # Production rate for f equation
        nut = self.nut()
        if self._grad_U is not None:
            S = self._strain_rate()
            P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))
        else:
            P_k = torch.zeros_like(k_safe)

        # RHS of elliptic equation
        rhs = (
            (C.C1 - 1.0) * (2.0 / 3.0 - v2_safe / k_safe) / T
            - C.C2 * P_k / k_safe
        )

        # Algebraic approximation: f = rhs / (1 + (L/d)²)
        # where d is wall distance
        y = self._y.clamp(min=1e-10)
        denom = 1.0 + (L / y) ** 2
        self._f = rhs / denom.clamp(min=1.0)

    def _solve_v2(self) -> None:
        """Solve the v² transport equation.

        ∂v²/∂t + ∇·(U v²) = ∇·((ν + ν_t/σ_k) ∇v²)
                           + k f - 6 v² ε / k
        """
        mesh = self._mesh
        C = self._C

        nu_eff = self._nu + self.nut() / C.sigma_k

        eqn = fvm.div(self._phi, self._v2, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(nu_eff, self._v2, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)
        v2_safe = self._v2.clamp(min=1e-16)

        source = k_safe * self._f - 6.0 * v2_safe * eps_safe / k_safe
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        v2_new = eqn.source / diag_safe
        self._v2 = v2_new.clamp(min=1e-10)

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
