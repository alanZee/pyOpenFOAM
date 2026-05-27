"""
SSG (Speziale-Sarkar-Gatski) Reynolds stress model.

Implements a full Reynolds stress model that solves transport equations for
each component of the Reynolds stress tensor R_ij = <u'_i u'_j>.  The
pressure-strain correlation uses the SSG (nonlinear) model which is quadratic
in the anisotropy tensor b_ij.

Transport equations (for each independent R_ij component):
    DR_ij/Dt = P_ij + Phi_ij - (2/3) eps delta_ij
             + d/dx_k [(nu + C_s k/eps R_km) d R_ij / dx_m]

Production:
    P_ij = -(R_ik dU_j/dx_k + R_jk dU_i/dx_k)

Pressure-strain (SSG):
    Phi_ij = Phi_ij^(s) + Phi_ij^(r) + Phi_ij^(w)

    Slow (return-to-isotropy):
        Phi_ij^(s) = -eps [C1 b_ij + C1* (b_ik b_kj - 1/3 II_b delta_ij)]

    Fast (rapid):
        Phi_ij^(r) = -k [C2 S_ij + C3 (b_ik S_kj + S_ik b_kj - 2/3 b_km S_km delta_ij)
                        + C4 (b_ik W_kj - W_ik b_kj)]

    where:
        b_ij = R_ij/(2k) - delta_ij/3  (anisotropy tensor)
        S_ij = 0.5 (dU_i/dx_j + dU_j/dx_i)  (strain rate)
        W_ij = 0.5 (dU_i/dx_j - dU_j/dx_i)  (rotation rate)
        II_b = b_ij b_jk  (second invariant of anisotropy tensor)

Dissipation equation:
    Deps/Dt = Ceps1 P_k eps/k - Ceps2 eps^2/k
             + d/dx_k [(nu + C_s k/eps R_km) d eps / dx_m]

Eddy viscosity (for coupling with mean flow):
    nut = Cmu k^2 / eps

Constants (Speziale, Sarkar & Gatski 1991):
    C1 = 3.4, C1* = 1.8, C2 = 4.2, C3 = 0.8, C4 = 1.2, C5 = 0.4
    Ceps1 = 1.44, Ceps2 = 1.06, Cmu = 0.09, Cs = 0.22

References
----------
Speziale, C.G., Sarkar, S. & Gatski, T.B. (1991). Modelling the
pressure-strain correlation of turbulence: an invariant dynamical
systems approach. Journal of Fluid Mechanics, 227, 245-272.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel

__all__ = ["SSGModel", "SSGConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SSGConstants:
    """Constants for the SSG Reynolds stress model.

    Attributes:
        C1: Slow pressure-strain linear coefficient (default: 3.4).
        C1star: Slow pressure-strain nonlinear coefficient (default: 1.8).
        C2: Fast pressure-strain strain-rate coefficient (default: 4.2).
        C3: Fast pressure-strain anisotropy-strain coefficient (default: 0.8).
        C4: Fast pressure-strain anisotropy-rotation coefficient (default: 1.2).
        C5: Fast pressure-strain trace coefficient (default: 0.4).
        Ceps1: Dissipation production coefficient (default: 1.44).
        Ceps2: Dissipation destruction coefficient (default: 1.06).
        Cmu: Eddy-viscosity constant (default: 0.09).
        Cs: Diffusion coefficient (default: 0.22).
        sigma_eps: Turbulent Prandtl number for epsilon (default: 1.0).
    """

    C1: float = 3.4
    C1star: float = 1.8
    C2: float = 4.2
    C3: float = 0.8
    C4: float = 1.2
    C5: float = 0.4
    Ceps1: float = 1.44
    Ceps2: float = 1.06
    Cmu: float = 0.09
    Cs: float = 0.22
    sigma_eps: float = 1.0


_DEFAULT_CONSTANTS = SSGConstants()


# Stress component index mapping (upper triangle of symmetric 3x3)
# (i, j) -> flat index: (0,0)=0, (1,1)=1, (2,2)=2, (0,1)=3, (0,2)=4, (1,2)=5
_COMPONENTS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


# ---------------------------------------------------------------------------
# SSG Reynolds stress model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("SSG")
class SSGModel(TurbulenceModel):
    """SSG Reynolds stress model.

    Solves transport equations for each component of the Reynolds stress
    tensor R_ij and dissipation rate epsilon.  Uses the nonlinear SSG
    pressure-strain correlation model.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : SSGConstants, optional
        Model constants.  Defaults to Speziale-Sarkar-Gatski values.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: SSGConstants | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)

        self._C = constants or _DEFAULT_CONSTANTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        # Reynolds stress tensor R_ij (n_cells, 3, 3)
        # Initialise to isotropic: R_ij = 2/3 k_0 delta_ij
        k0 = 1e-4
        self._R = torch.zeros(n_cells, 3, 3, device=device, dtype=dtype)
        self._R[:, 0, 0] = 2.0 / 3.0 * k0
        self._R[:, 1, 1] = 2.0 / 3.0 * k0
        self._R[:, 2, 2] = 2.0 / 3.0 * k0

        # Dissipation rate
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)

        # Velocity gradient tensor (n_cells, 3, 3)
        self._grad_U: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def R_field(self) -> torch.Tensor:
        """Reynolds stress tensor R_ij ``(n_cells, 3, 3)``."""
        return self._R

    @R_field.setter
    def R_field(self, value: torch.Tensor) -> None:
        self._R = value.to(device=self._device, dtype=self._dtype)

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
        """Turbulent viscosity: nut = Cmu k^2 / eps.

        Returns:
            ``(n_cells,)`` turbulent viscosity field.
        """
        k = self.k().clamp(min=1e-16)
        eps = self._eps.clamp(min=1e-16)
        return self._C.Cmu * k**2 / eps

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy: k = 0.5 * trace(R).

        Returns:
            ``(n_cells,)`` turbulent kinetic energy.
        """
        return 0.5 * (self._R[:, 0, 0] + self._R[:, 1, 1] + self._R[:, 2, 2]).clamp(
            min=1e-16
        )

    def epsilon(self) -> torch.Tensor:
        """Return dissipation rate ``(n_cells,)``."""
        return self._eps

    def devReff(self) -> torch.Tensor:
        """Return effective deviatoric Reynolds stress.

        tau_eff = nu_t * (gradU + gradU^T) - (2/3) k I

        Returns:
            ``(n_cells, 3, 3)`` effective deviatoric stress.
        """
        nut = self.nut().unsqueeze(-1).unsqueeze(-1)  # (n_cells, 1, 1)
        k = self.k()

        if self._grad_U is not None:
            S = self._strain_rate()
        else:
            S = torch.zeros_like(self._R)

        tau = nut * 2.0 * S
        # Subtract isotropic part
        tau[:, 0, 0] -= 2.0 / 3.0 * k
        tau[:, 1, 1] -= 2.0 / 3.0 * k
        tau[:, 2, 2] -= 2.0 / 3.0 * k
        return tau

    def correct(self) -> None:
        """Update the SSG model: compute gradients, solve R and eps equations."""
        mesh = self._mesh
        device = self._device
        dtype = self._dtype

        # Compute velocity gradient tensor (n_cells, 3, 3)
        grad_U = torch.zeros(mesh.n_cells, 3, 3, device=device, dtype=dtype)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=mesh
            )
        self._grad_U = grad_U

        # Solve R_ij transport equations
        self._solve_R()

        # Solve epsilon transport equation
        self._solve_eps()

    # ------------------------------------------------------------------
    # Reynolds stress transport
    # ------------------------------------------------------------------

    def _solve_R(self) -> None:
        """Solve transport equations for each R_ij component.

        DR_ij/Dt = P_ij + Phi_ij - (2/3) eps delta_ij
                 + diffusion
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        k = self.k()
        eps_safe = self._eps.clamp(min=1e-16)
        k_safe = k.clamp(min=1e-16)

        # Compute anisotropy tensor b_ij
        b = self._compute_anisotropy()

        # Compute strain and rotation rate tensors
        S = self._strain_rate()       # (n_cells, 3, 3)
        W = self._rotation_rate()     # (n_cells, 3, 3)

        # Compute production tensor P_ij
        P = self._production_tensor()

        # Compute pressure-strain tensor Phi_ij
        Phi = self._pressure_strain(b, S, W, k_safe, eps_safe)

        # Turbulent diffusivity: Cs * k/eps * R_ij (used as eddy diffusivity)
        # For diffusion of R_ij, we use scalar Cs * k/eps applied to each component
        nut_diff = C.Cs * k_safe / eps_safe  # (n_cells,)

        # Solve for each independent component
        new_R = self._R.clone()
        for ci, (i, j) in enumerate(_COMPONENTS):
            Rij = self._R[:, i, j]

            # Convection
            eqn = fvm.div(self._phi, Rij, "Gauss upwind", mesh=mesh)

            # Diffusion: d/dx_k [Cs k/eps R_km d R_ij / dx_m]
            # Simplified: use scalar diffusivity Cs * k / eps
            diff = fvm.laplacian(
                nut_diff, Rij, "Gauss linear corrected", mesh=mesh
            )
            eqn.lower = eqn.lower + diff.lower
            eqn.upper = eqn.upper + diff.upper
            eqn.diag = eqn.diag + diff.diag

            # Source: P_ij + Phi_ij - (2/3) eps delta_ij
            source = P[:, i, j] + Phi[:, i, j]
            if i == j:
                source = source - 2.0 / 3.0 * self._eps

            eqn.source = eqn.source + source

            # Solve
            diag_safe = eqn.diag.abs().clamp(min=1e-30)
            Rij_new = eqn.source / diag_safe
            new_R[:, i, j] = Rij_new

            # Mirror symmetric component
            if i != j:
                new_R[:, j, i] = Rij_new

        # Enforce realizability: ensure diagonal components are non-negative
        for i in range(3):
            new_R[:, i, i] = new_R[:, i, i].clamp(min=1e-16)

        self._R = new_R

    def _solve_eps(self) -> None:
        """Solve the epsilon transport equation.

        Deps/Dt = Ceps1 P_k eps/k - Ceps2 eps^2/k + diffusion

        where P_k = 0.5 * trace(production) = 0.5 * sum P_ii.
        """
        mesh = self._mesh
        C = self._C

        k_safe = self.k().clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        # Turbulent diffusivity
        nut_diff = C.Cs * k_safe / eps_safe

        # Effective diffusivity for epsilon
        nu_eff = self._nu + nut_diff / C.sigma_eps

        # Convection + diffusion
        eqn = fvm.div(self._phi, self._eps, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._eps, "Gauss linear corrected", mesh=mesh
        )
        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Production rate: P_k = -2 R_ik S_ik (trace of production tensor)
        P = self._production_tensor()
        P_k = -(P[:, 0, 0] + P[:, 1, 1] + P[:, 2, 2]).clamp(min=0.0)

        # Source: Ceps1 P_k eps/k - Ceps2 eps^2/k
        source = C.Ceps1 * P_k * eps_safe / k_safe - C.Ceps2 * eps_safe**2 / k_safe
        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        eps_new = eqn.source / diag_safe
        self._eps = eps_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Tensor computations
    # ------------------------------------------------------------------

    def _strain_rate(self) -> torch.Tensor:
        """Strain rate tensor S_ij = 0.5 (dU_i/dx_j + dU_j/dx_i).

        Returns:
            ``(n_cells, 3, 3)`` strain rate tensor.
        """
        grad_U = self._grad_U
        return 0.5 * (grad_U + grad_U.transpose(-1, -2))

    def _rotation_rate(self) -> torch.Tensor:
        """Rotation (vorticity) tensor W_ij = 0.5 (dU_i/dx_j - dU_j/dx_i).

        Returns:
            ``(n_cells, 3, 3)`` rotation tensor.
        """
        grad_U = self._grad_U
        return 0.5 * (grad_U - grad_U.transpose(-1, -2))

    def _compute_anisotropy(self) -> torch.Tensor:
        """Anisotropy tensor b_ij = R_ij/(2k) - delta_ij/3.

        Returns:
            ``(n_cells, 3, 3)`` anisotropy tensor.
        """
        k = self.k().clamp(min=1e-16)  # (n_cells,)
        b = self._R / (2.0 * k.unsqueeze(-1).unsqueeze(-1))
        b[:, 0, 0] -= 1.0 / 3.0
        b[:, 1, 1] -= 1.0 / 3.0
        b[:, 2, 2] -= 1.0 / 3.0
        return b

    def _production_tensor(self) -> torch.Tensor:
        """Production tensor P_ij = -(R_ik dU_j/dx_k + R_jk dU_i/dx_k).

        Returns:
            ``(n_cells, 3, 3)`` production tensor.
        """
        grad_U = self._grad_U  # (n_cells, 3, 3): grad_U[:, i, j] = dU_i/dx_j
        R = self._R

        # P_ij = -(R_ik G_jk + R_jk G_ik) where G_jk = dU_j/dx_k
        P = -(torch.bmm(R, grad_U.transpose(-1, -2))
              + torch.bmm(grad_U, R.transpose(-1, -2)))
        return P

    def _pressure_strain(
        self,
        b: torch.Tensor,
        S: torch.Tensor,
        W: torch.Tensor,
        k: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """SSG pressure-strain correlation tensor Phi_ij.

        Phi_ij = Phi_ij^(s) + Phi_ij^(r)

        Slow:  Phi^(s) = -eps [C1 b + C1* (b^2 - 1/3 II_b I)]
        Fast:  Phi^(r) = -k [C2 S + C3 (b S + S b - 2/3 b:S I)
                            + C4 (b W - W b)]

        Args:
            b: Anisotropy tensor ``(n_cells, 3, 3)``.
            S: Strain rate tensor ``(n_cells, 3, 3)``.
            W: Rotation rate tensor ``(n_cells, 3, 3)``.
            k: Turbulent kinetic energy ``(n_cells,)``.
            eps: Dissipation rate ``(n_cells,)``.

        Returns:
            ``(n_cells, 3, 3)`` pressure-strain tensor.
        """
        C = self._C
        device = self._device
        dtype = self._dtype
        n_cells = b.shape[0]

        # Second invariant: II_b = b_ij b_ji = trace(b^2)
        b2 = torch.bmm(b, b)  # (n_cells, 3, 3)
        II_b = b2[:, 0, 0] + b2[:, 1, 1] + b2[:, 2, 2]  # (n_cells,)

        # ---- Slow pressure-strain (return-to-isotropy) ----
        # Phi^(s) = -eps [C1 b_ij + C1* (b_ik b_kj - 1/3 II_b delta_ij)]
        Phi_slow = -eps.unsqueeze(-1).unsqueeze(-1) * (
            C.C1 * b
            + C.C1star * (
                b2
                - (1.0 / 3.0) * II_b.unsqueeze(-1).unsqueeze(-1)
                * torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            )
        )

        # ---- Fast pressure-strain (rapid) ----
        # b_ik S_kj + S_ik b_kj
        bS = torch.bmm(b, S)
        Sb = torch.bmm(S, b)

        # b:S = b_ik S_ki (contraction)
        bS_trace = (b * S).sum(dim=(1, 2))  # (n_cells,)

        # b_ik W_kj - W_ik b_kj
        bW = torch.bmm(b, W)
        Wb = torch.bmm(W, b)

        I3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # (1, 3, 3)

        Phi_fast = -k.unsqueeze(-1).unsqueeze(-1) * (
            C.C2 * S
            + C.C3 * (
                bS + Sb
                - (2.0 / 3.0) * bS_trace.unsqueeze(-1).unsqueeze(-1) * I3
            )
            + C.C4 * (bW - Wb)
        )

        return Phi_slow + Phi_fast
