"""
LRR (Launder-Reece-Rodi) Reynolds stress transport model.

Implements the LRR RSM (Reynolds Stress Model) which solves transport
equations for each component of the Reynolds stress tensor R_ij rather
than using the eddy-viscosity hypothesis.

Unlike eddy-viscosity models (k-ε, k-ω), the LRR model directly
computes the Reynolds stresses from their own transport equations,
allowing it to capture anisotropy effects such as streamline curvature
and swirl.

Transport equation for R_ij::

    ∂R_ij/∂t + ∇·(U R_ij) = ∇·((ν + ν_t/σ_k) ∇R_ij)
                             + P_ij + Φ_ij - (2/3) ε δ_ij

Transport equation for ε::

    ∂ε/∂t + ∇·(U ε) = ∇·((ν + ν_t/σ_ε) ∇ε)
                      + C1 P_k ε/k - C2 ε²/k

where:
- P_ij = -(R_ik ∂U_j/∂x_k + R_jk ∂U_i/∂x_k)  (production)
- Φ_ij = Φ_ij,1 + Φ_ij,2                          (pressure-strain)
- Φ_ij,1 = -C1 (ε/k) (R_ij - (2/3) k δ_ij)       (slow Rotta term)
- Φ_ij,2 = -C2 (P_ij - (2/3) P_k δ_ij)            (rapid IP model)
- k = 0.5 R_ii,  P_k = 0.5 P_ii

Constants (Launder, Reece & Rodi 1975):
    C1 = 1.8,  C2 = 0.6,  Cε1 = 1.44,  Cε2 = 1.92,
    Cμ = 0.09,  σ_k = 1.0,  σ_ε = 1.3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel

__all__ = ["LRRModel", "LRRConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LRRConstants:
    """Constants for the LRR Reynolds stress model.

    Attributes:
        C1: Slow pressure-strain (Rotta) constant.
        C2: Rapid pressure-strain constant.
        Ceps1: Production coefficient in ε equation.
        Ceps2: Destruction coefficient in ε equation.
        Cmu: Eddy-viscosity constant (used for nut only).
        sigmaK: Turbulent Prandtl number for R_ij diffusion.
        sigmaEps: Turbulent Prandtl number for ε diffusion.
    """

    C1: float = 1.8
    C2: float = 0.6
    Ceps1: float = 1.44
    Ceps2: float = 1.92
    Cmu: float = 0.09
    sigmaK: float = 1.0
    sigmaEps: float = 1.3


# Default constants (Launder, Reece & Rodi 1975)
_DEFAULT_CONSTANTS = LRRConstants()


# ---------------------------------------------------------------------------
# LRR model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("LRR")
class LRRModel(TurbulenceModel):
    """LRR Reynolds stress transport model.

    Solves transport equations for the 6 independent components of the
    symmetric Reynolds stress tensor R_ij plus the dissipation rate ε.

    R_ij is stored as a full (n_cells, 3, 3) tensor with symmetry enforced
    after each update.  Turbulent kinetic energy is derived as::

        k = 0.5 * trace(R_ij)

    Turbulent viscosity is computed via the Boussinesq-like relation::

        ν_t = Cμ k² / ε

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : LRRConstants, optional
        Model constants.  Defaults to Launder-Reece-Rodi values.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: LRRConstants | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)

        self._C = constants if constants is not None else _DEFAULT_CONSTANTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        # Reynolds stress tensor R_ij (n_cells, 3, 3) — initialised to
        # isotropic turbulence with k0 = 1e-4
        k0 = 1e-4
        self._R = torch.zeros(n_cells, 3, 3, device=device, dtype=dtype)
        self._R[:, 0, 0] = 2.0 / 3.0 * k0
        self._R[:, 1, 1] = 2.0 / 3.0 * k0
        self._R[:, 2, 2] = 2.0 / 3.0 * k0

        # Dissipation rate ε (n_cells,)
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)

        # Velocity gradient tensor (n_cells, 3, 3) — computed in correct()
        self._grad_U: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def R_field(self) -> torch.Tensor:
        """Reynolds stress tensor ``(n_cells, 3, 3)``."""
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
        """Turbulent viscosity: ν_t = Cμ k² / ε.

        Returns:
            ``(n_cells,)`` turbulent viscosity field.
        """
        k = self.k().clamp(min=1e-16)
        eps = self._eps.clamp(min=1e-16)
        return self._C.Cmu * k**2 / eps

    def k(self) -> torch.Tensor:
        """Turbulent kinetic energy: k = 0.5 * trace(R_ij).

        Returns:
            ``(n_cells,)`` turbulent kinetic energy field.
        """
        return 0.5 * (self._R[:, 0, 0] + self._R[:, 1, 1] + self._R[:, 2, 2])

    def epsilon(self) -> torch.Tensor:
        """Dissipation rate field ``(n_cells,)``."""
        return self._eps

    def devReff(self) -> torch.Tensor:
        """Effective deviatoric Reynolds stress.

        τ_eff = R_ij - (2/3) k δ_ij + ν_eff (∇U + ∇U^T)

        Returns:
            ``(n_cells, 3, 3)`` effective stress tensor.
        """
        if self._grad_U is None:
            # No gradient yet — return anisotropic stress only
            k = self.k()
            aniso = self._R.clone()
            aniso[:, 0, 0] -= 2.0 / 3.0 * k
            aniso[:, 1, 1] -= 2.0 / 3.0 * k
            aniso[:, 2, 2] -= 2.0 / 3.0 * k
            return aniso

        nut = self.nut()
        k = self.k()
        grad_U = self._grad_U

        # Effective viscosity contribution: ν_eff (∇U + ∇U^T)
        nu_eff = self._nu + nut
        viscous = nu_eff.unsqueeze(-1).unsqueeze(-1) * (grad_U + grad_U.transpose(-1, -2))

        # Anisotropic Reynolds stress: R_ij - (2/3) k δ_ij
        aniso = self._R.clone()
        aniso[:, 0, 0] -= 2.0 / 3.0 * k
        aniso[:, 1, 1] -= 2.0 / 3.0 * k
        aniso[:, 2, 2] -= 2.0 / 3.0 * k

        return aniso + viscous

    def correct(self) -> None:
        """Update the LRR model: compute grad(U), solve R_ij and ε equations."""
        # Compute velocity gradient tensor ∂U_i/∂x_j → (n_cells, 3, 3)
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        # Solve R_ij transport equations (all 6 independent components)
        self._solve_R()

        # Compute production for ε equation (P_k = 0.5 * P_ii)
        P_k = self._compute_P_k()

        # Solve ε transport equation
        self._solve_eps(P_k)

    # ------------------------------------------------------------------
    # Internal: transport equations
    # ------------------------------------------------------------------

    def _solve_R(self) -> None:
        """Solve the Reynolds stress transport equations.

        For each component R_ij (i <= j, 6 independent):
            ∂R_ij/∂t + ∇·(U R_ij) = ∇·((ν + ν_t/σ_k) ∇R_ij)
                                     + P_ij + Φ_ij - (2/3) ε δ_ij

        Full tensor symmetry is enforced after solving.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        grad_U = self._grad_U

        # Effective diffusivity for all stress components
        nu_eff = self._nu + self.nut() / C.sigmaK

        # Compute production tensor P_ij = -(R_ik ∂U_j/∂x_k + R_jk ∂U_i/∂x_k)
        P = self._compute_P_ij()

        # Compute pressure-strain tensor Φ_ij = Φ_ij,1 + Φ_ij,2
        Phi = self._compute_phi(P)

        # Solve each independent component (i <= j)
        for i in range(3):
            for j in range(i, 3):
                R_ij = self._R[:, i, j]

                # Convection + diffusion (same structure as k-ε)
                eqn = fvm.div(self._phi, R_ij, "Gauss upwind", mesh=mesh)
                diff = fvm.laplacian(
                    nu_eff, R_ij, "Gauss linear corrected", mesh=mesh
                )
                eqn.lower = eqn.lower + diff.lower
                eqn.upper = eqn.upper + diff.upper
                eqn.diag = eqn.diag + diff.diag

                # Source: P_ij + Φ_ij - (2/3) ε δ_ij
                source = P[:, i, j] + Phi[:, i, j]
                if i == j:
                    source = source - (2.0 / 3.0) * self._eps

                eqn.source = eqn.source + source

                # Solve (Jacobi-like)
                diag_safe = eqn.diag.abs().clamp(min=1e-30)
                R_new = eqn.source / diag_safe
                R_new = R_new.clamp(min=-1e10, max=1e10)

                # Update both R_ij and R_ji (symmetry)
                self._R[:, i, j] = R_new
                if i != j:
                    self._R[:, j, i] = R_new

        # Enforce positive semi-definiteness on diagonal (R_ii >= 0)
        for i in range(3):
            self._R[:, i, i] = self._R[:, i, i].clamp(min=1e-16)

    def _solve_eps(self, P_k: torch.Tensor) -> None:
        """Solve the ε transport equation.

        ∂ε/∂t + ∇·(U ε) = ∇·((ν + ν_t/σ_ε) ∇ε)
                          + Cε1 P_k ε/k - Cε2 ε²/k
        """
        mesh = self._mesh
        C = self._C

        # Effective diffusivity
        nu_eff = self._nu + self.nut() / C.sigmaEps

        # Build equation
        eqn = fvm.div(self._phi, self._eps, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._eps, "Gauss linear corrected", mesh=mesh
        )
        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: Cε1 * ε/k * P_k - Cε2 * ε²/k
        k_safe = self.k().clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)
        source = (
            C.Ceps1 * eps_safe / k_safe * P_k
            - C.Ceps2 * eps_safe**2 / k_safe
        )
        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        eps_new = eqn.source / diag_safe
        self._eps = eps_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Internal: tensor computations
    # ------------------------------------------------------------------

    def _compute_P_ij(self) -> torch.Tensor:
        """Compute the production tensor P_ij.

        P_ij = -(R_ik ∂U_j/∂x_k + R_jk ∂U_i/∂x_k)

        Returns:
            ``(n_cells, 3, 3)`` production tensor.
        """
        grad_U = self._grad_U
        R = self._R

        # R_ik * ∂U_j/∂x_k  →  einsum: (cells, i, k) * (cells, j, k) → (cells, i, j)
        # grad_U[:, j, k] = ∂U_j/∂x_k
        # term1[i,j] = sum_k R[i,k] * grad_U[j,k]
        term1 = torch.einsum("bik,bjk->bij", R, grad_U)
        # term2[i,j] = sum_k R[j,k] * grad_U[i,k]
        term2 = torch.einsum("bik,bjk->bij", R, grad_U.transpose(-1, -2))

        return -(term1 + term2)

    def _compute_P_k(self) -> torch.Tensor:
        """Compute turbulent production P_k = 0.5 * P_ii.

        Returns:
            ``(n_cells,)`` scalar production rate.
        """
        P = self._compute_P_ij()
        return 0.5 * (P[:, 0, 0] + P[:, 1, 1] + P[:, 2, 2])

    def _compute_phi(self, P: torch.Tensor) -> torch.Tensor:
        """Compute the pressure-strain correlation Φ_ij.

        LRR linear return-to-isotropy model::

            Φ_ij = Φ_ij,1 + Φ_ij,2

        where:
            Φ_ij,1 = -C1 (ε/k) (R_ij - (2/3) k δ_ij)   [slow / Rotta]
            Φ_ij,2 = -C2 (P_ij - (2/3) P_k δ_ij)        [rapid / IP model]

        Args:
            P: Production tensor ``(n_cells, 3, 3)``.

        Returns:
            ``(n_cells, 3, 3)`` pressure-strain tensor.
        """
        C = self._C
        k = self.k()
        k_safe = k.clamp(min=1e-16)
        eps = self._eps.clamp(min=1e-16)

        # P_k = 0.5 * trace(P_ij)
        P_k = 0.5 * (P[:, 0, 0] + P[:, 1, 1] + P[:, 2, 2])

        # --- Slow term (Rotta): Φ_ij,1 = -C1 (ε/k) (R_ij - (2/3) k δ_ij) ---
        ratio = eps / k_safe  # (n_cells,)
        aniso = self._R.clone()
        aniso[:, 0, 0] -= 2.0 / 3.0 * k
        aniso[:, 1, 1] -= 2.0 / 3.0 * k
        aniso[:, 2, 2] -= 2.0 / 3.0 * k
        phi_slow = -C.C1 * ratio.unsqueeze(-1).unsqueeze(-1) * aniso

        # --- Rapid term (IP model): Φ_ij,2 = -C2 (P_ij - (2/3) P_k δ_ij) ---
        P_aniso = P.clone()
        P_aniso[:, 0, 0] -= 2.0 / 3.0 * P_k
        P_aniso[:, 1, 1] -= 2.0 / 3.0 * P_k
        P_aniso[:, 2, 2] -= 2.0 / 3.0 * P_k
        phi_fast = -C.C2 * P_aniso

        return phi_slow + phi_fast
