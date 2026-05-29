"""
Enhanced displacement solver with large deformation support.

Extends :class:`~pyfoam.structural.displacement_solver.DisplacementSolver` with:

- Green-Lagrange strain tensor (large deformation)
- Updated Lagrangian formulation
- Newton-Raphson iteration for nonlinear problems
- Geometric stiffness matrix for large deformation analysis

Usage::

    solver = EnhancedDisplacementSolver(model)
    strain_gl = solver.green_lagrange_strain(grad_u)
    result = solver.solve_nonlinear_1d(
        area=0.01, length=1.0, force=1e6,
        max_iterations=20, tolerance=1e-6,
    )

References
----------
- OpenFOAM ``solidDisplacementFoam`` with large deformation support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver import DisplacementSolver

__all__ = ["EnhancedDisplacementSolver", "NonlinearSolveResult"]

logger = logging.getLogger(__name__)


@dataclass
class NonlinearSolveResult:
    """Result of a nonlinear displacement solve.

    Attributes:
        displacement: Final displacement vector.
        n_iterations: Number of Newton-Raphson iterations.
        converged: Whether the solve converged.
        residual: Final residual norm.
        strain_energy: Total strain energy at convergence.
    """

    displacement: torch.Tensor
    n_iterations: int = 0
    converged: bool = True
    residual: float = 0.0
    strain_energy: float = 0.0


class EnhancedDisplacementSolver(DisplacementSolver):
    """Enhanced displacement solver with large deformation capabilities.

    Parameters
    ----------
    model : LinearElasticModel
        Constitutive model.
    """

    def __init__(self, model: LinearElasticModel) -> None:
        super().__init__(model)

    # ------------------------------------------------------------------
    # Large deformation strain measures
    # ------------------------------------------------------------------

    def green_lagrange_strain(self, grad_u: torch.Tensor) -> torch.Tensor:
        """Compute Green-Lagrange strain tensor.

        E = 0.5 * (F^T * F - I)  where F = I + grad_u

        For small deformations, this reduces to the linear (Cauchy) strain.
        In Voigt notation: ``(E_xx, E_yy, E_zz, 2*E_yz, 2*E_xz, 2*E_xy)``.

        Args:
            grad_u: ``(3, 3)`` displacement gradient ``du_i/dx_j``.

        Returns:
            ``(6,)`` Green-Lagrange strain in Voigt notation.
        """
        grad_u = grad_u.to(dtype=torch.float64)
        F = torch.eye(3, dtype=torch.float64) + grad_u
        E = 0.5 * (F.T @ F - torch.eye(3, dtype=torch.float64))
        return torch.tensor([
            E[0, 0],
            E[1, 1],
            E[2, 2],
            2.0 * E[1, 2],
            2.0 * E[0, 2],
            2.0 * E[0, 1],
        ], dtype=torch.float64)

    def deformation_gradient(
        self, grad_u: torch.Tensor
    ) -> torch.Tensor:
        """Compute deformation gradient tensor F = I + grad_u.

        Args:
            grad_u: ``(3, 3)`` displacement gradient.

        Returns:
            ``(3, 3)`` deformation gradient tensor.
        """
        return torch.eye(3, dtype=torch.float64) + grad_u.to(dtype=torch.float64)

    def jacobian_det(self, grad_u: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian determinant J = det(F).

        J > 0 for physically valid deformations. J = 1 for
        isochoric (volume-preserving) deformation.

        Args:
            grad_u: ``(3, 3)`` displacement gradient.

        Returns:
            Scalar J.
        """
        F = self.deformation_gradient(grad_u)
        return torch.det(F)

    # ------------------------------------------------------------------
    # Geometric stiffness
    # ------------------------------------------------------------------

    def geometric_stiffness_1d(
        self,
        axial_force: float,
        length: float,
    ) -> torch.Tensor:
        """Assemble 1D geometric stiffness matrix (initial stress stiffness).

        For a bar element with axial force N and length L::

            K_G = (N/L) * [[1, -1], [-1, 1]]

        This captures the stiffening/softening effect of pre-stress.

        Args:
            axial_force: Axial force in the element (N, positive = tension).
            length: Element length (m).

        Returns:
            ``(2, 2)`` geometric stiffness matrix.
        """
        k = axial_force / length
        return torch.tensor([[k, -k], [-k, k]], dtype=torch.float64)

    # ------------------------------------------------------------------
    # Newton-Raphson solver for 1D bar
    # ------------------------------------------------------------------

    def solve_nonlinear_1d(
        self,
        area: float,
        length: float,
        force: float,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
    ) -> NonlinearSolveResult:
        """Solve a 1D bar problem with Newton-Raphson iteration.

        Solves K(u) * u = F where K(u) includes both material and
        geometric stiffness.

        The bar has:
        - One end fixed (DOF 0)
        - Force applied at the free end (DOF 1)

        Args:
            area: Cross-sectional area (m^2).
            length: Element length (m).
            force: Applied force at free end (N).
            max_iterations: Maximum Newton-Raphson iterations.
            tolerance: Convergence tolerance on residual.

        Returns:
            :class:`NonlinearSolveResult`.
        """
        E = self._model.youngs_modulus
        K_mat = self.stiffness_matrix_1d(area, length)

        u = torch.zeros(2, dtype=torch.float64)
        F = torch.tensor([0.0, force], dtype=torch.float64)
        converged = False
        residual = float("inf")

        for iteration in range(max_iterations):
            # Current axial strain and force
            strain = (u[1] - u[0]) / length
            N = E * area * strain

            # Total stiffness = material + geometric
            K_geo = self.geometric_stiffness_1d(N, length)
            K_total = K_mat + K_geo

            # Residual: R = F - K(u) * u
            R = F - K_total @ u
            R_free = R[1:]  # Only free DOF

            residual = float(R_free.norm().item())
            if residual < tolerance:
                converged = True
                break

            # Solve for correction: dK * du = R (free DOFs only)
            K_free = K_total[1:, 1:]
            du_free = torch.linalg.solve(K_free, R_free)
            u[1] += du_free[0]

        # Compute strain energy
        strain_final = (u[1] - u[0]) / length
        U = 0.5 * E * strain_final ** 2 * area * length

        return NonlinearSolveResult(
            displacement=u,
            n_iterations=iteration + 1,
            converged=converged,
            residual=residual,
            strain_energy=U.item(),
        )

    # ------------------------------------------------------------------
    # Updated Lagrangian strain increment
    # ------------------------------------------------------------------

    def updated_lagrangian_strain_increment(
        self,
        grad_u_old: torch.Tensor,
        grad_u_new: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the incremental strain between two configurations.

        Uses the updated Lagrangian approach: the strain increment
        is computed relative to the current (old) configuration.

        delta_epsilon = 0.5 * (F_new^T * F_new - F_old^T * F_old)

        Args:
            grad_u_old: ``(3, 3)`` displacement gradient (old config).
            grad_u_new: ``(3, 3)`` displacement gradient (new config).

        Returns:
            ``(6,)`` incremental Green-Lagrange strain in Voigt notation.
        """
        F_old = self.deformation_gradient(grad_u_old)
        F_new = self.deformation_gradient(grad_u_new)

        C_old = F_old.T @ F_old
        C_new = F_new.T @ F_new

        dE = 0.5 * (C_new - C_old)

        return torch.tensor([
            dE[0, 0],
            dE[1, 1],
            dE[2, 2],
            2.0 * dE[1, 2],
            2.0 * dE[0, 2],
            2.0 * dE[0, 1],
        ], dtype=torch.float64)

    def __repr__(self) -> str:
        return f"EnhancedDisplacementSolver(model={self._model!r})"
