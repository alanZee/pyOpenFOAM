"""
Displacement solver for structural mechanics.

Solves for displacement fields given boundary conditions and material
properties, using a simple direct stiffness approach on a single-element
level (extendable to mesh-based solvers).

In OpenFOAM, the ``solidDisplacementFoam`` solver handles displacement
field computation iteratively with the momentum equation.  This module
provides a Python equivalent for single-element and small-system analysis.

Usage::

    model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
    solver = DisplacementSolver(model)

    # Apply uniform strain and get displacement
    strain = torch.tensor([0.001, -0.0003, -0.0003, 0, 0, 0], dtype=torch.float64)
    disp = solver.solve_from_strain(strain, element_size=0.1)
"""

from __future__ import annotations

import torch

from pyfoam.structural.elastic_model import LinearElasticModel

__all__ = ["DisplacementSolver"]


class DisplacementSolver:
    """Solve for displacement fields in linear elastic solids.

    Provides utility methods for:

    - Computing displacement from strain
    - Computing strain from displacement gradient
    - Assembling simple stiffness systems (single element)
    - Solving ``K * u = F`` for small systems

    Args:
        model: Constitutive model (:class:`LinearElasticModel`).
    """

    def __init__(self, model: LinearElasticModel) -> None:
        self._model = model

    @property
    def model(self) -> LinearElasticModel:
        return self._model

    def strain_from_displacement_gradient(
        self, grad_u: torch.Tensor
    ) -> torch.Tensor:
        """Compute strain from the displacement gradient tensor.

        The symmetric strain tensor is::

            epsilon = 0.5 * (grad_u + grad_u^T)

        In Voigt notation ``(eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy)``
        where ``gamma = 2 * epsilon_offdiagonal``.

        Args:
            grad_u: ``(3, 3)`` displacement gradient tensor ``du_i/dx_j``.

        Returns:
            ``(6,)`` strain in Voigt notation.
        """
        grad_u = grad_u.to(dtype=torch.float64)
        eps = 0.5 * (grad_u + grad_u.T)
        return torch.tensor([
            eps[0, 0],        # eps_xx
            eps[1, 1],        # eps_yy
            eps[2, 2],        # eps_zz
            2.0 * eps[1, 2],  # gamma_yz
            2.0 * eps[0, 2],  # gamma_xz
            2.0 * eps[0, 1],  # gamma_xy
        ], dtype=torch.float64)

    def solve_from_strain(
        self, strain: torch.Tensor, element_size: float = 1.0
    ) -> torch.Tensor:
        """Compute displacement from strain for a single element.

        Uses the relationship ``u = epsilon * L`` where ``L`` is the
        element size.  This is exact for uniform strain fields.

        Args:
            strain: ``(6,)`` strain in Voigt notation.
            element_size: Characteristic element length (m).

        Returns:
            ``(3,)`` displacement vector (approximate).
        """
        strain = strain.to(dtype=torch.float64)
        # Extract normal strains (xx, yy, zz) and scale by element size
        return strain[:3] * element_size

    def stiffness_matrix_1d(
        self, area: float, length: float
    ) -> torch.Tensor:
        """Assemble 1D bar element stiffness matrix.

        For a bar element with Young's modulus *E*, area *A*, length *L*::

            K = (E*A/L) * [[1, -1], [-1, 1]]

        Args:
            area: Cross-sectional area (m^2).
            length: Element length (m).

        Returns:
            ``(2, 2)`` stiffness matrix.
        """
        E = self._model.youngs_modulus
        k = E * area / length
        return torch.tensor([[k, -k], [-k, k]], dtype=torch.float64)

    def stiffness_matrix_2d_truss(
        self, area: float, length: float, angle: float
    ) -> torch.Tensor:
        """Assemble 2D truss element stiffness matrix (4x4).

        For a truss element at angle *theta* in the 2D plane::

            K = (E*A/L) * T^T * [[1,-1],[-1,1]] * T

        where ``T`` is the rotation matrix.

        Args:
            area: Cross-sectional area (m^2).
            length: Element length (m).
            angle: Element orientation angle (radians).

        Returns:
            ``(4, 4)`` element stiffness matrix in global coords.
        """
        E = self._model.youngs_modulus
        c = torch.cos(torch.tensor(angle, dtype=torch.float64))
        s = torch.sin(torch.tensor(angle, dtype=torch.float64))
        k = E * area / length

        # Direction cosines contribution
        return k * torch.tensor([
            [c*c,   c*s,  -c*c,  -c*s],
            [c*s,   s*s,  -c*s,  -s*s],
            [-c*c, -c*s,   c*c,   c*s],
            [-c*s, -s*s,   c*s,   s*s],
        ], dtype=torch.float64)

    def solve_bar(
        self,
        stiffness: torch.Tensor,
        forces: torch.Tensor,
        fixed_dofs: list[int],
    ) -> torch.Tensor:
        """Solve ``K * u = F`` with prescribed (fixed) DOFs.

        Eliminates fixed DOFs from the system, solves the reduced
        system, and reconstructs the full displacement vector.

        Args:
            stiffness: ``(n, n)`` global stiffness matrix.
            forces: ``(n,)`` force vector.
            fixed_dofs: Indices of fixed (zero-displacement) DOFs.

        Returns:
            ``(n,)`` displacement vector.
        """
        stiffness = stiffness.to(dtype=torch.float64)
        forces = forces.to(dtype=torch.float64)
        n = stiffness.shape[0]

        # Free DOFs
        free_dofs = [i for i in range(n) if i not in fixed_dofs]

        if len(free_dofs) == 0:
            return torch.zeros(n, dtype=torch.float64)

        # Extract sub-matrices
        K_ff = stiffness[free_dofs][:, free_dofs]
        F_f = forces[free_dofs]

        # Solve reduced system
        u_f = torch.linalg.solve(K_ff, F_f)

        # Reconstruct full vector
        u = torch.zeros(n, dtype=torch.float64)
        for i, dof in enumerate(free_dofs):
            u[dof] = u_f[i]
        return u

    def strain_energy(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute strain energy density: U = 0.5 * epsilon^T * C * epsilon.

        Args:
            strain: ``(6,)`` strain in Voigt notation.

        Returns:
            Scalar strain energy density (J/m^3).
        """
        strain = strain.to(dtype=torch.float64)
        C = self._model.elasticity_matrix
        return 0.5 * strain @ C @ strain

    def __repr__(self) -> str:
        return f"DisplacementSolver(model={self._model!r})"
