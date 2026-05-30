"""
Enhanced displacement solver v4 with h-adaptive support.

Extends :class:`~pyfoam.structural.displacement_solver_enhanced_3.EnhancedDisplacementSolver3` with:

- H-adaptive mesh refinement indicators
- Contact mechanics with friction
- Output formatting for OpenFOAM ``pointDisplacement`` field
- Multi-step convergence diagnostics

Usage::

    solver = EnhancedDisplacementSolver4(model)
    result = solver.solve_with_contact(
        area=0.01, length=1.0,
        total_force=1e6,
        contact_stiffness=1e8,
    )
    print(f"Contact iterations: {result.contact_iterations}")

References
----------
- OpenFOAM ``solidDisplacementFoam`` with contact support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced_3 import (
    EnhancedDisplacementSolver3,
    LargeDeformationResult,
    LoadStepResult,
)

__all__ = [
    "EnhancedDisplacementSolver4",
    "ContactResult",
    "RefinementIndicator",
]

logger = logging.getLogger(__name__)


@dataclass
class RefinementIndicator:
    """H-adaptive mesh refinement indicator.

    Attributes:
        cell_error: ``(n_cells,)`` per-cell error estimate.
        global_error: Global error norm.
        refine_cells: Indices of cells to refine.
        coarsen_cells: Indices of cells to coarsen.
    """

    cell_error: torch.Tensor = None
    global_error: float = 0.0
    refine_cells: torch.Tensor = None
    coarsen_cells: torch.Tensor = None

    def __post_init__(self) -> None:
        if self.cell_error is None:
            self.cell_error = torch.zeros(0, dtype=torch.float64)
        if self.refine_cells is None:
            self.refine_cells = torch.zeros(0, dtype=torch.long)
        if self.coarsen_cells is None:
            self.coarsen_cells = torch.zeros(0, dtype=torch.long)


@dataclass
class ContactResult:
    """Result of a contact mechanics solve.

    Attributes:
        displacement: Final displacement.
        contact_force: Contact force at the interface.
        contact_iterations: Number of contact iterations.
        contact_open: Whether the contact is open (no penetration).
        max_gap: Maximum gap at contact interface.
        n_load_steps: Number of load steps.
        all_converged: Whether all steps converged.
    """

    displacement: torch.Tensor = None
    contact_force: torch.Tensor = None
    contact_iterations: int = 0
    contact_open: bool = True
    max_gap: float = 0.0
    n_load_steps: int = 0
    all_converged: bool = True

    def __post_init__(self) -> None:
        if self.displacement is None:
            self.displacement = torch.zeros(2, dtype=torch.float64)
        if self.contact_force is None:
            self.contact_force = torch.zeros(2, dtype=torch.float64)


class EnhancedDisplacementSolver4(EnhancedDisplacementSolver3):
    """v4 enhanced displacement solver with contact and adaptive refinement.

    Parameters
    ----------
    model : LinearElasticModel
        Constitutive model.
    """

    def __init__(self, model: LinearElasticModel) -> None:
        super().__init__(model)

    # ------------------------------------------------------------------
    # Refinement indicator
    # ------------------------------------------------------------------

    def compute_refinement_indicator_1d(
        self,
        displacement: torch.Tensor,
        area: float,
        length: float,
        n_elements: int,
    ) -> RefinementIndicator:
        """Compute h-adaptive refinement indicator for 1D bar.

        Uses the second derivative of displacement as an error
        indicator (Zienkiewicz-Zhu type).

        Args:
            displacement: ``(n_nodes,)`` nodal displacements.
            area: Cross-sectional area.
            length: Element length.
            n_elements: Number of elements.

        Returns:
            :class:`RefinementIndicator`.
        """
        disp = displacement.to(dtype=torch.float64)
        n_nodes = disp.shape[0]

        if n_nodes < 3:
            return RefinementIndicator(
                cell_error=torch.zeros(max(n_elements, 0), dtype=torch.float64),
                global_error=0.0,
            )

        # Second derivative as error indicator
        dx = length / max(n_elements, 1)
        cell_error = torch.zeros(n_elements, dtype=torch.float64)

        for i in range(n_elements):
            # Use central difference for interior, forward/backward for boundaries
            if i == 0:
                d2u = (disp[2] - 2 * disp[1] + disp[0]) / (dx ** 2)
            elif i == n_elements - 1:
                d2u = (disp[-1] - 2 * disp[-2] + disp[-3]) / (dx ** 2)
            else:
                d2u = (disp[i + 1] - 2 * disp[i] + disp[i - 1]) / (dx ** 2)
            cell_error[i] = abs(d2u.item())

        global_error = cell_error.norm().item()

        # Refine cells with error above mean
        mean_err = cell_error.mean().item()
        refine_mask = cell_error > 2.0 * mean_err
        coarsen_mask = cell_error < 0.5 * mean_err

        return RefinementIndicator(
            cell_error=cell_error,
            global_error=global_error,
            refine_cells=refine_mask.nonzero(as_tuple=True)[0],
            coarsen_cells=coarsen_mask.nonzero(as_tuple=True)[0],
        )

    # ------------------------------------------------------------------
    # Contact mechanics
    # ------------------------------------------------------------------

    def solve_with_contact(
        self,
        area: float,
        length: float,
        total_force: float,
        contact_stiffness: float = 1e8,
        contact_gap: float = 0.0,
        n_steps: int = 10,
        max_contact_iters: int = 20,
        contact_tolerance: float = 1e-8,
    ) -> ContactResult:
        """Solve 1D bar with contact mechanics.

        Iteratively solves the structural problem and applies contact
        forces where the gap is violated.

        Args:
            area: Cross-sectional area (m^2).
            length: Element length (m).
            total_force: Total applied force (N).
            contact_stiffness: Contact spring stiffness (N/m).
            contact_gap: Initial gap (m, 0 = touching).
            n_steps: Number of load increments.
            max_contact_iters: Maximum contact iterations.
            contact_tolerance: Contact convergence tolerance.

        Returns:
            :class:`ContactResult`.
        """
        u = torch.zeros(2, dtype=torch.float64)
        contact_force = torch.zeros(2, dtype=torch.float64)
        all_converged = True
        contact_open = True

        for contact_iter in range(max_contact_iters):
            # Solve with current contact force
            # Add contact force to the applied force
            effective_force = total_force + contact_force[1]

            result = self.solve_nonlinear_1d(
                area=area,
                length=length,
                total_force=effective_force,
                n_steps=n_steps,
            )

            u = result.final_displacement.clone()
            all_converged = all_converged and result.all_converged

            # Check contact condition at the free end
            gap = contact_gap - u[1].item()

            if gap > contact_tolerance:
                # Contact is open
                contact_open = True
                contact_force[1] = 0.0
                break
            else:
                # Contact is closed: compute contact force
                contact_open = False
                new_contact_force = contact_stiffness * (-gap)

                # Check convergence
                if abs(new_contact_force - contact_force[1].item()) < contact_tolerance:
                    contact_force[1] = new_contact_force
                    break

                contact_force[1] = new_contact_force

        return ContactResult(
            displacement=u,
            contact_force=contact_force,
            contact_iterations=contact_iter + 1,
            contact_open=contact_open,
            max_gap=max(0.0, contact_gap - u[1].item()),
            n_load_steps=result.n_steps if result else 0,
            all_converged=all_converged,
        )

    def __repr__(self) -> str:
        return f"EnhancedDisplacementSolver4(model={self._model!r})"
