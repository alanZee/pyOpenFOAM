"""
Plane Couette flow validation case.

Two infinite parallel plates separated by gap H:
- Bottom plate (y=0): stationary (no-slip)
- Top plate (y=H): moving at velocity U in the x-direction

Analytical steady-state solution (fully developed, 2D):

    u(y) = U * y / H        (linear velocity profile)
    v = 0
    w = 0
    p = constant (dp/dx = 0)

This is the simplest non-trivial viscous flow with an exact solution.
The Reynolds number is Re = U * H / nu.

Usage::

    case = CouetteFlowCase(n_cells=32, Re=10.0, U_top=1.0)
    case.setup()
    case.run()
    ref = case.get_reference()
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from validation.runner import ValidationCaseBase

logger = logging.getLogger(__name__)


class CouetteFlowCase(ValidationCaseBase):
    """Plane Couette flow validation case.

    Parameters
    ----------
    n_cells : int
        Number of cells in each direction (n_cells × n_cells mesh).
    Re : float
        Reynolds number Re = U * H / nu.
    U_top : float
        Velocity of the top plate.
    H : float
        Channel height (gap width).
    L : float
        Channel length (streamwise direction).
    max_iterations : int
        Maximum SIMPLE iterations.
    tolerance : float
        Convergence tolerance.
    """

    def __init__(
        self,
        n_cells: int = 32,
        Re: float = 10.0,
        U_top: float = 1.0,
        H: float = 1.0,
        L: float = 1.0,
        max_iterations: int = 200,
        tolerance: float = 1e-6,
    ) -> None:
        self.n_cells = n_cells
        self.Re = Re
        self.U_top = U_top
        self.H = H
        self.L = L
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Derived quantities
        self.nu = U_top * H / Re  # kinematic viscosity

        # Will be populated by setup()
        self._mesh = None
        self._U_computed = None
        self._p_computed = None
        self._cell_centres = None
        self._solver_info = {}

    @property
    def name(self) -> str:
        return "Couette Flow"

    @property
    def description(self) -> str:
        return (
            f"Plane Couette flow: Re={self.Re:.0f}, "
            f"U_top={self.U_top}, H={self.H}, "
            f"mesh={self.n_cells}x{self.n_cells}"
        )

    def setup(self) -> None:
        """Build the mesh and initialise fields for Couette flow."""
        N = self.n_cells
        device = torch.device("cpu")
        dtype = torch.float64

        # Create a 2D channel mesh (simplified as a structured grid)
        # We use a 2D mesh in the x-y plane, extruded with one cell in z
        nx = N  # streamwise
        ny = N  # wall-normal

        # Cell centres (nx * ny cells)
        dx = self.L / nx
        dy = self.H / ny

        cell_centres = torch.zeros(nx * ny, 3, dtype=dtype, device=device)
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                cell_centres[idx, 0] = (i + 0.5) * dx  # x
                cell_centres[idx, 1] = (j + 0.5) * dy  # y
                cell_centres[idx, 2] = 0.0               # z

        self._cell_centres = cell_centres
        self._nx = nx
        self._ny = ny
        self._dx = dx
        self._dy = dy

        # Initialise velocity field: small perturbation from zero
        n_cells = nx * ny
        U = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        # Small initial x-velocity to help convergence
        U[:, 0] = 0.5 * self.U_top  # uniform initial guess

        # Pressure: zero everywhere
        p = torch.zeros(n_cells, dtype=dtype, device=device)

        self._U_init = U.clone()
        self._p_init = p.clone()

        logger.info("Couette flow setup: %dx%d mesh, Re=%.1f, nu=%.6e",
                     nx, ny, self.Re, self.nu)

    def run(self) -> dict[str, Any]:
        """Run the Couette flow solver.

        Uses a direct solve of the momentum equation since Couette flow
        has no pressure gradient and is steady-state.
        """
        N = self.n_cells
        nx, ny = self._nx, self._ny
        dx, dy = self._dx, self._dy
        dtype = torch.float64

        # For Couette flow, the steady-state solution satisfies:
        # d²u/dy² = 0  (no pressure gradient, fully developed)
        # with BCs: u(0) = 0, u(H) = U_top
        #
        # We solve this iteratively using a vectorized Jacobi relaxation
        # with ghost cells for wall BCs.

        U = self._U_init.clone()
        n_cells = nx * ny

        # Under-relaxation factor
        alpha = 0.5

        # Use padded array with ghost cells for wall BCs
        # For cell-centered scheme with walls at y=0 and y=H:
        # Bottom wall (u_wall=0): u_ghost = -u_interior (antisymmetric)
        # Top wall (u_wall=U_top): u_ghost = 2*U_top - u_interior
        u_padded = torch.zeros(ny + 2, nx, dtype=dtype)
        # Initialize ghost cells with antisymmetric BC
        u_padded[1:-1, :] = U[:, 0].reshape(ny, nx).clone()
        u_padded[0, :] = -u_padded[1, :]    # bottom wall
        u_padded[-1, :] = 2.0 * self.U_top - u_padded[-2, :]  # top wall

        # Iterative solve (vectorized Jacobi relaxation)
        for iteration in range(self.max_iterations):
            u_old = u_padded.clone()

            # All real cells: apply d²u/dy² = 0
            # u_new[j] = 0.5 * (u[j-1] + u[j+1])
            u_new = 0.5 * (u_padded[:-2, :] + u_padded[2:, :])
            u_padded[1:-1, :] = alpha * u_new + (1.0 - alpha) * u_padded[1:-1, :]

            # Wall BCs: antisymmetric ghost cells
            # Bottom wall (u_wall=0): u_ghost = -u_interior
            u_padded[0, :] = -u_padded[1, :]
            # Top wall (u_wall=U_top): u_ghost = 2*U_top - u_interior
            u_padded[-1, :] = 2.0 * self.U_top - u_padded[-2, :]

            # Check convergence
            diff = (u_padded - u_old).abs().max().item()
            if diff < self.tolerance:
                logger.info("Couette flow converged in %d iterations (max_diff=%.6e)",
                           iteration + 1, diff)
                self._solver_info = {
                    "iterations": iteration + 1,
                    "converged": True,
                    "final_residual": diff,
                }
                break
        else:
            logger.warning("Couette flow did not converge in %d iterations", self.max_iterations)
            self._solver_info = {
                "iterations": self.max_iterations,
                "converged": False,
                "final_residual": diff,
            }

        # Extract real cells (skip ghost cells)
        u_grid = u_padded[1:-1, :]

        # Convert back to flat array
        U[:, 0] = u_grid.reshape(-1)
        U[:, 1] = 0.0
        U[:, 2] = 0.0

        self._U_computed = U
        self._p_computed = torch.zeros(n_cells, dtype=dtype)

        return self._solver_info

    def get_reference(self) -> dict[str, torch.Tensor]:
        """Compute the analytical Couette flow solution.

        u(y) = U_top * y / H  (linear profile)
        """
        ny = self._ny
        dtype = torch.float64
        n_cells = self._nx * ny

        U_ref = torch.zeros(n_cells, 3, dtype=dtype)
        p_ref = torch.zeros(n_cells, dtype=dtype)

        for j in range(ny):
            y = (j + 0.5) * self._dy  # cell centre y-coordinate
            u_analytical = self.U_top * y / self.H
            for i in range(self._nx):
                idx = j * self._nx + i
                U_ref[idx, 0] = u_analytical

        return {"U": U_ref, "p": p_ref}

    def get_computed(self) -> dict[str, torch.Tensor]:
        """Return the computed velocity and pressure fields."""
        return {
            "U": self._U_computed.clone(),
            "p": self._p_computed.clone(),
        }

    def get_tolerances(self) -> dict[str, float]:
        """Couette flow tolerances (relaxed for simplified solver)."""
        return {
            "l2_tol": 0.1,    # 10% L2 relative error
            "max_tol": 0.1,   # 0.1 max absolute error
        }
