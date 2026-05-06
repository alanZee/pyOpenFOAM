"""
SIMPLE algorithm for steady-state incompressible flow.

Implements the Semi-Implicit Method for Pressure-Linked Equations (SIMPLE)
as described by Patankar (1980). This is the workhorse algorithm for
steady-state incompressible CFD.

Algorithm (per outer iteration):
1. **Momentum predictor**: Solve A_p * U* = H(U) - grad(p_old)
   using under-relaxation.
2. **Compute HbyA**: HbyA = H(U*) / A_p
3. **Compute phiHbyA**: Face flux from HbyA (linear interpolation).
4. **Assemble pressure equation**: laplacian(1/A_p, p') = div(phiHbyA)
5. **Solve pressure correction** p'.
6. **Correct velocity**: U = U* + (1/A_p) * (-grad(p'))
   (equivalently: U = HbyA - (1/A_p) * grad(p_old + p'))
7. **Correct flux**: phi = phiHbyA - (1/A_p)_f * grad(p')_f
8. **Check convergence** on continuity residual.

Under-relaxation is applied implicitly to the momentum equation:
    diag' = diag / α_U
    source' += (1 - α_U) / α_U * diag * U_old

And optionally to the pressure correction:
    p += α_p * p'

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.solvers.coupled_solver import (
    CoupledSolverBase,
    CoupledSolverConfig,
    ConvergenceData,
)
from pyfoam.solvers.rhie_chow import (
    compute_HbyA,
    compute_face_flux_HbyA,
    rhie_chow_correction,
)
from pyfoam.solvers.pressure_equation import (
    assemble_pressure_equation,
    solve_pressure_equation,
    correct_velocity,
    correct_face_flux,
)

__all__ = ["SIMPLESolver", "SIMPLEConfig"]


logger = logging.getLogger(__name__)


class SIMPLEConfig(CoupledSolverConfig):
    """Configuration for the SIMPLE algorithm.

    Extends the base config with SIMPLE-specific parameters.

    Attributes
    ----------
    n_correctors : int
        Number of pressure correction steps per outer iteration (default 1).
    nu : float
        Kinematic viscosity (default 1.0).
    """

    def __init__(
        self,
        n_correctors: int = 1,
        nu: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_correctors = n_correctors
        self.nu = nu


class SIMPLESolver(CoupledSolverBase):
    """SIMPLE algorithm for steady-state incompressible flow.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    config : SIMPLEConfig
        Solver configuration.

    Examples::

        config = SIMPLEConfig(
            relaxation_factor_U=0.7,
            relaxation_factor_p=0.3,
        )
        solver = SIMPLESolver(mesh, config)

        U, p, phi, convergence = solver.solve(U, p, phi, ...)
    """

    def __init__(
        self,
        mesh: Any,
        config: SIMPLEConfig | None = None,
    ) -> None:
        if config is None:
            config = SIMPLEConfig()
        super().__init__(mesh, config)
        self._simple_config = config

    def solve(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        *,
        U_old: torch.Tensor | None = None,
        p_old: torch.Tensor | None = None,
        max_outer_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run the SIMPLE algorithm.

        Args:
            U: ``(n_cells, 3)`` — velocity field.
            p: ``(n_cells,)`` — pressure field.
            phi: ``(n_faces,)`` — face flux field.
            U_old: Previous time-step velocity (unused in SIMPLE, kept for API).
            p_old: Previous time-step pressure (unused in SIMPLE, kept for API).
            max_outer_iterations: Maximum outer-loop iterations.
            tolerance: Convergence tolerance on continuity residual.

        Returns:
            Tuple of ``(U, p, phi, convergence_data)``.
        """
        device = self._device
        dtype = self._dtype
        mesh = self._mesh
        config = self._simple_config

        # Ensure tensors are on correct device/dtype
        U = U.to(device=device, dtype=dtype)
        p = p.to(device=device, dtype=dtype)
        phi = phi.to(device=device, dtype=dtype)

        convergence = ConvergenceData()

        # Store old values for relaxation
        U_old_iter = U.clone()
        p_old_iter = p.clone()

        for outer in range(max_outer_iterations):
            # Store previous iteration for convergence check
            U_prev = U.clone()
            p_prev = p.clone()

            # ============================================
            # Step 1: Momentum predictor
            # ============================================
            U, A_p, H = self._momentum_predictor(U, p, phi)

            # ============================================
            # Step 2: Compute HbyA
            # ============================================
            HbyA = compute_HbyA(H, A_p)

            # ============================================
            # Step 3: Compute phiHbyA
            # ============================================
            phiHbyA = compute_face_flux_HbyA(
                HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
                mesh.n_internal_faces, mesh.face_weights,
            )

            # ============================================
            # Step 4-5: Assemble and solve pressure equation
            # ============================================
            p_eqn = assemble_pressure_equation(
                phiHbyA, A_p, mesh, mesh.face_weights,
            )

            p, p_iters, p_res = solve_pressure_equation(
                p_eqn, p, self._p_solver,
                tolerance=config.p_tolerance,
                max_iter=config.p_max_iter,
            )

            # Under-relax pressure correction
            alpha_p = config.relaxation_factor_p
            if alpha_p < 1.0:
                p = alpha_p * p + (1.0 - alpha_p) * p_prev

            # ============================================
            # Step 6: Correct velocity
            # ============================================
            U = correct_velocity(U, HbyA, p, A_p, mesh)

            # ============================================
            # Step 7: Correct face flux
            # ============================================
            phi = correct_face_flux(phi, p, A_p, mesh, mesh.face_weights)

            # ============================================
            # Step 8: Check convergence
            # ============================================
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)

            # Continuity error: sum of flux imbalance per cell
            continuity_error = self._compute_continuity_error(phi)

            convergence.p_residual = p_residual
            convergence.U_residual = U_residual
            convergence.continuity_error = continuity_error
            convergence.outer_iterations = outer + 1

            convergence.residual_history.append({
                "outer": outer,
                "U_residual": U_residual,
                "p_residual": p_residual,
                "continuity_error": continuity_error,
                "p_linear_iters": p_iters,
                "p_linear_res": p_res,
            })

            if outer % 10 == 0 or outer < 5:
                logger.info(
                    "SIMPLE iteration %d: U_res=%.6e, p_res=%.6e, "
                    "continuity=%.6e",
                    outer, U_residual, p_residual, continuity_error,
                )

            # Check convergence
            if continuity_error < tolerance and outer > 0:
                convergence.converged = True
                logger.info(
                    "SIMPLE converged in %d iterations (continuity=%.6e)",
                    outer + 1, continuity_error,
                )
                break

        if not convergence.converged:
            logger.warning(
                "SIMPLE did not converge in %d iterations "
                "(continuity=%.6e)",
                max_outer_iterations, continuity_error,
            )

        return U, p, phi, convergence

    def _momentum_predictor(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the momentum equation with under-relaxation.

        The momentum equation:
            A_p * U = H(U) - grad(p)

        With under-relaxation:
            A_p/α_U * U = H(U) - grad(p) + (1-α_U)/α_U * A_p * U_old

        Args:
            U: ``(n_cells, 3)`` — current velocity.
            p: ``(n_cells,)`` — current pressure.
            phi: ``(n_faces,)`` — current face flux.

        Returns:
            Tuple of ``(U_new, A_p, H)`` where:
            - U_new: ``(n_cells, 3)`` — relaxed velocity.
            - A_p: ``(n_cells,)`` — diagonal coefficients.
            - H: ``(n_cells, 3)`` — H(U) vector.
        """
        mesh = self._mesh
        device = self._device
        dtype = self._dtype
        config = self._simple_config
        alpha_U = config.relaxation_factor_U

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # Build the momentum matrix
        # For simplicity, we assemble a convection-diffusion matrix
        # In a full implementation, this would use fvm.div(phi, U) + fvm.laplacian(nu, U)
        # Here we build the matrix directly from the mesh topology

        mat = FvMatrix(
            n_cells, owner[:n_internal], neighbour,
            device=device, dtype=dtype,
        )

        # Simple convection-diffusion coefficients
        # Diffusion: lower[f] = upper[f] = -coeff
        # Convection: upwind bias
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients
        cell_volumes_safe = cell_volumes.clamp(min=1e-30)

        int_owner = owner[:n_internal]
        int_neigh = neighbour

        # Diffusion coefficient (viscous)
        nu = config.nu  # kinematic viscosity from config
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = nu * S_mag * delta_f

        # Convection (upwind)
        flux = phi[:n_internal]
        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        # Matrix coefficients
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        # Lower = -diff_coeff + flux_neg (owner receives from neighbour)
        mat.lower = (-diff_coeff + flux_neg) / V_P
        # Upper = -diff_coeff + flux_pos (neighbour receives from owner)
        mat.upper = (-diff_coeff - flux_pos) / V_N

        # Diagonal: sum of absolute off-diagonal + convection
        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add((diff_coeff - flux_neg) / V_P, int_owner, n_cells)
        diag = diag + scatter_add((diff_coeff + flux_pos) / V_N, int_neigh, n_cells)
        mat.diag = diag

        # Store A_p (diagonal) for later use
        A_p = diag.clone()

        # Compute H(U): off-diagonal contributions
        # H = -sum(off-diag * U_neigh) + source
        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        # Owner receives from neighbour: lower * U_neigh
        # Use direct indexing for 2D U tensor and index_add_ for 2D scatter
        U_neigh = U[int_neigh]  # (n_internal, 3)
        owner_contrib = mat.lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        # Neighbour receives from owner: upper * U_owner
        U_own = U[int_owner]  # (n_internal, 3)
        neigh_contrib = mat.upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        # Note: H already includes the off-diagonal terms multiplied by V
        # The momentum equation is: A_p * U = H - grad(p)
        # where A_p and H are per-volume quantities

        # Pressure gradient contribution to source
        # grad(p) using Gauss theorem
        w = mesh.face_weights[:n_internal]
        p_P = gather(p, int_owner)
        p_N = gather(p, int_neigh)
        p_face = w * p_P + (1.0 - w) * p_N
        p_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_p.index_add_(0, int_owner, p_contrib)
        grad_p.index_add_(0, int_neigh, -p_contrib)

        # Source = H - grad(p) (per cell, not per volume)
        source = H - grad_p

        # Solve momentum equation: A_p * U = source
        # U = source / A_p (point Jacobi for simplicity)
        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)

        # Under-relaxation: U_new = α * U_solved + (1-α) * U_old
        U_new = alpha_U * U_solved + (1.0 - alpha_U) * U

        return U_new, A_p, H

    def _compute_continuity_error(self, phi: torch.Tensor) -> float:
        """Compute the global continuity error.

        The continuity error is the sum of flux imbalances per cell,
        normalised by the total volume.

        Args:
            phi: ``(n_faces,)`` — face flux.

        Returns:
            Global continuity error (scalar).
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # Sum flux per cell
        div_phi = torch.zeros(n_cells, dtype=phi.dtype, device=phi.device)

        # Internal faces
        div_phi = div_phi + scatter_add(phi[:n_internal], owner[:n_internal], n_cells)
        div_phi = div_phi + scatter_add(-phi[:n_internal], neighbour, n_cells)

        # Boundary faces
        if mesh.n_faces > n_internal:
            div_phi = div_phi + scatter_add(phi[n_internal:], owner[n_internal:], n_cells)

        # Normalise by cell volume
        V = cell_volumes.clamp(min=1e-30)
        div_phi = div_phi / V

        # Return L1 norm (mean absolute divergence)
        return float(div_phi.abs().mean().item())
