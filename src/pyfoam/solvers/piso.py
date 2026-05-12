"""
PISO algorithm for transient incompressible flow.

Implements the Pressure-Implicit with Splitting of Operators (PISO)
algorithm as described by Issa (1986). This is designed for transient
incompressible flow simulations.

Algorithm (per time step):
1. **Momentum predictor**: Solve A_p * U* = H(U) - grad(p)
   (no under-relaxation for PISO).
2. **First pressure correction**:
   a. Compute HbyA = H / A_p
   b. Compute phiHbyA = flux(HbyA)
   c. Solve laplacian(1/A_p, p') = div(phiHbyA)
   d. Correct velocity: U = HbyA - (1/A_p) * grad(p')
   e. Correct flux: phi = phiHbyA - (1/A_p)_f * grad(p')_f
3. **Subsequent pressure corrections** (n_correctors - 1):
   a. Recompute H from corrected U
   b. Solve pressure correction with updated H
   c. Correct velocity and flux
4. **Advance time**.

Key differences from SIMPLE:
- No under-relaxation (transient, not iterative)
- Multiple pressure corrections per step (typically 2-3)
- Requires a time-stepping outer loop

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
)
from pyfoam.solvers.pressure_equation import (
    assemble_pressure_equation,
    solve_pressure_equation,
    correct_velocity,
    correct_face_flux,
)

__all__ = ["PISOSolver", "PISOConfig"]


logger = logging.getLogger(__name__)


class PISOConfig(CoupledSolverConfig):
    """Configuration for the PISO algorithm.

    Attributes
    ----------
    n_correctors : int
        Number of pressure correction steps per time step (default 2).
    """

    def __init__(
        self,
        n_correctors: int = 2,
        **kwargs,
    ) -> None:
        # PISO does not use under-relaxation by default
        kwargs.setdefault("relaxation_factor_U", 1.0)
        kwargs.setdefault("relaxation_factor_p", 1.0)
        super().__init__(**kwargs)
        self.n_correctors = n_correctors


class PISOSolver(CoupledSolverBase):
    """PISO algorithm for transient incompressible flow.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    config : PISOConfig
        Solver configuration.

    Examples::

        config = PISOConfig(n_correctors=3)
        solver = PISOSolver(mesh, config)

        # Time loop
        for t in range(n_steps):
            U, p, phi, convergence = solver.solve(U, p, phi, ...)
    """

    def __init__(
        self,
        mesh: Any,
        config: PISOConfig | None = None,
    ) -> None:
        if config is None:
            config = PISOConfig()
        super().__init__(mesh, config)
        self._piso_config = config

    def solve(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        *,
        U_bc: torch.Tensor | None = None,
        U_old: torch.Tensor | None = None,
        p_old: torch.Tensor | None = None,
        max_outer_iterations: int = 1,
        tolerance: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run the PISO algorithm for one time step.

        Args:
            U: ``(n_cells, 3)`` — velocity field.
            p: ``(n_cells,)`` — pressure field.
            phi: ``(n_faces,)`` — face flux field.
            U_bc: ``(n_cells, 3)`` — prescribed velocity for boundary cells.
                Cells with fixed-value BCs should have their prescribed values;
                cells without BCs should have NaN. If None, no BC enforcement.
            U_old: Previous time-step velocity (for time derivative).
            p_old: Previous time-step pressure (for time derivative).
            max_outer_iterations: Not used for PISO (always 1 time step).
            tolerance: Convergence tolerance.

        Returns:
            Tuple of ``(U, p, phi, convergence_data)``.
        """
        device = self._device
        dtype = self._dtype
        mesh = self._mesh
        config = self._piso_config

        # Ensure tensors are on correct device/dtype
        U = U.to(device=device, dtype=dtype)
        p = p.to(device=device, dtype=dtype)
        phi = phi.to(device=device, dtype=dtype)

        convergence = ConvergenceData()

        # ============================================
        # Step 1: Momentum predictor
        # ============================================
        U, A_p, H = self._momentum_predictor(U, p, phi, U_old, U_bc=U_bc)

        # ============================================
        # Pressure correction loop
        # ============================================
        for corr in range(config.n_correctors):
            # Compute HbyA
            HbyA = compute_HbyA(H, A_p)

            # Compute phiHbyA
            phiHbyA = compute_face_flux_HbyA(
                HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
                mesh.n_internal_faces, mesh.face_weights,
            )

            # Assemble and solve pressure equation
            p_eqn = assemble_pressure_equation(
                phiHbyA, A_p, mesh, mesh.face_weights,
            )

            p, p_iters, p_res = solve_pressure_equation(
                p_eqn, p, self._p_solver,
                tolerance=config.p_tolerance,
                max_iter=config.p_max_iter,
            )

            # Correct velocity
            U = correct_velocity(U, HbyA, p, A_p, mesh)

            # Correct face flux
            phi = correct_face_flux(phi, p, A_p, mesh, mesh.face_weights)

            # Re-apply BCs after velocity correction
            if U_bc is not None:
                bc_mask = ~torch.isnan(U_bc[:, 0])
                if bc_mask.any():
                    U[bc_mask] = U_bc[bc_mask]

            # Recompute H for subsequent corrections (not needed for last)
            if corr < config.n_correctors - 1:
                H = self._recompute_H(U, phi)

            # Track convergence
            continuity_error = self._compute_continuity_error(phi)
            convergence.residual_history.append({
                "correction": corr,
                "p_linear_iters": p_iters,
                "p_linear_res": p_res,
                "continuity_error": continuity_error,
            })

        # Final convergence metrics
        convergence.p_residual = p_res
        convergence.continuity_error = continuity_error
        convergence.outer_iterations = 1
        convergence.converged = continuity_error < tolerance

        # Re-apply boundary conditions after pressure correction loop
        if U_bc is not None:
            bc_mask = ~torch.isnan(U_bc[:, 0])
            if bc_mask.any():
                U[bc_mask] = U_bc[bc_mask]

        logger.info(
            "PISO: %d corrections, continuity=%.6e",
            config.n_correctors, continuity_error,
        )

        return U, p, phi, convergence

    def _momentum_predictor(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        U_old: torch.Tensor | None = None,
        U_bc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the momentum equation without under-relaxation.

        For PISO, the momentum predictor is solved without relaxation
        (α_U = 1.0) since we're doing transient simulation.

        Args:
            U: ``(n_cells, 3)`` — current velocity.
            p: ``(n_cells,)`` — current pressure.
            phi: ``(n_faces,)`` — current face flux.
            U_old: Previous time-step velocity.
            U_bc: ``(n_cells, 3)`` — prescribed velocity for boundary cells.

        Returns:
            Tuple of ``(U_new, A_p, H)``.
        """
        mesh = self._mesh
        device = self._device
        dtype = self._dtype

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        int_owner = owner[:n_internal]
        int_neigh = neighbour

        # Diffusion coefficient
        nu = 1.0
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = nu * S_mag * delta_f

        # Convection (upwind)
        flux = phi[:n_internal]
        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        # Matrix coefficients
        lower = (-diff_coeff + flux_neg) / V_P
        upper = (-diff_coeff - flux_pos) / V_N

        # Diagonal
        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add((diff_coeff - flux_neg) / V_P, int_owner, n_cells)
        diag = diag + scatter_add((diff_coeff + flux_pos) / V_N, int_neigh, n_cells)

        # Source term
        source = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        # ============================================
        # Boundary condition enforcement (implicit BC method)
        # Matches SIMPLE's approach: add face diffusion coefficient
        # to diagonal and source for fixed-value boundary cells.
        # ============================================
        if U_bc is not None:
            bc_mask = ~torch.isnan(U_bc[:, 0])
            if bc_mask.any() and n_faces > n_internal:
                bnd_owner = owner[n_internal:]
                bnd_areas = mesh.face_areas[n_internal:]
                bnd_face_centres = mesh.face_centres[n_internal:]

                # Compute boundary delta using 2×d_P to match internal face convention
                owner_centres = mesh.cell_centres[bnd_owner]
                d_P = bnd_face_centres - owner_centres
                d_full = 2.0 * d_P
                bnd_S_mag = bnd_areas.norm(dim=1)
                safe_S_mag = torch.where(bnd_S_mag > 1e-30, bnd_S_mag, torch.ones_like(bnd_S_mag))
                n_hat = bnd_areas / safe_S_mag.unsqueeze(-1)
                d_dot_n = (d_full * n_hat).sum(dim=1).abs()
                bnd_delta = 1.0 / d_dot_n.clamp(min=1e-30)

                # Face diffusion coefficient: nu * |S_f| * delta_bnd
                bnd_face_coeff = nu * bnd_S_mag * bnd_delta

                # Only apply to cells that have BCs
                bnd_bc_mask = bc_mask[bnd_owner]
                bnd_face_coeff_masked = bnd_face_coeff * bnd_bc_mask.float()

                # Divide by cell volume to match per-unit-volume form
                bnd_V = gather(cell_volumes_safe, bnd_owner)
                bnd_face_coeff_pv = bnd_face_coeff_masked / bnd_V

                # Add to diagonal: internalCoeffs = face_coeff / V
                diag = diag + scatter_add(bnd_face_coeff_pv, bnd_owner, n_cells)

                # Add to source: boundaryCoeffs = face_coeff * U_bc / V
                for comp in range(3):
                    u_bc_comp = U_bc[bnd_owner, comp].nan_to_num(0.0)
                    source_contrib = bnd_face_coeff_pv * u_bc_comp
                    source[:, comp] = source[:, comp] + scatter_add(source_contrib, bnd_owner, n_cells)

        # Compute H(U): off-diagonal contributions
        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        # Use direct indexing for 2D U tensor and index_add_ for 2D scatter
        U_neigh = U[int_neigh]  # (n_internal, 3)
        U_own = U[int_owner]  # (n_internal, 3)

        # Owner receives from neighbour: lower * U_neigh * V_P
        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        # Neighbour receives from owner: upper * U_owner * V_N
        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        # H now includes off-diagonal product + BC source (no pressure gradient)
        H = H + source

        # Store BC source for _recompute_H
        self._bc_source = source.clone()

        # Pressure gradient
        w = mesh.face_weights[:n_internal]
        p_P = gather(p, int_owner)
        p_N = gather(p, int_neigh)
        p_face = w * p_P + (1.0 - w) * p_N
        p_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_p.index_add_(0, int_owner, p_contrib)
        grad_p.index_add_(0, int_neigh, -p_contrib)

        # Solve: A_p * U = H - grad(p)
        total_source = H - grad_p

        diag_safe = diag.abs().clamp(min=1e-30)
        U_new = total_source / diag_safe.unsqueeze(-1)

        # Re-apply boundary conditions directly after solve
        if U_bc is not None:
            bc_mask = ~torch.isnan(U_bc[:, 0])
            if bc_mask.any():
                U_new[bc_mask] = U_bc[bc_mask]

        return U_new, diag, H

    def _recompute_H(
        self,
        U: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute H(U) from the corrected velocity field.

        This is used in subsequent pressure corrections to get a
        better estimate of the off-diagonal contributions.
        Includes BC source contributions stored from momentum predictor.

        Args:
            U: ``(n_cells, 3)`` — corrected velocity.
            phi: ``(n_faces,)`` — corrected face flux.

        Returns:
            ``(n_cells, 3)`` — H(U) vector.
        """
        mesh = self._mesh
        device = self._device
        dtype = self._dtype

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        int_owner = owner[:n_internal]
        int_neigh = neighbour

        # Diffusion
        nu = 1.0
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = nu * S_mag * delta_f

        # Convection
        flux = phi[:n_internal]
        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        lower = (-diff_coeff + flux_neg) / V_P
        upper = (-diff_coeff - flux_pos) / V_N

        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        # Use direct indexing for 2D U tensor
        U_neigh = U[int_neigh]  # (n_internal, 3)
        U_own = U[int_owner]  # (n_internal, 3)

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        # Add stored BC source contributions
        if hasattr(self, '_bc_source'):
            H = H + self._bc_source

        return H

    def _compute_continuity_error(self, phi: torch.Tensor) -> float:
        """Compute the global continuity error.

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

        div_phi = torch.zeros(n_cells, dtype=phi.dtype, device=phi.device)
        div_phi = div_phi + scatter_add(phi[:n_internal], owner[:n_internal], n_cells)
        div_phi = div_phi + scatter_add(-phi[:n_internal], neighbour, n_cells)

        if mesh.n_faces > n_internal:
            div_phi = div_phi + scatter_add(phi[n_internal:], owner[n_internal:], n_cells)

        V = cell_volumes.clamp(min=1e-30)
        div_phi = div_phi / V

        return float(div_phi.abs().mean().item())
