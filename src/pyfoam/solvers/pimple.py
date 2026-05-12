"""
PIMPLE algorithm — merged PISO + SIMPLE for transient incompressible flow.

Implements the PIMPLE algorithm used in OpenFOAM, which combines:
- **Outer SIMPLE loop**: under-relaxation for stability with large time steps
- **Inner PISO corrections**: multiple pressure corrections per outer iteration

This is the recommended algorithm for transient incompressible flow when:
- The time step is too large for pure PISO
- Under-relaxation is needed for stability
- Both transient accuracy and convergence are required

Algorithm (per time step):
1. **Outer loop** (SIMPLE-like):
   a. **Momentum predictor**: Solve with under-relaxation.
   b. **PISO correction loop** (2+ corrections):
      - Compute HbyA, phiHbyA
      - Solve pressure equation
      - Correct velocity and flux
   c. **Check convergence** (outer loop residual)
2. **Advance time**.

When n_outer_correctors = 1, PIMPLE reduces to PISO.
When n_outer_correctors > 1, it adds the SIMPLE outer loop on top.

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

__all__ = ["PIMPLESolver", "PIMPLEConfig"]


logger = logging.getLogger(__name__)


class PIMPLEConfig(CoupledSolverConfig):
    """Configuration for the PIMPLE algorithm.

    Attributes
    ----------
    n_outer_correctors : int
        Number of outer SIMPLE-like iterations per time step (default 3).
    n_correctors : int
        Number of PISO pressure corrections per outer iteration (default 2).
    """

    def __init__(
        self,
        n_outer_correctors: int = 3,
        n_correctors: int = 2,
        **kwargs,
    ) -> None:
        kwargs.setdefault("relaxation_factor_U", 0.7)
        kwargs.setdefault("relaxation_factor_p", 0.3)
        super().__init__(**kwargs)
        self.n_outer_correctors = n_outer_correctors
        self.n_correctors = n_correctors


class PIMPLESolver(CoupledSolverBase):
    """PIMPLE algorithm for transient incompressible flow.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    config : PIMPLEConfig
        Solver configuration.

    Examples::

        config = PIMPLEConfig(
            n_outer_correctors=3,
            n_correctors=2,
            relaxation_factor_U=0.7,
            relaxation_factor_p=0.3,
        )
        solver = PIMPLESolver(mesh, config)

        # Time loop
        for t in range(n_steps):
            U, p, phi, convergence = solver.solve(U, p, phi, ...)
    """

    def __init__(
        self,
        mesh: Any,
        config: PIMPLEConfig | None = None,
    ) -> None:
        if config is None:
            config = PIMPLEConfig()
        super().__init__(mesh, config)
        self._pimple_config = config

    def solve(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        *,
        U_bc: torch.Tensor | None = None,
        U_old: torch.Tensor | None = None,
        p_old: torch.Tensor | None = None,
        max_outer_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run the PIMPLE algorithm for one time step.

        Args:
            U: ``(n_cells, 3)`` — velocity field.
            p: ``(n_cells,)`` — pressure field.
            phi: ``(n_faces,)`` — face flux field.
            U_bc: ``(n_cells, 3)`` — prescribed velocity for boundary cells.
                Cells with fixed-value BCs should have their prescribed values;
                cells without BCs should have NaN. If None, no BC enforcement.
            U_old: Previous time-step velocity (for time derivative).
            p_old: Previous time-step pressure (for time derivative).
            max_outer_iterations: Maximum outer-loop iterations (overrides config).
            tolerance: Convergence tolerance on continuity residual.

        Returns:
            Tuple of ``(U, p, phi, convergence_data)``.
        """
        device = self._device
        dtype = self._dtype
        mesh = self._mesh
        config = self._pimple_config

        # Ensure tensors are on correct device/dtype
        U = U.to(device=device, dtype=dtype)
        p = p.to(device=device, dtype=dtype)
        phi = phi.to(device=device, dtype=dtype)

        convergence = ConvergenceData()
        n_outer = min(config.n_outer_correctors, max_outer_iterations)

        # Store for relaxation
        U_old_iter = U.clone()
        p_old_iter = p.clone()

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()

            # ============================================
            # Momentum predictor (with relaxation)
            # ============================================
            U, A_p, H = self._momentum_predictor(U, p, phi, U_old, U_bc=U_bc)

            # ============================================
            # PISO correction loop
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

                # Recompute H for subsequent corrections
                if corr < config.n_correctors - 1:
                    H = self._recompute_H(U, phi)

            # ============================================
            # Under-relaxation (SIMPLE-like)
            # ============================================
            alpha_U = config.relaxation_factor_U
            alpha_p = config.relaxation_factor_p

            if alpha_U < 1.0:
                U = alpha_U * U + (1.0 - alpha_U) * U_prev
            if alpha_p < 1.0:
                p = alpha_p * p + (1.0 - alpha_p) * p_prev

            # Re-apply boundary conditions after relaxation
            if U_bc is not None:
                bc_mask = ~torch.isnan(U_bc[:, 0])
                if bc_mask.any():
                    U[bc_mask] = U_bc[bc_mask]

            # ============================================
            # Check convergence
            # ============================================
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)
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
            })

            if outer % 5 == 0 or outer < 3:
                logger.info(
                    "PIMPLE outer %d: U_res=%.6e, p_res=%.6e, "
                    "continuity=%.6e",
                    outer, U_residual, p_residual, continuity_error,
                )

            if continuity_error < tolerance and outer > 0:
                convergence.converged = True
                logger.info(
                    "PIMPLE converged in %d outer iterations (continuity=%.6e)",
                    outer + 1, continuity_error,
                )
                break

        if not convergence.converged:
            logger.warning(
                "PIMPLE did not converge in %d outer iterations "
                "(continuity=%.6e)",
                n_outer, continuity_error,
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
        """Solve the momentum equation with under-relaxation.

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
        config = self._pimple_config
        alpha_U = config.relaxation_factor_U

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

        # Diffusion
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

        # H(U) - off-diagonal contributions
        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]  # (n_internal, 3)
        U_own = U[int_owner]  # (n_internal, 3)

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        # H includes off-diagonal product + BC source (no pressure gradient)
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
        U_solved = total_source / diag_safe.unsqueeze(-1)

        # Under-relaxation
        U_new = alpha_U * U_solved + (1.0 - alpha_U) * U

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

        nu = 1.0
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = nu * S_mag * delta_f

        flux = phi[:n_internal]
        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        lower = (-diff_coeff + flux_neg) / V_P
        upper = (-diff_coeff - flux_pos) / V_N

        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
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
