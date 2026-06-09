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
    adjust_phi,
)

__all__ = ["PISOSolver", "PISOConfig"]


logger = logging.getLogger(__name__)


class PISOConfig(CoupledSolverConfig):
    """Configuration for the PISO algorithm.

    Attributes
    ----------
    n_correctors : int
        Number of pressure correction steps per time step (default 2).
    nu : float
        Kinematic viscosity (default 1.0).
    dt : float
        Time step size (default 0.01).
    """

    def __init__(
        self,
        n_correctors: int = 2,
        nu: float = 1.0,
        dt: float = 0.01,
        **kwargs,
    ) -> None:
        # PISO does not use under-relaxation by default
        kwargs.setdefault("relaxation_factor_U", 1.0)
        kwargs.setdefault("relaxation_factor_p", 1.0)
        super().__init__(**kwargs)
        self.n_correctors = n_correctors
        self.nu = nu
        self.dt = dt


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
        body_force: torch.Tensor | None = None,
        max_outer_iterations: int = 1,
        tolerance: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run the PISO algorithm for one time step.

        Args:
            U: ``(n_cells, 3)`` — velocity field.
            p: ``(n_cells,)`` — pressure field.
            phi: ``(n_faces,)`` — face flux field.
            U_bc: ``(n_cells, 3)`` — prescribed velocity for boundary cells.
            U_old: Previous time-step velocity (for time derivative).
            p_old: Previous time-step pressure (for time derivative).
            body_force: ``(n_cells, 3)`` — body force per unit volume.
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
        U_pred, A_p, H = self._momentum_predictor(U, p, phi, U_old, U_bc=U_bc, body_force=body_force)

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

            # Fix boundary face fluxes using prescribed BC velocities.
            # compute_face_flux_HbyA uses cell-centre HbyA for boundary
            # faces, but fixedValue BCs prescribe the face velocity directly.
            # Using cell-centre values causes non-zero spurious boundary
            # fluxes in closed domains (Couette, Poiseuille), corrupting
            # the pressure equation RHS and destroying the solution.
            if U_bc is not None and mesh.n_faces > mesh.n_internal_faces:
                self._fix_boundary_flux(phiHbyA, U_bc, mesh)

            # Zero out empty patch faces (2-D approximation)
            self._zero_empty_patches(phiHbyA, mesh)

            # Fix boundary fluxes for zero-gradient patches.
            # For zero-gradient BCs, the face velocity = owner cell velocity.
            # These fluxes must be included in the pressure equation source
            # so the discrete divergence is consistent with the HbyA field.
            self._fix_zerogradient_boundary_flux(phiHbyA, HbyA, mesh)

            # Adjust boundary fluxes for global conservation on CLOSED domains
            # with NO moving walls.  For domains with moving walls or open
            # boundaries, skip adjustment — the fluxes are physical.
            if U_bc is None or not self._has_moving_wall_or_open_bc(mesh, U_bc):
                adjust_phi(phiHbyA, mesh, closed=True)

            # Assemble and solve pressure equation
            p_eqn = assemble_pressure_equation(
                phiHbyA, A_p, mesh, mesh.face_weights,
            )

            saved_rel_tol = self._p_solver._rel_tol
            self._p_solver._rel_tol = 0.0
            p_old_iter = p.clone()
            p, p_iters, p_res = solve_pressure_equation(
                p_eqn, p, self._p_solver,
                tolerance=config.p_tolerance,
                max_iter=config.p_max_iter,
            )
            self._p_solver._rel_tol = saved_rel_tol

            # Pressure correction: p' = p_new - p_old
            p_prime = p - p_old_iter

            # Correct velocity: U = U - (1/A_p) * grad(p')
            # Use the CURRENT velocity U as base (not HbyA, which is tiny
            # due to V/dt dominance in the diagonal solve).  The correction
            # is only the pressure gradient adjustment to enforce continuity.
            U = correct_velocity(U, U, p_prime, A_p, mesh)

            # Re-apply boundary conditions
            if U_bc is not None:
                bc_mask = ~torch.isnan(U_bc[:, 0])
                if bc_mask.any():
                    U[bc_mask] = U_bc[bc_mask]

            # 速度限制器（防止发散）
            U_mag = U.norm(dim=1, keepdim=True).clamp(min=1e-30)
            U_limit = 1e4
            U = torch.where(U_mag > U_limit, U * (U_limit / U_mag), U)

            # Correct face flux using pressure correction
            phi = correct_face_flux(phi, p_prime, A_p, mesh, mesh.face_weights)

            # Fix boundary face fluxes using prescribed BC velocities
            if U_bc is not None and mesh.n_faces > mesh.n_internal_faces:
                self._fix_boundary_flux(phi, U_bc, mesh)

            # Zero out empty patch faces on corrected phi
            self._zero_empty_patches(phi, mesh)

            # Re-apply BCs only for non-zero prescribed velocity cells
            # (moving walls, inlets).  Do NOT re-apply for zero-velocity
            # walls — the face-based diffusion already drives those cells
            # toward zero, and forcing them to exactly zero destroys the
            # velocity gradient in closed domains like Couette flow.
            if U_bc is not None:
                bc_mask = ~torch.isnan(U_bc[:, 0])
                nonzero_bc = bc_mask & (U_bc.abs().sum(dim=1) > 1e-10)
                if nonzero_bc.any():
                    U[nonzero_bc] = U_bc[nonzero_bc]

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
        # Track velocity change (not residual — PISO is transient)
        if U_pred is not None:
            U_change = (U - U_pred).norm() / U.norm().clamp(min=1e-30)
            convergence.U_residual = float(U_change)
        else:
            convergence.U_residual = 0.0
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
        body_force: torch.Tensor | None = None,
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
        nu = self._piso_config.nu
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

        # Source term (per-unit-volume)
        source = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        # Add face-based diffusion at FIXED-VALUE wall boundary faces.
        # This is the discrete equivalent of the wall shear stress.
        # For walls with U=0: adds face_coeff to diagonal (pulls velocity to 0)
        # For moving walls: adds face_coeff*U_top to source (drives flow)
        # Skip empty patches (2D approximation) and non-wall patches —
        # only wall patches impose a velocity constraint via penalty.
        if n_faces > n_internal and hasattr(mesh, 'boundary') and U_bc is not None:
            bc_mask = ~torch.isnan(U_bc[:, 0])
            for patch in mesh.boundary:
                ptype = patch.get("type", "")
                # Only process wall patches (fixedValue velocity BCs)
                if ptype != "wall":
                    continue
                sf = patch.get("startFace", 0)
                nf = patch.get("nFaces", 0)
                if nf <= 0:
                    continue

                patch_owner = owner[sf:sf + nf]
                # Only process patches with fixedValue BCs
                if not bc_mask[patch_owner].any():
                    continue

                patch_areas = face_areas[sf:sf + nf]
                patch_fc = mesh.face_centres[sf:sf + nf]
                patch_oc = mesh.cell_centres[patch_owner]

                d_P = patch_fc - patch_oc
                patch_S_mag = patch_areas.norm(dim=1)
                safe_mag = torch.where(patch_S_mag > 1e-30, patch_S_mag, torch.ones_like(patch_S_mag))
                n_hat = patch_areas / safe_mag.unsqueeze(-1)
                d_full = 2.0 * d_P
                d_dot_n = (d_full * n_hat).sum(dim=1).abs()
                patch_delta = 1.0 / d_dot_n.clamp(min=1e-30)

                patch_coeff = nu * patch_S_mag * patch_delta
                patch_V = gather(cell_volumes_safe, patch_owner)
                patch_coeff_pv = patch_coeff / patch_V

                # Only add for cells that have BCs
                patch_bc = bc_mask[patch_owner]
                patch_coeff_masked = patch_coeff_pv * patch_bc.float()

                diag = diag + scatter_add(patch_coeff_masked, patch_owner, n_cells)

                for comp in range(3):
                    u_bc_comp = U_bc[patch_owner, comp].nan_to_num(0.0)
                    source_contrib = patch_coeff_masked * u_bc_comp
                    source[:, comp] = source[:, comp] + scatter_add(source_contrib, patch_owner, n_cells)

        # Store BC source for _recompute_H (spatial operator only, no time derivative)
        self._bc_source = source.clone()

        # Add time derivative V/dt * U_old to H and V/dt to diagonal.
        dt = self._piso_config.dt if hasattr(self._piso_config, 'dt') else 1.0
        V_over_dt = torch.zeros(1, dtype=dtype, device=device)
        if U_old is not None and dt > 0:
            V_over_dt = cell_volumes_safe / dt
            diag = diag + V_over_dt

        # Compute H from off-diagonal contributions (absolute form).
        # H is used for HbyA and the pressure equation, NOT for the
        # momentum solve (which uses the diagonal approximation).
        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]
        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)
        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)
        H = H + source * cell_volumes_safe.unsqueeze(-1)
        if body_force is not None:
            H = H + body_force.to(device=device, dtype=dtype) * cell_volumes_safe.unsqueeze(-1)
        if U_old is not None and dt > 0:
            H = H + V_over_dt.unsqueeze(-1) * U_old

        # Diagonal solve: U_new = (H - grad(p)) / A_p
        # This is the standard PISO approach (Issa 1986).
        # Pressure gradient
        w = mesh.face_weights[:n_internal]
        p_P = gather(p, int_owner)
        p_N = gather(p, int_neigh)
        p_face = w * p_P + (1.0 - w) * p_N
        p_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_p.index_add_(0, int_owner, p_contrib)
        grad_p.index_add_(0, int_neigh, -p_contrib)

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
        nu = self._piso_config.nu
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
        U_neigh = U[int_neigh]  # (n_internal, 3)
        U_own = U[int_owner]  # (n_internal, 3)

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        # Add stored BC source contributions (scale to absolute form)
        if hasattr(self, '_bc_source'):
            H = H + self._bc_source * cell_volumes_safe.unsqueeze(-1)

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

    def _fix_boundary_flux(
        self,
        phi: torch.Tensor,
        U_bc: torch.Tensor,
        mesh: Any,
    ) -> None:
        """Overwrite boundary face fluxes using per-patch prescribed velocities.

        For fixedValue boundary faces, the face velocity is the prescribed
        value from the PATCH definition, not the owner cell's velocity.
        This is critical for corner cells shared between patches (e.g.,
        movingWall and fixedWalls) where the owner cell may have the
        wrong prescribed velocity for some of its boundary faces.

        Args:
            phi: ``(n_faces,)`` — face flux (modified in-place).
            U_bc: ``(n_cells, 3)`` — prescribed velocity (NaN where no BC).
            mesh: The finite volume mesh.
        """
        n_internal = mesh.n_internal_faces
        bnd_owner = mesh.owner[n_internal:]
        bnd_areas = mesh.face_areas[n_internal:]

        # Build per-face prescribed velocity from patch definitions
        U_bnd = torch.full_like(bnd_areas, float('nan'))
        for patch in mesh.boundary:
            if patch.get("type", "") == "empty":
                continue  # skip empty patches
            sf = patch.get("startFace", 0) - n_internal
            nf = patch.get("nFaces", 0)
            if sf < 0 or nf <= 0:
                continue
            # Get prescribed velocity from the first owner cell of this patch
            first_owner = bnd_owner[sf].item()
            u_patch = U_bc[first_owner]
            if torch.isnan(u_patch[0]):
                continue
            # Apply to all faces in this patch
            U_bnd[sf:sf+nf] = u_patch.unsqueeze(0)

        has_bc = ~torch.isnan(U_bnd[:, 0])
        if has_bc.any():
            U_bnd_clean = U_bnd.nan_to_num(0.0)
            phi_bnd = (U_bnd_clean * bnd_areas).sum(dim=1)
            phi[n_internal:] = torch.where(has_bc, phi_bnd, phi[n_internal:])

    @staticmethod
    def _has_moving_wall_or_open_bc(mesh: Any, U_bc: torch.Tensor) -> bool:
        """Check if domain has moving walls or open (non-wall) boundaries.

        Returns True if adjust_phi should NOT be applied (domain is not
        a simple closed cavity with all stationary walls).
        """
        if not hasattr(mesh, 'boundary'):
            return False
        n_internal = mesh.n_internal_faces
        bc_mask = ~torch.isnan(U_bc[:, 0])
        bnd_owner = mesh.owner[n_internal:]
        for patch in mesh.boundary:
            ptype = patch.get("type", "")
            # Open patches (not wall, not empty) → don't adjust
            if ptype not in ("wall", "empty", ""):
                return True
            # Check if any wall has non-zero prescribed velocity
            if ptype == "wall":
                sf = patch.get("startFace", 0) - n_internal
                nf = patch.get("nFaces", 0)
                if sf >= 0 and nf > 0:
                    owners = bnd_owner[sf:sf+nf]
                    u_vals = U_bc[owners]
                    if (u_vals.abs().sum() > 1e-10):
                        return True
        return False

    @staticmethod
    def _fix_zerogradient_boundary_flux(
        phi: torch.Tensor,
        HbyA: torch.Tensor,
        mesh: Any,
    ) -> None:
        """Fix boundary fluxes for non-fixedValue patches.

        For zero-gradient patches (inletOutlet, etc.), the face velocity
        equals the owner cell's HbyA velocity.  These fluxes must be
        consistent with HbyA so the pressure equation source (which only
        sums over internal faces and fixedValue boundary faces) captures
        the full divergence.
        """
        if not hasattr(mesh, 'boundary'):
            return
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        face_areas = mesh.face_areas
        for patch in mesh.boundary:
            ptype = patch.get("type", "")
            # Skip empty patches (already zeroed) and wall patches
            # (handled by _fix_boundary_flux)
            if ptype in ("empty", "wall", ""):
                continue
            sf = patch.get("startFace", 0)
            nf = patch.get("nFaces", 0)
            if nf <= 0:
                continue
            for fi in range(sf, sf + nf):
                bfi = fi - n_internal
                if bfi < 0:
                    continue
                c = owner[fi].item()
                phi[fi] = (HbyA[c] * face_areas[fi]).sum()

    @staticmethod
    def _zero_empty_patches(phi: torch.Tensor, mesh: Any) -> None:
        """Zero out face fluxes on empty patch faces."""
        if not hasattr(mesh, 'boundary'):
            return
        n_internal = mesh.n_internal_faces
        for patch in mesh.boundary:
            if patch.get("type", "") == "empty":
                start = patch.get("startFace", 0) - n_internal
                n = patch.get("nFaces", 0)
                if start >= 0 and n > 0:
                    phi[n_internal + start: n_internal + start + n] = 0.0

    @staticmethod
    def _apply_boundary_pressure_correction(
        p_eqn: FvMatrix,
        phiHbyA: torch.Tensor,
        A_p: torch.Tensor,
        U_bc: torch.Tensor,
        mesh: Any,
    ) -> None:
        """Apply boundary pressure correction (fixedFluxPressure equivalent).

        For fixedValue velocity BCs, the pressure equation needs a boundary
        source term to make the corrected flux match the prescribed velocity:

            source[P] += (phiHbyA_bnd - U_bc · S_f) * (1/A_p)_f / V_P

        This ensures the pressure gradient at walls is consistent with
        the velocity BC, preventing spurious pressure accumulation.

        Args:
            p_eqn: Assembled pressure equation (source modified in-place).
            phiHbyA: ``(n_faces,)`` — face flux from HbyA.
            A_p: ``(n_cells,)`` — diagonal momentum coefficients.
            U_bc: ``(n_cells, 3)`` — prescribed velocity (NaN where no BC).
            mesh: The finite volume mesh.
        """
        n_internal = mesh.n_internal_faces
        bnd_owner = mesh.owner[n_internal:]
        bnd_areas = mesh.face_areas[n_internal:]

        # Build per-face prescribed velocity
        U_bnd = torch.full_like(bnd_areas, float('nan'))
        for patch in mesh.boundary:
            if patch.get("type", "") == "empty":
                continue
            sf = patch.get("startFace", 0) - n_internal
            nf = patch.get("nFaces", 0)
            if sf < 0 or nf <= 0:
                continue
            first_owner = bnd_owner[sf].item()
            u_patch = U_bc[first_owner]
            if torch.isnan(u_patch[0]):
                continue
            U_bnd[sf:sf+nf] = u_patch.unsqueeze(0)

        has_bc = ~torch.isnan(U_bnd[:, 0])
        if not has_bc.any():
            return

        # Prescribed boundary flux: U_bc · S_f
        U_bnd_clean = U_bnd.nan_to_num(0.0)
        phi_bc = (U_bnd_clean * bnd_areas).sum(dim=1)

        # Flux correction: (phiHbyA - U_bc · S_f)
        flux_correction = phiHbyA[n_internal:] - phi_bc

        # Face-interpolated 1/A_p for boundary faces
        A_p_safe = A_p.abs().clamp(min=1e-30)
        inv_A_p = 1.0 / A_p_safe
        inv_A_p_bnd = gather(inv_A_p, bnd_owner)

        # Source correction (per-unit-volume)
        source_correction = flux_correction * inv_A_p_bnd
        source_correction = torch.where(has_bc, source_correction, torch.zeros_like(source_correction))

        V = mesh.cell_volumes.clamp(min=1e-30)
        V_bnd = gather(V, bnd_owner)
        p_eqn.source.scatter_add_(0, bnd_owner, source_correction / V_bnd)
