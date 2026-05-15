"""
SRFSimpleFoam — steady-state incompressible solver with Single Rotating Reference Frame.

Implements the SIMPLE algorithm for steady-state incompressible
Navier-Stokes equations in a rotating reference frame.

Extends :class:`SimpleFoam` with SRF-specific source terms:

- **Coriolis force**: ``-2 ω × U``  (velocity-dependent)
- **Centrifugal force**: ``-ω × (ω × r)``  (position-dependent)

where ``ω`` is the angular velocity vector and ``r`` is the cell centre
position relative to the rotation origin.

The SRF model reads from ``constant/SRFProperties``::

    SRFProperties
    {
        selectionMode   all;        // or cellZone
        origin          (0 0 0);    // rotation centre
        axis            (0 0 1);    // rotation axis (normalised internally)
        omega           10;         // angular velocity [rad/s]
    }

Algorithm modifications (per outer iteration):

1. Compute centrifugal source: ``S_cent = -ω × (ω × r)`` (explicit)
2. Assemble momentum matrix with Coriolis implicit diagonal augmentation
3. Solve momentum predictor (includes SRF source terms)
4. Standard SIMPLE pressure-correction steps

The Coriolis force ``-2ω × U`` is treated semi-implicitly:
    - Diagonal: augmented by ``2|ω|`` (ensures diagonal dominance)
    - Source: ``-2ω × U_old`` (explicit part from previous iteration)

This matches OpenFOAM's ``SRFSimpleFoam`` treatment.

Usage::

    from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

    solver = SrfSimpleFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig
from pyfoam.solvers.coupled_solver import ConvergenceData
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

from .solver_base import SolverBase
from .simple_foam import SimpleFoam
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SrfSimpleFoam"]

logger = logging.getLogger(__name__)


class SRFProperties:
    """Container for Single Rotating Reference Frame properties.

    Parameters
    ----------
    origin : tuple[float, float, float]
        Rotation centre ``(x, y, z)``.
    axis : tuple[float, float, float]
        Rotation axis direction (will be normalised).
    omega : float
        Angular velocity in rad/s.
    """

    def __init__(
        self,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        omega: float = 0.0,
    ) -> None:
        self.origin = origin
        self.axis = self._normalise(axis)
        self.omega = omega

        # Angular velocity vector: ω_vec = omega * axis_hat
        self.omega_vec = (
            self.omega * self.axis[0],
            self.omega * self.axis[1],
            self.omega * self.axis[2],
        )

    @staticmethod
    def _normalise(v: tuple[float, float, float]) -> tuple[float, float, float]:
        """Normalise a 3-vector."""
        mag = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        if mag < 1e-30:
            return (0.0, 0.0, 1.0)
        return (v[0] / mag, v[1] / mag, v[2] / mag)

    def __repr__(self) -> str:
        return (
            f"SRFProperties(origin={self.origin}, axis={self.axis}, "
            f"omega={self.omega:.4g} rad/s)"
        )


class SrfSimpleFoam(SimpleFoam):
    """Steady-state incompressible SIMPLE solver with Single Rotating Reference Frame.

    Extends :class:`SimpleFoam` with Coriolis and centrifugal force source
    terms for flows in a rotating reference frame (e.g. mixers, turbomachinery).

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    srf_props : SRFProperties, optional
        SRF properties. If None, reads from ``constant/SRFProperties``.

    Attributes
    ----------
    srf : SRFProperties
        The SRF model properties.
    omega_vec : torch.Tensor
        ``(3,)`` angular velocity vector.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        srf_props: SRFProperties | None = None,
    ) -> None:
        # Call SimpleFoam.__init__ (not super().__init__ which calls SolverBase)
        # We need SimpleFoam's initialisation first
        super().__init__(case_path)

        # Read or use provided SRF properties
        self.srf = srf_props or self._read_srf_properties()

        # Precompute angular velocity vector as tensor
        device = get_device()
        dtype = get_default_dtype()
        self.omega_vec = torch.tensor(
            self.srf.omega_vec, dtype=dtype, device=device,
        )

        # Precompute rotation origin as tensor
        self.origin = torch.tensor(
            self.srf.origin, dtype=dtype, device=device,
        )

        # Precompute cell position vectors relative to origin: r = centre - origin
        self._r = self.mesh.cell_centres - self.origin.unsqueeze(0)

        # Precompute centrifugal force per unit volume: -ω × (ω × r)
        self._centrifugal_source = self._compute_centrifugal_force()

        # Precompute |ω| for Coriolis diagonal augmentation
        self._omega_mag = self.omega_vec.norm().item()

        logger.info(
            "SrfSimpleFoam ready: %s, omega_vec=%s, omega_mag=%.4g",
            self.srf,
            self.omega_vec.tolist(),
            self._omega_mag,
        )

    # ------------------------------------------------------------------
    # SRF property reading
    # ------------------------------------------------------------------

    def _read_srf_properties(self) -> SRFProperties:
        """Read SRF properties from ``constant/SRFProperties``.

        Returns:
            Parsed :class:`SRFProperties`.
        """
        srf_path = self.case_path / "constant" / "SRFProperties"
        if not srf_path.exists():
            logger.warning(
                "constant/SRFProperties not found; using defaults "
                "(origin=(0,0,0), axis=(0,0,1), omega=0)"
            )
            return SRFProperties()

        try:
            from pyfoam.io.dictionary import parse_dict_file
            srf = parse_dict_file(srf_path)

            origin = self._parse_vector(srf.get("origin", (0, 0, 0)))
            axis = self._parse_vector(srf.get("axis", (0, 0, 1)))
            omega = float(srf.get("omega", 0.0))

            return SRFProperties(origin=origin, axis=axis, omega=omega)

        except Exception as e:
            logger.warning("Could not read SRFProperties: %s; using defaults", e)
            return SRFProperties()

    @staticmethod
    def _parse_vector(value: Any) -> tuple[float, float, float]:
        """Parse a vector from dictionary value.

        Handles tuple/list and string formats like ``'( 0 0 1 )'``.
        """
        if isinstance(value, (tuple, list)) and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))

        if isinstance(value, str):
            match = re.search(
                r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)",
                value,
            )
            if match:
                return (
                    float(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                )

        return (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # SRF force computation
    # ------------------------------------------------------------------

    def _compute_centrifugal_force(self) -> torch.Tensor:
        """Compute centrifugal force per unit volume for each cell.

        The centrifugal force is:
            F_cent = -ω × (ω × r)

        Using the BAC-CAB rule:
            ω × (ω × r) = ω(ω·r) - r(ω·ω)
            F_cent = r|ω|² - ω(ω·r)

        This is a position-dependent explicit source term.

        Returns:
            ``(n_cells, 3)`` centrifugal force per unit volume.
        """
        omega = self.omega_vec  # (3,)
        r = self._r  # (n_cells, 3)

        # ω·r for each cell
        omega_dot_r = (r * omega.unsqueeze(0)).sum(dim=1)  # (n_cells,)

        # |ω|²
        omega_sq = omega.dot(omega)

        # F_cent = r * |ω|² - ω * (ω·r)
        F_cent = r * omega_sq - omega.unsqueeze(0) * omega_dot_r.unsqueeze(-1)

        # Negate: the source term in the momentum equation is -ω × (ω × r)
        # But OpenFOAM convention: the source added to the RHS of the momentum
        # equation for centrifugal is: -ω × (ω × r)
        # Using BAC-CAB: ω × (ω × r) = ω(ω·r) - r|ω|²
        # So -ω × (ω × r) = r|ω|² - ω(ω·r) = F_cent computed above
        # Wait, let me re-derive:
        # Centrifugal force = -ω × (ω × r)
        # ω × (ω × r) = ω(ω·r) - r(ω·ω) = ω(ω·r) - r|ω|²
        # -ω × (ω × r) = r|ω|² - ω(ω·r) = F_cent
        # So F_cent is correct as computed.

        return F_cent

    def _compute_coriolis_source(self, U: torch.Tensor) -> torch.Tensor:
        """Compute Coriolis force source term: -2 ω × U.

        The Coriolis force depends on velocity and is treated semi-implicitly:
        - The diagonal augmentation is handled in the momentum matrix
        - This computes the explicit source part

        For semi-implicit treatment:
            -2ω × U = -2ω × U_diag + (-2ω × (U - U_diag))
        where U_diag is the part absorbed into the diagonal.

        In OpenFOAM, the full Coriolis is treated implicitly by augmenting
        the diagonal of the momentum matrix by 2|ω| and adding
        -2ω × U_old to the source.

        This function returns the full explicit Coriolis source:
            S_coriolis = -2ω × U

        Args:
            U: ``(n_cells, 3)`` velocity field.

        Returns:
            ``(n_cells, 3)`` Coriolis force per unit volume.
        """
        omega = self.omega_vec  # (3,)

        # Cross product: ω × U
        # (ω × U)_i = ε_ijk ω_j U_k
        omega_cross_U = torch.zeros_like(U)
        omega_cross_U[:, 0] = omega[1] * U[:, 2] - omega[2] * U[:, 1]
        omega_cross_U[:, 1] = omega[2] * U[:, 0] - omega[0] * U[:, 2]
        omega_cross_U[:, 2] = omega[0] * U[:, 1] - omega[1] * U[:, 0]

        # Coriolis force: -2ω × U
        return -2.0 * omega_cross_U

    def _compute_coriolis_contribution(
        self,
        U: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Coriolis semi-implicit treatment contributions.

        In OpenFOAM's SRFSimpleFoam, the Coriolis force is treated
        semi-implicitly to enhance stability:

            -2ω × U = -(2|ω|) * U + (2|ω| * U - 2ω × U)

        The first term is absorbed into the diagonal (implicit),
        and the second is the explicit source correction.

        However, a simpler and more common approach is:
            Diagonal augmentation: +2|ω| (ensures diagonal dominance)
            Source: -2ω × U_old (explicit)

        This function returns:
            diag_augment: scalar value to add to diagonal per cell
            source: ``(n_cells, 3)`` explicit source

        Args:
            U: ``(n_cells, 3)`` current velocity field.

        Returns:
            Tuple of ``(diag_augment, source)``.
        """
        # Diagonal augmentation for Coriolis implicit treatment
        # Adding 2|ω| to diagonal ensures the matrix remains diagonally
        # dominant even with the rotational source term
        diag_augment = 2.0 * self._omega_mag

        # The explicit source is the full Coriolis force
        source = self._compute_coriolis_source(U)

        return diag_augment, source

    # ------------------------------------------------------------------
    # Modified SIMPLE iteration with SRF
    # ------------------------------------------------------------------

    def _build_solver_with_srf(self) -> "_SRFSIMPLESolver":
        """Build an SRF-aware SIMPLE solver.

        Returns:
            An :class:`_SRFSIMPLESolver` that incorporates SRF source terms.
        """
        config = SIMPLEConfig(
            n_correctors=1,
            p_solver=self.p_solver,
            U_solver=self.U_solver,
            p_tolerance=self.p_tolerance,
            U_tolerance=self.U_tolerance,
            p_max_iter=self.p_max_iter,
            U_max_iter=self.U_max_iter,
            n_non_orthogonal_correctors=self.n_non_orth_correctors,
            relaxation_factor_p=self.alpha_p,
            relaxation_factor_U=self.alpha_U,
            nu=self.nu,
        )
        return _SRFSIMPLESolver(
            self.mesh,
            config,
            omega_vec=self.omega_vec,
            centrifugal_source=self._centrifugal_source,
            omega_mag=self._omega_mag,
        )

    def run(self) -> ConvergenceData:
        """Run the SRFSimpleFoam solver.

        Executes the SIMPLE algorithm with SRF source terms in a
        time-stepping loop until convergence or ``endTime`` is reached.

        Returns:
            Final :class:`ConvergenceData`.
        """
        solver = self._build_solver_with_srf()

        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=self.convergence_tolerance,
            min_steps=1,
        )

        logger.info("Starting SRFSimpleFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  SRF: %s", self.srf)
        logger.info("  relaxation: alpha_U=%.2f, alpha_p=%.2f", self.alpha_U, self.alpha_p)

        # Build boundary conditions
        U_bc = self._build_boundary_conditions()

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Update turbulence model (if active)
            nu_field = self._update_turbulence()

            # Run one SIMPLE outer iteration with SRF
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                nu_field=nu_field,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Check convergence
            residuals = {
                "U": conv.U_residual,
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            converged = convergence.update(step + 1, residuals)

            # Write fields if needed
            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        # Write final fields
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("SRFSimpleFoam completed successfully (converged)")
            else:
                logger.warning("SRFSimpleFoam completed without full convergence")

        return last_convergence or ConvergenceData()


class _SRFSIMPLESolver(SIMPLESolver):
    """SIMPLE solver with SRF (Single Rotating Reference Frame) source terms.

    Extends :class:`SIMPLESolver` to add Coriolis and centrifugal forces
    to the momentum equation before solving.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    config : SIMPLEConfig
        Solver configuration.
    omega_vec : torch.Tensor
        ``(3,)`` angular velocity vector.
    centrifugal_source : torch.Tensor
        ``(n_cells, 3)`` centrifugal force per unit volume.
    omega_mag : float
        Magnitude of angular velocity.
    """

    def __init__(
        self,
        mesh: Any,
        config: SIMPLEConfig,
        *,
        omega_vec: torch.Tensor,
        centrifugal_source: torch.Tensor,
        omega_mag: float,
    ) -> None:
        super().__init__(mesh, config)
        self._omega_vec = omega_vec
        self._centrifugal_source = centrifugal_source
        self._omega_mag = omega_mag

    def solve(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        *,
        U_bc: torch.Tensor | None = None,
        U_old: torch.Tensor | None = None,
        p_old: torch.Tensor | None = None,
        nu_field: torch.Tensor | None = None,
        max_outer_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run the SIMPLE algorithm with SRF source terms.

        The SRF modifications are:
        1. Add centrifugal force to momentum source
        2. Add Coriolis force (semi-implicit) to momentum equation
        3. Standard pressure-correction steps

        Args:
            U: ``(n_cells, 3)`` velocity field.
            p: ``(n_cells,)`` pressure field.
            phi: ``(n_faces,)`` face flux field.
            U_bc: ``(n_cells, 3)`` prescribed velocity for boundary cells.
            nu_field: ``(n_cells,)`` per-cell effective viscosity.
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

        for outer in range(max_outer_iterations):
            U_prev = U.clone()
            p_prev = p.clone()

            # ============================================
            # Step 1: Momentum predictor with SRF
            # ============================================
            U, A_p, H, mat_lower, mat_upper = self._momentum_predictor_srf(
                U, p, phi, U_bc=U_bc, nu_field=nu_field,
            )

            # ============================================
            # Step 2: Compute HbyA
            # ============================================
            HbyA = compute_HbyA(H, A_p)

            # Constrain HbyA at boundary cells
            if U_bc is not None:
                bc_mask = ~torch.isnan(U_bc[:, 0])
                if bc_mask.any():
                    HbyA[bc_mask] = U_bc[bc_mask]

            # ============================================
            # Step 3: Compute phiHbyA
            # ============================================
            phiHbyA = compute_face_flux_HbyA(
                HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
                mesh.n_internal_faces, mesh.face_weights,
            )

            # ============================================
            # Step 4-5: Pressure equation
            # ============================================
            A_p_eff = A_p.clone()

            p_eqn = assemble_pressure_equation(
                phiHbyA, A_p_eff, mesh, mesh.face_weights,
            )

            p_prime, p_iters, p_res = solve_pressure_equation(
                p_eqn, torch.zeros_like(p), self._p_solver,
                tolerance=config.p_tolerance,
                max_iter=config.p_max_iter,
            )

            # ============================================
            # Step 6: Correct flux
            # ============================================
            phi = correct_face_flux(phiHbyA, p_prime, A_p_eff, mesh, mesh.face_weights)

            # Under-relax pressure
            alpha_p = config.relaxation_factor_p
            p_prime = alpha_p * p_prime

            # Accumulate pressure
            p = p_prev + p_prime

            # ============================================
            # Step 7: Correct velocity
            # ============================================
            U = correct_velocity(U, HbyA, p, A_p_eff, mesh)

            # Re-apply boundary conditions
            if U_bc is not None:
                bc_mask = ~torch.isnan(U_bc[:, 0])
                if bc_mask.any():
                    U[bc_mask] = U_bc[bc_mask]

            # ============================================
            # Step 8: Check convergence
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

            if outer % 10 == 0 or outer < 5:
                logger.info(
                    "SRFSIMPLE iteration %d: U_res=%.6e, p_res=%.6e, "
                    "continuity=%.6e",
                    outer, U_residual, p_residual, continuity_error,
                )

            if continuity_error < tolerance and outer > 0:
                convergence.converged = True
                logger.info(
                    "SRFSIMPLE converged in %d iterations (continuity=%.6e)",
                    outer + 1, continuity_error,
                )
                break

            if torch.isnan(U).any() or torch.isnan(p).any():
                logger.error("SRFSIMPLE diverged at iteration %d (NaN)", outer + 1)
                break

        if not convergence.converged:
            logger.warning(
                "SRFSIMPLE did not converge in %d iterations (continuity=%.6e)",
                max_outer_iterations, continuity_error,
            )

        return U, p, phi, convergence

    def _momentum_predictor_srf(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        U_bc: torch.Tensor | None = None,
        nu_field: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the momentum equation with SRF source terms.

        Adds the following to the standard momentum equation:
        - Centrifugal source: ``-ω × (ω × r)`` (explicit)
        - Coriolis semi-implicit: diagonal augmented by ``2|ω|``,
          source includes ``-2ω × U_old``

        Args:
            U: ``(n_cells, 3)`` velocity field.
            p: ``(n_cells,)`` pressure field.
            phi: ``(n_faces,)`` face flux.
            U_bc: ``(n_cells, 3)`` boundary conditions.
            nu_field: ``(n_cells,)`` effective viscosity.

        Returns:
            Tuple of ``(U_new, A_p_eff, H, lower, upper)``.
        """
        mesh = self._mesh
        device = self._device
        dtype = self._dtype
        config = self._simple_config
        alpha_U = config.relaxation_factor_U

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # ============================================
        # Build momentum matrix (same as standard SIMPLE)
        # ============================================
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients
        cell_volumes_safe = cell_volumes.clamp(min=1e-30)

        mat = FvMatrix(
            n_cells, owner[:n_internal], neighbour,
            device=device, dtype=dtype,
        )

        int_owner = owner[:n_internal]
        int_neigh = neighbour

        # Diffusion coefficient
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        if nu_field is not None:
            nu_field = nu_field.to(device=device, dtype=dtype)
            nu_face = 0.5 * (
                gather(nu_field, int_owner) + gather(nu_field, int_neigh)
            )
            diff_coeff = nu_face * S_mag * delta_f
        else:
            nu = config.nu
            diff_coeff = nu * S_mag * delta_f

        # Convection (upwind)
        flux = phi[:n_internal]
        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        mat.lower = (-diff_coeff + flux_neg) / V_P
        mat.upper = (-diff_coeff - flux_pos) / V_N

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add((diff_coeff - flux_neg) / V_P, int_owner, n_cells)
        diag = diag + scatter_add((diff_coeff + flux_pos) / V_N, int_neigh, n_cells)
        mat.diag = diag.clone()

        # ============================================
        # Source term: -grad(p) + SRF
        # ============================================
        w = mesh.face_weights[:n_internal]
        p_P = gather(p, int_owner)
        p_N = gather(p, int_neigh)
        p_face = w * p_P + (1.0 - w) * p_N
        p_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_p.index_add_(0, int_owner, p_contrib)
        grad_p.index_add_(0, int_neigh, -p_contrib)
        grad_p = grad_p / cell_volumes_safe.unsqueeze(-1)

        source = -grad_p

        # ============================================
        # SRF: Add centrifugal force source
        # S_cent = -ω × (ω × r) per unit volume
        # ============================================
        source = source + self._centrifugal_source

        # ============================================
        # SRF: Add Coriolis semi-implicit treatment
        # Diagonal: +2|ω| (implicit part for stability)
        # Source: -2ω × U (explicit part)
        # ============================================
        # Coriolis explicit source: -2ω × U
        omega = self._omega_vec
        omega_cross_U = torch.zeros_like(U)
        omega_cross_U[:, 0] = omega[1] * U[:, 2] - omega[2] * U[:, 1]
        omega_cross_U[:, 1] = omega[2] * U[:, 0] - omega[0] * U[:, 2]
        omega_cross_U[:, 2] = omega[0] * U[:, 1] - omega[1] * U[:, 0]
        coriolis_source = -2.0 * omega_cross_U

        source = source + coriolis_source

        # Coriolis diagonal augmentation: +2|ω|
        coriolis_diag = 2.0 * self._omega_mag
        diag = diag + coriolis_diag

        # ============================================
        # Boundary conditions (same as standard SIMPLE)
        # ============================================
        if U_bc is not None:
            bc_mask = ~torch.isnan(U_bc[:, 0])
            if bc_mask.any() and n_faces > n_internal:
                bnd_owner = owner[n_internal:]
                bnd_areas = mesh.face_areas[n_internal:]
                bnd_face_centres = mesh.face_centres[n_internal:]

                owner_centres = mesh.cell_centres[bnd_owner]
                d_P = bnd_face_centres - owner_centres
                d_full = 2.0 * d_P
                bnd_S_mag = bnd_areas.norm(dim=1)
                safe_S_mag = torch.where(bnd_S_mag > 1e-30, bnd_S_mag, torch.ones_like(bnd_S_mag))
                n_hat = bnd_areas / safe_S_mag.unsqueeze(-1)
                d_dot_n = (d_full * n_hat).sum(dim=1).abs()
                bnd_delta = 1.0 / d_dot_n.clamp(min=1e-30)

                if nu_field is not None:
                    bnd_nu = gather(nu_field, bnd_owner)
                    bnd_face_coeff = bnd_nu * bnd_S_mag * bnd_delta
                else:
                    bnd_face_coeff = nu * bnd_S_mag * bnd_delta

                bnd_bc_mask = bc_mask[bnd_owner]
                bnd_face_coeff_masked = bnd_face_coeff * bnd_bc_mask.float()

                bnd_V = gather(cell_volumes_safe, bnd_owner)
                bnd_face_coeff_pv = bnd_face_coeff_masked / bnd_V

                diag = diag + scatter_add(bnd_face_coeff_pv, bnd_owner, n_cells)

                for comp in range(3):
                    u_bc_comp = U_bc[bnd_owner, comp].nan_to_num(0.0)
                    source_contrib = bnd_face_coeff_pv * u_bc_comp
                    source[:, comp] = source[:, comp] + scatter_add(source_contrib, bnd_owner, n_cells)

        # ============================================
        # Implicit under-relaxation
        # ============================================
        sum_off = torch.zeros(n_cells, dtype=dtype, device=device)
        sum_off = sum_off + scatter_add(mat.lower.abs(), int_owner, n_cells)
        sum_off = sum_off + scatter_add(mat.upper.abs(), int_neigh, n_cells)

        D_dominant = torch.max(diag.abs(), sum_off)
        D_new = D_dominant / alpha_U
        mat.diag = D_new

        source = source + (D_new - diag).unsqueeze(-1) * U
        A_p_eff = D_new.clone()

        mat.source = source

        # ============================================
        # Solve momentum equation
        # ============================================
        U_solved = torch.zeros_like(U)
        for comp in range(3):
            mat.source = source[:, comp]
            U_comp, _, _ = mat.solve(
                self._U_solver, U[:, comp],
                tolerance=config.U_tolerance,
                max_iter=config.U_max_iter,
            )
            U_solved[:, comp] = U_comp

        # Re-apply BCs
        if U_bc is not None:
            bc_mask = ~torch.isnan(U_bc[:, 0])
            if bc_mask.any():
                U_solved[bc_mask] = U_bc[bc_mask]

        # ============================================
        # Compute H from solved U
        # ============================================
        H_from_Ustar = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh_solved = U_solved[int_neigh]
        H_from_Ustar.index_add_(0, int_owner, -mat.lower.unsqueeze(-1) * U_neigh_solved)
        U_own_solved = U_solved[int_owner]
        H_from_Ustar.index_add_(0, int_neigh, -mat.upper.unsqueeze(-1) * U_own_solved)

        H_from_Ustar = H_from_Ustar + source

        return U_solved, A_p_eff, H_from_Ustar, mat.lower.clone(), mat.upper.clone()
