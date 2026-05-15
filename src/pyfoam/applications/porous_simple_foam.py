"""
PorousSimpleFoam — steady-state incompressible solver with porous media zones.

Implements the SIMPLE algorithm for steady-state incompressible
Navier-Stokes equations with Darcy-Forchheimer porous media resistance
and optional Multiple Reference Frame (MRF) zones.

Extends :class:`SimpleFoam` with:

- **Darcy-Forchheimer resistance**: ``S = -(μ d + ρ|U| f/2) U``
- **MRF zones**: Coriolis and centrifugal forces in rotating regions

The porous media model reads from ``constant/porosityProperties``::

    porosity1
    {
        type            DarcyForchheimer;
        cellZone        porosity;
        d   (5e7 -1000 -1000);     // Darcy coefficients [1/m²]
        f   (0 0 0);               // Forchheimer coefficients [1/m]
    }

The MRF zone reads from ``constant/MRFProperties``::

    MRFProperties
    {
        cellZone        rotor;
        origin          (0 0 0);
        axis            (0 0 1);
        omega           100;
    }

Algorithm modifications (per outer iteration):

1. Assemble standard momentum matrix (diffusion + convection)
2. Add porous resistance to diagonal and source in porous cells
3. Add MRF Coriolis/centrifugal forces in MRF cells
4. Apply implicit under-relaxation
5. Solve momentum predictor
6. Standard SIMPLE pressure-correction steps

**Darcy-Forchheimer Model**:

The resistance tensor per cell is:

    Cd = μ * D + ρ * |U| * F

where ``D = diag(d_x, d_y, d_z)`` is the Darcy tensor and
``F = 0.5 * diag(f_x, f_y, f_z)`` is the Forchheimer tensor.

The contribution to the momentum matrix is:
    - Diagonal: ``+V * tr(Cd)`` (implicit isotropic part)
    - Source: ``-V * (Cd - I * tr(Cd)) · U`` (explicit anisotropic part)

This matches OpenFOAM's ``DarcyForchheimer::apply()`` implementation.

Usage::

    from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

    solver = PorousSimpleFoam("path/to/case")
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

__all__ = ["PorousSimpleFoam"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Porous properties container
# ---------------------------------------------------------------------------

class PorousZoneProperties:
    """Container for Darcy-Forchheimer porous zone properties.

    Parameters
    ----------
    name : str
        Zone name (for logging).
    cell_zone : str
        Name of the cell zone where porosity applies.
    d : tuple[float, float, float]
        Darcy coefficients ``(d_x, d_y, d_z)`` [1/m²].
        These represent viscous resistance: ``S_viscous = -μ d U``.
    f : tuple[float, float, float]
        Forchheimer coefficients ``(f_x, f_y, f_z)`` [1/m].
        These represent inertial resistance: ``S_inertial = -ρ |U| f/2 U``.
    """

    def __init__(
        self,
        name: str = "porosity1",
        cell_zone: str = "porosity",
        d: tuple[float, float, float] = (0.0, 0.0, 0.0),
        f: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.name = name
        self.cell_zone = cell_zone
        self.d = d
        self.f = f

    def __repr__(self) -> str:
        return (
            f"PorousZoneProperties(name={self.name!r}, "
            f"d={self.d}, f={self.f})"
        )


class MRFZoneProperties:
    """Container for Multiple Reference Frame zone properties.

    Parameters
    ----------
    name : str
        Zone name.
    cell_zone : str
        Name of the cell zone where MRF applies.
    origin : tuple[float, float, float]
        Rotation centre ``(x, y, z)``.
    axis : tuple[float, float, float]
        Rotation axis direction (will be normalised).
    omega : float
        Angular velocity in rad/s.
    """

    def __init__(
        self,
        name: str = "MRF",
        cell_zone: str = "rotor",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        omega: float = 0.0,
    ) -> None:
        self.name = name
        self.cell_zone = cell_zone
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
            f"MRFZoneProperties(name={self.name!r}, "
            f"axis={self.axis}, omega={self.omega:.4g} rad/s)"
        )


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

class PorousSimpleFoam(SimpleFoam):
    """Steady-state incompressible SIMPLE solver with porous media zones.

    Extends :class:`SimpleFoam` with Darcy-Forchheimer porous resistance
    and optional MRF (Multiple Reference Frame) zones for flows through
    porous media (e.g. filters, packed beds, heat exchangers).

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    porous_zones : list[PorousZoneProperties], optional
        Porous zone definitions. If None, reads from ``constant/porosityProperties``.
    mrf_zones : list[MRFZoneProperties], optional
        MRF zone definitions. If None, reads from ``constant/MRFProperties``.

    Attributes
    ----------
    porous_zones : list[PorousZoneProperties]
        The porous zone definitions.
    mrf_zones : list[MRFZoneProperties]
        The MRF zone definitions.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        porous_zones: list[PorousZoneProperties] | None = None,
        mrf_zones: list[MRFZoneProperties] | None = None,
    ) -> None:
        super().__init__(case_path)

        # Read or use provided porous properties
        self.porous_zones = porous_zones if porous_zones is not None else self._read_porous_properties()

        # Read or use provided MRF properties
        self.mrf_zones = mrf_zones if mrf_zones is not None else self._read_mrf_properties()

        # Precompute MRF tensors
        device = get_device()
        dtype = get_default_dtype()
        self._mrf_data: list[dict[str, Any]] = []

        for mrf in self.mrf_zones:
            omega_vec = torch.tensor(mrf.omega_vec, dtype=dtype, device=device)
            origin = torch.tensor(mrf.origin, dtype=dtype, device=device)
            omega_mag = omega_vec.norm().item()

            # Build cell mask for MRF zone
            cell_mask = self._build_cell_mask(mrf.cell_zone)

            # Precompute r = centre - origin for MRF cells
            r = self.mesh.cell_centres[cell_mask] - origin.unsqueeze(0)

            # Precompute centrifugal force for MRF cells
            omega_sq = omega_vec.dot(omega_vec)
            omega_dot_r = (r * omega_vec.unsqueeze(0)).sum(dim=1)
            centrifugal = r * omega_sq - omega_vec.unsqueeze(0) * omega_dot_r.unsqueeze(-1)

            self._mrf_data.append({
                "name": mrf.name,
                "cell_mask": cell_mask,
                "omega_vec": omega_vec,
                "omega_mag": omega_mag,
                "centrifugal": centrifugal,
                "origin": origin,
            })

        logger.info(
            "PorousSimpleFoam ready: %d porous zone(s), %d MRF zone(s)",
            len(self.porous_zones), len(self.mrf_zones),
        )
        for pz in self.porous_zones:
            logger.info("  Porous zone: %s", pz)
        for mrf in self.mrf_zones:
            logger.info("  MRF zone: %s", mrf)

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_porous_properties(self) -> list[PorousZoneProperties]:
        """Read porous properties from ``constant/porosityProperties``.

        Returns:
            List of parsed :class:`PorousZoneProperties`.
        """
        pp_path = self.case_path / "constant" / "porosityProperties"
        if not pp_path.exists():
            logger.info("constant/porosityProperties not found; no porous zones")
            return []

        try:
            from pyfoam.io.dictionary import parse_dict_file
            pp = parse_dict_file(pp_path)

            zones = []
            for key, value in pp.items():
                if not isinstance(value, dict):
                    continue
                zone_type = str(value.get("type", "")).strip()
                if zone_type != "DarcyForchheimer":
                    logger.warning("Skipping non-DarcyForchheimer zone: %s", key)
                    continue

                cell_zone = str(value.get("cellZone", key)).strip()
                d = self._parse_resistance_vector(value.get("d", (0, 0, 0)))
                f = self._parse_resistance_vector(value.get("f", (0, 0, 0)))

                zones.append(PorousZoneProperties(
                    name=key,
                    cell_zone=cell_zone,
                    d=d,
                    f=f,
                ))

            return zones

        except Exception as e:
            logger.warning("Could not read porosityProperties: %s", e)
            return []

    def _read_mrf_properties(self) -> list[MRFZoneProperties]:
        """Read MRF properties from ``constant/MRFProperties``.

        Returns:
            List of parsed :class:`MRFZoneProperties`.
        """
        mrf_path = self.case_path / "constant" / "MRFProperties"
        if not mrf_path.exists():
            logger.info("constant/MRFProperties not found; no MRF zones")
            return []

        try:
            from pyfoam.io.dictionary import parse_dict_file
            mrf = parse_dict_file(mrf_path)

            zones = []
            for key, value in mrf.items():
                if not isinstance(value, dict):
                    continue

                cell_zone = str(value.get("cellZone", key)).strip()
                origin = self._parse_vector(value.get("origin", (0, 0, 0)))
                axis = self._parse_vector(value.get("axis", (0, 0, 1)))
                omega = float(value.get("omega", 0.0))

                zones.append(MRFZoneProperties(
                    name=key,
                    cell_zone=cell_zone,
                    origin=origin,
                    axis=axis,
                    omega=omega,
                ))

            return zones

        except Exception as e:
            logger.warning("Could not read MRFProperties: %s", e)
            return []

    def _build_cell_mask(self, zone_name: str) -> torch.Tensor:
        """Build a boolean mask for cells in a named zone.

        If the zone name is ``"all"``, returns a mask selecting all cells.
        Otherwise, reads the cell zone from the mesh.

        Args:
            zone_name: Name of the cell zone.

        Returns:
            ``(n_cells,)`` boolean tensor.
        """
        n_cells = self.mesh.n_cells

        if zone_name == "all":
            return torch.ones(n_cells, dtype=torch.bool)

        # Try to read cell zones from the mesh
        try:
            zone_path = self.case_path / "constant" / "polyMesh" / "cellZones"
            if zone_path.exists():
                # Parse cellZones file
                from pyfoam.io.dictionary import parse_dict_file
                zones = parse_dict_file(zone_path)
                if zone_name in zones:
                    zone_data = zones[zone_name]
                    if isinstance(zone_data, (list, tuple)):
                        indices = [int(i) for i in zone_data]
                        mask = torch.zeros(n_cells, dtype=torch.bool)
                        for idx in indices:
                            if 0 <= idx < n_cells:
                                mask[idx] = True
                        return mask
        except Exception as e:
            logger.debug("Could not read cellZones: %s", e)

        # Fallback: if zone not found, use all cells
        logger.warning(
            "Cell zone '%s' not found; applying to ALL cells", zone_name,
        )
        return torch.ones(n_cells, dtype=torch.bool)

    @staticmethod
    def _parse_vector(value: Any) -> tuple[float, float, float]:
        """Parse a vector from dictionary value."""
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

    @staticmethod
    def _parse_resistance_vector(value: Any) -> tuple[float, float, float]:
        """Parse a resistance vector (d or f coefficients).

        Negative values are treated as multipliers of the maximum positive
        component (OpenFOAM convention).
        """
        vec = PorousSimpleFoam._parse_vector(value)

        # Handle negative resistance (OpenFOAM convention)
        max_cmpt = max(vec)
        if max_cmpt < 0:
            # All negative: invalid, return zeros
            return (0.0, 0.0, 0.0)

        result = list(vec)
        for i in range(3):
            if result[i] < 0:
                result[i] = result[i] * (-max_cmpt)

        return tuple(result)

    # ------------------------------------------------------------------
    # Build solver
    # ------------------------------------------------------------------

    def _build_solver_with_porosity(self) -> "_PorousSIMPLESolver":
        """Build a porous-aware SIMPLE solver.

        Returns:
            A :class:`_PorousSIMPLESolver` that incorporates porous resistance.
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

        # Build porous zone data
        device = get_device()
        dtype = get_default_dtype()
        porous_data: list[dict[str, Any]] = []

        for pz in self.porous_zones:
            cell_mask = self._build_cell_mask(pz.cell_zone)

            # Darcy tensor components: d_x, d_y, d_z
            d_tensor = torch.tensor(pz.d, dtype=dtype, device=device)
            # Forchheimer tensor components: f_x, f_y, f_z
            f_tensor = torch.tensor(pz.f, dtype=dtype, device=device)

            porous_data.append({
                "name": pz.name,
                "cell_mask": cell_mask,
                "d": d_tensor,
                "f": f_tensor,
            })

        return _PorousSIMPLESolver(
            self.mesh,
            config,
            porous_data=porous_data,
            mrf_data=self._mrf_data,
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the PorousSimpleFoam solver.

        Executes the SIMPLE algorithm with porous media resistance in a
        time-stepping loop until convergence or ``endTime`` is reached.

        Returns:
            Final :class:`ConvergenceData`.
        """
        solver = self._build_solver_with_porosity()

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

        logger.info("Starting PorousSimpleFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  porous zones: %d", len(self.porous_zones))
        logger.info("  MRF zones: %d", len(self.mrf_zones))
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

            # Run one SIMPLE outer iteration with porous resistance
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
                logger.info("PorousSimpleFoam completed successfully (converged)")
            else:
                logger.warning("PorousSimpleFoam completed without full convergence")

        return last_convergence or ConvergenceData()


# ---------------------------------------------------------------------------
# Internal solver with porous resistance
# ---------------------------------------------------------------------------

class _PorousSIMPLESolver(SIMPLESolver):
    """SIMPLE solver with Darcy-Forchheimer porous resistance.

    Extends :class:`SIMPLESolver` to add porous media resistance and
    optional MRF source terms to the momentum equation.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    config : SIMPLEConfig
        Solver configuration.
    porous_data : list[dict]
        Porous zone data with cell masks and resistance coefficients.
    mrf_data : list[dict]
        MRF zone data with cell masks and rotation parameters.
    """

    def __init__(
        self,
        mesh: Any,
        config: SIMPLEConfig,
        *,
        porous_data: list[dict[str, Any]],
        mrf_data: list[dict[str, Any]],
    ) -> None:
        super().__init__(mesh, config)
        self._porous_data = porous_data
        self._mrf_data = mrf_data

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
        """Run the SIMPLE algorithm with porous resistance.

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
            # Step 1: Momentum predictor with porous resistance
            # ============================================
            U, A_p, H, mat_lower, mat_upper = self._momentum_predictor_porous(
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
                    "PorousSIMPLE iteration %d: U_res=%.6e, p_res=%.6e, "
                    "continuity=%.6e",
                    outer, U_residual, p_residual, continuity_error,
                )

            if continuity_error < tolerance and outer > 0:
                convergence.converged = True
                logger.info(
                    "PorousSIMPLE converged in %d iterations (continuity=%.6e)",
                    outer + 1, continuity_error,
                )
                break

            if torch.isnan(U).any() or torch.isnan(p).any():
                logger.error("PorousSIMPLE diverged at iteration %d (NaN)", outer + 1)
                break

        if not convergence.converged:
            logger.warning(
                "PorousSIMPLE did not converge in %d iterations (continuity=%.6e)",
                max_outer_iterations, continuity_error,
            )

        return U, p, phi, convergence

    def _momentum_predictor_porous(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        U_bc: torch.Tensor | None = None,
        nu_field: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the momentum equation with porous resistance and MRF forces.

        Adds the following to the standard momentum equation:
        - Darcy-Forchheimer resistance in porous cells
        - MRF Coriolis and centrifugal forces in MRF cells

        The Darcy-Forchheimer model:
            Cd = μ * D + ρ * |U| * F
            Diagonal: += V * tr(Cd)
            Source: -= V * ((Cd - I*tr(Cd)) · U)

        For isotropic resistance (d_x = d_y = d_z):
            Cd = (μ*d + |U|*f/2) * I
            tr(Cd) = 3 * (μ*d + |U|*f/2)
            Cd - I*tr(Cd) = 0 (no anisotropic source)

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
        # Source term: -grad(p)
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
        # Porous resistance: Darcy-Forchheimer model
        # S = -(μ d + |U| f/2) U
        # ============================================
        for zone in self._porous_data:
            mask = zone["cell_mask"]
            if not mask.any():
                continue

            d = zone["d"]  # (3,) Darcy coefficients
            f = zone["f"]  # (3,) Forchheimer coefficients

            # Get cell data for this zone
            U_zone = U[mask]  # (n_zone, 3)
            V_zone = cell_volumes_safe[mask]  # (n_zone,)

            # Effective viscosity for this zone
            if nu_field is not None:
                nu_zone = nu_field[mask]  # (n_zone,)
            else:
                nu_zone = torch.full((mask.sum(),), config.nu, dtype=dtype, device=device)

            # Velocity magnitude
            U_mag = U_zone.norm(dim=1)  # (n_zone,)

            # Resistance tensor components (per cell, per direction)
            # Cd_i = μ * d_i + |U| * f_i / 2
            # Using kinematic viscosity (μ = ρ*ν, but for incompressible
            # we work with ν directly since pressure is p/ρ)
            Cd = nu_zone.unsqueeze(-1) * d.unsqueeze(0) + \
                 U_mag.unsqueeze(-1) * f.unsqueeze(0) * 0.5  # (n_zone, 3)

            # Isotropic part: tr(Cd) = sum of diagonal
            isoCd = Cd.sum(dim=1)  # (n_zone,)

            # Diagonal contribution: += V * tr(Cd)
            diag[mask] = diag[mask] + V_zone * isoCd

            # Anisotropic source: -= V * (Cd - I*isoCd) · U
            # For diagonal resistance tensor: (Cd - I*isoCd)_ii = Cd_i - isoCd
            # This is zero for isotropic case (d_x = d_y = d_z, f_x = f_y = f_z)
            aniso = Cd - isoCd.unsqueeze(-1) * torch.ones_like(Cd)  # (n_zone, 3)
            aniso_source = V_zone.unsqueeze(-1) * aniso * U_zone  # (n_zone, 3)
            source[mask] = source[mask] - aniso_source

        # ============================================
        # MRF: Coriolis and centrifugal forces
        # ============================================
        for mrf in self._mrf_data:
            mask = mrf["cell_mask"]
            if not mask.any():
                continue

            omega_vec = mrf["omega_vec"]
            omega_mag = mrf["omega_mag"]
            centrifugal = mrf["centrifugal"]

            # Add centrifugal force (explicit, position-dependent)
            source[mask] = source[mask] + centrifugal

            # Add Coriolis force (semi-implicit)
            # Explicit part: -2ω × U
            U_zone = U[mask]
            omega_cross_U = torch.zeros_like(U_zone)
            omega_cross_U[:, 0] = omega_vec[1] * U_zone[:, 2] - omega_vec[2] * U_zone[:, 1]
            omega_cross_U[:, 1] = omega_vec[2] * U_zone[:, 0] - omega_vec[0] * U_zone[:, 2]
            omega_cross_U[:, 2] = omega_vec[0] * U_zone[:, 1] - omega_vec[1] * U_zone[:, 0]
            coriolis_source = -2.0 * omega_cross_U

            source[mask] = source[mask] + coriolis_source

            # Implicit part: +2|ω| on diagonal
            diag[mask] = diag[mask] + 2.0 * omega_mag

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
