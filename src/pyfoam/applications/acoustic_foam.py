"""
acousticFoam — acoustic wave propagation solver.

Solves the linearized Euler equations for acoustic perturbations about
a uniform mean flow.  The governing equations are:

    ∂p'/∂t + U₀·∇p' + ρ₀ c² ∇·u' = 0
    ∂u'/∂t + U₀·∇u' + (1/ρ₀) ∇p' = 0

where:
    p'  — pressure perturbation
    u'  — velocity perturbation vector
    U₀  — mean flow velocity (uniform)
    ρ₀  — mean density (uniform)
    c   — speed of sound

Uses explicit Euler time-stepping with CFL-limited Δt.
Boundary conditions:
    - fixedValue: prescribed perturbation (wall / source)
    - advective: non-reflecting outflow (∂φ/∂t + c_n ∂φ/∂n = 0)
    - zeroGradient: symmetry / reflective

Usage::

    from pyfoam.applications.acoustic_foam import AcousticFoam

    solver = AcousticFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["AcousticFoam"]

logger = logging.getLogger(__name__)


class AcousticFoam(SolverBase):
    """Acoustic wave propagation solver.

    Solves the linearized Euler equations for acoustic perturbations
    using explicit time-stepping.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    c0 : float, optional
        Speed of sound (m/s).  Reads from ``constant/acousticProperties``
        if None.  Default 343.0.
    U0 : tuple[float, float, float], optional
        Uniform mean flow velocity (m/s).  Default (0, 0, 0).
    rho0 : float, optional
        Mean density (kg/m³).  Default 1.225.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        c0: float | None = None,
        U0: tuple[float, float, float] | None = None,
        rho0: float | None = None,
    ) -> None:
        super().__init__(case_path)

        self._read_fv_solution_settings()
        self._read_fv_schemes_settings()

        # Acoustic properties
        props = self._read_acoustic_properties()
        self.c0 = c0 if c0 is not None else props.get("c0", 343.0)
        self.U0 = torch.tensor(
            U0 if U0 is not None else props.get("U0", (0.0, 0.0, 0.0)),
            dtype=get_default_dtype(),
            device=get_device(),
        )
        self.rho0 = rho0 if rho0 is not None else props.get("rho0", 1.225)

        # Initialise perturbation fields
        self.p_prime, self._p_data = self._init_scalar_field("p'")
        self.u_prime, self._u_data = self._init_vector_field("u'")

        # Parse boundary conditions
        self._p_bcs = self._parse_bcs(self._p_data)
        self._u_bcs = self._parse_bcs(self._u_data)

        logger.info(
            "AcousticFoam ready: c0=%.1f, U0=%s, rho0=%.4f",
            self.c0, tuple(self.U0.tolist()), self.rho0,
        )

    # ------------------------------------------------------------------
    # Settings reading
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        """Read solver settings from fvSolution."""
        fv = self.case.fvSolution
        self.max_iter = int(fv.get_path("solvers/p'/maxIter", 1000))

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings."""
        fs = self.case.fvSchemes
        self.grad_scheme = str(fs.get_path("gradSchemes/default", "Gauss linear"))

    def _read_acoustic_properties(self) -> dict[str, Any]:
        """Read acoustic properties from constant/acousticProperties."""
        props_path = self.case_path / "constant" / "acousticProperties"
        if not props_path.exists():
            return {}

        try:
            from pyfoam.io.dictionary import parse_dict_file
            return parse_dict_file(props_path)
        except Exception as e:
            logger.warning("Could not parse acousticProperties: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_scalar_field(
        self, name: str,
    ) -> tuple[torch.Tensor, Any]:
        """Initialise a scalar perturbation field."""
        device = get_device()
        dtype = get_default_dtype()
        tensor, data = self.read_field_tensor(name, 0)
        return tensor.to(device=device, dtype=dtype), data

    def _init_vector_field(
        self, name: str,
    ) -> tuple[torch.Tensor, Any]:
        """Initialise a vector perturbation field."""
        device = get_device()
        dtype = get_default_dtype()
        tensor, data = self.read_field_tensor(name, 0)
        return tensor.to(device=device, dtype=dtype), data

    # ------------------------------------------------------------------
    # Boundary condition parsing
    # ------------------------------------------------------------------

    def _parse_bcs(self, field_data: Any) -> dict[str, dict[str, Any]]:
        """Parse boundary conditions from field data."""
        bcs: dict[str, dict[str, Any]] = {}
        mesh_boundary = self.case.boundary

        for i, patch in enumerate(field_data.boundary_field.patches):
            bc_info: dict[str, Any] = {
                "type": patch.patch_type,
                "value": None,
            }

            # Parse fixed value
            if patch.patch_type == "fixedValue" and patch.value is not None:
                val = patch.value
                if isinstance(val, str):
                    val = val.strip()
                    if val.startswith("uniform"):
                        val = val[len("uniform"):].strip()
                    try:
                        bc_info["value"] = float(val)
                    except ValueError:
                        pass
                elif isinstance(val, (int, float)):
                    bc_info["value"] = float(val)

            # Get face info from mesh boundary
            if i < len(mesh_boundary):
                bp = mesh_boundary[i]
                bc_info["start_face"] = bp.start_face
                bc_info["n_faces"] = bp.n_faces

            bcs[patch.name] = bc_info

        return bcs

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the acousticFoam solver.

        Uses explicit Euler time-stepping with CFL-limited Δt.

        Returns:
            Final :class:`ConvergenceData`.
        """
        # Limit Δt for explicit stability (CFL < 1)
        c_max = float(self.c0 + self.U0.norm().item())
        max_delta_t = self._compute_max_delta_t(c_max)
        actual_dt = min(self.delta_t, max_delta_t)

        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=actual_dt,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(tolerance=1e-10, min_steps=1)

        logger.info("Starting acousticFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g (limited=%.6g)",
                     self.end_time, self.delta_t, actual_dt)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            p_old = self.p_prime.clone()
            u_old = self.u_prime.clone()

            # Explicit Euler: advance one time step
            self.p_prime, self.u_prime = self._advance(
                self.p_prime, self.u_prime, actual_dt,
            )

            # Apply boundary conditions
            self.p_prime = self._apply_scalar_bcs(self.p_prime, self._p_bcs)
            self.u_prime = self._apply_vector_bcs(self.u_prime, self._u_bcs)

            # Compute residual
            p_res = self._compute_residual(self.p_prime, p_old)
            u_res = self._compute_residual(self.u_prime, u_old)
            residual = max(p_res, u_res)

            conv = ConvergenceData()
            conv.T_residual = p_res
            conv.U_residual = u_res
            conv.converged = residual < 1e-10
            last_convergence = conv

            residuals = {"p'": p_res, "u'": u_res}
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + actual_dt)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * actual_dt
        self._write_fields(final_time)

        if last_convergence is not None:
            logger.info(
                "acousticFoam completed: p_res=%.6e, u_res=%.6e",
                last_convergence.T_residual, last_convergence.U_residual,
            )

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # Explicit time advancement
    # ------------------------------------------------------------------

    def _advance(
        self,
        p: torch.Tensor,
        u: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one explicit Euler time step.

        Equations:
            ∂p'/∂t = -U₀·∇p' - ρ₀ c² ∇·u'
            ∂u'/∂t = -U₀·∇u' - (1/ρ₀) ∇p'

        Discretised with central differences for spatial gradients.
        """
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = get_device()
        dtype = get_default_dtype()

        V = mesh.cell_volumes
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # ---- Advection: U₀·∇φ (central difference) ----
        dp_adv = torch.zeros(n_cells, dtype=dtype, device=device)
        du_adv = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        if self.U0.norm() > 1e-10:
            # Compute face values (linear interpolation)
            p_O = gather(p, int_owner)
            p_N = gather(p, int_neigh)
            p_face = 0.5 * (p_O + p_N)

            u_O = u[int_owner]  # (n_int, 3)
            u_N = u[int_neigh]
            u_face = 0.5 * (u_O + u_N)

            # Upwind scheme for advection
            # Face mass flux: φ_f = U₀ · S_f
            S_f = mesh.face_areas[:n_internal]  # (n_int, 3) or (n_int,)
            if S_f.dim() == 1:
                S_f = torch.zeros(n_internal, 3, dtype=dtype, device=device)

            phi_f = (self.U0.unsqueeze(0) * S_f).sum(dim=1)  # (n_int,)

            # Upwind selection: if phi_f > 0, use owner; else use neighbour
            upwind_mask = phi_f > 0
            p_upwind = torch.where(upwind_mask, p_O, p_N)
            u_upwind = torch.where(
                upwind_mask.unsqueeze(-1),
                u_O,
                u_N,
            )

            # Flux = phi_f * upwind_value
            p_flux = phi_f * p_upwind
            u_flux = phi_f.unsqueeze(-1) * u_upwind  # (n_int, 3)

            # Scatter flux divergence to cells
            # For owner: +flux/V, for neighbour: -flux/V
            dp_adv = dp_adv - scatter_add(p_flux / gather(V, int_owner), int_owner, n_cells)
            dp_adv = dp_adv + scatter_add(p_flux / gather(V, int_neigh), int_neigh, n_cells)

            for dim in range(3):
                du_adv[:, dim] = du_adv[:, dim] - scatter_add(
                    u_flux[:, dim] / gather(V, int_owner), int_owner, n_cells,
                )
                du_adv[:, dim] = du_adv[:, dim] + scatter_add(
                    u_flux[:, dim] / gather(V, int_neigh), int_neigh, n_cells,
                )

        # ---- Pressure gradient: (1/ρ₀) ∇p' ----
        # Central difference approximation
        p_O = gather(p, int_owner)
        p_N = gather(p, int_neigh)
        dp = p_N - p_O  # (n_int,)

        S_f = mesh.face_areas[:n_internal]
        if S_f.dim() == 1:
            S_f_vec = torch.zeros(n_internal, 3, dtype=dtype, device=device)
        else:
            S_f_vec = S_f

        # grad_p contribution: dp * S_f / V
        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        for dim in range(3):
            flux_grad = dp * S_f_vec[:, dim]
            grad_p[:, dim] = scatter_add(
                flux_grad / gather(V, int_owner), int_owner, n_cells,
            ) - scatter_add(
                flux_grad / gather(V, int_neigh), int_neigh, n_cells,
            )

        # ---- Velocity divergence: ρ₀ c² ∇·u' ----
        div_u = torch.zeros(n_cells, dtype=dtype, device=device)
        u_O = u[int_owner]
        u_N = u[int_neigh]
        for dim in range(3):
            du = (u_N[:, dim] - u_O[:, dim]) * S_f_vec[:, dim]
            div_u = div_u + scatter_add(
                du / gather(V, int_owner), int_owner, n_cells,
            ) - scatter_add(
                du / gather(V, int_neigh), int_neigh, n_cells,
            )

        # ---- Update fields ----
        dp_dt = -dp_adv - self.rho0 * self.c0**2 * div_u
        du_dt = -du_adv - grad_p / self.rho0

        p_new = p + dt * dp_dt
        u_new = u + dt * du_dt

        return p_new, u_new

    # ------------------------------------------------------------------
    # Boundary condition application
    # ------------------------------------------------------------------

    def _apply_scalar_bcs(
        self,
        field: torch.Tensor,
        bcs: dict[str, dict[str, Any]],
    ) -> torch.Tensor:
        """Apply scalar boundary conditions to owner cells of boundary faces."""
        mesh = self.mesh
        for patch_name, bc_info in bcs.items():
            start = bc_info.get("start_face", 0)
            n = bc_info.get("n_faces", 0)
            if n == 0:
                continue

            bc_type = bc_info["type"]
            owners = mesh.owner[start:start + n]

            if bc_type == "fixedValue" and bc_info["value"] is not None:
                field[owners] = bc_info["value"]
            elif bc_type == "advective":
                # Non-reflecting: owner cells keep their interior values
                pass
            # zeroGradient: no action needed (already extrapolated)

        return field

    def _apply_vector_bcs(
        self,
        field: torch.Tensor,
        bcs: dict[str, dict[str, Any]],
    ) -> torch.Tensor:
        """Apply vector boundary conditions to owner cells of boundary faces."""
        mesh = self.mesh
        for patch_name, bc_info in bcs.items():
            start = bc_info.get("start_face", 0)
            n = bc_info.get("n_faces", 0)
            if n == 0:
                continue

            bc_type = bc_info["type"]
            owners = mesh.owner[start:start + n]

            if bc_type == "fixedValue" and bc_info["value"] is not None:
                field[owners] = bc_info["value"]
            elif bc_type == "advective":
                # Non-reflecting: owner cells keep their interior values
                pass
            # zeroGradient: no action

        return field

    # ------------------------------------------------------------------
    # Stability and utilities
    # ------------------------------------------------------------------

    def _compute_max_delta_t(self, c_max: float) -> float:
        """Compute maximum stable Δt from CFL condition.

        For explicit Euler on an unstructured mesh, CFL = c_max * Δt / Δx < 1.
        """
        mesh = self.mesh
        cell_volumes = mesh.cell_volumes
        # Estimate characteristic cell size from volume
        delta_x = cell_volumes.pow(1.0 / 3.0).min().item()
        if c_max > 1e-10:
            return 0.5 * delta_x / c_max  # CFL = 0.5 for safety
        return self.delta_t

    @staticmethod
    def _compute_residual(
        field: torch.Tensor,
        field_old: torch.Tensor,
    ) -> float:
        """Compute L∞ residual normalised by field magnitude."""
        diff = field - field_old
        max_diff = float(diff.abs().max().item())
        max_field = float(field.abs().max().item())
        if max_field > 1e-30:
            return max_diff / max_field
        return max_diff

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write acoustic perturbation fields."""
        time_str = f"{time:g}"
        self.write_field("p'", self.p_prime, time_str, self._p_data)
        self.write_field("u'", self.u_prime, time_str, self._u_data)
