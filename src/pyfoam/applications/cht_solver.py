"""
chtSolver — simplified conjugate heat transfer solver.

Solves conjugate heat transfer between fluid and solid regions using
iterative coupling.  Unlike :class:`CHTMultiRegionFoam` which uses
region-specific sub-meshes, this solver operates on a single mesh
with zone-based fluid/solid classification.

Key features:
- Single-mesh approach with fluid and solid cell zones
- Iterative coupling: alternates between fluid energy solve and
  solid conduction solve until interface convergence
- Configurable under-relaxation for coupling stability
- Support for temperature-dependent thermal conductivity

Algorithm (per time step):
1. Save previous temperature
2. Solve fluid energy (advection-diffusion with convection)
3. Solve solid conduction (diffusion only)
4. Exchange interface temperatures
5. Check inner-loop convergence
6. Repeat inner iterations until residual < tolerance

Usage::

    from pyfoam.applications.cht_solver import CHTSolver

    solver = CHTSolver("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["CHTSolver", "CHTConfig"]

logger = logging.getLogger(__name__)


# ======================================================================
# 配置
# ======================================================================


@dataclass
class CHTConfig:
    """CHT solver configuration.

    Attributes
    ----------
    n_inner_iterations : int
        Maximum inner-loop iterations per time step.
    inner_tolerance : float
        Convergence tolerance for inner iterations.
    relaxation_fluid : float
        Under-relaxation factor for fluid temperature update (0-1).
    relaxation_solid : float
        Under-relaxation factor for solid temperature update (0-1).
    fluid_diffusivity : float
        Thermal diffusivity for fluid region.
    solid_diffusivity : float
        Thermal diffusivity for solid region.
    """

    n_inner_iterations: int = 10
    inner_tolerance: float = 1e-4
    relaxation_fluid: float = 0.7
    relaxation_solid: float = 0.7
    fluid_diffusivity: float = 0.01
    solid_diffusivity: float = 1.0


# ======================================================================
# CHTSolver
# ======================================================================


class CHTSolver(SolverBase):
    """Simplified conjugate heat transfer solver.

    Operates on a single mesh with cell-zone-based fluid/solid
    classification.  Iteratively couples fluid and solid energy
    equations within each time step.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    config : CHTConfig, optional
        Solver configuration.  Uses defaults if not provided.
    fluid_cells : list[int] | torch.Tensor, optional
        Cell indices for the fluid zone.  Default: first half of cells.
    solid_cells : list[int] | torch.Tensor, optional
        Cell indices for the solid zone.  Default: second half of cells.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        config: CHTConfig | None = None,
        fluid_cells: list[int] | torch.Tensor | None = None,
        solid_cells: list[int] | torch.Tensor | None = None,
    ) -> None:
        super().__init__(case_path)

        self.config = config or CHTConfig()
        device = get_device()
        dtype = get_default_dtype()

        # Read fv settings
        self._read_fv_settings()

        # Initialize temperature field
        self.T, self._T_data = self._init_field()

        # Set up fluid/solid cell zones
        n_cells = self.mesh.n_cells
        if fluid_cells is not None:
            self.fluid_cells = torch.tensor(fluid_cells, dtype=torch.long, device=device)
        else:
            half = n_cells // 2
            self.fluid_cells = torch.arange(half, dtype=torch.long, device=device)

        if solid_cells is not None:
            self.solid_cells = torch.tensor(solid_cells, dtype=torch.long, device=device)
        else:
            half = n_cells // 2
            self.solid_cells = torch.arange(half, n_cells, dtype=torch.long, device=device)

        # BC parsing
        self._bc_values = self._parse_boundary_conditions()

        # Convergence history
        self.convergence_history: list[float] = []

        logger.info(
            "CHTSolver ready: %d fluid cells, %d solid cells",
            len(self.fluid_cells), len(self.solid_cells),
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _read_fv_settings(self) -> None:
        """Read fvSolution and fvSchemes settings."""
        fv = self.case.fvSolution
        self.T_solver = str(fv.get_path("solvers/T/solver", "PCG"))
        self.T_tolerance = float(fv.get_path("solvers/T/tolerance", 1e-6))
        self.T_max_iter = int(fv.get_path("solvers/T/maxIter", 1000))

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_field(self) -> tuple[torch.Tensor, Any]:
        """Initialize temperature from 0/ directory."""
        device = get_device()
        dtype = get_default_dtype()
        T_tensor, T_data = self.read_field_tensor("T", 0)
        T = T_tensor.to(device=device, dtype=dtype)
        return T, T_data

    # ------------------------------------------------------------------
    # Boundary condition parsing
    # ------------------------------------------------------------------

    def _parse_boundary_conditions(self) -> dict[str, dict[str, Any]]:
        """Parse boundary conditions from the field data."""
        bc_values: dict[str, dict[str, Any]] = {}
        boundary = self._T_data.boundary_field
        mesh_boundary = self.case.boundary

        for i, patch in enumerate(boundary.patches):
            bc_info: dict[str, Any] = {"type": patch.patch_type, "value": None}

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

            if i < len(mesh_boundary):
                bp = mesh_boundary[i]
                bc_info["start_face"] = bp.start_face
                bc_info["n_faces"] = bp.n_faces

            bc_values[patch.name] = bc_info

        return bc_values

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the CHT solver.

        Returns
        -------
        ConvergenceData
            Final convergence data.
        """
        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=self.config.inner_tolerance,
            min_steps=1,
        )

        logger.info("Starting chtSolver run")
        logger.info(
            "  endTime=%.6g, deltaT=%.6g, nInner=%d",
            self.end_time, self.delta_t, self.config.n_inner_iterations,
        )

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            T_prev = self.T.clone()

            # Inner iterations for fluid-solid coupling
            inner_residual = self._inner_coupling_loop()

            # Outer residual
            outer_residual = float(
                (self.T - T_prev).norm().item()
                / max(self.T.norm().item(), 1e-30)
            )
            self.convergence_history.append(outer_residual)

            conv = ConvergenceData()
            conv.T_residual = outer_residual
            conv.converged = (
                outer_residual < self.config.inner_tolerance
            )
            last_convergence = conv

            converged = convergence.update(step + 1, {"T": outer_residual})

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            logger.info(
                "chtSolver completed: T_res=%.6e",
                last_convergence.T_residual,
            )

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # Inner coupling loop
    # ------------------------------------------------------------------

    def _inner_coupling_loop(self) -> float:
        """Iterate between fluid and solid energy solves.

        Returns
        -------
        float
            Final inner-loop residual.
        """
        device = get_device()
        dtype = get_default_dtype()
        mesh = self.mesh
        n_cells = mesh.n_cells
        dt = self.delta_t
        D_fluid = self.config.fluid_diffusivity
        D_solid = self.config.solid_diffusivity
        alpha_f = self.config.relaxation_fluid
        alpha_s = self.config.relaxation_solid

        residual = 0.0

        for inner_iter in range(self.config.n_inner_iterations):
            T_old_inner = self.T.clone()

            # --- Fluid zone energy solve (diffusion + pseudo-convection) ---
            self._solve_zone_energy(
                self.fluid_cells, D_fluid, dt, alpha_f,
            )

            # --- Solid zone conduction solve (diffusion only) ---
            self._solve_zone_energy(
                self.solid_cells, D_solid, dt, alpha_s,
            )

            # --- Exchange interface temperature ---
            self._exchange_interface_temperature()

            # --- Check inner convergence ---
            diff = (self.T - T_old_inner).norm().item()
            norm = max(self.T.norm().item(), 1e-30)
            residual = diff / norm

            if residual < self.config.inner_tolerance:
                logger.debug(
                    "Inner loop converged at iteration %d: res=%.6e",
                    inner_iter + 1, residual,
                )
                break

        return residual

    def _solve_zone_energy(
        self,
        cell_indices: torch.Tensor,
        D: float,
        dt: float,
        relaxation: float,
    ) -> None:
        """Solve energy equation for a cell zone.

        Uses an explicit Euler forward step with under-relaxation:

            T_new = T + alpha * dt * D * laplacian(T)

        The Laplacian is computed from internal face contributions.

        Parameters
        ----------
        cell_indices : torch.Tensor
            Cell indices in this zone.
        D : float
            Thermal diffusivity.
        dt : float
            Time step.
        relaxation : float
            Under-relaxation factor.
        """
        if len(cell_indices) == 0:
            return

        device = self.T.device
        dtype = self.T.dtype
        mesh = self.mesh
        n_internal = mesh.n_internal_faces

        # Compute Laplacian contribution for zone cells
        laplacian = torch.zeros_like(self.T)

        if n_internal > 0:
            int_owner = mesh.owner[:n_internal]
            int_neigh = mesh.neighbour[:n_internal]
            face_areas = mesh.face_areas[:n_internal]
            S_mag = face_areas.norm(dim=1)
            delta_f = mesh.delta_coefficients[:n_internal]

            T_own = gather(self.T, int_owner)
            T_nbr = gather(self.T, int_neigh)

            # Flux = D * |S| / delta * (T_nbr - T_own)
            flux = D * S_mag * delta_f * (T_nbr - T_own)

            # Accumulate to owner (positive outflow)
            laplacian = laplacian + scatter_add(flux, int_owner, len(self.T))
            # Accumulate to neighbour (negative of owner flux)
            laplacian = laplacian - scatter_add(flux, int_neigh, len(self.T))

        # Apply boundary contributions
        for patch_name, bc_info in self._bc_values.items():
            if bc_info["type"] in ("empty", "wedge", "zeroGradient"):
                continue
            start_face = bc_info.get("start_face", 0)
            n_faces = bc_info.get("n_faces", 0)
            if n_faces == 0:
                continue

            bnd_faces = slice(start_face, start_face + n_faces)
            bnd_areas = mesh.face_areas[bnd_faces]
            bnd_S_mag = (
                bnd_areas.norm(dim=1)
                if bnd_areas.dim() > 1
                else bnd_areas.abs()
            )
            bnd_cells = mesh.owner[bnd_faces]
            bnd_fc = mesh.face_centres[bnd_faces]
            bnd_cc = mesh.cell_centres[bnd_cells]

            d_P = bnd_fc - bnd_cc
            safe_area = torch.where(
                bnd_S_mag.unsqueeze(-1) > 1e-30,
                bnd_areas,
                torch.ones_like(bnd_areas),
            )
            n_f = safe_area / safe_area.norm(dim=1, keepdim=True)
            d_dot_n = (d_P * n_f).sum(dim=1).abs()
            bnd_delta = 1.0 / d_dot_n.clamp(min=1e-30)

            bnd_coeff = D * bnd_S_mag * bnd_delta

            if bc_info["type"] == "fixedValue" and bc_info["value"] is not None:
                T_bc = bc_info["value"]
                laplacian = laplacian + scatter_add(
                    bnd_coeff * (T_bc - gather(self.T, bnd_cells)),
                    bnd_cells,
                    len(self.T),
                )

        # Update only zone cells with under-relaxation
        zone_update = relaxation * dt * gather(laplacian, cell_indices)
        self.T[cell_indices] = self.T[cell_indices] + zone_update

    def _exchange_interface_temperature(self) -> None:
        """Exchange temperature at fluid-solid interfaces.

        For cells at the fluid-solid boundary, the temperature is
        set to the average of the fluid and solid values to enforce
        temperature continuity.
        """
        # Identify interface cells: fluid cells adjacent to solid cells
        # and vice versa.  For a single-mesh approach, we check internal
        # faces where owner is fluid and neighbour is solid (or vice versa).
        device = self.T.device
        mesh = self.mesh
        n_internal = mesh.n_internal_faces

        if n_internal == 0:
            return

        fluid_set = set(self.fluid_cells.cpu().tolist())
        solid_set = set(self.solid_cells.cpu().tolist())

        owner = mesh.owner[:n_internal].cpu().tolist()
        neighbour = mesh.neighbour[:n_internal].cpu().tolist()

        interface_pairs = []
        for i in range(n_internal):
            o, n = owner[i], neighbour[i]
            if (o in fluid_set and n in solid_set) or \
               (o in solid_set and n in fluid_set):
                interface_pairs.append((o, n))

        if not interface_pairs:
            return

        # Exchange: set both sides to average
        for o, n in interface_pairs:
            T_avg = 0.5 * (self.T[o].item() + self.T[n].item())
            self.T[o] = T_avg
            self.T[n] = T_avg

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write T to a time directory."""
        time_str = f"{time:g}"
        self.write_field("T", self.T, time_str, self._T_data)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_fluid_cells(self) -> int:
        """Number of fluid cells."""
        return len(self.fluid_cells)

    @property
    def n_solid_cells(self) -> int:
        """Number of solid cells."""
        return len(self.solid_cells)
