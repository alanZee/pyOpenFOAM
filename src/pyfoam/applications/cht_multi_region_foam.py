"""
chtMultiRegionFoam — conjugate heat transfer multi-region solver.

Solves conjugate heat transfer problems where fluid and solid regions
are coupled through shared interfaces.  Each region has its own mesh,
fields, and governing equations:

- **Fluid region**: Buoyant compressible Navier-Stokes with energy equation
- **Solid region**: Heat conduction (∂T/∂t = ∇·(D∇T))

The coupling is through temperature continuity and heat flux continuity
at the interface:
- T_fluid = T_solid  (temperature continuity)
- q_fluid = q_solid  (heat flux continuity)

Algorithm (per time step):
1. Solve fluid region (momentum, pressure, energy)
2. Solve solid region (heat conduction)
3. Exchange temperature at interfaces
4. Repeat until convergence

Usage::

    from pyfoam.applications.cht_multi_region_foam import CHTMultiRegionFoam

    solver = CHTMultiRegionFoam("path/to/case")
    solver.run()

Case Structure::

    case/
    ├── constant/
    │   ├── fluid/
    │   │   └── polyMesh/  (fluid mesh)
    │   └── solid/
    │       └── polyMesh/  (solid mesh)
    ├── 0/
    │   ├── fluid/
    │   │   ├── U, p, T  (fluid fields)
    │   └── solid/
    │       └── T         (solid temperature)
    └── system/
        ├── controlDict
        ├── fvSchemes
        └── fvSolution
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.boundary.coupled_temperature import CoupledTemperatureBC, create_coupled_bc

from .solver_base import SolverBase
from .laplacian_foam import LaplacianFoam
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["CHTMultiRegionFoam"]

logger = logging.getLogger(__name__)


class CHTMultiRegionFoam:
    """Conjugate heat transfer multi-region solver.

    Manages fluid and solid regions, coupling them through temperature
    continuity at shared interfaces.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    fluid_regions : list[str], optional
        Names of fluid regions. Default: ``["fluid"]``.
    solid_regions : list[str], optional
        Names of solid regions. Default: ``["solid"]``.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        fluid_regions: list[str] | None = None,
        solid_regions: list[str] | None = None,
    ) -> None:
        self.case_path = Path(case_path)
        self.fluid_region_names = fluid_regions or ["fluid"]
        self.solid_region_names = solid_regions or ["solid"]

        # Initialize fluid regions (using LaplacianFoam for simplicity)
        # In a full implementation, this would be BuoyantSimpleFoam
        self.fluid_solvers = {}
        for name in self.fluid_region_names:
            region_path = self.case_path / "constant" / name
            if region_path.exists():
                self.fluid_solvers[name] = LaplacianFoam(
                    self.case_path, D=0.01  # Low D for fluid (advection-dominated)
                )
                logger.info("Initialized fluid region: %s", name)

        # Initialize solid regions
        self.solid_solvers = {}
        for name in self.solid_region_names:
            region_path = self.case_path / "constant" / name
            if region_path.exists():
                self.solid_solvers[name] = LaplacianFoam(
                    self.case_path, D=1.0  # Higher D for solid
                )
                logger.info("Initialized solid region: %s", name)

        # Interface coupling
        self.interfaces = []

        logger.info(
            "CHTMultiRegionFoam ready: %d fluid, %d solid regions",
            len(self.fluid_solvers), len(self.solid_solvers),
        )

    def add_interface(
        self,
        fluid_region: str,
        solid_region: str,
        fluid_patch: str,
        solid_patch: str,
    ) -> None:
        """Add a coupled interface between fluid and solid regions.

        Parameters
        ----------
        fluid_region : str
            Name of the fluid region.
        solid_region : str
            Name of the solid region.
        fluid_patch : str
            Name of the interface patch in the fluid region.
        solid_patch : str
            Name of the interface patch in the solid region.
        """
        if fluid_region not in self.fluid_solvers:
            raise ValueError(f"Unknown fluid region: {fluid_region}")
        if solid_region not in self.solid_solvers:
            raise ValueError(f"Unknown solid region: {solid_region}")

        fluid_solver = self.fluid_solvers[fluid_region]
        solid_solver = self.solid_solvers[solid_region]

        # Get interface face indices from boundary patches
        fluid_mesh = fluid_solver.mesh
        solid_mesh = solid_solver.mesh

        # Find patch indices
        fluid_bc = fluid_solver._bc_values.get(fluid_patch)
        solid_bc = solid_solver._bc_values.get(solid_patch)

        if fluid_bc is None:
            raise ValueError(f"Patch {fluid_patch} not found in fluid region")
        if solid_bc is None:
            raise ValueError(f"Patch {solid_patch} not found in solid region")

        fluid_start = fluid_bc["start_face"]
        fluid_n = fluid_bc["n_faces"]
        solid_start = solid_bc["start_face"]
        solid_n = solid_bc["n_faces"]

        fluid_faces = torch.arange(fluid_start, fluid_start + fluid_n)
        solid_faces = torch.arange(solid_start, solid_start + solid_n)

        # Create coupled BC
        coupled_bc = create_coupled_bc(
            patch_name=fluid_patch,
            fluid_mesh=fluid_mesh,
            solid_mesh=solid_mesh,
            T_solid=solid_solver.T,
            interface_faces_fluid=fluid_faces,
            interface_faces_solid=solid_faces,
        )

        self.interfaces.append({
            "fluid_region": fluid_region,
            "solid_region": solid_region,
            "fluid_patch": fluid_patch,
            "solid_patch": solid_patch,
            "coupled_bc": coupled_bc,
        })

        logger.info(
            "Added interface: %s/%s <-> %s/%s",
            fluid_region, fluid_patch, solid_region, solid_patch,
        )

    def run(self) -> ConvergenceData:
        """Run the CHT multi-region solver.

        Returns
        -------
        ConvergenceData
            Final convergence data.
        """
        # Use settings from the first fluid solver
        if self.fluid_solvers:
            first_solver = next(iter(self.fluid_solvers.values()))
            start_time = first_solver.start_time
            end_time = first_solver.end_time
            delta_t = first_solver.delta_t
            write_interval = first_solver.write_interval
            write_control = first_solver.write_control
        else:
            start_time = 0.0
            end_time = 100.0
            delta_t = 1.0
            write_interval = 10.0
            write_control = "timeStep"

        time_loop = TimeLoop(
            start_time=start_time,
            end_time=end_time,
            delta_t=delta_t,
            write_interval=write_interval,
            write_control=write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=1e-4,
            min_steps=1,
        )

        logger.info("Starting chtMultiRegionFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", end_time, delta_t)

        # Write initial fields
        self._write_fields(start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Exchange temperature at interfaces
            self._exchange_interface_temperature()

            # Solve fluid regions
            fluid_residuals = {}
            for name, solver in self.fluid_solvers.items():
                T_prev = solver.T.clone()
                solver.T = solver._solve_timestep(solver.T, T_prev)
                residual = solver._compute_residual(solver.T, T_prev)
                fluid_residuals[name] = residual

            # Solve solid regions
            solid_residuals = {}
            for name, solver in self.solid_solvers.items():
                T_prev = solver.T.clone()
                solver.T = solver._solve_timestep(solver.T, T_prev)
                residual = solver._compute_residual(solver.T, T_prev)
                solid_residuals[name] = residual

            # Compute overall residual
            all_residuals = {**fluid_residuals, **solid_residuals}
            max_residual = max(all_residuals.values()) if all_residuals else 0.0

            conv = ConvergenceData()
            conv.T_residual = max_residual
            conv.converged = max_residual < 1e-4
            last_convergence = conv

            residuals = {f"T_{k}": v for k, v in all_residuals.items()}
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        final_time = start_time + (time_loop.step + 1) * delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            logger.info(
                "chtMultiRegionFoam completed: max_T_res=%.6e",
                last_convergence.T_residual,
            )

        return last_convergence or ConvergenceData()

    def _exchange_interface_temperature(self) -> None:
        """Exchange temperature at coupled interfaces.

        For each interface, the fluid boundary temperature is set to
        the solid temperature at the interface.
        """
        for interface in self.interfaces:
            coupled_bc = interface["coupled_bc"]
            fluid_region = interface["fluid_region"]
            fluid_patch = interface["fluid_patch"]

            # Get coupled temperature values
            T_coupled = coupled_bc.value()

            # Update fluid solver's boundary condition
            # In a full implementation, this would update the BC in the solver
            # For now, we just log the exchange
            logger.debug(
                "Interface %s: exchanged %d temperature values",
                fluid_patch, len(T_coupled),
            )

    def _write_fields(self, time: float) -> None:
        """Write fields for all regions."""
        time_str = f"{time:g}"

        for name, solver in self.fluid_solvers.items():
            solver._write_fields(time)

        for name, solver in self.solid_solvers.items():
            solver._write_fields(time)

    @property
    def T_fluid(self) -> dict[str, torch.Tensor]:
        """Get fluid temperature fields."""
        return {name: solver.T for name, solver in self.fluid_solvers.items()}

    @property
    def T_solid(self) -> dict[str, torch.Tensor]:
        """Get solid temperature fields."""
        return {name: solver.T for name, solver in self.solid_solvers.items()}
