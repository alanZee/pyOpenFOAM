"""
ParticleTracks — particle track generation for Lagrangian visualization.

Generates particle trajectory data by advecting massless tracer
particles through a velocity field, producing VTK-compatible track
data for visualization in ParaView.

Physics
-------
Particles are advected using the velocity field:

    dx/dt = U(x, t)

Integration methods:

- ``RK1``: Forward Euler (1st order)
- ``RK2``: Midpoint method (2nd order)
- ``RK4``: Classic Runge-Kutta (4th order)

The particle position is updated as:

    x^{n+1} = x^n + Δt * U(x^n)

(RK1) or with the appropriate Runge-Kutta weights.

References
----------
- OpenFOAM ``particleTracks`` function object source
- OpenFOAM ``streamlines`` function object source
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["ParticleTracks"]

logger = logging.getLogger(__name__)


class ParticleTracks(FunctionObject):
    """Generate particle track data by advecting tracer particles.

    Releases particles from specified seed points and advects them
    through the velocity field using Runge-Kutta time integration.

    Configuration keys:

    - ``nParticleTracks``: number of particles to release (default: 100)
    - ``seedPoints``: explicit list of ``(x, y, z)`` seed locations
      (overrides nParticleTracks if provided)
    - ``trackLength``: maximum number of time steps per track (default: 1000)
    - ``integrationScheme``: ``"RK1"``, ``"RK2"``, or ``"RK4"`` (default: ``"RK4"``)
    - ``dt``: particle advection time step (default: uses simulation deltaT)
    - ``fields``: velocity field name (default: ``"U"``)
    - ``lifeTime``: maximum particle lifetime in time units (default: inf)
    - ``cloudName``: name for the particle cloud (default: ``"particleTracks"``)

    Example controlDict entry::

        particleTracks1
        {
            type            particleTracks;
            libs            ("liblagrangian.so");
            nParticleTracks 50;
            trackLength     500;
            integrationScheme RK4;
            cloudName       trackCloud;
        }
    """

    SCHEMES = {"RK1", "RK2", "RK4"}

    def __init__(self, name: str = "particleTracks", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._n_particles: int = int(self.config.get("nParticleTracks", 100))
        self._track_length: int = int(self.config.get("trackLength", 1000))
        self._scheme: str = self.config.get("integrationScheme", "RK4")
        self._dt_override: Optional[float] = self.config.get("dt")
        self._field_name: str = self.config.get("fields", "U")
        self._life_time: float = float(self.config.get("lifeTime", 1e30))
        self._cloud_name: str = self.config.get("cloudName", "particleTracks")
        self._seed_points: Optional[List[List[float]]] = self.config.get("seedPoints")

        if self._scheme not in self.SCHEMES:
            raise ValueError(
                f"Unknown integration scheme '{self._scheme}'. "
                f"Available: {self.SCHEMES}"
            )

        # Particle state
        self._positions: Optional[torch.Tensor] = None  # (n_particles, 3)
        self._tracks: List[List[torch.Tensor]] = []  # list of track point lists
        self._active: Optional[torch.Tensor] = None  # (n_particles,) bool
        self._n_steps: int = 0
        self._current_time: float = 0.0

        # Velocity field cache (updated each execute call)
        self._U: Optional[torch.Tensor] = None

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and initialise particle positions."""
        self._mesh = mesh
        self._fields = fields

        device = get_device()
        dtype = get_default_dtype()

        # Determine particle seed positions
        if self._seed_points is not None:
            self._positions = torch.tensor(
                self._seed_points, dtype=dtype, device=device,
            )
            self._n_particles = len(self._seed_points)
        else:
            self._positions = self._generate_random_seeds(mesh, dtype, device)

        # Track storage: one list per particle
        self._tracks = [[self._positions[i].clone().cpu()] for i in range(self._n_particles)]
        self._active = torch.ones(self._n_particles, dtype=torch.bool, device=device)
        self._n_steps = 0

        logger.info(
            "ParticleTracks '%s' initialised: %d particles, scheme=%s",
            self.name, self._n_particles, self._scheme,
        )

    def execute(self, time: float) -> None:
        """Advect particles using current velocity field."""
        if not self._enabled or self._mesh is None or self._positions is None:
            return

        if not self._active.any():
            return

        device = get_device()
        dtype = get_default_dtype()

        # Get velocity field
        U_field = self._fields.get(self._field_name)
        if U_field is None:
            logger.warning("Field '%s' not found. Skipping.", self._field_name)
            return

        if hasattr(U_field, "internal_field"):
            self._U = U_field.internal_field.to(device=device, dtype=dtype)
        elif hasattr(U_field, "data"):
            self._U = U_field.data.to(device=device, dtype=dtype)
        else:
            self._U = U_field.to(device=device, dtype=dtype)

        dt = self._dt_override if self._dt_override is not None else max(
            time - self._current_time, 1e-10
        )

        # Advect active particles
        self._advect_particles(dt, dtype, device)

        # Store track points
        for i in range(self._n_particles):
            if self._active[i]:
                self._tracks[i].append(self._positions[i].clone().cpu())

        self._current_time = time
        self._n_steps += 1

        # Check lifetime
        if self._current_time > self._life_time:
            self._active[:] = False

        self._log.info(
            "t=%g  active=%d/%d  steps=%d",
            time, self._active.sum().item(), self._n_particles, self._n_steps,
        )

    def _advect_particles(
        self, dt: float, dtype: torch.dtype, device: torch.device,
    ) -> None:
        """Advect particles one time step using the selected RK scheme."""
        if self._scheme == "RK1":
            self._advect_rk1(dt, dtype, device)
        elif self._scheme == "RK2":
            self._advect_rk2(dt, dtype, device)
        elif self._scheme == "RK4":
            self._advect_rk4(dt, dtype, device)

    def _sample_velocity(
        self, positions: torch.Tensor, dtype: torch.dtype, device: torch.device,
    ) -> torch.Tensor:
        """Interpolate velocity at particle positions (nearest-cell).

        Parameters
        ----------
        positions : torch.Tensor
            ``(n, 3)`` particle positions.

        Returns
        -------
        torch.Tensor
            ``(n, 3)`` interpolated velocity.
        """
        mesh = self._mesh
        U = self._U

        if U is None:
            return torch.zeros_like(positions)

        cc = mesh.cell_centres.to(device=device, dtype=dtype)

        # Nearest-cell interpolation
        # positions: (n, 3), cc: (n_cells, 3)
        # Compute distances: (n, n_cells)
        diff = positions.unsqueeze(1) - cc.unsqueeze(0)  # (n, n_cells, 3)
        dist = diff.norm(dim=2)  # (n, n_cells)
        nearest_cell = dist.argmin(dim=1)  # (n,)

        # Mask out-of-domain particles
        cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)
        in_domain = (nearest_cell < len(U))

        vel = torch.zeros_like(positions)
        valid_mask = self._active & in_domain
        if valid_mask.any():
            vel[valid_mask] = U[nearest_cell[valid_mask]]

        return vel

    def _advect_rk1(self, dt: float, dtype: torch.dtype, device: torch.device) -> None:
        """Forward Euler advection."""
        vel = self._sample_velocity(self._positions, dtype, device)
        self._positions[self._active] += dt * vel[self._active]

    def _advect_rk2(self, dt: float, dtype: torch.dtype, device: torch.device) -> None:
        """Midpoint method advection."""
        active = self._active
        pos = self._positions[active]

        vel1 = self._sample_velocity(self._positions, dtype, device)[active]
        pos_mid = pos + 0.5 * dt * vel1

        # Create temp positions for midpoint sampling
        temp_pos = self._positions.clone()
        temp_pos[active] = pos_mid
        vel2 = self._sample_velocity(temp_pos, dtype, device)[active]

        self._positions[active] = pos + dt * vel2

    def _advect_rk4(self, dt: float, dtype: torch.dtype, device: torch.device) -> None:
        """Classic Runge-Kutta 4th order advection."""
        active = self._active
        pos = self._positions[active]

        # k1
        vel1 = self._sample_velocity(self._positions, dtype, device)[active]
        k1 = dt * vel1

        # k2
        temp = self._positions.clone()
        temp[active] = pos + 0.5 * k1
        vel2 = self._sample_velocity(temp, dtype, device)[active]
        k2 = dt * vel2

        # k3
        temp = self._positions.clone()
        temp[active] = pos + 0.5 * k2
        vel3 = self._sample_velocity(temp, dtype, device)[active]
        k3 = dt * vel3

        # k4
        temp = self._positions.clone()
        temp[active] = pos + k3
        vel4 = self._sample_velocity(temp, dtype, device)[active]
        k4 = dt * vel4

        self._positions[active] = pos + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def _generate_random_seeds(
        self, mesh, dtype: torch.dtype, device: torch.device,
    ) -> torch.Tensor:
        """Generate random seed points inside the mesh bounding box.

        Parameters
        ----------
        mesh : FvMesh
            The mesh.
        dtype : torch.dtype
            Tensor dtype.
        device : torch.device
            Tensor device.

        Returns
        -------
        torch.Tensor
            ``(n_particles, 3)`` seed positions.
        """
        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        bb_min = cc.min(dim=0).values
        bb_max = cc.max(dim=0).values

        # Random points in bounding box
        rand = torch.rand(self._n_particles, 3, dtype=dtype, device=device)
        seeds = bb_min + rand * (bb_max - bb_min)

        return seeds

    def finalise(self) -> None:
        """Mark all particles as inactive."""
        if self._active is not None:
            self._active[:] = False
        logger.info("ParticleTracks '%s' finalised: %d steps", self.name, self._n_steps)

    def write(self) -> None:
        """Write particle track data to output files."""
        if self._output_path is None:
            return

        if not self._tracks:
            return

        # Write tracks as simple VTK-like text format
        tracks_file = self._output_path / f"{self._cloud_name}.vtk"
        with open(tracks_file, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"Particle tracks - {self._cloud_name}\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")

            # Count total points and lines
            total_points = 0
            total_lines = 0
            for track in self._tracks:
                n = len(track)
                if n >= 2:
                    total_points += n
                    total_lines += 1

            f.write(f"POINTS {total_points} float\n")
            for track in self._tracks:
                if len(track) < 2:
                    continue
                for pt in track:
                    f.write(f"{pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n")

            # Lines connectivity
            line_sizes = [len(t) for t in self._tracks if len(t) >= 2]
            total_line_entries = sum(line_sizes) + total_lines
            f.write(f"LINES {total_lines} {total_line_entries}\n")
            offset = 0
            for track in self._tracks:
                if len(track) < 2:
                    continue
                n = len(track)
                indices = " ".join(str(offset + j) for j in range(n))
                f.write(f"{n} {indices}\n")
                offset += n

        logger.info("Wrote particle tracks to %s (%d tracks)", tracks_file, total_lines)

        # Write track data as CSV
        csv_file = self._output_path / f"{self._cloud_name}_tracks.csv"
        with open(csv_file, "w") as f:
            f.write("track_id,step,x,y,z\n")
            for i, track in enumerate(self._tracks):
                for j, pt in enumerate(track):
                    f.write(f"{i},{j},{pt[0]:.6e},{pt[1]:.6e},{pt[2]:.6e}\n")
        logger.info("Wrote track CSV to %s", csv_file)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def positions(self) -> Optional[torch.Tensor]:
        """Current particle positions ``(n, 3)``."""
        return self._positions

    @property
    def tracks(self) -> List[List[torch.Tensor]]:
        """Particle track history. Each track is a list of ``(3,)`` position tensors."""
        return self._tracks

    @property
    def active(self) -> Optional[torch.Tensor]:
        """Boolean mask of active particles."""
        return self._active

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self._n_particles

    @property
    def n_steps(self) -> int:
        """Number of advection steps performed."""
        return self._n_steps

    @property
    def cloud_name(self) -> str:
        """Name of the particle cloud."""
        return self._cloud_name


# Register
FunctionObjectRegistry.register("particleTracks", ParticleTracks)
