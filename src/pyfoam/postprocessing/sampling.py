"""
Sampling — point probes, line sampling, and surface sampling.

Provides function objects for sampling field values at specific locations
in the domain, mirroring OpenFOAM's ``probes``, ``sets``, and ``surfaces``
function objects.

Supported sampling types:

- ``Probes``: Sample at specific points (interpolated from cell values)
- ``LineSample``: Sample along a line (sets)
- ``SurfaceSample``: Sample on a surface

References
----------
- OpenFOAM ``probes`` function object source
- OpenFOAM ``sets`` function object source
- OpenFOAM ``surfaces`` function object source
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["Probes", "LineSample", "SurfaceSample"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: find cell containing a point
# ---------------------------------------------------------------------------


def _find_cell_for_point(
    point: torch.Tensor,
    mesh,
) -> int:
    """Find the cell index containing *point* using nearest-cell-centre.

    This is a simplified approach.  For accurate interpolation, a proper
    cell-location algorithm (walk + tree search) would be needed.

    Args:
        point: ``(3,)`` tensor of coordinates.
        mesh: The FvMesh.

    Returns:
        Index of the nearest cell, or -1 if not found.
    """
    device = mesh.device
    dtype = mesh.dtype
    p = point.to(device=device, dtype=dtype).unsqueeze(0)  # (1, 3)
    cc = mesh.cell_centres.to(device=device, dtype=dtype)  # (n_cells, 3)

    dist = (cc - p).norm(dim=1)  # (n_cells,)
    idx = dist.argmin().item()

    return idx


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


class Probes(FunctionObject):
    """Sample fields at specific probe points.

    Configuration keys:

    - ``fields``: list of field names to sample
    - ``probeLocations``: list of (x, y, z) coordinates
    - ``interpolationScheme``: ``"cellCentre"`` (default) or ``"linear"``

    Example controlDict entry::

        probes1
        {
            type            probes;
            libs            ("libfieldFunctionObjects.so");
            fields          (p U);
            probeLocations
            (
                (0.5 0.5 0.5)
                (1.0 0.5 0.5)
            );
            interpolationScheme cellCentre;
        }
    """

    def __init__(self, name: str = "probes", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._field_names: List[str] = self.config.get("fields", [])
        raw_locations = self.config.get("probeLocations", [])
        self._locations: List[torch.Tensor] = [
            torch.tensor(loc, dtype=torch.float64) for loc in raw_locations
        ]
        self._interpolation: str = self.config.get("interpolationScheme", "cellCentre")

        # Cell indices for each probe (computed in initialise)
        self._cell_indices: List[int] = []

        # Results: {field_name: {probe_idx: [values]}}
        self._results: Dict[str, Dict[int, List[float]]] = {}
        self._vector_results: Dict[str, Dict[int, List[List[float]]]] = {}
        self._times: List[float] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Locate probe cells and initialise storage."""
        self._mesh = mesh
        self._fields = fields

        # Find cells for each probe
        self._cell_indices = []
        for i, loc in enumerate(self._locations):
            cell_idx = _find_cell_for_point(loc, mesh)
            self._cell_indices.append(cell_idx)
            if cell_idx >= 0:
                logger.info("Probe %d at %s -> cell %d", i, loc.tolist(), cell_idx)
            else:
                logger.warning("Probe %d at %s not found in mesh", i, loc.tolist())

        # Initialise result storage
        for fname in self._field_names:
            self._results[fname] = {i: [] for i in range(len(self._locations))}
            self._vector_results[fname] = {i: [] for i in range(len(self._locations))}

        logger.info("Probes '%s' initialised: %d probes, fields=%s", self.name, len(self._locations), self._field_names)

    def execute(self, time: float) -> None:
        """Sample fields at probe locations."""
        if not self._enabled or self._mesh is None:
            return

        for fname in self._field_names:
            field = self._fields.get(fname)
            if field is None:
                continue

            if hasattr(field, "internal_field"):
                data = field.internal_field
            else:
                data = field

            for i, cell_idx in enumerate(self._cell_indices):
                if cell_idx < 0:
                    continue

                value = data[cell_idx]
                if value.dim() == 0:
                    # Scalar
                    self._results[fname][i].append(value.item())
                else:
                    # Vector/tensor
                    self._vector_results[fname][i].append(value.detach().cpu().tolist())

        self._times.append(time)

    def write(self) -> None:
        """Write probe data to output files."""
        if self._output_path is None or not self._times:
            return

        for fname in self._field_names:
            probe_file = self._output_path / f"{fname}_probe.dat"

            # Check if scalar or vector
            is_vector = any(
                len(vals) > 0 and isinstance(vals[0], list)
                for vals in self._vector_results.get(fname, {}).values()
            )

            with open(probe_file, "w") as f:
                if is_vector:
                    # Vector field
                    header = "# Time"
                    for i in range(len(self._locations)):
                        header += f"  probe{i}_x  probe{i}_y  probe{i}_z"
                    f.write(header + "\n")

                    for t_idx, t in enumerate(self._times):
                        line = f"{t:.6e}"
                        for i in range(len(self._locations)):
                            vals = self._vector_results[fname].get(i, [])
                            if t_idx < len(vals):
                                v = vals[t_idx]
                                line += f"  {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}"
                            else:
                                line += "  0.0  0.0  0.0"
                        f.write(line + "\n")
                else:
                    # Scalar field
                    header = "# Time"
                    for i in range(len(self._locations)):
                        header += f"  probe{i}"
                    f.write(header + "\n")

                    for t_idx, t in enumerate(self._times):
                        line = f"{t:.6e}"
                        for i in range(len(self._locations)):
                            vals = self._results[fname].get(i, [])
                            if t_idx < len(vals):
                                line += f"  {vals[t_idx]:.6e}"
                            else:
                                line += "  0.0"
                        f.write(line + "\n")

            logger.info("Wrote probe data to %s", probe_file)

    @property
    def results(self) -> Dict[str, Dict[int, List[float]]]:
        """Scalar probe results."""
        return self._results

    @property
    def vector_results(self) -> Dict[str, Dict[int, List[List[float]]]]:
        """Vector probe results."""
        return self._vector_results

    @property
    def times(self) -> List[float]:
        """Time values."""
        return self._times


# ---------------------------------------------------------------------------
# LineSample (sets)
# ---------------------------------------------------------------------------


class LineSample(FunctionObject):
    """Sample fields along a line in the domain.

    Configuration keys:

    - ``fields``: list of field names to sample
    - ``start``: start point (x, y, z)
    - ``end``: end point (x, y, z)
    - ``nPoints``: number of sample points along the line (default: 100)

    Example controlDict entry::

        lineSample1
        {
            type            sets;
            libs            ("libsampling.so");
            fields          (p U);
            start           (0 0 0);
            end             (1 0 0);
            nPoints         50;
        }
    """

    def __init__(self, name: str = "lineSample", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._field_names: List[str] = self.config.get("fields", [])
        self._start = torch.tensor(self.config.get("start", [0.0, 0.0, 0.0]), dtype=torch.float64)
        self._end = torch.tensor(self.config.get("end", [1.0, 0.0, 0.0]), dtype=torch.float64)
        self._n_points: int = int(self.config.get("nPoints", 100))

        # Sample points along the line
        self._sample_points: List[torch.Tensor] = []
        self._cell_indices: List[int] = []

        # Results
        self._results: Dict[str, List[List[float]]] = {}  # {field: [[values_at_t0], [values_at_t1], ...]}
        self._vector_results: Dict[str, List[List[List[float]]]] = {}
        self._times: List[float] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Generate sample points and locate cells."""
        self._mesh = mesh
        self._fields = fields

        # Generate points along the line
        self._sample_points = []
        self._cell_indices = []
        for i in range(self._n_points):
            alpha = i / max(self._n_points - 1, 1)
            point = self._start + alpha * (self._end - self._start)
            self._sample_points.append(point)
            cell_idx = _find_cell_for_point(point, mesh)
            self._cell_indices.append(cell_idx)

        for fname in self._field_names:
            self._results[fname] = []
            self._vector_results[fname] = []

        logger.info(
            "LineSample '%s' initialised: %d points from %s to %s",
            self.name, self._n_points, self._start.tolist(), self._end.tolist()
        )

    def execute(self, time: float) -> None:
        """Sample fields along the line."""
        if not self._enabled or self._mesh is None:
            return

        for fname in self._field_names:
            field = self._fields.get(fname)
            if field is None:
                continue

            if hasattr(field, "internal_field"):
                data = field.internal_field
            else:
                data = field

            scalar_values = []
            vector_values = []
            for cell_idx in self._cell_indices:
                if cell_idx < 0:
                    if data.dim() == 0 or (data.dim() >= 1 and data.shape[-1] == 1):
                        scalar_values.append(0.0)
                    else:
                        vector_values.append([0.0, 0.0, 0.0])
                    continue

                value = data[cell_idx]
                if value.dim() == 0:
                    scalar_values.append(value.item())
                else:
                    vector_values.append(value.detach().cpu().tolist())

            if scalar_values:
                self._results[fname].append(scalar_values)
            if vector_values:
                self._vector_results[fname].append(vector_values)

        self._times.append(time)

    def write(self) -> None:
        """Write line sample data."""
        if self._output_path is None or not self._times:
            return

        for fname in self._field_names:
            line_file = self._output_path / f"{fname}_line.dat"

            # Compute distances along line
            line_length = (self._end - self._start).norm().item()
            distances = [
                i / max(self._n_points - 1, 1) * line_length
                for i in range(self._n_points)
            ]

            with open(line_file, "w") as f:
                # Write header
                f.write(f"# Line from {self._start.tolist()} to {self._end.tolist()}\n")
                f.write(f"# {self._n_points} points, {len(self._times)} time steps\n")

                if self._results.get(fname):
                    # Scalar field
                    f.write("# distance")
                    for t in self._times:
                        f.write(f"  t={t:.4e}")
                    f.write("\n")

                    for i in range(self._n_points):
                        line = f"{distances[i]:.6e}"
                        for t_idx in range(len(self._times)):
                            vals = self._results[fname][t_idx]
                            if i < len(vals):
                                line += f"  {vals[i]:.6e}"
                        f.write(line + "\n")
                elif self._vector_results.get(fname):
                    # Vector field - write magnitude
                    f.write("# distance")
                    for t in self._times:
                        f.write(f"  |U|_t={t:.4e}")
                    f.write("\n")

                    for i in range(self._n_points):
                        line = f"{distances[i]:.6e}"
                        for t_idx in range(len(self._times)):
                            vals = self._vector_results[fname][t_idx]
                            if i < len(vals):
                                v = vals[i]
                                mag = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
                                line += f"  {mag:.6e}"
                        f.write(line + "\n")

            logger.info("Wrote line sample to %s", line_file)

    @property
    def sample_points(self) -> List[torch.Tensor]:
        """Sample point coordinates."""
        return self._sample_points

    @property
    def results(self) -> Dict[str, List[List[float]]]:
        """Scalar results: {field: [[values_at_t0], ...]}."""
        return self._results

    @property
    def times(self) -> List[float]:
        """Time values."""
        return self._times


# ---------------------------------------------------------------------------
# SurfaceSample
# ---------------------------------------------------------------------------


class SurfaceSample(FunctionObject):
    """Sample fields on a surface (plane, sphere, etc.).

    Configuration keys:

    - ``fields``: list of field names to sample
    - ``surfaceType``: type of surface (``"plane"``, ``"sphere"``)
    - ``surfaceParams``: parameters for the surface
    - ``nPoints``: number of sample points (default: 1000)

    For plane surfaces:

    - ``point``: point on the plane
    - ``normal``: normal vector

    For sphere surfaces:

    - ``centre``: sphere centre
    - ``radius``: sphere radius

    Example controlDict entry::

        surfaceSample1
        {
            type            surfaces;
            libs            ("libsampling.so");
            fields          (p U);
            surfaceType     plane;
            surfaceParams
            {
                point   (0.5 0.5 0.5);
                normal  (0 0 1);
            }
            nPoints         500;
        }
    """

    def __init__(self, name: str = "surfaceSample", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._field_names: List[str] = self.config.get("fields", [])
        self._surface_type: str = self.config.get("surfaceType", "plane")
        self._surface_params: Dict[str, Any] = self.config.get("surfaceParams", {})
        self._n_points: int = int(self.config.get("nPoints", 1000))

        # Sample points on surface
        self._sample_points: List[torch.Tensor] = []
        self._cell_indices: List[int] = []

        # Results
        self._results: Dict[str, List[Dict[str, float]]] = {}
        self._times: List[float] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Generate surface points and locate cells."""
        self._mesh = mesh
        self._fields = fields

        # Generate points on the surface
        if self._surface_type == "plane":
            self._generate_plane_points()
        elif self._surface_type == "sphere":
            self._generate_sphere_points()
        else:
            logger.warning("Unknown surface type '%s'. Using plane.", self._surface_type)
            self._generate_plane_points()

        # Locate cells
        self._cell_indices = []
        for point in self._sample_points:
            cell_idx = _find_cell_for_point(point, mesh)
            self._cell_indices.append(cell_idx)

        for fname in self._field_names:
            self._results[fname] = []

        logger.info(
            "SurfaceSample '%s' initialised: type=%s, %d points",
            self.name, self._surface_type, len(self._sample_points)
        )

    def _generate_plane_points(self) -> None:
        """Generate points on a plane."""
        point = torch.tensor(
            self._surface_params.get("point", [0.5, 0.5, 0.5]), dtype=torch.float64
        )
        normal = torch.tensor(
            self._surface_params.get("normal", [0.0, 0.0, 1.0]), dtype=torch.float64
        )
        normal = normal / normal.norm()

        # Find two tangent vectors
        if abs(normal[2]) < 0.9:
            t1 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        else:
            t1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        t1 = t1 - torch.dot(t1, normal) * normal
        t1 = t1 / t1.norm()
        t2 = torch.cross(normal, t1)

        # Generate grid of points
        n_side = int(self._n_points ** 0.5)
        self._sample_points = []
        for i in range(n_side):
            for j in range(n_side):
                u = (i / max(n_side - 1, 1) - 0.5) * 2.0
                v = (j / max(n_side - 1, 1) - 0.5) * 2.0
                p = point + u * t1 + v * t2
                self._sample_points.append(p)

    def _generate_sphere_points(self) -> None:
        """Generate points on a sphere using Fibonacci spiral."""
        centre = torch.tensor(
            self._surface_params.get("centre", [0.5, 0.5, 0.5]), dtype=torch.float64
        )
        radius = float(self._surface_params.get("radius", 0.5))

        # Fibonacci sphere
        golden_ratio = (1 + 5**0.5) / 2
        self._sample_points = []
        for i in range(self._n_points):
            theta = 2 * 3.141592653589793 * i / golden_ratio
            phi = torch.acos(torch.tensor(1 - 2 * (i + 0.5) / self._n_points))
            x = radius * torch.sin(phi) * torch.cos(torch.tensor(theta))
            y = radius * torch.sin(phi) * torch.sin(torch.tensor(theta))
            z = radius * torch.cos(phi)
            p = centre + torch.tensor([x.item(), y.item(), z.item()], dtype=torch.float64)
            self._sample_points.append(p)

    def execute(self, time: float) -> None:
        """Sample fields on the surface."""
        if not self._enabled or self._mesh is None:
            return

        for fname in self._field_names:
            field = self._fields.get(fname)
            if field is None:
                continue

            if hasattr(field, "internal_field"):
                data = field.internal_field
            else:
                data = field

            stats = {}
            values = []
            for cell_idx in self._cell_indices:
                if cell_idx < 0:
                    continue
                value = data[cell_idx]
                if value.dim() == 0:
                    values.append(value.item())

            if values:
                vals = torch.tensor(values)
                stats["mean"] = vals.mean().item()
                stats["min"] = vals.min().item()
                stats["max"] = vals.max().item()
                stats["std"] = vals.std().item()

            self._results[fname].append(stats)

        self._times.append(time)

    def write(self) -> None:
        """Write surface sample statistics."""
        if self._output_path is None or not self._times:
            return

        for fname in self._field_names:
            surf_file = self._output_path / f"{fname}_surface.dat"
            with open(surf_file, "w") as f:
                f.write(f"# Surface: {self._surface_type}\n")
                f.write("# Time  mean  min  max  std\n")
                for t_idx, t in enumerate(self._times):
                    stats = self._results[fname][t_idx]
                    if stats:
                        f.write(
                            f"{t:.6e}  {stats.get('mean', 0):.6e}  "
                            f"{stats.get('min', 0):.6e}  {stats.get('max', 0):.6e}  "
                            f"{stats.get('std', 0):.6e}\n"
                        )
            logger.info("Wrote surface sample to %s", surf_file)

    @property
    def sample_points(self) -> List[torch.Tensor]:
        """Sample point coordinates."""
        return self._sample_points

    @property
    def results(self) -> Dict[str, List[Dict[str, float]]]:
        """Results per field per time step."""
        return self._results

    @property
    def times(self) -> List[float]:
        """Time values."""
        return self._times


# Register
FunctionObjectRegistry.register("probes", Probes)
FunctionObjectRegistry.register("sets", LineSample)
FunctionObjectRegistry.register("surfaces", SurfaceSample)
