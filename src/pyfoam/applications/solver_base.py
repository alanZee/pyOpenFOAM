"""
Base class for application-level solvers.

Bridges the gap between OpenFOAM case I/O and the low-level solver
infrastructure (FvMesh, fields, SIMPLESolver, etc.).

Provides:
- Case loading from a directory path
- Mesh construction from ``constant/polyMesh`` data
- Field initialisation from ``0/`` directory
- Control-dict parsing (endTime, deltaT, writeControl, etc.)
- Field writing in OpenFOAM format

Usage::

    class MySolver(SolverBase):
        def run(self):
            ...

    solver = MySolver("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.io.case import Case
from pyfoam.io.field_io import FieldData, BoundaryPatch as FieldBoundaryPatch
from pyfoam.io.foam_file import FoamFileHeader, FileFormat
from pyfoam.io.mesh_io import BoundaryPatch as MeshBoundaryPatch
from pyfoam.mesh.poly_mesh import PolyMesh
from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SolverBase"]

logger = logging.getLogger(__name__)


class SolverBase:
    """Base class for application-level solvers.

    Reads an OpenFOAM case directory and constructs the mesh, fields,
    and solver infrastructure needed by concrete solvers.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    """

    def __init__(self, case_path: Union[str, Path]) -> None:
        self.case_path = Path(case_path)
        self.case = Case(self.case_path)

        # Build FvMesh from case data
        self.mesh = self._build_mesh()

        # Read control settings
        self._read_control_settings()

        logger.info("SolverBase initialised: %s", self.case_path)
        logger.info("  Mesh: %s", self.mesh)

    # ------------------------------------------------------------------
    # Mesh construction
    # ------------------------------------------------------------------

    def _build_mesh(self) -> FvMesh:
        """Build an FvMesh from the case's polyMesh data.

        Converts the ``MeshData`` (numpy faces, torch tensors) into a
        :class:`PolyMesh`, then wraps it in an :class:`FvMesh` with
        geometry computed.
        """
        mesh_data = self.case.mesh
        boundary_patches = self.case.boundary

        # Convert numpy face arrays to torch tensors
        face_tensors = [
            torch.tensor(f, dtype=INDEX_DTYPE, device="cpu")
            for f in mesh_data.faces
        ]

        # Convert BoundaryPatch objects to dicts for PolyMesh
        boundary_dicts = []
        for bp in boundary_patches:
            boundary_dicts.append({
                "name": bp.name,
                "type": bp.patch_type,
                "startFace": bp.start_face,
                "nFaces": bp.n_faces,
            })

        poly = PolyMesh(
            points=mesh_data.points,
            faces=face_tensors,
            owner=mesh_data.owner,
            neighbour=mesh_data.neighbour,
            boundary=boundary_dicts,
            validate=True,
        )

        fv = FvMesh.from_poly_mesh(poly)
        return fv

    # ------------------------------------------------------------------
    # Control settings
    # ------------------------------------------------------------------

    def _read_control_settings(self) -> None:
        """Read simulation control parameters from controlDict."""
        cd = self.case.controlDict

        self.delta_t: float = float(cd.get("deltaT", 1.0))
        self.end_time: float = float(cd.get("endTime", 100.0))
        self.start_time: float = float(cd.get("startTime", 0.0))
        self.write_interval: float = float(cd.get("writeInterval", 1.0))
        self.write_control: str = str(cd.get("writeControl", "timeStep"))
        self.application: str = str(cd.get("application", ""))

    # ------------------------------------------------------------------
    # Field I/O helpers
    # ------------------------------------------------------------------

    def read_field_tensor(
        self,
        name: str,
        time: Union[str, int, float] = 0,
    ) -> tuple[torch.Tensor, FieldData]:
        """Read a field from a time directory and return as a tensor.

        For uniform fields, broadcasts to all cells/faces.  For nonuniform
        fields, returns the tensor directly.

        Args:
            name: Field name (e.g. ``"U"``, ``"p"``).
            time: Time directory to read from.

        Returns:
            Tuple of ``(tensor, raw_field_data)``.
        """
        field_data = self.case.read_field(name, time)
        device = get_device()
        dtype = get_default_dtype()

        if field_data.is_uniform:
            value = field_data.internal_field
            if field_data.scalar_type == "vector":
                # value is a tuple (x, y, z)
                n = self.mesh.n_cells
                tensor = torch.tensor(
                    [list(value)] * n, dtype=dtype, device=device,
                )
            else:
                # scalar
                n = self.mesh.n_cells
                tensor = torch.full((n,), float(value), dtype=dtype, device=device)
        else:
            tensor = field_data.internal_field.to(device=device, dtype=dtype)

        return tensor, field_data

    def write_field(
        self,
        name: str,
        tensor: torch.Tensor,
        time: Union[str, int, float],
        field_data: FieldData,
    ) -> None:
        """Write a field to a time directory in OpenFOAM format.

        Args:
            name: Field name.
            tensor: Field values.
            time: Time value (directory name).
            field_data: Original :class:`FieldData` (for header/dimensions).
        """
        time_dir = self.case_path / str(time)
        time_dir.mkdir(parents=True, exist_ok=True)

        # Build updated FieldData
        new_field = FieldData(
            header=FoamFileHeader(
                version=field_data.header.version,
                format=FileFormat.ASCII,
                class_name=field_data.header.class_name,
                location=str(time),
                object=name,
            ),
            dimensions=field_data.dimensions,
            internal_field=tensor.detach().cpu(),
            boundary_field=field_data.boundary_field,
            is_uniform=False,
            scalar_type=field_data.scalar_type,
        )

        from pyfoam.io.field_io import write_field as _write_field
        _write_field(time_dir / name, new_field, overwrite=True)
        logger.info("Wrote field %s to time %s", name, time)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the solver.  Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")
