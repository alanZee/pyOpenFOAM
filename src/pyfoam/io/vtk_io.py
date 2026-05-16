"""
VTK file I/O and foamToVTK converter.

Provides standalone VTK writing utilities for converting OpenFOAM meshes
and fields to VTK format for visualization in ParaView.

Supported formats:

- VTK Legacy (.vtk) — ASCII unstructured grid
- VTK XML (.vtu) — ASCII unstructured grid XML

The postprocessing module's ``VTKWriter`` and ``FoamToVTK`` function objects
use this module internally for the actual file writing.

References
----------
- VTK file format specification
- OpenFOAM ``foamToVTK`` utility source
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.mesh_io import BoundaryPatch, MeshData

__all__ = [
    "write_vtk_unstructured",
    "write_vtu_unstructured",
    "foam_to_vtk",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VTK Legacy format writing
# ---------------------------------------------------------------------------


def write_vtk_unstructured(
    path: Union[str, Path],
    points: torch.Tensor,
    faces: List[np.ndarray],
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_cells: int,
    cell_data: Optional[Dict[str, torch.Tensor]] = None,
    point_data: Optional[Dict[str, torch.Tensor]] = None,
    title: str = "pyOpenFOAM data",
) -> None:
    """Write an unstructured grid in VTK legacy format.

    Args:
        path: Output file path.
        points: Vertex coordinates, shape ``(n_points, 3)``.
        faces: List of face vertex-index arrays.
        owner: Owner cell indices, shape ``(n_faces,)``.
        neighbour: Neighbour cell indices (internal faces only).
        n_cells: Total number of cells.
        cell_data: Optional dict of cell data fields.
        point_data: Optional dict of point data fields.
        title: VTK file title.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_points = points.shape[0]
    n_faces = len(faces)
    n_internal = neighbour.shape[0]

    # Build cell -> faces mapping
    cell_faces: Dict[int, List[int]] = {i: [] for i in range(n_cells)}
    for face_idx in range(n_faces):
        own = owner[face_idx].item()
        cell_faces[own].append(face_idx)
        if face_idx < n_internal:
            nei = neighbour[face_idx].item()
            cell_faces[nei].append(face_idx)

    # Convert points to numpy
    pts = points.detach().cpu().numpy()

    with open(path, "w") as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write(f"POINTS {n_points} double\n")
        for i in range(n_points):
            f.write(f"{pts[i, 0]:.10e} {pts[i, 1]:.10e} {pts[i, 2]:.10e}\n")

        # Cells (polyhedra)
        # VTK polyhedron format: nFaces face0 face1 ...
        # Each face: nVerts v0 v1 ...
        total_entries = 0
        for cell_idx in range(n_cells):
            cf = cell_faces[cell_idx]
            total_entries += 1  # nFaces
            for face_idx in cf:
                n_verts = len(faces[face_idx])
                total_entries += 1 + n_verts  # nVerts + vertex indices

        f.write(f"CELLS {n_cells} {total_entries}\n")
        for cell_idx in range(n_cells):
            cf = cell_faces[cell_idx]
            f.write(f"{len(cf)}")
            for face_idx in cf:
                verts = faces[face_idx]
                f.write(f" {len(verts)}")
                for v in verts:
                    f.write(f" {v}")
            f.write("\n")

        # Cell types
        f.write(f"CELL_TYPES {n_cells}\n")
        for _ in range(n_cells):
            f.write("42\n")  # VTK_POLYHEDRON

        # Cell data
        if cell_data:
            f.write(f"CELL_DATA {n_cells}\n")
            for name, data in cell_data.items():
                _write_cell_data_field(f, name, data)

        # Point data
        if point_data:
            f.write(f"POINT_DATA {n_points}\n")
            for name, data in point_data.items():
                _write_point_data_field(f, name, data)

    logger.info("Wrote VTK file: %s", path)


def _write_cell_data_field(f, name: str, data: torch.Tensor) -> None:
    """Write a cell data field to VTK file."""
    data_np = data.detach().cpu().numpy()

    if data_np.ndim == 1:
        # Scalar
        f.write(f"SCALARS {name} double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in data_np:
            f.write(f"{val:.10e}\n")
    elif data_np.ndim == 2 and data_np.shape[1] == 3:
        # Vector
        f.write(f"VECTORS {name} double\n")
        for i in range(data_np.shape[0]):
            f.write(f"{data_np[i, 0]:.10e} {data_np[i, 1]:.10e} {data_np[i, 2]:.10e}\n")
    elif data_np.ndim == 3 and data_np.shape[1] == 3 and data_np.shape[2] == 3:
        # Tensor
        f.write(f"TENSORS {name} double\n")
        for i in range(data_np.shape[0]):
            for row in range(3):
                f.write(
                    f"{data_np[i, row, 0]:.10e} "
                    f"{data_np[i, row, 1]:.10e} "
                    f"{data_np[i, row, 2]:.10e}\n"
                )


def _write_point_data_field(f, name: str, data: torch.Tensor) -> None:
    """Write a point data field to VTK file."""
    data_np = data.detach().cpu().numpy()

    if data_np.ndim == 1:
        f.write(f"SCALARS {name} double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in data_np:
            f.write(f"{val:.10e}\n")
    elif data_np.ndim == 2 and data_np.shape[1] == 3:
        f.write(f"VECTORS {name} double\n")
        for i in range(data_np.shape[0]):
            f.write(f"{data_np[i, 0]:.10e} {data_np[i, 1]:.10e} {data_np[i, 2]:.10e}\n")


# ---------------------------------------------------------------------------
# VTK XML format writing
# ---------------------------------------------------------------------------


def write_vtu_unstructured(
    path: Union[str, Path],
    points: torch.Tensor,
    faces: List[np.ndarray],
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_cells: int,
    cell_data: Optional[Dict[str, torch.Tensor]] = None,
    point_data: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Write an unstructured grid in VTK XML format (.vtu).

    Args:
        path: Output file path.
        points: Vertex coordinates, shape ``(n_points, 3)``.
        faces: List of face vertex-index arrays.
        owner: Owner cell indices.
        neighbour: Neighbour cell indices (internal faces only).
        n_cells: Total number of cells.
        cell_data: Optional dict of cell data fields.
        point_data: Optional dict of point data fields.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_points = points.shape[0]
    n_faces_total = len(faces)
    n_internal = neighbour.shape[0]

    # Build cell -> faces mapping
    cell_faces: Dict[int, List[int]] = {i: [] for i in range(n_cells)}
    for face_idx in range(n_faces_total):
        own = owner[face_idx].item()
        cell_faces[own].append(face_idx)
        if face_idx < n_internal:
            nei = neighbour[face_idx].item()
            cell_faces[nei].append(face_idx)

    pts = points.detach().cpu().numpy()

    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">\n')

        # Points
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for i in range(n_points):
            f.write(f"          {pts[i, 0]:.10e} {pts[i, 1]:.10e} {pts[i, 2]:.10e}\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")

        # Cells
        f.write("      <Cells>\n")

        # Connectivity
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for cell_idx in range(n_cells):
            cell_verts = set()
            for face_idx in cell_faces[cell_idx]:
                for v in faces[face_idx]:
                    cell_verts.add(v)
            for v in sorted(cell_verts):
                f.write(f"          {v}\n")
        f.write("        </DataArray>\n")

        # Offsets
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        offset = 0
        for cell_idx in range(n_cells):
            cell_verts = set()
            for face_idx in cell_faces[cell_idx]:
                for v in faces[face_idx]:
                    cell_verts.add(v)
            offset += len(cell_verts)
            f.write(f"          {offset}\n")
        f.write("        </DataArray>\n")

        # Types (42 = VTK_POLYHEDRON)
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for _ in range(n_cells):
            f.write("          42\n")
        f.write("        </DataArray>\n")

        f.write("      </Cells>\n")

        # Cell data
        if cell_data:
            f.write("      <CellData>\n")
            for name, data in cell_data.items():
                _write_vtu_data_array(f, name, data)
            f.write("      </CellData>\n")

        # Point data
        if point_data:
            f.write("      <PointData>\n")
            for name, data in point_data.items():
                _write_vtu_data_array(f, name, data)
            f.write("      </PointData>\n")

        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")

    logger.info("Wrote VTU file: %s", path)


def _write_vtu_data_array(f, name: str, data: torch.Tensor) -> None:
    """Write a data array in VTU format."""
    data_np = data.detach().cpu().numpy()

    if data_np.ndim == 1:
        f.write(f'        <DataArray type="Float64" Name="{name}" format="ascii">\n')
        for val in data_np:
            f.write(f"          {val:.10e}\n")
        f.write("        </DataArray>\n")
    elif data_np.ndim == 2 and data_np.shape[1] == 3:
        f.write(
            f'        <DataArray type="Float64" Name="{name}" '
            f'NumberOfComponents="3" format="ascii">\n'
        )
        for i in range(data_np.shape[0]):
            f.write(f"          {data_np[i, 0]:.10e} {data_np[i, 1]:.10e} {data_np[i, 2]:.10e}\n")
        f.write("        </DataArray>\n")


# ---------------------------------------------------------------------------
# foamToVTK conversion
# ---------------------------------------------------------------------------


def foam_to_vtk(
    case_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    *,
    time_name: Optional[str] = None,
    fmt: str = "vtk",
    fields: Optional[List[str]] = None,
) -> List[Path]:
    """Convert an OpenFOAM case to VTK format.

    Reads mesh and field files from the case directory and writes
    VTK files for visualization.

    Args:
        case_dir: Path to the OpenFOAM case directory.
        output_dir: Output directory for VTK files.
            Defaults to ``case_dir/VTK``.
        time_name: Specific time directory to convert.
            If None, converts all time directories.
        fmt: Output format (``"vtk"`` or ``"vtu"``).
        fields: List of field names to convert.
            If None, converts all found fields.

    Returns:
        List of generated VTK/VTU file paths.
    """
    case_dir = Path(case_dir)
    if output_dir is None:
        output_dir = case_dir / "VTK"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read mesh
    poly_mesh_dir = case_dir / "constant" / "polyMesh"
    if not poly_mesh_dir.exists():
        raise FileNotFoundError(f"polyMesh directory not found: {poly_mesh_dir}")

    from pyfoam.io.mesh_io import read_mesh
    mesh = read_mesh(poly_mesh_dir)

    # Find time directories
    if time_name:
        time_dirs = [case_dir / time_name]
    else:
        time_dirs = _find_time_dirs(case_dir)

    vtk_files: List[Path] = []

    for time_dir in time_dirs:
        if not time_dir.exists():
            continue

        time_val = time_dir.name
        vtk_path = output_dir / f"{time_val}.{fmt}"

        # Read fields from this time directory
        cell_data: Dict[str, torch.Tensor] = {}
        field_files = _find_field_files(time_dir, fields)

        for field_name, field_path in field_files.items():
            try:
                data = _read_field_internal(field_path, mesh.n_cells)
                if data is not None:
                    cell_data[field_name] = data
            except Exception as e:
                logger.warning("Failed to read field %s: %s", field_name, e)

        # Write VTK file
        if fmt == "vtu":
            write_vtu_unstructured(
                vtk_path,
                mesh.points,
                mesh.faces,
                mesh.owner,
                mesh.neighbour,
                mesh.n_cells,
                cell_data=cell_data or None,
            )
        else:
            write_vtk_unstructured(
                vtk_path,
                mesh.points,
                mesh.faces,
                mesh.owner,
                mesh.neighbour,
                mesh.n_cells,
                cell_data=cell_data or None,
                title=f"pyOpenFOAM t={time_val}",
            )

        vtk_files.append(vtk_path)
        logger.info("Converted time %s to %s", time_val, vtk_path)

    return vtk_files


def _find_time_dirs(case_dir: Path) -> List[Path]:
    """Find all time directories in a case."""
    time_dirs = []
    for item in case_dir.iterdir():
        if item.is_dir():
            try:
                float(item.name)
                time_dirs.append(item)
            except ValueError:
                continue
    return sorted(time_dirs, key=lambda p: float(p.name))


def _find_field_files(
    time_dir: Path, field_names: Optional[List[str]]
) -> Dict[str, Path]:
    """Find field files in a time directory."""
    field_files: Dict[str, Path] = {}

    for item in time_dir.iterdir():
        if not item.is_file():
            continue

        name = item.name

        # Skip non-field files
        if name in ("uniform", "polyMesh"):
            continue

        # Check if it's a field file (has FoamFile header)
        try:
            content = item.read_text(encoding="utf-8", errors="replace")
            if "FoamFile" not in content:
                continue
        except Exception:
            continue

        if field_names is None or name in field_names:
            field_files[name] = item

    return field_files


def _read_field_internal(
    field_path: Path, expected_cells: int
) -> Optional[torch.Tensor]:
    """Read internal field data from a field file.

    Returns:
        Tensor of shape ``(n_cells,)`` for scalars or ``(n_cells, 3)``
        for vectors, or None if parsing fails.
    """
    from pyfoam.io.foam_file import read_foam_file

    header, body = read_foam_file(field_path)

    # Find internalField
    import re

    # Look for "internalField" keyword
    match = re.search(r"internalField\s+(.+?);", body, re.DOTALL)
    if match is None:
        return None

    field_str = match.group(1).strip()

    # Parse uniform field
    if field_str.startswith("uniform"):
        value_str = field_str[len("uniform"):].strip()
        if value_str.startswith("("):
            # Vector: (x y z)
            values = value_str.strip("()").split()
            vec = torch.tensor(
                [float(v) for v in values],
                dtype=get_default_dtype(),
                device=get_device(),
            )
            return vec.unsqueeze(0).expand(expected_cells, -1).contiguous()
        else:
            # Scalar
            val = float(value_str)
            return torch.full(
                (expected_cells,), val, dtype=get_default_dtype(), device=get_device()
            )

    # Parse nonuniform field
    if field_str.startswith("nonuniform"):
        rest = field_str[len("nonuniform"):].strip()

        # Find list: N ( ... ) or N ((v1) (v2) ...)
        list_match = re.match(r"(\d+)\s*\n?\s*(.+)", rest, re.DOTALL)
        if list_match is None:
            return None

        n = int(list_match.group(1))
        data_str = list_match.group(2).strip()

        if data_str.startswith("("):
            # Check if vector list: ((x1 y1 z1) (x2 y2 z2) ...)
            inner = data_str.strip("()")
            if inner.startswith("("):
                # Vector list
                values = []
                for m in re.finditer(r"\(\s*([^)]+)\)", inner):
                    coords = [float(v) for v in m.group(1).split()]
                    values.append(coords)
                if len(values) == n:
                    return torch.tensor(
                        values, dtype=get_default_dtype(), device=get_device()
                    )
            else:
                # Scalar list
                values = [float(v) for v in inner.split()]
                if len(values) == n:
                    return torch.tensor(
                        values, dtype=get_default_dtype(), device=get_device()
                    )

    return None
