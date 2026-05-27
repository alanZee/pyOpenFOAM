"""
foamToVTK — export an OpenFOAM case to VTK format.

Mirrors OpenFOAM's ``foamToVTK`` utility.  Writes:

- ``.vtu`` files (VTK UnstructuredGrid) for volume mesh and cell-centred
  fields.
- ``.vtp`` files (VTK PolyData) for surface/boundary patch fields.

Output is the VTK XML serial format (ASCII).  Both standard element types
(hexahedron, tetrahedron, wedge, pyramid) and general polyhedra are
supported.

Usage::

    from pyfoam.tools.foam_to_vtk import foam_to_vtk

    foam_to_vtk(case_path, mesh=mesh, fields=fields, time_range=[0, 1])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["foam_to_vtk"]

# VTK cell type constants
_VTK_VERTEX = 1
_VTK_POLY_VERTEX = 2
_VTK_LINE = 3
_VTK_POLY_LINE = 4
_VTK_TRIANGLE = 5
_VTK_QUAD = 9
_VTK_TETRA = 10
_VTK_HEXAHEDRON = 12
_VTK_WEDGE = 13
_VTK_PYRAMID = 14
_VTK_POLYGON = 7
_VTK_POLYHEDRON = 42


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_vtk(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Export an OpenFOAM case to VTK XML format.

    When *mesh* and *fields* are provided directly (e.g. from a running
    simulation), they are used directly.  Otherwise the function scans
    *case_path* for on-disk time directories and field files.

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    time_range : sequence of float, optional
        Subset of time values to export.  ``None`` exports all available
        times (requires on-disk data).
    mesh : FvMesh, optional
        Pre-loaded mesh.  When given, geometry and connectivity are
        extracted from this object rather than read from disk.
    fields : dict, optional
        ``{field_name: numpy_array}`` of per-cell field values to export.
        Scalar arrays have shape ``(n_cells,)``, vector arrays
        ``(n_cells, 3)``.
    output_dir : str or Path, optional
        Directory for VTK output.  Defaults to ``<case_path>/VTK``.

    Returns
    -------
    Path
        Path to the output directory containing ``.vtu`` / ``.vtp`` files.

    Raises
    ------
    FileNotFoundError
        If *case_path* does not exist.
    ValueError
        If no mesh is available (neither provided nor on disk).
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    # Determine output directory
    if output_dir is None:
        vtk_dir = case_dir / "VTK"
    else:
        vtk_dir = Path(output_dir)
    os.makedirs(vtk_dir, exist_ok=True)

    # Determine time values
    if time_range is not None:
        times = sorted(float(t) for t in time_range)
    else:
        from pyfoam.tools.foam_list_times import foam_list_times

        times = foam_list_times(case_dir)
        if not times and mesh is not None:
            times = [0.0]

    if mesh is None:
        raise ValueError(
            "No mesh provided.  Pass a mesh object directly."
        )

    # Pre-compute connectivity once
    cell_verts, vtk_cell_types = _compute_cell_vertices_and_types(mesh)

    # Extract boundary patch data
    patch_data = _extract_boundary_patches(mesh)

    # Write files per time step
    for t in times:
        t_name = _format_time(t)
        # Volume mesh (.vtu)
        vtu_path = vtk_dir / f"{t_name}.vtu"
        _write_vtu(vtu_path, mesh, cell_verts, vtk_cell_types, fields)

        # Surface patches (.vtp)
        for patch_name, patch_faces, patch_owner_cells in patch_data:
            vtp_path = vtk_dir / f"{t_name}_{patch_name}.vtp"
            _write_vtp(vtp_path, mesh, patch_faces, patch_owner_cells, fields)

    return vtk_dir


# ---------------------------------------------------------------------------
# Cell-vertex connectivity and VTK types
# ---------------------------------------------------------------------------


def _compute_cell_vertices_and_types(mesh: "FvMesh"):
    """Build cell-to-unique-vertices mapping and VTK cell type codes.

    Returns
    -------
    cell_verts : list[list[int]]
        Point indices for each cell (sorted, unique).
    vtk_types : list[int]
        VTK cell type code for each cell.
    """
    n_cells = mesh.n_cells
    faces = mesh.faces
    owner = mesh.owner.detach().cpu().numpy()
    neighbour = mesh.neighbour.detach().cpu().numpy()
    n_internal = mesh.n_internal_faces

    cell_to_verts: list[set[int]] = [set() for _ in range(n_cells)]

    for fi, face in enumerate(faces):
        face_nodes = face.detach().cpu().numpy().tolist()
        c_own = int(owner[fi])
        cell_to_verts[c_own].update(face_nodes)

        if fi < n_internal:
            c_nbr = int(neighbour[fi])
            cell_to_verts[c_nbr].update(face_nodes)

    cell_verts = [sorted(verts) for verts in cell_to_verts]

    # Classify VTK cell type by unique node count
    vtk_types = []
    for verts in cell_verts:
        nn = len(verts)
        if nn == 4:
            vtk_types.append(_VTK_TETRA)
        elif nn == 5:
            vtk_types.append(_VTK_PYRAMID)
        elif nn == 6:
            vtk_types.append(_VTK_WEDGE)
        elif nn == 8:
            vtk_types.append(_VTK_HEXAHEDRON)
        else:
            vtk_types.append(_VTK_POLYHEDRON)

    return cell_verts, vtk_types


# ---------------------------------------------------------------------------
# Boundary patch extraction
# ---------------------------------------------------------------------------


def _extract_boundary_patches(mesh: "FvMesh"):
    """Extract boundary face indices grouped by patch.

    Returns
    -------
    list of (patch_name, face_indices, owner_cells)
    """
    patches = []
    owner = mesh.owner.detach().cpu().numpy()
    n_internal = mesh.n_internal_faces

    for patch_info in mesh.boundary:
        name = patch_info["name"]
        start = patch_info["startFace"]
        n_faces = patch_info["nFaces"]

        face_indices = list(range(start, start + n_faces))
        owner_cells = [int(owner[fi]) for fi in face_indices]
        patches.append((name, face_indices, owner_cells))

    return patches


# ---------------------------------------------------------------------------
# VTU (volume) writing
# ---------------------------------------------------------------------------


def _write_vtu(
    path: Path,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
    vtk_types: list[int],
    fields: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write a VTK UnstructuredGrid (.vtu) file."""
    pts = mesh.points.detach().cpu().numpy()
    n_points = pts.shape[0]
    n_cells = len(cell_verts)

    # Build total connectivity and offsets
    connectivity = []
    offsets = []
    running = 0
    for verts in cell_verts:
        connectivity.extend(verts)
        running += len(verts)
        offsets.append(running)

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("<UnstructuredGrid>\n")
        f.write(f'<Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">\n')

        # Points
        f.write("<Points>\n")
        _write_data_array(f, "Points", pts.flatten(), n_components=3)
        f.write("</Points>\n")

        # Cells
        f.write("<Cells>\n")
        _write_int_data_array(f, "connectivity", np.array(connectivity, dtype=np.int64))
        _write_int_data_array(f, "offsets", np.array(offsets, dtype=np.int64))
        _write_int_data_array(f, "types", np.array(vtk_types, dtype=np.uint8))
        f.write("</Cells>\n")

        # Cell data (fields)
        if fields:
            f.write("<CellData>\n")
            for name, data in fields.items():
                if data.ndim == 1:
                    _write_data_array(f, name, data, n_components=1)
                elif data.ndim == 2 and data.shape[1] == 3:
                    _write_data_array(f, name, data.flatten(), n_components=3)
            f.write("</CellData>\n")

        f.write("</Piece>\n")
        f.write("</UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


# ---------------------------------------------------------------------------
# VTP (surface) writing
# ---------------------------------------------------------------------------


def _write_vtp(
    path: Path,
    mesh: "FvMesh",
    face_indices: list[int],
    owner_cells: list[int],
    fields: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write a VTK PolyData (.vtp) file for a boundary patch."""
    pts = mesh.points.detach().cpu().numpy()
    faces = mesh.faces

    # Collect unique point indices across all patch faces
    all_point_indices: set[int] = set()
    for fi in face_indices:
        face_nodes = faces[fi].detach().cpu().numpy().tolist()
        all_point_indices.update(face_nodes)

    sorted_points = sorted(all_point_indices)
    point_map = {old: new for new, old in enumerate(sorted_points)}

    # Subset of points used by this patch
    sub_pts = pts[sorted_points]
    n_sub_points = len(sorted_points)

    # Build polygon connectivity
    polygons = []
    polygon_offsets = []
    running = 0
    for fi in face_indices:
        face_nodes = faces[fi].detach().cpu().numpy().tolist()
        remapped = [point_map[n] for n in face_nodes]
        polygons.extend(remapped)
        running += len(remapped)
        polygon_offsets.append(running)

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("<PolyData>\n")
        n_polys = len(face_indices)
        f.write(f'<Piece NumberOfPoints="{n_sub_points}" NumberOfPolys="{n_polys}">\n')

        # Points
        f.write("<Points>\n")
        _write_data_array(f, "Points", sub_pts.flatten(), n_components=3)
        f.write("</Points>\n")

        # Polygons
        f.write("<Polys>\n")
        _write_int_data_array(f, "connectivity", np.array(polygons, dtype=np.int64))
        _write_int_data_array(f, "offsets", np.array(polygon_offsets, dtype=np.int64))
        f.write("</Polys>\n")

        # Cell data (fields mapped to owner cells)
        if fields:
            f.write("<CellData>\n")
            for name, data in fields.items():
                if data.ndim == 1:
                    patch_data = data[owner_cells]
                    _write_data_array(f, name, patch_data, n_components=1)
                elif data.ndim == 2 and data.shape[1] == 3:
                    patch_data = data[owner_cells].flatten()
                    _write_data_array(f, name, patch_data, n_components=3)
            f.write("</CellData>\n")

        f.write("</Piece>\n")
        f.write("</PolyData>\n")
        f.write("</VTKFile>\n")


# ---------------------------------------------------------------------------
# XML helper functions
# ---------------------------------------------------------------------------


def _write_data_array(
    f,
    name: str,
    data: np.ndarray,
    n_components: int = 1,
) -> None:
    """Write a VTK DataArray element (Float64)."""
    f.write(
        f'<DataArray type="Float64" Name="{name}" '
        f'NumberOfComponents="{n_components}" format="ascii">\n'
    )
    for val in data:
        f.write(f"{val:18.10E}\n")
    f.write("</DataArray>\n")


def _write_int_data_array(f, name: str, data: np.ndarray) -> None:
    """Write a VTK DataArray element (Int64 or UInt8)."""
    dtype_name = "UInt8" if data.dtype == np.uint8 else "Int64"
    f.write(f'<DataArray type="{dtype_name}" Name="{name}" format="ascii">\n')
    for val in data:
        f.write(f"{int(val)}\n")
    f.write("</DataArray>\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_time(t: float) -> str:
    """Format a time value for use in file names."""
    if t == int(t):
        return str(int(t))
    return f"{t:g}"
