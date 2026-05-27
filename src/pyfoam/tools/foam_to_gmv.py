"""
foamToGMV — export an OpenFOAM case to GMV (General Mesh Viewer) format.

Mirrors OpenFOAM's ``foamToGMV`` utility.  Writes ASCII GMV files that
can be read by the GMV visualisation tool.

GMV file format overview:

- ``gmvinput ascii`` header
- ``nodev <n>`` block with node coordinates (1-based in original GMV,
  but we use 0-based indices consistent with internal mesh numbering)
- ``cells <n>`` block with cell connectivity
- Variable blocks (``velocity``, ``p``) for per-node field data
- ``endgmv`` terminator

Usage::

    from pyfoam.tools.foam_to_gmv import foam_to_gmv

    foam_to_gmv(case_path, mesh=mesh, fields=fields, time_range=[0, 1])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["foam_to_gmv"]

# GMV cell type strings
_GMV_HEX = "hex"
_GMV_TET = "tet"
_GMV_PYR = "pyramid"
_GMV_PRISM = "prism"
_GMV_GENERAL = "general"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_gmv(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Export an OpenFOAM case to GMV ASCII format.

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
        ``{field_name: numpy_array}`` of per-cell or per-node field
        values to export.  Scalar arrays have shape ``(n,)``, vector
        arrays ``(n, 3)``.
    output_dir : str or Path, optional
        Directory for GMV output.  Defaults to ``<case_path>/GMV``.

    Returns
    -------
    Path
        Path to the output directory containing ``.gmv`` files.

    Raises
    ------
    FileNotFoundError
        If *case_path* does not exist.
    ValueError
        If no mesh is available (neither provided nor on-disk).
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    # Determine output directory
    if output_dir is None:
        gmv_dir = case_dir / "GMV"
    else:
        gmv_dir = Path(output_dir)
    os.makedirs(gmv_dir, exist_ok=True)

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
    cell_verts, gmv_cell_types = _compute_cell_vertices_and_types(mesh)

    # Write one GMV file per time step
    for t in times:
        t_name = _format_time(t)
        gmv_path = gmv_dir / f"{t_name}.gmv"
        _write_gmv(gmv_path, mesh, cell_verts, gmv_cell_types, fields)

    return gmv_dir


# ---------------------------------------------------------------------------
# Cell-vertex connectivity and GMV types
# ---------------------------------------------------------------------------


def _compute_cell_vertices_and_types(mesh: "FvMesh"):
    """Build cell-to-unique-vertices mapping and GMV cell type strings.

    Returns
    -------
    cell_verts : list[list[int]]
        Point indices for each cell (sorted, unique).
    gmv_types : list[str]
        GMV cell type string for each cell.
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

    # Classify GMV cell type by unique node count
    gmv_types = []
    for verts in cell_verts:
        nn = len(verts)
        if nn == 8:
            gmv_types.append(_GMV_HEX)
        elif nn == 4:
            gmv_types.append(_GMV_TET)
        elif nn == 5:
            gmv_types.append(_GMV_PYR)
        elif nn == 6:
            gmv_types.append(_GMV_PRISM)
        else:
            gmv_types.append(_GMV_GENERAL)

    return cell_verts, gmv_types


# ---------------------------------------------------------------------------
# GMV file writing
# ---------------------------------------------------------------------------


def _write_gmv(
    path: Path,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
    gmv_types: list[str],
    fields: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write a complete GMV ASCII file."""
    pts = mesh.points.detach().cpu().numpy()
    n_points = pts.shape[0]
    n_cells = len(cell_verts)

    with open(path, "w", encoding="utf-8") as f:
        # Header
        f.write("gmvinput ascii\n")

        # Nodes
        f.write(f"nodev {n_points}\n")
        for i in range(n_points):
            f.write(f" {pts[i, 0]:18.10E} {pts[i, 1]:18.10E} {pts[i, 2]:18.10E}\n")

        # Cells
        f.write(f"cells {n_cells}\n")
        for ci, (verts, ctype) in enumerate(zip(cell_verts, gmv_types)):
            if ctype == _GMV_GENERAL:
                # General polyhedron: write as "general <n_faces>"
                # We approximate by writing node count and all vertices
                n_verts = len(verts)
                f.write(f"general {n_verts}\n")
                f.write(f" {n_verts}\n")
                line = "".join(f" {v + 1}" for v in verts)
                f.write(line + "\n")
            else:
                f.write(f"{ctype} {len(verts)}\n")
                line = "".join(f" {v + 1}" for v in verts)
                f.write(line + "\n")

        # Fields
        if fields:
            _write_fields(f, fields, n_points, n_cells)

        # Terminator
        f.write("endgmv\n")


def _write_fields(
    f,
    fields: Dict[str, np.ndarray],
    n_points: int,
    n_cells: int,
) -> None:
    """Write variable data blocks to the GMV file.

    Fields are classified as per-node (length == n_points) or
    per-cell (length == n_cells).  Vector fields (ndim == 2, shape[1] == 3)
    are written with ``velocity`` or ``<name>`` as GMV variable names.
    """
    for name, data in fields.items():
        is_vector = data.ndim == 2 and data.shape[1] == 3
        is_nodal = data.shape[0] == n_points

        if is_vector:
            # GMV velocity block (always per-node in GMV)
            f.write(f"variable\n")
            f.write(f" {name} 1\n")
            if is_nodal:
                f.write(" 0\n")  # 0 = node data
                for i in range(n_points):
                    f.write(f" {data[i, 0]:18.10E}\n")
                f.write(" 0\n")
                for i in range(n_points):
                    f.write(f" {data[i, 1]:18.10E}\n")
                f.write(" 0\n")
                for i in range(n_points):
                    f.write(f" {data[i, 2]:18.10E}\n")
            else:
                # Cell-based vector: write as 3 separate scalar components
                f.write(" 1\n")  # 1 = cell data
                for i in range(n_cells):
                    f.write(f" {data[i, 0]:18.10E}\n")
                f.write(" 1\n")
                for i in range(n_cells):
                    f.write(f" {data[i, 1]:18.10E}\n")
                f.write(" 1\n")
                for i in range(n_cells):
                    f.write(f" {data[i, 2]:18.10E}\n")
            f.write(f"endvars\n")
        else:
            # Scalar field
            f.write(f"variable\n")
            f.write(f" {name} 1\n")
            if is_nodal:
                f.write(" 0\n")  # 0 = node data
                for i in range(n_points):
                    f.write(f" {data[i]:18.10E}\n")
            else:
                f.write(" 1\n")  # 1 = cell data
                for i in range(n_cells):
                    f.write(f" {data[i]:18.10E}\n")
            f.write(f"endvars\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_time(t: float) -> str:
    """Format a time value for use in file names."""
    if t == int(t):
        return str(int(t))
    return f"{t:g}"
