"""
foamToStar — export an OpenFOAM case to Star-CD format.

Mirrors the functionality of converting OpenFOAM mesh and field data
to CD-adapco Star-CD's ``.vrt``, ``.cel``, and ``.bnd`` mesh format
and ``.pst`` field format.

Star-CD mesh file overview:

- ``<name>.vrt`` — Vertex (node) coordinates file
  - One line per vertex: ``x  y  z``
- ``<name>.cel`` — Cell connectivity file
  - One line per cell: ``type  v1  v2  v3  v4  [v5  v6  v7  v8]``
  - Cell types: 1=hex, 2=prism/wedge, 3=tet, 4=pyramid
- ``<name>.bnd`` — Boundary face connectivity file
  - One line per face: ``type  bc_type  v1  v2  [v3  v4]  region_id``
- ``<name>.pst`` — Solution field data (per-cell values)

Usage::

    from pyfoam.tools.foam_to_star import foam_to_star

    foam_to_star(case_path, output_path, mesh=mesh, fields=fields)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["foam_to_star"]

# Star-CD cell type constants
_STAR_HEX = 1
_STAR_WEDGE = 2
_STAR_TET = 3
_STAR_PYRAMID = 4

# Star-CD boundary type codes
_STAR_WALL = 1
_STAR_INLET = 2
_STAR_OUTLET = 3
_STAR_SYMMETRY = 4
_STAR_PERIODIC = 5
_STAR_INTERIOR = 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_star(
    case_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """Export an OpenFOAM case to Star-CD format.

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    output_path : str or Path, optional
        Directory for Star-CD output.  Defaults to ``<case_path>/StarCD``.
    time_range : sequence of float, optional
        Subset of time values to export.  ``None`` exports all available.
    mesh : FvMesh, optional
        Pre-loaded mesh.  Required for connectivity and geometry.
    fields : dict, optional
        ``{field_name: numpy_array}`` of per-cell field values to export.

    Returns
    -------
    Path
        Path to the output directory containing ``.vrt``, ``.cel``,
        ``.bnd``, and (optionally) ``.pst`` files.

    Raises
    ------
    FileNotFoundError
        If *case_path* does not exist.
    ValueError
        If no mesh is provided.
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    if mesh is None:
        raise ValueError("No mesh provided.  Pass a mesh object directly.")

    # Determine output directory
    if output_path is None:
        star_dir = case_dir / "StarCD"
    else:
        star_dir = Path(output_path)
    os.makedirs(star_dir, exist_ok=True)

    # Extract mesh data
    points = mesh.points.detach().cpu().numpy()
    faces = mesh.faces
    owner = mesh.owner.detach().cpu().numpy()
    neighbour = mesh.neighbour.detach().cpu().numpy()
    n_internal = mesh.n_internal_faces
    n_cells = mesh.n_cells

    # --- Write .vrt file (vertices) ---
    vrt_path = star_dir / "mesh.vrt"
    _write_vrt(vrt_path, points)

    # --- Write .cel file (cell connectivity) ---
    cel_path = star_dir / "mesh.cel"
    cell_types = _compute_cell_types(mesh)
    cell_verts = _compute_cell_vertices(mesh)
    _write_cel(cel_path, cell_types, cell_verts)

    # --- Write .bnd file (boundary faces) ---
    bnd_path = star_dir / "mesh.bnd"
    _write_bnd(bnd_path, mesh)

    # --- Write .pst file (field data) ---
    if fields is not None:
        pst_path = star_dir / "solution.pst"
        _write_pst(pst_path, fields, n_cells)

    return star_dir


# ---------------------------------------------------------------------------
# Cell type and vertex computation
# ---------------------------------------------------------------------------


def _compute_cell_types(mesh: "FvMesh") -> list[int]:
    """Compute Star-CD cell type for each cell.

    Returns
    -------
    list of int
        Star-CD type code per cell (1=hex, 2=wedge, 3=tet, 4=pyramid).
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

    cell_types = []
    for verts in cell_to_verts:
        nn = len(verts)
        if nn == 4:
            cell_types.append(_STAR_TET)
        elif nn == 5:
            cell_types.append(_STAR_PYRAMID)
        elif nn == 6:
            cell_types.append(_STAR_WEDGE)
        elif nn == 8:
            cell_types.append(_STAR_HEX)
        else:
            cell_types.append(_STAR_HEX)  # fallback

    return cell_types


def _compute_cell_vertices(mesh: "FvMesh") -> list[list[int]]:
    """Build cell-to-unique-vertices mapping (sorted).

    Returns
    -------
    list of list of int
        Point indices for each cell (sorted, unique, 1-based).
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

    # Star-CD uses 1-based node indices
    return [sorted(v + 1 for v in verts) for verts in cell_to_verts]


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------


def _write_vrt(path: Path, points: np.ndarray) -> None:
    """Write Star-CD vertex (.vrt) file.

    Format: one vertex per line, ``x  y  z``.
    """
    with open(path, "w", encoding="utf-8") as f:
        for i in range(points.shape[0]):
            f.write(f"  {points[i, 0]:18.10E}  {points[i, 1]:18.10E}  {points[i, 2]:18.10E}\n")


def _write_cel(path: Path, cell_types: list[int], cell_verts: list[list[int]]) -> None:
    """Write Star-CD cell (.cel) file.

    Format: one cell per line, ``type  v1  v2  ...  vN``.
    """
    with open(path, "w", encoding="utf-8") as f:
        for ctype, verts in zip(cell_types, cell_verts):
            verts_str = "  ".join(f"{v:d}" for v in verts)
            f.write(f"  {ctype:d}  {verts_str}\n")


def _write_bnd(path: Path, mesh: "FvMesh") -> None:
    """Write Star-CD boundary (.bnd) file.

    Format: one face per line,
    ``bc_type  user_type  v1  v2  [v3  v4]  region_id``.
    """
    faces = mesh.faces
    owner = mesh.owner.detach().cpu().numpy()
    region_id = 0

    with open(path, "w", encoding="utf-8") as f:
        for patch_info in mesh.boundary:
            region_id += 1
            pname = patch_info["name"]
            pstart = patch_info["startFace"]
            n_faces_patch = patch_info["nFaces"]
            ptype = patch_info.get("type", "wall")

            star_bc_type = _map_patch_type(ptype)

            for fi in range(pstart, pstart + n_faces_patch):
                face_nodes = faces[fi].detach().cpu().numpy().tolist()
                # Star-CD uses 1-based node indices
                nodes_1based = [n + 1 for n in face_nodes]
                n_verts = len(nodes_1based)
                c_own = int(owner[fi]) + 1  # 1-based cell index

                verts_str = "  ".join(f"{v:d}" for v in nodes_1based)
                f.write(f"  {star_bc_type:d}  {n_verts:d}  {verts_str}  {region_id:d}\n")


def _write_pst(path: Path, fields: Dict[str, np.ndarray], n_cells: int) -> None:
    """Write Star-CD solution (.pst) file.

    Format:
    - Header line: field names
    - Data lines: one per cell, space-separated values
    """
    with open(path, "w", encoding="utf-8") as f:
        # Write header
        field_names = list(fields.keys())
        header = "  ".join(field_names)
        f.write(f"  {header}\n")

        # Write data
        for ic in range(n_cells):
            values = []
            for name in field_names:
                data = fields[name]
                if data.ndim == 1:
                    values.append(f"{data[ic]:18.10E}")
                elif data.ndim == 2 and data.shape[1] == 3:
                    # Vector: write magnitude
                    mag = np.sqrt(data[ic, 0] ** 2 + data[ic, 1] ** 2 + data[ic, 2] ** 2)
                    values.append(f"{mag:18.10E}")
                elif data.ndim == 2:
                    # Tensor or multi-component: write all components
                    for c in range(data.shape[1]):
                        values.append(f"{data[ic, c]:18.10E}")
            f.write(f"  {'  '.join(values)}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _map_patch_type(patch_type: str) -> int:
    """Map OpenFOAM patch type to Star-CD boundary type code."""
    pt = patch_type.lower()
    if pt == "wall":
        return _STAR_WALL
    elif pt in ("inlet", "pressureinlet"):
        return _STAR_INLET
    elif pt in ("outlet", "pressureoutlet"):
        return _STAR_OUTLET
    elif pt == "symmetry":
        return _STAR_SYMMETRY
    elif pt in ("cyclic", "periodic"):
        return _STAR_PERIODIC
    elif pt == "interior":
        return _STAR_INTERIOR
    else:
        return _STAR_WALL  # default
