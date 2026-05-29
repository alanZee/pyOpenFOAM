"""
foamToStarMesh — export OpenFOAM polyMesh to Star-CD format.

Reads the OpenFOAM polyMesh files directly from disk and writes
Star-CD mesh files (``.vrt``, ``.cel``, ``.bnd``).

This tool differs from :func:`pyfoam.tools.foam_to_star.foam_to_star` in
that it reads mesh data from the polyMesh directory on disk rather than
requiring a pre-loaded ``FvMesh`` object.

Star-CD file overview:

- ``<name>.vrt`` — Vertex coordinates (``x  y  z`` per line)
- ``<name>.cel`` — Cell connectivity (``type  v1..v8`` per line)
- ``<name>.bnd`` — Boundary faces (``type  bc_type  v1..v4  region_id``)

Star-CD cell types: 1=hex, 2=prism/wedge, 3=tet, 4=pyramid
Star-CD boundary types: 0=interior, 1=wall, 2=inlet, 3=outlet, 4=symmetry, 5=periodic

Usage::

    from pyfoam.tools.foam_to_star_mesh import foam_to_star_mesh

    foam_to_star_mesh(case_dir, output_dir)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from pyfoam.io.mesh_io import BoundaryPatch, MeshData, read_mesh

__all__ = ["foam_to_star_mesh"]

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


def foam_to_star_mesh(
    case_dir: Union[str, Path],
    output_dir: Union[str, Path, None] = None,
) -> Path:
    """Export an OpenFOAM mesh to Star-CD format.

    Reads polyMesh files from disk and writes ``.vrt``, ``.cel``, ``.bnd``
    files in Star-CD format.

    Parameters
    ----------
    case_dir : str or Path
        Root of the OpenFOAM case directory (must contain
        ``constant/polyMesh``).
    output_dir : str or Path, optional
        Directory for Star-CD output.  Defaults to ``<case_dir>/StarCD``.

    Returns
    -------
    Path
        Path to the output directory.

    Raises
    ------
    FileNotFoundError
        If the polyMesh directory does not exist.
    """
    case_path = Path(case_dir).resolve()
    poly_mesh_dir = case_path / "constant" / "polyMesh"

    if not poly_mesh_dir.is_dir():
        raise FileNotFoundError(f"polyMesh directory not found: {poly_mesh_dir}")

    # Determine output directory
    if output_dir is None:
        star_dir = case_path / "StarCD"
    else:
        star_dir = Path(output_dir)
    os.makedirs(star_dir, exist_ok=True)

    # Read mesh from disk
    mesh_data = read_mesh(poly_mesh_dir)

    # Extract arrays
    points = mesh_data.points.detach().cpu().numpy()
    faces = mesh_data.faces
    owner = mesh_data.owner.detach().cpu().numpy()
    neighbour = mesh_data.neighbour.detach().cpu().numpy()
    n_internal = mesh_data.n_internal_faces

    # Build cell-to-vertices mapping
    cell_verts, cell_types = _compute_cell_data(
        faces, owner, neighbour, n_internal, mesh_data.n_cells,
    )

    # --- Write .vrt ---
    _write_vrt(star_dir / "mesh.vrt", points)

    # --- Write .cel ---
    _write_cel(star_dir / "mesh.cel", cell_types, cell_verts)

    # --- Write .bnd ---
    _write_bnd(star_dir / "mesh.bnd", faces, owner, mesh_data.boundary, n_internal)

    return star_dir


# ---------------------------------------------------------------------------
# Cell type / vertex computation
# ---------------------------------------------------------------------------


def _compute_cell_data(
    faces: list,
    owner: np.ndarray,
    neighbour: np.ndarray,
    n_internal: int,
    n_cells: int,
) -> tuple[list[list[int]], list[int]]:
    """Compute cell vertices and Star-CD types.

    Returns:
        Tuple of (cell_vertices, cell_types).
    """
    cell_to_verts: list[set[int]] = [set() for _ in range(n_cells)]

    for fi, face in enumerate(faces):
        face_nodes = face.tolist() if hasattr(face, 'tolist') else list(face)
        c_own = int(owner[fi])
        cell_to_verts[c_own].update(face_nodes)

        if fi < n_internal:
            c_nbr = int(neighbour[fi])
            cell_to_verts[c_nbr].update(face_nodes)

    cell_verts: list[list[int]] = []
    cell_types: list[int] = []

    for verts in cell_to_verts:
        nn = len(verts)
        # Star-CD uses 1-based node indices
        cell_verts.append(sorted(v + 1 for v in verts))

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

    return cell_verts, cell_types


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------


def _write_vrt(path: Path, points: np.ndarray) -> None:
    """Write Star-CD vertex (.vrt) file."""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(points.shape[0]):
            f.write(
                f"  {points[i, 0]:18.10E}  {points[i, 1]:18.10E}"
                f"  {points[i, 2]:18.10E}\n"
            )


def _write_cel(
    path: Path, cell_types: list[int], cell_verts: list[list[int]],
) -> None:
    """Write Star-CD cell (.cel) file."""
    with open(path, "w", encoding="utf-8") as f:
        for ctype, verts in zip(cell_types, cell_verts):
            verts_str = "  ".join(f"{v:d}" for v in verts)
            f.write(f"  {ctype:d}  {verts_str}\n")


def _write_bnd(
    path: Path,
    faces: list,
    owner: np.ndarray,
    boundary: list[BoundaryPatch],
    n_internal: int,
) -> None:
    """Write Star-CD boundary (.bnd) file."""
    with open(path, "w", encoding="utf-8") as f:
        region_id = 0
        for patch in boundary:
            region_id += 1
            star_bc_type = _map_patch_type(patch.patch_type)

            for fi in range(patch.start_face, patch.start_face + patch.n_faces):
                face_nodes = faces[fi]
                if hasattr(face_nodes, 'tolist'):
                    face_nodes = face_nodes.tolist()
                nodes_1based = [n + 1 for n in face_nodes]
                n_verts = len(nodes_1based)
                verts_str = "  ".join(f"{v:d}" for v in nodes_1based)
                f.write(
                    f"  {star_bc_type:d}  {n_verts:d}  {verts_str}"
                    f"  {region_id:d}\n"
                )


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
        return _STAR_WALL
