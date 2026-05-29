"""
foamToFluentMesh — export OpenFOAM polyMesh to Fluent ASCII mesh format.

Reads the OpenFOAM polyMesh files directly from disk and writes a
Fluent ASCII mesh (``.msh``) file.

This tool differs from :func:`pyfoam.tools.foam_to_fluent.foam_to_fluent`
in that it reads mesh data from the polyMesh directory on disk rather than
requiring a pre-loaded ``FvMesh`` object.

Fluent ASCII mesh format (``.msh``) overview::

    (0 "comment")                    — comment header
    (2 dimensions)                   — dimensionality
    (10 zone (first last type) ...)  — node zone
    (12 zone (first last type) ...)  — cell zone
    (13 zone (first last type) ...)  — face zone
    (45 (zone name type)())          — zone labels

Zone types: 0=interior, 3=wall, 4=pressure-inlet, 5=pressure-outlet,
            7=symmetry, 2=interior (cells), 24=axis

Usage::

    from pyfoam.tools.foam_to_fluent_mesh import foam_to_fluent_mesh

    foam_to_fluent_mesh(case_dir, output_dir)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

import numpy as np

from pyfoam.io.mesh_io import BoundaryPatch, MeshData, read_mesh

__all__ = ["foam_to_fluent_mesh"]

# Fluent zone types
_FL_INTERIOR = 2
_FL_WALL = 3
_FL_PRESSURE_INLET = 4
_FL_PRESSURE_OUTLET = 5
_FL_SYMMETRY = 7
_FL_AXIS = 24

# Fluent element types
_FL_MIXED = 0
_FL_TRI = 1
_FL_QUAD = 2
_FL_TET = 3
_FL_HEX = 4
_FL_PYRAMID = 5
_FL_WEDGE = 6


def foam_to_fluent_mesh(
    case_dir: Union[str, Path],
    output_dir: Union[str, Path, None] = None,
) -> Path:
    """Export an OpenFOAM mesh to Fluent ASCII format.

    Reads polyMesh files from disk and writes a ``.msh`` file.

    Parameters
    ----------
    case_dir : str or Path
        Root of the OpenFOAM case directory (must contain
        ``constant/polyMesh``).
    output_dir : str or Path, optional
        Directory for Fluent output.  Defaults to ``<case_dir>/Fluent``.

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
        fluent_dir = case_path / "Fluent"
    else:
        fluent_dir = Path(output_dir)
    os.makedirs(fluent_dir, exist_ok=True)

    # Read mesh from disk
    mesh_data = read_mesh(poly_mesh_dir)

    # Write .msh file
    msh_path = fluent_dir / "mesh.msh"
    _write_msh(msh_path, mesh_data)

    return fluent_dir


# ---------------------------------------------------------------------------
# Fluent mesh file writing
# ---------------------------------------------------------------------------


def _write_msh(path: Path, mesh_data: MeshData) -> None:
    """Write a Fluent ASCII mesh (.msh) file."""
    points = mesh_data.points.detach().cpu().numpy()
    faces = mesh_data.faces
    owner = mesh_data.owner.detach().cpu().numpy()
    neighbour = mesh_data.neighbour.detach().cpu().numpy()
    n_internal = mesh_data.n_internal_faces
    n_cells = mesh_data.n_cells

    n_points = points.shape[0]

    # Build cell types
    cell_types = _compute_cell_types(mesh_data)

    # Build zone info
    zone_info = _build_zone_info(mesh_data, n_internal)

    with open(path, "w", encoding="utf-8") as f:
        # Header
        f.write('(0 "OpenFOAM to Fluent export")\n\n')

        # Dimensions
        f.write('(2 3)\n\n')

        # Nodes section
        f.write(f'(10 (0 1 {n_points:x} 0 3))\n')
        f.write(f'(10 (1 1 {n_points:x} 1))\n')
        for i in range(n_points):
            f.write(
                f' {points[i, 0]:18.10E} {points[i, 1]:18.10E}'
                f' {points[i, 2]:18.10E}\n'
            )
        f.write('\n')

        # Cells section
        f.write(f'(12 (0 1 {n_cells:x} 0 0))\n')
        f.write(f'(12 (2 1 {n_cells:x} 1))\n')
        for ct in cell_types:
            f.write(f' {ct}\n')
        f.write('\n')

        # Faces section
        n_all_faces = len(faces)
        f.write(f'(13 (0 1 {n_all_faces:x} 0 0))\n')

        # Internal faces (zone 3, type interior=2)
        if n_internal > 0:
            f.write(f'(13 (3 1 {n_internal:x} {_FL_INTERIOR}))\n')
            for fi in range(n_internal):
                face_nodes = faces[fi]
                if hasattr(face_nodes, 'tolist'):
                    face_nodes = face_nodes.tolist()
                nn_str = " ".join(str(n + 1) for n in face_nodes)  # 1-based
                c_own = int(owner[fi]) + 1
                c_neigh = int(neighbour[fi]) + 1
                f.write(f' {nn_str} {c_own:x} {c_neigh:x}\n')
            f.write('\n')

        # Boundary face zones
        zone_offset = 4
        for zi, patch in enumerate(mesh_data.boundary):
            zid = zone_offset + zi
            fluent_type = _map_patch_type(patch.patch_type)
            first_f = patch.start_face + 1  # 1-based
            last_f = patch.start_face + patch.n_faces

            f.write(f'(13 ({zid:x} {first_f:x} {last_f:x} {fluent_type:x}))\n')

            for fi in range(patch.start_face, patch.start_face + patch.n_faces):
                face_nodes = faces[fi]
                if hasattr(face_nodes, 'tolist'):
                    face_nodes = face_nodes.tolist()
                nn_str = " ".join(str(n + 1) for n in face_nodes)
                c_own = int(owner[fi]) + 1
                f.write(f' {nn_str} {c_own:x} 0\n')
            f.write('\n')

        # Zone labels
        f.write('(45 (2 fluid fluid)())\n')
        for zi, patch in enumerate(mesh_data.boundary):
            zid = zone_offset + zi
            f.write(f'(45 ({zid} {patch.name} {patch.patch_type})())\n')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_cell_types(mesh_data: MeshData) -> list[int]:
    """Compute Fluent cell type for each cell."""
    faces = mesh_data.faces
    owner = mesh_data.owner.detach().cpu().numpy()
    neighbour = mesh_data.neighbour.detach().cpu().numpy()
    n_internal = mesh_data.n_internal_faces
    n_cells = mesh_data.n_cells

    cell_to_verts: list[set[int]] = [set() for _ in range(n_cells)]

    for fi, face in enumerate(faces):
        face_nodes = face.tolist() if hasattr(face, 'tolist') else list(face)
        c_own = int(owner[fi])
        cell_to_verts[c_own].update(face_nodes)

        if fi < n_internal:
            c_nbr = int(neighbour[fi])
            cell_to_verts[c_nbr].update(face_nodes)

    cell_types: list[int] = []
    for verts in cell_to_verts:
        nn = len(verts)
        if nn == 4:
            cell_types.append(_FL_TET)
        elif nn == 5:
            cell_types.append(_FL_PYRAMID)
        elif nn == 6:
            cell_types.append(_FL_WEDGE)
        elif nn == 8:
            cell_types.append(_FL_HEX)
        else:
            cell_types.append(_FL_MIXED)

    return cell_types


def _build_zone_info(
    mesh_data: MeshData, n_internal: int,
) -> list[dict]:
    """Build zone metadata list."""
    zone_info: list[dict] = []

    # Interior zone
    zone_info.append({
        "id": 3,
        "type": "interior",
        "fluent_type": _FL_INTERIOR,
        "name": "interior",
        "first_face": 1,
        "last_face": n_internal,
    })

    # Boundary zones
    for zi, patch in enumerate(mesh_data.boundary):
        zone_info.append({
            "id": 4 + zi,
            "type": patch.patch_type,
            "fluent_type": _map_patch_type(patch.patch_type),
            "name": patch.name,
            "first_face": patch.start_face + 1,
            "last_face": patch.start_face + patch.n_faces,
        })

    return zone_info


def _map_patch_type(patch_type: str) -> int:
    """Map OpenFOAM patch type to Fluent zone type."""
    pt = patch_type.lower()
    if pt == "wall":
        return _FL_WALL
    elif pt in ("pressureinlet", "inlet"):
        return _FL_PRESSURE_INLET
    elif pt in ("pressureoutlet", "outlet"):
        return _FL_PRESSURE_OUTLET
    elif pt == "symmetry":
        return _FL_SYMMETRY
    elif pt == "interior":
        return _FL_INTERIOR
    elif pt == "axis":
        return _FL_AXIS
    else:
        return _FL_WALL
