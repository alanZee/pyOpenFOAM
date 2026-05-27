"""
foamToFluent — export an OpenFOAM case to Fluent mesh format.

Mirrors the functionality of converting OpenFOAM mesh and field data
to ANSYS Fluent's ASCII mesh format (``.msh``) and data format (``.dat``).

Fluent ASCII mesh format (``.msh``) overview:

- ``(0 "comment")`` — comment header
- ``(2 dimensions)`` — dimensionality
- ``(10 zone index (first last type) (nodes))`` — node coordinates
- ``(12 zone index (first last type) (cells))`` — cell definitions
- ``(13 zone index (first last type) (faces))`` — face definitions

Fluent data format (``.dat``) contains per-zone field values.

Usage::

    from pyfoam.tools.foam_to_fluent import foam_to_fluent

    foam_to_fluent(case_path, mesh=mesh, fields=fields, time_range=[0, 1])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["foam_to_fluent"]

# Fluent zone types
_FLUID_ZONE = 0
_WALL_ZONE = 3
_PRESSURE_INLET_ZONE = 4
_PRESSURE_OUTLET_ZONE = 5
_SYMMETRY_ZONE = 7
_INTERIOR_ZONE = 2
_AXIS_ZONE = 10

# Fluent element types
_FL_MIXED = 0
_FL_TRI = 1
_FL_QUAD = 2
_FL_TET = 3
_FL_HEX = 4
_FL_PYRAMID = 5
_FL_WEDGE = 6
_FL_POLY = 7


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_fluent(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Export an OpenFOAM case to Fluent ASCII format.

    When *mesh* and *fields* are provided directly, they are used
    directly.  Otherwise the function requires on-disk data or a mesh
    object.

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    time_range : sequence of float, optional
        Subset of time values to export.  ``None`` exports all available
        times.  Only used when *fields* are provided per time step.
    mesh : FvMesh, optional
        Pre-loaded mesh.  Required for connectivity and geometry.
    fields : dict, optional
        ``{field_name: numpy_array}`` of per-cell field values to export.
    output_dir : str or Path, optional
        Directory for Fluent output.  Defaults to ``<case_path>/Fluent``.

    Returns
    -------
    Path
        Path to the output directory.

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

    if output_dir is None:
        fluent_dir = case_dir / "Fluent"
    else:
        fluent_dir = Path(output_dir)
    os.makedirs(fluent_dir, exist_ok=True)

    if mesh is None:
        raise ValueError(
            "No mesh provided.  Pass a mesh object directly."
        )

    # Determine time values
    if time_range is not None:
        times = sorted(float(t) for t in time_range)
    else:
        times = [0.0]

    # Build mesh data
    nodes, cell_data, face_data, zone_info = _extract_fluent_data(mesh)

    # Write .msh file (mesh only, written once)
    msh_path = fluent_dir / "mesh.msh"
    _write_msh(msh_path, mesh, nodes, cell_data, face_data, zone_info)

    # Write .dat files per time step (field data)
    for t in times:
        t_name = _format_time(t)
        dat_path = fluent_dir / f"{t_name}.dat"
        _write_dat(dat_path, mesh, zone_info, fields)

    return fluent_dir


# ---------------------------------------------------------------------------
# Mesh data extraction
# ---------------------------------------------------------------------------


def _extract_fluent_data(mesh: "FvMesh"):
    """Extract mesh data in Fluent format.

    Returns
    -------
    nodes : np.ndarray
        Node coordinates ``(n_points, 3)``.
    cell_data : dict
        Cell zone data.
    face_data : dict
        Face zone data.
    zone_info : list of dict
        Zone metadata.
    """
    points = mesh.points.detach().cpu().numpy()
    faces = mesh.faces
    owner = mesh.owner.detach().cpu().numpy()
    neighbour = mesh.neighbour.detach().cpu().numpy()
    n_internal = mesh.n_internal_faces
    n_cells = mesh.n_cells

    # Build node array
    nodes = points

    # Build cell type data
    cell_types = _compute_cell_types(mesh)

    # Build face data with zone assignment
    face_zones = []
    zone_info_list = []

    # Internal face zone (zone 1)
    internal_faces_list = []
    for fi in range(n_internal):
        face_nodes = faces[fi].detach().cpu().numpy().tolist()
        c_own = int(owner[fi])
        c_neigh = int(neighbour[fi])
        internal_faces_list.append((face_nodes, c_own + 1, c_neigh + 1))

    zone_id = 1
    zone_info_list.append({
        "id": zone_id,
        "type": "interior",
        "fluent_type": _INTERIOR_ZONE,
        "name": "interior",
        "first_face": 1,
        "last_face": len(internal_faces_list),
    })

    # Boundary face zones
    boundary_start = len(internal_faces_list) + 1
    boundary_zones = []
    for patch_info in mesh.boundary:
        zone_id += 1
        pname = patch_info["name"]
        pstart = patch_info["startFace"]
        n_faces_patch = patch_info["nFaces"]

        patch_face_list = []
        for fi in range(pstart, pstart + n_faces_patch):
            face_nodes = faces[fi].detach().cpu().numpy().tolist()
            c_own = int(owner[fi]) + 1  # Fluent uses 1-based
            patch_face_list.append((face_nodes, c_own, 0))  # 0 = no neighbour

        # Map patch type to Fluent zone type
        ptype = patch_info.get("type", "wall")
        fluent_type = _map_patch_type(ptype)

        zone_info_list.append({
            "id": zone_id,
            "type": ptype,
            "fluent_type": fluent_type,
            "name": pname,
            "first_face": boundary_start,
            "last_face": boundary_start + len(patch_face_list) - 1,
        })
        boundary_zones.append(patch_face_list)
        boundary_start += len(patch_face_list)

    face_data = {
        "internal": internal_faces_list,
        "boundary": boundary_zones,
    }

    cell_data = {
        "types": cell_types,
        "n_cells": n_cells,
    }

    return nodes, cell_data, face_data, zone_info_list


def _compute_cell_types(mesh: "FvMesh") -> list[int]:
    """Compute Fluent cell type for each cell."""
    n_cells = mesh.n_cells
    faces = mesh.faces
    owner = mesh.owner.detach().cpu().numpy()
    neighbour = mesh.neighbour.detach().cpu().numpy()
    n_internal = mesh.n_internal_faces

    cell_to_verts: list[set[int]] = [set() for _ in range(n_cells)]
    cell_face_count = [0] * n_cells

    for fi, face in enumerate(faces):
        face_nodes = face.detach().cpu().numpy().tolist()
        c_own = int(owner[fi])
        cell_to_verts[c_own].update(face_nodes)
        cell_face_count[c_own] += 1

        if fi < n_internal:
            c_neigh = int(neighbour[fi])
            cell_to_verts[c_neigh].update(face_nodes)
            cell_face_count[c_neigh] += 1

    cell_types = []
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


def _map_patch_type(patch_type: str) -> int:
    """Map OpenFOAM patch type to Fluent zone type."""
    pt = patch_type.lower()
    if pt == "wall":
        return _WALL_ZONE
    elif pt in ("pressureinlet", "inlet"):
        return _PRESSURE_INLET_ZONE
    elif pt in ("pressureoutlet", "outlet"):
        return _PRESSURE_OUTLET_ZONE
    elif pt == "symmetry":
        return _SYMMETRY_ZONE
    elif pt == "interior":
        return _INTERIOR_ZONE
    elif pt == "axis":
        return _AXIS_ZONE
    else:
        return _WALL_ZONE  # default


# ---------------------------------------------------------------------------
# Fluent mesh file writing (.msh)
# ---------------------------------------------------------------------------


def _write_msh(
    path: Path,
    mesh: "FvMesh",
    nodes: np.ndarray,
    cell_data: dict,
    face_data: dict,
    zone_info: list[dict],
) -> None:
    """Write a Fluent ASCII mesh (.msh) file."""
    n_points = nodes.shape[0]
    n_cells = cell_data["n_cells"]
    cell_types = cell_data["types"]

    with open(path, "w", encoding="utf-8") as f:
        # Header
        f.write('(0 "OpenFOAM to Fluent export")\n\n')

        # Dimensions
        f.write('(2 3)\n\n')

        # Nodes section
        f.write(f'(10 (0 1 {n_points:x} 0 3))\n')
        # Write all nodes in a single zone (zone 0 = all nodes)
        f.write(f'(10 (1 1 {n_points:x} 1))\n')
        for i in range(n_points):
            f.write(f' {nodes[i, 0]:18.10E} {nodes[i, 1]:18.10E} {nodes[i, 2]:18.10E}\n')
        f.write('\n')

        # Cells section
        f.write(f'(12 (0 1 {n_cells:x} 0 0))\n')

        # Single fluid cell zone (zone 2)
        f.write(f'(12 (2 1 {n_cells:x} 1))\n')
        # Fluent expects mixed element type; write element types
        for ct in cell_types:
            f.write(f' {ct}\n')
        f.write('\n')

        # Faces section
        # Internal faces first
        internal_faces = face_data["internal"]
        n_int = len(internal_faces)
        boundary_zones = face_data["boundary"]

        # Total face count header
        n_bnd_total = sum(len(bz) for bz in boundary_zones)
        n_all_faces = n_int + n_bnd_total
        f.write(f'(13 (0 1 {n_all_faces:x} 0 0))\n')

        # Internal faces (zone 3, type = interior=2)
        f.write(f'(13 (3 1 {n_int:x} 2))\n')
        for face_nodes, c1, c2 in internal_faces:
            nn_str = " ".join(str(n + 1) for n in face_nodes)  # 1-based
            f.write(f' {nn_str} {c1:x} {c2:x}\n')
        f.write('\n')

        # Boundary face zones (starting from zone 4)
        zone_offset = 4
        for zi, bz in enumerate(boundary_zones):
            zid = zone_offset + zi
            zinfo = zone_info[zi + 1]  # skip interior zone info
            first_f = zinfo["first_face"]
            last_f = zinfo["last_face"]
            fluent_t = zinfo["fluent_type"]
            f.write(f'(13 ({zid:x} {first_f:x} {last_f:x} {fluent_t:x}))\n')
            for face_nodes, c_own, _ in bz:
                nn_str = " ".join(str(n + 1) for n in face_nodes)
                f.write(f' {nn_str} {c_own:x} 0\n')
            f.write('\n')

        # Zone labels
        f.write('(45 (2 fluid fluid)())\n')
        for zi, zi_info in enumerate(zone_info[1:], start=1):
            zid = zone_offset + zi - 1
            name = zi_info["name"]
            ztype = zi_info["type"]
            f.write(f'(45 ({zid} {name} {ztype})())\n')


# ---------------------------------------------------------------------------
# Fluent data file writing (.dat)
# ---------------------------------------------------------------------------


def _write_dat(
    path: Path,
    mesh: "FvMesh",
    zone_info: list[dict],
    fields: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write a Fluent ASCII data (.dat) file with per-zone field values."""
    if fields is None:
        # Write a minimal data file with no field data
        with open(path, "w", encoding="utf-8") as f:
            f.write('(0 "OpenFOAM to Fluent data export")\n')
        return

    n_cells = mesh.n_cells

    with open(path, "w", encoding="utf-8") as f:
        f.write('(0 "OpenFOAM to Fluent data export")\n\n')

        # Write field data for the cell zone (zone 2)
        # Fluent data format: (30 zone_id (field_name n_comp))
        for field_name, data in fields.items():
            if data.ndim == 1:
                # Scalar field
                f.write(f'(30 2 ({field_name} 1))\n')
                for i in range(min(n_cells, data.shape[0])):
                    f.write(f' {data[i]:18.10E}\n')
                f.write('\n')
            elif data.ndim == 2 and data.shape[1] == 3:
                # Vector field — write as 3 separate components
                for comp, suffix in enumerate(["x", "y", "z"]):
                    fname = f"{field_name}_{suffix}"
                    f.write(f'(30 2 ({fname} 1))\n')
                    for i in range(min(n_cells, data.shape[0])):
                        f.write(f' {data[i, comp]:18.10E}\n')
                    f.write('\n')
            elif data.ndim == 2 and data.shape[1] == 9:
                # Tensor field (symmetric: 6 components)
                fname = f"{field_name}"
                f.write(f'(30 2 ({fname} {data.shape[1]}))\n')
                for i in range(min(n_cells, data.shape[0])):
                    comps = " ".join(f"{v:18.10E}" for v in data[i])
                    f.write(f' {comps}\n')
                f.write('\n')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_time(t: float) -> str:
    """Format a time value for use in file names."""
    if t == int(t):
        return str(int(t))
    return f"{t:g}"
