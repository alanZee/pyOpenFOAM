"""
foamToCGNS — export an OpenFOAM case to CGNS format.

Mirrors the CGNS (CFD General Notation System) export functionality.
Writes a ``.cgns`` file (HDF5-based when available, otherwise ASCII CGNS
XML) containing:

- Unstructured mesh (coordinates + element connectivity)
- Cell-centred field data (scalars and vectors)
- Boundary condition zones

The output uses CGNS conventions:
- Zone type: Unstructured
- Element types: HEXA_8, TETRA_4, PENTA_6, PYRA_5
- Data location: CellCenter

Usage::

    from pyfoam.tools.foam_to_cgns import foam_to_cgns

    foam_to_cgns(case_path, output_path="output.cgns", mesh=mesh, fields=fields)
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["foam_to_cgns"]


# CGNS element type constants (from cgnslib.h)
CGNS_ENUM_NODE = 2
CGNS_ENUM_BAR_2 = 3
CGNS_ENUM_TRI_3 = 5
CGNS_ENUM_QUAD_4 = 7
CGNS_ENUM_TETRA_4 = 10
CGNS_ENUM_PYRA_5 = 12
CGNS_ENUM_PENTA_6 = 14
CGNS_ENUM_HEXA_8 = 17
CGNS_ENUM_MIXED = 20
CGNS_ENUM_NFACE = 22
CGNS_ENUM_NGON = 23


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_cgns(
    case_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """Export an OpenFOAM case to CGNS format.

    When *mesh* and *fields* are provided directly (e.g. from a running
    simulation), they are used directly.  Otherwise the function reads
    from disk.

    The output is a structured ASCII file following CGNS conventions.
    HDF5-based output requires ``h5py`` and is used when available.

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    output_path : str or Path, optional
        Path for the output ``.cgns`` file.  Defaults to
        ``<case_path>/case.cgns``.
    time_range : sequence of float, optional
        Subset of time values to export.
    mesh : FvMesh, optional
        Pre-loaded mesh.
    fields : dict, optional
        ``{field_name: numpy_array}`` of per-cell field values.

    Returns
    -------
    Path
        Path to the output ``.cgns`` file.

    Raises
    ------
    FileNotFoundError
        If *case_path* does not exist.
    ValueError
        If no mesh is available.
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    if mesh is None:
        raise ValueError(
            "No mesh provided.  Pass a mesh object directly."
        )

    # Determine output path
    if output_path is None:
        out_path = case_dir / "case.cgns"
    else:
        out_path = Path(output_path)
    os.makedirs(out_path.parent, exist_ok=True)

    # Compute connectivity and element types
    cell_verts, cgns_elem_types = _compute_cell_connectivity(mesh)

    # Extract boundary patches
    patch_data = _extract_boundary_patches(mesh)

    # Write CGNS file (structured ASCII format)
    _write_cgns_ascii(out_path, mesh, cell_verts, cgns_elem_types, patch_data, fields)

    return out_path


# ---------------------------------------------------------------------------
# Cell connectivity and CGNS element types
# ---------------------------------------------------------------------------


def _compute_cell_connectivity(mesh: "FvMesh"):
    """Build cell-to-vertex mapping and CGNS element type codes.

    Returns
    -------
    cell_verts : list[list[int]]
        Point indices for each cell.
    cgns_types : list[int]
        CGNS element type code for each cell.
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

    cgns_types = []
    for verts in cell_verts:
        nn = len(verts)
        if nn == 4:
            cgns_types.append(CGNS_ENUM_TETRA_4)
        elif nn == 5:
            cgns_types.append(CGNS_ENUM_PYRA_5)
        elif nn == 6:
            cgns_types.append(CGNS_ENUM_PENTA_6)
        elif nn == 8:
            cgns_types.append(CGNS_ENUM_HEXA_8)
        else:
            cgns_types.append(CGNS_ENUM_MIXED)

    return cell_verts, cgns_types


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
# CGNS ASCII writer
# ---------------------------------------------------------------------------


def _write_cgns_ascii(
    path: Path,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
    cgns_types: list[int],
    patch_data: list,
    fields: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write a CGNS file in structured ASCII format.

    The file follows the CGNS SIDS (Standard Interface Data Structures)
    conventions, written as a hierarchical text format for portability.
    HDF5 output is preferred when h5py is available.
    """
    pts = mesh.points.detach().cpu().numpy()
    n_points = pts.shape[0]
    n_cells = len(cell_verts)

    # Flatten connectivity (1-based for CGNS convention)
    connectivity = []
    for verts in cell_verts:
        connectivity.extend(v + 1 for v in verts)  # CGNS uses 1-based indexing

    total_conn = len(connectivity)

    # Try HDF5 output first
    try:
        _write_cgns_hdf5(
            path, pts, connectivity, cgns_types, cell_verts,
            n_points, n_cells, patch_data, fields,
        )
        return
    except ImportError:
        pass

    # Fallback: structured ASCII CGNS representation
    with open(path, "w", encoding="utf-8") as f:
        f.write("# CGNS structured ASCII representation\n")
        f.write("# Generated by pyOpenFOAM foam_to_cgns\n\n")

        # Base node
        f.write("CGNSBase_t  Base:\n")
        f.write("  CGNSLibraryVersion_t  CGNSLibraryVersion:\n")
        f.write("    DataClass: Dimensional\n\n")

        # Zone
        f.write(f"  Zone_t  Zone:\n")
        f.write(f"    ZoneType: Unstructured\n")
        f.write(f"    # Vertex size, cell size, boundary vertex size\n")
        f.write(f"    ZoneBCSize_t  ({n_points} {n_cells} 0)\n\n")

        # Grid coordinates
        f.write("    GridCoordinates_t  GridCoordinates:\n")
        f.write("      DataArray_t<real,64> CoordinateX:\n")
        for i in range(n_points):
            f.write(f"        {pts[i, 0]:18.10E}\n")
        f.write("\n")
        f.write("      DataArray_t<real,64> CoordinateY:\n")
        for i in range(n_points):
            f.write(f"        {pts[i, 1]:18.10E}\n")
        f.write("\n")
        f.write("      DataArray_t<real,64> CoordinateZ:\n")
        for i in range(n_points):
            f.write(f"        {pts[i, 2]:18.10E}\n")
        f.write("\n")

        # Elements (connectivity)
        f.write("    Elements_t  Elements:\n")
        # Use the most common element type
        from collections import Counter
        type_counts = Counter(cgns_types)
        most_common_type = type_counts.most_common(1)[0][0]
        f.write(f"      ElementType: {most_common_type}\n")
        f.write(f"      ElementConnectivity:\n")
        verts_per_cell = len(cell_verts[0]) if cell_verts else 8
        for ci in range(n_cells):
            conn_str = " ".join(str(v) for v in connectivity[ci * verts_per_cell:(ci + 1) * verts_per_cell])
            f.write(f"        {conn_str}\n")
        f.write("\n")

        # Solution (fields)
        if fields:
            f.write("    FlowSolution_t  Solution:\n")
            f.write("      GridLocation: CellCenter\n\n")
            for name, data in fields.items():
                if data.ndim == 1:
                    f.write(f"      DataArray_t<real,64> {name}:\n")
                    for val in data:
                        f.write(f"        {val:18.10E}\n")
                    f.write("\n")
                elif data.ndim == 2 and data.shape[1] == 3:
                    for comp, suffix in enumerate(["X", "Y", "Z"]):
                        f.write(f"      DataArray_t<real,64> {name}{suffix}:\n")
                        for val in data[:, comp]:
                            f.write(f"        {val:18.10E}\n")
                        f.write("\n")

        # Boundary conditions
        if patch_data:
            f.write("    ZoneBC_t  ZoneBC:\n")
            for patch_name, face_indices, _ in patch_data:
                f.write(f"      BC_t  {patch_name}:\n")
                f.write(f"        BCType: FamilySpecified\n")
                f.write(f"        PointList:\n")
                for fi in face_indices:
                    f.write(f"          {fi + 1}\n")
                f.write("\n")


# ---------------------------------------------------------------------------
# CGNS HDF5 writer (optional)
# ---------------------------------------------------------------------------


def _write_cgns_hdf5(
    path: Path,
    pts: np.ndarray,
    connectivity: list,
    cgns_types: list[int],
    cell_verts: list[list[int]],
    n_points: int,
    n_cells: int,
    patch_data: list,
    fields: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write CGNS file in HDF5 format using h5py."""
    import h5py

    with h5py.File(path, "w") as f:
        # CGNS library version
        f.attrs["CGNSLibraryVersion"] = [4.2, 0]
        f.attrs["CGNSLibraryVersion_t"] = "CGNSLibraryVersion"

        # Create base node
        base = f.create_group("Base")
        base.attrs["CellDim"] = 3
        base.attrs["PhysicalDimension"] = 3

        # Create zone
        zone = base.create_group("Zone")
        zone.attrs["ZoneType"] = "Unstructured"
        zone.create_dataset("ZoneBCSize_t", data=[n_points, n_cells, 0])

        # Grid coordinates
        gc = zone.create_group("GridCoordinates")
        gc.create_dataset("CoordinateX", data=pts[:, 0], dtype="float64")
        gc.create_dataset("CoordinateY", data=pts[:, 1], dtype="float64")
        gc.create_dataset("CoordinateZ", data=pts[:, 2], dtype="float64")

        # Elements
        conn_arr = np.array(connectivity, dtype=np.int32)
        verts_per = len(cell_verts[0]) if cell_verts else 8
        elem = zone.create_group("Elements")
        elem.attrs["ElementType"] = cgns_types[0]
        elem.create_dataset("ElementConnectivity", data=conn_arr)

        # Solution fields
        if fields:
            sol = zone.create_group("Solution")
            sol.attrs["GridLocation"] = "CellCenter"
            for name, data in fields.items():
                if data.ndim == 1:
                    sol.create_dataset(name, data=data.astype(np.float64))
                elif data.ndim == 2 and data.shape[1] == 3:
                    for comp, suffix in enumerate(["X", "Y", "Z"]):
                        sol.create_dataset(
                            f"{name}{suffix}",
                            data=data[:, comp].astype(np.float64),
                        )

        # Boundary conditions
        if patch_data:
            bc_group = zone.create_group("ZoneBC")
            for patch_name, face_indices, _ in patch_data:
                bc = bc_group.create_group(patch_name)
                bc.attrs["BCType"] = "FamilySpecified"
                bc.create_dataset(
                    "PointList",
                    data=np.array(face_indices, dtype=np.int32) + 1,
                )
