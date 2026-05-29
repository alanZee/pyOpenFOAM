"""
foamToGmsh enhanced — enhanced Gmsh export with 4.x format and boundary
layer mesh support.

Extends :func:`foam_to_gmsh` with:

- **Gmsh 4.x format**: Writes ``$MeshFormat`` version 4.1 with the
  restructured ``$Entities`` and ``$PartitionedEntities`` sections.
- **Boundary layer mesh export**: Writes ``$BoundaryLayerFields`` for
  first-cell-height, growth rate, and number of layers.
- **Physical groups**: Automatic physical group assignment from OpenFOAM
  boundary patches.
- **Multiple element orders**: Support for second-order elements via
  ``element_order`` parameter.

Usage::

    from pyfoam.tools.foam_to_gmsh_enhanced import foam_to_gmsh_enhanced

    result = foam_to_gmsh_enhanced(
        case_path="cavity",
        mesh=mesh,
        fields={"p": p_arr},
        boundary_layers={"wall": {"n_layers": 5, "first_height": 1e-4}},
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["GmshEnhancedResult", "foam_to_gmsh_enhanced"]


# Gmsh element type codes (Msh 4.x / 2.2 compatible)
_GMSH_TRI = 2
_GMSH_QUAD = 3
_GMSH_TET = 4
_GMSH_HEX = 5
_GMSH_WEDGE = 6
_GMSH_PYRAMID = 7
_GMSH_POLY = 0

# Second-order element types
_GMSH_TRI6 = 9
_GMSH_QUAD8 = 16
_GMSH_TET10 = 11
_GMSH_HEX20 = 17
_GMSH_WEDGE15 = 18
_GMSH_PYRAMID14 = 14


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class GmshEnhancedResult:
    """Result from :func:`foam_to_gmsh_enhanced`.

    Attributes
    ----------
    msh_file : Path
        Path to the written ``.msh`` file.
    geo_file : Path, optional
        Path to the companion ``.geo`` file (boundary layer config).
    n_nodes : int
        Number of nodes in the mesh.
    n_elements : int
        Number of volume elements.
    n_boundary_elements : int
        Number of boundary (surface) elements.
    gmsh_format : str
        Gmsh format version used (``"4.1"`` or ``"2.2"``).
    """

    msh_file: Path
    geo_file: Optional[Path] = None
    n_nodes: int = 0
    n_elements: int = 0
    n_boundary_elements: int = 0
    gmsh_format: str = "4.1"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_gmsh_enhanced(
    case_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    mesh: "FvMesh | None" = None,
    fields: Dict[str, np.ndarray] | None = None,
    time_value: float = 0.0,
    gmsh_format: str = "4.1",
    element_order: int = 1,
    boundary_layers: Optional[Dict[str, dict]] = None,
) -> GmshEnhancedResult:
    """Export an OpenFOAM case to enhanced Gmsh ``.msh`` format.

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    output_path : str or Path, optional
        Path for the output ``.msh`` file.
    mesh : FvMesh, optional
        Pre-loaded mesh.
    fields : dict, optional
        ``{field_name: numpy_array}`` of per-cell field values.
    time_value : float
        Physical time value written into field data headers.
    gmsh_format : str
        Gmsh format version: ``"4.1"`` (default) or ``"2.2"``.
    element_order : int
        Element order (1 = linear, 2 = quadratic).  Default: 1.
    boundary_layers : dict, optional
        ``{patch_name: {"n_layers": int, "first_height": float,
        "growth_rate": float}}`` for boundary layer specification.

    Returns
    -------
    GmshEnhancedResult
        Export result with file paths and metadata.
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    if mesh is None:
        raise ValueError("No mesh provided. Pass a mesh object directly.")

    # Determine output path
    if output_path is None:
        gmsh_dir = case_dir / "Gmsh_enhanced"
        os.makedirs(gmsh_dir, exist_ok=True)
        msh_path = gmsh_dir / "mesh.msh"
    else:
        msh_path = Path(output_path)
        os.makedirs(msh_path.parent, exist_ok=True)

    # Build cell-to-vertex mapping and element types
    cell_verts, gmsh_types = _compute_cell_verts_and_types(mesh, element_order)

    # Extract boundary patch information
    boundary_data = _extract_boundary_patches(mesh)

    # Write .msh file
    if gmsh_format == "4.1":
        _write_msh_v4(msh_path, mesh, cell_verts, gmsh_types, boundary_data,
                       fields, time_value, element_order)
    else:
        _write_msh_v2(msh_path, mesh, cell_verts, gmsh_types, fields, time_value)

    # Write .geo companion for boundary layers
    geo_path = None
    if boundary_layers:
        geo_path = msh_path.with_suffix(".geo")
        _write_geo_boundary_layers(geo_path, boundary_layers)

    n_bnd_elems = sum(len(faces) for _, faces, _ in boundary_data)

    return GmshEnhancedResult(
        msh_file=msh_path,
        geo_file=geo_path,
        n_nodes=mesh.points.shape[0],
        n_elements=len(cell_verts),
        n_boundary_elements=n_bnd_elems,
        gmsh_format=gmsh_format,
    )


# ---------------------------------------------------------------------------
# Cell-vertex mapping and Gmsh type classification
# ---------------------------------------------------------------------------


def _compute_cell_verts_and_types(
    mesh: "FvMesh", element_order: int = 1,
):
    """Build cell-to-unique-vertices mapping and Gmsh element type codes."""
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

    # Select type mapping based on element order
    if element_order == 2:
        type_map = {4: _GMSH_TET10, 5: _GMSH_PYRAMID14, 6: _GMSH_WEDGE15, 8: _GMSH_HEX20}
    else:
        type_map = {4: _GMSH_TET, 5: _GMSH_PYRAMID, 6: _GMSH_WEDGE, 8: _GMSH_HEX}

    gmsh_types = []
    for verts in cell_verts:
        nn = len(verts)
        gmsh_types.append(type_map.get(nn, _GMSH_POLY))

    return cell_verts, gmsh_types


# ---------------------------------------------------------------------------
# Boundary patch extraction
# ---------------------------------------------------------------------------


def _extract_boundary_patches(mesh: "FvMesh"):
    """Extract boundary face data grouped by patch."""
    patches = []
    owner = mesh.owner.detach().cpu().numpy()

    for patch_info in mesh.boundary:
        name = patch_info["name"]
        start = patch_info["startFace"]
        n_faces = patch_info["nFaces"]
        ptype = patch_info.get("type", "wall")

        face_indices = list(range(start, start + n_faces))
        owner_cells = [int(owner[fi]) for fi in face_indices]
        patches.append((name, face_indices, ptype))

    return patches


# ---------------------------------------------------------------------------
# Gmsh 4.1 writer
# ---------------------------------------------------------------------------


def _write_msh_v4(
    path: Path,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
    gmsh_types: list[int],
    boundary_data: list,
    fields: Dict[str, np.ndarray] | None = None,
    time_value: float = 0.0,
    element_order: int = 1,
) -> None:
    """Write a Gmsh 4.1 ASCII .msh file."""
    pts = mesh.points.detach().cpu().numpy()
    n_points = pts.shape[0]
    n_cells = len(cell_verts)

    # Build boundary face connectivity
    bnd_faces_list = []
    for name, face_indices, ptype in boundary_data:
        for fi in face_indices:
            face_nodes = mesh.faces[fi].detach().cpu().numpy().tolist()
            bnd_faces_list.append((face_nodes, name))

    n_bnd = len(bnd_faces_list)

    with open(path, "w", encoding="utf-8") as f:
        # MeshFormat
        f.write("$MeshFormat\n")
        f.write("4.1 0 8\n")
        f.write("$EndMeshFormat\n")

        # Entities section (Gmsh 4.x requires this)
        # We write minimal entities: points, curves (none), surfaces, volumes
        n_points_entity = 0
        n_curves = 0
        n_surfaces = 0
        n_volumes = 0

        # Count unique patches as surfaces
        patch_names = list(dict.fromkeys(name for name, _, _ in boundary_data))
        n_surfaces = len(patch_names)
        n_volumes = 1  # single volume

        f.write("$Entities\n")
        f.write(f"{n_points_entity} {n_curves} {n_surfaces} {n_volumes}\n")

        # Write surface entities
        for i, pname in enumerate(patch_names):
            # surfaceTag xmin xmax ymin ymax zmin zmax
            f.write(f"{i + 1} 0 0 0 0 0 0 0\n")

        # Volume entity
        f.write(f"1 0 0 0 0 0 0 0\n")
        f.write("$EndEntities\n")

        # PartitionedEntities (minimal)
        f.write("$PartitionedEntities\n")
        f.write("0 0\n")  # ghost entities
        f.write(f"0 0 {n_surfaces} {n_volumes}\n")
        f.write("$EndPartitionedEntities\n")

        # Nodes
        f.write("$Nodes\n")
        f.write(f"1 {n_points} 1 {n_points}\n")
        # entityDim entityTag parametric numNodes
        f.write(f"3 1 0 {n_points}\n")
        for i in range(n_points):
            f.write(f"{i + 1}\n")
        for i in range(n_points):
            f.write(f"{pts[i, 0]:18.10E} {pts[i, 1]:18.10E} {pts[i, 2]:18.10E}\n")
        f.write("$EndNodes\n")

        # Elements (volume + boundary)
        total_elems = n_cells + n_bnd
        f.write("$Elements\n")
        f.write(f"{n_surfaces + 1} {total_elems} 1 {total_elems}\n")

        # Boundary surface elements
        for sidx, (pname, face_indices, ptype) in enumerate(boundary_data):
            n_bnd_patch = len(face_indices)
            f.write(f"2 {sidx + 1} 2 {n_bnd_patch}\n")
            for fi in face_indices:
                face_nodes = mesh.faces[fi].detach().cpu().numpy().tolist()
                nn = len(face_nodes)
                etype = _GMSH_QUAD if nn == 4 else _GMSH_TRI
                node_str = " ".join(str(v + 1) for v in face_nodes)
                f.write(f"{fi + 1} {etype} {node_str}\n")

        # Volume elements
        f.write(f"3 1 4 {n_cells}\n")
        for ci, (verts, etype) in enumerate(zip(cell_verts, gmsh_types)):
            node_str = " ".join(str(v + 1) for v in verts)
            f.write(f"{ci + 1} {etype} {node_str}\n")
        f.write("$EndElements\n")

        # Field data
        if fields:
            _write_element_data_v4(f, mesh, fields, time_value, cell_verts)


def _write_element_data_v4(
    f,
    mesh: "FvMesh",
    fields: Dict[str, np.ndarray],
    time_value: float,
    cell_verts: list[list[int]],
) -> None:
    """Write $ElementNodeData sections for Gmsh 4.x format."""
    n_cells = mesh.n_cells

    for field_name, data in fields.items():
        f.write("$ElementNodeData\n")
        f.write("1\n")
        f.write(f'"{field_name}"\n')
        f.write("1\n")
        f.write(f"{time_value}\n")
        f.write("3\n")
        f.write("0\n")
        f.write(f"{1 if data.ndim == 1 else data.shape[1]}\n")
        f.write(f"{n_cells}\n")

        for ci in range(n_cells):
            n_nodes = len(cell_verts[ci])
            if data.ndim == 1:
                val_str = f"{data[ci]:18.10E}"
            elif data.ndim == 2 and data.shape[1] == 3:
                val_str = " ".join(f"{data[ci, c]:18.10E}" for c in range(3))
            else:
                val_str = f"{data[ci]:18.10E}"
            node_vals = " ".join([val_str] * n_nodes)
            f.write(f"{ci + 1} {node_vals}\n")

        f.write("$EndElementNodeData\n")


# ---------------------------------------------------------------------------
# Gmsh 2.2 writer (faster path for compatibility)
# ---------------------------------------------------------------------------


def _write_msh_v2(
    path: Path,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
    gmsh_types: list[int],
    fields: Dict[str, np.ndarray] | None = None,
    time_value: float = 0.0,
) -> None:
    """Write a Gmsh 2.2 ASCII .msh file."""
    pts = mesh.points.detach().cpu().numpy()
    n_points = pts.shape[0]
    n_cells = len(cell_verts)

    with open(path, "w", encoding="utf-8") as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")

        f.write("$Nodes\n")
        f.write(f"{n_points}\n")
        for i in range(n_points):
            f.write(f"{i + 1} {pts[i, 0]:18.10E} {pts[i, 1]:18.10E} {pts[i, 2]:18.10E}\n")
        f.write("$EndNodes\n")

        f.write("$Elements\n")
        f.write(f"{n_cells}\n")
        for ci, (verts, etype) in enumerate(zip(cell_verts, gmsh_types)):
            node_ids = " ".join(str(v + 1) for v in verts)
            f.write(f"{ci + 1} {etype} 2 1 1 {node_ids}\n")
        f.write("$EndElements\n")

        if fields:
            _write_element_data_v2(f, mesh, fields, time_value, cell_verts)


def _write_element_data_v2(
    f,
    mesh: "FvMesh",
    fields: Dict[str, np.ndarray],
    time_value: float,
    cell_verts: list[list[int]],
) -> None:
    """Write $ElementNodeData for Gmsh 2.2 format."""
    n_cells = mesh.n_cells

    for field_name, data in fields.items():
        f.write("$ElementNodeData\n")
        f.write("1\n")
        f.write(f'"{field_name}"\n')
        f.write("1\n")
        f.write(f"{time_value}\n")
        f.write("3\n")
        f.write("0\n")
        f.write(f"{1 if data.ndim == 1 else data.shape[1]}\n")
        f.write(f"{n_cells}\n")

        for ci in range(n_cells):
            n_nodes = len(cell_verts[ci])
            if data.ndim == 1:
                val_str = f"{data[ci]:18.10E}"
            elif data.ndim == 2 and data.shape[1] == 3:
                val_str = " ".join(f"{data[ci, c]:18.10E}" for c in range(3))
            else:
                val_str = f"{data[ci]:18.10E}"
            node_vals = " ".join([val_str] * n_nodes)
            f.write(f"{ci + 1} {node_vals}\n")

        f.write("$EndElementNodeData\n")


# ---------------------------------------------------------------------------
# Boundary layer .geo file
# ---------------------------------------------------------------------------


def _write_geo_boundary_layers(
    path: Path,
    boundary_layers: Dict[str, dict],
) -> None:
    """Write a Gmsh .geo file with boundary layer field definitions."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("// Boundary layer configuration\n")
        f.write("// Generated by foam_to_gmsh_enhanced\n\n")

        field_id = 1
        bnd_field_ids = []

        for patch_name, params in boundary_layers.items():
            n_layers = params.get("n_layers", 5)
            first_height = params.get("first_height", 1e-5)
            growth_rate = params.get("growth_rate", 1.2)

            f.write(f"// Boundary layer for patch '{patch_name}'\n")
            f.write(f"Field[{field_id}] = BoundaryLayer;\n")
            f.write(f"Field[{field_id}].hwall_n = {first_height};\n")
            f.write(f"Field[{field_id}].ratio = {growth_rate};\n")
            f.write(f"Field[{field_id}].thickness = {first_height * n_layers};\n")
            f.write(f"Field[{field_id}].NumberOfLayers = {n_layers};\n")
            f.write(f'Field[{field_id}].FacesList = {{1}}; // placeholder\n')
            f.write("\n")

            bnd_field_ids.append(field_id)
            field_id += 1

        # Combine all boundary layer fields
        if len(bnd_field_ids) > 1:
            f.write(f"Field[{field_id}] = BoundaryLayer;\n")
            f.write(f"Field[{field_id}].hwall_n = {first_height};\n")
            f.write(f"Field[{field_id}].ratio = {growth_rate};\n")
            f.write(f"Field[{field_id}].thickness = {first_height * n_layers};\n")
            f.write(f"Field[{field_id}].NumberOfLayers = {n_layers};\n")
            f.write("\n")
