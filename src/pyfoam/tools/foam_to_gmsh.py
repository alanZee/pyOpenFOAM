"""
foamToGmsh — export an OpenFOAM case to Gmsh mesh format.

Mirrors the conversion of OpenFOAM mesh and field data to the Gmsh ``.msh``
format (version 2.2 ASCII).

Gmsh mesh format overview:

- ``$MeshFormat`` — version, file-type (0=ASCII), data-size
- ``$Nodes`` — ``node_count`` then ``node_id x y z``
- ``$Elements`` — ``element_count`` then element lines
- ``$ElementNodeData`` / ``$NodeData`` — per-step field values

Supported element type mapping (OpenFOAM → Gmsh):

- Tet (4 verts) → Gmsh type 4
- Hex (8 verts) → Gmsh type 5
- Prism/Wedge (6 verts) → Gmsh type 6
- Pyramid (5 verts) → Gmsh type 7
- Other → Gmsh type 0 (generic polyline, not standard)

Usage::

    from pyfoam.tools.foam_to_gmsh import foam_to_gmsh

    foam_to_gmsh(case_path, output_path="mesh.msh", mesh=mesh, fields=fields)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["foam_to_gmsh"]

# Gmsh element type codes (Msh 2.2)
_GMSH_TRI = 2
_GMSH_QUAD = 3
_GMSH_TET = 4
_GMSH_HEX = 5
_GMSH_WEDGE = 6
_GMSH_PYRAMID = 7
_GMSH_POLY = 0  # fallback for unsupported


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_gmsh(
    case_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    mesh: "FvMesh | None" = None,
    fields: Dict[str, np.ndarray] | None = None,
    time_value: float = 0.0,
) -> Path:
    """Export an OpenFOAM case to Gmsh ``.msh`` format (version 2.2).

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    output_path : str or Path, optional
        Path for the output ``.msh`` file.  Defaults to
        ``<case_path>/Gmsh/mesh.msh``.
    mesh : FvMesh, optional
        Pre-loaded mesh.  Required for connectivity and geometry.
    fields : dict, optional
        ``{field_name: numpy_array}`` of per-cell field values to embed
        as ``$ElementNodeData`` sections.
    time_value : float
        Physical time value written into field data headers (default 0).

    Returns
    -------
    Path
        Path to the written ``.msh`` file.

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
        raise ValueError("No mesh provided. Pass a mesh object directly.")

    # Determine output path
    if output_path is None:
        gmsh_dir = case_dir / "Gmsh"
        os.makedirs(gmsh_dir, exist_ok=True)
        msh_path = gmsh_dir / "mesh.msh"
    else:
        msh_path = Path(output_path)
        os.makedirs(msh_path.parent, exist_ok=True)

    # Build cell-to-vertex mapping and Gmsh element types
    cell_verts, gmsh_types = _compute_cell_verts_and_types(mesh)

    # Write .msh file
    _write_msh(msh_path, mesh, cell_verts, gmsh_types, fields, time_value)

    return msh_path


# ---------------------------------------------------------------------------
# Cell-vertex mapping and Gmsh type classification
# ---------------------------------------------------------------------------


def _compute_cell_verts_and_types(mesh: "FvMesh"):
    """Build cell-to-unique-vertices mapping and Gmsh element type codes.

    Returns
    -------
    cell_verts : list[list[int]]
        0-based point indices for each cell (sorted, unique).
    gmsh_types : list[int]
        Gmsh element type code for each cell.
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

    gmsh_types = []
    for verts in cell_verts:
        nn = len(verts)
        if nn == 4:
            gmsh_types.append(_GMSH_TET)
        elif nn == 5:
            gmsh_types.append(_GMSH_PYRAMID)
        elif nn == 6:
            gmsh_types.append(_GMSH_WEDGE)
        elif nn == 8:
            gmsh_types.append(_GMSH_HEX)
        else:
            gmsh_types.append(_GMSH_POLY)

    return cell_verts, gmsh_types


# ---------------------------------------------------------------------------
# Gmsh .msh writer (version 2.2 ASCII)
# ---------------------------------------------------------------------------


def _write_msh(
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
        # Header
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")  # version 2.2, ASCII (0), sizeof(double)=8
        f.write("$EndMeshFormat\n")

        # Nodes
        f.write("$Nodes\n")
        f.write(f"{n_points}\n")
        for i in range(n_points):
            # Gmsh uses 1-based node IDs
            f.write(f"{i + 1} {pts[i, 0]:18.10E} {pts[i, 1]:18.10E} {pts[i, 2]:18.10E}\n")
        f.write("$EndNodes\n")

        # Elements
        f.write("$Elements\n")
        f.write(f"{n_cells}\n")
        for ci, (verts, etype) in enumerate(zip(cell_verts, gmsh_types)):
            # Gmsh element ID is 1-based; tags: physical-group, elementary-tag
            # Write: elemID elemType nTags physicalGrp elemGrp nodeIDs...
            # Node IDs are 1-based
            node_ids = " ".join(str(v + 1) for v in verts)
            f.write(f"{ci + 1} {etype} 2 1 1 {node_ids}\n")
        f.write("$EndElements\n")

        # Field data (as $ElementNodeData if fields provided)
        if fields:
            _write_element_data(f, mesh, fields, time_value)


def _write_element_data(
    f,
    mesh: "FvMesh",
    fields: Dict[str, np.ndarray],
    time_value: float,
) -> None:
    """Write $ElementNodeData sections for per-cell fields.

    In Gmsh, ``$ElementNodeData`` maps field values to element nodes.
    Here we assign the cell-average value to every node of each element
    (piecewise-constant interpolation).
    """
    n_cells = mesh.n_cells
    faces = mesh.faces
    owner = mesh.owner.detach().cpu().numpy()
    neighbour = mesh.neighbour.detach().cpu().numpy()
    n_internal = mesh.n_internal_faces

    # Rebuild cell-to-vertices for element data
    cell_to_verts: list[list[int]] = []
    cell_verts_set: list[set[int]] = [set() for _ in range(n_cells)]
    for fi, face in enumerate(faces):
        face_nodes = face.detach().cpu().numpy().tolist()
        c_own = int(owner[fi])
        cell_verts_set[c_own].update(face_nodes)
        if fi < n_internal:
            c_nbr = int(neighbour[fi])
            cell_verts_set[c_nbr].update(face_nodes)
    cell_to_verts = [sorted(vs) for vs in cell_verts_set]

    for field_name, data in fields.items():
        f.write("$ElementNodeData\n")
        # Header: string-tags-count, string-tags, real-tags-count, real-tags,
        #         int-tags-count, int-tags
        # We use 1 string tag (field name), 1 real tag (time), 1 int tag (step)
        f.write("1\n")  # number of string tags
        f.write(f'"{field_name}"\n')
        f.write("1\n")  # number of real tags
        f.write(f"{time_value}\n")
        f.write("3\n")  # number of integer tags
        f.write(f"0\n")  # time step index
        f.write(f"{1 if data.ndim == 1 else data.shape[1]}\n")  # components
        f.write(f"{n_cells}\n")  # number of elements

        for ci in range(n_cells):
            verts = cell_to_verts[ci]
            n_nodes = len(verts)
            if data.ndim == 1:
                val_str = f"{data[ci]:18.10E}"
            elif data.ndim == 2 and data.shape[1] == 3:
                val_str = " ".join(f"{data[ci, c]:18.10E}" for c in range(3))
            else:
                val_str = f"{data[ci]:18.10E}"

            # elemID value_per_node (repeated for each node)
            node_vals = " ".join([val_str] * n_nodes)
            f.write(f"{ci + 1} {node_vals}\n")

        f.write("$EndElementNodeData\n")
