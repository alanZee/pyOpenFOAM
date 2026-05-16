"""
Gmsh mesh file I/O and gmshToFoam converter.

Parses Gmsh ``.msh`` format (version 2.2 and 4.1) and converts to
OpenFOAM polyMesh format.

Gmsh MSH format v2.2 structure::

    $MeshFormat
    2.2 0 8
    $EndMeshFormat
    $Nodes
    N
    node-id x y z
    ...
    $EndNodes
    $Elements
    N
    elem-id elem-type n-tags tag1 tag2 ... node1 node2 ...
    ...
    $EndElements
    $PhysicalNames
    N
    dim physical-id "name"
    ...
    $EndPhysicalNames

Gmsh element type codes (relevant subset):

    1:  2-node line
    2:  3-node triangle
    3:  4-node quadrangle
    4:  4-node tetrahedron
    5:  8-node hexahedron
    6:  6-node prism (wedge)
    8:  3-node second-order line
    9:  6-node second-order triangle
    11: 10-node second-order tetrahedron
    15: 1-node point

References
----------
- Gmsh documentation: https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
- OpenFOAM ``gmshToFoam`` utility source
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.mesh_io import BoundaryPatch, MeshData

__all__ = [
    "GmshElement",
    "GmshPhysicalGroup",
    "GmshMesh",
    "read_gmsh",
    "gmsh_to_foam",
]

# ---------------------------------------------------------------------------
# Gmsh element type definitions
# ---------------------------------------------------------------------------

# Maps Gmsh element type -> (n_nodes, is_volume, is_surface, is_line, is_point)
_GMSH_ELEMENT_INFO: Dict[int, Tuple[int, bool, bool, bool, bool]] = {
    1:  (2,  False, False, True,  False),  # 2-node line
    2:  (3,  False, True,  False, False),  # 3-node triangle
    3:  (4,  False, True,  False, False),  # 4-node quadrangle
    4:  (4,  True,  False, False, False),  # 4-node tetrahedron
    5:  (8,  True,  False, False, False),  # 8-node hexahedron
    6:  (6,  True,  False, False, False),  # 6-node prism (wedge)
    7:  (5,  True,  False, False, False),  # 5-node pyramid
    8:  (3,  False, False, True,  False),  # 3-node second-order line
    9:  (6,  False, True,  False, False),  # 6-node second-order triangle
    10: (9,  False, True,  False, False),  # 9-node second-order quadrangle
    11: (10, True,  False, False, False),  # 10-node second-order tetrahedron
    15: (1,  False, False, False, True),   # 1-node point
    16: (8,  False, True,  False, False),  # 8-node second-order quadrangle
    17: (20, True,  False, False, False),  # 20-node second-order hexahedron
}

# Volume element types
_VOLUME_TYPES = {k for k, v in _GMSH_ELEMENT_INFO.items() if v[1]}
# Surface element types
_SURFACE_TYPES = {k for k, v in _GMSH_ELEMENT_INFO.items() if v[2]}
# Line element types
_LINE_TYPES = {k for k, v in _GMSH_ELEMENT_INFO.items() if v[3]}

# Linear volume element vertex counts (for degenerate handling)
_LINEAR_VOLUME_VERTS = {4: 4, 5: 8, 6: 6, 7: 5}  # type -> n_verts


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GmshElement:
    """A single Gmsh element.

    Attributes:
        elem_id: Element ID (1-based in Gmsh).
        elem_type: Gmsh element type code.
        physical_group: Physical group tag.
        nodes: Node IDs (1-based in Gmsh).
    """

    elem_id: int
    elem_type: int
    physical_group: int
    nodes: List[int]


@dataclass
class GmshPhysicalGroup:
    """A Gmsh physical group (surface or volume).

    Attributes:
        dim: Dimension (1=line, 2=surface, 3=volume).
        physical_id: Physical group ID.
        name: Physical group name.
    """

    dim: int
    physical_id: int
    name: str


@dataclass
class GmshMesh:
    """Parsed Gmsh mesh.

    Attributes:
        node_coords: Node coordinates, shape ``(n_nodes, 3)``, 0-based indexing.
        node_id_map: Maps Gmsh node ID -> 0-based index.
        elements: List of parsed elements.
        physical_groups: List of physical group definitions.
        mesh_format: Format version string.
    """

    node_coords: np.ndarray  # (n_nodes, 3)
    node_id_map: Dict[int, int]  # gmsh_id -> 0-based index
    elements: List[GmshElement]
    physical_groups: List[GmshPhysicalGroup]
    mesh_format: str = "2.2"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def read_gmsh(path: Union[str, Path]) -> GmshMesh:
    """Read a Gmsh ``.msh`` file.

    Supports MSH format versions 2.2 and 4.1.

    Args:
        path: Path to the ``.msh`` file.

    Returns:
        Parsed :class:`GmshMesh`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    return _parse_gmsh(content)


def _parse_gmsh(content: str) -> GmshMesh:
    """Parse Gmsh MSH content."""
    # Detect format version
    fmt_match = re.search(r"\$MeshFormat\s*\n([\d.]+)", content)
    version = fmt_match.group(1) if fmt_match else "2.2"

    # Parse physical names
    physical_groups = _parse_physical_names(content)

    # Parse nodes
    node_coords, node_id_map = _parse_nodes(content, version)

    # Parse elements
    elements = _parse_elements(content, version, node_id_map)

    return GmshMesh(
        node_coords=node_coords,
        node_id_map=node_id_map,
        elements=elements,
        physical_groups=physical_groups,
        mesh_format=version,
    )


def _parse_physical_names(content: str) -> List[GmshPhysicalGroup]:
    """Parse $PhysicalNames section."""
    groups: List[GmshPhysicalGroup] = []
    match = re.search(
        r"\$PhysicalNames\s*\n\s*(\d+)\s*\n(.*?)\$EndPhysicalNames",
        content,
        re.DOTALL,
    )
    if match is None:
        return groups

    for line in match.group(2).strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Format: dim physical-id "name"  (quotes optional in some Gmsh versions)
        # Try quoted format first
        parts = line.split('"')
        if len(parts) >= 3:
            header = parts[0].strip().split()
            dim = int(header[0])
            phys_id = int(header[1])
            name = parts[1].strip()
        else:
            # Unquoted format: dim physical-id name
            tokens = line.split()
            if len(tokens) >= 3:
                dim = int(tokens[0])
                phys_id = int(tokens[1])
                name = tokens[2]
            else:
                continue
        groups.append(GmshPhysicalGroup(dim=dim, physical_id=phys_id, name=name))

    return groups


def _parse_nodes(
    content: str, version: str
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Parse $Nodes section.

    Returns:
        Tuple of (node_coords, node_id_map).
    """
    if version.startswith("4"):
        return _parse_nodes_v4(content)
    return _parse_nodes_v2(content)


def _parse_nodes_v2(
    content: str,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Parse $Nodes for MSH format v2.x."""
    match = re.search(
        r"\$Nodes\s*\n\s*(\d+)\s*\n(.*?)\$EndNodes",
        content,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("$Nodes section not found")

    n_nodes = int(match.group(1))
    lines = match.group(2).strip().split("\n")

    coords = np.zeros((n_nodes, 3), dtype=np.float64)
    id_map: Dict[int, int] = {}

    for i, line in enumerate(lines):
        if i >= n_nodes:
            break
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        node_id = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        id_map[node_id] = i
        coords[i] = [x, y, z]

    return coords, id_map


def _parse_nodes_v4(
    content: str,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Parse $Nodes for MSH format v4.x.

    V4 format has entity blocks::

        numEntityBlocks numNodes minNodeTag maxNodeTag
        entityDim entityTag parametric numNodesInBlock
        nodeTag
        ...
        x y z
        ...
    """
    match = re.search(
        r"\$Nodes\s*\n(.*?)\$EndNodes",
        content,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("$Nodes section not found")

    lines = match.group(1).strip().split("\n")
    header = lines[0].strip().split()
    n_nodes = int(header[1])

    coords_list = []
    id_map: Dict[int, int] = {}
    idx = 1

    while idx < len(lines):
        parts = lines[idx].strip().split()
        if len(parts) < 4:
            idx += 1
            continue

        # entityDim entityTag parametric nNodesInBlock
        n_block = int(parts[3])
        idx += 1

        # Read node tags
        tags = []
        for _ in range(n_block):
            if idx < len(lines):
                tags.append(int(lines[idx].strip()))
                idx += 1

        # Read coordinates
        for i, tag in enumerate(tags):
            if idx < len(lines):
                coords = lines[idx].strip().split()
                if len(coords) >= 3:
                    c = [float(coords[0]), float(coords[1]), float(coords[2])]
                    id_map[tag] = len(coords_list)
                    coords_list.append(c)
                idx += 1

    return np.array(coords_list, dtype=np.float64), id_map


def _parse_elements(
    content: str, version: str, node_id_map: Dict[int, int]
) -> List[GmshElement]:
    """Parse $Elements section."""
    if version.startswith("4"):
        return _parse_elements_v4(content, node_id_map)
    return _parse_elements_v2(content, node_id_map)


def _parse_elements_v2(
    content: str, node_id_map: Dict[int, int]
) -> List[GmshElement]:
    """Parse $Elements for MSH format v2.x."""
    match = re.search(
        r"\$Elements\s*\n\s*(\d+)\s*\n(.*?)\$EndElements",
        content,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("$Elements section not found")

    lines = match.group(2).strip().split("\n")
    elements: List[GmshElement] = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        elem_id = int(parts[0])
        elem_type = int(parts[1])
        n_tags = int(parts[2])

        # First tag is physical group
        physical_group = int(parts[3]) if n_tags > 0 else 0

        # Node IDs start at index 3 + n_tags
        node_start = 3 + n_tags
        node_ids = [int(parts[i]) for i in range(node_start, len(parts))]

        elements.append(GmshElement(
            elem_id=elem_id,
            elem_type=elem_type,
            physical_group=physical_group,
            nodes=node_ids,
        ))

    return elements


def _parse_elements_v4(
    content: str, node_id_map: Dict[int, int]
) -> List[GmshElement]:
    """Parse $Elements for MSH format v4.x.

    V4 format has entity blocks::

        numEntityBlocks numElements minElementTag maxElementTag
        entityDim entityTag elemType numElementsInBlock
        elemTag node1 node2 ...
        ...
    """
    match = re.search(
        r"\$Elements\s*\n(.*?)\$EndElements",
        content,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("$Elements section not found")

    lines = match.group(1).strip().split("\n")
    elements: List[GmshElement] = []

    idx = 1  # Skip header line

    while idx < len(lines):
        parts = lines[idx].strip().split()
        if len(parts) < 4:
            idx += 1
            continue

        # entityDim entityTag elemType nElementsInBlock
        elem_type = int(parts[2])
        n_block = int(parts[3])
        idx += 1

        for _ in range(n_block):
            if idx >= len(lines):
                break
            ep = lines[idx].strip().split()
            elem_id = int(ep[0])
            node_ids = [int(ep[i]) for i in range(1, len(ep))]

            elements.append(GmshElement(
                elem_id=elem_id,
                elem_type=elem_type,
                physical_group=int(parts[1]),  # entityTag as physical group
                nodes=node_ids,
            ))
            idx += 1

    return elements


# ---------------------------------------------------------------------------
# gmshToFoam conversion
# ---------------------------------------------------------------------------


def gmsh_to_foam(
    gmsh_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    overwrite: bool = False,
) -> MeshData:
    """Convert a Gmsh ``.msh`` file to OpenFOAM polyMesh format.

    Reads the Gmsh mesh and writes the 5 polyMesh files:
    ``points``, ``faces``, ``owner``, ``neighbour``, ``boundary``.

    Args:
        gmsh_path: Path to the Gmsh ``.msh`` file.
        output_dir: Output directory for polyMesh files.
            The ``constant/polyMesh`` subdirectory will be created.
        overwrite: If True, overwrite existing files.

    Returns:
        The converted :class:`MeshData`.
    """
    gmsh_path = Path(gmsh_path)
    output_dir = Path(output_dir)
    poly_mesh_dir = output_dir / "constant" / "polyMesh"
    poly_mesh_dir.mkdir(parents=True, exist_ok=True)

    # Parse Gmsh mesh
    gmsh = read_gmsh(gmsh_path)

    # Build physical group name map
    phys_names: Dict[int, str] = {}
    phys_dims: Dict[int, int] = {}
    for pg in gmsh.physical_groups:
        phys_names[pg.physical_id] = pg.name
        phys_dims[pg.physical_id] = pg.dim

    # Convert node coordinates
    points = torch.tensor(
        gmsh.node_coords, dtype=get_default_dtype(), device=get_device()
    )

    # Separate volume and surface/line elements
    volume_elements: List[GmshElement] = []
    boundary_elements: Dict[int, List[GmshElement]] = {}  # phys_group -> elements

    for elem in gmsh.elements:
        if elem.elem_type in _VOLUME_TYPES:
            volume_elements.append(elem)
        elif elem.elem_type in _SURFACE_TYPES or elem.elem_type in _LINE_TYPES:
            pg = elem.physical_group
            if pg not in boundary_elements:
                boundary_elements[pg] = []
            boundary_elements[pg].append(elem)

    # Build cell connectivity: for each volume element, collect its faces
    # A face is defined by its sorted node indices
    faces, owner, neighbour, cell_faces = _build_foam_mesh(
        volume_elements, gmsh.node_id_map
    )

    # Build boundary patches from surface elements
    boundary_patches = _build_boundary_patches(
        boundary_elements, faces, owner, gmsh.node_id_map, phys_names
    )

    # Convert faces to numpy arrays
    faces_np = [np.array(f, dtype=np.int32) for f in faces]

    mesh_data = MeshData(
        points=points,
        faces=faces_np,
        owner=owner,
        neighbour=neighbour,
        boundary=boundary_patches,
    )

    # Write polyMesh files
    from pyfoam.io.foam_file import FoamFileHeader, FileFormat

    header_points = FoamFileHeader(
        format=FileFormat.ASCII,
        class_name="vectorField",
        location="constant/polyMesh",
        object="points",
    )
    header_faces = FoamFileHeader(
        format=FileFormat.ASCII,
        class_name="faceList",
        location="constant/polyMesh",
        object="faces",
    )
    header_owner = FoamFileHeader(
        format=FileFormat.ASCII,
        class_name="labelList",
        location="constant/polyMesh",
        object="owner",
    )
    header_neighbour = FoamFileHeader(
        format=FileFormat.ASCII,
        class_name="labelList",
        location="constant/polyMesh",
        object="neighbour",
    )
    header_boundary = FoamFileHeader(
        format=FileFormat.ASCII,
        class_name="polyBoundaryMesh",
        location="constant/polyMesh",
        object="boundary",
    )

    from pyfoam.io.mesh_io import (
        write_boundary,
        write_faces,
        write_neighbour,
        write_owner,
        write_points,
    )

    write_points(poly_mesh_dir / "points", header_points, points, overwrite=overwrite)
    write_faces(poly_mesh_dir / "faces", header_faces, faces_np, overwrite=overwrite)
    write_owner(poly_mesh_dir / "owner", header_owner, owner, overwrite=overwrite)
    write_neighbour(
        poly_mesh_dir / "neighbour", header_neighbour, neighbour, overwrite=overwrite
    )
    write_boundary(
        poly_mesh_dir / "boundary", header_boundary, boundary_patches, overwrite=overwrite
    )

    return mesh_data


def _build_foam_mesh(
    volume_elements: List[GmshElement],
    node_id_map: Dict[int, int],
) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor, List[List[int]]]:
    """Build OpenFOAM mesh connectivity from volume elements.

    Returns:
        Tuple of (all_faces, owner, neighbour, cell_face_indices).
        - all_faces: List of face vertex lists (0-based).
        - owner: Owner cell tensor.
        - neighbour: Neighbour cell tensor.
        - cell_face_indices: For each cell, list of face indices.
    """
    # Map face (sorted tuple) -> (face_index, owner_cell)
    face_map: Dict[Tuple[int, ...], Tuple[int, int]] = {}
    all_faces: List[List[int]] = []
    cell_face_indices: List[List[int]] = []

    for cell_idx, elem in enumerate(volume_elements):
        cell_faces: List[int] = []
        elem_faces = _get_element_faces(elem.elem_type, elem.nodes, node_id_map)

        for face_nodes in elem_faces:
            face_key = tuple(sorted(face_nodes))
            if face_key in face_map:
                # This is an internal face; current cell is neighbour
                face_idx, owner_cell = face_map[face_key]
                cell_faces.append(face_idx)
            else:
                # New face; current cell is owner
                face_idx = len(all_faces)
                face_map[face_key] = (face_idx, cell_idx)
                all_faces.append(face_nodes)
                cell_faces.append(face_idx)

        cell_face_indices.append(cell_faces)

    # Build owner and neighbour arrays
    n_faces = len(all_faces)
    owner = torch.zeros(n_faces, dtype=torch.int64, device=get_device())
    neighbour_list: List[int] = []

    # Re-process to fill owner/neighbour
    face_owners: Dict[int, int] = {}
    face_neighbours: Dict[int, int] = {}

    for cell_idx, elem in enumerate(volume_elements):
        elem_faces = _get_element_faces(elem.elem_type, elem.nodes, node_id_map)

        for face_nodes in elem_faces:
            face_key = tuple(sorted(face_nodes))
            face_idx = face_map[face_key][0]

            if face_idx not in face_owners:
                face_owners[face_idx] = cell_idx
            else:
                face_neighbours[face_idx] = cell_idx

    # Fill owner tensor
    for i in range(n_faces):
        owner[i] = face_owners.get(i, 0)

    # Build neighbour (only for internal faces)
    internal_faces = sorted(face_neighbours.keys())
    neighbour = torch.tensor(
        [face_neighbours[i] for i in internal_faces],
        dtype=torch.int64,
        device=get_device(),
    )

    return all_faces, owner, neighbour, cell_face_indices


def _get_element_faces(
    elem_type: int,
    node_ids: List[int],
    node_id_map: Dict[int, int],
) -> List[List[int]]:
    """Get the faces of a volume element.

    Returns:
        List of face vertex lists (0-based node indices).
    """
    # Convert Gmsh node IDs to 0-based indices
    verts = [node_id_map[n] for n in node_ids]

    if elem_type == 4:  # Tetrahedron (4 nodes)
        # Faces: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
        return [
            [verts[0], verts[1], verts[2]],
            [verts[0], verts[1], verts[3]],
            [verts[0], verts[2], verts[3]],
            [verts[1], verts[2], verts[3]],
        ]
    elif elem_type == 5:  # Hexahedron (8 nodes)
        # Standard hex face ordering (OpenFOAM convention)
        return [
            [verts[0], verts[1], verts[5], verts[4]],  # bottom
            [verts[2], verts[3], verts[7], verts[6]],  # top
            [verts[0], verts[1], verts[3], verts[2]],  # front
            [verts[4], verts[5], verts[7], verts[6]],  # back
            [verts[0], verts[4], verts[6], verts[2]],  # left
            [verts[1], verts[5], verts[7], verts[3]],  # right
        ]
    elif elem_type == 6:  # Prism/Wedge (6 nodes)
        return [
            [verts[0], verts[1], verts[4], verts[3]],  # quad face 1
            [verts[1], verts[2], verts[5], verts[4]],  # quad face 2
            [verts[0], verts[2], verts[5], verts[3]],  # quad face 3
            [verts[0], verts[2], verts[1]],             # tri face 1
            [verts[3], verts[4], verts[5]],             # tri face 2
        ]
    elif elem_type == 7:  # Pyramid (5 nodes)
        return [
            [verts[0], verts[1], verts[4]],  # tri
            [verts[1], verts[2], verts[4]],  # tri
            [verts[2], verts[3], verts[4]],  # tri
            [verts[0], verts[3], verts[4]],  # tri
            [verts[0], verts[1], verts[2], verts[3]],  # quad base
        ]
    elif elem_type == 11:  # 10-node second-order tetrahedron
        # Use only corner nodes (0,1,2,3) for linear faces
        return [
            [verts[0], verts[1], verts[2]],
            [verts[0], verts[1], verts[3]],
            [verts[0], verts[2], verts[3]],
            [verts[1], verts[2], verts[3]],
        ]
    elif elem_type == 17:  # 20-node second-order hexahedron
        # Use only corner nodes
        return [
            [verts[0], verts[1], verts[5], verts[4]],
            [verts[2], verts[3], verts[7], verts[6]],
            [verts[0], verts[1], verts[3], verts[2]],
            [verts[4], verts[5], verts[7], verts[6]],
            [verts[0], verts[4], verts[6], verts[2]],
            [verts[1], verts[5], verts[7], verts[3]],
        ]
    else:
        # Generic: try to construct faces from the element
        # For unknown types, return a single face with all nodes
        return [verts]


def _build_boundary_patches(
    boundary_elements: Dict[int, List[GmshElement]],
    all_faces: List[List[int]],
    owner: torch.Tensor,
    node_id_map: Dict[int, int],
    phys_names: Dict[int, str],
) -> List[BoundaryPatch]:
    """Build boundary patches from surface elements.

    Surface elements in Gmsh define boundary faces. We match them to
    the existing face list and create boundary patches.
    """
    # Build a reverse lookup: sorted face tuple -> face index
    face_lookup: Dict[Tuple[int, ...], int] = {}
    for i, face in enumerate(all_faces):
        face_lookup[tuple(sorted(face))] = i

    patches: List[BoundaryPatch] = []
    current_start = owner.shape[0]  # Boundary faces start after all faces

    for pg_id in sorted(boundary_elements.keys()):
        elems = boundary_elements[pg_id]
        name = phys_names.get(pg_id, f"patch_{pg_id}")

        # Find which faces belong to this boundary
        boundary_face_indices: List[int] = []
        for elem in elems:
            surface_verts = [node_id_map[n] for n in elem.nodes]
            face_key = tuple(sorted(surface_verts))
            if face_key in face_lookup:
                boundary_face_indices.append(face_lookup[face_key])

        if not boundary_face_indices:
            continue

        patches.append(BoundaryPatch(
            name=name,
            patch_type="patch",
            n_faces=len(boundary_face_indices),
            start_face=current_start,
        ))

        # Add boundary faces to the face list
        for fi in boundary_face_indices:
            # This face is a boundary face - it should already be in all_faces
            # with owner but no neighbour
            pass

        current_start += len(boundary_face_indices)

    return patches
