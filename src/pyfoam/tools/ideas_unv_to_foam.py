"""
ideasUNVToFoam — convert I-DEAS Universal (UNV) mesh to OpenFOAM polyMesh.

Parses I-DEAS Universal file format datasets and writes the 5 OpenFOAM
polyMesh files: ``points``, ``faces``, ``owner``, ``neighbour``, ``boundary``.

UNV dataset structure::

    -1
    dataset_number
    ...data...
    -1

Key datasets:
    2411 — Node coordinates (record 1: node_id, x, y, z)
    2412 — Elements (record 1: elem_id, type, n_nodes, ...)
           Element types: 11=beam, 22=tet4, 23=wedge6, 24=hex8, 25=tet10
    2467 — Groups (FE groups = boundary patches)
           Record 1: group_name
           Records 2+: 8-field lines (entity_type, entity_id, ...)

References
----------
- I-DEAS Universal file format documentation
- OpenFOAM ``ideasUnvToFoam`` utility source
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.mesh_io import BoundaryPatch, MeshData

__all__ = [
    "UnvNode",
    "UnvElement",
    "UnvGroup",
    "UnvMesh",
    "read_unv",
    "ideas_unv_to_foam",
]


# ---------------------------------------------------------------------------
# UNV element type definitions
# ---------------------------------------------------------------------------

# Maps UNV element type -> (n_nodes, is_volume)
_UNV_ELEMENT_INFO: Dict[int, Tuple[int, bool]] = {
    11: (2, False),   # beam
    21: (3, False),   # triangular plate (3-node)
    22: (4, True),    # tetrahedron (4-node)
    23: (6, True),    # wedge/prism (6-node)
    24: (8, True),    # hexahedron (8-node)
    25: (10, True),   # tetrahedron (10-node, second-order)
    41: (4, False),   # quadrilateral plate (4-node)
    44: (8, True),    # hexahedron (8-node, alternative)
    45: (4, True),    # tetrahedron (4-node, alternative)
    51: (2, False),   # rigid bar
    61: (6, True),    # wedge (6-node, alternative)
    73: (3, False),   # triangular thin shell
    74: (4, False),   # quadrilateral thin shell
    91: (3, False),   # triangular plate
    92: (4, False),   # quadrilateral plate
    94: (4, False),   # quadrilateral plate (alternative)
    111: (4, True),   # solid tet (4-node)
    112: (8, True),   # solid hex (8-node)
    113: (6, True),   # solid wedge (6-node)
    114: (5, True),   # solid pyramid (5-node)
    115: (4, True),   # tet (4-node, alternative)
    116: (8, True),   # hex (8-node, alternative)
    117: (6, True),   # wedge (6-node, alternative)
    118: (5, True),   # pyramid (5-node, alternative)
}

_VOLUME_TYPES = {k for k, v in _UNV_ELEMENT_INFO.items() if v[1]}
_SURFACE_TYPES = {k for k, v in _UNV_ELEMENT_INFO.items() if not v[1] and v[0] >= 3}
_LINE_TYPES = {k for k, v in _UNV_ELEMENT_INFO.items() if not v[1] and v[0] < 3}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class UnvNode:
    """A UNV node.

    Attributes:
        node_id: Node ID (1-based in UNV).
        coords: Node coordinates (x, y, z).
    """

    def __init__(self, node_id: int, coords: Tuple[float, float, float]) -> None:
        self.node_id = node_id
        self.coords = coords


class UnvElement:
    """A UNV element.

    Attributes:
        elem_id: Element ID (1-based in UNV).
        elem_type: UNV element type code.
        nodes: Node IDs (1-based in UNV).
        phys_prop: Physical property number.
        mat_prop: Material property number.
    """

    def __init__(
        self,
        elem_id: int,
        elem_type: int,
        nodes: List[int],
        phys_prop: int = 0,
        mat_prop: int = 0,
    ) -> None:
        self.elem_id = elem_id
        self.elem_type = elem_type
        self.nodes = nodes
        self.phys_prop = phys_prop
        self.mat_prop = mat_prop


class UnvGroup:
    """A UNV FE group (used as boundary patch).

    Attributes:
        name: Group name.
        entities: List of (entity_type, entity_id) tuples.
            entity_type: 7=element, 8=node
    """

    def __init__(
        self, name: str, entities: List[Tuple[int, int]]
    ) -> None:
        self.name = name
        self.entities = entities


class UnvMesh:
    """Parsed UNV mesh.

    Attributes:
        nodes: List of parsed nodes.
        node_id_map: Maps UNV node ID -> 0-based index.
        elements: List of parsed elements.
        groups: List of parsed groups.
    """

    def __init__(
        self,
        nodes: List[UnvNode],
        node_id_map: Dict[int, int],
        elements: List[UnvElement],
        groups: List[UnvGroup],
    ) -> None:
        self.nodes = nodes
        self.node_id_map = node_id_map
        self.elements = elements
        self.groups = groups


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def read_unv(path: Union[str, Path]) -> UnvMesh:
    """Read an I-DEAS Universal (UNV) file.

    Args:
        path: Path to the ``.unv`` file.

    Returns:
        Parsed :class:`UnvMesh`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no nodes or elements found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    return _parse_unv(content)


def _parse_unv(content: str) -> UnvMesh:
    """Parse UNV file content by splitting into datasets."""
    # Split content into datasets delimited by -1 lines
    datasets = _split_datasets(content)

    nodes: List[UnvNode] = []
    node_id_map: Dict[int, int] = {}
    elements: List[UnvElement] = []
    groups: List[UnvGroup] = []

    for ds_num, ds_lines in datasets:
        if ds_num == 2411:
            nodes, node_id_map = _parse_dataset_2411(ds_lines)
        elif ds_num == 2412:
            elements.extend(_parse_dataset_2412(ds_lines))
        elif ds_num == 2467:
            groups.extend(_parse_dataset_2467(ds_lines))

    if not nodes:
        raise ValueError("No nodes (dataset 2411) found in UNV file")
    if not elements:
        raise ValueError("No elements (dataset 2412) found in UNV file")

    return UnvMesh(
        nodes=nodes,
        node_id_map=node_id_map,
        elements=elements,
        groups=groups,
    )


def _split_datasets(content: str) -> List[Tuple[int, List[str]]]:
    """Split UNV content into (dataset_number, lines) pairs."""
    lines = content.splitlines()
    datasets: List[Tuple[int, List[str]]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "-1":
            # Start of dataset
            i += 1
            if i >= len(lines):
                break
            ds_num_line = lines[i].strip()
            try:
                ds_num = int(ds_num_line)
            except ValueError:
                i += 1
                continue
            i += 1

            # Collect lines until next -1
            ds_lines: List[str] = []
            while i < len(lines) and lines[i].strip() != "-1":
                ds_lines.append(lines[i])
                i += 1
            # Skip the closing -1
            i += 1

            datasets.append((ds_num, ds_lines))
        else:
            i += 1

    return datasets


def _parse_dataset_2411(
    lines: List[str],
) -> Tuple[List[UnvNode], Dict[int, int]]:
    """Parse dataset 2411: Node coordinates.

    Format (two lines per node):
        Line 1: node_id  exp_coord_sys_id  disp_coord_sys_id
        Line 2: x  y  z
    """
    nodes: List[UnvNode] = []
    node_id_map: Dict[int, int] = {}

    i = 0
    while i + 1 < len(lines):
        header_line = lines[i].strip()
        data_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if not header_line:
            i += 1
            continue

        try:
            parts = header_line.split()
            node_id = int(parts[0])
        except (ValueError, IndexError):
            i += 2
            continue

        try:
            coords_parts = data_line.split()
            if len(coords_parts) >= 3:
                x = float(coords_parts[0].replace("D", "E").replace("d", "e"))
                y = float(coords_parts[1].replace("D", "E").replace("d", "e"))
                z = float(coords_parts[2].replace("D", "E").replace("d", "e"))
            else:
                i += 2
                continue
        except (ValueError, IndexError):
            i += 2
            continue

        idx = len(nodes)
        node_id_map[node_id] = idx
        nodes.append(UnvNode(node_id=node_id, coords=(x, y, z)))
        i += 2

    return nodes, node_id_map


def _parse_dataset_2412(lines: List[str]) -> List[UnvElement]:
    """Parse dataset 2412: Elements.

    Format (two lines per element):
        Line 1: elem_id  fe_descriptor_id  phys_prop  mat_prop  color  n_nodes
        Line 2: node1  node2  ...  nodeN
    """
    elements: List[UnvElement] = []

    i = 0
    while i + 1 < len(lines):
        header_line = lines[i].strip()
        data_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if not header_line:
            i += 1
            continue

        try:
            parts = header_line.split()
            if len(parts) < 2:
                i += 2
                continue

            elem_id = int(parts[0])
            elem_type = int(parts[1])
            n_nodes_in_data = 0
            if len(parts) >= 6:
                n_nodes_in_data = int(parts[5])

            # Parse node IDs
            node_ids: List[int] = []
            if data_line:
                for tok in data_line.split():
                    try:
                        node_ids.append(int(tok))
                    except ValueError:
                        pass

            # Some element types have node IDs on multiple lines
            if n_nodes_in_data > 0 and len(node_ids) < n_nodes_in_data:
                # Read additional lines for node data
                extra_idx = i + 2
                while len(node_ids) < n_nodes_in_data and extra_idx < len(lines):
                    extra_line = lines[extra_idx].strip()
                    if not extra_line or extra_line == "-1":
                        break
                    for tok in extra_line.split():
                        if len(node_ids) >= n_nodes_in_data:
                            break
                        try:
                            node_ids.append(int(tok))
                        except ValueError:
                            pass
                    extra_idx += 1
                i = extra_idx
            else:
                i += 2

            phys_prop = int(parts[2]) if len(parts) >= 3 else 0
            mat_prop = int(parts[3]) if len(parts) >= 4 else 0

            if node_ids:
                elements.append(UnvElement(
                    elem_id=elem_id,
                    elem_type=elem_type,
                    nodes=node_ids,
                    phys_prop=phys_prop,
                    mat_prop=mat_prop,
                ))
        except (ValueError, IndexError):
            i += 2
            continue

    return elements


def _parse_dataset_2467(lines: List[str]) -> List[UnvGroup]:
    """Parse dataset 2467: Groups (FE groups = boundary patches).

    Format:
        Line 1: group_name (8 chars)
        Line 2: n_items  (negative = active group)
        Lines 3+: 8-field records: entity_type  entity_id  ...
    """
    groups: List[UnvGroup] = []

    i = 0
    while i < len(lines):
        name_line = lines[i].strip()
        if not name_line:
            i += 1
            continue

        group_name = name_line.strip()
        i += 1

        if i >= len(lines):
            break

        count_line = lines[i].strip()
        try:
            count = int(count_line)
        except ValueError:
            continue
        n_items = abs(count)
        i += 1

        entities: List[Tuple[int, int]] = []
        for _ in range(n_items):
            if i >= len(lines):
                break
            line = lines[i].strip()
            parts = line.split()
            if len(parts) >= 2:
                try:
                    entity_type = int(parts[0])
                    entity_id = int(parts[1])
                    entities.append((entity_type, entity_id))
                except ValueError:
                    pass
            i += 1

        if entities:
            groups.append(UnvGroup(name=group_name, entities=entities))

    return groups


# ---------------------------------------------------------------------------
# ideasUnvToFoam conversion
# ---------------------------------------------------------------------------


def ideas_unv_to_foam(
    unv_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    overwrite: bool = False,
) -> MeshData:
    """Convert an I-DEAS Universal (UNV) file to OpenFOAM polyMesh format.

    Args:
        unv_path: Path to the ``.unv`` file.
        output_dir: Output directory for polyMesh files.
            The ``constant/polyMesh`` subdirectory will be created.
        overwrite: If True, overwrite existing files.

    Returns:
        The converted :class:`MeshData`.
    """
    unv_path = Path(unv_path)
    output_dir = Path(output_dir)
    poly_mesh_dir = output_dir / "constant" / "polyMesh"
    poly_mesh_dir.mkdir(parents=True, exist_ok=True)

    # Parse UNV mesh
    unv = read_unv(unv_path)

    # Convert node coordinates
    coords = np.array([n.coords for n in unv.nodes], dtype=np.float64)
    points = torch.tensor(coords, dtype=get_default_dtype(), device=get_device())

    # Separate volume and boundary elements
    volume_elements: List[UnvElement] = []
    boundary_elements: Dict[int, List[UnvElement]] = {}  # phys_prop -> elements

    for elem in unv.elements:
        if elem.elem_type in _VOLUME_TYPES:
            volume_elements.append(elem)
        elif elem.elem_type in _SURFACE_TYPES or elem.elem_type in _LINE_TYPES:
            pg = elem.phys_prop
            if pg not in boundary_elements:
                boundary_elements[pg] = []
            boundary_elements[pg].append(elem)

    # Build faces, owner, neighbour from volume elements
    all_faces, owner, neighbour = _build_foam_mesh(volume_elements, unv.node_id_map)

    # Build boundary patches from groups or boundary elements
    boundary_patches = _build_boundary_patches(
        unv.groups, boundary_elements, all_faces, unv.node_id_map,
    )

    # Convert faces to numpy arrays
    faces_np = [np.array(f, dtype=np.int32) for f in all_faces]

    mesh_data = MeshData(
        points=points,
        faces=faces_np,
        owner=owner,
        neighbour=neighbour,
        boundary=boundary_patches,
    )

    # Write polyMesh files
    _write_polymesh(poly_mesh_dir, mesh_data, overwrite=overwrite)

    return mesh_data


# ---------------------------------------------------------------------------
# Mesh building
# ---------------------------------------------------------------------------


def _build_foam_mesh(
    volume_elements: List[UnvElement],
    node_id_map: Dict[int, int],
) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
    """Build OpenFOAM mesh connectivity from volume elements.

    Returns:
        Tuple of (all_faces, owner, neighbour).
    """
    # Map face (sorted tuple) -> (face_index, owner_cell)
    face_map: Dict[Tuple[int, ...], Tuple[int, int]] = {}
    all_faces: List[List[int]] = []
    face_owners: Dict[int, int] = {}
    face_neighbours: Dict[int, int] = {}

    for cell_idx, elem in enumerate(volume_elements):
        elem_faces = _get_element_faces(elem.elem_type, elem.nodes, node_id_map)

        for face_nodes in elem_faces:
            face_key = tuple(sorted(face_nodes))
            if face_key in face_map:
                face_idx, _ = face_map[face_key]
                face_neighbours[face_idx] = cell_idx
            else:
                face_idx = len(all_faces)
                face_map[face_key] = (face_idx, cell_idx)
                face_owners[face_idx] = cell_idx
                all_faces.append(face_nodes)

    # Build tensors
    n_faces = len(all_faces)
    owner = torch.zeros(n_faces, dtype=torch.int64, device=get_device())
    for i in range(n_faces):
        owner[i] = face_owners.get(i, 0)

    internal_faces = sorted(face_neighbours.keys())
    neighbour = torch.tensor(
        [face_neighbours[i] for i in internal_faces],
        dtype=torch.int64,
        device=get_device(),
    )

    return all_faces, owner, neighbour


def _get_element_faces(
    elem_type: int,
    node_ids: List[int],
    node_id_map: Dict[int, int],
) -> List[List[int]]:
    """Get faces of a volume element using 0-based node indices."""
    verts = [node_id_map[n] for n in node_ids]

    if elem_type in (22, 45, 111, 115):  # Tet4
        return [
            [verts[0], verts[1], verts[2]],
            [verts[0], verts[1], verts[3]],
            [verts[0], verts[2], verts[3]],
            [verts[1], verts[2], verts[3]],
        ]
    elif elem_type in (24, 44, 112, 116):  # Hex8
        return [
            [verts[0], verts[1], verts[5], verts[4]],
            [verts[2], verts[3], verts[7], verts[6]],
            [verts[0], verts[1], verts[3], verts[2]],
            [verts[4], verts[5], verts[7], verts[6]],
            [verts[0], verts[4], verts[6], verts[2]],
            [verts[1], verts[5], verts[7], verts[3]],
        ]
    elif elem_type in (23, 61, 113, 117):  # Wedge6
        return [
            [verts[0], verts[1], verts[4], verts[3]],
            [verts[1], verts[2], verts[5], verts[4]],
            [verts[0], verts[2], verts[5], verts[3]],
            [verts[0], verts[2], verts[1]],
            [verts[3], verts[4], verts[5]],
        ]
    elif elem_type in (114, 118):  # Pyramid5
        return [
            [verts[0], verts[1], verts[4]],
            [verts[1], verts[2], verts[4]],
            [verts[2], verts[3], verts[4]],
            [verts[0], verts[3], verts[4]],
            [verts[0], verts[1], verts[2], verts[3]],
        ]
    elif elem_type == 25:  # Tet10 (second-order, use corner nodes)
        return [
            [verts[0], verts[1], verts[2]],
            [verts[0], verts[1], verts[3]],
            [verts[0], verts[2], verts[3]],
            [verts[1], verts[2], verts[3]],
        ]
    else:
        # Fallback: treat as single face with all vertices
        return [verts]


def _build_boundary_patches(
    groups: List[UnvGroup],
    boundary_elements: Dict[int, List[UnvElement]],
    all_faces: List[List[int]],
    node_id_map: Dict[int, int],
) -> List[BoundaryPatch]:
    """Build boundary patches from UNV groups or surface elements."""
    # Build face lookup
    face_lookup: Dict[Tuple[int, ...], int] = {}
    for i, face in enumerate(all_faces):
        face_lookup[tuple(sorted(face))] = i

    patches: List[BoundaryPatch] = []
    n_all_faces = len(all_faces)

    # Prefer groups for boundary definition
    if groups:
        used_faces: set[int] = set()
        current_start = n_all_faces

        for group in groups:
            # Collect face nodes from element-type entities
            boundary_face_indices: List[int] = []
            for entity_type, entity_id in group.entities:
                if entity_type == 7:  # Element
                    # Find the element and check if it's a surface element
                    for elem_list in boundary_elements.values():
                        for elem in elem_list:
                            if elem.elem_id == entity_id:
                                verts = [node_id_map[n] for n in elem.nodes]
                                face_key = tuple(sorted(verts))
                                if face_key in face_lookup:
                                    fi = face_lookup[face_key]
                                    if fi not in used_faces:
                                        boundary_face_indices.append(fi)
                                        used_faces.add(fi)

            if boundary_face_indices:
                patches.append(BoundaryPatch(
                    name=group.name.strip(),
                    patch_type="patch",
                    n_faces=len(boundary_face_indices),
                    start_face=current_start,
                ))
                current_start += len(boundary_face_indices)

    # Fallback to physical property grouping
    if not patches and boundary_elements:
        current_start = n_all_faces
        for pg_id in sorted(boundary_elements.keys()):
            elems = boundary_elements[pg_id]
            name = f"patch_{pg_id}"

            face_indices: List[int] = []
            for elem in elems:
                verts = [node_id_map[n] for n in elem.nodes]
                face_key = tuple(sorted(verts))
                if face_key in face_lookup:
                    face_indices.append(face_lookup[face_key])

            if face_indices:
                patches.append(BoundaryPatch(
                    name=name,
                    patch_type="patch",
                    n_faces=len(face_indices),
                    start_face=current_start,
                ))
                current_start += len(face_indices)

    return patches


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def _write_polymesh(
    poly_mesh_dir: Path,
    mesh_data: MeshData,
    *,
    overwrite: bool = False,
) -> None:
    """Write polyMesh files from MeshData."""
    from pyfoam.io.foam_file import FoamFileHeader, FileFormat
    from pyfoam.io.mesh_io import (
        write_boundary,
        write_faces,
        write_neighbour,
        write_owner,
        write_points,
    )

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

    write_points(
        poly_mesh_dir / "points", header_points, mesh_data.points, overwrite=overwrite
    )
    write_faces(
        poly_mesh_dir / "faces", header_faces, mesh_data.faces, overwrite=overwrite
    )
    write_owner(
        poly_mesh_dir / "owner", header_owner, mesh_data.owner, overwrite=overwrite
    )
    write_neighbour(
        poly_mesh_dir / "neighbour",
        header_neighbour,
        mesh_data.neighbour,
        overwrite=overwrite,
    )
    write_boundary(
        poly_mesh_dir / "boundary",
        header_boundary,
        mesh_data.boundary,
        overwrite=overwrite,
    )
