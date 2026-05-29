"""
ansysToFoam — convert ANSYS mesh input to OpenFOAM polyMesh.

Parses ANSYS input file (``.inp`` / ``.cdb``) format and writes the 5
OpenFOAM polyMesh files: ``points``, ``faces``, ``owner``, ``neighbour``,
``boundary``.

ANSYS input format keywords::

    *NODE        — node definitions: id, x, y, z
    *ELEMENT     — element definitions: id, type, n1, n2, ...
    *ELSET       — element sets (used as cell zones / boundary patches)
    *NSET        — node sets

Element types (CPS/D/3D family):
    SOLID45  → 8-node hex
    SOLID92  → 10-node tet (use corner nodes)
    SOLID95  → 20-node hex (use corner nodes)
    SOLID185 → 8-node hex
    SOLID187 → 10-node tet
    PLANE42  → 4-node quad (2D)
    SHELL63  → 4-node quad shell

References
----------
- ANSYS input file format documentation
- OpenFOAM ``ansysToFoam`` utility source
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
    "AnsysElement",
    "AnsysMesh",
    "read_ansys",
    "ansys_to_foam",
]


# ---------------------------------------------------------------------------
# ANSYS element type definitions
# ---------------------------------------------------------------------------

# Maps ANSYS element type name -> (n_nodes, is_volume)
_ANSYS_ELEMENT_INFO: Dict[str, Tuple[int, bool]] = {
    "SOLID45": (8, True),
    "SOLID73": (8, True),
    "SOLID92": (10, True),
    "SOLID95": (20, True),
    "SOLID185": (8, True),
    "SOLID186": (20, True),
    "SOLID187": (10, True),
    "SOLID285": (4, True),   # pyramid
    "PLANE42": (4, False),
    "PLANE82": (8, False),
    "SHELL63": (4, False),
    "SHELL181": (4, False),
    "SHELL281": (8, False),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class AnsysElement:
    """An ANSYS element.

    Attributes:
        elem_id: Element ID.
        elem_type: Element type name (e.g., ``SOLID45``).
        nodes: Node IDs.
    """

    def __init__(
        self, elem_id: int, elem_type: str, nodes: List[int]
    ) -> None:
        self.elem_id = elem_id
        self.elem_type = elem_type
        self.nodes = nodes


class AnsysMesh:
    """Parsed ANSYS mesh.

    Attributes:
        coords: Node coordinates array, shape ``(n_nodes, 3)``.
        node_id_map: Maps ANSYS node ID -> 0-based index.
        elements: List of parsed elements.
        element_sets: Dict mapping set name -> list of element IDs.
        node_sets: Dict mapping set name -> list of node IDs.
    """

    def __init__(
        self,
        coords: np.ndarray,
        node_id_map: Dict[int, int],
        elements: List[AnsysElement],
        element_sets: Dict[str, List[int]],
        node_sets: Dict[str, List[int]],
    ) -> None:
        self.coords = coords
        self.node_id_map = node_id_map
        self.elements = elements
        self.element_sets = element_sets
        self.node_sets = node_sets


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def read_ansys(path: Union[str, Path]) -> AnsysMesh:
    """Read an ANSYS input file (``.inp`` / ``.cdb``).

    Args:
        path: Path to the ANSYS input file.

    Returns:
        Parsed :class:`AnsysMesh`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no nodes or elements found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    return _parse_ansys(content)


def _parse_ansys(content: str) -> AnsysMesh:
    """Parse ANSYS input file content."""
    # Split into records by * keywords
    records = _split_records(content)

    # Parse nodes
    node_coords: Dict[int, Tuple[float, float, float]] = {}
    elements: List[AnsysElement] = []
    element_sets: Dict[str, List[int]] = {}
    node_sets: Dict[str, List[int]] = {}

    # Current element type context
    current_elem_type = "UNKNOWN"

    for keyword, body in records:
        kw_upper = keyword.upper().strip()

        if kw_upper.startswith("*NODE") and "OUTPUT" not in kw_upper.upper():
            _parse_nodes(body, node_coords)
        elif kw_upper.startswith("*ELEMENT"):
            current_elem_type = _parse_element_type(kw_upper)
            _parse_elements(body, current_elem_type, elements)
        elif kw_upper.startswith("*ELSET"):
            set_name = _parse_set_name(kw_upper, "ELSET")
            set_ids = _parse_set_body(body, element_sets, set_name)
            if set_name not in element_sets:
                element_sets[set_name] = []
            element_sets[set_name].extend(set_ids)
        elif kw_upper.startswith("*NSET"):
            set_name = _parse_set_name(kw_upper, "NSET")
            set_ids = _parse_set_body(body, node_sets, set_name)
            if set_name not in node_sets:
                node_sets[set_name] = []
            node_sets[set_name].extend(set_ids)

    if not node_coords:
        raise ValueError("No nodes (*NODE) found in ANSYS file")
    if not elements:
        raise ValueError("No elements (*ELEMENT) found in ANSYS file")

    # Build compact arrays
    sorted_ids = sorted(node_coords.keys())
    node_id_map: Dict[int, int] = {}
    coords_list: List[List[float]] = []
    for idx, nid in enumerate(sorted_ids):
        node_id_map[nid] = idx
        c = node_coords[nid]
        coords_list.append([c[0], c[1], c[2]])

    return AnsysMesh(
        coords=np.array(coords_list, dtype=np.float64),
        node_id_map=node_id_map,
        elements=elements,
        element_sets=element_sets,
        node_sets=node_sets,
    )


def _split_records(content: str) -> List[Tuple[str, str]]:
    """Split content into (keyword, body) records at * keyword lines."""
    records: List[Tuple[str, str]] = []
    lines = content.splitlines()
    current_keyword: Optional[str] = None
    current_body_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("*"):
            # Save previous record
            if current_keyword is not None:
                records.append((current_keyword, "\n".join(current_body_lines)))
            current_keyword = stripped
            current_body_lines = []
        else:
            current_body_lines.append(line)

    if current_keyword is not None:
        records.append((current_keyword, "\n".join(current_body_lines)))

    return records


def _parse_nodes(
    body: str,
    node_coords: Dict[int, Tuple[float, float, float]],
) -> None:
    """Parse *NODE body lines.

    Format: id, x, y, z
    """
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove trailing comments
        line = line.split("!")[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            nid = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            node_coords[nid] = (x, y, z)
        except ValueError:
            continue


def _parse_element_type(keyword_line: str) -> str:
    """Extract element type from *ELEMENT keyword line."""
    for part in keyword_line.split(","):
        part = part.strip()
        if part.upper().startswith("TYPE="):
            return part.split("=", 1)[1].strip()
    return "SOLID185"  # default


def _parse_elements(
    body: str, elem_type: str, elements: List[AnsysElement],
) -> None:
    """Parse *ELEMENT body lines.

    Format: id, n1, n2, n3, n4, [n5, n6, n7, n8, ...]
    """
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.split("!")[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            elem_id = int(parts[0])
            node_ids = [int(p) for p in parts[1:] if p.strip()]
            if node_ids:
                elements.append(AnsysElement(
                    elem_id=elem_id,
                    elem_type=elem_type,
                    nodes=node_ids,
                ))
        except ValueError:
            continue


def _parse_set_name(keyword_line: str, keyword: str) -> str:
    """Extract set name from *ELSET or *NSET keyword line."""
    for part in keyword_line.split(","):
        part = part.strip()
        if part.upper().startswith(f"{keyword}="):
            return part.split("=", 1)[1].strip()
    return "unnamed"


def _parse_set_body(
    body: str,
    existing_sets: Dict[str, List[int]],
    set_name: str,
) -> List[int]:
    """Parse set body lines. Supports GENERATE format."""
    ids: List[int] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.split("!")[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]

        # Check for GENERATE
        if len(parts) >= 3:
            try:
                start = int(parts[0])
                end = int(parts[1])
                step = int(parts[2]) if len(parts) > 2 else 1
                ids.extend(range(start, end + 1, step))
                continue
            except ValueError:
                pass

        for p in parts:
            p = p.strip()
            if not p:
                continue
            # Reference to another set
            if p.upper() in existing_sets:
                ids.extend(existing_sets[p.upper()])
            else:
                try:
                    ids.append(int(p))
                except ValueError:
                    pass

    return ids


# ---------------------------------------------------------------------------
# ansysToFoam conversion
# ---------------------------------------------------------------------------


def ansys_to_foam(
    ansys_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    overwrite: bool = False,
) -> MeshData:
    """Convert an ANSYS input file to OpenFOAM polyMesh format.

    Args:
        ansys_path: Path to the ANSYS input file (``.inp`` / ``.cdb``).
        output_dir: Output directory for polyMesh files.
            The ``constant/polyMesh`` subdirectory will be created.
        overwrite: If True, overwrite existing files.

    Returns:
        The converted :class:`MeshData`.
    """
    ansys_path = Path(ansys_path)
    output_dir = Path(output_dir)
    poly_mesh_dir = output_dir / "constant" / "polyMesh"
    poly_mesh_dir.mkdir(parents=True, exist_ok=True)

    # Parse ANSYS mesh
    ansys = read_ansys(ansys_path)

    # Convert node coordinates
    points = torch.tensor(ansys.coords, dtype=get_default_dtype(), device=get_device())

    # Separate volume and boundary elements
    volume_elements: List[AnsysElement] = []
    boundary_sets: Dict[str, List[AnsysElement]] = {}
    elem_id_to_elem: Dict[int, AnsysElement] = {}

    for elem in ansys.elements:
        elem_id_to_elem[elem.elem_id] = elem
        info = _ANSYS_ELEMENT_INFO.get(elem.elem_type.upper())
        if info is not None and info[1]:
            volume_elements.append(elem)
        # Non-volume elements not directly added — handled via element sets

    # Build faces, owner, neighbour from volume elements
    all_faces, owner, neighbour = _build_foam_mesh(volume_elements, ansys.node_id_map)

    # Build boundary patches from element sets
    boundary_patches = _build_boundary_patches(
        ansys.element_sets, elem_id_to_elem, all_faces, ansys.node_id_map,
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
    volume_elements: List[AnsysElement],
    node_id_map: Dict[int, int],
) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
    """Build OpenFOAM mesh connectivity from volume elements."""
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
    elem_type: str,
    node_ids: List[int],
    node_id_map: Dict[int, int],
) -> List[List[int]]:
    """Get faces of a volume element."""
    verts = [node_id_map[n] for n in node_ids]
    etype = elem_type.upper()

    n_verts = len(verts)

    # Tetrahedron types
    if etype in ("SOLID92", "SOLID187"):
        # 10-node tet, use corner nodes 0-3
        return [
            [verts[0], verts[1], verts[2]],
            [verts[0], verts[1], verts[3]],
            [verts[0], verts[2], verts[3]],
            [verts[1], verts[2], verts[3]],
        ]
    if n_verts == 4:
        return [
            [verts[0], verts[1], verts[2]],
            [verts[0], verts[1], verts[3]],
            [verts[0], verts[2], verts[3]],
            [verts[1], verts[2], verts[3]],
        ]

    # Hex types
    if etype in ("SOLID45", "SOLID73", "SOLID185"):
        return [
            [verts[0], verts[1], verts[5], verts[4]],
            [verts[2], verts[3], verts[7], verts[6]],
            [verts[0], verts[1], verts[3], verts[2]],
            [verts[4], verts[5], verts[7], verts[6]],
            [verts[0], verts[4], verts[6], verts[2]],
            [verts[1], verts[5], verts[7], verts[3]],
        ]
    if etype in ("SOLID95", "SOLID186"):
        # 20-node hex, use corner nodes
        return [
            [verts[0], verts[1], verts[5], verts[4]],
            [verts[2], verts[3], verts[7], verts[6]],
            [verts[0], verts[1], verts[3], verts[2]],
            [verts[4], verts[5], verts[7], verts[6]],
            [verts[0], verts[4], verts[6], verts[2]],
            [verts[1], verts[5], verts[7], verts[3]],
        ]
    if n_verts == 8:
        return [
            [verts[0], verts[1], verts[5], verts[4]],
            [verts[2], verts[3], verts[7], verts[6]],
            [verts[0], verts[1], verts[3], verts[2]],
            [verts[4], verts[5], verts[7], verts[6]],
            [verts[0], verts[4], verts[6], verts[2]],
            [verts[1], verts[5], verts[7], verts[3]],
        ]

    # Pyramid
    if n_verts == 5:
        return [
            [verts[0], verts[1], verts[4]],
            [verts[1], verts[2], verts[4]],
            [verts[2], verts[3], verts[4]],
            [verts[0], verts[3], verts[4]],
            [verts[0], verts[1], verts[2], verts[3]],
        ]

    # Wedge
    if n_verts == 6:
        return [
            [verts[0], verts[1], verts[4], verts[3]],
            [verts[1], verts[2], verts[5], verts[4]],
            [verts[0], verts[2], verts[5], verts[3]],
            [verts[0], verts[2], verts[1]],
            [verts[3], verts[4], verts[5]],
        ]

    # Fallback
    return [verts]


def _build_boundary_patches(
    element_sets: Dict[str, List[int]],
    elem_id_to_elem: Dict[int, AnsysElement],
    all_faces: List[List[int]],
    node_id_map: Dict[int, int],
) -> List[BoundaryPatch]:
    """Build boundary patches from ANSYS element sets."""
    face_lookup: Dict[Tuple[int, ...], int] = {}
    for i, face in enumerate(all_faces):
        face_lookup[tuple(sorted(face))] = i

    patches: List[BoundaryPatch] = []
    current_start = len(all_faces)

    for set_name in sorted(element_sets.keys()):
        elem_ids = element_sets[set_name]
        # Find elements that are surface (non-volume) type
        face_indices: List[int] = []
        for eid in elem_ids:
            if eid not in elem_id_to_elem:
                continue
            elem = elem_id_to_elem[eid]
            info = _ANSYS_ELEMENT_INFO.get(elem.elem_type.upper())
            if info is not None and info[1]:
                continue  # Skip volume elements
            verts = [node_id_map[n] for n in elem.nodes]
            face_key = tuple(sorted(verts))
            if face_key in face_lookup:
                face_indices.append(face_lookup[face_key])

        if face_indices:
            patches.append(BoundaryPatch(
                name=set_name,
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
