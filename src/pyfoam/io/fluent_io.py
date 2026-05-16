"""
Fluent mesh file I/O and fluentMeshToFoam converter.

Parses ANSYS Fluent ``.msh`` format (ASCII) and converts to
OpenFOAM polyMesh format.

Fluent MSH format structure::

    (0 "Fluent mesh file")
    (2 2 2 3)                    ; dimensions
    (10 zone_id first_id last_id type)  ; cells/zones
    (12 zone_id first_id last_id type)
    (13 zone_id first_id last_id type)  ; faces
    (39 zone_id first_id last_id)       ; mixed cells
    (45 zone_id (node1 x1 y1 z1) ...)  ; nodes
    (  n_zones)
    ...

Zone types:
    0: interior
    2: wall
    3: pressure-inlet
    4: pressure-outlet
    5: symmetry
    7: velocity-inlet
    8: periodic-shadow
    9: periodic
    12: fan
    14: interface
    20: mass-flow-inlet
    24: axis
    31: parent (hanging node)
    36: pressure-far-field
    37: outflow
    61: elasto-taper
    63: grid-interface

Cell zone types (section 10):
    0: dead
    1: fluid
    2: solid

Element types in face section (section 13):
    0: mixed
    2: linear (3-node triangle)
    3: linear (4-node quad)
    4: linear (4-node tet)
    5: linear (5-node pyramid)
    6: linear (6-node wedge/prism)
    7: linear (8-node hex)

References
----------
- Fluent mesh format documentation
- OpenFOAM ``fluentMeshToFoam`` utility source
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
    "FluentZone",
    "FluentFace",
    "FluentMesh",
    "read_fluent",
    "fluent_to_foam",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FluentZone:
    """A Fluent zone definition.

    Attributes:
        zone_id: Zone ID.
        first_id: First entity ID in this zone.
        last_id: Last entity ID in this zone.
        zone_type: Zone type code.
        name: Zone name (from ``zone_name`` section if available).
    """

    zone_id: int
    first_id: int
    last_id: int
    zone_type: int
    name: str = ""


@dataclass
class FluentFace:
    """A single Fluent face.

    Attributes:
        face_id: Face ID (0-based).
        node_ids: Node IDs (0-based).
        c0: Owner cell ID (0-based).
        c1: Neighbour cell ID (0-based, -1 for boundary).
        zone_id: Zone this face belongs to.
    """

    face_id: int
    node_ids: List[int]
    c0: int
    c1: int
    zone_id: int


@dataclass
class FluentMesh:
    """Parsed Fluent mesh.

    Attributes:
        dim: Spatial dimension (2 or 3).
        node_coords: Node coordinates, shape ``(n_nodes, 3)``.
        nodes_per_cell: For mixed cells, the node count per cell.
        cell_zones: Cell zone definitions.
        face_zones: Face zone definitions.
        faces: List of parsed faces.
        n_cells: Total number of cells.
    """

    dim: int
    node_coords: np.ndarray  # (n_nodes, 3)
    cell_zones: List[FluentZone]
    face_zones: List[FluentZone]
    faces: List[FluentFace]
    n_cells: int


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def read_fluent(path: Union[str, Path]) -> FluentMesh:
    """Read a Fluent ``.msh`` file (ASCII format).

    Args:
        path: Path to the ``.msh`` file.

    Returns:
        Parsed :class:`FluentMesh`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    return _parse_fluent(content)


def _parse_fluent(content: str) -> FluentMesh:
    """Parse Fluent MSH content."""
    dim = 3
    node_coords: Optional[np.ndarray] = None
    cell_zones: List[FluentZone] = []
    face_zones: List[FluentZone] = []
    faces: List[FluentFace] = []
    n_cells = 0

    # Parse all parenthesized sections
    # Pattern: (section_id args... body...)
    # The content has sections like:
    # (0 "comment")
    # (2 2 2 3)
    # (10 zone first last type (cells...))
    # (13 zone first last type (faces...))

    # Remove comments
    content = re.sub(r";[^\n]*", "", content)

    # Split into sections at top-level parentheses
    sections = _split_sections(content)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Parse section header
        m = re.match(r"\((\d+)\s+(.*)", section, re.DOTALL)
        if m is None:
            continue

        section_id = int(m.group(1))
        rest = m.group(2)

        if section_id == 0:
            # Comment, skip
            continue

        elif section_id == 2:
            # Dimensions: (2 2 2 3)
            parts = rest.strip().rstrip(")").strip().split()
            if len(parts) >= 3:
                dim = int(parts[-1].rstrip(")"))

        elif section_id == 10:
            # Cells zone
            zone = _parse_zone_section(rest, section_id)
            if zone is not None:
                cell_zones.append(zone)
                n_cells = max(n_cells, zone.last_id)

        elif section_id == 12:
            # Cells zone (alternative)
            zone = _parse_zone_section(rest, section_id)
            if zone is not None:
                cell_zones.append(zone)
                n_cells = max(n_cells, zone.last_id)

        elif section_id == 13:
            # Faces
            zone_info, face_data = _parse_face_section(rest)
            if zone_info is not None:
                face_zones.append(zone_info)
            faces.extend(face_data)

        elif section_id == 39:
            # Mixed cells (additional info)
            pass

        elif section_id == 45 or section_id == 46:
            # Nodes (section 45 = 2D, 46 = 3D)
            node_coords = _parse_node_section(rest)

        elif section_id == 58 or section_id == 59:
            # Nodes with coordinates (alternative format)
            if node_coords is None:
                node_coords = _parse_node_section(rest)

    if node_coords is None:
        raise ValueError("No node coordinates found in Fluent mesh")

    if not faces:
        raise ValueError("No faces found in Fluent mesh")

    return FluentMesh(
        dim=dim,
        node_coords=node_coords,
        cell_zones=cell_zones,
        face_zones=face_zones,
        faces=faces,
        n_cells=n_cells,
    )


def _split_sections(content: str) -> List[str]:
    """Split content into top-level parenthesized sections."""
    sections: List[str] = []
    depth = 0
    start = 0

    for i, c in enumerate(content):
        if c == "(":
            if depth == 0:
                start = i
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                sections.append(content[start : i + 1])

    return sections


def _parse_zone_section(rest: str, section_id: int) -> Optional[FluentZone]:
    """Parse a zone section (10, 12, etc.)."""
    # Format: zone_id first_id last_id type
    # or: zone_id first_id last_id type (data...)
    rest = rest.rstrip(")")
    parts = rest.split("(")[0].strip().split()

    if len(parts) < 4:
        return None

    zone_id = int(parts[0])
    first_id = int(parts[1])
    last_id = int(parts[2])
    zone_type = int(parts[3])

    return FluentZone(
        zone_id=zone_id,
        first_id=first_id,
        last_id=last_id,
        zone_type=zone_type,
    )


def _parse_face_section(
    rest: str,
) -> Tuple[Optional[FluentZone], List[FluentFace]]:
    """Parse a face section (13).

    Format::

        zone_id first_id last_id type
        (n0 n1 n2 c0 c1)
        ...

    or for mixed::

        zone_id first_id last_id 0
        (n_nodes n0 n1 ... c0 c1)
        ...
    """
    # Split header from body
    paren_idx = rest.find("(")
    if paren_idx == -1:
        return None, []

    header_part = rest[:paren_idx].strip().rstrip(")")
    body_part = rest[paren_idx:]

    # Parse header
    parts = header_part.split()
    if len(parts) < 4:
        return None, []

    zone_id = int(parts[0])
    first_id = int(parts[1])
    last_id = int(parts[2])
    elem_type = int(parts[3])

    zone = FluentZone(
        zone_id=zone_id,
        first_id=first_id,
        last_id=last_id,
        zone_type=0,  # Will be filled later
    )

    # Parse face data
    faces: List[FluentFace] = []
    face_pattern = re.compile(r"\(([^)]+)\)")
    for m in face_pattern.finditer(body_part):
        values = [int(v) for v in m.group(1).split()]
        if not values:
            continue

        if elem_type == 0:
            # Mixed: first value is n_nodes
            n_nodes = values[0]
            node_ids = values[1 : 1 + n_nodes]
            c0 = values[1 + n_nodes]
            c1 = values[2 + n_nodes] if len(values) > 2 + n_nodes else -1
        elif elem_type == 2:
            # 3-node triangle
            node_ids = values[:3]
            c0 = values[3]
            c1 = values[4] if len(values) > 4 else -1
        elif elem_type == 3:
            # 4-node quad
            node_ids = values[:4]
            c0 = values[4]
            c1 = values[5] if len(values) > 5 else -1
        elif elem_type == 4:
            # 4-node tet
            node_ids = values[:4]
            c0 = values[4]
            c1 = values[5] if len(values) > 5 else -1
        else:
            # Unknown type, try generic parsing
            # Assume last two are cell indices
            if len(values) >= 3:
                c0 = values[-2]
                c1 = values[-1] if len(values) > 2 else -1
                node_ids = values[:-2]
            else:
                continue

        faces.append(FluentFace(
            face_id=first_id + len(faces),
            node_ids=node_ids,
            c0=c0,
            c1=c1,
            zone_id=zone_id,
        ))

    return zone, faces


def _parse_node_section(rest: str) -> Optional[np.ndarray]:
    """Parse a node section (45, 46, 58, 59).

    Format::

        zone_id first_id last_id dim
        (x1 y1 z1)
        (x2 y2 z2)
        ...

    or:

        zone_id first_id last_id dim
        x1 y1 z1
        x2 y2 z2
        ...
    """
    paren_idx = rest.find("(")
    if paren_idx == -1:
        return None

    header_part = rest[:paren_idx].strip().rstrip(")")
    body_part = rest[paren_idx:]

    # Parse header
    parts = header_part.split()
    if len(parts) < 4:
        return None

    first_id = int(parts[1])
    last_id = int(parts[2])
    n_nodes = last_id - first_id + 1

    # Parse coordinates
    coords: List[List[float]] = []

    # Try parenthesized format first: (x y z)
    paren_pattern = re.compile(r"\(([^)]+)\)")
    matches = list(paren_pattern.finditer(body_part))

    if matches:
        for m in matches:
            values = [float(v) for v in m.group(1).split()]
            if len(values) >= 3:
                coords.append(values[:3])
    else:
        # Try space-separated format
        for line in body_part.strip().split("\n"):
            values = [float(v) for v in line.strip().split() if v.strip()]
            if len(values) >= 3:
                coords.append(values[:3])

    if not coords:
        return None

    return np.array(coords, dtype=np.float64)


# ---------------------------------------------------------------------------
# fluentMeshToFoam conversion
# ---------------------------------------------------------------------------


def fluent_to_foam(
    fluent_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    overwrite: bool = False,
) -> MeshData:
    """Convert a Fluent ``.msh`` file to OpenFOAM polyMesh format.

    Reads the Fluent mesh and writes the 5 polyMesh files:
    ``points``, ``faces``, ``owner``, ``neighbour``, ``boundary``.

    Args:
        fluent_path: Path to the Fluent ``.msh`` file.
        output_dir: Output directory for polyMesh files.
            The ``constant/polyMesh`` subdirectory will be created.
        overwrite: If True, overwrite existing files.

    Returns:
        The converted :class:`MeshData`.
    """
    fluent_path = Path(fluent_path)
    output_dir = Path(output_dir)
    poly_mesh_dir = output_dir / "constant" / "polyMesh"
    poly_mesh_dir.mkdir(parents=True, exist_ok=True)

    # Parse Fluent mesh
    fluent = read_fluent(fluent_path)

    # Convert node coordinates
    points = torch.tensor(
        fluent.node_coords, dtype=get_default_dtype(), device=get_device()
    )

    # Build faces and connectivity
    # Fluent faces already have owner (c0) and neighbour (c1) info
    all_faces: List[List[int]] = []
    owner_list: List[int] = []
    neighbour_list: List[int] = []

    # Separate boundary and internal faces
    boundary_zone_faces: Dict[int, List[int]] = {}  # zone_id -> face indices

    for face in fluent.faces:
        face_idx = len(all_faces)
        all_faces.append(face.node_ids)
        owner_list.append(face.c0)

        if face.c1 >= 0:
            # Internal face
            neighbour_list.append(face.c1)
        else:
            # Boundary face
            if face.zone_id not in boundary_zone_faces:
                boundary_zone_faces[face.zone_id] = []
            boundary_zone_faces[face.zone_id].append(face_idx)

    # Separate internal and boundary faces
    # Internal faces must come first in OpenFOAM format
    internal_faces: List[List[int]] = []
    internal_owner: List[int] = []
    internal_neighbour: List[int] = []
    boundary_faces: List[List[int]] = []
    boundary_owner: List[int] = []

    for i, face in enumerate(all_faces):
        if i < len(neighbour_list) and i < len(owner_list):
            # This is an internal face (has both owner and neighbour)
            if len(neighbour_list) > 0:
                internal_faces.append(face)
                internal_owner.append(owner_list[i])
                if i < len(neighbour_list):
                    internal_neighbour.append(neighbour_list[i])

    # Rebuild with proper ordering
    # Actually, let's just use the face data directly
    # Fluent format already gives us c0 (owner) and c1 (neighbour)
    internal_faces = []
    internal_owner = []
    internal_neighbour = []
    boundary_faces_list = []
    boundary_owner_list = []
    boundary_zone_map: Dict[int, List[int]] = {}

    face_counter = 0
    for face in fluent.faces:
        if face.c1 >= 0:
            # Internal face
            internal_faces.append(face.node_ids)
            internal_owner.append(face.c0)
            internal_neighbour.append(face.c1)
        else:
            # Boundary face
            boundary_faces_list.append(face.node_ids)
            boundary_owner_list.append(face.c0)
            if face.zone_id not in boundary_zone_map:
                boundary_zone_map[face.zone_id] = []
            boundary_zone_map[face.zone_id].append(face_counter)
        face_counter += 1

    # Combine: internal faces first, then boundary faces
    all_faces_ordered = internal_faces + boundary_faces_list
    all_owner = internal_owner + boundary_owner_list

    owner = torch.tensor(all_owner, dtype=torch.int64, device=get_device())
    neighbour = torch.tensor(
        internal_neighbour, dtype=torch.int64, device=get_device()
    )

    # Build boundary patches
    boundary_patches: List[BoundaryPatch] = []
    n_internal = len(internal_faces)

    for zone_id in sorted(boundary_zone_map.keys()):
        face_indices = boundary_zone_map[zone_id]
        if not face_indices:
            continue

        # Find zone name
        zone_name = f"zone_{zone_id}"
        for fz in fluent.face_zones:
            if fz.zone_id == zone_id and fz.name:
                zone_name = fz.name
                break

        # Determine patch type from zone type
        patch_type = "patch"
        for fz in fluent.face_zones:
            if fz.zone_id == zone_id:
                if fz.zone_type == 2:
                    patch_type = "wall"
                elif fz.zone_type == 5:
                    patch_type = "symmetry"
                elif fz.zone_type in (3, 7, 20):
                    patch_type = "patch"
                elif fz.zone_type in (4, 37):
                    patch_type = "patch"
                break

        boundary_patches.append(BoundaryPatch(
            name=zone_name,
            patch_type=patch_type,
            n_faces=len(face_indices),
            start_face=n_internal + face_indices[0] if face_indices else 0,
        ))

    # Convert faces to numpy arrays
    faces_np = [np.array(f, dtype=np.int32) for f in all_faces_ordered]

    mesh_data = MeshData(
        points=points,
        faces=faces_np,
        owner=owner,
        neighbour=neighbour,
        boundary=boundary_patches,
    )

    # Write polyMesh files
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
