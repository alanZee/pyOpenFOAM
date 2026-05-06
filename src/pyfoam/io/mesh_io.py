"""
OpenFOAM mesh file reading and writing.

Handles the polyMesh directory files:
- ``points`` — N×3 vertex coordinates
- ``faces`` — face-to-vertex connectivity
- ``owner`` — owner cell for each face
- ``neighbour`` — neighbour cell for each internal face
- ``boundary`` — boundary patch definitions (always ASCII)

Supports both ASCII and binary formats.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.binary_io import BinaryReader, BinaryWriter
from pyfoam.io.dictionary import FoamDict, parse_dict
from pyfoam.io.foam_file import (
    FoamFileHeader,
    FileFormat,
    read_foam_file,
    split_header_body,
    write_foam_file,
)

__all__ = [
    "MeshData",
    "BoundaryPatch",
    "read_points",
    "read_faces",
    "read_owner",
    "read_neighbour",
    "read_boundary",
    "write_points",
    "write_faces",
    "write_owner",
    "write_neighbour",
    "write_boundary",
    "read_mesh",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class BoundaryPatch:
    """A boundary patch definition from the ``boundary`` file.

    Attributes:
        name: Patch name.
        patch_type: Patch type (e.g., ``wall``, ``patch``, ``inlet``).
        n_faces: Number of faces in this patch.
        start_face: Starting face index in the global face list.
        in_groups: Optional group membership list.
    """

    def __init__(
        self,
        name: str,
        patch_type: str = "patch",
        n_faces: int = 0,
        start_face: int = 0,
        in_groups: Optional[list[str]] = None,
    ) -> None:
        self.name = name
        self.patch_type = patch_type
        self.n_faces = n_faces
        self.start_face = start_face
        self.in_groups = in_groups or []

    def __repr__(self) -> str:
        return (
            f"BoundaryPatch(name={self.name!r}, type={self.patch_type!r}, "
            f"n_faces={self.n_faces}, start_face={self.start_face})"
        )


class MeshData:
    """Complete mesh data read from a polyMesh directory.

    Attributes:
        points: Vertex coordinates, shape ``(n_points, 3)``.
        faces: List of face vertex-index arrays.
        owner: Owner cell indices, shape ``(n_faces,)``.
        neighbour: Neighbour cell indices (internal faces only).
        boundary: List of boundary patch definitions.
        n_cells: Total number of cells.
        n_internal_faces: Number of internal faces.
    """

    def __init__(
        self,
        points: torch.Tensor,
        faces: list[np.ndarray],
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        boundary: list[BoundaryPatch],
    ) -> None:
        self.points = points
        self.faces = faces
        self.owner = owner
        self.neighbour = neighbour
        self.boundary = boundary

    @property
    def n_points(self) -> int:
        """Number of vertices."""
        return self.points.shape[0]

    @property
    def n_faces(self) -> int:
        """Total number of faces."""
        return len(self.faces)

    @property
    def n_internal_faces(self) -> int:
        """Number of internal faces (shared between two cells)."""
        return self.neighbour.shape[0]

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return int(self.owner.max().item()) + 1 if self.owner.numel() > 0 else 0

    @property
    def n_boundary_faces(self) -> int:
        """Number of boundary faces."""
        return self.n_faces - self.n_internal_faces

    def __repr__(self) -> str:
        return (
            f"MeshData(n_points={self.n_points}, n_faces={self.n_faces}, "
            f"n_cells={self.n_cells}, n_patches={len(self.boundary)})"
        )


# ---------------------------------------------------------------------------
# Points reading / writing
# ---------------------------------------------------------------------------

_POINTS_COUNT_PATTERN = re.compile(r"^\s*(\d+)", re.MULTILINE)


def _parse_ascii_points(content: str, n: int) -> torch.Tensor:
    """Parse ASCII points from content after the count line.

    Args:
        content: Content between ``(`` and ``)``.
        n: Expected number of points.

    Returns:
        Tensor of shape ``(n, 3)``.
    """
    content = content.strip().strip("()")
    points = []
    for match in re.finditer(r"\(\s*([^)]+)\)", content):
        coords = match.group(1).split()
        points.append([float(c) for c in coords])
    if len(points) != n:
        raise ValueError(f"Expected {n} points, got {len(points)}")
    return torch.tensor(points, dtype=get_default_dtype(), device=get_device())


def read_points(path: Union[str, Path]) -> tuple[FoamFileHeader, torch.Tensor]:
    """Read a ``points`` file.

    Args:
        path: Path to the points file.

    Returns:
        Tuple of (header, points_tensor) with shape ``(n_points, 3)``.
    """
    header, body = read_foam_file(path)

    # Find count
    match = _POINTS_COUNT_PATTERN.search(body)
    if match is None:
        raise ValueError("Cannot find point count")
    n = int(match.group(1))

    if header.is_binary:
        # Binary: N × 3 × 8 bytes (big-endian double)
        paren_start = body.find("(", match.end())
        if paren_start == -1:
            raise ValueError("Cannot find '(' for binary points data")
        binary_start = paren_start + 1
        # Read N*3 doubles from binary data
        raw = body[binary_start:].encode("latin-1")
        reader = BinaryReader(raw)
        arr = reader.read_doubles(n * 3)
        points = torch.tensor(
            arr.reshape(n, 3), dtype=get_default_dtype(), device=get_device()
        )
    else:
        # ASCII
        paren_start = body.find("(", match.end())
        if paren_start == -1:
            raise ValueError("Cannot find '(' for points data")
        paren_end = _find_matching_paren(body, paren_start)
        data_text = body[paren_start:paren_end + 1]
        points = _parse_ascii_points(data_text, n)

    return header, points


def write_points(
    path: Union[str, Path],
    header: FoamFileHeader,
    points: torch.Tensor,
    *,
    overwrite: bool = False,
) -> None:
    """Write a ``points`` file.

    Args:
        path: Output file path.
        header: FoamFile header (format field is updated).
        points: Points tensor of shape ``(n, 3)``.
        overwrite: If False, raise if file exists.
    """
    n = points.shape[0]

    if header.is_binary:
        writer = BinaryWriter()
        writer.write_marker_open()
        writer.write_binary_points(points)
        writer.write_marker_close()
        body = f"{n}\n{writer.get_bytes().decode('latin-1')}"
    else:
        lines = [f"{n}", "("]
        pts = points.detach().cpu().numpy()
        for i in range(n):
            lines.append(f"({pts[i, 0]:.10g} {pts[i, 1]:.10g} {pts[i, 2]:.10g})")
        lines.append(")")
        body = "\n".join(lines)

    write_foam_file(path, header, body, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Faces reading / writing
# ---------------------------------------------------------------------------


def _parse_ascii_faces(content: str, n: int) -> list[np.ndarray]:
    """Parse ASCII faces from content.

    Each face is ``NVertices(v0 v1 v2 ... vN)``.

    Args:
        content: Content between ``(`` and ``)``.
        n: Expected number of faces.

    Returns:
        List of numpy arrays, one per face.
    """
    content = content.strip().strip("()")
    faces: list[np.ndarray] = []
    for match in re.finditer(r"\d+\(([^)]*)\)", content):
        verts = [int(v) for v in match.group(1).split()]
        faces.append(np.array(verts, dtype=np.int32))
    if len(faces) != n:
        raise ValueError(f"Expected {n} faces, got {len(faces)}")
    return faces


def read_faces(path: Union[str, Path]) -> tuple[FoamFileHeader, list[np.ndarray]]:
    """Read a ``faces`` file.

    Args:
        path: Path to the faces file.

    Returns:
        Tuple of (header, faces_list).
    """
    header, body = read_foam_file(path)

    match = _POINTS_COUNT_PATTERN.search(body)
    if match is None:
        raise ValueError("Cannot find face count")
    n = int(match.group(1))

    if header.is_binary:
        # Binary: CompactListList encoding
        paren_start = body.find("(", match.end())
        if paren_start == -1:
            raise ValueError("Cannot find '(' for binary faces data")
        raw = body[paren_start + 1:].encode("latin-1")
        reader = BinaryReader(raw)
        faces = reader.read_binary_compact_list_list()
    else:
        # ASCII
        paren_start = body.find("(", match.end())
        if paren_start == -1:
            raise ValueError("Cannot find '(' for faces data")
        paren_end = _find_matching_paren(body, paren_start)
        data_text = body[paren_start:paren_end + 1]
        faces = _parse_ascii_faces(data_text, n)

    return header, faces


def write_faces(
    path: Union[str, Path],
    header: FoamFileHeader,
    faces: list[np.ndarray],
    *,
    overwrite: bool = False,
) -> None:
    """Write a ``faces`` file.

    Args:
        path: Output file path.
        header: FoamFile header.
        faces: List of face vertex-index arrays.
        overwrite: If False, raise if file exists.
    """
    n = len(faces)

    if header.is_binary:
        writer = BinaryWriter()
        writer.write_marker_open()
        writer.write_binary_compact_list_list(faces)
        writer.write_marker_close()
        body = f"{n}\n{writer.get_bytes().decode('latin-1')}"
    else:
        lines = [f"{n}", "("]
        for i, face in enumerate(faces):
            verts = " ".join(str(v) for v in face)
            lines.append(f"{len(face)}({verts})")
        lines.append(")")
        body = "\n".join(lines)

    write_foam_file(path, header, body, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Owner / Neighbour reading / writing
# ---------------------------------------------------------------------------


def _parse_ascii_label_list(content: str, n: int) -> torch.Tensor:
    """Parse ASCII label (int) list from content.

    Args:
        content: Content between ``(`` and ``)``.
        n: Expected number of values.

    Returns:
        1-D int64 tensor.
    """
    content = content.strip().strip("()")
    values = [int(v) for v in content.split() if v.strip()]
    if len(values) != n:
        raise ValueError(f"Expected {n} labels, got {len(values)}")
    return torch.tensor(values, dtype=torch.int64, device=get_device())


def _read_label_list_file(
    path: Union[str, Path],
) -> tuple[FoamFileHeader, torch.Tensor]:
    """Read a label list file (owner, neighbour).

    Args:
        path: Path to the file.

    Returns:
        Tuple of (header, label_tensor).
    """
    header, body = read_foam_file(path)

    match = _POINTS_COUNT_PATTERN.search(body)
    if match is None:
        raise ValueError("Cannot find label count")
    n = int(match.group(1))

    if header.is_binary:
        paren_start = body.find("(", match.end())
        if paren_start == -1:
            raise ValueError("Cannot find '(' for binary label data")
        raw = body[paren_start + 1:].encode("latin-1")
        reader = BinaryReader(raw)
        arr = reader.read_int32s(n)
        labels = torch.tensor(arr, dtype=torch.int64, device=get_device())
    else:
        paren_start = body.find("(", match.end())
        if paren_start == -1:
            raise ValueError("Cannot find '(' for label data")
        paren_end = _find_matching_paren(body, paren_start)
        data_text = body[paren_start:paren_end + 1]
        labels = _parse_ascii_label_list(data_text, n)

    return header, labels


def read_owner(path: Union[str, Path]) -> tuple[FoamFileHeader, torch.Tensor]:
    """Read an ``owner`` file.

    Args:
        path: Path to the owner file.

    Returns:
        Tuple of (header, owner_tensor) — one cell index per face.
    """
    return _read_label_list_file(path)


def read_neighbour(path: Union[str, Path]) -> tuple[FoamFileHeader, torch.Tensor]:
    """Read a ``neighbour`` file.

    Args:
        path: Path to the neighbour file.

    Returns:
        Tuple of (header, neighbour_tensor) — one cell index per internal face.
    """
    return _read_label_list_file(path)


def _write_label_list_file(
    path: Union[str, Path],
    header: FoamFileHeader,
    labels: torch.Tensor,
    *,
    overwrite: bool = False,
) -> None:
    """Write a label list file (owner or neighbour).

    Args:
        path: Output file path.
        header: FoamFile header.
        labels: 1-D int64 tensor.
        overwrite: If False, raise if file exists.
    """
    n = labels.shape[0]

    if header.is_binary:
        writer = BinaryWriter()
        writer.write_marker_open()
        writer.write_binary_label_list(labels)
        writer.write_marker_close()
        body = f"{n}\n{writer.get_bytes().decode('latin-1')}"
    else:
        lines = [f"{n}", "("]
        for val in labels:
            lines.append(str(val.item()))
        lines.append(")")
        body = "\n".join(lines)

    write_foam_file(path, header, body, overwrite=overwrite)


def write_owner(
    path: Union[str, Path],
    header: FoamFileHeader,
    owner: torch.Tensor,
    *,
    overwrite: bool = False,
) -> None:
    """Write an ``owner`` file."""
    _write_label_list_file(path, header, owner, overwrite=overwrite)


def write_neighbour(
    path: Union[str, Path],
    header: FoamFileHeader,
    neighbour: torch.Tensor,
    *,
    overwrite: bool = False,
) -> None:
    """Write a ``neighbour`` file."""
    _write_label_list_file(path, header, neighbour, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Boundary reading / writing
# ---------------------------------------------------------------------------


def read_boundary(path: Union[str, Path]) -> tuple[FoamFileHeader, list[BoundaryPatch]]:
    """Read a ``boundary`` file.

    The boundary file is always ASCII regardless of the format header.

    Args:
        path: Path to the boundary file.

    Returns:
        Tuple of (header, list_of_patches).
    """
    header, body = read_foam_file(path)

    # Find the patch list count
    match = _POINTS_COUNT_PATTERN.search(body)
    n_patches = int(match.group(1)) if match else 0

    # Parse the patch blocks
    patches: list[BoundaryPatch] = []

    # Find all patch blocks: name { ... }
    block_pattern = re.compile(r"(\w+)\s*\{([^}]*)\}", re.DOTALL)
    for block_match in block_pattern.finditer(body):
        name = block_match.group(1)
        block = block_match.group(2)

        patch_type = "patch"
        n_faces = 0
        start_face = 0
        in_groups: list[str] = []

        # Parse key-value pairs
        kv_pattern = re.compile(r"(\w+)\s+(.+?)\s*;")
        for kv in kv_pattern.finditer(block):
            key = kv.group(1)
            value = kv.group(2).strip().strip('"')
            if key == "type":
                patch_type = value
            elif key == "nFaces":
                n_faces = int(value)
            elif key == "startFace":
                start_face = int(value)
            elif key == "inGroups":
                # Parse group list: (group1 group2 ...)
                value = value.strip("()")
                if value:
                    in_groups = [g.strip().strip('"') for g in value.split()]

        patches.append(BoundaryPatch(
            name=name,
            patch_type=patch_type,
            n_faces=n_faces,
            start_face=start_face,
            in_groups=in_groups,
        ))

    return header, patches


def write_boundary(
    path: Union[str, Path],
    header: FoamFileHeader,
    patches: list[BoundaryPatch],
    *,
    overwrite: bool = False,
) -> None:
    """Write a ``boundary`` file.

    Args:
        path: Output file path.
        header: FoamFile header.
        patches: List of boundary patch definitions.
        overwrite: If False, raise if file exists.
    """
    lines = [f"{len(patches)}", "("]
    for patch in patches:
        lines.append(f"    {patch.name}")
        lines.append("    {")
        lines.append(f"        type            {patch.patch_type};")
        lines.append(f"        nFaces          {patch.n_faces};")
        lines.append(f"        startFace       {patch.start_face};")
        if patch.in_groups:
            groups = " ".join(f'"{g}"' for g in patch.in_groups)
            lines.append(f"        inGroups        ({groups});")
        lines.append("    }")
    lines.append(")")

    body = "\n".join(lines)
    write_foam_file(path, header, body, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Complete mesh reading
# ---------------------------------------------------------------------------


def read_mesh(mesh_dir: Union[str, Path]) -> MeshData:
    """Read all mesh files from a polyMesh directory.

    Args:
        mesh_dir: Path to the ``polyMesh`` directory (or ``constant/polyMesh``).

    Returns:
        :class:`MeshData` with all mesh information.
    """
    mesh_dir = Path(mesh_dir)

    _, points = read_points(mesh_dir / "points")
    _, faces = read_faces(mesh_dir / "faces")
    _, owner = read_owner(mesh_dir / "owner")
    _, neighbour = read_neighbour(mesh_dir / "neighbour")
    _, boundary = read_boundary(mesh_dir / "boundary")

    return MeshData(
        points=points,
        faces=faces,
        owner=owner,
        neighbour=neighbour,
        boundary=boundary,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_matching_paren(text: str, start: int) -> int:
    """Find the matching closing parenthesis."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unmatched parenthesis")
