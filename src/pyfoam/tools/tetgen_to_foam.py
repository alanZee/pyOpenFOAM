"""
tetgenToFoam — convert TetGen mesh to OpenFOAM polyMesh.

Reads TetGen mesh files (``.node``, ``.ele``, ``.face``) and writes
the 5 OpenFOAM polyMesh files: ``points``, ``faces``, ``owner``,
``neighbour``, ``boundary``.

TetGen file formats:

- ``.node`` — Node coordinates
  - Header: ``n_nodes  dimension  n_attributes  n_boundary_markers``
  - Data: ``id  x  y  z  [attribute]  [boundary_marker]``
- ``.ele`` — Tetrahedral elements
  - Header: ``n_elements  n_nodes_per_element  n_attributes``
  - Data: ``id  n1  n2  n3  n4  [attribute]``
- ``.face`` — Boundary faces (triangles)
  - Header: ``n_faces  n_boundary_markers``
  - Data: ``id  n1  n2  n3  [boundary_marker]``

References
----------
- TetGen documentation: http://wias-berlin.de/software/tetgen/
- OpenFOAM ``tetgenToFoam`` utility source
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.mesh_io import BoundaryPatch, MeshData

__all__ = [
    "TetGenMesh",
    "read_tetgen",
    "tetgen_to_foam",
]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

class TetGenMesh:
    """Parsed TetGen mesh.

    Attributes:
        coords: Node coordinates, shape ``(n_nodes, 3)``.
        elements: Element connectivity, shape ``(n_elements, 4)``
            (0-based node indices).
        faces: Boundary face connectivity, shape ``(n_faces, 3)``
            (0-based node indices).
        face_markers: Boundary marker per face.
        node_id_map: Maps TetGen 1-based node ID -> 0-based index.
    """

    def __init__(
        self,
        coords: np.ndarray,
        elements: np.ndarray,
        faces: np.ndarray,
        face_markers: Optional[np.ndarray] = None,
        node_id_map: Optional[Dict[int, int]] = None,
    ) -> None:
        self.coords = coords
        self.elements = elements
        self.faces = faces
        self.face_markers = face_markers
        self.node_id_map = node_id_map or {}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def read_tetgen(
    node_path: Union[str, Path],
    ele_path: Union[str, Path],
    face_path: Optional[Union[str, Path]] = None,
) -> TetGenMesh:
    """Read TetGen mesh from ``.node``, ``.ele``, and optionally ``.face`` files.

    Args:
        node_path: Path to the ``.node`` file.
        ele_path: Path to the ``.ele`` file.
        face_path: Path to the ``.face`` file (optional).

    Returns:
        Parsed :class:`TetGenMesh`.
    """
    coords, node_id_map = _parse_node(Path(node_path))
    elements = _parse_ele(Path(ele_path), node_id_map)

    faces: np.ndarray
    face_markers: Optional[np.ndarray] = None

    if face_path is not None:
        p = Path(face_path)
        if p.exists():
            faces, face_markers = _parse_face(p, node_id_map)
        else:
            # Generate faces from elements if no face file
            faces = _generate_boundary_faces(elements)
    else:
        faces = _generate_boundary_faces(elements)

    return TetGenMesh(
        coords=coords,
        elements=elements,
        faces=faces,
        face_markers=face_markers,
        node_id_map=node_id_map,
    )


def _parse_node(
    path: Path,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Parse ``.node`` file.

    Returns:
        Tuple of (coordinates array, node_id_map).
    """
    if not path.exists():
        raise FileNotFoundError(f"Node file not found: {path}")

    coords_list: List[List[float]] = []
    id_map: Dict[int, int] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header_done = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not header_done:
                # Header line: n_nodes dimension n_attributes n_boundary_markers
                header_done = True
                continue

            if len(parts) < 4:
                continue

            try:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                idx = len(coords_list)
                id_map[node_id] = idx
                coords_list.append([x, y, z])
            except ValueError:
                continue

    if not coords_list:
        raise ValueError(f"No node data found in {path}")

    return np.array(coords_list, dtype=np.float64), id_map


def _parse_ele(
    path: Path, node_id_map: Dict[int, int],
) -> np.ndarray:
    """Parse ``.ele`` file.

    Returns:
        Element connectivity array, shape ``(n_elements, 4)`` (0-based).
    """
    if not path.exists():
        raise FileNotFoundError(f"Element file not found: {path}")

    elem_list: List[List[int]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header_done = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not header_done:
                header_done = True
                continue

            if len(parts) < 5:
                continue

            try:
                # id n1 n2 n3 n4 [attribute]
                node_ids = [int(parts[i]) for i in range(1, 5)]
                # Convert 1-based to 0-based
                verts = [node_id_map[n] for n in node_ids]
                elem_list.append(verts)
            except (ValueError, KeyError):
                continue

    if not elem_list:
        raise ValueError(f"No element data found in {path}")

    return np.array(elem_list, dtype=np.int32)


def _parse_face(
    path: Path, node_id_map: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse ``.face`` file.

    Returns:
        Tuple of (face connectivity, face markers).
    """
    if not path.exists():
        raise FileNotFoundError(f"Face file not found: {path}")

    face_list: List[List[int]] = []
    markers: List[int] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header_done = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not header_done:
                header_done = True
                continue

            if len(parts) < 4:
                continue

            try:
                # id n1 n2 n3 [boundary_marker]
                node_ids = [int(parts[i]) for i in range(1, 4)]
                verts = [node_id_map[n] for n in node_ids]
                face_list.append(verts)
                marker = int(parts[4]) if len(parts) > 4 else 0
                markers.append(marker)
            except (ValueError, KeyError):
                continue

    if not face_list:
        return np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int32)

    return np.array(face_list, dtype=np.int32), np.array(markers, dtype=np.int32)


def _generate_boundary_faces(elements: np.ndarray) -> np.ndarray:
    """Generate boundary faces from elements by finding exposed faces.

    An exposed face appears only once across all elements.
    """
    face_count: Dict[Tuple[int, ...], int] = {}

    for elem in elements:
        # Tet faces: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
        tet_faces = [
            (elem[0], elem[1], elem[2]),
            (elem[0], elem[1], elem[3]),
            (elem[0], elem[2], elem[3]),
            (elem[1], elem[2], elem[3]),
        ]
        for face in tet_faces:
            key = tuple(sorted(face))
            face_count[key] = face_count.get(key, 0) + 1

    boundary = [list(k) for k, c in face_count.items() if c == 1]
    if not boundary:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(boundary, dtype=np.int32)


# ---------------------------------------------------------------------------
# tetgenToFoam conversion
# ---------------------------------------------------------------------------


def tetgen_to_foam(
    node_path: Union[str, Path],
    ele_path: Union[str, Path],
    face_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = ".",
    *,
    overwrite: bool = False,
) -> MeshData:
    """Convert TetGen mesh files to OpenFOAM polyMesh format.

    Args:
        node_path: Path to the ``.node`` file.
        ele_path: Path to the ``.ele`` file.
        face_path: Path to the ``.face`` file (optional).
        output_dir: Output directory for polyMesh files.
            The ``constant/polyMesh`` subdirectory will be created.
        overwrite: If True, overwrite existing files.

    Returns:
        The converted :class:`MeshData`.
    """
    output_dir = Path(output_dir)
    poly_mesh_dir = output_dir / "constant" / "polyMesh"
    poly_mesh_dir.mkdir(parents=True, exist_ok=True)

    # Parse TetGen mesh
    tg = read_tetgen(node_path, ele_path, face_path)

    # Convert node coordinates
    points = torch.tensor(tg.coords, dtype=get_default_dtype(), device=get_device())

    # Build faces, owner, neighbour from tetrahedral elements
    all_faces, owner, neighbour = _build_foam_mesh(tg.elements)

    # Build boundary patches from boundary faces
    boundary_patches = _build_boundary_patches(tg.faces, tg.face_markers, all_faces)

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
    elements: np.ndarray,
) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
    """Build OpenFOAM mesh connectivity from tetrahedral elements."""
    face_map: Dict[Tuple[int, ...], Tuple[int, int]] = {}
    all_faces: List[List[int]] = []
    face_owners: Dict[int, int] = {}
    face_neighbours: Dict[int, int] = {}

    for cell_idx in range(elements.shape[0]):
        elem = elements[cell_idx]
        tet_faces = [
            [elem[0], elem[1], elem[2]],
            [elem[0], elem[1], elem[3]],
            [elem[0], elem[2], elem[3]],
            [elem[1], elem[2], elem[3]],
        ]

        for face_nodes in tet_faces:
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


def _build_boundary_patches(
    boundary_faces: np.ndarray,
    face_markers: Optional[np.ndarray],
    all_faces: List[List[int]],
) -> List[BoundaryPatch]:
    """Build boundary patches from TetGen boundary faces."""
    if boundary_faces.shape[0] == 0:
        return []

    # Build face lookup
    face_lookup: Dict[Tuple[int, ...], int] = {}
    for i, face in enumerate(all_faces):
        face_lookup[tuple(sorted(face))] = i

    # Group by marker
    marker_faces: Dict[int, List[int]] = {}
    for i in range(boundary_faces.shape[0]):
        verts = boundary_faces[i].tolist()
        face_key = tuple(sorted(verts))
        marker = int(face_markers[i]) if face_markers is not None else 0
        if face_key in face_lookup:
            fi = face_lookup[face_key]
            if marker not in marker_faces:
                marker_faces[marker] = []
            marker_faces[marker].append(fi)

    patches: List[BoundaryPatch] = []
    current_start = len(all_faces)

    for marker in sorted(marker_faces.keys()):
        face_indices = marker_faces[marker]
        if not face_indices:
            continue

        patches.append(BoundaryPatch(
            name=f"patch_{marker}",
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
