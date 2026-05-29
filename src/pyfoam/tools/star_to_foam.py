"""
starToFoam — convert Star-CD/Star-CCM+ mesh to OpenFOAM polyMesh.

Reads Star-CD mesh files (``.vrt``, ``.cel``, ``.bnd``) and writes the
5 OpenFOAM polyMesh files: ``points``, ``faces``, ``owner``, ``neighbour``,
``boundary``.

Star-CD mesh file overview:

- ``<name>.vrt`` — Vertex (node) coordinates file
  - One line per vertex: ``x  y  z``
- ``<name>.cel`` — Cell connectivity file
  - One line per cell: ``type  v1  v2  v3  v4  [v5  v6  v7  v8]``
  - Cell types: 1=hex, 2=prism/wedge, 3=tet, 4=pyramid
- ``<name>.bnd`` — Boundary face connectivity file
  - One line per face: ``type  bc_type  v1  v2  [v3  v4]  region_id``

References
----------
- Star-CD mesh format documentation
- OpenFOAM ``starToFoam`` utility source
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.mesh_io import BoundaryPatch, MeshData

__all__ = [
    "StarCell",
    "StarBoundaryFace",
    "StarMesh",
    "read_star_mesh",
    "star_to_foam",
]


# ---------------------------------------------------------------------------
# Star-CD cell type constants
# ---------------------------------------------------------------------------

_STAR_HEX = 1
_STAR_WEDGE = 2
_STAR_TET = 3
_STAR_PYRAMID = 4

# Map Star-CD cell type -> number of vertices
_STAR_CELL_VERTS: Dict[int, int] = {
    _STAR_HEX: 8,
    _STAR_WEDGE: 6,
    _STAR_TET: 4,
    _STAR_PYRAMID: 5,
}

# Star-CD boundary type codes
_STAR_BC_WALL = 1
_STAR_BC_INLET = 2
_STAR_BC_OUTLET = 3
_STAR_BC_SYMMETRY = 4
_STAR_BC_PERIODIC = 5
_STAR_BC_INTERIOR = 0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class StarCell:
    """A Star-CD cell.

    Attributes:
        cell_type: Star-CD cell type code (1=hex, 2=wedge, 3=tet, 4=pyramid).
        nodes: Node IDs (1-based in Star-CD, stored as-is).
    """

    def __init__(self, cell_type: int, nodes: List[int]) -> None:
        self.cell_type = cell_type
        self.nodes = nodes


class StarBoundaryFace:
    """A Star-CD boundary face.

    Attributes:
        face_type: Number of vertices in face (3 or 4).
        bc_type: Boundary condition type code.
        nodes: Node IDs (1-based in Star-CD).
        region_id: Region/patch ID.
    """

    def __init__(
        self,
        face_type: int,
        bc_type: int,
        nodes: List[int],
        region_id: int,
    ) -> None:
        self.face_type = face_type
        self.bc_type = bc_type
        self.nodes = nodes
        self.region_id = region_id


class StarMesh:
    """Parsed Star-CD mesh.

    Attributes:
        coords: Node coordinates, shape ``(n_nodes, 3)``.
        cells: List of parsed cells.
        boundary_faces: List of parsed boundary faces.
    """

    def __init__(
        self,
        coords: np.ndarray,
        cells: List[StarCell],
        boundary_faces: List[StarBoundaryFace],
    ) -> None:
        self.coords = coords
        self.cells = cells
        self.boundary_faces = boundary_faces


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def read_star_mesh(
    vrt_path: Union[str, Path],
    cel_path: Union[str, Path],
    bnd_path: Union[str, Path],
) -> StarMesh:
    """Read Star-CD mesh from ``.vrt``, ``.cel``, ``.bnd`` files.

    Args:
        vrt_path: Path to the ``.vrt`` vertex file.
        cel_path: Path to the ``.cel`` cell file.
        bnd_path: Path to the ``.bnd`` boundary file.

    Returns:
        Parsed :class:`StarMesh`.
    """
    coords = _parse_vrt(Path(vrt_path))
    cells = _parse_cel(Path(cel_path))
    boundary_faces = _parse_bnd(Path(bnd_path))

    return StarMesh(coords=coords, cells=cells, boundary_faces=boundary_faces)


def _parse_vrt(path: Path) -> np.ndarray:
    """Parse Star-CD ``.vrt`` vertex file.

    Format: one vertex per line, ``x  y  z`` (space-separated).
    """
    if not path.exists():
        raise FileNotFoundError(f"Vertex file not found: {path}")

    coords: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    coords.append([x, y, z])
                except ValueError:
                    continue

    if not coords:
        raise ValueError(f"No vertex data found in {path}")

    return np.array(coords, dtype=np.float64)


def _parse_cel(path: Path) -> List[StarCell]:
    """Parse Star-CD ``.cel`` cell file.

    Format: one cell per line, ``type  v1  v2  v3  v4  [v5  v6  v7  v8]``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cell file not found: {path}")

    cells: List[StarCell] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                cell_type = int(parts[0])
                node_ids = [int(p) for p in parts[1:]]
                if node_ids:
                    cells.append(StarCell(cell_type=cell_type, nodes=node_ids))
            except ValueError:
                continue

    if not cells:
        raise ValueError(f"No cell data found in {path}")

    return cells


def _parse_bnd(path: Path) -> List[StarBoundaryFace]:
    """Parse Star-CD ``.bnd`` boundary file.

    Format: one face per line,
    ``type  bc_type  v1  v2  [v3  v4]  region_id``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found: {path}")

    faces: List[StarBoundaryFace] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                face_type = int(parts[0])
                bc_type = int(parts[1])
                # Last field is region_id
                region_id = int(parts[-1])
                # Middle fields are node IDs
                node_ids = [int(p) for p in parts[2:-1]]
                if node_ids:
                    faces.append(StarBoundaryFace(
                        face_type=face_type,
                        bc_type=bc_type,
                        nodes=node_ids,
                        region_id=region_id,
                    ))
            except ValueError:
                continue

    return faces


# ---------------------------------------------------------------------------
# starToFoam conversion
# ---------------------------------------------------------------------------


def star_to_foam(
    vrt_path: Union[str, Path],
    cel_path: Union[str, Path],
    bnd_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    overwrite: bool = False,
) -> MeshData:
    """Convert Star-CD mesh files to OpenFOAM polyMesh format.

    Reads the Star-CD ``.vrt``, ``.cel``, ``.bnd`` files and writes the
    5 polyMesh files.

    Args:
        vrt_path: Path to the ``.vrt`` vertex file.
        cel_path: Path to the ``.cel`` cell file.
        bnd_path: Path to the ``.bnd`` boundary file.
        output_dir: Output directory for polyMesh files.
            The ``constant/polyMesh`` subdirectory will be created.
        overwrite: If True, overwrite existing files.

    Returns:
        The converted :class:`MeshData`.
    """
    output_dir = Path(output_dir)
    poly_mesh_dir = output_dir / "constant" / "polyMesh"
    poly_mesh_dir.mkdir(parents=True, exist_ok=True)

    # Parse Star-CD mesh
    star = read_star_mesh(vrt_path, cel_path, bnd_path)

    # Convert node coordinates (Star-CD is 1-based, OpenFOAM is 0-based)
    points = torch.tensor(star.coords, dtype=get_default_dtype(), device=get_device())

    # Build faces, owner, neighbour from cells
    all_faces, owner, neighbour = _build_foam_mesh(star.cells)

    # Build boundary patches from boundary faces
    boundary_patches = _build_boundary_patches(star.boundary_faces, all_faces)

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
    cells: List[StarCell],
) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
    """Build OpenFOAM mesh connectivity from Star-CD cells.

    Star-CD uses 1-based node indices; OpenFOAM uses 0-based.
    """
    face_map: Dict[Tuple[int, ...], Tuple[int, int]] = {}
    all_faces: List[List[int]] = []
    face_owners: Dict[int, int] = {}
    face_neighbours: Dict[int, int] = {}

    for cell_idx, cell in enumerate(cells):
        # Convert 1-based to 0-based
        verts = [n - 1 for n in cell.nodes]
        elem_faces = _get_cell_faces(cell.cell_type, verts)

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


def _get_cell_faces(
    cell_type: int, verts: List[int],
) -> List[List[int]]:
    """Get faces of a Star-CD cell (verts are 0-based)."""
    if cell_type == _STAR_TET:
        return [
            [verts[0], verts[1], verts[2]],
            [verts[0], verts[1], verts[3]],
            [verts[0], verts[2], verts[3]],
            [verts[1], verts[2], verts[3]],
        ]
    elif cell_type == _STAR_HEX:
        return [
            [verts[0], verts[1], verts[5], verts[4]],
            [verts[2], verts[3], verts[7], verts[6]],
            [verts[0], verts[1], verts[3], verts[2]],
            [verts[4], verts[5], verts[7], verts[6]],
            [verts[0], verts[4], verts[6], verts[2]],
            [verts[1], verts[5], verts[7], verts[3]],
        ]
    elif cell_type == _STAR_WEDGE:
        return [
            [verts[0], verts[1], verts[4], verts[3]],
            [verts[1], verts[2], verts[5], verts[4]],
            [verts[0], verts[2], verts[5], verts[3]],
            [verts[0], verts[2], verts[1]],
            [verts[3], verts[4], verts[5]],
        ]
    elif cell_type == _STAR_PYRAMID:
        return [
            [verts[0], verts[1], verts[4]],
            [verts[1], verts[2], verts[4]],
            [verts[2], verts[3], verts[4]],
            [verts[0], verts[3], verts[4]],
            [verts[0], verts[1], verts[2], verts[3]],
        ]
    else:
        # Fallback
        return [verts]


def _build_boundary_patches(
    boundary_faces: List[StarBoundaryFace],
    all_faces: List[List[int]],
) -> List[BoundaryPatch]:
    """Build boundary patches from Star-CD boundary faces."""
    # Build face lookup
    face_lookup: Dict[Tuple[int, ...], int] = {}
    for i, face in enumerate(all_faces):
        face_lookup[tuple(sorted(face))] = i

    # Group boundary faces by region_id
    region_faces: Dict[int, List[int]] = {}
    for bface in boundary_faces:
        # Convert 1-based to 0-based
        verts = [n - 1 for n in bface.nodes]
        face_key = tuple(sorted(verts))
        if face_key in face_lookup:
            fi = face_lookup[face_key]
            rid = bface.region_id
            if rid not in region_faces:
                region_faces[rid] = []
            region_faces[rid].append(fi)

    # Map boundary type codes to OpenFOAM patch type names
    bc_type_map: Dict[int, str] = {
        _STAR_BC_WALL: "wall",
        _STAR_BC_INLET: "patch",
        _STAR_BC_OUTLET: "patch",
        _STAR_BC_SYMMETRY: "symmetry",
        _STAR_BC_PERIODIC: "cyclic",
    }

    # Determine per-region BC type (majority vote)
    region_bc: Dict[int, int] = {}
    for bface in boundary_faces:
        rid = bface.region_id
        if rid not in region_bc:
            region_bc[rid] = bface.bc_type

    patches: List[BoundaryPatch] = []
    current_start = len(all_faces)

    for rid in sorted(region_faces.keys()):
        face_indices = region_faces[rid]
        if not face_indices:
            continue

        bc = region_bc.get(rid, _STAR_BC_WALL)
        patch_name = f"patch_{rid}"
        patch_type = bc_type_map.get(bc, "patch")

        patches.append(BoundaryPatch(
            name=patch_name,
            patch_type=patch_type,
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
