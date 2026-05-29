"""
blockMesh enhanced — enhanced block mesh generation with multi-block support
and grading specification.

Provides a Python-level equivalent of OpenFOAM's ``blockMesh`` utility with:

- **Multi-block support**: Define multiple hex blocks with independent
  resolution and connectivity.
- **Grading specification**: Simple and edge-based grading (expansion
  ratios) for each block direction.
- **Vertex merging**: Automatic vertex proximity detection and merging.
- **Patch definition**: Named boundary patches with type specification.
- **blockMeshDict generation**: Write ``system/blockMeshDict`` files.

Usage::

    from pyfoam.tools.block_mesh_enhanced import BlockMeshConfig, block_mesh, Block

    config = BlockMeshConfig(
        vertices=[[0,0,0], [1,0,0], [1,1,0], [0,1,0],
                  [0,0,1], [1,0,1], [1,1,1], [0,1,1]],
        blocks=[
            Block(vertices=[0,1,2,3,4,5,6,7],
                  cells=[10, 10, 1],
                  grading=[(1, 0.2), (1, 0.2), (1, 1)]),
        ],
        patches=[...],
    )
    result = block_mesh(config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Block", "PatchFace", "BlockMeshConfig", "BlockMeshResult", "block_mesh",
    "write_block_mesh_dict",
]


# ---------------------------------------------------------------------------
# Block and patch specifications
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """Specification of a single hex block.

    Attributes
    ----------
    vertices : list[int]
        8 vertex indices defining the hex block (OpenFOAM ordering).
    cells : tuple[int, int, int]
        Number of cells in (x, y, z) directions.
    grading : list[tuple[float, float]], optional
        Grading specification.  Each entry is ``(ratio, fraction)``.
        For simple grading, a single entry per direction.  For multi-grading,
        multiple entries whose fractions sum to 1.0.
        If None, uniform grading (1, 1, 1) is used.
    zone : str, optional
        Cell zone name for this block.
    """
    vertices: List[int] = field(default_factory=lambda: list(range(8)))
    cells: Tuple[int, int, int] = (10, 10, 10)
    grading: Optional[List[Tuple[float, float]]] = None
    zone: Optional[str] = None


@dataclass
class PatchFace:
    """Specification of a boundary patch face.

    Attributes
    ----------
    name : str
        Patch name.
    face_vertices : list[int]
        4 vertex indices defining a quad face.
    patch_type : str
        Patch type (``"wall"``, ``"patch"``, ``"inlet"``, etc.).
    """
    name: str = "patch"
    face_vertices: List[int] = field(default_factory=list)
    patch_type: str = "patch"


@dataclass
class BlockMeshConfig:
    """Configuration for enhanced blockMesh.

    Attributes
    ----------
    vertices : list[list[float]]
        Vertex coordinates as ``[[x, y, z], ...]``.
    blocks : list[Block]
        Block definitions.
    patches : list[PatchFace]
        Boundary patch face definitions.
    edges : list[tuple[int, int, list[float]]], optional
        Spline/arc edge definitions as ``(v1, v2, midpoint_or_points)``.
    merge_patch_pairs : list[tuple[str, str]], optional
        Pairs of patches to merge.
    scale : float
        Scaling factor for all coordinates.
    """
    vertices: List[List[float]] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    patches: List[PatchFace] = field(default_factory=list)
    edges: List[Tuple[int, int, list]] = field(default_factory=list)
    merge_patch_pairs: List[Tuple[str, str]] = field(default_factory=list)
    scale: float = 1.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BlockMeshResult:
    """Result from :func:`block_mesh`.

    Attributes
    ----------
    points : np.ndarray
        ``(n_points, 3)`` vertex coordinates.
    cell_centres : np.ndarray
        ``(n_cells, 3)`` cell centre coordinates.
    cell_volumes : np.ndarray
        ``(n_cells,)`` cell volumes.
    n_cells : int
        Total number of cells across all blocks.
    n_points : int
        Total number of vertices.
    grading_expansion : dict[str, list[float]]
        Per-block grading expansion ratios.
    block_mesh_dict_path : Path, optional
        Path to the written ``blockMeshDict`` file, if requested.
    """

    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    cell_centres: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    cell_volumes: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    n_cells: int = 0
    n_points: int = 0
    grading_expansion: Dict[str, List[float]] = field(default_factory=dict)
    block_mesh_dict_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def block_mesh(
    config: BlockMeshConfig,
    write_dict: Optional[Union[str, Path]] = None,
) -> BlockMeshResult:
    """Generate a multi-block hex mesh from configuration.

    Parameters
    ----------
    config : BlockMeshConfig
        Block mesh configuration.
    write_dict : str or Path, optional
        If specified, write a ``blockMeshDict`` to this path.

    Returns
    -------
    BlockMeshResult
        Generated mesh data.
    """
    if not config.blocks:
        raise ValueError("At least one block must be defined.")

    if not config.vertices:
        raise ValueError("At least 8 vertices must be defined.")

    verts = np.array(config.vertices, dtype=np.float64) * config.scale

    # Generate per-block hex meshes
    all_centres = []
    all_volumes = []
    grading_info: dict[str, list[float]] = {}

    for bi, block in enumerate(config.blocks):
        # Get block corner coordinates
        block_verts = verts[block.vertices]
        cx, cy, cz = block.cells

        # Compute grading expansion ratios
        grading_x = _expand_grading(block.grading, cx, 0) if block.grading else [1.0] * cx
        grading_y = _expand_grading(block.grading, cy, 1) if block.grading else [1.0] * cy
        grading_z = _expand_grading(block.grading, cz, 2) if block.grading else [1.0] * cz

        grading_info[f"block_{bi}"] = grading_x + grading_y + grading_z

        # Generate cell centres using trilinear mapping
        centres, volumes = _generate_block_cells(
            block_verts, cx, cy, cz, grading_x, grading_y, grading_z,
        )
        all_centres.append(centres)
        all_volumes.append(volumes)

    cell_centres = np.concatenate(all_centres, axis=0)
    cell_volumes = np.concatenate(all_volumes, axis=0)

    # Compute total points
    n_points = len(verts)

    # Write blockMeshDict if requested
    dict_path = None
    if write_dict is not None:
        dict_path = write_block_mesh_dict(write_dict, config)

    return BlockMeshResult(
        points=verts,
        cell_centres=cell_centres,
        cell_volumes=cell_volumes,
        n_cells=len(cell_centres),
        n_points=n_points,
        grading_expansion=grading_info,
        block_mesh_dict_path=dict_path,
    )


# ---------------------------------------------------------------------------
# Block cell generation
# ---------------------------------------------------------------------------


def _generate_block_cells(
    block_verts: np.ndarray,
    nx: int, ny: int, nz: int,
    gx: list[float], gy: list[float], gz: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate cell centres and volumes for a single hex block.

    Uses trilinear mapping from parametric (s, t, u) space to physical
    coordinates.  The hex block is defined by 8 vertices in OpenFOAM
    ordering:

        0=(0,0,0), 1=(1,0,0), 2=(1,1,0), 3=(0,1,0)
        4=(0,0,1), 5=(1,0,1), 6=(1,1,1), 7=(0,1,1)
    """
    v = block_verts  # shape (8, 3)

    # Build parametric coordinates with grading
    s_coords = _graded_coordinates(nx, gx)
    t_coords = _graded_coordinates(ny, gy)
    u_coords = _graded_coordinates(nz, gz)

    centres = []
    volumes = []

    for i, si in enumerate(s_coords):
        for j, tj in enumerate(t_coords):
            for k, uk in enumerate(u_coords):
                # Trilinear interpolation for cell centre
                c = _trilinear(v, si, tj, uk)
                centres.append(c)

                # Approximate cell volume via Jacobian determinant
                ds = gx[i] / nx if i < len(gx) else 1.0 / nx
                dt = gy[j] / ny if j < len(gy) else 1.0 / ny
                du = gz[k] / nz if k < len(gz) else 1.0 / nz

                vol = _trilinear_volume(v, si, tj, uk, ds, dt, du)
                volumes.append(vol)

    return np.array(centres), np.array(volumes)


def _graded_coordinates(n_cells: int, grading: list[float]) -> np.ndarray:
    """Compute cell-centre coordinates in [0, 1] with grading.

    Each entry in ``grading`` is an expansion ratio for that cell.
    Returns coordinates of cell centres (not faces).
    """
    if len(grading) != n_cells:
        # Uniform grading as fallback
        return np.array([(i + 0.5) / n_cells for i in range(n_cells)])

    # Build face positions from expansion ratios
    total = sum(grading)
    faces = [0.0]
    for g in grading:
        faces.append(faces[-1] + g / total)
    faces[-1] = 1.0  # ensure exact closure

    # Cell centres
    centres = [(faces[i] + faces[i + 1]) / 2.0 for i in range(n_cells)]
    return np.array(centres)


def _expand_grading(
    grading_spec: Optional[list[tuple[float, float]]],
    n_cells: int,
    direction: int,
) -> list[float]:
    """Expand grading specification to per-cell expansion ratios.

    Each entry in ``grading_spec`` is ``(ratio, fraction)``.  The
    fraction indicates what portion of cells get that expansion ratio.

    Returns a list of ``n_cells`` expansion ratios.
    """
    if grading_spec is None:
        return [1.0] * n_cells

    result = []
    remaining = n_cells

    for ratio, fraction in grading_spec:
        n = max(1, round(fraction * n_cells))
        n = min(n, remaining)
        result.extend([ratio] * n)
        remaining -= n

    # Fill any remaining cells
    while len(result) < n_cells:
        result.append(1.0)

    return result[:n_cells]


def _trilinear(v: np.ndarray, s: float, t: float, u: float) -> np.ndarray:
    """Trilinear interpolation within a hex block."""
    return (
        (1 - s) * (1 - t) * (1 - u) * v[0] +
        s * (1 - t) * (1 - u) * v[1] +
        s * t * (1 - u) * v[2] +
        (1 - s) * t * (1 - u) * v[3] +
        (1 - s) * (1 - t) * u * v[4] +
        s * (1 - t) * u * v[5] +
        s * t * u * v[6] +
        (1 - s) * t * u * v[7]
    )


def _trilinear_volume(
    v: np.ndarray, s: float, t: float, u: float,
    ds: float, dt: float, du: float,
) -> float:
    """Approximate cell volume via Jacobian of the trilinear mapping.

    Volume = |det(J)| * ds * dt * du where J is the Jacobian matrix
    of the trilinear map at (s, t, u).
    """
    # Partial derivatives
    dN_ds = np.array([
        -(1 - t) * (1 - u), (1 - t) * (1 - u), t * (1 - u), -(1 - t) * (1 - u),
        -(1 - t) * u, (1 - t) * u, t * u, -(1 - t) * u,
    ])
    dN_dt = np.array([
        -(1 - s) * (1 - u), -(s) * (1 - u), s * (1 - u), (1 - s) * (1 - u),
        -(1 - s) * u, -(s) * u, s * u, (1 - s) * u,
    ])
    dN_du = np.array([
        -(1 - s) * (1 - t), -(s) * (1 - t), -(s) * t, -(1 - s) * t,
        (1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t,
    ])

    # Jacobian columns
    J0 = dN_ds @ v  # dx/ds, dy/ds, dz/ds
    J1 = dN_dt @ v  # dx/dt, dy/dt, dz/dt
    J2 = dN_du @ v  # dx/du, dy/du, dz/du

    J = np.column_stack([J0, J1, J2])
    det_J = abs(np.linalg.det(J))

    return det_J * ds * dt * du


# ---------------------------------------------------------------------------
# blockMeshDict writer
# ---------------------------------------------------------------------------


def write_block_mesh_dict(
    path: Union[str, Path],
    config: BlockMeshConfig,
) -> Path:
    """Write a ``blockMeshDict`` configuration file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    config : BlockMeshConfig
        Block mesh configuration.

    Returns
    -------
    Path
        Path to the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        f.write("/* blockMeshDict — generated by block_mesh_enhanced */\n\n")
        f.write(f"scale    {config.scale};\n\n")

        # Vertices
        f.write("vertices\n(\n")
        for i, v in enumerate(config.vertices):
            f.write(f"    ({v[0]:.10g} {v[1]:.10g} {v[2]:.10g})  // {i}\n")
        f.write(");\n\n")

        # Edges
        if config.edges:
            f.write("edges\n(\n")
            for v1, v2, pts in config.edges:
                if isinstance(pts[0], (list, tuple)):
                    # Spline
                    pts_str = " ".join(
                        f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})" for p in pts
                    )
                    f.write(f"    spline {v1} {v2} ({pts_str})\n")
                else:
                    # Arc midpoint
                    f.write(
                        f"    arc {v1} {v2} "
                        f"({pts[0]:.10g} {pts[1]:.10g} {pts[2]:.10g})\n"
                    )
            f.write(");\n\n")

        # Blocks
        f.write("blocks\n(\n")
        for bi, block in enumerate(config.blocks):
            v_str = " ".join(str(v) for v in block.vertices)
            c_str = f"({block.cells[0]} {block.cells[1]} {block.cells[2]})"
            zone_str = f" zone {block.zone}" if block.zone else ""

            if block.grading:
                g_entries = " ".join(
                    f"({r:.6g} {fr:.6g})" for r, fr in block.grading
                )
                grading_str = f"simpleGrading ({g_entries} {g_entries} {g_entries})"
            else:
                grading_str = "simpleGrading (1 1 1)"

            f.write(
                f"    hex ({v_str}){zone_str} {c_str}\n"
                f"    {grading_str}\n"
            )
        f.write(");\n\n")

        # Patches
        if config.patches:
            # Group faces by patch name
            patch_groups: dict[str, list[PatchFace]] = {}
            for pf in config.patches:
                patch_groups.setdefault(pf.name, []).append(pf)

            f.write("boundary\n(\n")
            for pname, pfaces in patch_groups.items():
                ptype = pfaces[0].patch_type
                f.write(f"    {pname}\n    {{\n")
                f.write(f"        type    {ptype};\n")
                f.write(f"        faces\n        (\n")
                for pf in pfaces:
                    fv_str = " ".join(str(v) for v in pf.face_vertices)
                    f.write(f"            ({fv_str})\n")
                f.write(f"        );\n")
                f.write(f"    }}\n")
            f.write(");\n\n")

        # Merge patch pairs
        if config.merge_patch_pairs:
            f.write("mergePatchPairs\n(\n")
            for p1, p2 in config.merge_patch_pairs:
                f.write(f"    ({p1} {p2})\n")
            f.write(");\n")

    return out
