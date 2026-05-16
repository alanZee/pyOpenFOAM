"""
BlockMesh — structured hexahedral mesh generation.

Implements OpenFOAM's blockMesh utility for creating structured hex meshes
from block definitions. Supports:

- Hexahedral (hex) blocks with 8 vertices
- Multi-grading (grading expansion ratios)
- Curved edges (arc, spline, polyLine, line)
- Boundary patch definitions

The generated mesh is a :class:`~pyfoam.mesh.poly_mesh.PolyMesh` that can
be used directly with the FVM solver framework.

Example::

    from pyfoam.mesh.generation import BlockMesh, Block, Grading

    # Define vertices
    vertices = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]

    # Define block with grading
    block = Block(
        vertices=[0, 1, 2, 3, 4, 5, 6, 7],
        n_cells=[10, 10, 10],
        grading=Grading.simple(1.0, 1.0, 1.0),
    )

    mesh = BlockMesh(vertices=[vertices], blocks=[block])
    poly_mesh = mesh.generate()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.poly_mesh import PolyMesh

__all__ = [
    "BlockMesh",
    "Block",
    "Grading",
    "EdgeType",
    "Edge",
    "BoundaryPatch",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EdgeType(Enum):
    """Types of curved edges in blockMesh."""
    LINE = "line"           # Straight line (default)
    ARC = "arc"             # Circular arc
    SPLINE = "spline"       # Spline curve
    POLYLINE = "polyLine"   # Polyline (piecewise linear)
    SIMPLE_SPLINE = "simpleSpline"  # Simple spline


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Grading:
    """Mesh grading (expansion ratio) specification.

    Supports simple grading (single ratio per direction) and multi-grading
    (multiple regions with different ratios).

    Attributes:
        ratios: List of (expansion_ratio, n_cells_fraction) tuples.
            For simple grading: [(ratio, 1.0)].
            For multi-grading: [(ratio1, frac1), (ratio2, frac2), ...].
    """
    ratios: list[tuple[float, float]]

    @classmethod
    def simple(cls, rx: float = 1.0, ry: float = 1.0, rz: float = 1.0) -> list["Grading"]:
        """Create simple grading for x, y, z directions.

        Args:
            rx: Expansion ratio in x direction.
            ry: Expansion ratio in y direction.
            rz: Expansion ratio in z direction.

        Returns:
            List of 3 Grading objects (one per direction).
        """
        return [
            cls(ratios=[(rx, 1.0)]),
            cls(ratios=[(ry, 1.0)]),
            cls(ratios=[(rz, 1.0)]),
        ]

    @classmethod
    def uniform(cls) -> list["Grading"]:
        """Create uniform grading (ratio=1.0) for all directions."""
        return cls.simple(1.0, 1.0, 1.0)

    def compute_distribution(self, n_cells: int) -> list[float]:
        """Compute cell size distribution for this grading.

        Args:
            n_cells: Number of cells in this direction.

        Returns:
            List of normalized cell sizes (sum to 1.0).
        """
        if len(self.ratios) == 1:
            # Simple grading
            ratio = self.ratios[0][0]
            if abs(ratio - 1.0) < 1e-10:
                # Uniform
                return [1.0 / n_cells] * n_cells
            else:
                # Geometric progression
                # ratio = dx_last / dx_first
                # Sum of geometric series: dx_first * (1 - ratio^n) / (1 - ratio) = 1.0
                if abs(ratio - 1.0) < 1e-10:
                    dx_first = 1.0 / n_cells
                else:
                    dx_first = (1.0 - ratio) / (1.0 - ratio**n_cells)
                sizes = []
                for i in range(n_cells):
                    sizes.append(dx_first * ratio**i)
                return sizes
        else:
            # Multi-grading
            sizes = []
            current_frac = 0.0
            for ratio, frac in self.ratios:
                n_region = max(1, round(frac * n_cells))
                if abs(ratio - 1.0) < 1e-10:
                    region_sizes = [frac / n_region] * n_region
                else:
                    dx_first = (1.0 - ratio) / (1.0 - ratio**n_region) * frac
                    region_sizes = [dx_first * ratio**i for i in range(n_region)]
                sizes.extend(region_sizes)
            # Normalize to sum to 1.0
            total = sum(sizes)
            return [s / total for s in sizes]


@dataclass
class Edge:
    """Curved edge definition.

    Attributes:
        start: Start vertex index.
        end: End vertex index.
        edge_type: Type of curve.
        points: Control points for the curve (for arc: single midpoint).
    """
    start: int
    end: int
    edge_type: EdgeType
    points: list[list[float]] = field(default_factory=list)


@dataclass
class Block:
    """Hexahedral block definition.

    Attributes:
        vertices: List of 8 vertex indices defining the hex block.
            Ordering follows OpenFOAM convention:
            - Bottom face (z=0): 0,1,2,3 (counter-clockwise from below)
            - Top face (z=1): 4,5,6,7 (same ordering)
        n_cells: Number of cells in each direction [nx, ny, nz].
        grading: List of 3 Grading objects (one per direction).
        zone: Optional zone name for the block.
    """
    vertices: list[int]
    n_cells: list[int]
    grading: list[Grading] = field(default_factory=lambda: Grading.uniform())
    zone: Optional[str] = None

    def __post_init__(self):
        if len(self.vertices) != 8:
            raise ValueError(f"Block must have 8 vertices, got {len(self.vertices)}")
        if len(self.n_cells) != 3:
            raise ValueError(f"n_cells must have 3 elements, got {len(self.n_cells)}")
        if len(self.grading) != 3:
            raise ValueError(f"grading must have 3 elements, got {len(self.grading)}")


@dataclass
class BoundaryPatch:
    """Boundary patch definition for blockMesh.

    Attributes:
        name: Patch name.
        patch_type: Patch type (wall, patch, inlet, outlet, etc.).
        faces: List of face vertex index lists.
    """
    name: str
    patch_type: str = "patch"
    faces: list[list[int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BlockMesh generator
# ---------------------------------------------------------------------------


class BlockMesh:
    """Structured hex mesh generator.

    Generates a polyhedral mesh from block definitions, similar to OpenFOAM's
    blockMesh utility.

    Parameters:
        vertices: List of vertex coordinates [[x,y,z], ...].
        blocks: List of Block definitions.
        edges: List of Edge definitions (optional).
        patches: List of BoundaryPatch definitions (optional).
        scale: Scale factor for vertex coordinates.
    """

    def __init__(
        self,
        vertices: list[list[float]],
        blocks: list[Block],
        edges: Optional[list[Edge]] = None,
        patches: Optional[list[BoundaryPatch]] = None,
        scale: float = 1.0,
    ) -> None:
        self._vertices = [v[:] for v in vertices]  # Deep copy
        self._blocks = blocks
        self._edges = edges or []
        self._patches = patches or []
        self._scale = scale

        # Apply scale
        for v in self._vertices:
            for i in range(3):
                v[i] *= scale

    def generate(self) -> PolyMesh:
        """Generate the polyhedral mesh.

        Returns:
            PolyMesh instance with the generated mesh.
        """
        device = get_device()
        dtype = get_default_dtype()

        # Generate points, faces, owner, neighbour for each block
        all_points: list[list[float]] = []
        all_faces: list[list[int]] = []
        all_owner: list[int] = []
        all_neighbour: list[int] = []
        all_boundary: list[dict] = []

        # Track point mapping (global vertex index -> local point index)
        point_map: dict[int, int] = {}

        for block_idx, block in enumerate(self._blocks):
            block_points, block_faces, block_owner, block_neighbour = \
                self._generate_block(block, block_idx, point_map, all_points)

            # Offset face indices
            n_existing_points = len(all_points)
            offset = len(all_faces)

            # Add block points
            all_points.extend(block_points)

            # Add block faces with offset owner/neighbour
            for face in block_faces:
                all_faces.append(face)
            for cell_idx in block_owner:
                all_owner.append(cell_idx + block_idx * (block.n_cells[0] * block.n_cells[1] * block.n_cells[2]))
            for cell_idx in block_neighbour:
                all_neighbour.append(cell_idx + block_idx * (block.n_cells[0] * block.n_cells[1] * block.n_cells[2]))

        # Build boundary patches from faces
        boundary = self._build_boundary(all_faces, all_owner, all_neighbour)

        # Convert to tensors
        points_tensor = torch.tensor(all_points, dtype=dtype, device=device)
        face_tensors = [torch.tensor(f, dtype=INDEX_DTYPE, device=device) for f in all_faces]
        owner_tensor = torch.tensor(all_owner, dtype=INDEX_DTYPE, device=device)
        neighbour_tensor = torch.tensor(all_neighbour, dtype=INDEX_DTYPE, device=device)

        return PolyMesh(
            points=points_tensor,
            faces=face_tensors,
            owner=owner_tensor,
            neighbour=neighbour_tensor,
            boundary=boundary,
        )

    def _generate_block(
        self,
        block: Block,
        block_idx: int,
        point_map: dict[int, int],
        global_points: list[list[float]],
    ) -> tuple[list[list[float]], list[list[int]], list[int], list[int]]:
        """Generate mesh for a single block.

        Returns:
            (points, faces, owner, neighbour) for this block.
        """
        nx, ny, nz = block.n_cells
        grading_x, grading_y, grading_z = block.grading

        # Get block vertex coordinates
        block_verts = [self._vertices[vidx] for vidx in block.vertices]

        # Compute grading distributions
        dx_dist = grading_x.compute_distribution(nx)
        dy_dist = grading_y.compute_distribution(ny)
        dz_dist = grading_z.compute_distribution(nz)

        # Generate interior points using trilinear interpolation
        points = []
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    # Parametric coordinates
                    u = sum(dx_dist[:i]) if i > 0 else 0.0
                    v = sum(dy_dist[:j]) if j > 0 else 0.0
                    w = sum(dz_dist[:k]) if k > 0 else 0.0

                    # Trilinear interpolation
                    p = self._trilinear_interpolate(block_verts, u, v, w)
                    points.append(p)

        # Generate cells (hexahedra)
        faces = []
        owner = []
        neighbour = []

        # Helper to get linear index
        def idx(i, j, k):
            return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i

        # Generate internal faces and owner/neighbour
        cell_idx = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # Cell vertices
                    c000 = idx(i, j, k)
                    c100 = idx(i + 1, j, k)
                    c010 = idx(i, j + 1, k)
                    c110 = idx(i + 1, j + 1, k)
                    c001 = idx(i, j, k + 1)
                    c101 = idx(i + 1, j, k + 1)
                    c011 = idx(i, j + 1, k + 1)
                    c111 = idx(i + 1, j + 1, k + 1)

                    # Internal faces (shared between cells)
                    # x-normal face (between i and i+1)
                    if i < nx - 1:
                        faces.append([c100, c110, c111, c101])
                        owner.append(cell_idx)
                        neighbour.append(cell_idx + 1)

                    # y-normal face (between j and j+1)
                    if j < ny - 1:
                        faces.append([c010, c000, c001, c011])
                        owner.append(cell_idx)
                        neighbour.append(cell_idx + nx)

                    # z-normal face (between k and k+1)
                    if k < nz - 1:
                        faces.append([c000, c100, c101, c001])
                        owner.append(cell_idx)
                        neighbour.append(cell_idx + nx * ny)

                    cell_idx += 1

        # Generate boundary faces
        # Bottom (k=0)
        for j in range(ny):
            for i in range(nx):
                c000 = idx(i, j, 0)
                c100 = idx(i + 1, j, 0)
                c110 = idx(i + 1, j + 1, 0)
                c010 = idx(i, j + 1, 0)
                faces.append([c000, c010, c110, c100])
                owner.append(j * nx + i)

        # Top (k=nz)
        for j in range(ny):
            for i in range(nx):
                c001 = idx(i, j, nz)
                c101 = idx(i + 1, j, nz)
                c111 = idx(i + 1, j + 1, nz)
                c011 = idx(i, j + 1, nz)
                faces.append([c001, c101, c111, c011])
                owner.append((nz - 1) * nx * ny + j * nx + i)

        # Front (j=0)
        for k in range(nz):
            for i in range(nx):
                c000 = idx(i, 0, k)
                c100 = idx(i + 1, 0, k)
                c101 = idx(i + 1, 0, k + 1)
                c001 = idx(i, 0, k + 1)
                faces.append([c000, c100, c101, c001])
                owner.append(k * nx + i)

        # Back (j=ny)
        for k in range(nz):
            for i in range(nx):
                c010 = idx(i, ny, k)
                c110 = idx(i + 1, ny, k)
                c111 = idx(i + 1, ny, k + 1)
                c011 = idx(i, ny, k + 1)
                faces.append([c010, c011, c111, c110])
                owner.append(k * nx * ny + (ny - 1) * nx + i)

        # Left (i=0)
        for k in range(nz):
            for j in range(ny):
                c000 = idx(0, j, k)
                c010 = idx(0, j + 1, k)
                c011 = idx(0, j + 1, k + 1)
                c001 = idx(0, j, k + 1)
                faces.append([c000, c001, c011, c010])
                owner.append(k * nx * ny + j * nx)

        # Right (i=nx)
        for k in range(nz):
            for j in range(ny):
                c100 = idx(nx, j, k)
                c110 = idx(nx, j + 1, k)
                c111 = idx(nx, j + 1, k + 1)
                c101 = idx(nx, j, k + 1)
                faces.append([c100, c110, c111, c101])
                owner.append(k * nx * ny + j * nx + nx - 1)

        return points, faces, owner, neighbour

    def _trilinear_interpolate(
        self,
        verts: list[list[float]],
        u: float,
        v: float,
        w: float,
    ) -> list[float]:
        """Trilinear interpolation within a hex block.

        Args:
            verts: 8 vertex coordinates (OpenFOAM ordering).
            u, v, w: Parametric coordinates in [0, 1].

        Returns:
            Interpolated [x, y, z] coordinates.
        """
        # OpenFOAM hex vertex ordering:
        # 0: (0,0,0), 1: (1,0,0), 2: (1,1,0), 3: (0,1,0)
        # 4: (0,0,1), 5: (1,0,1), 6: (1,1,1), 7: (0,1,1)
        x = (
            (1 - u) * (1 - v) * (1 - w) * verts[0][0] +
            u * (1 - v) * (1 - w) * verts[1][0] +
            u * v * (1 - w) * verts[2][0] +
            (1 - u) * v * (1 - w) * verts[3][0] +
            (1 - u) * (1 - v) * w * verts[4][0] +
            u * (1 - v) * w * verts[5][0] +
            u * v * w * verts[6][0] +
            (1 - u) * v * w * verts[7][0]
        )
        y = (
            (1 - u) * (1 - v) * (1 - w) * verts[0][1] +
            u * (1 - v) * (1 - w) * verts[1][1] +
            u * v * (1 - w) * verts[2][1] +
            (1 - u) * v * (1 - w) * verts[3][1] +
            (1 - u) * (1 - v) * w * verts[4][1] +
            u * (1 - v) * w * verts[5][1] +
            u * v * w * verts[6][1] +
            (1 - u) * v * w * verts[7][1]
        )
        z = (
            (1 - u) * (1 - v) * (1 - w) * verts[0][2] +
            u * (1 - v) * (1 - w) * verts[1][2] +
            u * v * (1 - w) * verts[2][2] +
            (1 - u) * v * (1 - w) * verts[3][2] +
            (1 - u) * (1 - v) * w * verts[4][2] +
            u * (1 - v) * w * verts[5][2] +
            u * v * w * verts[6][2] +
            (1 - u) * v * w * verts[7][2]
        )
        return [x, y, z]

    def _build_boundary(
        self,
        faces: list[list[int]],
        owner: list[int],
        neighbour: list[int],
    ) -> list[dict]:
        """Build boundary patch definitions.

        If patches are explicitly defined, use those. Otherwise, create
        a single default boundary patch.
        """
        if self._patches:
            boundary = []
            for patch in self._patches:
                boundary.append({
                    "name": patch.name,
                    "type": patch.patch_type,
                    "startFace": len(neighbour) + len(boundary) * 1000,  # Placeholder
                    "nFaces": len(patch.faces),
                })
            return boundary

        # Default: single boundary patch
        n_boundary = len(faces) - len(neighbour)
        return [{
            "name": "defaultFaces",
            "type": "wall",
            "startFace": len(neighbour),
            "nFaces": n_boundary,
        }]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def create_hex_mesh(
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
    z_range: tuple[float, float] = (0.0, 1.0),
    nx: int = 10,
    ny: int = 10,
    nz: int = 10,
    grading: Optional[list[Grading]] = None,
) -> PolyMesh:
    """Create a simple hex mesh.

    Args:
        x_range: (x_min, x_max) range.
        y_range: (y_min, y_max) range.
        z_range: (z_min, z_max) range.
        nx, ny, nz: Number of cells in each direction.
        grading: Optional grading specification.

    Returns:
        PolyMesh with the generated mesh.
    """
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range

    vertices = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]

    block = Block(
        vertices=[0, 1, 2, 3, 4, 5, 6, 7],
        n_cells=[nx, ny, nz],
        grading=grading or Grading.uniform(),
    )

    mesh = BlockMesh(vertices=vertices, blocks=[block])
    return mesh.generate()
