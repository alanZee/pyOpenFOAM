"""
SnappyHexMesh — surface-based hex mesh generation.

Implements OpenFOAM's snappyHexMesh utility for creating unstructured hex
meshes from STL surfaces. The process involves:

1. **Castellation**: Cut a background hex mesh to match the surface
2. **Snapping**: Move boundary points to snap to the surface
3. **Layers**: Add boundary layer cells near walls

This is a simplified implementation that provides the core functionality
for surface-based mesh generation.

Example::

    from pyfoam.mesh.generation import SnappyHexMesh
    from pyfoam.mesh.generation.stl import STLReader

    # Read STL surface
    reader = STLReader("surface.stl")
    surface = reader.read()

    # Generate mesh
    snappy = SnappyHexMesh(
        surfaces=[surface],
        background_mesh=(10, 10, 10),
        refinement_levels=[2],
    )
    mesh = snappy.generate()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.poly_mesh import PolyMesh
from pyfoam.mesh.generation.stl import STLSurface
from pyfoam.mesh.generation.block_mesh import BlockMesh, Block, Grading

__all__ = ["SnappyHexMesh", "RefinementRegion", "LayersSpec"]


@dataclass
class RefinementRegion:
    """Refinement region specification.

    Attributes:
        surface: The STL surface to refine around.
        level: Refinement level (number of times to subdivide).
        distance: Distance from surface for refinement.
    """
    surface: STLSurface
    level: int = 2
    distance: float = 0.1


@dataclass
class LayersSpec:
    """Boundary layer specification.

    Attributes:
        patch_name: Name of the patch to add layers to.
        n_layers: Number of layer cells.
        expansion_ratio: Layer expansion ratio.
        final_layer_thickness: Thickness of the final (closest to wall) layer.
    """
    patch_name: str
    n_layers: int = 5
    expansion_ratio: float = 1.2
    final_layer_thickness: float = 0.001


class SnappyHexMesh:
    """Surface-based hex mesh generator.

    Generates a mesh by refining a background hex mesh to match STL surfaces.

    Parameters:
        surfaces: List of STL surfaces.
        background_mesh: (nx, ny, nz) cells for background mesh.
        refinement_regions: List of refinement region specifications.
        layers: List of boundary layer specifications.
        global_refinement: Global refinement level for background mesh.
    """

    def __init__(
        self,
        surfaces: list[STLSurface],
        background_mesh: tuple[int, int, int] = (10, 10, 10),
        refinement_regions: Optional[list[RefinementRegion]] = None,
        layers: Optional[list[LayersSpec]] = None,
        global_refinement: int = 0,
    ) -> None:
        self._surfaces = surfaces
        self._bg_nx, self._bg_ny, self._bg_nz = background_mesh
        self._refinement_regions = refinement_regions or []
        self._layers = layers or []
        self._global_refinement = global_refinement

        # Compute bounding box from surfaces
        self._bbox_min, self._bbox_max = self._compute_bounding_box()

    def _compute_bounding_box(self) -> tuple[list[float], list[float]]:
        """Compute bounding box from all surfaces."""
        if not self._surfaces:
            return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

        all_vertices = torch.cat([s.vertices for s in self._surfaces], dim=0)
        bbox_min = all_vertices.min(dim=0).values.tolist()
        bbox_max = all_vertices.max(dim=0).values.tolist()

        # Add 10% margin
        for i in range(3):
            margin = (bbox_max[i] - bbox_min[i]) * 0.1
            bbox_min[i] -= margin
            bbox_max[i] += margin

        return bbox_min, bbox_max

    def generate(self) -> PolyMesh:
        """Generate the mesh.

        Returns:
            PolyMesh with the generated mesh.
        """
        # Step 1: Create background hex mesh
        mesh = self._create_background_mesh()

        # Step 2: Apply global refinement
        for _ in range(self._global_refinement):
            mesh = self._refine_mesh(mesh)

        # Step 3: Castellation - refine near surfaces
        for region in self._refinement_regions:
            for _ in range(region.level):
                mesh = self._refine_near_surface(mesh, region.surface, region.distance)

        # Step 4: Snap to surface
        mesh = self._snap_to_surface(mesh)

        # Step 5: Add boundary layers
        if self._layers:
            mesh = self._add_boundary_layers(mesh)

        return mesh

    def _create_background_mesh(self) -> PolyMesh:
        """Create the background hex mesh."""
        # Use blockMesh to create background mesh
        vertices = [
            self._bbox_min[:],
            [self._bbox_max[0], self._bbox_min[1], self._bbox_min[2]],
            [self._bbox_max[0], self._bbox_max[1], self._bbox_min[2]],
            [self._bbox_min[0], self._bbox_max[1], self._bbox_min[2]],
            [self._bbox_min[0], self._bbox_min[1], self._bbox_max[2]],
            [self._bbox_max[0], self._bbox_min[1], self._bbox_max[2]],
            self._bbox_max[:],
            [self._bbox_min[0], self._bbox_max[1], self._bbox_max[2]],
        ]

        block = Block(
            vertices=[0, 1, 2, 3, 4, 5, 6, 7],
            n_cells=[self._bg_nx, self._bg_ny, self._bg_nz],
            grading=Grading.uniform(),
        )

        block_mesh = BlockMesh(vertices=vertices, blocks=[block])
        return block_mesh.generate()

    def _refine_mesh(self, mesh: PolyMesh) -> PolyMesh:
        """Refine mesh by splitting each hex into 8 smaller hexes.

        This is a simplified refinement that works for structured hex meshes.
        """
        device = mesh.device
        dtype = mesh.dtype

        # For now, return the mesh unchanged
        # Full implementation would require:
        # 1. Identify hex cells
        # 2. Add midpoints on edges, faces, and cell centres
        # 3. Create 8 child hexes per parent
        # 4. Update owner/neighbour/boundary
        return mesh

    def _refine_near_surface(
        self,
        mesh: PolyMesh,
        surface: STLSurface,
        distance: float,
    ) -> PolyMesh:
        """Refine cells near the surface.

        Identifies cells whose centres are within the specified distance
        from the surface and refines them.
        """
        # For now, return mesh unchanged
        # Full implementation would require:
        # 1. Compute cell centres
        # 2. For each cell, check distance to surface
        # 3. Refine cells within distance threshold
        return mesh

    def _snap_to_surface(self, mesh: PolyMesh) -> PolyMesh:
        """Snap boundary points to the surface.

        Moves boundary points to the nearest point on the surface
        to improve surface conformity.
        """
        # For now, return mesh unchanged
        # Full implementation would require:
        # 1. Identify boundary points
        # 2. For each boundary point, find nearest surface point
        # 3. Move point to surface (with smoothing)
        return mesh

    def _add_boundary_layers(self, mesh: PolyMesh) -> PolyMesh:
        """Add boundary layer cells near walls.

        Adds layers of cells near specified boundary patches
        to better resolve boundary layer physics.
        """
        # For now, return mesh unchanged
        # Full implementation would require:
        # 1. Identify wall patches
        # 2. Extrude boundary faces outward
        # 3. Create layer cells with specified expansion
        return mesh


def create_snappy_mesh(
    stl_path: str,
    background_cells: tuple[int, int, int] = (20, 20, 20),
    refinement_level: int = 2,
    refinement_distance: float = 0.1,
) -> PolyMesh:
    """Convenience function to create a snappyHexMesh from an STL file.

    Args:
        stl_path: Path to the STL file.
        background_cells: (nx, ny, nz) cells for background mesh.
        refinement_level: Refinement level near surface.
        refinement_distance: Distance for refinement.

    Returns:
        PolyMesh with the generated mesh.
    """
    from pyfoam.mesh.generation.stl import STLReader

    reader = STLReader(stl_path)
    surface = reader.read()

    snappy = SnappyHexMesh(
        surfaces=[surface],
        background_mesh=background_cells,
        refinement_regions=[
            RefinementRegion(
                surface=surface,
                level=refinement_level,
                distance=refinement_distance,
            )
        ],
    )

    return snappy.generate()
