"""
pyfoam.mesh.generation — Mesh generation tools.

Provides OpenFOAM-compatible mesh generation utilities:

- **BlockMesh**: Structured hex mesh generation from block definitions
- **SnappyHexMesh**: Unstructured mesh generation from STL surfaces
- **ExtrudeMesh**: Extrude 2D mesh to 3D
- **ExtrudeToRegion**: Extrude faces into separate region
"""

from pyfoam.mesh.generation.block_mesh import BlockMesh, Block, Grading
from pyfoam.mesh.generation.snappy_hex_mesh import SnappyHexMesh
from pyfoam.mesh.generation.stl import STLReader
from pyfoam.mesh.generation.extrude_mesh import (
    ExtrudeMesh, ExtrudeModel, LinearExtrude, WedgeExtrude,
    RotationalExtrude, extrude_mesh,
)
from pyfoam.mesh.generation.extrude_to_region import (
    ExtrudeToRegion, RegionExtrudeSpec, extrude_to_region,
)

__all__ = [
    "BlockMesh",
    "Block",
    "Grading",
    "SnappyHexMesh",
    "STLReader",
    # Extrusion
    "ExtrudeMesh",
    "ExtrudeModel",
    "LinearExtrude",
    "WedgeExtrude",
    "RotationalExtrude",
    "extrude_mesh",
    # Region extrusion
    "ExtrudeToRegion",
    "RegionExtrudeSpec",
    "extrude_to_region",
]
