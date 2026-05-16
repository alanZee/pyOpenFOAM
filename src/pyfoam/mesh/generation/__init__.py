"""
pyfoam.mesh.generation — Mesh generation tools.

Provides OpenFOAM-compatible mesh generation utilities:

- **BlockMesh**: Structured hex mesh generation from block definitions
- **SnappyHexMesh**: Unstructured mesh generation from STL surfaces
"""

from pyfoam.mesh.generation.block_mesh import BlockMesh, Block, Grading
from pyfoam.mesh.generation.snappy_hex_mesh import SnappyHexMesh
from pyfoam.mesh.generation.stl import STLReader

__all__ = [
    "BlockMesh",
    "Block",
    "Grading",
    "SnappyHexMesh",
    "STLReader",
]
