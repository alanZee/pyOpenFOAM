"""
pyfoam.mesh — Mesh representation and geometry for finite volume methods.

Classes
-------
PolyMesh
    Raw polyhedral mesh topology (points, faces, owner, neighbour, boundary).
FvMesh
    Extends PolyMesh with computed geometric quantities for FVM.

Geometry functions
-----------------
compute_face_centres
    Arithmetic mean of face vertex positions.
compute_face_area_vectors
    Face normal vectors with magnitude equal to face area.
compute_cell_volumes_and_centres
    Cell volumes and centres via tetrahedral decomposition.
compute_face_weights
    Linear interpolation weights for face values.
compute_delta_coefficients
    Diffusion distance factors (1/|d·n|).

Topology utilities
-----------------
validate_owner_neighbour
    Check OpenFOAM owner/neighbour conventions.
build_cell_to_faces
    Cell → face index mapping.
build_face_to_cells
    Face → cell index mapping (n_faces, 2).
cell_neighbours
    Neighbouring cell indices for a given cell.
"""

from pyfoam.mesh.poly_mesh import PolyMesh
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.mesh.mesh_geometry import (
    compute_face_centres,
    compute_face_area_vectors,
    compute_cell_volumes_and_centres,
    compute_face_weights,
    compute_delta_coefficients,
)
from pyfoam.mesh.topology import (
    validate_owner_neighbour,
    build_cell_to_faces,
    build_face_to_cells,
    cell_neighbours,
)

__all__ = [
    # Mesh classes
    "PolyMesh",
    "FvMesh",
    # Geometry
    "compute_face_centres",
    "compute_face_area_vectors",
    "compute_cell_volumes_and_centres",
    "compute_face_weights",
    "compute_delta_coefficients",
    # Topology
    "validate_owner_neighbour",
    "build_cell_to_faces",
    "build_face_to_cells",
    "cell_neighbours",
]
