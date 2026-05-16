"""
pyfoam.io — OpenFOAM file format I/O.

Reads and writes all OpenFOAM file formats:
- FoamFile headers (version, format, class, location, object)
- ASCII and binary field files (volScalarField, volVectorField, etc.)
- Mesh files (points, faces, owner, neighbour, boundary)
- Configuration files (controlDict, fvSchemes, fvSolution)
- Complete case directory structure
- Mesh format converters (Gmsh, Fluent, VTK)
"""

from pyfoam.io.binary_io import (
    BinaryReader,
    BinaryWriter,
    read_binary_compact_list_list,
    read_binary_faces,
    read_binary_label_list,
    read_binary_points,
    read_binary_scalar,
    read_binary_scalar_list,
    write_binary_compact_list_list,
    write_binary_faces,
    write_binary_label_list,
    write_binary_points,
    write_binary_scalar,
    write_binary_scalar_list,
)
from pyfoam.io.case import Case
from pyfoam.io.dictionary import (
    FoamDict,
    FoamList,
    Token,
    Tokenizer,
    expand_macros,
    parse_dict,
    parse_dict_file,
)
from pyfoam.io.field_io import (
    BoundaryField,
    BoundaryPatch,
    FieldData,
    parse_boundary_field,
    parse_internal_field,
    read_dimensions,
    read_field,
    write_field,
)
from pyfoam.io.foam_file import (
    FileFormat,
    FoamFileHeader,
    detect_format,
    get_header_end,
    parse_header,
    read_foam_file,
    split_header_body,
    write_foam_file,
)
from pyfoam.io.mesh_io import (
    BoundaryPatch as MeshBoundaryPatch,
    MeshData,
    read_boundary,
    read_faces,
    read_mesh,
    read_neighbour,
    read_owner,
    read_points,
    write_boundary,
    write_faces,
    write_neighbour,
    write_owner,
    write_points,
)
from pyfoam.io.gmsh_io import (
    GmshElement,
    GmshMesh,
    GmshPhysicalGroup,
    gmsh_to_foam,
    read_gmsh,
)
from pyfoam.io.fluent_io import (
    FluentFace,
    FluentMesh,
    FluentZone,
    fluent_to_foam,
    read_fluent,
)
from pyfoam.io.vtk_io import (
    foam_to_vtk,
    write_vtk_unstructured,
    write_vtu_unstructured,
)

__all__ = [
    # Binary I/O
    "BinaryReader",
    "BinaryWriter",
    "read_binary_scalar",
    "read_binary_scalar_list",
    "read_binary_label_list",
    "read_binary_points",
    "read_binary_faces",
    "read_binary_compact_list_list",
    "write_binary_scalar",
    "write_binary_scalar_list",
    "write_binary_label_list",
    "write_binary_points",
    "write_binary_faces",
    "write_binary_compact_list_list",
    # Case
    "Case",
    # Dictionary
    "FoamDict",
    "FoamList",
    "Token",
    "Tokenizer",
    "parse_dict",
    "parse_dict_file",
    "expand_macros",
    # Field I/O
    "FieldData",
    "BoundaryField",
    "BoundaryPatch",
    "read_field",
    "write_field",
    "read_dimensions",
    "parse_internal_field",
    "parse_boundary_field",
    # FoamFile
    "FileFormat",
    "FoamFileHeader",
    "parse_header",
    "read_foam_file",
    "write_foam_file",
    "split_header_body",
    "detect_format",
    "get_header_end",
    # Mesh I/O
    "MeshData",
    "MeshBoundaryPatch",
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
    # Gmsh I/O (gmshToFoam)
    "GmshElement",
    "GmshMesh",
    "GmshPhysicalGroup",
    "read_gmsh",
    "gmsh_to_foam",
    # Fluent I/O (fluentMeshToFoam)
    "FluentFace",
    "FluentMesh",
    "FluentZone",
    "read_fluent",
    "fluent_to_foam",
    # VTK I/O (foamToVTK)
    "write_vtk_unstructured",
    "write_vtu_unstructured",
    "foam_to_vtk",
]
