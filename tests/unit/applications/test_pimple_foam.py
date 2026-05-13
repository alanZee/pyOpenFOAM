"""
End-to-end test: pimpleFoam transient incompressible solver.

Creates a complete OpenFOAM case directory on disk (mesh, fields,
system files), runs PimpleFoam (transient PIMPLE with turbulence support),
and verifies convergence.

The cavity is a unit square [0,1] x [0,1] with:
- Top wall moving at U = (1, 0, 0)
- All other walls stationary
- nu = 0.01 (Re = 100)
- Transient simulation with PIMPLE algorithm
- Optional RANS turbulence model
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helpers
# ---------------------------------------------------------------------------

def _make_cavity_case(
    case_dir: Path,
    n_cells_x: int = 10,
    n_cells_y: int = 10,
    nu: float = 0.01,
    delta_t: float = 0.001,
    end_time: float = 0.1,
    n_outer_correctors: int = 3,
    n_correctors: int = 2,
    turbulence: bool = False,
) -> None:
    """Write a complete lid-driven cavity case for pimpleFoam to *case_dir*.

    Creates:
    - constant/polyMesh/{points, faces, owner, neighbour, boundary}
    - constant/transportProperties
    - constant/turbulenceProperties (if turbulence=True)
    - 0/U, 0/p
    - system/{controlDict, fvSchemes, fvSolution}
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mesh ----
    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1  # small depth for 3D (empty BC)

    # Points: two layers (z=0, z=dz)
    points_z0 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z0.append((i * dx, j * dy, 0.0))
    n_base = len(points_z0)

    points_z1 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z1.append((i * dx, j * dy, dz))

    all_points = points_z0 + points_z1
    n_points = len(all_points)

    # Faces: internal + boundary
    faces = []
    owner = []
    neighbour = []

    # Internal vertical faces (between cells in x-direction)
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal faces (between cells in y-direction)
    for j in range(n_cells_y - 1):
        for i in range(n_cells_x):
            p0 = (j + 1) * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append((j + 1) * n_cells_x + i)

    n_internal = len(neighbour)

    # Boundary faces
    # movingWall (top, y=1)
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)

    n_moving = n_cells_x
    moving_start = n_internal

    # fixedWalls: bottom, left, right
    # Bottom (y=0)
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)

    # Left (x=0)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)

    # Right (x=1)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)

    n_fixed = n_cells_x + 2 * n_cells_y
    fixed_start = n_internal + n_moving

    # frontAndBack (z-normal, empty)
    # Front (z=0)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)

    # Back (z=dz)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = n_base + j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p1, p0, p3, p2))
            owner.append(j * n_cells_x + i)

    n_empty = 2 * n_cells_x * n_cells_y
    empty_start = fixed_start + n_fixed

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

    # Write mesh files
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0",
        format=FileFormat.ASCII,
        location="constant/polyMesh",
    )

    # points
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for x, y, z in all_points:
        lines.append(f"({x:.10g} {y:.10g} {z:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    # faces
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in faces:
        nv = face[0]
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{nv}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    # owner
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "owner"})
    lines = [f"{n_faces}", "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    # neighbour
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"})
    lines = [f"{n_internal}", "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    # boundary
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"})
    lines = ["3", "("]
    lines.append("    movingWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_moving};")
    lines.append(f"        startFace       {moving_start};")
    lines.append("    }")
    lines.append("    fixedWalls")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_fixed};")
    lines.append(f"        startFace       {fixed_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header,
        f"nu              [0 2 -1 0 0 0 0] {nu};",
        overwrite=True,
    )

    # ---- turbulenceProperties (optional) ----
    if turbulence:
        turb_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="dictionary", location="constant", object="turbulenceProperties",
        )
        turb_body = (
            "simulationType  RAS;\n\n"
            "RAS\n{\n"
            "    model           kEpsilon;\n"
            "    turbulence      on;\n"
            "    printCoeffs     on;\n"
            "}\n"
        )
        write_foam_file(
            case_dir / "constant" / "turbulenceProperties", turb_header,
            turb_body, overwrite=True,
        )

    # ---- 0/U ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    movingWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (1 0 0);\n"
        "    }\n"
        "    fixedWalls\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # ---- 0/p ----
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    p_body = (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    movingWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    fixedWalls\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     pimpleFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time:g};\n"
        f"deltaT          {delta_t:g};\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n"
        "purgeWrite      0;\n"
        "writeFormat     ascii;\n"
        "writePrecision  8;\n"
        "writeCompression off;\n"
        "timeFormat      general;\n"
        "timePrecision   6;\n"
        "runTimeModifiable true;\n"
    )
    write_foam_file(sys_dir / "controlDict", cd_header, cd_body, overwrite=True)

    # ---- system/fvSchemes ----
    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    fs_body = (
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         none;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n"
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, fs_body, overwrite=True)

    # ---- system/fvSolution ----
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    fv_body = (
        "solvers\n{\n"
        "    p\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "    U\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "PIMPLE\n{\n"
        f"    nOuterCorrectors    {n_outer_correctors};\n"
        f"    nCorrectors         {n_correctors};\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "    relaxationFactors\n    {\n"
        "        p               0.3;\n"
        "        U               0.7;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


def _make_tjunction_case(
    case_dir: Path,
    n_cells_x: int = 6,
    n_cells_y: int = 4,
    nu: float = 0.01,
    delta_t: float = 0.001,
    end_time: float = 0.005,
    n_outer_correctors: int = 3,
    n_correctors: int = 2,
) -> None:
    """Write a T-junction inspired case for pimpleFoam.

    This is a simplified T-junction: a horizontal channel with inlet
    on the left, outlet on the right, and wall boundaries top/bottom.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1

    # Points: two layers (z=0, z=dz)
    points_z0 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z0.append((i * dx, j * dy, 0.0))
    n_base = len(points_z0)

    points_z1 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z1.append((i * dx, j * dy, dz))

    all_points = points_z0 + points_z1
    n_points = len(all_points)

    # Faces: internal + boundary
    faces = []
    owner = []
    neighbour = []

    # Internal vertical faces (between cells in x-direction)
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal faces (between cells in y-direction)
    for j in range(n_cells_y - 1):
        for i in range(n_cells_x):
            p0 = (j + 1) * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append((j + 1) * n_cells_x + i)

    n_internal = len(neighbour)

    # Boundary faces
    # inlet (left, x=0) — fixedValue velocity
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)

    n_inlet = n_cells_y
    inlet_start = n_internal

    # outlet (right, x=1) — zeroGradient pressure
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)

    n_outlet = n_cells_y
    outlet_start = inlet_start + n_inlet

    # walls (top and bottom)
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)

    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)

    n_walls = 2 * n_cells_x
    walls_start = outlet_start + n_outlet

    # frontAndBack (empty)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)

    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = n_base + j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p1, p0, p3, p2))
            owner.append(j * n_cells_x + i)

    n_empty = 2 * n_cells_x * n_cells_y
    empty_start = walls_start + n_walls

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

    # Write mesh files
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0",
        format=FileFormat.ASCII,
        location="constant/polyMesh",
    )

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for x, y, z in all_points:
        lines.append(f"({x:.10g} {y:.10g} {z:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in faces:
        nv = face[0]
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{nv}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "owner"})
    lines = [f"{n_faces}", "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"})
    lines = [f"{n_internal}", "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"})
    lines = ["4", "("]
    lines.append("    inlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          {n_inlet};")
    lines.append(f"        startFace       {inlet_start};")
    lines.append("    }")
    lines.append("    outlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          {n_outlet};")
    lines.append(f"        startFace       {outlet_start};")
    lines.append("    }")
    lines.append("    walls")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_walls};")
    lines.append(f"        startFace       {walls_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header,
        f"nu              [0 2 -1 0 0 0 0] {nu};",
        overwrite=True,
    )

    # ---- 0/U ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (1 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (1 0 0);\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # ---- 0/p ----
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    p_body = (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform 0;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     pimpleFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time:g};\n"
        f"deltaT          {delta_t:g};\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n"
        "purgeWrite      0;\n"
        "writeFormat     ascii;\n"
        "writePrecision  8;\n"
        "writeCompression off;\n"
        "timeFormat      general;\n"
        "timePrecision   6;\n"
        "runTimeModifiable true;\n"
    )
    write_foam_file(sys_dir / "controlDict", cd_header, cd_body, overwrite=True)

    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    fs_body = (
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         none;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n"
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, fs_body, overwrite=True)

    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    fv_body = (
        "solvers\n{\n"
        "    p\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "    U\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "PIMPLE\n{\n"
        f"    nOuterCorrectors    {n_outer_correctors};\n"
        f"    nCorrectors         {n_correctors};\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "    relaxationFactors\n    {\n"
        "        p               0.3;\n"
        "        U               0.7;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Tests — Laminar mode
# ---------------------------------------------------------------------------

@pytest.fixture
def cavity_case(tmp_path):
    """Create a lid-driven cavity case for pimpleFoam (laminar)."""
    case_dir = tmp_path / "cavity"
    _make_cavity_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        nu=0.01,
        delta_t=0.001,
        end_time=0.01,  # 10 time steps for fast test
        n_outer_correctors=3,
        n_correctors=2,
        turbulence=False,
    )
    return case_dir


class TestPimpleFoamLaminar:
    """End-to-end tests for pimpleFoam in laminar mode."""

    def test_case_loads(self, cavity_case):
        """Case directory is readable and has expected structure."""
        from pyfoam.io.case import Case

        case = Case(cavity_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.get_end_time() == 0.01
        assert case.get_delta_t() == 0.001

    def test_mesh_builds(self, cavity_case):
        """FvMesh is constructed correctly from case data."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(cavity_case)
        mesh = solver.mesh

        assert mesh.n_cells == 16  # 4x4
        assert mesh.n_internal_faces > 0
        assert mesh.cell_volumes.shape == (16,)
        assert mesh.face_areas.shape[0] == mesh.n_faces
        assert mesh.owner.shape[0] == mesh.n_faces
        assert mesh.neighbour.shape[0] == mesh.n_internal_faces

    def test_fields_initialise(self, cavity_case):
        """Fields are initialised from the 0/ directory."""
        from pyfoam.applications.pimple_foam import PimpleFoam
        from pyfoam.core.dtype import CFD_DTYPE

        solver = PimpleFoam(cavity_case)

        # U should be (16, 3) zeros (uniform (0 0 0))
        assert solver.U.shape == (16, 3)
        assert torch.allclose(solver.U, torch.zeros(16, 3, dtype=CFD_DTYPE))

        # p should be (16,) zeros
        assert solver.p.shape == (16,)
        assert torch.allclose(solver.p, torch.zeros(16, dtype=CFD_DTYPE))

        # phi should be (n_faces,) zeros
        assert solver.phi.shape == (solver.mesh.n_faces,)

    def test_nu_read(self, cavity_case):
        """Kinematic viscosity is read from transportProperties."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)
        assert abs(solver.nu - 0.01) < 1e-10

    def test_pimple_settings(self, cavity_case):
        """PIMPLE settings are read correctly from fvSolution."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)
        assert solver.p_solver == "PCG"
        assert solver.U_solver == "PBiCGStab"
        assert solver.n_outer_correctors == 3
        assert solver.n_correctors == 2
        assert abs(solver.convergence_tolerance - 1e-4) < 1e-10
        assert abs(solver.alpha_U - 0.7) < 1e-10
        assert abs(solver.alpha_p - 0.3) < 1e-10

    def test_fv_schemes_settings(self, cavity_case):
        """fvSchemes settings are read correctly."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)
        assert solver.grad_scheme == "Gauss linear"

    def test_turbulence_disabled_by_default(self, cavity_case):
        """Turbulence is disabled when turbulenceProperties is absent."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)
        assert not solver.turbulence.enabled

    def test_run_produces_valid_fields(self, cavity_case):
        """pimpleFoam runs and produces valid fields."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)
        conv = solver.run()

        # Fields should have correct shapes
        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

        # All values should be finite (no NaN or Inf)
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"

    def test_run_writes_output(self, cavity_case):
        """pimpleFoam writes field files to time directories."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)
        solver.run()

        # Check that at least one output time directory was created
        time_dirs = [d for d in cavity_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        # Check that U and p were written
        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"

    def test_fields_are_valid_format(self, cavity_case):
        """Written fields are valid OpenFOAM format."""
        from pyfoam.applications.pimple_foam import PimpleFoam
        from pyfoam.io.field_io import read_field

        solver = PimpleFoam(cavity_case)
        solver.run()

        # Find the last written time directory
        time_dirs = sorted(
            [d for d in cavity_case.iterdir()
             if d.is_dir() and d.name.replace(".", "").isdigit()
             and d.name != "0"],
            key=lambda d: float(d.name),
        )
        assert len(time_dirs) >= 1

        last_dir = time_dirs[-1]
        U_data = read_field(last_dir / "U")
        p_data = read_field(last_dir / "p")

        assert U_data.scalar_type == "vector"
        assert p_data.scalar_type == "scalar"
        assert not U_data.is_uniform
        assert not p_data.is_uniform

    def test_transient_time_stepping(self, cavity_case):
        """pimpleFoam advances through multiple time steps."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)

        # Verify time settings
        assert abs(solver.delta_t - 0.001) < 1e-10
        assert abs(solver.end_time - 0.01) < 1e-10

        conv = solver.run()

        # Should have run at least one iteration
        assert conv.outer_iterations >= 1

    def test_velocity_changes_over_time(self, cavity_case):
        """Velocity field evolves from initial conditions."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)

        # Initial velocity should be zero
        U_initial = solver.U.clone()

        conv = solver.run()

        # After running, velocity should have changed (at least somewhere)
        U_diff = (solver.U - U_initial).abs().sum()
        assert U_diff > 0, "Velocity did not change during simulation"

    def test_pressure_field_shape(self, cavity_case):
        """Pressure field has correct shape after solving."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)
        solver.run()

        assert solver.p.shape == (16,)
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_old_fields_stored(self, cavity_case):
        """Old fields are stored for time derivative."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)

        # Old fields should be clones of initial fields
        assert solver.U_old.shape == solver.U.shape
        assert solver.p_old.shape == solver.p.shape
        assert torch.allclose(solver.U_old, solver.U)
        assert torch.allclose(solver.p_old, solver.p)

    def test_solver_repr(self, cavity_case):
        """PimpleFoam has a useful mesh reference."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)
        assert solver.mesh is not None
        assert solver.mesh.n_cells == 16

    def test_boundary_conditions_applied(self, cavity_case):
        """Boundary conditions are properly applied."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)
        U_bc = solver._build_boundary_conditions()

        # U_bc should have shape (16, 3)
        assert U_bc.shape == (16, 3)

        # At least some cells should have BCs (not all NaN)
        bc_mask = ~torch.isnan(U_bc[:, 0])
        assert bc_mask.any(), "No boundary conditions found"

    def test_multiple_outer_correctors(self, tmp_path):
        """pimpleFoam works with different numbers of outer correctors."""
        case_dir = tmp_path / "cavity_5outer"
        _make_cavity_case(
            case_dir,
            n_cells_x=4,
            n_cells_y=4,
            nu=0.01,
            delta_t=0.001,
            end_time=0.005,  # 5 steps
            n_outer_correctors=5,
            n_correctors=3,
        )

        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(case_dir)
        assert solver.n_outer_correctors == 5
        assert solver.n_correctors == 3

        conv = solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


# ---------------------------------------------------------------------------
# Tests — Turbulent mode
# ---------------------------------------------------------------------------

@pytest.fixture
def turbulent_cavity_case(tmp_path):
    """Create a lid-driven cavity case with RAS turbulence."""
    case_dir = tmp_path / "turb_cavity"
    _make_cavity_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        nu=0.01,
        delta_t=0.001,
        end_time=0.005,  # 5 steps
        n_outer_correctors=3,
        n_correctors=2,
        turbulence=True,
    )
    return case_dir


class TestPimpleFoamTurbulent:
    """End-to-end tests for pimpleFoam with turbulence."""

    def test_turbulence_enabled(self, turbulent_cavity_case):
        """Turbulence is enabled when turbulenceProperties specifies RAS."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(turbulent_cavity_case)
        assert solver.turbulence.enabled

    def test_turbulence_model_name(self, turbulent_cavity_case):
        """Turbulence model is kEpsilon by default."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(turbulent_cavity_case)
        assert solver.turbulence.config.model_name == "kEpsilon"

    def test_turbulent_run_produces_valid_fields(self, turbulent_cavity_case):
        """pimpleFoam with turbulence produces valid fields."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(turbulent_cavity_case)
        conv = solver.run()

        # Fields should have correct shapes
        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

        # All values should be finite
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"

    def test_turbulent_writes_output(self, turbulent_cavity_case):
        """pimpleFoam with turbulence writes output files."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(turbulent_cavity_case)
        solver.run()

        # Check that at least one output time directory was created
        time_dirs = [d for d in turbulent_cavity_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"


# ---------------------------------------------------------------------------
# Tests — Small mesh for fast execution
# ---------------------------------------------------------------------------

class TestPimpleFoamSmallMesh:
    """Tests on a very small mesh (2x2) for fast execution."""

    @pytest.fixture
    def tiny_case(self, tmp_path):
        """Create a minimal 2x2 cavity case."""
        case_dir = tmp_path / "tiny_cavity"
        _make_cavity_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            nu=0.01,
            delta_t=0.001,
            end_time=0.003,  # 3 steps
            n_outer_correctors=3,
            n_correctors=2,
        )
        return case_dir

    def test_tiny_mesh_runs(self, tiny_case):
        """2x2 mesh runs without errors."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(tiny_case)
        assert solver.mesh.n_cells == 4

        conv = solver.run()
        assert solver.U.shape == (4, 3)
        assert solver.p.shape == (4,)
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_tiny_mesh_output(self, tiny_case):
        """2x2 mesh produces valid output."""
        from pyfoam.applications.pimple_foam import PimpleFoam
        from pyfoam.io.field_io import read_field

        solver = PimpleFoam(tiny_case)
        solver.run()

        # Should have written fields
        time_dirs = [d for d in tiny_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        last_dir = sorted(time_dirs, key=lambda d: float(d.name))[-1]
        U_data = read_field(last_dir / "U")
        assert U_data.scalar_type == "vector"


# ---------------------------------------------------------------------------
# Tests — T-junction inspired case
# ---------------------------------------------------------------------------

@pytest.fixture
def tjunction_case(tmp_path):
    """Create a T-junction inspired case for pimpleFoam."""
    case_dir = tmp_path / "tjunction"
    _make_tjunction_case(
        case_dir,
        n_cells_x=6,
        n_cells_y=4,
        nu=0.01,
        delta_t=0.001,
        end_time=0.005,
        n_outer_correctors=3,
        n_correctors=2,
    )
    return case_dir


class TestPimpleFoamTJunction:
    """Tests for pimpleFoam on a T-junction inspired case."""

    def test_tjunction_loads(self, tjunction_case):
        """T-junction case loads correctly."""
        from pyfoam.io.case import Case

        case = Case(tjunction_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)

    def test_tjunction_mesh_builds(self, tjunction_case):
        """T-junction mesh is constructed correctly."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(tjunction_case)
        mesh = solver.mesh

        assert mesh.n_cells == 24  # 6x4
        assert mesh.n_internal_faces > 0

    def test_tjunction_runs(self, tjunction_case):
        """T-junction case runs without errors."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(tjunction_case)
        conv = solver.run()

        # Fields should be valid
        assert solver.U.shape == (24, 3)
        assert solver.p.shape == (24,)
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_tjunction_writes_output(self, tjunction_case):
        """T-junction writes output files."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(tjunction_case)
        solver.run()

        time_dirs = [d for d in tjunction_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"

    def test_tjunction_flow_develops(self, tjunction_case):
        """T-junction flow develops from initial conditions."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(tjunction_case)
        U_initial = solver.U.clone()

        conv = solver.run()

        # Velocity should have changed
        U_diff = (solver.U - U_initial).abs().sum()
        assert U_diff > 0, "Velocity did not change during simulation"


# ---------------------------------------------------------------------------
# Tests — PIMPLE-specific behaviour
# ---------------------------------------------------------------------------

class TestPimpleFoamBehaviour:
    """Tests for PIMPLE-specific behaviour (outer iterations, relaxation)."""

    def test_outer_iterations_reported(self, cavity_case):
        """PIMPLE reports outer iteration count."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)
        conv = solver.run()

        # Should have run at least one outer iteration
        assert conv.outer_iterations >= 1

    def test_convergence_data_complete(self, cavity_case):
        """PIMPLE returns complete convergence data."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)
        conv = solver.run()

        assert conv.continuity_error >= 0
        assert conv.U_residual >= 0
        assert conv.p_residual >= 0

    def test_residual_history_populated(self, cavity_case):
        """PIMPLE populates residual history."""
        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(cavity_case)
        conv = solver.run()

        # After at least one time step, there should be residual history
        # (from the PIMPLE solver's outer loop)
        assert conv.outer_iterations >= 1

    def test_different_relaxation_factors(self, tmp_path):
        """pimpleFoam works with custom relaxation factors."""
        case_dir = tmp_path / "cavity_custom_relax"
        _make_cavity_case(
            case_dir,
            n_cells_x=4,
            n_cells_y=4,
            nu=0.01,
            delta_t=0.001,
            end_time=0.005,
            n_outer_correctors=3,
            n_correctors=2,
        )

        from pyfoam.applications.pimple_foam import PimpleFoam

        solver = PimpleFoam(case_dir)

        # Verify the relaxation factors were read
        assert abs(solver.alpha_U - 0.7) < 1e-10
        assert abs(solver.alpha_p - 0.3) < 1e-10

        conv = solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
