"""
Tutorial validation framework — shared helpers.

Provides reusable mesh generation and case setup for OpenFOAM tutorial tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ── 2-D structured hex mesh ────────────────────────────────────────────────

def make_structured_mesh(
    mesh_dir: Path,
    nx: int,
    ny: int,
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    dz: float = 0.1,
) -> None:
    """Create a 2-D structured hex mesh (1 cell thick in z).

    Default patches:
        movingWall   — top face (y = y_max), type wall
        fixedWalls   — left/right/bottom, type wall
        frontAndBack — z faces, type empty
    """
    mesh_dir.mkdir(parents=True, exist_ok=True)

    x0, x1 = x_range
    y0, y1 = y_range
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    # Points: z=0 layer then z=dz layer
    all_points = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            all_points.append((x0 + i * dx, y0 + j * dy, 0.0))
    n_base = len(all_points)
    for j in range(ny + 1):
        for i in range(nx + 1):
            all_points.append((x0 + i * dx, y0 + j * dy, dz))
    n_points = len(all_points)

    faces: list = []
    owner: list = []
    neighbour: list = []

    # Internal x-faces
    for j in range(ny):
        for i in range(nx - 1):
            p0 = j * (nx + 1) + i + 1
            p1 = p0 + nx + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * nx + i)
            neighbour.append(j * nx + i + 1)

    # Internal y-faces
    for j in range(ny - 1):
        for i in range(nx):
            p0 = (j + 1) * (nx + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * nx + i)
            neighbour.append((j + 1) * nx + i)

    n_internal = len(neighbour)

    # movingWall (top, y = y_max)
    for i in range(nx):
        p0 = ny * (nx + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((ny - 1) * nx + i)
    n_moving = nx
    moving_start = n_internal

    # fixedWalls: bottom (y = y_min)
    for i in range(nx):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)

    # fixedWalls: left (x = x_min)
    for j in range(ny):
        p0 = j * (nx + 1)
        p1 = p0 + nx + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * nx)

    # fixedWalls: right (x = x_max)
    for j in range(ny):
        p0 = j * (nx + 1) + nx
        p1 = p0 + nx + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * nx + nx - 1)

    n_fixed = nx + 2 * ny
    fixed_start = n_internal + n_moving

    # frontAndBack: Front (z=0) — outward normal in -z
    for j in range(ny):
        for i in range(nx):
            p0 = j * (nx + 1) + i
            p1 = p0 + 1
            p2 = p1 + nx + 1
            p3 = p0 + nx + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * nx + i)

    # frontAndBack: Back (z=dz) — outward normal in +z (reversed winding)
    for j in range(ny):
        for i in range(nx):
            p0 = n_base + j * (nx + 1) + i
            p1 = p0 + 1
            p2 = p1 + nx + 1
            p3 = p0 + nx + 1
            faces.append((4, p1, p0, p3, p2))
            owner.append(j * nx + i)

    n_empty = 2 * nx * ny
    empty_start = fixed_start + n_fixed
    n_faces = len(faces)

    # ── Write mesh files ──
    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        location="constant/polyMesh",
    )

    # points
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [str(n_points), "("]
    for x, y, z in all_points:
        lines.append(f"({x:.10g} {y:.10g} {z:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    # faces — format: NVertices(v0 v1 v2 vN)
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [str(n_faces), "("]
    for f in faces:
        nv = f[0]
        verts = " ".join(str(v) for v in f[1:])
        lines.append(f"{nv}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    # owner
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "owner"})
    lines = [str(n_faces), "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    # neighbour
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"})
    lines = [str(n_internal), "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    # boundary
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"})
    lines = ["3", "("]
    for name, start, count, bc_type in [
        ("movingWall", moving_start, n_moving, "wall"),
        ("fixedWalls", fixed_start, n_fixed, "wall"),
        ("frontAndBack", empty_start, n_empty, "empty"),
    ]:
        lines.append(f"    {name}")
        lines.append("    {")
        lines.append(f"        type            {bc_type};")
        lines.append(f"        nFaces          {count};")
        lines.append(f"        startFace       {start};")
        lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)


# ── Case file helpers ─────────────────────────────────────────────────────

def write_transport_properties(case_dir: Path, nu: float) -> None:
    h = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", h,
        f"nu              [0 2 -1 0 0 0 0] {nu};",
        overwrite=True,
    )


def write_control_dict(
    case_dir: Path,
    solver: str = "incompressibleFluid",
    delta_t: float = 0.005,
    end_time: float = 1.0,
    write_interval: int = 100,
) -> None:
    h = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    content = (
        f"solver          {solver};\n"
        f"startFrom       startTime;\nstartTime       0;\n"
        f"stopAt          endTime;\nendTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        f"writeControl    timeStep;\nwriteInterval   {write_interval};\n"
        f"purgeWrite      0;\nwriteFormat     ascii;\nwritePrecision  6;\n"
        f"writeCompression off;\ntimeFormat      general;\ntimePrecision   6;\n"
        f"runTimeModifiable true;\n"
    )
    write_foam_file(case_dir / "system" / "controlDict", h, content, overwrite=True)


def write_fv_schemes(case_dir: Path) -> None:
    h = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    content = """\
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
    write_foam_file(case_dir / "system" / "fvSchemes", h, content, overwrite=True)


def write_fv_solution(case_dir: Path, algorithm: str = "SIMPLE") -> None:
    h = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    if algorithm == "PISO":
        content = """\
solvers
{
    p { solver PCG; preconditioner DIC; tolerance 1e-6; relTol 0.01; }
    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.01; }
}
PISO { nCorrectors 2; nNonOrthogonalCorrectors 0; }
"""
    else:
        content = """\
solvers
{
    p { solver GAMG; tolerance 1e-06; relTol 0.01; smoother DICGaussSeidel; }
    "(U|k|epsilon|omega|nuTilda)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-06; relTol 0.01; }
}
SIMPLE { nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }
relaxationFactors { fields { p 0.3; } equations { U 0.7; k 0.7; epsilon 0.7; omega 0.7; nuTilda 0.7; } }
"""
    write_foam_file(case_dir / "system" / "fvSolution", h, content, overwrite=True)


def write_velocity_field(
    case_dir: Path,
    patches: Dict[str, Tuple[float, float, float]],
    internal: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    bc_types: Optional[Dict[str, str]] = None,
) -> None:
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)
    h = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    lines = [
        "dimensions      [0 1 -1 0 0 0 0];",
        f"internalField   uniform ({internal[0]} {internal[1]} {internal[2]});",
        "boundaryField {",
    ]
    for name, val in patches.items():
        bc = (bc_types or {}).get(name, "fixedValue")
        lines.append(f"    {name} {{")
        if bc == "fixedValue":
            lines.append(f"        type            fixedValue;")
            lines.append(f"        value           uniform ({val[0]} {val[1]} {val[2]});")
        elif bc == "noSlip":
            lines.append(f"        type            noSlip;")
        elif bc == "empty":
            lines.append(f"        type            empty;")
        elif bc == "zeroGradient":
            lines.append(f"        type            zeroGradient;")
        lines.append("    }")
    lines.append("}")
    write_foam_file(zero_dir / "U", h, "\n".join(lines), overwrite=True)


def write_pressure_field(
    case_dir: Path,
    patches: Dict[str, str],
    internal: float = 0.0,
) -> None:
    """Write ``0/p``. *patches* maps name -> BC type ('zeroGradient', 'fixedValue', 'empty')."""
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)
    h = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    lines = [
        "dimensions      [0 2 -2 0 0 0 0];",
        f"internalField   uniform {internal};",
        "boundaryField {",
    ]
    for name, bc in patches.items():
        lines.append(f"    {name} {{")
        if bc == "fixedValue":
            lines.append(f"        type            fixedValue;")
            lines.append(f"        value           uniform {internal};")
        elif bc == "zeroGradient":
            lines.append(f"        type            zeroGradient;")
        elif bc == "empty":
            lines.append(f"        type            empty;")
        lines.append("    }")
    lines.append("}")
    write_foam_file(zero_dir / "p", h, "\n".join(lines), overwrite=True)
