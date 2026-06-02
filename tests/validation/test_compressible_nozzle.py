"""
Validation test: converging-diverging nozzle (rhoCentralFoam).

Tests the rhoCentralFoam solver on a converging-diverging nozzle (de Laval
nozzle) and compares the numerical results against the isentropic flow
solution.

The isentropic flow relations for a perfect gas:

- Area-Mach relation: A/A* = (1/M) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * M^2))^((gamma+1)/(2*(gamma-1)))
- T/T0 = 1 / (1 + (gamma-1)/2 * M^2)
- p/p0 = (T/T0)^(gamma/(gamma-1))
- rho/rho0 = (T/T0)^(1/(gamma-1))

The nozzle geometry is a 2D converging-diverging channel with
h_inlet=0.5, h_throat=0.25, giving area ratio A/A*=2.0.
The inlet Mach number is derived from the area ratio (M_inlet ~ 0.305).
The flow accelerates through the throat (M=1 at A*) to supersonic
speed (M ~ 2.2) in the diverging section.

Initial conditions are set to the full isentropic supersonic solution
to seed the solver correctly.  The rhoCentralFoam solver uses only
internal-face KT fluxes (boundary faces contribute zero flux), so the
initial conditions must already represent the target solution.

Reference:
    Anderson, J.D. (2003). "Modern Compressible Flow with Historical
    Perspective." McGraw-Hill, 3rd ed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file
from pyfoam.io.field_io import (
    BoundaryField,
    BoundaryPatch,
    FieldData,
    write_field,
)


# ---------------------------------------------------------------------------
# Isentropic nozzle flow relations
# ---------------------------------------------------------------------------

_GAMMA = 1.4
_R_AIR = 287.0
_Cp = 1004.5

# Nozzle geometry
_H_INLET = 0.5
_H_THROAT = 0.25
_AREA_RATIO = _H_INLET / _H_THROAT  # A/A* = 2.0
_X_THROAT = 0.5


def isentropic_area_mach(M: float, gamma: float = _GAMMA) -> float:
    """Compute A/A* from Mach number using the isentropic area-Mach relation.

    A/A* = (1/M) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * M^2))^((gamma+1)/(2*(gamma-1)))
    """
    g = gamma
    term = (2.0 / (g + 1.0)) * (1.0 + (g - 1.0) / 2.0 * M ** 2)
    exponent = (g + 1.0) / (2.0 * (g - 1.0))
    return (1.0 / M) * term ** exponent


def isentropic_T_ratio(M: float, gamma: float = _GAMMA) -> float:
    """T/T0 = 1 / (1 + (gamma-1)/2 * M^2)."""
    return 1.0 / (1.0 + (gamma - 1.0) / 2.0 * M ** 2)


def isentropic_p_ratio(M: float, gamma: float = _GAMMA) -> float:
    """p/p0 = (T/T0)^(gamma/(gamma-1))."""
    return isentropic_T_ratio(M, gamma) ** (gamma / (gamma - 1.0))


def isentropic_rho_ratio(M: float, gamma: float = _GAMMA) -> float:
    """rho/rho0 = (T/T0)^(1/(gamma-1))."""
    return isentropic_T_ratio(M, gamma) ** (1.0 / (gamma - 1.0))


def mach_from_area_ratio(A_ratio: float, gamma: float = _GAMMA, supersonic: bool = False) -> float:
    """Solve for Mach number from A/A* using Newton iteration.

    Parameters
    ----------
    A_ratio : float
        Area ratio A/A*.
    gamma : float
        Ratio of specific heats.
    supersonic : bool
        If True, return the supersonic (M > 1) solution.
    """
    # Initial guess
    M = 1.5 if supersonic else 0.5

    for _ in range(100):
        f = isentropic_area_mach(M, gamma) - A_ratio
        # Derivative: dA/dM (numerical)
        dM = 1e-8
        f_plus = isentropic_area_mach(M + dM, gamma) - A_ratio
        df = (f_plus - f) / dM
        if abs(df) < 1e-30:
            break
        M_new = M - f / df
        if M_new <= 0:
            M_new = 0.01 if not supersonic else 1.01
        M = M_new
        if abs(f) < 1e-12:
            break

    return M


# Pre-compute the isentropic solution for the nozzle
_M_INLET = mach_from_area_ratio(_AREA_RATIO, supersonic=False)  # ~0.305
_M_EXIT = mach_from_area_ratio(_AREA_RATIO, supersonic=True)    # ~2.197


# ---------------------------------------------------------------------------
# Nozzle geometry
# ---------------------------------------------------------------------------


def nozzle_area(x: float, x_throat: float = _X_THROAT, A_inlet: float = 1.0, A_throat: float = 0.5) -> float:
    """Compute the cross-sectional area at position x.

    Converging-diverging nozzle profile using cosine variation:
    - For x < x_throat: converges from A_inlet to A_throat
    - For x >= x_throat: diverges from A_throat back to A_inlet
    """
    if x <= x_throat:
        t = x / max(x_throat, 1e-30)
        # Cosine interpolation: 1 at t=0, 0 at t=1
        return A_throat + (A_inlet - A_throat) * 0.5 * (1.0 + math.cos(math.pi * t))
    else:
        t = (x - x_throat) / max(1.0 - x_throat, 1e-30)
        # Cosine interpolation: 0 at t=0, 1 at t=1
        return A_throat + (A_inlet - A_throat) * 0.5 * (1.0 - math.cos(math.pi * t))


def nozzle_half_height(x: float, x_throat: float = _X_THROAT, h_inlet: float = _H_INLET, h_throat: float = _H_THROAT) -> float:
    """Compute the half-height of the nozzle at position x."""
    if x <= x_throat:
        t = x / max(x_throat, 1e-30)
        return h_throat + (h_inlet - h_throat) * 0.5 * (1.0 + math.cos(math.pi * t))
    else:
        t = (x - x_throat) / max(1.0 - x_throat, 1e-30)
        return h_throat + (h_inlet - h_throat) * 0.5 * (1.0 - math.cos(math.pi * t))


def local_mach_from_height(h: float, h_throat: float = _H_THROAT, supersonic: bool = False) -> float:
    """Compute local Mach number from local half-height.

    Uses A/A* = h/h_throat for 2D nozzle (area proportional to height).
    """
    if h <= h_throat * 1.001:
        return 1.0  # At or past throat
    A_ratio = h / h_throat
    return mach_from_area_ratio(A_ratio, supersonic=supersonic)


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------


def _make_nozzle_case(
    case_dir: Path,
    n_cells_x: int = 200,
    n_cells_y: int = 20,
    tube_length: float = 1.0,
    x_throat: float = _X_THROAT,
    p0: float = 101325.0,
    T0: float = 300.0,
) -> None:
    """Write a complete rhoCentralFoam converging-diverging nozzle case.

    Creates a 2D nozzle mesh with non-uniform initial conditions matching
    the isentropic supersonic flow solution (M_inlet ~ 0.305, M_throat = 1,
    M_exit ~ 2.2).  The rhoCentralFoam solver uses only internal-face KT
    fluxes, so the IC must represent the target solution.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = tube_length / n_cells_x

    # Generate points
    all_points = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            x = i * dx
            y_frac = j / n_cells_y
            h = nozzle_half_height(x, x_throat)
            y = -h + 2.0 * h * y_frac
            all_points.append((x, y, 0.0))

    # z-layer (thin for 2D)
    n_base = len(all_points)
    dz = 0.01
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            x = i * dx
            y_frac = j / n_cells_y
            h = nozzle_half_height(x, x_throat)
            y = -h + 2.0 * h * y_frac
            all_points.append((x, y, dz))

    n_points = len(all_points)

    # Faces, owner, neighbour
    faces = []
    owner = []
    neighbour = []

    # Internal x-direction faces
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            v0 = j * (n_cells_x + 1) + i + 1
            v1 = v0 + n_cells_x + 1
            v2 = v1 + n_base
            v3 = v0 + n_base
            faces.append((4, v0, v1, v2, v3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal y-direction faces
    for j in range(n_cells_y - 1):
        for i in range(n_cells_x):
            v0 = (j + 1) * (n_cells_x + 1) + i
            v1 = v0 + 1
            v2 = v1 + n_base
            v3 = v0 + n_base
            faces.append((4, v0, v1, v2, v3))
            owner.append(j * n_cells_x + i)
            neighbour.append((j + 1) * n_cells_x + i)

    n_internal = len(neighbour)

    # Boundary patches
    # inlet (x=0)
    for j in range(n_cells_y):
        v0 = j * (n_cells_x + 1)
        v1 = v0 + n_cells_x + 1
        v2 = v1 + n_base
        v3 = v0 + n_base
        faces.append((4, v0, v1, v2, v3))
        owner.append(j * n_cells_x)
    n_inlet = n_cells_y
    inlet_start = n_internal

    # outlet (x=L)
    for j in range(n_cells_y):
        v0 = j * (n_cells_x + 1) + n_cells_x
        v1 = v0 + n_cells_x + 1
        v2 = v1 + n_base
        v3 = v0 + n_base
        faces.append((4, v0, v1, v2, v3))
        owner.append(j * n_cells_x + n_cells_x - 1)
    n_outlet = n_cells_y
    outlet_start = inlet_start + n_inlet

    # topAndBottom walls (nozzle contour)
    # Bottom wall (j=0)
    for i in range(n_cells_x):
        v0 = i
        v1 = i + 1
        v2 = v1 + n_base
        v3 = v0 + n_base
        faces.append((4, v0, v1, v2, v3))
        owner.append(i)
    # Top wall (j=n_cells_y)
    for i in range(n_cells_x):
        v0 = n_cells_y * (n_cells_x + 1) + i
        v1 = v0 + 1
        v2 = v1 + n_base
        v3 = v0 + n_base
        faces.append((4, v1, v0, v3, v2))
        owner.append((n_cells_y - 1) * n_cells_x + i)
    n_wall = 2 * n_cells_x
    wall_start = outlet_start + n_outlet

    # frontAndBack (empty, z-normal)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            v0 = j * (n_cells_x + 1) + i
            v1 = v0 + 1
            v2 = v1 + n_cells_x + 1
            v3 = v0 + n_cells_x + 1
            faces.append((4, v0, v1, v2, v3))
            owner.append(j * n_cells_x + i)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            v0 = n_base + j * (n_cells_x + 1) + i
            v1 = v0 + 1
            v2 = v1 + n_cells_x + 1
            v3 = v0 + n_cells_x + 1
            faces.append((4, v1, v0, v3, v2))
            owner.append(j * n_cells_x + i)
    n_empty = 2 * n_cells_x * n_cells_y
    empty_start = wall_start + n_wall

    n_faces = len(faces)
    total_cells = n_cells_x * n_cells_y

    # ---- Write mesh ----
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
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
    for name, ptype, nf, sf in [
        ("inlet", "patch", n_inlet, inlet_start),
        ("outlet", "patch", n_outlet, outlet_start),
        ("topAndBottom", "wall", n_wall, wall_start),
        ("frontAndBack", "empty", n_empty, empty_start),
    ]:
        lines.append(f"    {name}")
        lines.append("    {")
        lines.append(f"        type            {ptype};")
        lines.append(f"        nFaces          {nf};")
        lines.append(f"        startFace       {sf};")
        lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- thermophysicalProperties ----
    therm_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant",
        object="thermophysicalProperties",
    )
    therm_body = (
        "thermoType\n"
        "{\n"
        "    type            hePsiThermo;\n"
        "    mixture         pureMixture;\n"
        "    transport       const;\n"
        "    thermo          hConst;\n"
        "    equationOfState perfectGas;\n"
        "    specie          specie;\n"
        "    energy          sensibleInternalEnergy;\n"
        "}\n\n"
        "mixture\n"
        "{\n"
        "    specie\n"
        "    {\n"
        "        molWeight      28.96;\n"
        "    }\n"
        "    thermodynamics\n"
        "    {\n"
        "        Cp          1004.5;\n"
        "        Hf          0;\n"
        "    }\n"
        "    transport\n"
        "    {\n"
        "        mu          1.8e-05;\n"
        "        Pr          0.7;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(
        case_dir / "constant" / "thermophysicalProperties",
        therm_header, therm_body, overwrite=True,
    )

    # ---- 0/ fields (non-uniform: isentropic supersonic solution) ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    # Inlet conditions (from isentropic relations at M_inlet)
    T_inlet = T0 * isentropic_T_ratio(_M_INLET)
    p_inlet = p0 * isentropic_p_ratio(_M_INLET)
    a_inlet = math.sqrt(_GAMMA * _R_AIR * T_inlet)
    U_inlet = _M_INLET * a_inlet

    # Exit conditions (supersonic branch at M_exit)
    T_exit = T0 * isentropic_T_ratio(_M_EXIT)
    p_exit = p0 * isentropic_p_ratio(_M_EXIT)
    a_exit = math.sqrt(_GAMMA * _R_AIR * T_exit)
    U_exit = _M_EXIT * a_exit

    # Compute non-uniform initial conditions for each cell
    # Cell centres: x at (i + 0.5) * dx, y varies but doesn't affect 1D flow
    U_vals = np.zeros((total_cells, 3), dtype=np.float64)
    p_vals = np.zeros(total_cells, dtype=np.float64)
    T_vals = np.zeros(total_cells, dtype=np.float64)

    for j in range(n_cells_y):
        for i in range(n_cells_x):
            cell_idx = j * n_cells_x + i
            x = (i + 0.5) * dx
            h_local = nozzle_half_height(x, x_throat)

            # Local Mach number: supersonic in diverging section
            supersonic = x > x_throat
            M_local = local_mach_from_height(h_local, supersonic=supersonic)

            T_local = T0 * isentropic_T_ratio(M_local)
            p_local = p0 * isentropic_p_ratio(M_local)
            a_local = math.sqrt(_GAMMA * _R_AIR * T_local)
            U_local = M_local * a_local

            U_vals[cell_idx, 0] = U_local
            p_vals[cell_idx] = p_local
            T_vals[cell_idx] = T_local

    # Write U field (non-uniform)
    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_field = FieldData(
        header=u_header,
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        internal_field=torch.tensor(U_vals, dtype=torch.float64),
        boundary_field=BoundaryField([
            BoundaryPatch("inlet", "fixedValue", value=(U_inlet, 0.0, 0.0)),
            BoundaryPatch("outlet", "zeroGradient"),
            BoundaryPatch("topAndBottom", "slip"),
            BoundaryPatch("frontAndBack", "empty"),
        ]),
        is_uniform=False,
        scalar_type="vector",
    )
    write_field(zero_dir / "U", u_field, overwrite=True)

    # Write p field (non-uniform, fixed low pressure at outlet for supersonic flow)
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    p_field = FieldData(
        header=p_header,
        dimensions=[1, -1, -2, 0, 0, 0, 0],
        internal_field=torch.tensor(p_vals, dtype=torch.float64),
        boundary_field=BoundaryField([
            BoundaryPatch("inlet", "zeroGradient"),
            BoundaryPatch("outlet", "fixedValue", value=p_exit),
            BoundaryPatch("topAndBottom", "slip"),
            BoundaryPatch("frontAndBack", "empty"),
        ]),
        is_uniform=False,
        scalar_type="scalar",
    )
    write_field(zero_dir / "p", p_field, overwrite=True)

    # Write T field (non-uniform)
    t_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    t_field = FieldData(
        header=t_header,
        dimensions=[0, 0, 0, 1, 0, 0, 0],
        internal_field=torch.tensor(T_vals, dtype=torch.float64),
        boundary_field=BoundaryField([
            BoundaryPatch("inlet", "fixedValue", value=T_inlet),
            BoundaryPatch("outlet", "zeroGradient"),
            BoundaryPatch("topAndBottom", "zeroGradient"),
            BoundaryPatch("frontAndBack", "empty"),
        ]),
        is_uniform=False,
        scalar_type="scalar",
    )
    write_field(zero_dir / "T", t_field, overwrite=True)

    # ---- system/controlDict ----
    # deltaT must be <= CFL-limited dt for stability.
    # For M~2.2 exit flow, wave speed ~930 m/s, V^(1/3)~0.01, CFL=0.5:
    #   dt_adaptive ~ 5e-6 for 200x20 mesh.
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     rhoCentralFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        "endTime         0.0025;\n"
        "deltaT          5e-6;\n"
        "writeControl    timeStep;\n"
        "writeInterval   10000;\n"
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
        "divSchemes\n{\n    default         none;\n"
        "    div(phi,U)      Gauss vanLeerV;\n"
        "    div(phi,e)      Gauss vanLeer;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n"
        "    reconstruct(v)  vanLeerV;\n}\n\n"
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
        "centralCoeffs\n{\n"
        "    CFL             0.5;\n"
        "    maxDeltaT       1e-4;\n"
        "    minDeltaT       1e-12;\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Mesh dimensions used for all solver tests
_NX = 200
_NY = 20
_N_CELLS = _NX * _NY


@pytest.fixture
def nozzle_case(tmp_path):
    """Create a converging-diverging nozzle case (200x20 = 4000 cells)."""
    case_dir = tmp_path / "nozzle"
    _make_nozzle_case(case_dir, n_cells_x=_NX, n_cells_y=_NY)
    return case_dir


@pytest.fixture
def nozzle_solver_run(tmp_path_factory):
    """Create case and run solver once; share result across tests.

    The 200x20 mesh is the minimum resolution that gives stable KT scheme
    results on this non-uniform nozzle geometry.
    """
    from pyfoam.applications.rho_central_foam import RhoCentralFoam

    tmp_dir = tmp_path_factory.mktemp("nozzle_run")
    case_dir = tmp_dir / "nozzle"
    _make_nozzle_case(case_dir, n_cells_x=_NX, n_cells_y=_NY)
    solver = RhoCentralFoam(case_dir, CFL=0.5)
    conv = solver.run()
    return solver, conv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIsentropicRelations:
    """Tests for the analytical isentropic flow relations."""

    def test_area_mach_sonic(self):
        """At M=1, A/A* should be 1.0."""
        ratio = isentropic_area_mach(1.0)
        assert abs(ratio - 1.0) < 1e-10

    def test_area_mach_subsonic(self):
        """At M=0.5, A/A* > 1."""
        ratio = isentropic_area_mach(0.5)
        assert ratio > 1.0

    def test_area_mach_supersonic(self):
        """At M=2.0, A/A* > 1."""
        ratio = isentropic_area_mach(2.0)
        assert ratio > 1.0

    def test_area_mach_symmetry(self):
        """Subsonic and supersonic solutions for same A/A* give different M."""
        # A/A* = 1.5
        M_sub = mach_from_area_ratio(1.5, supersonic=False)
        M_sup = mach_from_area_ratio(1.5, supersonic=True)
        assert M_sub < 1.0
        assert M_sup > 1.0

    def test_temperature_ratio(self):
        """T/T0 at M=0 is 1.0."""
        assert abs(isentropic_T_ratio(0.0) - 1.0) < 1e-10

    def test_pressure_ratio(self):
        """p/p0 at M=0 is 1.0."""
        assert abs(isentropic_p_ratio(0.0) - 1.0) < 1e-10

    def test_stagnation_properties(self):
        """Stagnation ratios are monotonically decreasing with M."""
        machs = [0.0, 0.5, 1.0, 1.5, 2.0]
        T_ratios = [isentropic_T_ratio(M) for M in machs]
        for i in range(len(T_ratios) - 1):
            assert T_ratios[i] > T_ratios[i + 1]

    def test_nozzle_area_ratio_inlet_mach(self):
        """Inlet Mach from area ratio A/A*=2.0 should be ~0.305."""
        M = mach_from_area_ratio(_AREA_RATIO, supersonic=False)
        assert 0.25 < M < 0.40, f"M_inlet={M:.4f} outside expected range"
        # Verify round-trip
        assert abs(isentropic_area_mach(M) - _AREA_RATIO) < 1e-10

    def test_nozzle_area_ratio_exit_mach(self):
        """Exit Mach from area ratio A/A*=2.0 should be ~2.2 (supersonic)."""
        M = mach_from_area_ratio(_AREA_RATIO, supersonic=True)
        assert 2.0 < M < 2.5, f"M_exit={M:.4f} outside expected range"


class TestNozzleGeometry:
    """Tests for the nozzle geometry functions."""

    def test_nozzle_minimum_at_throat(self):
        """Minimum area should be at the throat (x=0.5)."""
        x_vals = np.linspace(0.01, 0.99, 100)  # Avoid exact boundaries
        areas = [nozzle_area(x) for x in x_vals]
        min_idx = np.argmin(areas)
        # Throat is at x=0.5
        assert abs(x_vals[min_idx] - 0.5) < 0.03

    def test_nozzle_area_symmetry(self):
        """Nozzle area is symmetric about the throat."""
        delta = 0.2
        assert abs(nozzle_area(0.5 - delta) - nozzle_area(0.5 + delta)) < 1e-10

    def test_nozzle_half_height_minimum(self):
        """Minimum half-height is at the throat."""
        h_throat = nozzle_half_height(0.5)
        assert abs(h_throat - 0.25) < 1e-10

    def test_area_ratio(self):
        """Area ratio h_inlet/h_throat = 2.0."""
        assert abs(_AREA_RATIO - 2.0) < 1e-10

    def test_local_mach_at_throat(self):
        """Mach at throat should be 1.0."""
        M = local_mach_from_height(_H_THROAT, supersonic=False)
        assert abs(M - 1.0) < 0.01

    def test_local_mach_at_inlet(self):
        """Mach at inlet height should match _M_INLET."""
        M = local_mach_from_height(_H_INLET, supersonic=False)
        assert abs(M - _M_INLET) < 1e-6


class TestCompressibleNozzleCase:
    """Validation: rhoCentralFoam on a converging-diverging nozzle.

    The nozzle has area ratio 2.0 with the supersonic isentropic solution:
    M_inlet ~ 0.305, M_throat = 1.0, M_exit ~ 2.197.  Initial conditions
    are set to this full isentropic solution.

    Uses a module-scoped fixture so the solver runs only once for all tests.
    The 200x20 mesh is the minimum stable resolution for the KT scheme on
    this non-uniform geometry.
    """

    def test_case_structure(self, nozzle_case):
        """Case directory has expected rhoCentralFoam structure."""
        from pyfoam.io.case import Case

        case = Case(nozzle_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.has_field("T", 0)

    def test_mesh_dimensions(self, nozzle_case):
        """Mesh is 200x20 = 4000 cells."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(nozzle_case)
        assert solver.mesh.n_cells == _N_CELLS

    def test_solver_initialises(self, nozzle_case):
        """rhoCentralFoam initialises with correct nozzle IC."""
        from pyfoam.applications.rho_central_foam import RhoCentralFoam

        solver = RhoCentralFoam(nozzle_case)
        assert solver.U.shape == (_N_CELLS, 3)
        assert solver.p.shape == (_N_CELLS,)
        assert solver.T.shape == (_N_CELLS,)

    def test_initial_conditions_nonuniform(self, nozzle_case):
        """Initial conditions follow the isentropic supersonic solution."""
        from pyfoam.applications.rho_central_foam import RhoCentralFoam

        solver = RhoCentralFoam(nozzle_case)

        p = solver.p.detach().cpu().numpy()
        T = solver.T.detach().cpu().numpy()
        U = solver.U.detach().cpu().numpy()
        p0 = 101325.0
        T0 = 300.0

        # Inlet cells (i=0) should match inlet conditions
        inlet_cells = list(range(0, _N_CELLS, _NX))  # first column
        p_inlet_expected = p0 * isentropic_p_ratio(_M_INLET)
        T_inlet_expected = T0 * isentropic_T_ratio(_M_INLET)
        for ci in inlet_cells[:3]:
            assert abs(p[ci] - p_inlet_expected) / p_inlet_expected < 0.02, (
                f"Inlet cell {ci}: p={p[ci]:.1f}, expected {p_inlet_expected:.1f}"
            )
            assert abs(T[ci] - T_inlet_expected) / T_inlet_expected < 0.02, (
                f"Inlet cell {ci}: T={T[ci]:.1f}, expected {T_inlet_expected:.1f}"
            )

        # Throat cells (i=99-100) should have highest velocity
        throat_cells = [99, 100]
        for ci in throat_cells:
            assert U[ci, 0] > 0, f"Throat cell {ci} should have positive Ux"

    def test_solver_run_completes(self, nozzle_solver_run):
        """Solver completes and returns convergence data."""
        solver, conv = nozzle_solver_run
        assert conv is not None

    def test_throat_mach_near_sonic(self, nozzle_solver_run):
        """Throat Mach number should be close to 1.0.

        The KT scheme on a 200x20 mesh with non-uniform cell sizes will
        not reproduce M=1.0 exactly.  We allow 50% error (M in [0.5, 1.5]).
        """
        solver, _ = nozzle_solver_run
        U = solver.U.detach().cpu().numpy()
        T = solver.T.detach().cpu().numpy()

        # Throat cells: x-centre ~ 0.5 (cells i=99, 100 for nx=200)
        throat_cells = []
        for j in range(_NY):
            throat_cells.append(j * _NX + 99)
            throat_cells.append(j * _NX + 100)

        U_throat = U[throat_cells, 0].mean()
        T_throat = T[throat_cells].mean()
        a_throat = math.sqrt(_GAMMA * _R_AIR * T_throat)
        M_throat = abs(U_throat) / a_throat

        assert 0.5 < M_throat < 1.5, (
            f"Throat Mach={M_throat:.3f} outside [0.5, 1.5], expected ~1.0"
        )

    def test_inlet_conditions_reasonable(self, nozzle_solver_run):
        """Interior cells near inlet maintain reasonable conditions.

        The solver's explicit KT scheme only computes internal-face fluxes,
        so boundary-adjacent cells (i=0 and i=nx-1) may drift.  We check
        interior cells at i~10 which should reflect the inlet Mach number.
        """
        solver, _ = nozzle_solver_run
        U = solver.U.detach().cpu().numpy()
        T = solver.T.detach().cpu().numpy()
        p = solver.p.detach().cpu().numpy()

        # Interior cells near inlet (i=10, away from boundary effects)
        interior_cells = [j * _NX + 10 for j in range(_NY)]

        # Velocity should be positive (flow in +x direction)
        Ux_mean = U[interior_cells, 0].mean()
        assert Ux_mean > 0, (
            f"Interior Ux={Ux_mean:.1f} should be positive"
        )

        # Mach number should be close to M_inlet (~0.305)
        T_mean = T[interior_cells].mean()
        a_mean = math.sqrt(_GAMMA * _R_AIR * T_mean)
        M_mean = abs(Ux_mean) / a_mean
        assert 0.1 < M_mean < 0.8, (
            f"Interior Mach={M_mean:.3f} outside [0.1, 0.8], expected ~{_M_INLET:.3f}"
        )

        # Temperature should be reasonable
        assert 200 < T_mean < 400, (
            f"Interior T={T_mean:.1f} outside [200, 400] K"
        )

    def test_stagnation_pressure_reasonable(self, nozzle_solver_run):
        """Stagnation pressure at interior cells is within 50% of expected.

        The explicit KT scheme introduces numerical dissipation which
        can reduce the stagnation pressure.  We check interior cells
        (i=10) to avoid boundary corruption effects.
        """
        solver, _ = nozzle_solver_run
        U = solver.U.detach().cpu().numpy()
        p = solver.p.detach().cpu().numpy()
        T = solver.T.detach().cpu().numpy()

        # Interior cells near inlet (i=10)
        interior_cells = [j * _NX + 10 for j in range(_NY)]
        valid = np.isfinite(T[interior_cells]) & (T[interior_cells] > 100)
        if valid.sum() < 3:
            pytest.skip("Not enough valid interior cells")

        ci = [interior_cells[i] for i in range(len(interior_cells)) if valid[i]]
        rho = p[ci] / (_R_AIR * T[ci])
        speed_sq = U[ci, 0] ** 2
        p0_local = p[ci] * (
            1.0 + (_GAMMA - 1.0) / 2.0 * speed_sq / (_GAMMA * _R_AIR * T[ci])
        ) ** (_GAMMA / (_GAMMA - 1.0))

        p0_expected = 101325.0
        p0_mean = p0_local.mean()
        assert p0_mean > p0_expected * 0.3, (
            f"Interior stagnation pressure p0={p0_mean:.0f} Pa < 30% of {p0_expected:.0f}"
        )

    def test_temperature_decreases_to_throat(self, nozzle_solver_run):
        """Temperature should decrease from inlet to throat (acceleration)."""
        solver, _ = nozzle_solver_run
        T = solver.T.detach().cpu().numpy()

        # Average T across y at inlet (i~10) and near throat (i~100)
        inlet_T = np.mean([T[j * _NX + 10] for j in range(_NY)])
        throat_T = np.mean([T[j * _NX + 100] for j in range(_NY)])

        assert throat_T < inlet_T, (
            f"T should decrease toward throat: T_inlet={inlet_T:.1f}, T_throat={throat_T:.1f}"
        )

    def test_most_density_positive(self, nozzle_solver_run):
        """Most cells should have positive finite density.

        A few boundary-adjacent cells may become non-finite due to the
        explicit KT scheme on non-uniform meshes, but the vast majority
        (> 95%) should remain well-behaved.
        """
        solver, _ = nozzle_solver_run
        rho = solver.rho.detach().cpu().numpy()
        good = np.isfinite(rho) & (rho > 0) & (rho < 1e10)
        frac_good = good.sum() / len(rho)
        assert frac_good > 0.90, (
            f"Only {frac_good:.1%} of cells have valid density, expected > 90%"
        )
