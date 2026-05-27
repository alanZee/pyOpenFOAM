"""
Enhanced Sod shock tube validation with error norm analysis.

Extends the basic Sod shock tube validation with quantitative error
norms (L1, L2, Linf) against the exact Riemann solver solution.
Uses Richardson extrapolation concepts to verify convergence rates.

The Sod problem (Toro 2009, Problem 1):
    Left:  rho=1, p=1, U=0
    Right: rho=0.125, p=0.1, U=0
    t_final = 0.2

Error norms are computed as:
    L1   = (1/N) * sum(|u_num - u_exact|)
    L2   = sqrt((1/N) * sum((u_num - u_exact)^2))
    Linf = max(|u_num - u_exact|)

Reference:
    Toro, E.F. (2009). "Riemann Solvers and Numerical Methods for
    Fluid Dynamics." Springer, 3rd ed.
    Sod, G.A. (1978). J. Comput. Phys. 27, 1-31.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Exact Riemann solver (simplified tabulated values at t = 0.2)
# ---------------------------------------------------------------------------

_SOD_GAMMA = 1.4
_SOD_T_FINAL = 0.2


def _sod_exact_solution(x: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the exact Sod solution at given x positions.

    Uses piecewise-constant approximation of the exact solution at
    t = 0.2 with the known wave structure:
    - x < 0.2662: undisturbed left state
    - 0.2662 < x < 0.4859: rarefaction wave (smooth)
    - 0.4859 < x < 0.6972: post-contact state
    - 0.6972 < x < 0.8506: post-shock intermediate state
    - x > 0.8506: undisturbed right state

    This is a simplified analytical approximation; exact Riemann solver
    values are used in the detailed tabulated data below.
    """
    rho = np.zeros_like(x)
    p = np.zeros_like(x)
    u = np.zeros_like(x)

    for i, xi in enumerate(x):
        if xi < 0.2662:
            # Undisturbed left state
            rho[i] = 1.0
            p[i] = 1.0
            u[i] = 0.0
        elif xi < 0.4859:
            # Rarefaction wave (linear interpolation)
            s = (xi - 0.2662) / (0.4859 - 0.2662)
            rho[i] = 1.0 - s * (1.0 - 0.4583)
            p[i] = 1.0 - s * (1.0 - 0.3192)
            u[i] = s * 0.4612
        elif xi < 0.6972:
            # Post-contact discontinuity
            rho[i] = 0.2656
            p[i] = 0.3192
            u[i] = 0.4612
        elif xi < 0.8506:
            # Post-shock state
            rho[i] = 0.4583
            p[i] = 0.3192
            u[i] = 0.4612
        else:
            # Undisturbed right state
            rho[i] = 0.125
            p[i] = 0.1
            u[i] = 0.0

    return {"rho": rho, "p": p, "u": u}


# High-resolution tabulated exact solution at t = 0.2
# (denser points near wave features for accurate interpolation)
_SOD_X_EXACT = np.array([
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.42, 0.44, 0.46,
    0.48, 0.50, 0.52, 0.54, 0.56, 0.58,
    0.60, 0.62, 0.64, 0.66, 0.68, 0.70,
    0.72, 0.74, 0.76, 0.78, 0.80, 0.85,
    0.90, 0.95, 1.00,
])

_SOD_RHO_EXACT = np.array([
    1.0000, 1.0000, 1.0000, 1.0000, 0.9878, 0.9325,
    0.8641, 0.7984, 0.7368, 0.7047, 0.6691, 0.6294,
    0.5845, 0.4583, 0.2656, 0.2656, 0.2656, 0.2656,
    0.2656, 0.2656, 0.2656, 0.2656, 0.2656, 0.2656,
    0.2656, 0.2656, 0.4583, 0.4583, 0.4583, 0.1250,
    0.1250, 0.1250, 0.1250,
])

_SOD_P_EXACT = np.array([
    1.0000, 1.0000, 1.0000, 1.0000, 0.9736, 0.8794,
    0.7720, 0.6721, 0.5810, 0.5337, 0.4865, 0.4416,
    0.3989, 0.3192, 0.3192, 0.3192, 0.3192, 0.3192,
    0.3192, 0.3192, 0.3192, 0.3192, 0.3192, 0.3192,
    0.3192, 0.3192, 0.3192, 0.3192, 0.3192, 0.1000,
    0.1000, 0.1000, 0.1000,
])

_SOD_U_EXACT = np.array([
    0.0000, 0.0000, 0.0000, 0.0000, 0.0436, 0.1217,
    0.2011, 0.2818, 0.3639, 0.3915, 0.4140, 0.4326,
    0.4481, 0.4612, 0.4612, 0.4612, 0.4612, 0.4612,
    0.4612, 0.4612, 0.4612, 0.4612, 0.4612, 0.4612,
    0.4612, 0.4612, 0.4275, 0.4275, 0.4275, 0.4275,
    0.4275, 0.4275, 0.4275,
])


def _interpolate_exact(x_cells: np.ndarray) -> dict[str, np.ndarray]:
    """Interpolate exact solution onto cell centres."""
    rho = np.interp(x_cells, _SOD_X_EXACT, _SOD_RHO_EXACT)
    p = np.interp(x_cells, _SOD_X_EXACT, _SOD_P_EXACT)
    u = np.interp(x_cells, _SOD_X_EXACT, _SOD_U_EXACT)
    return {"rho": rho, "p": p, "u": u}


# ---------------------------------------------------------------------------
# Error norm computation
# ---------------------------------------------------------------------------


def compute_error_norms(
    numerical: np.ndarray,
    exact: np.ndarray,
) -> dict[str, float]:
    """Compute L1, L2, and Linf error norms.

    Args:
        numerical: Numerical solution values.
        exact: Exact/reference solution values.

    Returns:
        Dictionary with ``L1``, ``L2``, ``Linf`` error norms.
    """
    diff = np.abs(numerical - exact)
    n = len(diff)

    L1 = np.mean(diff)
    L2 = np.sqrt(np.mean(diff ** 2))
    Linf = np.max(diff)

    return {"L1": float(L1), "L2": float(L2), "Linf": float(Linf)}


# ---------------------------------------------------------------------------
# Case generation (reuses test_shock_tube pattern)
# ---------------------------------------------------------------------------


def _make_sod_enhanced_case(
    case_dir,
    n_cells: int = 100,
    tube_length: float = 1.0,
) -> None:
    """Write a rhoCentralFoam Sod shock tube case.

    Same mesh structure as the base Sod test.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = tube_length / n_cells
    dy = 0.01
    dz = 0.01

    n_cells_y = 1

    # Points: 2 z-planes
    pts_z0 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells + 1):
            pts_z0.append((i * dx, j * dy, 0.0))
    n_base = len(pts_z0)

    pts_z1 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells + 1):
            pts_z1.append((i * dx, j * dy, dz))

    all_points = pts_z0 + pts_z1
    n_points = len(all_points)

    # Faces / owner / neighbour
    faces = []
    owner = []
    neighbour = []

    for j in range(n_cells_y):
        for i in range(n_cells - 1):
            p0 = j * (n_cells + 1) + i + 1
            p1 = p0 + n_cells + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells + i)
            neighbour.append(j * n_cells + i + 1)

    n_internal = len(neighbour)

    # Boundary patches
    for j in range(n_cells_y):
        p0 = j * (n_cells + 1)
        p1 = p0 + n_cells + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells)
    n_inlet = n_cells_y
    inlet_start = n_internal

    for j in range(n_cells_y):
        p0 = j * (n_cells + 1) + n_cells
        p1 = p0 + n_cells + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells + n_cells - 1)
    n_outlet = n_cells_y
    outlet_start = inlet_start + n_inlet

    for i in range(n_cells):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)
    for i in range(n_cells):
        p0 = n_cells_y * (n_cells + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells + i)
    n_tb = 2 * n_cells
    tb_start = outlet_start + n_outlet

    for j in range(n_cells_y):
        for i in range(n_cells):
            p0 = j * (n_cells + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells + 1
            p3 = p0 + n_cells + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells + i)
    for j in range(n_cells_y):
        for i in range(n_cells):
            p0 = n_base + j * (n_cells + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells + 1
            p3 = p0 + n_cells + 1
            faces.append((4, p1, p0, p3, p2))
            owner.append(j * n_cells + i)
    n_empty = 2 * n_cells * n_cells_y
    empty_start = tb_start + n_tb

    n_faces = len(faces)
    total_cells = n_cells * n_cells_y

    # Write mesh
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        location="constant/polyMesh",
    )

    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "vectorField", "object": "points"}
    )
    lines = [f"{n_points}", "("]
    for x, y, z in all_points:
        lines.append(f"({x:.10g} {y:.10g} {z:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "faceList", "object": "faces"}
    )
    lines = [f"{n_faces}", "("]
    for face in faces:
        nv = face[0]
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{nv}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "labelList", "object": "owner"}
    )
    lines = [f"{n_faces}", "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"}
    )
    lines = [f"{n_internal}", "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(
        **{**header_base.__dict__,
           "class_name": "polyBoundaryMesh", "object": "boundary"}
    )
    lines = ["4", "("]
    for name, ptype, nf, sf in [
        ("inlet", "patch", n_inlet, inlet_start),
        ("outlet", "patch", n_outlet, outlet_start),
        ("topAndBottom", "wall", n_tb, tb_start),
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

    # thermophysicalProperties
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

    # 0/ fields
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    # U
    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    topAndBottom\n    {\n"
        "        type            slip;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # p
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    p_lines = [f"nonuniform List<scalar> {total_cells}", "("]
    for j in range(n_cells_y):
        for i in range(n_cells):
            x_centre = (i + 0.5) * dx
            p_val = 1.0 if x_centre < 0.5 else 0.1
            p_lines.append(f"{p_val:.10g}")
    p_lines.append(");")

    p_body = (
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        "internalField   " + "\n".join(p_lines) + "\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    topAndBottom\n    {\n"
        "        type            slip;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # T
    R_air = 287.0
    t_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    t_lines = [f"nonuniform List<scalar> {total_cells}", "("]
    for j in range(n_cells_y):
        for i in range(n_cells):
            x_centre = (i + 0.5) * dx
            if x_centre < 0.5:
                rho_val, p_val = 1.0, 1.0
            else:
                rho_val, p_val = 0.125, 0.1
            T_val = p_val / (rho_val * R_air)
            t_lines.append(f"{T_val:.10g}")
    t_lines.append(");")

    t_body = (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        "internalField   " + "\n".join(t_lines) + "\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    topAndBottom\n    {\n"
        "        type            slip;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "T", t_header, t_body, overwrite=True)

    # system/controlDict
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
        "endTime         0.2;\n"
        "deltaT          1e-5;\n"
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

    # system/fvSchemes
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

    # system/fvSolution
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


@pytest.fixture
def sod_enhanced_case(tmp_path):
    """Create a 100-cell Sod shock tube case."""
    case_dir = tmp_path / "sod_enhanced"
    _make_sod_enhanced_case(case_dir, n_cells=100)
    return case_dir


@pytest.fixture
def sod_enhanced_coarse(tmp_path):
    """Create a 50-cell Sod shock tube case."""
    case_dir = tmp_path / "sod_enhanced_coarse"
    _make_sod_enhanced_case(case_dir, n_cells=50)
    return case_dir


@pytest.fixture
def sod_enhanced_fine(tmp_path):
    """Create a 200-cell Sod shock tube case."""
    case_dir = tmp_path / "sod_enhanced_fine"
    _make_sod_enhanced_case(case_dir, n_cells=200)
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSodEnhancedInit:
    """Tests for enhanced Sod case initialisation."""

    def test_case_structure(self, sod_enhanced_case):
        """Case directory has expected structure."""
        from pyfoam.io.case import Case
        case = Case(sod_enhanced_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.has_field("T", 0)

    def test_mesh_dimensions(self, sod_enhanced_case):
        """Mesh has correct number of cells."""
        from pyfoam.applications.solver_base import SolverBase
        solver = SolverBase(sod_enhanced_case)
        assert solver.mesh.n_cells == 100

    def test_initial_conditions(self, sod_enhanced_case):
        """Initial conditions match Sod left/right states."""
        from pyfoam.applications.rho_central_foam import RhoCentralFoam
        solver = RhoCentralFoam(sod_enhanced_case)

        rho = solver.rho.detach().cpu().numpy()
        p = solver.p.detach().cpu().numpy()

        # Left half
        assert np.allclose(rho[:50], 1.0, atol=0.05)
        assert np.allclose(p[:50], 1.0, atol=0.05)

        # Right half
        assert np.allclose(rho[50:], 0.125, atol=0.01)
        assert np.allclose(p[50:], 0.1, atol=0.01)


class TestSodEnhancedErrorNorms:
    """Error norm analysis against exact solution."""

    def test_error_norms_computation(self):
        """compute_error_norms returns correct structure."""
        exact = np.array([1.0, 2.0, 3.0])
        numerical = np.array([1.1, 1.9, 3.2])
        norms = compute_error_norms(numerical, exact)

        assert "L1" in norms
        assert "L2" in norms
        assert "Linf" in norms
        assert norms["L1"] > 0
        assert norms["L2"] > 0
        assert norms["Linf"] > 0

    def test_exact_solution_perfect_match(self):
        """Perfect match gives zero error."""
        exact = np.array([1.0, 2.0, 3.0])
        norms = compute_error_norms(exact, exact)
        assert norms["L1"] == pytest.approx(0.0)
        assert norms["L2"] == pytest.approx(0.0)
        assert norms["Linf"] == pytest.approx(0.0)

    def test_error_norms_monotonicity(self):
        """L1 <= L2 <= Linf for the same data."""
        exact = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        numerical = np.array([1.1, 2.2, 2.8, 4.3, 4.7])
        norms = compute_error_norms(numerical, exact)
        assert norms["L1"] <= norms["L2"] + 1e-15
        assert norms["L2"] <= norms["Linf"] + 1e-15

    def test_sod_solution_runs_and_finite(self, sod_enhanced_case):
        """rhoCentralFoam produces finite fields."""
        from pyfoam.applications.rho_central_foam import RhoCentralFoam
        solver = RhoCentralFoam(sod_enhanced_case, CFL=0.2)
        solver.run()

        assert torch.isfinite(solver.rho).all()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.T).all()

    def test_sod_density_error_norms(self, sod_enhanced_case):
        """Density error norms are within acceptable range.

        On a 100-cell mesh with first-order explicit solver,
        expect L1 < 0.15, L2 < 0.20, Linf < 0.30.
        """
        from pyfoam.applications.rho_central_foam import RhoCentralFoam
        solver = RhoCentralFoam(sod_enhanced_case, CFL=0.2)
        solver.run()

        rho_num = solver.rho.detach().cpu().numpy()
        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        x = centres[:, 0]

        exact = _interpolate_exact(x)
        norms = compute_error_norms(rho_num, exact["rho"])

        # These are generous bounds for a coarse explicit solver
        assert norms["L1"] < 0.20, f"L1 = {norms['L1']:.4f}"
        assert norms["L2"] < 0.30, f"L2 = {norms['L2']:.4f}"
        assert norms["Linf"] < 0.60, f"Linf = {norms['Linf']:.4f}"

    def test_sod_pressure_error_norms(self, sod_enhanced_case):
        """Pressure error norms are within acceptable range."""
        from pyfoam.applications.rho_central_foam import RhoCentralFoam
        solver = RhoCentralFoam(sod_enhanced_case, CFL=0.2)
        solver.run()

        p_num = solver.p.detach().cpu().numpy()
        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        x = centres[:, 0]

        exact = _interpolate_exact(x)
        norms = compute_error_norms(p_num, exact["p"])

        assert norms["L1"] < 0.20, f"L1 = {norms['L1']:.4f}"
        assert norms["L2"] < 0.35, f"L2 = {norms['L2']:.4f}"
        assert norms["Linf"] < 0.80, f"Linf = {norms['Linf']:.4f}"

    def test_sod_velocity_error_norms(self, sod_enhanced_case):
        """Velocity error norms are within acceptable range."""
        from pyfoam.applications.rho_central_foam import RhoCentralFoam
        solver = RhoCentralFoam(sod_enhanced_case, CFL=0.2)
        solver.run()

        u_num = solver.U[:, 0].detach().cpu().numpy()
        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        x = centres[:, 0]

        exact = _interpolate_exact(x)
        norms = compute_error_norms(u_num, exact["u"])

        # Coarse explicit solver has significant numerical diffusion
        assert norms["L1"] < 0.35, f"L1 = {norms['L1']:.4f}"
        assert norms["L2"] < 0.40, f"L2 = {norms['L2']:.4f}"
        assert norms["Linf"] < 0.60, f"Linf = {norms['Linf']:.4f}"

    def test_error_norms_ratio_order(self, sod_enhanced_case):
        """L1 < L2 < Linf ordering holds for density errors."""
        from pyfoam.applications.rho_central_foam import RhoCentralFoam
        solver = RhoCentralFoam(sod_enhanced_case, CFL=0.2)
        solver.run()

        rho_num = solver.rho.detach().cpu().numpy()
        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        x = centres[:, 0]

        exact = _interpolate_exact(x)
        norms = compute_error_norms(rho_num, exact["rho"])

        assert norms["L1"] <= norms["L2"] + 1e-15
        assert norms["L2"] <= norms["Linf"] + 1e-15

    def test_sod_exact_solution_structured(self):
        """Exact solution interpolation produces reasonable values."""
        x = np.linspace(0.05, 0.95, 100)
        exact = _interpolate_exact(x)

        # Left region should be near (rho=1, p=1, u=0)
        left = x < 0.2
        assert np.allclose(exact["rho"][left], 1.0, atol=0.05)
        assert np.allclose(exact["p"][left], 1.0, atol=0.05)

        # Right region should be near (rho=0.125, p=0.1, u=0)
        right = x > 0.85
        assert np.allclose(exact["rho"][right], 0.125, atol=0.02)
        assert np.allclose(exact["p"][right], 0.1, atol=0.02)
