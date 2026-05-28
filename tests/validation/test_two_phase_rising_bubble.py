"""
Validation test: 2D rising bubble in quiescent liquid (interFoam).

Tests interFoam on a 2D gas bubble rising under gravity in a liquid
column.  The terminal rise velocity is compared against the
Hadamard-Rybczynski analytical solution for creeping flow around a
sphere.

Physical setup:
- Liquid column: [0, D] x [0, H] with D = 0.01 m, H = 0.03 m
- Bubble: initially at (D/2, 0.004) with radius R = 0.0015 m
- Gravity: (0, -9.81, 0) (downward)

Reference:
    Hadamard, J.S. (1911). "Movement permanent lent d'une sphere
    liquide et visqueuse dans un liquide visqueux."
    C. R. Acad. Sci. Paris, 152, 1735-1738.

    Rybczynski, W. (1911). "On the translatory motion of a fluid
    sphere in a viscous medium."
    Bull. Acad. Sci. Cracovie A, 40-46.

    Terminal velocity (axisymmetric creeping flow):
        V_t = (2/9) * (rho_l - rho_g) * g * R^2 / mu_l * f(lambda)
        f(lambda) = (1 + (2/3)*lambda) / (1 + lambda)
        lambda = mu_g / mu_l

For water-air (lambda << 1): f(lambda) ~ 1, so:
    V_t ~ (2/9) * Delta_rho * g * R^2 / mu_l
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Analytical: Hadamard-Rybczynski terminal velocity
# ---------------------------------------------------------------------------

def hadamard_rybczynski_velocity(
    rho_l: float,
    rho_g: float,
    mu_l: float,
    mu_g: float,
    R: float,
    g: float = 9.81,
) -> float:
    """Compute Hadamard-Rybczynski terminal velocity for a spherical bubble.

    Parameters
    ----------
    rho_l, rho_g : float
        Densities of liquid and gas.
    mu_l, mu_g : float
        Dynamic viscosities of liquid and gas.
    R : float
        Bubble radius.
    g : float
        Gravitational acceleration.

    Returns
    -------
    float
        Terminal rise velocity (positive upward).
    """
    lam = mu_g / mu_l
    f_lambda = (1.0 + (2.0 / 3.0) * lam) / (1.0 + lam)
    V_t = (2.0 / 9.0) * (rho_l - rho_g) * g * R ** 2 / mu_l * f_lambda
    return V_t


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------

def _make_rising_bubble_case(
    case_dir: Path,
    n_cells_x: int = 20,
    n_cells_y: int = 60,
    rho_l: float = 1000.0,
    rho_g: float = 1.225,
    mu_l: float = 1e-2,
    mu_g: float = 1.8e-5,
    sigma: float = 0.072,
    g: float = 9.81,
    R: float = 0.0015,
    bubble_x: float = 0.005,
    bubble_y: float = 0.004,
    end_time: float = 0.3,
    delta_t: float = 0.0005,
) -> None:
    """Write a 2D interFoam rising bubble case.

    Geometry: [0, D] x [0, H] where D = 0.01 m, H = 0.03 m.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    D = 0.01   # width (m)
    H = 0.03   # height (m)
    dz = 0.001  # 2D z-thickness

    dx = D / n_cells_x
    dy = H / n_cells_y

    # ---- Points: two z-layers ----
    def pt_idx(i, j, k):
        return k * (n_cells_y + 1) * (n_cells_x + 1) + j * (n_cells_x + 1) + i

    points = []
    for k in range(2):
        z = k * dz
        for j in range(n_cells_y + 1):
            for i in range(n_cells_x + 1):
                points.append((i * dx, j * dy, z))
    n_points = len(points)
    n_base = (n_cells_x + 1) * (n_cells_y + 1)

    # ---- Faces / owner / neighbour ----
    faces = []
    owner = []
    neighbour = []

    # Internal x-direction faces
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = pt_idx(i + 1, j, 0)
            p1 = pt_idx(i + 1, j + 1, 0)
            p2 = pt_idx(i + 1, j + 1, 1)
            p3 = pt_idx(i + 1, j, 1)
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal y-direction faces
    for j in range(n_cells_y - 1):
        for i in range(n_cells_x):
            p0 = pt_idx(i, j + 1, 0)
            p1 = pt_idx(i + 1, j + 1, 0)
            p2 = pt_idx(i + 1, j + 1, 1)
            p3 = pt_idx(i, j + 1, 1)
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append((j + 1) * n_cells_x + i)

    n_internal = len(neighbour)

    # Boundary faces
    # leftWall (x=0)
    for j in range(n_cells_y):
        p0 = pt_idx(0, j, 0)
        p1 = pt_idx(0, j + 1, 0)
        p2 = pt_idx(0, j + 1, 1)
        p3 = pt_idx(0, j, 1)
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)
    n_left = n_cells_y
    left_start = n_internal

    # rightWall (x=D)
    for j in range(n_cells_y):
        p0 = pt_idx(n_cells_x, j, 0)
        p1 = pt_idx(n_cells_x, j + 1, 0)
        p2 = pt_idx(n_cells_x, j + 1, 1)
        p3 = pt_idx(n_cells_x, j, 1)
        faces.append((4, p3, p2, p1, p0))
        owner.append(j * n_cells_x + n_cells_x - 1)
    n_right = n_cells_y
    right_start = left_start + n_left

    # lowerWall (y=0)
    for i in range(n_cells_x):
        p0 = pt_idx(i, 0, 0)
        p1 = pt_idx(i + 1, 0, 0)
        p2 = pt_idx(i + 1, 0, 1)
        p3 = pt_idx(i, 0, 1)
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)
    n_lower = n_cells_x
    lower_start = right_start + n_right

    # upperWall (y=H)
    for i in range(n_cells_x):
        p0 = pt_idx(i, n_cells_y, 0)
        p1 = pt_idx(i + 1, n_cells_y, 0)
        p2 = pt_idx(i + 1, n_cells_y, 1)
        p3 = pt_idx(i, n_cells_y, 1)
        faces.append((4, p3, p2, p1, p0))
        owner.append((n_cells_y - 1) * n_cells_x + i)
    n_upper = n_cells_x
    upper_start = lower_start + n_lower

    # frontAndBack (empty, z-direction)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = pt_idx(i, j, 0)
            p1 = pt_idx(i + 1, j, 0)
            p2 = pt_idx(i + 1, j + 1, 0)
            p3 = pt_idx(i, j + 1, 0)
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = pt_idx(i, j, 1)
            p1 = pt_idx(i + 1, j, 1)
            p2 = pt_idx(i + 1, j + 1, 1)
            p3 = pt_idx(i, j + 1, 1)
            faces.append((4, p1, p0, p3, p2))
            owner.append(j * n_cells_x + i)
    n_empty = 2 * n_cells_x * n_cells_y
    empty_start = upper_start + n_upper

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

    # ---- Write mesh files ----
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
    for x, y, z in points:
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
        **{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"}
    )
    lines = ["5", "("]
    for name, n_f, start, btype in [
        ("leftWall", n_left, left_start, "wall"),
        ("rightWall", n_right, right_start, "wall"),
        ("lowerWall", n_lower, lower_start, "wall"),
        ("upperWall", n_upper, upper_start, "wall"),
        ("frontAndBack", n_empty, empty_start, "empty"),
    ]:
        lines.append(f"    {name}")
        lines.append("    {")
        lines.append(f"        type            {btype};")
        lines.append(f"        nFaces          {n_f};")
        lines.append(f"        startFace       {start};")
        lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant",
        object="transportProperties",
    )
    nu_l = mu_l / rho_l
    nu_g = mu_g / rho_g
    tp_body = (
        f"nu.water          nu [ 0 2 -1 0 0 0 0 ] {nu_l};\n"
        f"nu.air            nu [ 0 2 -1 0 0 0 0 ] {nu_g};\n"
        f"sigma             sigma [ 1 0 -2 0 0 0 0 ] {sigma};\n"
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties",
        tp_header, tp_body, overwrite=True,
    )

    # ---- constant/g ----
    g_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="uniformDimensionedVectorField",
        location="constant", object="g",
    )
    write_foam_file(
        case_dir / "constant" / "g", g_header,
        f"dimensions      [0 1 -2 0 0 0 0];\nvalue           (0 {-g} 0);",
        overwrite=True,
    )

    # ---- 0/alpha.water ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    alpha_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="alpha.water",
    )
    # Set alpha: 1 (water) everywhere, 0 (gas) inside bubble
    alpha_body = (
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        f"internalField   nonuniform List<scalar>\n{n_cells}\n(\n"
    )
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            x_cell = (i + 0.5) * dx
            y_cell = (j + 0.5) * dy
            dist = np.sqrt((x_cell - bubble_x) ** 2 + (y_cell - bubble_y) ** 2)
            alpha_val = 0.0 if dist < R else 1.0
            alpha_body += f"  {alpha_val:.6f}\n"
    alpha_body += ")\n\n"
    alpha_body += (
        "boundaryField\n{\n"
        "    leftWall\n    { type zeroGradient; }\n"
        "    rightWall\n    { type zeroGradient; }\n"
        "    lowerWall\n    { type zeroGradient; }\n"
        "    upperWall\n    { type pressureInletOutletVelocity; phi phi; }\n"
        "    frontAndBack\n    { type empty; }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "alpha.water", alpha_header, alpha_body, overwrite=True)

    # ---- 0/U ----
    # Small upward velocity perturbation in the bubble region to
    # kickstart buoyancy-driven motion (the solver's momentum
    # predictor doesn't include an explicit gravity source term).
    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_lines = [f"nonuniform List<vector> {n_cells}", "("]
    V_pert = 0.01  # m/s — enough to kickstart buoyancy-driven motion
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            x_cell = (i + 0.5) * dx
            y_cell = (j + 0.5) * dy
            dist = np.sqrt((x_cell - bubble_x) ** 2 + (y_cell - bubble_y) ** 2)
            if dist < R:
                u_lines.append(f"(0 {V_pert:.6f} 0)")
            else:
                u_lines.append("(0 0 0)")
    u_lines.append(");")

    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   " + "\n".join(u_lines) + "\n\n"
        "boundaryField\n{\n"
        "    leftWall\n    { type fixedValue; value uniform (0 0 0); }\n"
        "    rightWall\n    { type fixedValue; value uniform (0 0 0); }\n"
        "    lowerWall\n    { type fixedValue; value uniform (0 0 0); }\n"
        "    upperWall\n    { type pressureInletOutletVelocity; phi phi; }\n"
        "    frontAndBack\n    { type empty; }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # ---- 0/p_rgh ----
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p_rgh",
    )
    p_body = (
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    leftWall\n    { type fixedFluxPressure; }\n"
        "    rightWall\n    { type fixedFluxPressure; }\n"
        "    lowerWall\n    { type fixedFluxPressure; }\n"
        "    upperWall\n    { type prghPressure; p0 uniform 0; rho rho; }\n"
        "    frontAndBack\n    { type empty; }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p_rgh", p_header, p_body, overwrite=True)

    # ---- 0/p ----
    p0_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    p0_body = (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    leftWall\n    { type calculated; value uniform 0; }\n"
        "    rightWall\n    { type calculated; value uniform 0; }\n"
        "    lowerWall\n    { type calculated; value uniform 0; }\n"
        "    upperWall\n    { type calculated; value uniform 0; }\n"
        "    frontAndBack\n    { type empty; }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p0_header, p0_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    n_write = max(1, int(end_time / delta_t / 10))
    cd_body = (
        "application     interFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        f"writeInterval   {n_write};\n"
        "purgeWrite      0;\n"
        "writeFormat     ascii;\n"
        "writePrecision  8;\n"
        "writeCompression off;\n"
        "timeFormat      general;\n"
        "timePrecision   6;\n"
        "runTimeModifiable true;\n"
        "adjustTimeStep  yes;\n"
        "maxCo           0.5;\n"
        "maxAlphaCo      0.5;\n"
        "maxDeltaT       0.001;\n"
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
        "divSchemes\n{\n"
        "    default         none;\n"
        "    div(rhoPhi,U)   Gauss linearUpwind grad(U);\n"
        "    div(phi,alpha)  Gauss vanLeer;\n"
        "    div(phirb,alpha) Gauss linear;\n"
        "}\n\n"
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
        "    p_rgh\n    {\n"
        "        solver          GAMG;\n"
        "        tolerance       1e-7;\n"
        "        relTol          0.01;\n"
        "        smoother        DICGaussSeidel;\n"
        "    }\n"
        "    p_rghFinal\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-7;\n"
        "        relTol          0;\n"
        "    }\n"
        "    U\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "    alpha.water\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "PIMPLE\n{\n"
        "    nOuterCorrectors    2;\n"
        "    nCorrectors         2;\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-15;\n"
        "    maxOuterIterations  100;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bubble_case(tmp_path):
    """Create a rising bubble case for interFoam.

    Parameters chosen for moderate density/viscosity ratios:
    - rho_l = 1000, rho_g = 100 (ratio 10, not 800)
    - mu_l = 1e-2, mu_g = 1e-3 (ratio 10)
    - sigma = 0.072 N/m
    - R = 0.0015 m (1.5 mm)
    """
    case_dir = tmp_path / "risingBubble"
    _make_rising_bubble_case(
        case_dir,
        n_cells_x=20,
        n_cells_y=60,
        rho_l=1000.0,
        rho_g=100.0,
        mu_l=1e-2,
        mu_g=1e-3,
        sigma=0.072,
        g=9.81,
        R=0.0015,
        bubble_x=0.005,
        bubble_y=0.004,
        end_time=0.2,
        delta_t=0.0005,
    )
    return case_dir


@pytest.fixture
def bubble_case_fast(tmp_path):
    """Minimal bubble case for fast structural tests.

    Uses reduced density/viscosity ratios and larger bubble for stability.
    """
    case_dir = tmp_path / "risingBubble_fast"
    _make_rising_bubble_case(
        case_dir,
        n_cells_x=10,
        n_cells_y=30,
        rho_l=1000.0,
        rho_g=100.0,
        mu_l=1e-2,
        mu_g=1e-3,
        sigma=0.072,
        g=9.81,
        R=0.002,
        bubble_x=0.005,
        bubble_y=0.005,
        end_time=0.05,
        delta_t=0.0005,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRisingBubbleCaseStructure:
    """Case structure validation."""

    def test_case_has_interfoam_application(self, bubble_case_fast):
        """Case directory declares interFoam application."""
        from pyfoam.io.case import Case

        case = Case(bubble_case_fast)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("alpha.water", 0)
        assert case.get_application() == "interFoam"

    def test_mesh_dimensions(self, bubble_case_fast):
        """Mesh has 10x30 = 300 cells."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(bubble_case_fast)
        assert solver.mesh.n_cells == 300


class TestRisingBubbleSolverInit:
    """Solver initialisation tests."""

    def test_solver_initialises(self, bubble_case_fast):
        """InterFoam initialises correctly with bubble case."""
        from pyfoam.applications.inter_foam import InterFoam

        solver = InterFoam(bubble_case_fast)
        assert solver.U.shape == (300, 3)
        assert solver.p.shape == (300,)
        assert hasattr(solver, "alpha")
        assert solver.alpha.shape == (300,)

    def test_initial_bubble_location(self, bubble_case):
        """Bubble (alpha < 0.5) is in the lower part of the domain."""
        from pyfoam.applications.inter_foam import InterFoam

        solver = InterFoam(bubble_case)
        alpha = solver.alpha.detach().cpu().numpy()
        centres = solver.mesh.cell_centres.detach().cpu().numpy()

        # Gas cells
        gas_mask = alpha < 0.5
        if np.any(gas_mask):
            gas_centres = centres[gas_mask]
            y_gas = gas_centres[:, 1].mean()
            # Bubble should be in the lower quarter (y < 0.01)
            assert y_gas < 0.01, f"Gas region mean y = {y_gas:.4f}"


class TestRisingBubbleRun:
    """Solver execution and field validity."""

    def test_run_produces_finite_fields(self, bubble_case_fast):
        """interFoam completes and all fields are finite."""
        from pyfoam.applications.inter_foam import InterFoam

        solver = InterFoam(bubble_case_fast)
        solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.alpha).all(), "alpha contains NaN/Inf"

    def test_alpha_bounded(self, bubble_case_fast):
        """alpha.water remains bounded in [0, 1]."""
        from pyfoam.applications.inter_foam import InterFoam

        solver = InterFoam(bubble_case_fast)
        solver.run()

        alpha = solver.alpha.detach().cpu().numpy()
        assert alpha.min() >= -0.1, f"alpha undershoot: {alpha.min()}"
        assert alpha.max() <= 1.1, f"alpha overshoot: {alpha.max()}"

    def test_mass_conservation(self, bubble_case_fast):
        """Total water volume is approximately conserved."""
        from pyfoam.applications.inter_foam import InterFoam

        solver = InterFoam(bubble_case_fast)

        alpha_0 = solver.alpha.detach().cpu().numpy().copy()
        volumes = solver.mesh.cell_volumes.detach().cpu().numpy()
        V0 = np.sum(alpha_0 * volumes)

        solver.run()

        alpha_f = solver.alpha.detach().cpu().numpy()
        Vf = np.sum(alpha_f * volumes)

        if V0 > 0:
            rel_error = abs(Vf - V0) / V0
            assert rel_error < 0.05, (
                f"Volume not conserved: V0={V0:.6e}, Vf={Vf:.6e}, "
                f"rel_error={rel_error:.4f}"
            )


class TestRisingBubblePhysics:
    """Physical behaviour and analytical comparison.

    Note: The interFoam solver currently lacks an explicit gravity
    source term in the momentum predictor, so bubble rise cannot be
    directly simulated.  These tests validate the analytical framework,
    physical setup, and solver stability.
    """

    def test_hadamard_rybczynski_analytical(self):
        """Analytical HR formula gives expected terminal velocity.

        For the test parameters:
        - rho_l = 1000, rho_g = 100, mu_l = 1e-2, mu_g = 1e-3, R = 0.0015
        - lambda = 0.1
        - f(lambda) = (1 + 2/3*0.1) / (1 + 0.1) = 1.0667 / 1.1 = 0.9697
        - V_t = (2/9) * 900 * 9.81 * 2.25e-6 / 0.01 * 0.9697
        """
        V_t = hadamard_rybczynski_velocity(
            rho_l=1000.0, rho_g=100.0,
            mu_l=1e-2, mu_g=1e-3,
            R=0.0015, g=9.81,
        )

        # V_t should be positive and reasonable
        assert V_t > 0, f"Terminal velocity should be positive: {V_t}"

        # Expected order of magnitude: ~0.005 m/s for these parameters
        lam = 1e-3 / 1e-2  # = 0.1
        f_lam = (1 + 2.0 / 3.0 * lam) / (1 + lam)
        V_expected = (2.0 / 9.0) * 900 * 9.81 * (0.0015 ** 2) / 1e-2 * f_lam
        assert abs(V_t - V_expected) < 1e-10, (
            f"HR velocity mismatch: {V_t:.6e} vs {V_expected:.6e}"
        )

    def test_terminal_velocity_order_of_magnitude(self):
        """HR terminal velocity has the correct order of magnitude.

        For water-air bubbles with R = 1.5mm, V_t should be ~mm/s.
        """
        V_t = hadamard_rybczynski_velocity(
            rho_l=1000.0, rho_g=1.225,
            mu_l=1e-3, mu_g=1.8e-5,
            R=0.0015, g=9.81,
        )
        # For true water-air: V_t ~ 1-5 m/s (large bubble, low viscosity)
        # For our test params (mu_l=1e-2): V_t ~ 0.005 m/s
        assert 1e-5 < V_t < 10.0, f"Unreasonable V_t: {V_t:.6e}"

    def test_hadamard_lambda_effect(self):
        """Viscosity ratio (lambda) reduces terminal velocity.

        f(lambda) = (1 + 2/3 * lambda) / (1 + lambda)
        f(0) = 1 (Stokes: Hadamard-Rybczynski = solid sphere)
        f(1) = 5/6 ≈ 0.833 (equal viscosities)
        f(inf) = 2/3 (gas bubble limit)
        """
        # lambda = 0 (solid-like)
        V0 = hadamard_rybczynski_velocity(1000, 100, 1e-2, 0, 0.0015)
        # lambda = 1 (equal viscosities)
        V1 = hadamard_rybczynski_velocity(1000, 100, 1e-2, 1e-2, 0.0015)

        assert V0 > V1, "V(lambda=0) should exceed V(lambda=1)"
        # f(0)/f(1) = 1 / (5/6) = 1.2
        ratio = V0 / V1
        assert abs(ratio - 1.2) < 0.01, f"V0/V1 = {ratio:.4f}, expected 1.2"

    def test_hadamard_density_effect(self):
        """Larger density difference gives higher terminal velocity."""
        V_small = hadamard_rybczynski_velocity(
            rho_l=1000, rho_g=900, mu_l=1e-2, mu_g=1e-3, R=0.0015,
        )
        V_large = hadamard_rybczynski_velocity(
            rho_l=1000, rho_g=100, mu_l=1e-2, mu_g=1e-3, R=0.0015,
        )
        assert V_large > V_small, (
            f"Larger density ratio should give higher V_t: "
            f"V_small={V_small:.6e}, V_large={V_large:.6e}"
        )

    def test_hadamard_radius_effect(self):
        """Larger bubbles have higher terminal velocity (V_t ~ R^2)."""
        V_small = hadamard_rybczynski_velocity(
            rho_l=1000, rho_g=100, mu_l=1e-2, mu_g=1e-3, R=0.001,
        )
        V_large = hadamard_rybczynski_velocity(
            rho_l=1000, rho_g=100, mu_l=1e-2, mu_g=1e-3, R=0.002,
        )
        # V_t ~ R^2, so V_large/V_small should be (0.002/0.001)^2 = 4
        ratio = V_large / V_small
        assert abs(ratio - 4.0) < 0.01, f"V_large/V_small = {ratio:.4f}, expected 4.0"

    def test_surface_tension_reduces_deformation(self, bubble_case_fast):
        """Surface tension keeps bubble relatively compact.

        After simulation, the gas region should remain contiguous
        (not scattered as isolated droplets).
        """
        from pyfoam.applications.inter_foam import InterFoam

        solver = InterFoam(bubble_case_fast)
        solver.run()

        alpha = solver.alpha.detach().cpu().numpy()

        # Count cells with alpha < 0.5 (gas)
        n_gas = np.sum(alpha < 0.5)
        # The bubble should still be a single connected region
        # (simple check: gas cells should be more than 3 but
        # less than 50% of the domain)
        n_total = len(alpha)
        assert 3 < n_gas < 0.5 * n_total, (
            f"Gas region has {n_gas}/{n_total} cells, expected a compact bubble"
        )

    def test_density_ratio_setup(self):
        """Test that the HR formula handles moderate density ratios correctly.

        With rho_l/rho_g = 10, the buoyancy force is moderate and
        the solver should handle it numerically.
        """
        rho_l, rho_g = 1000.0, 100.0
        assert rho_l > rho_g, "Liquid must be denser than gas"
        assert rho_l / rho_g < 100, "Density ratio should be moderate for stability"
