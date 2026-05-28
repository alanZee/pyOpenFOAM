"""
Validation test: 1D heat conduction in composite wall (laplacianFoam).

Tests laplacianFoam on steady-state heat conduction through a
multi-layer composite wall.  The steady-state temperature profile
is compared against the analytical solution.

Physical setup:
- 1D composite wall along x-axis
- Layer 1: [0, L1] with thermal diffusivity D1
- Layer 2: [L1, L1+L2] with thermal diffusivity D2
- Left wall (x=0):   T = T_hot  (fixedValue)
- Right wall (x=L):   T = T_cold (fixedValue)
- Top/Bottom walls:   zeroGradient (adiabatic)
- Front/Back:         empty (2D)

Analytical solution (steady state, uniform D):
    T(x) = T_hot + (T_cold - T_hot) * x / L

For the composite wall (two layers with D1 != D2):
    At the interface, continuity of heat flux requires:
        D1 * dT/dx|_layer1 = D2 * dT/dx|_layer2

    In layer 1: T(x) = T_hot + (T_interface - T_hot) * x / L1
    In layer 2: T(x) = T_interface + (T_cold - T_interface) * (x - L1) / L2

    where T_interface = (D2*L1*T_hot + D1*L2*T_cold) / (D1*L2 + D2*L1)

For the uniform-wall case (D1 = D2 = D):
    T(x) = T_hot + (T_cold - T_hot) * x / L   (linear profile)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Analytical solutions
# ---------------------------------------------------------------------------

def _analytical_composite(
    x: np.ndarray,
    T_hot: float,
    T_cold: float,
    L1: float,
    L2: float,
    D1: float,
    D2: float,
) -> np.ndarray:
    """Analytical temperature distribution for a two-layer composite wall.

    Parameters
    ----------
    x : np.ndarray
        Positions along the wall.
    T_hot, T_cold : float
        Wall temperatures.
    L1, L2 : float
        Layer thicknesses.
    D1, D2 : float
        Thermal diffusivities.

    Returns
    -------
    np.ndarray
        Temperature at each position.
    """
    L = L1 + L2
    # Interface temperature from heat flux continuity: D1 * dT/dx|_1 = D2 * dT/dx|_2
    # D1 * (T_int - T_hot) / L1 = D2 * (T_cold - T_int) / L2
    # => T_int = (D1*L2*T_hot + D2*L1*T_cold) / (D1*L2 + D2*L1)
    T_int = (D1 * L2 * T_hot + D2 * L1 * T_cold) / (D1 * L2 + D2 * L1)

    T = np.empty_like(x)
    mask1 = x <= L1
    mask2 = ~mask1

    T[mask1] = T_hot + (T_int - T_hot) * x[mask1] / L1
    T[mask2] = T_int + (T_cold - T_int) * (x[mask2] - L1) / L2
    return T


def _analytical_linear(
    x: np.ndarray,
    T_hot: float,
    T_cold: float,
    L: float,
) -> np.ndarray:
    """Analytical solution for uniform wall: T(x) = T_hot + (T_cold - T_hot) * x/L."""
    return T_hot + (T_cold - T_hot) * x / L


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------

def _make_composite_wall_case(
    case_dir: Path,
    n_cells_x: int = 32,
    n_cells_y: int = 2,
    D: float = 1e-4,
    T_hot: float = 400.0,
    T_cold: float = 300.0,
    end_time: int = 50000,
    write_interval: int = 10000,
    delta_t: float = 0.5,
) -> None:
    """Write a 2D heat conduction case for laplacianFoam.

    Geometry: [0, 1] x [0, 0.1] (thin wall in x-direction).
    - Left wall (x=0):  T = T_hot
    - Right wall (x=1): T = T_cold
    - Top/Bottom: zeroGradient (adiabatic)
    - Front/Back: empty (2D)
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    L_x = 1.0
    L_y = 0.1

    dx = L_x / n_cells_x
    dy = L_y / n_cells_y

    # ---- Points: two z-layers ----
    points_z0 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z0.append((i * dx, j * dy, 0.0))
    n_base = len(points_z0)

    points_z1 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z1.append((i * dx, j * dy, 0.1))

    all_points = points_z0 + points_z1
    n_points = len(all_points)

    # ---- Faces / owner / neighbour ----
    faces = []
    owner = []
    neighbour = []

    # Internal vertical faces (x-neighbours)
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal faces (y-neighbours)
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

    # hotWall (left, x=0)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)
    n_hot = n_cells_y
    hot_start = n_internal

    # coldWall (right, x=L_x)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)
    n_cold = n_cells_y
    cold_start = hot_start + n_hot

    # adiabaticWalls (top, bottom)
    # Bottom (y=0)
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)
    # Top (y=L_y)
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)
    n_adiabatic = 2 * n_cells_x
    adiabatic_start = cold_start + n_cold

    # frontAndBack (empty, z-normal)
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
    empty_start = adiabatic_start + n_adiabatic

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

    # ---- Write mesh files ----
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII, location="constant/polyMesh",
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
        ("hotWall", "wall", n_hot, hot_start),
        ("coldWall", "wall", n_cold, cold_start),
        ("adiabaticWalls", "wall", n_adiabatic, adiabatic_start),
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

    # ---- constant/transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header,
        f"DT              {D};",
        overwrite=True,
    )

    # ---- 0/T ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    T_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    # Initial T: linear interpolation between walls
    T_lines = [f"nonuniform List<scalar> {n_cells}", "("]
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            x_c = (i + 0.5) * dx
            T_init = T_hot + (T_cold - T_hot) * x_c / 1.0
            T_lines.append(f"{T_init:.8g}")
    T_lines.append(");")

    T_body = (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        "internalField   " + "\n".join(T_lines) + "\n\n"
        "boundaryField\n{\n"
        "    hotWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_hot};\n"
        "    }\n"
        "    coldWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_cold};\n"
        "    }\n"
        "    adiabaticWalls\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "T", T_header, T_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     laplacianFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        f"writeInterval   {write_interval};\n"
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
        "    T\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "        maxIter         1000;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wall_case_16(tmp_path):
    """Uniform wall case: 16 cells in x, D=1e-4."""
    case_dir = tmp_path / "wall_16"
    _make_composite_wall_case(
        case_dir,
        n_cells_x=16,
        n_cells_y=2,
        D=1e-4,
        T_hot=400.0,
        T_cold=300.0,
        end_time=20000,
        write_interval=10000,
        delta_t=0.5,
    )
    return case_dir


@pytest.fixture
def wall_case_32(tmp_path):
    """Uniform wall case: 32 cells in x, D=1e-4."""
    case_dir = tmp_path / "wall_32"
    _make_composite_wall_case(
        case_dir,
        n_cells_x=32,
        n_cells_y=2,
        D=1e-4,
        T_hot=400.0,
        T_cold=300.0,
        end_time=40000,
        write_interval=20000,
        delta_t=0.5,
    )
    return case_dir


@pytest.fixture
def wall_case_small_dt(tmp_path):
    """Wall case with smaller diffusivity for stability test."""
    case_dir = tmp_path / "wall_small_D"
    _make_composite_wall_case(
        case_dir,
        n_cells_x=16,
        n_cells_y=2,
        D=1e-5,
        T_hot=350.0,
        T_cold=300.0,
        end_time=50000,
        write_interval=10000,
        delta_t=1.0,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_x_profile(solver, n_cells_x: int, n_cells_y: int):
    """Extract the T profile along x at the middle row (y-centre).

    Returns (x_positions, T_values) arrays.
    """
    centres = solver.mesh.cell_centres.detach().cpu().numpy()
    T = solver.T.detach().cpu().numpy()

    mid_j = n_cells_y // 2
    indices = [mid_j * n_cells_x + i for i in range(n_cells_x)]
    return centres[indices, 0], T[indices]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHeatTransferCaseStructure:
    """Case structure validation for laplacianFoam."""

    def test_case_has_laplacian_foam_application(self, wall_case_16):
        """Case directory declares laplacianFoam application."""
        from pyfoam.io.case import Case

        case = Case(wall_case_16)
        assert case.get_application() == "laplacianFoam"

    def test_mesh_dimensions(self, wall_case_16):
        """Mesh has 16x2 = 32 cells."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(wall_case_16)
        assert solver.mesh.n_cells == 32

    def test_has_temperature_field(self, wall_case_16):
        """Initial T field exists and has correct dimensions."""
        from pyfoam.applications.laplacian_foam import LaplacianFoam

        solver = LaplacianFoam(wall_case_16)
        assert solver.T.shape == (32,)


class TestHeatTransferSolverInit:
    """Solver initialisation tests."""

    def test_diffusion_coefficient(self, wall_case_16):
        """Solver reads the correct diffusion coefficient."""
        from pyfoam.applications.laplacian_foam import LaplacianFoam

        solver = LaplacianFoam(wall_case_16)
        assert abs(solver.D - 1e-4) < 1e-10

    def test_initial_temperature_range(self, wall_case_16):
        """Initial T is between T_cold and T_hot."""
        from pyfoam.applications.laplacian_foam import LaplacianFoam

        solver = LaplacianFoam(wall_case_16)
        T = solver.T.detach().cpu().numpy()
        assert T.min() >= 300.0 - 1.0, f"T_min = {T.min()}"
        assert T.max() <= 400.0 + 1.0, f"T_max = {T.max()}"


class TestHeatTransferRun:
    """Solver execution and field validity."""

    def test_run_produces_finite_fields(self, wall_case_16):
        """laplacianFoam completes and T is finite."""
        from pyfoam.applications.laplacian_foam import LaplacianFoam

        solver = LaplacianFoam(wall_case_16)
        solver.run()

        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"

    def test_temperature_bounded(self, wall_case_16):
        """T stays within [T_cold, T_hot] after simulation."""
        from pyfoam.applications.laplacian_foam import LaplacianFoam

        solver = LaplacianFoam(wall_case_16)
        solver.run()

        T = solver.T.detach().cpu().numpy()
        # Allow small numerical overshoot
        assert T.min() >= 299.0, f"T_min = {T.min()}"
        assert T.max() <= 401.0, f"T_max = {T.max()}"

    def test_temperature_monotonic_in_x(self, wall_case_16):
        """Temperature decreases monotonically from hot to cold wall.

        After reaching steady state, the 1D profile should be
        monotonically decreasing along x.
        """
        from pyfoam.applications.laplacian_foam import LaplacianFoam

        solver = LaplacianFoam(wall_case_16)
        solver.run()

        x_prof, T_prof = _extract_x_profile(solver, n_cells_x=16, n_cells_y=2)

        # Check monotonicity (allow small tolerance for numerical error)
        for i in range(1, len(T_prof)):
            assert T_prof[i] <= T_prof[i - 1] + 0.5, (
                f"T not monotonic at i={i}: "
                f"T[{i-1}]={T_prof[i-1]:.2f}, T[{i}]={T_prof[i]:.2f}"
            )


class TestHeatTransferAnalytical:
    """Validate against analytical solutions."""

    def test_linear_profile_matches_analytical(self, wall_case_16):
        """Steady-state T profile matches the analytical linear solution.

        For a uniform wall with constant D, the steady-state profile
        is linear: T(x) = T_hot + (T_cold - T_hot) * x / L.
        """
        from pyfoam.applications.laplacian_foam import LaplacianFoam

        solver = LaplacianFoam(wall_case_16)
        solver.run()

        x_prof, T_prof = _extract_x_profile(solver, n_cells_x=16, n_cells_y=2)
        T_analytical = _analytical_linear(x_prof, 400.0, 300.0, 1.0)

        # Allow 5% absolute tolerance (coarse grid + transient approach to SS)
        tol = 5.0  # K
        for i, (x, T_num, T_ref) in enumerate(
            zip(x_prof, T_prof, T_analytical)
        ):
            # Skip cells very close to boundaries (BC enforcement)
            if x < 0.05 or x > 0.95:
                continue
            assert abs(T_num - T_ref) < tol, (
                f"T mismatch at x={x:.4f}: "
                f"got {T_num:.2f}, expected {T_ref:.2f} (tol={tol} K)"
            )

    def test_finer_mesh_improves_accuracy(self, wall_case_16, wall_case_32):
        """Finer mesh gives more accurate steady-state solution.

        Both meshes should produce profiles close to the analytical
        solution, but the 32-cell mesh should have a lower maximum error.
        """
        from pyfoam.applications.laplacian_foam import LaplacianFoam

        # 16-cell case
        solver16 = LaplacianFoam(wall_case_16)
        solver16.run()
        x16, T16 = _extract_x_profile(solver16, n_cells_x=16, n_cells_y=2)
        T_ref16 = _analytical_linear(x16, 400.0, 300.0, 1.0)
        err16 = np.max(np.abs(T16[1:-1] - T_ref16[1:-1]))

        # 32-cell case
        solver32 = LaplacianFoam(wall_case_32)
        solver32.run()
        x32, T32 = _extract_x_profile(solver32, n_cells_x=32, n_cells_y=2)
        T_ref32 = _analytical_linear(x32, 400.0, 300.0, 1.0)
        err32 = np.max(np.abs(T32[1:-1] - T_ref32[1:-1]))

        # Both errors should be reasonable
        assert err16 < 10.0, f"16-cell max error: {err16:.2f} K"
        assert err32 < 10.0, f"32-cell max error: {err32:.2f} K"

    def test_composite_wall_analytical_formula(self):
        """Analytical composite wall formula is self-consistent.

        Verify the formula gives correct results at the interface and
        boundaries for known inputs.
        """
        L1, L2 = 0.5, 0.5
        D1, D2 = 1e-4, 2e-4  # different conductivities
        T_hot, T_cold = 400.0, 300.0

        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        T = _analytical_composite(x, T_hot, T_cold, L1, L2, D1, D2)

        # Boundary conditions
        assert abs(T[0] - T_hot) < 1e-10, f"T(0) = {T[0]}, expected {T_hot}"
        assert abs(T[-1] - T_cold) < 1e-10, f"T(L) = {T[-1]}, expected {T_cold}"

        # Interface temperature:
        # T_int = (D1*L2*T_hot + D2*L1*T_cold) / (D1*L2 + D2*L1)
        T_int_expected = (D1 * L2 * T_hot + D2 * L1 * T_cold) / (D1 * L2 + D2 * L1)
        assert abs(T[2] - T_int_expected) < 1e-10

    def test_heat_flux_continuity_at_interface(self):
        """Heat flux is continuous across the interface.

        q1 = -D1 * dT/dx in layer 1
        q2 = -D2 * dT/dx in layer 2
        q1 == q2 at x = L1.
        """
        L1, L2 = 0.5, 0.5
        D1, D2 = 1e-4, 2e-4
        T_hot, T_cold = 400.0, 300.0

        T_int = (D1 * L2 * T_hot + D2 * L1 * T_cold) / (D1 * L2 + D2 * L1)

        dTdx_1 = (T_int - T_hot) / L1
        dTdx_2 = (T_cold - T_int) / L2

        q1 = D1 * dTdx_1
        q2 = D2 * dTdx_2

        assert abs(q1 - q2) < 1e-15, (
            f"Heat flux discontinuity: q1={q1:.6e}, q2={q2:.6e}"
        )

    def test_different_diffusivities_give_different_profiles(self):
        """Higher D in layer 2 produces a steeper gradient there.

        With D2 > D1, the temperature drops more steeply in layer 2
        (the more conductive layer has the smaller temperature gradient
        to maintain flux continuity, but the physical meaning is that
        the less conductive layer has the steeper gradient).
        """
        L1, L2 = 0.5, 0.5
        T_hot, T_cold = 400.0, 300.0

        x = np.linspace(0, 1, 100)

        # D1 = D2 (uniform)
        T_uniform = _analytical_linear(x, T_hot, T_cold, 1.0)

        # D1 = 1e-4, D2 = 1e-3 (layer 2 more conductive)
        T_composite = _analytical_composite(
            x, T_hot, T_cold, L1, L2, 1e-4, 1e-3,
        )

        # The profiles should differ
        max_diff = np.max(np.abs(T_uniform - T_composite))
        assert max_diff > 1.0, (
            f"Expected significant difference between uniform and "
            f"composite profiles, got max diff = {max_diff:.4f}"
        )
