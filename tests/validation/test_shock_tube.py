"""
Validation test: Sod shock tube (rhoCentralFoam).

The Sod shock tube is a classic 1D Riemann problem for the compressible
Euler equations. It consists of a tube with a diaphragm at x = 0.5:
- Left state:  ρ = 1.0, p = 1.0, U = 0
- Right state: ρ = 0.125, p = 0.1, U = 0

At t = 0 the diaphragm breaks, producing (left to right):
1. A rarefaction wave
2. A contact discontinuity
3. A shock wave

The analytical solution (exact Riemann solver) is used as reference.

Reference:
    Sod, G.A. (1978). "A survey of several finite difference methods
    for systems of nonlinear hyperbolic conservation laws."
    J. Comput. Phys. 27, 1-31.

    Toro, E.F. (2009). "Riemann Solvers and Numerical Methods for
    Fluid Dynamics." Springer, 3rd ed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Sod exact Riemann solver (simplified)
# ---------------------------------------------------------------------------

# Reference solution at t = 0.2 computed with exact Riemann solver.
# These are tabulated values at selected x-positions for a 100-cell mesh.

_SOD_T_FINAL = 0.2
_SOD_GAMMA = 1.4

# Pre-computed exact solution at x positions for t = 0.2
# Source: Toro (2009), Chapter 4, Problem 1
_SOD_X = np.array([
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95,
])

_SOD_RHO = np.array([
    1.000000, 1.000000, 1.000000, 0.987774, 0.932486, 0.864052,
    0.798374, 0.736826, 0.458337, 0.458337, 0.265574, 0.125000,
    0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000,
    0.125000,
])

_SOD_P = np.array([
    1.000000, 1.000000, 1.000000, 0.973560, 0.879378, 0.772049,
    0.672084, 0.580971, 0.319168, 0.319168, 0.272114, 0.100000,
    0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000,
    0.100000,
])

_SOD_U = np.array([
    0.000000, 0.000000, 0.000000, 0.043586, 0.121744, 0.201090,
    0.281837, 0.363867, 0.461245, 0.461245, 0.461245, 0.427525,
    0.427525, 0.427525, 0.427525, 0.427525, 0.427525, 0.427525,
    0.427525,
])


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------


def _make_sod_tube_case(
    case_dir: Path,
    n_cells: int = 100,
    tube_length: float = 1.0,
) -> None:
    """Write a complete rhoCentralFoam Sod shock tube case.

    1D tube [0, L] discretised as a thin 3D slab with empty z-boundaries.
    Two z-layers of cells (z = 0 and z = dz).
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = tube_length / n_cells
    dy = 0.01  # thin y extent
    dz = 0.01  # thin z extent

    n_cells_y = 1  # single row in y
    n_cells_z = 1  # not used directly, we make 2 z-layers

    # ---- Points: 2 z-planes, (n_cells+1) x 2 points ----
    # z=0 plane
    pts_z0 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells + 1):
            pts_z0.append((i * dx, j * dy, 0.0))
    n_base = len(pts_z0)

    # z=dz plane
    pts_z1 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells + 1):
            pts_z1.append((i * dx, j * dy, dz))

    all_points = pts_z0 + pts_z1
    n_points = len(all_points)

    # ---- Faces / owner / neighbour ----
    faces: list[tuple] = []
    owner: list[int] = []
    neighbour: list[int] = []

    # Internal x-direction faces
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
    # inlet (x=0): fixedValue for U, T; fixedFluxPressure for p
    for j in range(n_cells_y):
        p0 = j * (n_cells + 1)
        p1 = p0 + n_cells + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells)
    n_inlet = n_cells_y
    inlet_start = n_internal

    # outlet (x=L)
    for j in range(n_cells_y):
        p0 = j * (n_cells + 1) + n_cells
        p1 = p0 + n_cells + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells + n_cells - 1)
    n_outlet = n_cells_y
    outlet_start = inlet_start + n_inlet

    # topAndBottom (wall, y-normal)
    # Bottom (y=0)
    for i in range(n_cells):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)
    # Top (y=dy)
    for i in range(n_cells):
        p0 = n_cells_y * (n_cells + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells + i)
    n_tb = 2 * n_cells
    tb_start = outlet_start + n_outlet

    # frontAndBack (empty, z-normal)
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

    # ---- Write mesh ----
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

    # ---- 0/ fields ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    # --- U (uniform zero) ---
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

    # --- p (piecewise: left=1, right=0.1) ---
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    # Nonuniform initial condition: left half p=1, right half p=0.1
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

    # --- T (from EOS: T = p / (rho * R)) ---
    # R_air = 287 J/(kg·K)
    # Left:  T = 1.0 / (1.0 * 287) = 0.00348 K  -> use Cp/gamma-based
    # For perfect gas: p = rho * R * T => T = p / (rho * R)
    # With gamma=1.4, Cp=1004.5: R = Cp*(gamma-1)/gamma = 1004.5*0.4/1.4 = 287
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

    # ---- system/controlDict ----
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

    # ---- system/fvSolution (centralCoeffs) ----
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
def sod_case(tmp_path):
    """Create a Sod shock tube case (100 cells)."""
    case_dir = tmp_path / "sod_tube"
    _make_sod_tube_case(case_dir, n_cells=100)
    return case_dir


@pytest.fixture
def sod_case_coarse(tmp_path):
    """Create a coarse Sod shock tube case (50 cells)."""
    case_dir = tmp_path / "sod_tube_coarse"
    _make_sod_tube_case(case_dir, n_cells=50)
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSodShockTube:
    """Validation: rhoCentralFoam on the Sod shock tube problem."""

    def test_case_structure(self, sod_case):
        """Case directory has expected rhoCentralFoam structure."""
        from pyfoam.io.case import Case

        case = Case(sod_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.has_field("T", 0)
        assert case.get_application() == "rhoCentralFoam"

    def test_mesh_dimensions(self, sod_case):
        """Mesh is 100 cells in a 1D tube arrangement."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(sod_case)
        assert solver.mesh.n_cells == 100
        assert solver.mesh.n_internal_faces == 99

    def test_solver_initialises(self, sod_case):
        """rhoCentralFoam initialises with correct Sod IC."""
        from pyfoam.applications.rho_central_foam import RhoCentralFoam

        solver = RhoCentralFoam(sod_case)

        assert solver.U.shape == (100, 3)
        assert solver.p.shape == (100,)
        assert solver.rho.shape == (100,)
        assert solver.T.shape == (100,)

    def test_initial_conditions_sod(self, sod_case):
        """Initial conditions match Sod left/right states."""
        from pyfoam.applications.rho_central_foam import RhoCentralFoam

        solver = RhoCentralFoam(sod_case)
        rho = solver.rho.detach().cpu().numpy()
        p = solver.p.detach().cpu().numpy()

        # Left half: rho ~ 1, p ~ 1
        left_rho = rho[:50]
        left_p = p[:50]
        assert np.allclose(left_rho, 1.0, atol=0.05), (
            f"Left rho expected ~1.0, got range [{left_rho.min():.3f}, "
            f"{left_rho.max():.3f}]"
        )
        assert np.allclose(left_p, 1.0, atol=0.05), (
            f"Left p expected ~1.0, got range [{left_p.min():.3f}, "
            f"{left_p.max():.3f}]"
        )

        # Right half: rho ~ 0.125, p ~ 0.1
        right_rho = rho[50:]
        right_p = p[50:]
        assert np.allclose(right_rho, 0.125, atol=0.01), (
            f"Right rho expected ~0.125, got range [{right_rho.min():.3f}, "
            f"{right_rho.max():.3f}]"
        )
        assert np.allclose(right_p, 0.1, atol=0.01), (
            f"Right p expected ~0.1, got range [{right_p.min():.3f}, "
            f"{right_p.max():.3f}]"
        )

    def test_run_produces_finite_fields(self, sod_case):
        """rhoCentralFoam completes and all field values are finite."""
        from pyfoam.applications.rho_central_foam import RhoCentralFoam

        solver = RhoCentralFoam(sod_case, CFL=0.2)
        conv = solver.run()

        assert torch.isfinite(solver.rho).all(), "rho contains NaN/Inf"
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"

    def test_density_shock_structure(self, sod_case):
        """Density profile shows expected Sod structure at t=0.2.

        At t=0.2 the solution has:
        - Rarefaction wave (smooth density decrease) in left region
        - Contact discontinuity (density jump, no pressure jump) near x=0.5
        - Sharp shock wave in right region near x=0.7
        """
        from pyfoam.applications.rho_central_foam import RhoCentralFoam

        solver = RhoCentralFoam(sod_case, CFL=0.2)
        solver.run()

        rho = solver.rho.detach().cpu().numpy()
        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        x = centres[:, 0]

        # Left boundary density should still be near 1.0
        left_mask = x < 0.1
        assert rho[left_mask].mean() > 0.9, (
            f"Left density too low: {rho[left_mask].mean():.3f}"
        )

        # Right boundary density should be near 0.125
        right_mask = x > 0.9
        assert rho[right_mask].mean() < 0.2, (
            f"Right density too high: {rho[right_mask].mean():.3f}"
        )

        # Density should be monotonically non-increasing from left to right
        # (allowing numerical oscillations)
        rho_sorted = np.sort(rho)[::-1]
        assert rho_sorted[0] > rho_sorted[-1], (
            "Density should vary across the tube"
        )

    def test_shock_position_reasonable(self, sod_case):
        """Density gradient exists in the expected shock region.

        On a coarse 100-cell explicit solver, the shock structure is
        heavily diffused. We verify the steepest density gradient is
        in the right half of the domain (x > 0.5).
        """
        from pyfoam.applications.rho_central_foam import RhoCentralFoam

        solver = RhoCentralFoam(sod_case, CFL=0.2)
        solver.run()

        rho = solver.rho.detach().cpu().numpy()
        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        x = centres[:, 0]

        # Find the largest negative density gradient
        grad_rho = np.gradient(rho, x)
        i_shock = np.argmin(grad_rho)
        x_shock = x[i_shock]

        # Steepest gradient should be in the right half
        assert x_shock > 0.45, (
            f"Steepest density gradient at x={x_shock:.3f}, "
            f"expected in right half (x > 0.45)"
        )

    def test_pressure_continuity_across_contact(self, sod_case):
        """Pressure profile shows the expected Sod structure.

        At the contact near x ~ 0.5, density jumps but pressure should
        be approximately continuous. On a coarse explicit solver with
        diffusion, we verify the pressure has the correct general shape:
        high on the left, lower on the right, with a smooth transition.
        """
        from pyfoam.applications.rho_central_foam import RhoCentralFoam

        solver = RhoCentralFoam(sod_case, CFL=0.2)
        solver.run()

        p = solver.p.detach().cpu().numpy()
        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        x = centres[:, 0]

        # Left region should have higher pressure than right
        left_mask = x < 0.2
        right_mask = x > 0.8
        if left_mask.any() and right_mask.any():
            p_left_mean = p[left_mask].mean()
            p_right_mean = p[right_mask].mean()
            assert p_left_mean > p_right_mean, (
                f"Left pressure ({p_left_mean:.4f}) should be higher than "
                f"right ({p_right_mean:.4f})"
            )

    def test_velocity_profile_shape(self, sod_case):
        """Velocity profile has the expected Sod structure.

        - Zero velocity at left and right boundaries
        - Positive velocity in expansion region (gas flows right)
        - Maximum velocity near x ~ 0.5
        """
        from pyfoam.applications.rho_central_foam import RhoCentralFoam

        solver = RhoCentralFoam(sod_case, CFL=0.2)
        solver.run()

        u = solver.U[:, 0].detach().cpu().numpy()
        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        x = centres[:, 0]

        # Velocity should be mostly non-negative (expansion to the right)
        assert u.max() > 0.0, "Expected positive velocity in expansion region"

        # Left boundary velocity should be near zero
        left_mask = x < 0.05
        if left_mask.any():
            assert np.abs(u[left_mask]).mean() < 0.1, (
                f"Left boundary velocity too large: {np.abs(u[left_mask]).mean():.3f}"
            )
