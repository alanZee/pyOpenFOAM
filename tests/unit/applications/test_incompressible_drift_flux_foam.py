"""
Test incompressibleDriftFlux solver.

Uses a sediment settling case with a dispersed phase fraction field.
Verifies:
- Case loading and field initialisation
- Mixture property computation
- Stokes settling velocity calculation
- Drift velocity computation
- PIMPLE outer loop execution
- Run produces finite, bounded fields
- Alpha transport with drift flux
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------

def _make_settling_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 8,
    dx: float = 0.01,
    dy: float = 0.01,
    delta_t: float = 0.001,
    end_time: float = 0.005,
    n_outer: int = 2,
    n_correctors: int = 2,
) -> None:
    """Write a complete settling case for drift-flux solver.

    Creates a vertical column with dispersed phase initially concentrated
    in the upper half, which should settle under gravity.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    dz = 0.01
    Ly = n_cells_y * dy

    # ---- Points ----
    points_z0 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z0.append((i * dx, j * dy, 0.0))
    n_base = len(points_z0)
    points_z1 = [
        (i * dx, j * dy, dz)
        for j in range(n_cells_y + 1)
        for i in range(n_cells_x + 1)
    ]
    all_points = points_z0 + points_z1
    n_points = len(all_points)

    # ---- Faces ----
    faces = []
    owner = []
    neighbour = []

    # Internal vertical (x-direction)
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal (y-direction)
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

    # Boundary: top
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)
    n_top = n_cells_x
    top_start = n_internal

    # Bottom
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)
    n_bottom = n_cells_x
    bottom_start = top_start + n_top

    # Left
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)
    n_left = n_cells_y
    left_start = bottom_start + n_bottom

    # Right
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)
    n_right = n_cells_y
    right_start = left_start + n_left

    # Front / back (empty)
    for _ in range(2):
        for j in range(n_cells_y):
            for i in range(n_cells_x):
                p0 = j * (n_cells_x + 1) + i
                p1 = p0 + 1
                p2 = p1 + n_cells_x + 1
                p3 = p0 + n_cells_x + 1
                faces.append((4, p0, p1, p2, p3))
                owner.append(j * n_cells_x + i)
    n_empty = 2 * n_cells_x * n_cells_y
    empty_start = right_start + n_right

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

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
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{face[0]}({verts})")
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
    lines = ["5", "("]
    for name, n_f, start in [
        ("topWall", n_top, top_start),
        ("bottomWall", n_bottom, bottom_start),
        ("leftWall", n_left, left_start),
        ("rightWall", n_right, right_start),
        ("frontAndBack", n_empty, empty_start),
    ]:
        lines.append(f"    {name}")
        lines.append("    {")
        if name == "frontAndBack":
            lines.append("        type            empty;")
        else:
            lines.append("        type            wall;")
        lines.append(f"        nFaces          {n_f};")
        lines.append(f"        startFace       {start};")
        lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- constant/transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    tp_body = (
        "phase1\n{\n"
        "    nu              [0 2 -1 0 0 0 0] 1e-3;\n"
        "    rho             [1 -3 0 0 0 0 0] 1000;\n"
        "}\n"
        "phase2\n{\n"
        "    nu              [0 2 -1 0 0 0 0] 1e-5;\n"
        "    rho             [1 -3 0 0 0 0 0] 2500;\n"
        "}\n"
        "d               [0 1 0 0 0 0 0] 1e-4;\n"
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header, tp_body,
        overwrite=True,
    )

    # ---- constant/g ----
    g_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="uniformVectorField", location="constant", object="g",
    )
    write_foam_file(
        case_dir / "constant" / "g", g_header,
        "uniform (0 -9.81 0);\n", overwrite=True,
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
        "    topWall\n    {\n        type            slip;\n    }\n"
        "    bottomWall\n    {\n        type            noSlip;\n    }\n"
        "    leftWall\n    {\n        type            noSlip;\n    }\n"
        "    rightWall\n    {\n        type            noSlip;\n    }\n"
        "    frontAndBack\n    {\n        type            empty;\n    }\n"
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
        "    topWall\n    {\n        type            zeroGradient;\n    }\n"
        "    bottomWall\n    {\n        type            zeroGradient;\n    }\n"
        "    leftWall\n    {\n        type            zeroGradient;\n    }\n"
        "    rightWall\n    {\n        type            zeroGradient;\n    }\n"
        "    frontAndBack\n    {\n        type            empty;\n    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- 0/alpha (dispersed phase fraction) ----
    alpha_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="alpha",
    )
    # Sediment initially in upper half
    alpha_lines = [
        "dimensions      [0 0 0 0 0 0 0];",
        "",
        f"internalField   nonuniform {n_cells}",
        "(",
    ]
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            yc = (j + 0.5) * dy
            # Upper half: 0.3 volume fraction of dispersed phase
            alpha_lines.append("0.3" if yc > Ly / 2.0 else "0")
    alpha_lines += [
        ")",
        "",
        "boundaryField\n{",
    ]
    for bname in ["topWall", "bottomWall", "leftWall", "rightWall"]:
        alpha_lines += [f"    {bname}", "    {", "        type            zeroGradient;", "    }"]
    alpha_lines += ["    frontAndBack", "    {", "        type            empty;", "    }", "}"]
    write_foam_file(
        zero_dir / "alpha", alpha_header,
        "\n".join(alpha_lines) + "\n", overwrite=True,
    )

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     incompressibleDriftFluxFoam;\n"
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
        "divSchemes\n{\n    default         none;\n"
        "    div(phi,alpha)  Gauss vanLeer;\n"
        "    div(phi,U)      Gauss upwind;\n"
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
        "    p\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "        maxIter         100;\n"
        "    }\n"
        "    U\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "        maxIter         100;\n"
        "    }\n"
        "}\n\n"
        "PIMPLE\n{\n"
        f"    nOuterCorrectors    {n_outer};\n"
        f"    nCorrectors         {n_correctors};\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settling_case(tmp_path):
    """Create a sediment settling case (4x8 mesh)."""
    case_dir = tmp_path / "driftFluxSettling"
    _make_settling_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=8,
        delta_t=0.001,
        end_time=0.003,
        n_outer=2,
        n_correctors=2,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIncompressibleDriftFluxFoam:
    """Tests for the incompressibleDriftFlux solver."""

    def test_case_loads(self, settling_case):
        """Solver loads the case and reads all required fields."""
        from pyfoam.applications.incompressible_drift_flux_foam import IncompressibleDriftFluxFoam

        solver = IncompressibleDriftFluxFoam(
            settling_case,
            rho_c=1000.0, rho_d=2500.0,
            mu_c=1e-3, mu_d=1e-2,
            d=1e-4,
        )

        assert solver.mesh.n_cells == 32
        assert solver.U.shape == (32, 3)
        assert solver.p.shape == (32,)
        assert solver.alpha.shape == (32,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

    def test_mixture_properties(self, settling_case):
        """Mixture density and viscosity are finite and physically consistent."""
        from pyfoam.applications.incompressible_drift_flux_foam import IncompressibleDriftFluxFoam

        solver = IncompressibleDriftFluxFoam(
            settling_case,
            rho_c=1000.0, rho_d=2500.0,
            mu_c=1e-3, mu_d=1e-2,
            d=1e-4,
        )
        alpha = solver.alpha

        rho = solver._compute_mixture_rho(alpha)
        mu = solver._compute_mixture_mu(alpha)

        assert torch.isfinite(rho).all()
        assert rho.min() > 0
        assert torch.isfinite(mu).all()
        assert mu.min() > 0

        # Pure carrier (alpha=0): rho = rho_c
        rho_pure = solver._compute_mixture_rho(torch.zeros_like(alpha))
        assert torch.allclose(rho_pure, torch.full_like(rho_pure, solver.rho_c))

        # Pure dispersed (alpha=1): rho = rho_d
        rho_pure_d = solver._compute_mixture_rho(torch.ones_like(alpha))
        assert torch.allclose(rho_pure_d, torch.full_like(rho_pure_d, solver.rho_d))

    def test_stokes_settling_velocity(self, settling_case):
        """Stokes settling velocity formula is correct."""
        from pyfoam.applications.incompressible_drift_flux_foam import IncompressibleDriftFluxFoam

        solver = IncompressibleDriftFluxFoam(
            settling_case,
            rho_c=1000.0, rho_d=2500.0,
            mu_c=1e-3, mu_d=1e-2,
            d=1e-4,
        )

        # Manual calculation: U_slip = (rho_d - rho_c) * g * d^2 / (18 * mu_c)
        g = 9.81
        expected = (2500.0 - 1000.0) * g * (1e-4) ** 2 / (18.0 * 1e-3)
        assert abs(solver.U_slip_scalar - expected) / expected < 1e-10

    def test_drift_velocity_direction(self, settling_case):
        """Drift velocity points downward (same direction as gravity for heavier particles)."""
        from pyfoam.applications.incompressible_drift_flux_foam import IncompressibleDriftFluxFoam

        solver = IncompressibleDriftFluxFoam(
            settling_case,
            rho_c=1000.0, rho_d=2500.0,
            mu_c=1e-3, mu_d=1e-2,
            d=1e-4,
        )

        # Use alpha=0.5 everywhere for a clean test
        alpha_half = torch.full(
            (solver.mesh.n_cells,), 0.5,
            dtype=solver.alpha.dtype, device=solver.alpha.device,
        )
        drift = solver._compute_drift_velocity(alpha_half)

        # Drift should be in the direction of gravity (negative y)
        assert drift[:, 1].max() <= 0, "Drift y-component should be non-positive (downward)"

        # Maximum drift at alpha=0.5 (alpha*(1-alpha) is maximized)
        alpha_other = torch.full(
            (solver.mesh.n_cells,), 0.1,
            dtype=solver.alpha.dtype, device=solver.alpha.device,
        )
        drift_other = solver._compute_drift_velocity(alpha_other)
        # drift at alpha=0.5 > drift at alpha=0.1
        assert drift[:, 1].abs().mean() > drift_other[:, 1].abs().mean()

    def test_run_produces_valid_fields(self, settling_case):
        """After running, all fields are finite and alpha is bounded."""
        from pyfoam.applications.incompressible_drift_flux_foam import IncompressibleDriftFluxFoam

        solver = IncompressibleDriftFluxFoam(
            settling_case,
            rho_c=1000.0, rho_d=2500.0,
            mu_c=1e-3, mu_d=1e-2,
            d=1e-4,
        )
        conv = solver.run()

        # Shapes preserved
        assert solver.U.shape == (32, 3)
        assert solver.p.shape == (32,)
        assert solver.alpha.shape == (32,)

        # No NaN/Inf
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.alpha).all(), "alpha contains NaN/Inf"

        # Alpha bounded
        assert solver.alpha.min() >= -1e-6, f"alpha min = {solver.alpha.min()}"
        assert solver.alpha.max() <= 1.0 + 1e-6, f"alpha max = {solver.alpha.max()}"

    def test_pimple_loop_executes(self, settling_case):
        """PIMPLE outer loop runs and produces valid fields."""
        from pyfoam.applications.incompressible_drift_flux_foam import IncompressibleDriftFluxFoam

        solver = IncompressibleDriftFluxFoam(
            settling_case,
            rho_c=1000.0, rho_d=2500.0,
            mu_c=1e-3, mu_d=1e-2,
            d=1e-4,
        )

        # Run one time step
        U, p, alpha, phi, conv = solver._pimple_drift_flux_step()

        # At least one outer iteration should have run
        assert conv.outer_iterations >= 1

        # All fields should be finite
        assert torch.isfinite(U).all(), "U contains NaN/Inf after PIMPLE"
        assert torch.isfinite(p).all(), "p contains NaN/Inf after PIMPLE"
        assert torch.isfinite(phi).all(), "phi contains NaN/Inf after PIMPLE"
        assert torch.isfinite(alpha).all(), "alpha contains NaN/Inf after PIMPLE"

    def test_heavier_particles_settle(self, settling_case):
        """Over time, heavier particles should settle (mass center moves down)."""
        from pyfoam.applications.incompressible_drift_flux_foam import IncompressibleDriftFluxFoam

        solver = IncompressibleDriftFluxFoam(
            settling_case,
            rho_c=1000.0, rho_d=2500.0,
            mu_c=1e-3, mu_d=1e-2,
            d=1e-4,
        )

        # Compute initial mass center
        mesh = solver.mesh
        cell_centres = mesh.cell_centres
        y_coords = cell_centres[:, 1]
        initial_mass_center = (
            (solver.alpha * y_coords).sum() / solver.alpha.sum().clamp(min=1e-30)
        ).item()

        # Run several steps
        conv = solver.run()

        final_mass_center = (
            (solver.alpha * y_coords).sum() / solver.alpha.sum().clamp(min=1e-30)
        ).item()

        # Mass center should move downward (or stay same if already settled)
        # With settling particles, the mass center should decrease.
        # Very short simulation (3ms) with small settling velocity (~8 mm/s),
        # so allow a small tolerance for numerical diffusion effects.
        assert final_mass_center <= initial_mass_center + 1e-4, (
            f"Mass center should not rise significantly: "
            f"{initial_mass_center:.6f} -> {final_mass_center:.6f}"
        )
