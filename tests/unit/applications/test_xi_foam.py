"""
Test XiFoam — premixed/partially-premixed combustion solver.

Creates a minimal 1D reacting case with U, p, b, T fields and verifies:
- Case loading and field initialisation
- Density computation from progress variable
- Flame speed model
- Solver run completion
- Finite output fields
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

def _make_combustion_case(
    case_dir: Path,
    n_cells: int = 5,
    L: float = 1.0,
    delta_t: float = 0.001,
    end_time: float = 0.005,
    n_outer: int = 3,
) -> None:
    """Write a minimal 1D combustion case for XiFoam.

    Same channel mesh as ReactingFoam test.  Provides U, p, b, T fields.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = L / n_cells
    dy = 0.1
    dz = 0.1

    # Points
    points = []
    for i in range(n_cells + 1):
        x = i * dx
        points.append((x, 0.0, 0.0))
        points.append((x, dy, 0.0))
        points.append((x, dy, dz))
        points.append((x, 0.0, dz))

    n_points = len(points)

    # Faces
    faces = []
    owner = []
    neighbour = []

    for i in range(n_cells - 1):
        faces.append((4, i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3))
        owner.append(i)
        neighbour.append(i + 1)

    n_internal = len(neighbour)

    # Inlet
    inlet_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    # Outlet
    outlet_start = inlet_start + 1
    level = n_cells
    faces.append((4, level * 4 + 0, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

    # Empty patches (y and z faces)
    empty_start = outlet_start + 1
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, (i + 1) * 4 + 0, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, i * 4 + 1, (i + 1) * 4 + 1, (i + 1) * 4 + 0))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 3, (i + 1) * 4 + 3, (i + 1) * 4 + 2, i * 4 + 2))
        owner.append(i)

    n_faces = len(faces)
    n_empty = 4 * n_cells

    # Write mesh files
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII, location="constant/polyMesh",
    )

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for p in points:
        lines.append(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})")
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
    lines = ["2", "("]
    lines.append("    inlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {inlet_start};")
    lines.append("    }")
    lines.append("    outlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {outlet_start};")
    lines.append("    }")
    lines.append("    walls")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- constant/momentumTransport ----
    mt_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="momentumTransport",
    )
    write_foam_file(
        case_dir / "constant" / "momentumTransport", mt_header,
        "simulationType  laminar;\n",
        overwrite=True,
    )

    # ---- constant/thermophysicalProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="thermophysicalProperties",
    )
    write_foam_file(
        case_dir / "constant" / "thermophysicalProperties", tp_header,
        "R               8.314;\n"
        "Cp              1005;\n",
        overwrite=True,
    )

    # ---- 0/U ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    write_foam_file(zero_dir / "U", u_header, (
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
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # ---- 0/p ----
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    write_foam_file(zero_dir / "p", p_header, (
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        "internalField   uniform 101325;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # ---- 0/b (progress variable) ----
    b_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="b",
    )
    write_foam_file(zero_dir / "b", b_header, (
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        "internalField   uniform 0.01;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform 0.01;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # ---- 0/T ----
    t_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    write_foam_file(zero_dir / "T", t_header, (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        "internalField   uniform 300;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform 300;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    write_foam_file(sys_dir / "controlDict", cd_header, (
        "application     XiFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time:g};\n"
        f"deltaT          {delta_t:g};\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n"
    ), overwrite=True)

    # ---- system/fvSchemes ----
    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, (
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n"
    ), overwrite=True)

    # ---- system/fvSolution ----
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, (
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
        f"    nOuterCorrectors    {n_outer};\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def combustion_case(tmp_path):
    """Create a 1D combustion case for XiFoam."""
    case_dir = tmp_path / "xiCombustion"
    _make_combustion_case(case_dir, n_cells=3, end_time=0.003, delta_t=0.001)
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestXiFoamInit:
    """Tests for XiFoam initialisation."""

    def test_case_loads(self, combustion_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case

        case = Case(combustion_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)

    def test_fields_initialise(self, combustion_case):
        """All fields are initialised from the 0/ directory."""
        from pyfoam.applications.xi_foam import XiFoam

        solver = XiFoam(combustion_case)

        assert solver.U.shape == (3, 3)
        assert solver.p.shape == (3,)
        assert solver.b.shape == (3,)
        assert solver.Xi.shape == (3,)
        assert solver.T.shape == (3,)
        assert solver.phi.shape == (solver.mesh.n_faces,)
        assert solver.rho.shape == (3,)

    def test_progress_variable_initialised(self, combustion_case):
        """Progress variable b is read from 0/b."""
        from pyfoam.applications.xi_foam import XiFoam

        solver = XiFoam(combustion_case)

        # b should be initialised to 0.01 from the field file
        assert torch.allclose(
            solver.b, torch.full((3,), 0.01, dtype=CFD_DTYPE), atol=1e-6,
        )

    def test_temperature_initialised(self, combustion_case):
        """Temperature is read from 0/T."""
        from pyfoam.applications.xi_foam import XiFoam

        solver = XiFoam(combustion_case)

        assert torch.allclose(
            solver.T, torch.full((3,), 300.0, dtype=CFD_DTYPE), atol=1e-3,
        )

    def test_xi_initialised(self, combustion_case):
        """Flame wrinkling factor Xi is set to Xi0."""
        from pyfoam.applications.xi_foam import XiFoam

        solver = XiFoam(combustion_case, Xi0=5.0)

        assert torch.allclose(
            solver.Xi, torch.full((3,), 5.0, dtype=CFD_DTYPE),
        )


class TestXiFoamModels:
    """Tests for XiFoam physical models."""

    def test_density_from_progress_variable(self, combustion_case):
        """Density is computed correctly from progress variable."""
        from pyfoam.applications.xi_foam import XiFoam

        solver = XiFoam(
            combustion_case,
            rho_unburnt=1.2,
            rho_burnt=0.15,
        )

        # For b=0 (unburnt): rho should be close to rho_u
        b_unburnt = torch.zeros(3, dtype=CFD_DTYPE)
        rho_u = solver._compute_density(b_unburnt)
        assert torch.allclose(rho_u, torch.full((3,), 1.2, dtype=CFD_DTYPE), atol=1e-6)

        # For b=1 (burnt): rho should be close to rho_b
        b_burnt = torch.ones(3, dtype=CFD_DTYPE)
        rho_b = solver._compute_density(b_burnt)
        assert torch.allclose(rho_b, torch.full((3,), 0.15, dtype=CFD_DTYPE), atol=1e-6)

    def test_density_finite(self, combustion_case):
        """Density is always finite and positive."""
        from pyfoam.applications.xi_foam import XiFoam

        solver = XiFoam(combustion_case)

        assert torch.isfinite(solver.rho).all()
        assert (solver.rho > 0).all()

    def test_flame_speed(self, combustion_case):
        """Turbulent flame speed is computed from Xi model."""
        from pyfoam.applications.xi_foam import XiFoam

        solver = XiFoam(combustion_case, SL0=0.4, Xi0=5.0)

        Xi = torch.full((3,), 5.0, dtype=CFD_DTYPE)
        S_T = solver._compute_flame_speed(Xi)

        assert torch.isfinite(S_T).all()
        assert torch.allclose(S_T, torch.full((3,), 0.4 * 5.0, dtype=CFD_DTYPE))


class TestXiFoamSolver:
    """Tests for XiFoam solver execution."""

    def test_run_completes(self, combustion_case):
        """Solver runs without errors."""
        from pyfoam.applications.xi_foam import XiFoam

        solver = XiFoam(combustion_case)
        result = solver.run()

        assert hasattr(result, "converged")

    def test_fields_finite_after_run(self, combustion_case):
        """All fields are finite after solving."""
        from pyfoam.applications.xi_foam import XiFoam

        solver = XiFoam(combustion_case)
        solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.b).all(), "b contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"
        assert torch.isfinite(solver.rho).all(), "rho contains NaN/Inf"

    def test_progress_variable_bounded(self, combustion_case):
        """Progress variable b stays in [0, 1] after solving."""
        from pyfoam.applications.xi_foam import XiFoam

        solver = XiFoam(combustion_case)
        solver.run()

        # b should be clamped to [0, 1] during solving
        assert solver.b.min() >= -1e-6, f"b min = {solver.b.min()}"
        assert solver.b.max() <= 1.0 + 1e-6, f"b max = {solver.b.max()}"

    def test_temperature_positive(self, combustion_case):
        """Temperature remains non-negative after solving."""
        from pyfoam.applications.xi_foam import XiFoam

        solver = XiFoam(combustion_case)
        solver.run()

        assert solver.T.min() >= 0, f"T min = {solver.T.min()}"
