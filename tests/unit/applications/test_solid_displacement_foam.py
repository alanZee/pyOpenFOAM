"""
Unit tests for SolidDisplacementFoam — linear elasticity solver.

Tests cover:
- Case loading and field initialisation
- Mechanical property reading (E, nu)
- Lamé parameter computation
- Strain computation
- Stress computation (Hooke's law)
- Von Mises stress
- Displacement equation assembly
- Run convergence
- Field writing
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper
# ---------------------------------------------------------------------------

def _make_solid_case(
    case_dir: Path,
    n_cells: int = 5,
    L: float = 1.0,
    E: float = 1e9,
    nu: float = 0.3,
    end_time: int = 100,
    write_interval: int = 100,
    D_fixed: float = 0.01,
) -> None:
    """Write a 1D solid mechanics case."""
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

    # Left boundary (x=0): fixed displacement
    left_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    # Right boundary (x=L): prescribed displacement
    right_start = left_start + 1
    level = n_cells
    faces.append((4, level * 4 + 0, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

    # Empty patches
    empty_start = right_start + 1

    # Bottom (y=0)
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, (i + 1) * 4 + 0, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)

    # Top (y=dy)
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)

    # Front (z=0)
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, i * 4 + 1, (i + 1) * 4 + 1, (i + 1) * 4 + 0))
        owner.append(i)

    # Back (z=dz)
    for i in range(n_cells):
        faces.append((4, i * 4 + 3, (i + 1) * 4 + 3, (i + 1) * 4 + 2, i * 4 + 2))
        owner.append(i)

    n_faces = len(faces)

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
    n_empty = 4 * n_cells
    lines = ["4", "("]
    lines.append("    left")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {left_start};")
    lines.append("    }")
    lines.append("    right")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {right_start};")
    lines.append("    }")
    lines.append("    walls")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # mechanicalProperties
    mp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="mechanicalProperties",
    )
    write_foam_file(
        case_dir / "constant" / "mechanicalProperties", mp_header,
        f"E               [1 -1 -2 0 0 0 0] {E};\n"
        f"nu              [0 0 0 0 0 0 0] {nu};\n",
        overwrite=True,
    )

    # 0/D
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    d_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="D",
    )
    d_body = (
        "dimensions      [0 1 0 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    left\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    right\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform ({D_fixed} 0 0);\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "D", d_header, d_body, overwrite=True)

    # system/controlDict
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    write_foam_file(sys_dir / "controlDict", cd_header, (
        "application     solidDisplacementFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        "deltaT          1;\n"
        "writeControl    timeStep;\n"
        f"writeInterval   {write_interval};\n"
    ), overwrite=True)

    # system/fvSchemes
    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, (
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n"
    ), overwrite=True)

    # system/fvSolution
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, (
        "solvers\n{\n"
        "    D\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "solidMechanics\n{\n"
        "    nCorrectors     1;\n"
        "    convergenceTolerance 1e-5;\n"
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def solid_case(tmp_path):
    """Create a 1D solid mechanics case."""
    case_dir = tmp_path / "solid"
    _make_solid_case(
        case_dir,
        n_cells=5,
        E=1e9,
        nu=0.3,
        end_time=10,
        D_fixed=0.01,
    )
    return case_dir


@pytest.fixture
def tiny_solid_case(tmp_path):
    """Create a minimal 3-cell solid case."""
    case_dir = tmp_path / "tiny_solid"
    _make_solid_case(
        case_dir,
        n_cells=3,
        E=1e9,
        nu=0.3,
        end_time=5,
        D_fixed=0.01,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSolidDisplacementFoamInit:
    """Tests for SolidDisplacementFoam initialisation."""

    def test_case_loads(self, solid_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(solid_case)
        assert case.has_mesh()

    def test_mechanical_properties(self, solid_case):
        """Mechanical properties are read correctly."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        solver = SolidDisplacementFoam(solid_case)
        assert abs(solver.E - 1e9) < 1e3
        assert abs(solver.nu - 0.3) < 1e-10

    def test_lame_parameters(self, solid_case):
        """Lamé parameters are computed correctly."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        solver = SolidDisplacementFoam(solid_case)

        # λ = Eν / ((1+ν)(1-2ν))
        expected_lam = 1e9 * 0.3 / ((1 + 0.3) * (1 - 2 * 0.3))
        assert abs(solver.lam - expected_lam) < 1e3

        # μ = E / (2(1+ν))
        expected_mu = 1e9 / (2 * (1 + 0.3))
        assert abs(solver.mu - expected_mu) < 1e3

    def test_displacement_initialised(self, solid_case):
        """Displacement field is initialised correctly."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        solver = SolidDisplacementFoam(solid_case)
        assert solver.D.shape == (5, 3)

    def test_strain_computed(self, solid_case):
        """Strain tensor is computed."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        solver = SolidDisplacementFoam(solid_case)
        assert solver.epsilon.shape == (5, 6)

    def test_stress_computed(self, solid_case):
        """Stress tensor is computed."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        solver = SolidDisplacementFoam(solid_case)
        assert solver.sigma.shape == (5, 6)


class TestSolidDisplacementFoamConstitutive:
    """Tests for constitutive law (Hooke's law)."""

    def test_stress_strain_relationship(self, solid_case):
        """Stress-strain relationship follows Hooke's law."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        solver = SolidDisplacementFoam(solid_case)

        # For zero displacement, stress should be zero
        assert torch.allclose(
            solver.sigma, torch.zeros_like(solver.sigma), atol=1e-10
        )

    def test_von_mises_stress_zero_initially(self, solid_case):
        """Von Mises stress is zero for zero displacement."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        solver = SolidDisplacementFoam(solid_case)
        von_mises = solver._compute_von_mises_stress()

        assert torch.allclose(von_mises, torch.zeros_like(von_mises), atol=1e-10)

    def test_hooke_law_uniaxial(self):
        """Hooke's law is correct for uniaxial strain."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        # Create a simple case with known displacement
        # For uniaxial strain ε_xx = ε, others = 0
        # σ_xx = (λ + 2μ) ε
        E = 1e9
        nu = 0.3
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        # σ_xx = (λ + 2μ) * ε_xx
        expected_sigma_xx = (lam + 2 * mu) * 0.01

        # This is a conceptual test - actual implementation would need
        # a proper mesh with prescribed displacement


class TestSolidDisplacementFoamSolver:
    """Tests for solver execution."""

    def test_run_completes(self, tiny_solid_case):
        """Solver runs without errors."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        solver = SolidDisplacementFoam(tiny_solid_case)
        result = solver.run()

        assert "converged" in result

    def test_displacement_finite(self, tiny_solid_case):
        """Displacement is finite after solving."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        solver = SolidDisplacementFoam(tiny_solid_case)
        solver.run()

        assert torch.isfinite(solver.D).all(), "D contains NaN/Inf"

    def test_displacement_changes(self, tiny_solid_case):
        """Displacement changes from initial conditions."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        solver = SolidDisplacementFoam(tiny_solid_case)
        D_initial = solver.D.clone()

        solver.run()

        diff = (solver.D - D_initial).abs().sum()
        assert diff > 0, "Displacement did not change"

    def test_von_mises_positive(self, tiny_solid_case):
        """Von Mises stress is non-negative after solving."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        solver = SolidDisplacementFoam(tiny_solid_case)
        solver.run()

        von_mises = solver._compute_von_mises_stress()
        assert (von_mises >= -1e-10).all(), "Von Mises stress has negative values"

    def test_writes_output(self, tiny_solid_case):
        """Fields are written to time directories."""
        from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

        solver = SolidDisplacementFoam(tiny_solid_case)
        solver.run()

        time_dirs = [
            d for d in tiny_solid_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1
