"""
Unit tests for EnergyFoam — enhanced energy equation solver.

Tests cover:
- Solver initialisation with default and custom properties
- Viscous dissipation flag
- Compressibility work flag
- Run produces finite values
- Run converges
- Under-relaxation effect
- Output writing
- Source term toggling
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper (reused from heat_transfer_foam tests)
# ---------------------------------------------------------------------------

def _make_energy_case(
    case_dir: Path,
    n_cells_x: int = 10,
    n_cells_y: int = 10,
    T_init: float = 300.0,
    T_hot: float = 310.0,
    T_cold: float = 290.0,
    kappa: float = 0.026,
    end_time: int = 50,
    write_interval: int = 50,
    max_outer_iterations: int = 50,
    convergence_tolerance: float = 1e-4,
    alpha_T: float = 0.7,
) -> None:
    """Write a complete energy equation case to *case_dir*."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1

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

    faces = []
    owner = []
    neighbour = []

    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

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

    # hotWall (left)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)
    n_hot = n_cells_y
    hot_start = n_internal

    # coldWall (right)
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
    n_adiabatic = 2 * n_cells_x
    adiabatic_start = cold_start + n_cold

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
    empty_start = adiabatic_start + n_adiabatic

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

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
    lines = ["4", "("]
    lines.append("    hotWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_hot};")
    lines.append(f"        startFace       {hot_start};")
    lines.append("    }")
    lines.append("    coldWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_cold};")
    lines.append(f"        startFace       {cold_start};")
    lines.append("    }")
    lines.append("    adiabaticWalls")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_adiabatic};")
    lines.append(f"        startFace       {adiabatic_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # 0/T
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    T_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    T_body = (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        f"internalField   uniform {T_init};\n\n"
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

    # system/controlDict
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     energyFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time};\n"
        "deltaT          1;\n"
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

    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    fs_body = (
        "ddtSchemes\n{\n    default         steadyState;\n}\n\n"
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
        "    T\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        maxIter         1000;\n"
        "    }\n"
        "}\n\n"
        "SIMPLE\n{\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    residualControl\n    {\n"
        "        T               1e-4;\n"
        "    }\n"
        "    relaxationFactors\n    {\n"
        f"        T               {alpha_T};\n"
        "    }\n"
        f"    convergenceTolerance {convergence_tolerance};\n"
        f"    maxOuterIterations  {max_outer_iterations};\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def energy_case(tmp_path):
    """Create an energy equation case (2x2 mesh for fast tests)."""
    case_dir = tmp_path / "energy_case"
    _make_energy_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_hot=310.0,
        T_cold=290.0,
        kappa=0.026,
        end_time=10,
        write_interval=10,
        max_outer_iterations=10,
    )
    return case_dir


@pytest.fixture
def energy_case_4x4(tmp_path):
    """Create a 4x4 energy equation case."""
    case_dir = tmp_path / "energy_4x4"
    _make_energy_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        T_init=300.0,
        T_hot=310.0,
        T_cold=290.0,
        end_time=20,
        write_interval=20,
        max_outer_iterations=20,
    )
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestEnergyFoamInit:
    """Tests for EnergyFoam initialisation."""

    def test_case_loads(self, energy_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(energy_case)
        assert case.has_mesh()
        assert case.has_field("T", 0)

    def test_mesh_builds(self, energy_case):
        """FvMesh is constructed correctly."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(energy_case)
        assert solver.mesh.n_cells == 4  # 2x2

    def test_default_properties(self, energy_case):
        """Default material properties are set."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(energy_case)
        assert solver.kappa > 0
        assert solver.Cp > 0
        assert solver.rho_const > 0
        assert solver.mu > 0
        assert solver.beta > 0

    def test_custom_properties_injection(self, energy_case):
        """Custom material properties can be injected."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(
            energy_case,
            kappa=50.0,
            Cp=2000.0,
            rho=5.0,
            mu=1e-3,
            beta=1e-4,
        )
        assert solver.kappa == 50.0
        assert solver.Cp == 2000.0
        assert solver.rho_const == 5.0
        assert solver.mu == 1e-3
        assert solver.beta == 1e-4

    def test_viscous_dissipation_flag(self, energy_case):
        """Viscous dissipation flag is stored."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(energy_case, viscous_dissipation=True)
        assert solver.viscous_dissipation is True

        solver2 = EnergyFoam(energy_case, viscous_dissipation=False)
        assert solver2.viscous_dissipation is False

    def test_compressibility_work_flag(self, energy_case):
        """Compressibility work flag is stored."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(energy_case, compressibility_work=True)
        assert solver.compressibility_work is True

    def test_temperature_field_shape(self, energy_case):
        """Temperature field has correct shape."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(energy_case)
        assert solver.T.shape == (4,)


class TestEnergyFoamRun:
    """Tests for solver execution."""

    def test_run_produces_finite_values(self, energy_case):
        """Solver produces finite temperature values."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(energy_case)
        solver.run()
        assert torch.isfinite(solver.T).all()

    def test_run_converges(self, energy_case):
        """Solver reports convergence data."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(energy_case)
        conv = solver.run()
        assert conv.T_residual >= 0

    def test_run_with_viscous_dissipation(self, energy_case):
        """Solver runs with viscous dissipation enabled."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(energy_case, viscous_dissipation=True, alpha_T=1.0)
        solver.run()
        assert torch.isfinite(solver.T).all()

    def test_run_with_compressibility_work(self, energy_case):
        """Solver runs with compressibility work enabled."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(energy_case, compressibility_work=True, alpha_T=1.0)
        solver.run()
        assert torch.isfinite(solver.T).all()

    def test_run_with_all_sources(self, energy_case):
        """Solver runs with all source terms enabled simultaneously."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(
            energy_case,
            viscous_dissipation=True,
            compressibility_work=True,
            alpha_T=1.0,
        )
        solver.run()
        assert torch.isfinite(solver.T).all()

    def test_pure_conduction_temperature_gradient(self, energy_case_4x4):
        """1D conduction produces a temperature gradient from hot to cold."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(energy_case_4x4, alpha_T=1.0)
        solver.run()

        mesh = solver.mesh
        x = mesh.cell_centres[:, 0]

        left_mask = x < 0.25
        right_mask = x > 0.75

        if left_mask.any() and right_mask.any():
            T_left = solver.T[left_mask].mean()
            T_right = solver.T[right_mask].mean()
            assert T_left > T_right, (
                f"T_left={T_left:.4f} should be > T_right={T_right:.4f}"
            )

    def test_under_relaxation_effect(self, energy_case):
        """Lower under-relaxation still produces finite results."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(energy_case, alpha_T=0.3)
        conv = solver.run()
        assert torch.isfinite(solver.T).all()

    def test_run_writes_output(self, energy_case):
        """Solver writes T to time directories."""
        from pyfoam.applications.energy_foam import EnergyFoam
        solver = EnergyFoam(energy_case)
        solver.run()

        time_dirs = [
            d for d in energy_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1
        for td in time_dirs:
            assert (td / "T").exists(), f"T not found in {td}"

    def test_exports_in_all(self):
        """EnergyFoam is importable from the applications package."""
        from pyfoam.applications import EnergyFoam
        assert EnergyFoam is not None
