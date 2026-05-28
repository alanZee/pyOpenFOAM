"""
Unit tests for CombustionFoam — general combustion solver.

Tests cover:
- Case loading and field initialisation
- Mechanism type selection
- Reaction mechanism reading
- Arrhenius kinetics computation
- Eddy-dissipation rate computation
- EDC rate computation
- Species source terms
- Solver execution
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

def _make_combustion_case(
    case_dir: Path,
    n_cells: int = 5,
    L: float = 1.0,
    end_time: int = 10,
    delta_t: float = 0.1,
    write_interval: int = 100,
    mechanism: str = "Arrhenius",
) -> None:
    """Write a 1D combustion case."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = L / n_cells
    dy = 0.1
    dz = 0.1

    points = []
    for i in range(n_cells + 1):
        x = i * dx
        points.append((x, 0.0, 0.0))
        points.append((x, dy, 0.0))
        points.append((x, dy, dz))
        points.append((x, 0.0, dz))

    n_points = len(points)

    faces = []
    owner = []
    neighbour = []

    for i in range(n_cells - 1):
        faces.append((4, i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3))
        owner.append(i)
        neighbour.append(i + 1)

    n_internal = len(neighbour)

    inlet_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    outlet_start = inlet_start + 1
    level = n_cells
    faces.append((4, level * 4 + 0, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

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

    # thermophysicalProperties
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="thermophysicalProperties",
    )
    write_foam_file(
        case_dir / "constant" / "thermophysicalProperties", tp_header,
        "R               8.314;\n"
        "Cp              1005;\n"
        "species\n{\n"
        "    fuel   1.0;\n"
        "    O2     1.0;\n"
        "    prod   1.0;\n"
        "}\n",
        overwrite=True,
    )

    # combustionProperties
    cp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="combustionProperties",
    )
    write_foam_file(
        case_dir / "constant" / "combustionProperties", cp_header,
        f"mechanism       {mechanism};\n"
        "eddyDissipationA    4.0;\n"
        "eddyDissipationB    0.5;\n",
        overwrite=True,
    )

    # reactions
    rxn_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="reactions",
    )
    write_foam_file(
        case_dir / "constant" / "reactions", rxn_header,
        "reaction1\n{\n"
        "    A               1e6;\n"
        "    beta            0.0;\n"
        "    Ea              50000;\n"
        "    reactants\n    {\n"
        "        fuel        1;\n"
        "        O2          1;\n"
        "    }\n"
        "    products\n    {\n"
        "        prod        1;\n"
        "    }\n"
        "    fuel            fuel;\n"
        "    oxidiser        O2;\n"
        "    stoichRatio     1.0;\n"
        "}\n",
        overwrite=True,
    )

    # 0/Yfuel
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    for fname, init_val in [("Yfuel", 0.05), ("YO2", 0.23), ("Yprod", 0.0)]:
        header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="volScalarField", location="0", object=fname,
        )
        write_foam_file(zero_dir / fname, header, (
            "dimensions      [0 0 0 0 0 0 0];\n\n"
            f"internalField   uniform {init_val};\n\n"
            "boundaryField\n{\n"
            "    inlet\n    {\n"
            "        type            fixedValue;\n"
            f"        value           uniform {init_val};\n"
            "    }\n"
            "    outlet\n    {\n"
            "        type            zeroGradient;\n"
            "    }\n"
            "    walls\n    {\n"
            "        type            empty;\n"
            "    }\n"
            "}\n"
        ), overwrite=True)

    # 0/T
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

    # 0/U
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

    # 0/p
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

    # system/controlDict
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    write_foam_file(sys_dir / "controlDict", cd_header, (
        "application     combustionFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        f"writeInterval   {write_interval};\n"
    ), overwrite=True)

    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, (
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n"
    ), overwrite=True)

    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, (
        "solvers\n{\n"
        "    Y\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "    T\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "combustionFoam\n{\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def combustion_case_arrhenius(tmp_path):
    """Combustion case with Arrhenius mechanism."""
    case_dir = tmp_path / "combustion_arr"
    _make_combustion_case(case_dir, n_cells=3, end_time=1, delta_t=0.01, mechanism="Arrhenius")
    return case_dir


@pytest.fixture
def combustion_case_ed(tmp_path):
    """Combustion case with EddyDissipation mechanism."""
    case_dir = tmp_path / "combustion_ed"
    _make_combustion_case(case_dir, n_cells=3, end_time=1, delta_t=0.01, mechanism="EddyDissipation")
    return case_dir


@pytest.fixture
def combustion_case_edc(tmp_path):
    """Combustion case with EDC mechanism."""
    case_dir = tmp_path / "combustion_edc"
    _make_combustion_case(case_dir, n_cells=3, end_time=1, delta_t=0.01, mechanism="EDC")
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCombustionFoamInit:
    """Tests for CombustionFoam initialisation."""

    def test_case_loads(self, combustion_case_arrhenius):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(combustion_case_arrhenius)
        assert case.has_mesh()

    def test_species_detected(self, combustion_case_arrhenius):
        """Species are detected from 0/ directory."""
        from pyfoam.applications.combustion_foam import CombustionFoam

        solver = CombustionFoam(combustion_case_arrhenius)
        assert "fuel" in solver.species
        assert "O2" in solver.species
        assert "prod" in solver.species

    def test_reactions_read(self, combustion_case_arrhenius):
        """Reactions are read from constant/reactions."""
        from pyfoam.applications.combustion_foam import CombustionFoam

        solver = CombustionFoam(combustion_case_arrhenius)
        assert len(solver.reactions) == 1
        assert solver.reactions[0].A == 1e6
        assert solver.reactions[0].fuel == "fuel"
        assert solver.reactions[0].oxidiser == "O2"


class TestCombustionFoamMechanism:
    """Tests for mechanism type selection."""

    def test_arrhenius_mechanism(self, combustion_case_arrhenius):
        """Arrhenius mechanism is selected correctly."""
        from pyfoam.applications.combustion_foam import CombustionFoam, MechanismType

        solver = CombustionFoam(combustion_case_arrhenius)
        assert solver.mechanism == MechanismType.ARRHENIUS

    def test_eddy_dissipation_mechanism(self, combustion_case_ed):
        """Eddy-dissipation mechanism is selected correctly."""
        from pyfoam.applications.combustion_foam import CombustionFoam, MechanismType

        solver = CombustionFoam(combustion_case_ed)
        assert solver.mechanism == MechanismType.EDDY_DISSIPATION

    def test_edc_mechanism(self, combustion_case_edc):
        """EDC mechanism is selected correctly."""
        from pyfoam.applications.combustion_foam import CombustionFoam, MechanismType

        solver = CombustionFoam(combustion_case_edc)
        assert solver.mechanism == MechanismType.EDC


class TestCombustionFoamKinetics:
    """Tests for combustion kinetics computation."""

    def test_arrhenius_rate_shape(self, combustion_case_arrhenius):
        """Arrhenius rate has correct shape."""
        from pyfoam.applications.combustion_foam import CombustionFoam

        solver = CombustionFoam(combustion_case_arrhenius)
        rate = solver._compute_arrhenius_rate(
            solver.reactions[0], solver.T, solver.Y,
        )
        assert rate.shape == (3,)
        assert torch.isfinite(rate).all()

    def test_arrhenius_rate_positive(self, combustion_case_arrhenius):
        """Arrhenius rate is non-negative."""
        from pyfoam.applications.combustion_foam import CombustionFoam

        solver = CombustionFoam(combustion_case_arrhenius)
        rate = solver._compute_arrhenius_rate(
            solver.reactions[0], solver.T, solver.Y,
        )
        assert (rate >= 0).all(), "Rate has negative values"

    def test_eddy_dissipation_rate_finite(self, combustion_case_ed):
        """Eddy-dissipation rate is finite."""
        from pyfoam.applications.combustion_foam import CombustionFoam

        solver = CombustionFoam(combustion_case_ed)
        rate = solver._compute_eddy_dissipation_rate(
            solver.reactions[0], solver.T, solver.Y,
        )
        assert rate.shape == (3,)
        assert torch.isfinite(rate).all()

    def test_edc_rate_finite(self, combustion_case_edc):
        """EDC rate is finite."""
        from pyfoam.applications.combustion_foam import CombustionFoam

        solver = CombustionFoam(combustion_case_edc)
        rate = solver._compute_edc_rate(
            solver.reactions[0], solver.T, solver.Y,
        )
        assert rate.shape == (3,)
        assert torch.isfinite(rate).all()

    def test_species_source_terms(self, combustion_case_arrhenius):
        """Species source terms have correct shape."""
        from pyfoam.applications.combustion_foam import CombustionFoam

        solver = CombustionFoam(combustion_case_arrhenius)
        omega = solver._compute_combustion_source_terms(solver.T, solver.Y)

        assert "fuel" in omega
        assert "O2" in omega
        assert "prod" in omega


class TestCombustionFoamSolver:
    """Tests for solver execution."""

    def test_run_completes_arrhenius(self, combustion_case_arrhenius):
        """Solver runs with Arrhenius mechanism."""
        from pyfoam.applications.combustion_foam import CombustionFoam

        solver = CombustionFoam(combustion_case_arrhenius)
        result = solver.run()
        assert "converged" in result

    def test_run_completes_ed(self, combustion_case_ed):
        """Solver runs with eddy-dissipation mechanism."""
        from pyfoam.applications.combustion_foam import CombustionFoam

        solver = CombustionFoam(combustion_case_ed)
        result = solver.run()
        assert "converged" in result

    def test_fields_finite(self, combustion_case_arrhenius):
        """All fields remain finite after solving."""
        from pyfoam.applications.combustion_foam import CombustionFoam

        solver = CombustionFoam(combustion_case_arrhenius)
        solver.run()

        for name in solver.species:
            assert torch.isfinite(solver.Y[name]).all(), f"Y_{name} contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"
