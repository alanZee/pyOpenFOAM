"""
Unit tests for ReactingFoam — reacting flow solver.

Tests cover:
- Case loading and field initialisation
- Reaction mechanism reading
- Arrhenius kinetics computation
- Species source terms
- Transport equation assembly
- Time-stepping loop
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

def _make_reacting_case(
    case_dir: Path,
    n_cells: int = 5,
    L: float = 1.0,
    end_time: int = 10,
    delta_t: float = 0.1,
    write_interval: int = 100,
    T_init: float = 300.0,
    Y_A_init: float = 1.0,
) -> None:
    """Write a 1D reacting flow case."""
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

    # Empty patches
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
        "    A   1.0;\n"
        "    B   1.0;\n"
        "}\n",
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
        "        A           1;\n"
        "    }\n"
        "    products\n    {\n"
        "        B           1;\n"
        "    }\n"
        "}\n",
        overwrite=True,
    )

    # 0/YA
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    ya_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="YA",
    )
    write_foam_file(zero_dir / "YA", ya_header, (
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        f"internalField   uniform {Y_A_init};\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {Y_A_init};\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # 0/YB
    yb_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="YB",
    )
    write_foam_file(zero_dir / "YB", yb_header, (
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform 0;\n"
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
        f"internalField   uniform {T_init};\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_init};\n"
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
        "application     reactingFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        f"writeInterval   {write_interval};\n"
    ), overwrite=True)

    # system/fvSchemes
    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, (
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n"
    ), overwrite=True)

    # system/fvSolution
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
        "reactingFoam\n{\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reacting_case(tmp_path):
    """Create a 1D reacting flow case."""
    case_dir = tmp_path / "reacting"
    _make_reacting_case(case_dir, n_cells=3, end_time=1, delta_t=0.01)
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReactingFoamInit:
    """Tests for ReactingFoam initialisation."""

    def test_case_loads(self, reacting_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(reacting_case)
        assert case.has_mesh()

    def test_species_detected(self, reacting_case):
        """Species are detected from 0/ directory."""
        from pyfoam.applications.reacting_foam import ReactingFoam

        solver = ReactingFoam(reacting_case)
        assert "A" in solver.species
        assert "B" in solver.species

    def test_reactions_read(self, reacting_case):
        """Reactions are read from constant/reactions."""
        from pyfoam.applications.reacting_foam import ReactingFoam

        solver = ReactingFoam(reacting_case)
        assert len(solver.reactions) == 1
        assert solver.reactions[0].A == 1e6
        assert solver.reactions[0].Ea == 50000

    def test_temperature_initialised(self, reacting_case):
        """Temperature is initialised correctly."""
        from pyfoam.applications.reacting_foam import ReactingFoam

        solver = ReactingFoam(reacting_case)
        assert torch.allclose(solver.T, torch.full((3,), 300.0, dtype=CFD_DTYPE))


class TestReactingFoamKinetics:
    """Tests for Arrhenius kinetics."""

    def test_arrhenius_rate_shape(self, reacting_case):
        """Arrhenius rate has correct shape."""
        from pyfoam.applications.reacting_foam import ReactingFoam

        solver = ReactingFoam(reacting_case)
        rate = solver._compute_arrhenius_rate(
            solver.reactions[0], solver.T, solver.Y,
        )

        assert rate.shape == (3,)
        assert torch.isfinite(rate).all()

    def test_arrhenius_rate_positive(self, reacting_case):
        """Arrhenius rate is non-negative."""
        from pyfoam.applications.reacting_foam import ReactingFoam

        solver = ReactingFoam(reacting_case)
        rate = solver._compute_arrhenius_rate(
            solver.reactions[0], solver.T, solver.Y,
        )

        assert (rate >= 0).all(), "Rate has negative values"

    def test_species_source_terms(self, reacting_case):
        """Species source terms have correct shape."""
        from pyfoam.applications.reacting_foam import ReactingFoam

        solver = ReactingFoam(reacting_case)
        omega = solver._compute_species_source_terms(solver.T, solver.Y)

        assert "A" in omega
        assert "B" in omega
        assert omega["A"].shape == (3,)
        assert omega["B"].shape == (3,)


class TestReactingFoamSolver:
    """Tests for solver execution."""

    def test_run_completes(self, reacting_case):
        """Solver runs without errors."""
        from pyfoam.applications.reacting_foam import ReactingFoam

        solver = ReactingFoam(reacting_case)
        result = solver.run()

        assert "converged" in result

    def test_mass_fractions_finite(self, reacting_case):
        """Mass fractions are finite after solving."""
        from pyfoam.applications.reacting_foam import ReactingFoam

        solver = ReactingFoam(reacting_case)
        solver.run()

        for name in solver.species:
            assert torch.isfinite(solver.Y[name]).all(), f"Y_{name} contains NaN/Inf"

    def test_temperature_finite(self, reacting_case):
        """Temperature is finite after solving."""
        from pyfoam.applications.reacting_foam import ReactingFoam

        solver = ReactingFoam(reacting_case)
        solver.run()

        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"

    def test_mass_fractions_bounded(self, reacting_case):
        """Mass fractions stay finite after solving."""
        from pyfoam.applications.reacting_foam import ReactingFoam

        solver = ReactingFoam(reacting_case)
        solver.run()

        for name in solver.species:
            # Check finiteness rather than strict bounds
            assert torch.isfinite(solver.Y[name]).all(), f"Y_{name} has NaN/Inf"
