"""
Unit tests for MultiphaseReactingFoam — multiphase reacting solver.

Tests cover:
- Phase configuration and field initialisation
- Species detection and reaction mechanism reading
- Arrhenius kinetics computation
- Species source terms
- Heat release computation
- Multiphase iteration
- Time-stepping run
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------

def _make_multiphase_reacting_case(
    case_dir: Path,
    n_cells: int = 3,
    L: float = 1.0,
    end_time: int = 1,
    delta_t: float = 0.01,
    write_interval: int = 100,
) -> None:
    """Write a 1D multiphase reacting case."""
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
        "    A               1e4;\n"
        "    beta            0.0;\n"
        "    Ea              40000;\n"
        "    heatOfReaction  1000;\n"
        "    reactants\n    {\n"
        "        A           1;\n"
        "    }\n"
        "    products\n    {\n"
        "        B           1;\n"
        "    }\n"
        "}\n",
        overwrite=True,
    )

    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    # YA
    ya_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="YA",
    )
    write_foam_file(zero_dir / "YA", ya_header, (
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        "internalField   uniform 1;\n\n"
        "boundaryField\n{\n"
        "    inlet    { type fixedValue; value uniform 1; }\n"
        "    outlet   { type zeroGradient; }\n"
        "    walls    { type empty; }\n"
        "}\n"
    ), overwrite=True)

    # T
    t_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    write_foam_file(zero_dir / "T", t_header, (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        "internalField   uniform 300;\n\n"
        "boundaryField\n{\n"
        "    inlet    { type fixedValue; value uniform 300; }\n"
        "    outlet   { type zeroGradient; }\n"
        "    walls    { type empty; }\n"
        "}\n"
    ), overwrite=True)

    # U
    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    write_foam_file(zero_dir / "U", u_header, (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (1 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet    { type fixedValue; value uniform (1 0 0); }\n"
        "    outlet   { type zeroGradient; }\n"
        "    walls    { type empty; }\n"
        "}\n"
    ), overwrite=True)

    # p
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    write_foam_file(zero_dir / "p", p_header, (
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        "internalField   uniform 101325;\n\n"
        "boundaryField\n{\n"
        "    inlet    { type zeroGradient; }\n"
        "    outlet   { type zeroGradient; }\n"
        "    walls    { type empty; }\n"
        "}\n"
    ), overwrite=True)

    # alpha_gas
    alpha_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="alpha_gas",
    )
    write_foam_file(zero_dir / "alpha_gas", alpha_header, (
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        "internalField   uniform 0.3;\n\n"
        "boundaryField\n{\n"
        "    inlet    { type fixedValue; value uniform 0.3; }\n"
        "    outlet   { type zeroGradient; }\n"
        "    walls    { type empty; }\n"
        "}\n"
    ), overwrite=True)

    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    write_foam_file(sys_dir / "controlDict", cd_header, (
        "application     multiphaseReactingFoam;\n"
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
        "divSchemes\n{\n    default         Gauss linear;\n}\n"
    ), overwrite=True)

    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, (
        "PIMPLE\n{\n"
        "    nOuterCorrectors    2;\n"
        "    convergenceTolerance 1e-4;\n"
        "    maxOuterIterations  10;\n"
        "}\n\n"
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
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def multiphase_reacting_case(tmp_path):
    """Create a 1D multiphase reacting case."""
    case_dir = tmp_path / "mpr"
    _make_multiphase_reacting_case(case_dir, n_cells=3, end_time=1, delta_t=0.01)
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiphaseReactingFoamInit:
    """Tests for MultiphaseReactingFoam initialisation."""

    def test_case_loads(self, multiphase_reacting_case):
        from pyfoam.io.case import Case
        case = Case(multiphase_reacting_case)
        assert case.has_mesh()

    def test_phases_configured(self, multiphase_reacting_case):
        from pyfoam.applications.multiphase_reacting_foam import MultiphaseReactingFoam
        phases = [
            {"name": "gas", "rho": 1.225, "mu": 1.8e-5},
            {"name": "liquid", "rho": 1000.0, "mu": 1e-3},
        ]
        solver = MultiphaseReactingFoam(multiphase_reacting_case, phases=phases)
        assert solver.n_phases == 2
        assert solver.phase_names == ["gas", "liquid"]

    def test_species_detected(self, multiphase_reacting_case):
        from pyfoam.applications.multiphase_reacting_foam import MultiphaseReactingFoam
        phases = [
            {"name": "gas", "rho": 1.225, "mu": 1.8e-5},
            {"name": "liquid", "rho": 1000.0, "mu": 1e-3},
        ]
        solver = MultiphaseReactingFoam(multiphase_reacting_case, phases=phases)
        assert "A" in solver.species

    def test_reactions_read(self, multiphase_reacting_case):
        from pyfoam.applications.multiphase_reacting_foam import MultiphaseReactingFoam
        phases = [
            {"name": "gas", "rho": 1.225, "mu": 1.8e-5},
            {"name": "liquid", "rho": 1000.0, "mu": 1e-3},
        ]
        solver = MultiphaseReactingFoam(multiphase_reacting_case, phases=phases)
        assert len(solver.reactions) == 1
        assert solver.reactions[0].A == 1e4

    def test_alphas_sum_to_one(self, multiphase_reacting_case):
        from pyfoam.applications.multiphase_reacting_foam import MultiphaseReactingFoam
        phases = [
            {"name": "gas", "rho": 1.225, "mu": 1.8e-5},
            {"name": "liquid", "rho": 1000.0, "mu": 1e-3},
        ]
        solver = MultiphaseReactingFoam(multiphase_reacting_case, phases=phases)
        alpha_sum = sum(solver.alphas)
        assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-6)


class TestMultiphaseReactingFoamKinetics:
    """Tests for reaction kinetics."""

    def test_arrhenius_rate_shape(self, multiphase_reacting_case):
        from pyfoam.applications.multiphase_reacting_foam import MultiphaseReactingFoam
        phases = [
            {"name": "gas", "rho": 1.225, "mu": 1.8e-5},
            {"name": "liquid", "rho": 1000.0, "mu": 1e-3},
        ]
        solver = MultiphaseReactingFoam(multiphase_reacting_case, phases=phases)
        rate = solver._compute_arrhenius_rate(
            solver.reactions[0], solver.T, solver.Y,
        )
        assert rate.shape == (3,)
        assert torch.isfinite(rate).all()

    def test_species_source_terms(self, multiphase_reacting_case):
        from pyfoam.applications.multiphase_reacting_foam import MultiphaseReactingFoam
        phases = [
            {"name": "gas", "rho": 1.225, "mu": 1.8e-5},
            {"name": "liquid", "rho": 1000.0, "mu": 1e-3},
        ]
        solver = MultiphaseReactingFoam(multiphase_reacting_case, phases=phases)
        omega = solver._compute_species_source_terms(solver.T, solver.Y)
        assert "A" in omega
        assert omega["A"].shape == (3,)

    def test_heat_release(self, multiphase_reacting_case):
        from pyfoam.applications.multiphase_reacting_foam import MultiphaseReactingFoam
        phases = [
            {"name": "gas", "rho": 1.225, "mu": 1.8e-5},
            {"name": "liquid", "rho": 1000.0, "mu": 1e-3},
        ]
        solver = MultiphaseReactingFoam(multiphase_reacting_case, phases=phases)
        heat = solver._compute_heat_release(solver.T, solver.Y)
        assert heat.shape == (3,)
        assert torch.isfinite(heat).all()


class TestMultiphaseReactingFoamSolver:
    """Tests for solver execution."""

    def test_run_completes(self, multiphase_reacting_case):
        from pyfoam.applications.multiphase_reacting_foam import MultiphaseReactingFoam
        phases = [
            {"name": "gas", "rho": 1.225, "mu": 1.8e-5},
            {"name": "liquid", "rho": 1000.0, "mu": 1e-3},
        ]
        solver = MultiphaseReactingFoam(multiphase_reacting_case, phases=phases)
        result = solver.run()
        assert result is not None

    def test_fields_finite_after_run(self, multiphase_reacting_case):
        from pyfoam.applications.multiphase_reacting_foam import MultiphaseReactingFoam
        phases = [
            {"name": "gas", "rho": 1.225, "mu": 1.8e-5},
            {"name": "liquid", "rho": 1000.0, "mu": 1e-3},
        ]
        solver = MultiphaseReactingFoam(multiphase_reacting_case, phases=phases)
        solver.run()
        for name in solver.species:
            assert torch.isfinite(solver.Y[name]).all()
        assert torch.isfinite(solver.T).all()
