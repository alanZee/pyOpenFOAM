"""
Unit tests for ReactingMultiphaseFoam — reacting multiphase solver.

Tests cover:
- Phase initialisation and species detection
- Reaction mechanism reading
- Arrhenius kinetics computation
- Species source terms
- Run loop completion
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper (reuse from test_reacting_foam pattern)
# ---------------------------------------------------------------------------


def _make_reacting_multiphase_case(
    case_dir: Path,
    n_cells: int = 5,
    L: float = 1.0,
    end_time: int = 10,
    delta_t: float = 0.1,
    write_interval: int = 100,
) -> None:
    """Write a 1D multiphase reacting case."""
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

    # 0/U
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

    # 0/alpha_gas
    ag_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="alpha_gas",
    )
    write_foam_file(zero_dir / "alpha_gas", ag_header, (
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        "internalField   uniform 0.3;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform 0.3;\n"
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

    # system/controlDict
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    write_foam_file(sys_dir / "controlDict", cd_header, (
        "application     reactingMultiphaseEulerFoam;\n"
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
        "divSchemes\n{\n    default         Gauss linear;\n}\n"
    ), overwrite=True)

    # system/fvSolution
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, (
        "PIMPLE\n{\n"
        "    nOuterCorrectors     2;\n"
        "    convergenceTolerance 1e-4;\n"
        "    maxOuterIterations   10;\n"
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reacting_mp_case(tmp_path):
    """Create a 1D reacting multiphase case."""
    case_dir = tmp_path / "reactingMp"
    _make_reacting_multiphase_case(case_dir, n_cells=3, end_time=1, delta_t=0.01)
    return case_dir


PHASES = [
    {"name": "gas", "rho": 1.225, "mu": 1.8e-5, "species": ["A", "B"]},
    {"name": "liquid", "rho": 1000.0, "mu": 1e-3, "species": []},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReactingMultiphaseFoamInit:
    """Tests for ReactingMultiphaseFoam initialisation."""

    def test_case_loads(self, reacting_mp_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(reacting_mp_case)
        assert case.has_mesh()

    def test_n_phases(self, reacting_mp_case):
        """Number of phases is correct."""
        from pyfoam.applications.reacting_multiphase_foam import ReactingMultiphaseFoam

        solver = ReactingMultiphaseFoam(reacting_mp_case, phases=PHASES)
        assert solver.n_phases == 2

    def test_phase_names(self, reacting_mp_case):
        """Phase names are stored correctly."""
        from pyfoam.applications.reacting_multiphase_foam import ReactingMultiphaseFoam

        solver = ReactingMultiphaseFoam(reacting_mp_case, phases=PHASES)
        assert solver.phase_names == ["gas", "liquid"]

    def test_species_detected(self, reacting_mp_case):
        """Species are detected from phase definitions."""
        from pyfoam.applications.reacting_multiphase_foam import ReactingMultiphaseFoam

        solver = ReactingMultiphaseFoam(reacting_mp_case, phases=PHASES)
        assert "A" in solver.species["gas"]
        assert "B" in solver.species["gas"]

    def test_reactions_read(self, reacting_mp_case):
        """Reactions are read from constant/reactions."""
        from pyfoam.applications.reacting_multiphase_foam import ReactingMultiphaseFoam

        solver = ReactingMultiphaseFoam(reacting_mp_case, phases=PHASES)
        assert len(solver.reactions) == 1
        assert solver.reactions[0].A == 1e6


class TestReactingMultiphaseFoamKinetics:
    """Tests for Arrhenius kinetics in multiphase context."""

    def test_arrhenius_rate_shape(self, reacting_mp_case):
        """Arrhenius rate has correct shape."""
        from pyfoam.applications.reacting_multiphase_foam import ReactingMultiphaseFoam

        solver = ReactingMultiphaseFoam(reacting_mp_case, phases=PHASES)
        Y_flat = solver._get_flattened_species()
        rate = solver._compute_arrhenius_rate(
            solver.reactions[0], solver.T, Y_flat,
        )
        assert rate.shape == (3,)
        assert torch.isfinite(rate).all()

    def test_species_source_terms(self, reacting_mp_case):
        """Species source terms have correct shape."""
        from pyfoam.applications.reacting_multiphase_foam import ReactingMultiphaseFoam

        solver = ReactingMultiphaseFoam(reacting_mp_case, phases=PHASES)
        Y_flat = solver._get_flattened_species()
        omega = solver._compute_species_source_terms(solver.T, Y_flat)

        assert "A" in omega
        assert "B" in omega
        assert omega["A"].shape == (3,)

    def test_arrhenius_rate_positive(self, reacting_mp_case):
        """Arrhenius rate is non-negative."""
        from pyfoam.applications.reacting_multiphase_foam import ReactingMultiphaseFoam

        solver = ReactingMultiphaseFoam(reacting_mp_case, phases=PHASES)
        Y_flat = solver._get_flattened_species()
        rate = solver._compute_arrhenius_rate(
            solver.reactions[0], solver.T, Y_flat,
        )
        assert (rate >= 0).all()


class TestReactingMultiphaseFoamSolver:
    """Tests for solver execution."""

    def test_run_completes(self, reacting_mp_case):
        """Solver runs without errors."""
        from pyfoam.applications.reacting_multiphase_foam import ReactingMultiphaseFoam

        solver = ReactingMultiphaseFoam(reacting_mp_case, phases=PHASES)
        result = solver.run()
        assert result is not None

    def test_volume_fractions_bounded(self, reacting_mp_case):
        """Volume fractions are bounded after solving."""
        from pyfoam.applications.reacting_multiphase_foam import ReactingMultiphaseFoam

        solver = ReactingMultiphaseFoam(reacting_mp_case, phases=PHASES)
        solver.run()

        for a in solver.alphas:
            assert torch.isfinite(a).all()
            assert (a >= -1e-6).all()
            assert (a <= 1.0 + 1e-6).all()

    def test_species_finite(self, reacting_mp_case):
        """Species mass fractions are finite after solving."""
        from pyfoam.applications.reacting_multiphase_foam import ReactingMultiphaseFoam

        solver = ReactingMultiphaseFoam(reacting_mp_case, phases=PHASES)
        solver.run()

        for phase_name, species_dict in solver.Y.items():
            for sp_name, sp_tensor in species_dict.items():
                assert torch.isfinite(sp_tensor).all(), (
                    f"Y_{phase_name}_{sp_name} contains NaN/Inf"
                )
