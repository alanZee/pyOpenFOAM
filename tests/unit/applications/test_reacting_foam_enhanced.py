"""
Unit tests for ReactingFoamEnhanced — enhanced reacting flow solver.

Tests cover:
- Case loading and field initialisation
- Enhanced reaction mechanism (third-body support)
- Temperature-dependent diffusivity
- RK2 time integration
- Mass-fraction conservation diagnostics
- Solver execution
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper (reuses same structure as reactingFoam tests)
# ---------------------------------------------------------------------------

def _make_enhanced_reacting_case(
    case_dir: Path,
    n_cells: int = 5,
    L: float = 1.0,
    end_time: int = 10,
    delta_t: float = 0.1,
    write_interval: int = 100,
    T_init: float = 300.0,
    Y_A_init: float = 0.8,
    Y_B_init: float = 0.2,
) -> None:
    """Write a 1D enhanced reacting flow case with 3 species."""
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

    # Empty patches (4 per cell)
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

    # thermophysicalProperties (with 3 species + diffusivity)
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="thermophysicalProperties",
    )
    write_foam_file(
        case_dir / "constant" / "thermophysicalProperties", tp_header,
        "R               8.314;\n"
        "Cp              1005;\n"
        "species\n{\n"
        "    A   28.0;\n"
        "    B   32.0;\n"
        "    C   44.0;\n"
        "}\n"
        "diffusivity\n{\n"
        "    A   1.5e-5;\n"
        "    B   1.2e-5;\n"
        "    C   1.0e-5;\n"
        "}\n",
        overwrite=True,
    )

    # reactions (two reactions: A->B and B+C->products)
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
        "}\n"
        "reaction2\n{\n"
        "    A               5e4;\n"
        "    beta            0.5;\n"
        "    Ea              30000;\n"
        "    reactants\n    {\n"
        "        B           1;\n"
        "        C           1;\n"
        "    }\n"
        "    products\n    {\n"
        "        A           0.5;\n"
        "    }\n"
        "}\n",
        overwrite=True,
    )

    # 0/ directory with 3 species
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    for sp_name, sp_init in [("A", Y_A_init), ("B", Y_B_init), ("C", 0.0)]:
        header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="volScalarField", location="0", object=f"Y{sp_name}",
        )
        write_foam_file(zero_dir / f"Y{sp_name}", header, (
            "dimensions      [0 0 0 0 0 0 0];\n\n"
            f"internalField   uniform {sp_init};\n\n"
            "boundaryField\n{\n"
            "    inlet\n    {\n"
            "        type            fixedValue;\n"
            f"        value           uniform {sp_init};\n"
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
def enhanced_case(tmp_path):
    """Create a 1D enhanced reacting flow case (3 species, 2 reactions)."""
    case_dir = tmp_path / "enhanced_reacting"
    _make_enhanced_reacting_case(case_dir, n_cells=3, end_time=1, delta_t=0.01)
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReactingFoamEnhancedInit:
    """Tests for ReactingFoamEnhanced initialisation."""

    def test_case_loads(self, enhanced_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(enhanced_case)
        assert case.has_mesh()

    def test_species_detected(self, enhanced_case):
        """Three species are detected from 0/ directory."""
        from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced

        solver = ReactingFoamEnhanced(enhanced_case)
        assert "A" in solver.species
        assert "B" in solver.species
        assert "C" in solver.species
        assert len(solver.species) == 3

    def test_reactions_read(self, enhanced_case):
        """Two reactions are read from constant/reactions."""
        from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced

        solver = ReactingFoamEnhanced(enhanced_case)
        assert len(solver.enhanced_reactions) == 2

    def test_molecular_weights(self, enhanced_case):
        """Molecular weights are read correctly."""
        from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced

        solver = ReactingFoamEnhanced(enhanced_case)
        assert solver.W.get("A") == 28.0
        assert solver.W.get("B") == 32.0
        assert solver.W.get("C") == 44.0

    def test_initial_mass_conservation(self, enhanced_case):
        """Initial mass conservation error is zero."""
        from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced

        solver = ReactingFoamEnhanced(enhanced_case)
        errors = solver.check_mass_conservation()
        for name, err in errors.items():
            assert err < 1e-10, f"Initial mass error for {name}: {err}"


class TestReactingFoamEnhancedDiffusivity:
    """Tests for temperature-dependent diffusivity."""

    def test_diffusivity_shape(self, enhanced_case):
        """Diffusivity tensor has correct shape."""
        from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced

        solver = ReactingFoamEnhanced(enhanced_case)
        T = torch.full((3,), 300.0, dtype=CFD_DTYPE)
        D = solver._compute_diffusivity("A", T)

        assert D.shape == (3,)
        assert torch.isfinite(D).all()

    def test_diffusivity_increases_with_temperature(self, enhanced_case):
        """Diffusivity increases with temperature (power-law model)."""
        from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced

        solver = ReactingFoamEnhanced(enhanced_case)
        T_low = torch.full((3,), 300.0, dtype=CFD_DTYPE)
        T_high = torch.full((3,), 600.0, dtype=CFD_DTYPE)

        D_low = solver._compute_diffusivity("A", T_low)
        D_high = solver._compute_diffusivity("A", T_high)

        assert (D_high > D_low).all(), "Diffusivity should increase with T"


class TestReactingFoamEnhancedKinetics:
    """Tests for enhanced kinetics with multiple reactions."""

    def test_enhanced_rate_shape(self, enhanced_case):
        """Enhanced reaction rate has correct shape."""
        from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced

        solver = ReactingFoamEnhanced(enhanced_case)
        rate = solver._compute_enhanced_rate(
            solver.enhanced_reactions[0], solver.T, solver.Y,
        )
        assert rate.shape == (3,)
        assert torch.isfinite(rate).all()

    def test_species_source_terms(self, enhanced_case):
        """Source terms from multiple reactions have correct shape."""
        from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced

        solver = ReactingFoamEnhanced(enhanced_case)
        omega = solver._compute_species_source_terms(solver.T, solver.Y)

        assert "A" in omega
        assert "B" in omega
        assert "C" in omega
        for name in solver.species:
            assert omega[name].shape == (3,)
            assert torch.isfinite(omega[name]).all()


class TestReactingFoamEnhancedSolver:
    """Tests for solver execution."""

    def test_run_euler_completes(self, enhanced_case):
        """Euler integration completes without errors."""
        from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced

        solver = ReactingFoamEnhanced(enhanced_case, integration="euler")
        result = solver.run()

        assert "converged" in result
        assert "mass_conservation_error" in result

    def test_run_rk2_completes(self, enhanced_case):
        """RK2 integration completes without errors."""
        from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced

        solver = ReactingFoamEnhanced(enhanced_case, integration="rk2")
        result = solver.run()

        assert "converged" in result
        assert "mass_conservation_error" in result

    def test_fields_finite_after_solve(self, enhanced_case):
        """All fields remain finite after solving."""
        from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced

        solver = ReactingFoamEnhanced(enhanced_case, integration="euler")
        solver.run()

        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"
        for name in solver.species:
            assert torch.isfinite(solver.Y[name]).all(), f"Y_{name} contains NaN/Inf"
