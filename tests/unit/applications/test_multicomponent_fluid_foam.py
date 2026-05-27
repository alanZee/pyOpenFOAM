"""
Unit tests for MulticomponentFluidFoam — multi-species compressible solver.

Tests cover:
- Case loading and field initialisation
- Species detection and normalisation
- Mixture property computation
- Solver execution
- Mass fraction boundedness
- Field output
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# 网格生成辅助函数
# ---------------------------------------------------------------------------

def _make_multicomponent_case(
    case_dir: Path,
    n_cells: int = 3,
    L: float = 1.0,
    end_time: float = 3e-5,
    delta_t: float = 1e-5,
    n_outer_correctors: int = 3,
    n_correctors: int = 2,
    p_init: float = 101325.0,
    T_init: float = 300.0,
    Y_N2: float = 0.76,
    Y_O2: float = 0.24,
) -> None:
    """Write a 1D multicomponent channel case."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = L / n_cells
    dy = 0.1
    dz = 0.1

    # 节点
    points = []
    for i in range(n_cells + 1):
        x = i * dx
        points.append((x, 0.0, 0.0))
        points.append((x, dy, 0.0))
        points.append((x, dy, dz))
        points.append((x, 0.0, dz))

    n_points = len(points)

    # 面
    faces = []
    owner = []
    neighbour = []

    for i in range(n_cells - 1):
        faces.append((4, i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3))
        owner.append(i)
        neighbour.append(i + 1)

    n_internal = len(neighbour)

    # 入口
    inlet_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    # 出口
    outlet_start = inlet_start + 1
    level = n_cells
    faces.append((4, level * 4 + 0, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

    # 空 patches
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

    # 写网格文件
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

    # thermophysicalProperties（含物种信息）
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant",
        object="thermophysicalProperties",
    )
    tp_body = (
        "species\n{\n"
        "    N2\n    {\n"
        "        W       0.028;\n"
        "        Cp      1040;\n"
        "        mu      1.76e-5;\n"
        "        kappa   0.0258;\n"
        "        D       2.1e-5;\n"
        "    }\n"
        "    O2\n    {\n"
        "        W       0.032;\n"
        "        Cp      918;\n"
        "        mu      2.06e-5;\n"
        "        kappa   0.0265;\n"
        "        D       2.1e-5;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(
        case_dir / "constant" / "thermophysicalProperties", tp_header,
        tp_body, overwrite=True,
    )

    # 0/ 目录
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

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
        f"internalField   uniform {p_init};\n\n"
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

    # 0/YN2
    yn2_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="YN2",
    )
    write_foam_file(zero_dir / "YN2", yn2_header, (
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        f"internalField   uniform {Y_N2};\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {Y_N2};\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # 0/YO2
    yo2_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="YO2",
    )
    write_foam_file(zero_dir / "YO2", yo2_header, (
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        f"internalField   uniform {Y_O2};\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {Y_O2};\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # system
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    write_foam_file(sys_dir / "controlDict", cd_header, (
        "application     multicomponentFluidFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time:g};\n"
        f"deltaT          {delta_t:g};\n"
        "writeControl    timeStep;\n"
        f"writeInterval   100;\n"
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
        "    T\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "    Y\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "PIMPLE\n{\n"
        f"    nOuterCorrectors    {n_outer_correctors};\n"
        f"    nCorrectors         {n_correctors};\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "    maxOuterIterations  100;\n"
        "    relaxationFactors\n    {\n"
        "        p               0.3;\n"
        "        U               0.7;\n"
        "        T               1.0;\n"
        "        Y               1.0;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mc_case(tmp_path):
    """Create a 1D multicomponent case."""
    case_dir = tmp_path / "multicomponent"
    _make_multicomponent_case(
        case_dir,
        n_cells=3,
        end_time=3e-5,
        delta_t=1e-5,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests — 初始化
# ---------------------------------------------------------------------------

class TestMulticomponentFluidFoamInit:
    """Tests for MulticomponentFluidFoam initialisation."""

    def test_case_loads(self, mc_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(mc_case)
        assert case.has_mesh()

    def test_species_detected(self, mc_case):
        """Species are detected from 0/ directory."""
        from pyfoam.applications.multicomponent_fluid_foam import (
            MulticomponentFluidFoam,
        )

        solver = MulticomponentFluidFoam(mc_case)
        assert "N2" in solver.species
        assert "O2" in solver.species

    def test_mass_fractions_sum_to_one(self, mc_case):
        """Mass fractions sum to 1 at initialisation."""
        from pyfoam.applications.multicomponent_fluid_foam import (
            MulticomponentFluidFoam,
        )

        solver = MulticomponentFluidFoam(mc_case)
        Y_sum = sum(solver.Y.values())
        assert torch.allclose(Y_sum, torch.ones_like(Y_sum), atol=1e-6)

    def test_species_properties_loaded(self, mc_case):
        """Species properties are loaded from thermophysicalProperties."""
        from pyfoam.applications.multicomponent_fluid_foam import (
            MulticomponentFluidFoam,
        )

        solver = MulticomponentFluidFoam(mc_case)
        assert "N2" in solver.species_props
        assert "O2" in solver.species_props
        assert abs(solver.species_props["N2"].W - 0.028) < 1e-6
        assert abs(solver.species_props["O2"].W - 0.032) < 1e-6


# ---------------------------------------------------------------------------
# Tests — 混合物物性
# ---------------------------------------------------------------------------

class TestMulticomponentMixtureProperties:
    """Tests for mixture property computation."""

    def test_mixture_R(self, mc_case):
        """Mixture gas constant is mass-fraction-weighted average."""
        from pyfoam.applications.multicomponent_fluid_foam import (
            MulticomponentFluidFoam,
        )

        solver = MulticomponentFluidFoam(mc_case)
        R_mix = solver._mixture_R(solver.Y)

        R_N2 = 8.314 / 0.028
        R_O2 = 8.314 / 0.032
        expected = 0.76 * R_N2 + 0.24 * R_O2

        assert torch.allclose(
            R_mix, torch.full_like(R_mix, expected), rtol=1e-3
        )

    def test_mixture_Cp(self, mc_case):
        """Mixture Cp is mass-fraction-weighted average."""
        from pyfoam.applications.multicomponent_fluid_foam import (
            MulticomponentFluidFoam,
        )

        solver = MulticomponentFluidFoam(mc_case)
        Cp_mix = solver._mixture_Cp(solver.Y)

        expected = 0.76 * 1040.0 + 0.24 * 918.0
        assert torch.allclose(
            Cp_mix, torch.full_like(Cp_mix, expected), rtol=1e-3
        )

    def test_mixture_rho(self, mc_case):
        """Mixture density uses mixture R."""
        from pyfoam.applications.multicomponent_fluid_foam import (
            MulticomponentFluidFoam,
        )

        solver = MulticomponentFluidFoam(mc_case)
        rho = solver.rho
        assert rho.shape == (solver.mesh.n_cells,)
        assert (rho > 0).all(), "Density non-positive"


# ---------------------------------------------------------------------------
# Tests — 求解器执行
# ---------------------------------------------------------------------------

class TestMulticomponentFluidFoamRun:
    """Tests for solver execution."""

    def test_run_completes(self, mc_case):
        """Solver runs without errors."""
        from pyfoam.applications.multicomponent_fluid_foam import (
            MulticomponentFluidFoam,
        )

        solver = MulticomponentFluidFoam(mc_case)
        conv = solver.run()

        assert conv is not None
        assert conv.continuity_error >= 0

    def test_mass_fractions_bounded(self, mc_case):
        """Mass fractions stay in [0, 1] after solving."""
        from pyfoam.applications.multicomponent_fluid_foam import (
            MulticomponentFluidFoam,
        )

        solver = MulticomponentFluidFoam(mc_case)
        solver.run()

        for name in solver.species:
            assert (solver.Y[name] >= 0.0).all(), f"Y_{name} < 0"
            assert (solver.Y[name] <= 1.0).all(), f"Y_{name} > 1"

    def test_fields_finite(self, mc_case):
        """All fields are finite after solving."""
        from pyfoam.applications.multicomponent_fluid_foam import (
            MulticomponentFluidFoam,
        )

        solver = MulticomponentFluidFoam(mc_case)
        solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"
        for name in solver.species:
            assert torch.isfinite(solver.Y[name]).all(), f"Y_{name} has NaN/Inf"
