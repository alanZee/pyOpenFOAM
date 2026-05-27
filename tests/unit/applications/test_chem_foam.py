"""
Unit tests for ChemFoam — 0D chemistry solver.

Tests cover:
- Case loading and field initialisation
- Reaction mechanism reading
- Arrhenius kinetics computation
- Species source terms
- Temperature evolution
- Forward Euler time-stepping
- Field writing
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# 网格 + 算例生成
# ---------------------------------------------------------------------------

def _make_chem_case(
    case_dir: Path,
    n_cells: int = 1,
    end_time: int = 1,
    delta_t: float = 0.001,
    write_interval: int = 100,
    T_init: float = 1000.0,
    Y_A_init: float = 1.0,
) -> None:
    """Write a 1D case for chemFoam (single or few cells)."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / n_cells
    dy, dz = 0.1, 0.1

    # ---- 网格点 ----
    points = []
    for i in range(n_cells + 1):
        x = i * dx
        points.extend([(x, 0.0, 0.0), (x, dy, 0.0), (x, dy, dz), (x, 0.0, dz)])

    n_points = len(points)
    faces, owner, neighbour = [], [], []

    # 内部面
    for i in range(n_cells - 1):
        faces.append((4, i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3))
        owner.append(i)
        neighbour.append(i + 1)
    n_internal = len(neighbour)

    # 入口面
    inlet_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    # 出口面
    outlet_start = inlet_start + 1
    level = n_cells
    faces.append((4, level * 4, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

    # Empty patches
    empty_start = outlet_start + 1
    for i in range(n_cells):
        faces.append((4, i * 4, (i + 1) * 4, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4, i * 4 + 1, (i + 1) * 4 + 1, (i + 1) * 4))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 3, (i + 1) * 4 + 3, (i + 1) * 4 + 2, i * 4 + 2))
        owner.append(i)

    n_faces = len(faces)
    n_empty = 4 * n_cells

    # ---- 写网格文件 ----
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    hb = FoamFileHeader(version="2.0", format=FileFormat.ASCII, location="constant/polyMesh")

    def _header(class_name: str, obj: str) -> FoamFileHeader:
        return FoamFileHeader(**{**hb.__dict__, "class_name": class_name, "object": obj})

    lines = [str(n_points), "("]
    for p in points:
        lines.append(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", _header("vectorField", "points"), "\n".join(lines), overwrite=True)

    lines = [str(n_faces), "("]
    for face in faces:
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{face[0]}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", _header("faceList", "faces"), "\n".join(lines), overwrite=True)

    lines = [str(n_faces), "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", _header("labelList", "owner"), "\n".join(lines), overwrite=True)

    lines = [str(n_internal), "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", _header("labelList", "neighbour"), "\n".join(lines), overwrite=True)

    lines = ["2", "("]
    lines += [
        "    inlet", "    {", "        type            patch;",
        f"        nFaces          1;", f"        startFace       {inlet_start};", "    }",
        "    outlet", "    {", "        type            patch;",
        f"        nFaces          1;", f"        startFace       {outlet_start};", "    }",
        "    walls", "    {", "        type            empty;",
        f"        nFaces          {n_empty};", f"        startFace       {empty_start};", "    }",
        ")",
    ]
    write_foam_file(mesh_dir / "boundary", _header("polyBoundaryMesh", "boundary"), "\n".join(lines), overwrite=True)

    # ---- thermophysicalProperties ----
    tp_header = FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                               location="constant", object="thermophysicalProperties")
    write_foam_file(
        case_dir / "constant" / "thermophysicalProperties", tp_header,
        "R               8.314;\n"
        "Cp              1005;\n"
        "species\n{\n    A   1.0;\n    B   1.0;\n}\n",
        overwrite=True,
    )

    # ---- reactions ----
    rxn_header = FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                                location="constant", object="reactions")
    write_foam_file(
        case_dir / "constant" / "reactions", rxn_header,
        "reaction1\n{\n"
        "    A               1e6;\n"
        "    beta            0.0;\n"
        "    Ea              50000;\n"
        "    reactants\n    {\n        A           1;\n    }\n"
        "    products\n    {\n        B           1;\n    }\n"
        "}\n",
        overwrite=True,
    )

    # ---- 0/ 目录 ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    def _field_body(dims: str, value: str, obj_name: str) -> str:
        return (
            f"dimensions      {dims};\n\n"
            f"internalField   uniform {value};\n\n"
            "boundaryField\n{\n"
            "    inlet\n    {\n        type            fixedValue;\n"
            f"        value           uniform {value};\n    }}\n"
            "    outlet\n    {\n        type            zeroGradient;\n    }\n"
            "    walls\n    {\n        type            empty;\n    }\n"
            "}\n"
        )

    # YA
    write_foam_file(zero_dir / "YA",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="YA"),
        _field_body("[0 0 0 0 0 0 0]", str(Y_A_init), "YA"), overwrite=True)

    # YB
    write_foam_file(zero_dir / "YB",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="YB"),
        _field_body("[0 0 0 0 0 0 0]", "0", "YB"), overwrite=True)

    # T
    write_foam_file(zero_dir / "T",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="T"),
        _field_body("[0 0 0 1 0 0 0]", str(T_init), "T"), overwrite=True)

    # U (SolverBase 需要)
    write_foam_file(zero_dir / "U",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volVectorField",
                       location="0", object="U"),
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    # p (SolverBase 需要)
    write_foam_file(zero_dir / "p",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="p"),
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        "internalField   uniform 101325;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            zeroGradient;\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    # ---- system/ ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    write_foam_file(sys_dir / "controlDict",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="controlDict"),
        "application     chemFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        f"writeInterval   {write_interval};\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSchemes",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSchemes"),
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSolution",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSolution"),
        "chemFoam\n{\n    convergenceTolerance 1e-6;\n}\n",
        overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chem_case(tmp_path):
    """Create a 1-cell chemistry case."""
    case_dir = tmp_path / "chem"
    _make_chem_case(case_dir, n_cells=1, end_time=1, delta_t=0.001)
    return case_dir


@pytest.fixture
def multi_cell_chem_case(tmp_path):
    """Create a 3-cell chemistry case."""
    case_dir = tmp_path / "chem3"
    _make_chem_case(case_dir, n_cells=3, end_time=1, delta_t=0.001)
    return case_dir


# ---------------------------------------------------------------------------
# 初始化测试
# ---------------------------------------------------------------------------

class TestChemFoamInit:
    """ChemFoam 初始化测试。"""

    def test_case_loads(self, chem_case):
        """算例目录可读取。"""
        from pyfoam.io.case import Case
        case = Case(chem_case)
        assert case.has_mesh()

    def test_species_detected(self, chem_case):
        """从 0/ 目录检测到组分。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        assert "A" in solver.species
        assert "B" in solver.species

    def test_reactions_read(self, chem_case):
        """从 constant/reactions 读取反应机理。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        assert len(solver.reactions) == 1
        assert solver.reactions[0].A == 1e6
        assert solver.reactions[0].Ea == 50000

    def test_temperature_initialised(self, chem_case):
        """温度初始化正确。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        assert torch.allclose(solver.T, torch.full((1,), 1000.0, dtype=CFD_DTYPE))

    def test_multi_cell_species(self, multi_cell_chem_case):
        """多胞组分形状正确。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(multi_cell_chem_case)
        assert solver.Y["A"].shape == (3,)
        assert solver.Y["B"].shape == (3,)


# ---------------------------------------------------------------------------
# 化学动力学测试
# ---------------------------------------------------------------------------

class TestChemFoamKinetics:
    """Arrhenius 动力学测试。"""

    def test_arrhenius_rate_shape(self, chem_case):
        """Arrhenius 速率形状正确。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        rate = solver._compute_arrhenius_rate(
            solver.reactions[0], solver.T, solver.Y,
        )
        assert rate.shape == (1,)
        assert torch.isfinite(rate).all()

    def test_arrhenius_rate_positive(self, chem_case):
        """Arrhenius 速率非负。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        rate = solver._compute_arrhenius_rate(
            solver.reactions[0], solver.T, solver.Y,
        )
        assert (rate >= 0).all()

    def test_species_source_terms(self, chem_case):
        """组分源项形状正确。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        omega = solver._compute_species_source(solver.T, solver.Y)

        assert "A" in omega
        assert "B" in omega
        assert omega["A"].shape == (1,)
        assert omega["B"].shape == (1,)

    def test_reactant_consumed(self, chem_case):
        """反应物源项为负（消耗）。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        omega = solver._compute_species_source(solver.T, solver.Y)
        # A 是反应物，ω_A < 0
        assert omega["A"].item() < 0

    def test_product_formed(self, chem_case):
        """生成物源项为正（产生）。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        omega = solver._compute_species_source(solver.T, solver.Y)
        # B 是生成物，ω_B > 0
        assert omega["B"].item() > 0


# ---------------------------------------------------------------------------
# 求解器执行测试
# ---------------------------------------------------------------------------

class TestChemFoamSolver:
    """ChemFoam 求解器执行测试。"""

    def test_run_completes(self, chem_case):
        """求解器运行无报错。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        solver.end_time = 0.05
        result = solver.run()
        assert "converged" in result

    def test_mass_fractions_finite(self, chem_case):
        """质量分数在求解后保持有限。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        solver.end_time = 0.05
        solver.run()
        for name in solver.species:
            assert torch.isfinite(solver.Y[name]).all(), f"Y_{name} contains NaN/Inf"

    def test_temperature_finite(self, chem_case):
        """温度在求解后保持有限。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        solver.end_time = 0.05
        solver.run()
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"

    def test_species_conversion(self, chem_case):
        """长时间运行后 A 减少、B 增加。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        solver.end_time = 0.1
        solver.delta_t = 0.001

        Y_A_init = solver.Y["A"].clone()
        Y_B_init = solver.Y["B"].clone()

        solver.run()

        # A 应减少，B 应增加
        assert solver.Y["A"].mean() < Y_A_init.mean(), "A did not decrease"
        assert solver.Y["B"].mean() > Y_B_init.mean(), "B did not increase"

    def test_writes_output(self, chem_case):
        """场写入到时间目录。"""
        from pyfoam.applications.chem_foam import ChemFoam

        solver = ChemFoam(chem_case)
        solver.end_time = 0.05
        solver.run()

        time_dirs = [
            d for d in chem_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1
