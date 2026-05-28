"""
Unit tests for ViscousFoam — viscous flow solver for high-viscosity fluids.

Tests cover:
- Case loading and field initialisation
- Viscosity model reading (constant, powerLaw, BirdCarreau, Cross)
- Non-Newtonian viscosity model construction
- Strain rate computation
- SIMPLE solver construction
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
# Case generation
# ---------------------------------------------------------------------------

def _make_viscous_case(
    case_dir: Path,
    n_cells: int = 3,
    end_time: int = 500,
    delta_t: float = 1.0,
    write_interval: int = 100,
    viscosity_model: str = "constant",
    extra_tp: str = "",
) -> None:
    """Write a simple 1D case for ViscousFoam."""
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

    # Empty patches (side walls)
    empty_start = outlet_start + 1
    for i in range(n_cells):
        faces.append((4, i * 4, (i + 1) * 4, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4, i * 4 + 1, (i + 1) * 4 + 1, i * 4))
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

    # ---- transportProperties ----
    tp_header = FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                               location="constant", object="transportProperties")

    tp_content = f"viscosityModel  {viscosity_model};\nnu              0.01;\n{extra_tp}\n"
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header,
        tp_content, overwrite=True,
    )

    # ---- 0/ 目录 ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    # U
    write_foam_file(zero_dir / "U",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volVectorField",
                       location="0", object="U"),
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (1 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            fixedValue;\n"
        "        value           uniform (1 0 0);\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    # p
    write_foam_file(zero_dir / "p",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="p"),
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            zeroGradient;\n    }\n"
        "    outlet\n    {\n        type            fixedValue;\n"
        "        value           uniform 0;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    # ---- system/ ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    write_foam_file(sys_dir / "controlDict",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="controlDict"),
        "application     viscousFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        f"writeInterval   {write_interval};\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSchemes",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSchemes"),
        "ddtSchemes\n{\n    default         steadyState;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSolution",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSolution"),
        "SIMPLE\n{\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    nOuterCorrectors    100;\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n\n"
        "solvers\n{\n"
        "    p\n    {\n        solver          PCG;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n    }\n"
        "    U\n    {\n        solver          PBiCGStab;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n    }\n"
        "}\n",
        overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def constant_case(tmp_path):
    """Case with constant (Newtonian) viscosity."""
    case_dir = tmp_path / "constant_viscous"
    _make_viscous_case(case_dir, viscosity_model="constant")
    return case_dir


@pytest.fixture
def power_law_case(tmp_path):
    """Case with power-law viscosity model."""
    case_dir = tmp_path / "powerlaw_viscous"
    _make_viscous_case(
        case_dir,
        viscosity_model="powerLaw",
        extra_tp="K               0.01;\nn               0.5;\nnu_min          1e-6;\nnu_max          1e4;\n",
    )
    return case_dir


@pytest.fixture
def bird_carreau_case(tmp_path):
    """Case with Bird-Carreau viscosity model."""
    case_dir = tmp_path / "birdcarreau_viscous"
    _make_viscous_case(
        case_dir,
        viscosity_model="BirdCarreau",
        extra_tp="mu_0            0.05;\nmu_inf          0.001;\nlambda_         1.0;\nn               0.4;\n",
    )
    return case_dir


@pytest.fixture
def cross_case(tmp_path):
    """Case with Cross power-law viscosity model."""
    case_dir = tmp_path / "cross_viscous"
    _make_viscous_case(
        case_dir,
        viscosity_model="Cross",
        extra_tp="mu_0            0.05;\nmu_inf          0.001;\nlambda_         1.0;\nm               1.0;\n",
    )
    return case_dir


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestViscousFoamInit:
    """ViscousFoam 初始化测试。"""

    def test_case_loads(self, constant_case):
        """算例目录可读取。"""
        from pyfoam.io.case import Case
        case = Case(constant_case)
        assert case.has_mesh()

    def test_init_constant(self, constant_case):
        """常粘度模型初始化成功。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(constant_case)
        assert solver.viscosity_model == "constant"
        assert solver.nu == 0.01

    def test_init_power_law(self, power_law_case):
        """幂律模型初始化成功。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(power_law_case)
        assert solver.viscosity_model == "powerLaw"

    def test_init_bird_carreau(self, bird_carreau_case):
        """Bird-Carreau 模型初始化成功。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(bird_carreau_case)
        assert solver.viscosity_model == "BirdCarreau"

    def test_init_cross(self, cross_case):
        """Cross 模型初始化成功。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(cross_case)
        assert solver.viscosity_model == "Cross"

    def test_field_shapes(self, constant_case):
        """场张量形状正确。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(constant_case)
        assert solver.U.shape[1] == 3  # vector field
        assert solver.p.shape[0] == solver.U.shape[0]  # same n_cells
        assert solver.phi.shape[0] > 0

    def test_fields_finite(self, constant_case):
        """所有场值在初始化后有限。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(constant_case)
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


# ---------------------------------------------------------------------------
# Viscosity model tests
# ---------------------------------------------------------------------------

class TestViscousFoamViscosityModels:
    """粘度模型测试。"""

    def test_constant_viscosity_model(self, constant_case):
        """常粘度模型返回恒定值。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(constant_case)
        assert not solver._visco.is_non_newtonian()
        gd = torch.tensor([0.1, 1.0, 10.0], dtype=torch.float64)
        mu = solver._visco.mu(gd)
        assert torch.allclose(mu, torch.full_like(mu, 0.01))

    def test_power_law_non_newtonian(self, power_law_case):
        """幂律模型是非牛顿的。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(power_law_case)
        assert solver._visco.is_non_newtonian()

    def test_power_law_shear_thinning(self, power_law_case):
        """幂律模型（n<1）是剪切变稀的。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(power_law_case)
        gd_low = torch.tensor([0.01], dtype=torch.float64)
        gd_high = torch.tensor([10.0], dtype=torch.float64)
        assert solver._visco.mu(gd_low) > solver._visco.mu(gd_high)

    def test_bird_carreau_limits(self, bird_carreau_case):
        """Bird-Carreau 模型在零剪切和高剪切极限正确。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(bird_carreau_case)
        gd_zero = torch.tensor([0.0], dtype=torch.float64)
        gd_high = torch.tensor([1e10], dtype=torch.float64)
        assert torch.allclose(solver._visco.mu(gd_zero), torch.tensor([0.05], dtype=torch.float64), atol=1e-6)
        assert torch.allclose(solver._visco.mu(gd_high), torch.tensor([0.001], dtype=torch.float64), atol=1e-4)

    def test_cross_model_limits(self, cross_case):
        """Cross 模型在零剪切和高剪切极限正确。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(cross_case)
        gd_zero = torch.tensor([0.0], dtype=torch.float64)
        gd_high = torch.tensor([1e10], dtype=torch.float64)
        assert torch.allclose(solver._visco.mu(gd_zero), torch.tensor([0.05], dtype=torch.float64), atol=1e-6)
        assert torch.allclose(solver._visco.mu(gd_high), torch.tensor([0.001], dtype=torch.float64), atol=1e-4)


# ---------------------------------------------------------------------------
# Strain rate tests
# ---------------------------------------------------------------------------

class TestViscousFoamStrainRate:
    """应变率计算测试。"""

    def test_strain_rate_shape(self, constant_case):
        """应变率张量形状正确。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(constant_case)
        mag_S = solver._compute_strain_rate_magnitude(solver.U)
        assert mag_S.shape == (solver.mesh.n_cells,)

    def test_strain_rate_non_negative(self, constant_case):
        """应变率幅值非负。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(constant_case)
        mag_S = solver._compute_strain_rate_magnitude(solver.U)
        assert (mag_S >= 0).all()

    def test_strain_rate_finite(self, constant_case):
        """应变率值有限。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(constant_case)
        mag_S = solver._compute_strain_rate_magnitude(solver.U)
        assert torch.isfinite(mag_S).all()


# ---------------------------------------------------------------------------
# Solver execution tests
# ---------------------------------------------------------------------------

class TestViscousFoamSolver:
    """ViscousFoam 求解器执行测试。"""

    def test_run_constant(self, constant_case):
        """常粘度模型运行完成。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(constant_case)
        solver.end_time = 10
        result = solver.run()
        assert result is not None

    def test_run_power_law(self, power_law_case):
        """幂律模型运行完成。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(power_law_case)
        solver.end_time = 10
        result = solver.run()
        assert result is not None

    def test_fields_finite_after_run(self, constant_case):
        """运行后场值有限。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(constant_case)
        solver.end_time = 10
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_writes_output(self, constant_case):
        """场写入到时间目录。"""
        from pyfoam.applications.viscous_foam import ViscousFoam
        solver = ViscousFoam(constant_case)
        solver.end_time = 10
        solver.run()
        time_dirs = [
            d for d in constant_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1
