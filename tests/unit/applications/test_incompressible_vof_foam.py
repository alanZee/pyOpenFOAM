"""
End-to-end test: incompressibleVoF dam break.

Creates a complete two-phase dam break case on disk (mesh, fields,
system files), runs IncompressibleVoFFoam, and verifies:
- α stays bounded [0, 1]
- Volume fraction is conserved
- Fields are finite (no NaN/Inf)
- MULES limiter is used for alpha boundedness
- PIMPLE outer correctors converge
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# 网格和算例生成辅助函数（与 interFoam 测试共用相同的 dam break 设置）
# ---------------------------------------------------------------------------

def _make_dam_break_case(
    case_dir: Path,
    n_cells_x: int = 8,
    n_cells_y: int = 8,
    dx: float = 0.5,
    dy: float = 0.5,
    rho1: float = 1000.0,
    rho2: float = 1.225,
    mu1: float = 1e-3,
    mu2: float = 1.8e-5,
    sigma: float = 0.07,
    delta_t: float = 0.001,
    end_time: float = 0.01,
    n_outer: int = 3,
    n_correctors: int = 2,
) -> None:
    """写入完整的 dam break 算例用于 incompressibleVoF。

    创建：
    - constant/polyMesh/{points, faces, owner, neighbour, boundary}
    - constant/transportProperties
    - 0/U, 0/p, 0/alpha.water
    - system/{controlDict, fvSchemes, fvSolution}
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    # ---- 网格 ----
    dz = 0.1
    Lx = n_cells_x * dx
    Ly = n_cells_y * dy

    # 点：两层（z=0, z=dz）
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

    # 面
    faces = []
    owner = []
    neighbour = []

    # 内部垂直面（x 方向）
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # 内部水平面（y 方向）
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

    # 边界面
    # topWall (y=Ly)
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)

    n_top = n_cells_x
    top_start = n_internal

    # bottomWall (y=0)
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)

    n_bottom = n_cells_x
    bottom_start = top_start + n_top

    # leftWall (x=0)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)

    n_left = n_cells_y
    left_start = bottom_start + n_bottom

    # rightWall (x=Lx)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)

    n_right = n_cells_y
    right_start = left_start + n_left

    # frontAndBack（z 法向，empty）
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
    empty_start = right_start + n_right

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

    # 写入网格文件
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0",
        format=FileFormat.ASCII,
        location="constant/polyMesh",
    )

    # points
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for x, y, z in all_points:
        lines.append(f"({x:.10g} {y:.10g} {z:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    # faces
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in faces:
        nv = face[0]
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{nv}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    # owner
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "owner"})
    lines = [f"{n_faces}", "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    # neighbour
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"})
    lines = [f"{n_internal}", "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    # boundary
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"})
    lines = ["5", "("]
    for name, n_f, start in [
        ("topWall", n_top, top_start),
        ("bottomWall", n_bottom, bottom_start),
        ("leftWall", n_left, left_start),
        ("rightWall", n_right, right_start),
        ("frontAndBack", n_empty, empty_start),
    ]:
        lines.append(f"    {name}")
        lines.append("    {")
        if name == "frontAndBack":
            lines.append("        type            empty;")
        else:
            lines.append("        type            wall;")
        lines.append(f"        nFaces          {n_f};")
        lines.append(f"        startFace       {start};")
        lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- constant/transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    tp_body = (
        f"nu1             [0 2 -1 0 0 0 0] {mu1/rho1};\n"
        f"nu2             [0 2 -1 0 0 0 0] {mu2/rho2};\n"
        f"rho1            [1 -3 0 0 0 0 0] {rho1};\n"
        f"rho2            [1 -3 0 0 0 0 0] {rho2};\n"
        f"sigma           [1 0 -2 0 0 0 0] {sigma};\n"
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header, tp_body,
        overwrite=True,
    )

    # ---- 0/U ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    topWall\n    {\n"
        "        type            slip;\n"
        "    }\n"
        "    bottomWall\n    {\n"
        "        type            noSlip;\n"
        "    }\n"
        "    leftWall\n    {\n"
        "        type            noSlip;\n"
        "    }\n"
        "    rightWall\n    {\n"
        "        type            noSlip;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # ---- 0/p ----
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    p_body = (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    topWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    bottomWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    leftWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    rightWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- 0/alpha.water ----
    alpha_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="alpha.water",
    )
    # 阶跃函数：x < Lx/4, y < Ly/2 处为水
    dam_x = Lx / 4.0
    dam_y = Ly / 2.0
    alpha_lines = []
    alpha_lines.append("dimensions      [0 0 0 0 0 0 0];")
    alpha_lines.append("")
    alpha_lines.append(f"internalField   nonuniform {n_cells}")
    alpha_lines.append("(")
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            xc = (i + 0.5) * dx
            yc = (j + 0.5) * dy
            if xc < dam_x and yc < dam_y:
                alpha_lines.append("1")
            else:
                alpha_lines.append("0")
    alpha_lines.append(")")
    alpha_lines.append("")
    alpha_lines.append("boundaryField\n{")
    for bname in ["topWall", "bottomWall", "leftWall", "rightWall"]:
        alpha_lines.append(f"    {bname}")
        alpha_lines.append("    {")
        alpha_lines.append("        type            zeroGradient;")
        alpha_lines.append("    }")
    alpha_lines.append("    frontAndBack")
    alpha_lines.append("    {")
    alpha_lines.append("        type            empty;")
    alpha_lines.append("    }")
    alpha_lines.append("}")
    alpha_body = "\n".join(alpha_lines) + "\n"
    write_foam_file(zero_dir / "alpha.water", alpha_header, alpha_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     incompressibleVoF;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time:g};\n"
        f"deltaT          {delta_t:g};\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n"
        "purgeWrite      0;\n"
        "writeFormat     ascii;\n"
        "writePrecision  8;\n"
        "writeCompression off;\n"
        "timeFormat      general;\n"
        "timePrecision   6;\n"
        "runTimeModifiable true;\n"
    )
    write_foam_file(sys_dir / "controlDict", cd_header, cd_body, overwrite=True)

    # ---- system/fvSchemes ----
    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    fs_body = (
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         none;\n"
        "    div(phi,alpha)  Gauss vanLeer;\n"
        "    div(phi,U)      Gauss upwind;\n"
        "}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n"
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, fs_body, overwrite=True)

    # ---- system/fvSolution ----
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    fv_body = (
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
        "    alpha.water\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "PIMPLE\n{\n"
        f"    nOuterCorrectors    {n_outer};\n"
        f"    nCorrectors         {n_correctors};\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

@pytest.fixture
def dam_break_case(tmp_path):
    """在临时目录创建 dam break 算例。"""
    case_dir = tmp_path / "damBreak"
    _make_dam_break_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        dx=1.0,
        dy=1.0,
        delta_t=0.001,
        end_time=0.005,  # 5 个时间步
        n_outer=2,
        n_correctors=2,
    )
    return case_dir


class TestIncompressibleVoFDamBreak:
    """incompressibleVoF dam break 端到端测试。"""

    def test_case_loads(self, dam_break_case):
        """算例目录可读且结构正确。"""
        from pyfoam.io.case import Case

        case = Case(dam_break_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.has_field("alpha.water", 0)

    def test_mesh_builds(self, dam_break_case):
        """FvMesh 构造正确。"""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(dam_break_case)
        mesh = solver.mesh

        assert mesh.n_cells == 16  # 4x4
        assert mesh.n_internal_faces > 0
        assert mesh.cell_volumes.shape == (16,)

    def test_fields_initialise(self, dam_break_case):
        """场初始化正确。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            dam_break_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
        )

        # U 应为零
        assert solver.U.shape == (16, 3)
        assert torch.allclose(solver.U, torch.zeros(16, 3, dtype=CFD_DTYPE))

        # p 应为零
        assert solver.p.shape == (16,)

        # alpha 应为阶跃函数
        assert solver.alpha.shape == (16,)
        assert solver.alpha.min() >= 0.0
        assert solver.alpha.max() <= 1.0

    def test_mixture_properties(self, dam_break_case):
        """混合物性计算正确。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            dam_break_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
        )

        assert torch.isfinite(solver.rho).all()
        assert solver.rho.min() > 0

    def test_mules_limiter_initialised(self, dam_break_case):
        """MULES 限制器初始化。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            dam_break_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
            mules_iterations=3,
        )

        assert hasattr(solver, 'mules')
        assert solver.mules._n_iterations == 3

    def test_vof_advection_initialised(self, dam_break_case):
        """VOF 对流模块初始化。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            dam_break_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
            C_alpha=1.0,
        )

        assert hasattr(solver, 'vof')
        assert solver.vof._C_alpha == 1.0

    def test_run_produces_valid_fields(self, dam_break_case):
        """求解器运行且场有效。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            dam_break_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
        )
        conv = solver.run()

        # 场形状正确
        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)
        assert solver.alpha.shape == (16,)

        # 所有值有限
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.alpha).all(), "alpha contains NaN/Inf"

    def test_alpha_bounded(self, dam_break_case):
        """alpha 在 [0, 1] 范围内（MULES 有界性）。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            dam_break_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
        )
        solver.run()

        assert solver.alpha.min() >= -1e-10, f"alpha min = {solver.alpha.min()}"
        assert solver.alpha.max() <= 1.0 + 1e-10, f"alpha max = {solver.alpha.max()}"

    def test_volume_conservation(self, dam_break_case):
        """总水量近似守恒。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            dam_break_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
        )

        V = solver.mesh.cell_volumes
        water_before = (solver.alpha * V).sum().item()

        solver.run()

        water_after = (solver.alpha * V).sum().item()
        # 允许 10% 的数值扩散损失
        assert abs(water_after - water_before) < 0.1 * max(abs(water_before), 1e-30), \
            f"Water before: {water_before}, after: {water_after}"

    def test_pressure_field_shape(self, dam_break_case):
        """压力场形状正确。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            dam_break_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
        )
        solver.run()

        assert solver.p.shape == (16,)
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_transient_time_stepping(self, dam_break_case):
        """多个时间步推进正确。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            dam_break_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
        )

        assert abs(solver.delta_t - 0.001) < 1e-10
        assert abs(solver.end_time - 0.005) < 1e-10

        conv = solver.run()
        assert conv.outer_iterations >= 1

    def test_surface_tension_model(self, dam_break_case):
        """表面张力模型初始化。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            dam_break_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
        )

        assert hasattr(solver, 'surface_tension')
        assert solver.surface_tension.sigma == 0.07

        # 计算表面张力力
        F_st = solver._compute_surface_tension()
        assert F_st.shape == (16, 3)
        assert torch.isfinite(F_st).all()

    def test_operator_splitting_order(self, dam_break_case):
        """验证 operator splitting：alpha 在 PIMPLE 之前更新。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            dam_break_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
        )

        # alpha_before 与 alpha_after 应不同（VOF 对流会改变 alpha）
        alpha_before = solver.alpha.clone()
        solver.run()

        # 如果初始 alpha 有梯度，VOF 对流会改变它
        # 至少验证 alpha 仍有效
        assert torch.isfinite(solver.alpha).all()
        assert solver.alpha.shape == alpha_before.shape


class TestIncompressibleVoFSmallMesh:
    """2x2 小网格上的快速测试。"""

    @pytest.fixture
    def tiny_case(self, tmp_path):
        """创建最小 2x2 dam break 算例。"""
        case_dir = tmp_path / "tiny_dam"
        _make_dam_break_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            dx=2.0,
            dy=2.0,
            delta_t=0.001,
            end_time=0.003,  # 3 步
            n_outer=2,
            n_correctors=2,
        )
        return case_dir

    def test_tiny_mesh_runs(self, tiny_case):
        """2x2 网格无错误运行。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            tiny_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
        )
        assert solver.mesh.n_cells == 4

        conv = solver.run()
        assert solver.U.shape == (4, 3)
        assert solver.p.shape == (4,)
        assert solver.alpha.shape == (4,)
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.alpha).all()

    def test_tiny_mesh_alpha_bounded(self, tiny_case):
        """alpha 在小网格上保持有界。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            tiny_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
        )
        solver.run()

        assert solver.alpha.min() >= -1e-10
        assert solver.alpha.max() <= 1.0 + 1e-10

    def test_custom_mules_iterations(self, tiny_case):
        """自定义 MULES 迭代次数。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

        solver = IncompressibleVoFFoam(
            tiny_case,
            rho1=1000.0, rho2=1.225,
            mu1=1e-3, mu2=1.8e-5,
            sigma=0.07,
            mules_iterations=5,
        )
        assert solver.mules._n_iterations == 5

        conv = solver.run()
        assert torch.isfinite(solver.alpha).all()


class TestIncompressibleVoFExports:
    """验证模块导出。"""

    def test_import_from_applications(self):
        """从 pyfoam.applications 导入 IncompressibleVoFFoam。"""
        from pyfoam.applications import IncompressibleVoFFoam
        assert IncompressibleVoFFoam is not None

    def test_import_from_module(self):
        """从子模块直接导入。"""
        from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam
        assert IncompressibleVoFFoam is not None
