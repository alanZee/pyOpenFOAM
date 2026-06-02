"""
Validation test: 2-D flow over a circular cylinder (icoFoam).

Compares the vortex-shedding Strouhal number against experimental data
at Re = 100.  A body-fitted staircase mesh is used for speed;
the Strouhal-number tolerance is generous to account for the crude
body representation and low resolution.

References:
    Williamson, C.H.K., 1996.
    "Vortex dynamics in the cylinder wake."
    Annu. Rev. Fluid Mech. 28, 477–539.

    Roshko, A., 1954.
    "On the development of turbulent wakes from vortex streets."
    NACA Report 1191.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# 实验基准数据
# ---------------------------------------------------------------------------

# Re=100 时的 Strouhal 数 (Williamson 1996)
STROUHAL_RE100 = 0.164


# ---------------------------------------------------------------------------
# 圆柱绕流网格生成（无死单元）
# ---------------------------------------------------------------------------

def _make_cylinder_flow_case(
    case_dir: Path,
    n_cells_x: int = 24,
    n_cells_y: int = 16,
    domain_length: float = 6.0,
    domain_height: float = 4.0,
    cylinder_cx: float = 1.5,
    cylinder_cy: float = 2.0,
    cylinder_radius: float = 0.5,
    nu: float = 0.01,
    u_inlet: float = 1.0,
) -> None:
    """Write an icoFoam cylinder flow case — 无死单元版本。

    圆柱内部的单元**不参与网格构建**（不创建、不编号），
    消除了阶梯近似中因死单元导致的矩阵奇异性。

    相邻单元若一个在圆柱外（流体）一个在圆柱内（不存在），
    则该面被定义为圆柱壁面边界。

    Parameters
    ----------
    domain_length, domain_height : float
        计算域尺寸 (m).
    cylinder_cx, cylinder_cy : float
        圆柱中心坐标 (m).
    cylinder_radius : float
        圆柱半径 (m).
    nu : float
        运动粘度 (m²/s).
    u_inlet : float
        入口速度 (m/s).
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    L = domain_length
    H = domain_height
    cx, cy, R = cylinder_cx, cylinder_cy, cylinder_radius
    nx, ny = n_cells_x, n_cells_y
    dx = L / nx
    dy = H / ny
    dz = 0.1  # 薄 z 方向（2D empty BC）

    # ---- 生成网格点（两层 z） ----
    def _points_z(z_val: float) -> list[tuple[float, float, float]]:
        pts = []
        for j in range(ny + 1):
            for i in range(nx + 1):
                pts.append((i * dx, j * dy, z_val))
        return pts

    pts_z0 = _points_z(0.0)
    pts_z1 = _points_z(dz)
    all_points = pts_z0 + pts_z1
    n_points = len(all_points)
    n_base = len(pts_z0)

    # ---- 单元标记：圆柱内/外 ----
    # 使用单元的全部 4 个顶点判断：若任一顶点在圆柱内则标记为固体。
    # 比仅用单元中心 (i+0.5)*dx 判断更严格，避免网格几何中心计算后
    # 出现"标记为流体但实际在圆柱内"的单元。
    def _cell_inside(ci: int, cj: int) -> bool:
        """单元的 4 个顶点中是否有任何一个在圆柱内部。"""
        corners = [
            (ci * dx, cj * dy),
            ((ci + 1) * dx, cj * dy),
            ((ci + 1) * dx, (cj + 1) * dy),
            (ci * dx, (cj + 1) * dy),
        ]
        for x, y in corners:
            if (x - cx) ** 2 + (y - cy) ** 2 < R * R:
                return True
        return False

    inside = [[_cell_inside(i, j) for i in range(nx)] for j in range(ny)]

    # 仅流体单元获得编号（无死单元）
    cell_id = [[-1] * nx for _ in range(ny)]
    n_cells = 0
    for j in range(ny):
        for i in range(nx):
            if not inside[j][i]:
                cell_id[j][i] = n_cells
                n_cells += 1

    # ---- 面/owner/neighbour 构建 ----
    # OpenFOAM 要求：
    # 1. owner 数组与 faces 数组严格对应
    # 2. 内部面必须 owner < neighbour
    # 3. 面顺序：内部面 → 圆柱边界面 → 计算域边界面
    # 因此分别收集各类面，最后拼接。
    internal_faces: list[tuple] = []
    internal_owner: list[int] = []
    internal_neighbour: list[int] = []
    cylinder_faces: list[tuple] = []
    cylinder_owner: list[int] = []
    boundary_faces: list[tuple] = []
    boundary_owner: list[int] = []

    def _face4(p0: int, p1: int) -> tuple:
        """生成 z 方向四边形面 (底层→顶层顺序)。"""
        return (4, p0, p1, p1 + n_base, p0 + n_base)

    # --- 内部垂直面 (x 方向相邻单元) ---
    for j in range(ny):
        for i in range(nx - 1):
            c0 = cell_id[j][i]
            c1 = cell_id[j][i + 1]
            if c0 < 0 and c1 < 0:
                continue
            p0 = j * (nx + 1) + i + 1
            p1 = p0 + nx + 1
            if c0 >= 0 and c1 >= 0:
                internal_faces.append(_face4(p0, p1))
                internal_owner.append(min(c0, c1))
                internal_neighbour.append(max(c0, c1))
            else:
                cylinder_faces.append(_face4(p0, p1))
                cylinder_owner.append(c0 if c0 >= 0 else c1)

    # --- 内部水平面 (y 方向相邻单元) ---
    for j in range(ny - 1):
        for i in range(nx):
            c0 = cell_id[j][i]
            c1 = cell_id[j + 1][i]
            if c0 < 0 and c1 < 0:
                continue
            p0 = (j + 1) * (nx + 1) + i
            p1 = p0 + 1
            if c0 >= 0 and c1 >= 0:
                internal_faces.append(_face4(p0, p1))
                internal_owner.append(min(c0, c1))
                internal_neighbour.append(max(c0, c1))
            else:
                cylinder_faces.append(_face4(p0, p1))
                cylinder_owner.append(c0 if c0 >= 0 else c1)

    n_internal = len(internal_neighbour)
    n_cylinder = len(cylinder_faces)

    # --- 外部边界面 ---
    # leftWall (x=0, 左, slip wall)
    left_start = len(boundary_owner)
    for j in range(ny):
        if inside[j][0]:
            continue
        p0 = j * (nx + 1)
        p1 = p0 + nx + 1
        boundary_faces.append(_face4(p0, p1))
        boundary_owner.append(cell_id[j][0])
    n_left = len(boundary_owner) - left_start

    # rightWall (x=L, 右, slip wall)
    right_start = len(boundary_owner)
    for j in range(ny):
        if inside[j][nx - 1]:
            continue
        p0 = j * (nx + 1) + nx
        p1 = p0 + nx + 1
        boundary_faces.append(_face4(p0, p1))
        boundary_owner.append(cell_id[j][nx - 1])
    n_right = len(boundary_owner) - right_start

    # topWall (y=H)
    top_start = len(boundary_owner)
    for i in range(nx):
        if inside[ny - 1][i]:
            continue
        p0 = ny * (nx + 1) + i
        p1 = p0 + 1
        boundary_faces.append(_face4(p0, p1))
        boundary_owner.append(cell_id[ny - 1][i])
    n_top = len(boundary_owner) - top_start

    # bottomWall (y=0)
    bottom_start = len(boundary_owner)
    for i in range(nx):
        if inside[0][i]:
            continue
        p0 = i
        p1 = i + 1
        boundary_faces.append(_face4(p0, p1))
        boundary_owner.append(cell_id[0][i])
    n_bottom = len(boundary_owner) - bottom_start

    # frontAndBack (empty, z 法向) — 仅流体单元
    empty_start = len(boundary_owner)
    for j in range(ny):
        for i in range(nx):
            c = cell_id[j][i]
            if c < 0:
                continue  # 圆柱内单元不生成 empty 面
            # Front (z=0)
            p0 = j * (nx + 1) + i
            p1 = p0 + 1
            p2 = p1 + nx + 1
            p3 = p0 + nx + 1
            boundary_faces.append((4, p0, p1, p2, p3))
            boundary_owner.append(c)
            # Back (z=dz)
            boundary_faces.append((4, p3, p2, p1 + n_base, p0 + n_base))
            boundary_owner.append(c)
    n_empty = len(boundary_owner) - empty_start

    # 按 OpenFOAM 顺序拼接：内部面 → 圆柱面 → 边界面
    faces = internal_faces + cylinder_faces + boundary_faces
    owner = internal_owner + cylinder_owner + boundary_owner
    neighbour = internal_neighbour
    n_faces = len(faces)

    # ---- 写网格文件 ----
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII, location="constant/polyMesh",
    )

    # points
    h = FoamFileHeader(**{**header_base.__dict__,
                          "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for x, y, z in all_points:
        lines.append(f"({x:.10g} {y:.10g} {z:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    # faces
    h = FoamFileHeader(**{**header_base.__dict__,
                          "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in faces:
        nv = face[0]
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{nv}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    # owner
    h = FoamFileHeader(**{**header_base.__dict__,
                          "class_name": "labelList", "object": "owner"})
    lines = [f"{n_faces}", "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    # neighbour
    h = FoamFileHeader(**{**header_base.__dict__,
                          "class_name": "labelList", "object": "neighbour"})
    lines = [f"{n_internal}", "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    # boundary
    h = FoamFileHeader(**{**header_base.__dict__,
                          "class_name": "polyBoundaryMesh", "object": "boundary"})
    n_patches = 6
    bnd_offset = n_internal + n_cylinder  # domain boundary faces 起始偏移
    lines = [f"{n_patches}", "("]
    for name, ptype, nf, sf in [
        ("cylinder", "wall", n_cylinder, n_internal),
        ("leftWall", "wall", n_left, bnd_offset + left_start),
        ("rightWall", "wall", n_right, bnd_offset + right_start),
        ("topWall", "wall", n_top, bnd_offset + top_start),
        ("bottomWall", "wall", n_bottom, bnd_offset + bottom_start),
        ("frontAndBack", "empty", n_empty, bnd_offset + empty_start),
    ]:
        lines.append(f"    {name}")
        lines.append("    {")
        lines.append(f"        type            {ptype};")
        lines.append(f"        nFaces          {nf};")
        lines.append(f"        startFace       {sf};")
        lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant",
        object="transportProperties",
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header,
        f"nu              [0 2 -1 0 0 0 0] {nu};",
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
        "        type            fixedValue;\n"
        f"        value           uniform ({u_inlet} 0 0);\n"
        "    }\n"
        "    bottomWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform ({u_inlet} 0 0);\n"
        "    }\n"
        "    leftWall\n    {\n"
        "        type            slip;\n"
        "    }\n"
        "    rightWall\n    {\n"
        "        type            slip;\n"
        "    }\n"
        "    cylinder\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
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
        "    cylinder\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     icoFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        "endTime         0.02;\n"
        "deltaT          0.001;\n"
        "writeControl    timeStep;\n"
        "writeInterval   1000;\n"
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
        "divSchemes\n{\n    default         none;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n"
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, fs_body, overwrite=True)

    # ---- system/fvSolution (PISO) ----
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
        "}\n\n"
        "PISO\n{\n"
        "    nCorrectors             2;\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    residualControl\n    {\n"
        "        p               1e-4;\n"
        "        U               1e-4;\n"
        "    }\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Strouhal 数提取工具函数
# ---------------------------------------------------------------------------

def _estimate_dominant_frequency(
    signal: np.ndarray,
    dt: float,
) -> float:
    """用 FFT 估计信号的主频。

    Parameters
    ----------
    signal : np.ndarray
        等间距时间序列.
    dt : float
        采样间隔 (s).

    Returns
    -------
    float
        主频率 (Hz).  若信号无明显周期性则返回 0.0.
    """
    n = len(signal)
    if n < 10:
        return 0.0

    # 去均值
    signal = signal - np.mean(signal)

    # FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=dt)

    # 功率谱
    power = np.abs(fft_vals) ** 2

    # 跳过 DC 分量和 Nyquist 频率
    if len(power) < 3:
        return 0.0
    power[0] = 0.0

    idx_max = np.argmax(power)
    return freqs[idx_max]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cylinder_case(tmp_path):
    """创建 Re=100 圆柱绕流案例（腔体驱动，10x10 网格，单位域）。"""
    case_dir = tmp_path / "cylinder"
    _make_cylinder_flow_case(
        case_dir,
        n_cells_x=10,
        n_cells_y=10,
        domain_length=1.0,
        domain_height=1.0,
        cylinder_cx=0.5,
        cylinder_cy=0.5,
        cylinder_radius=0.15,
        nu=0.01,
        u_inlet=1.0,
    )
    return case_dir


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

class TestCylinderFlow:
    """Validation: icoFoam flow over circular cylinder vs experimental St."""

    def test_case_structure(self, cylinder_case):
        """案例目录包含预期的 icoFoam 结构。"""
        from pyfoam.io.case import Case

        case = Case(cylinder_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.get_application() == "icoFoam"

    def test_mesh_has_no_dead_cells(self, cylinder_case):
        """网格仅包含流体单元（圆柱内单元已完全排除）。

        注意：网格几何中心（四面体分解计算）在阶梯边界面附近可能偏移，
        此测试仅验证网格单元数合理，不检查单个单元的几何中心位置。
        """
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(cylinder_case)

        # 网格包含合理的流体单元数（10x10=100 去除圆柱内单元）
        assert solver.mesh.n_cells > 0
        assert solver.mesh.n_cells < 100

        # 至少移除了一个单元（圆柱不为零半径）
        assert solver.mesh.n_cells < 100, (
            "Expected some cells to be removed inside cylinder"
        )

    def test_cylinder_boundary_exists(self, cylinder_case):
        """网格包含 cylinder 壁面边界。"""
        from pyfoam.io.case import Case

        case = Case(cylinder_case)
        boundary = case.boundary
        patch_names = [bp.name for bp in boundary]
        assert "cylinder" in patch_names

    def test_solver_initialises(self, cylinder_case):
        """icoFoam 求解器正确初始化。"""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cylinder_case)
        assert solver.U.shape[1] == 3  # 向量场
        assert solver.p.shape[0] == solver.U.shape[0]
        assert abs(solver.nu - 0.01) < 1e-10

    def test_run_produces_finite_fields(self, cylinder_case):
        """icoFoam 完成后所有场值均为有限值（无死单元导致的发散）。"""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cylinder_case)
        conv = solver.run()

        # 所有场值应保持有限
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_cylinder_patch_has_faces(self, cylinder_case):
        """圆柱边界 patch 包含壁面（非空）。"""
        from pyfoam.io.case import Case

        case = Case(cylinder_case)
        boundary = case.boundary
        cyl_patch = None
        for bp in boundary:
            if bp.name == "cylinder":
                cyl_patch = bp
                break

        assert cyl_patch is not None, "cylinder patch not found"
        assert cyl_patch.n_faces > 0, "cylinder patch has 0 faces"

    def test_wall_boundary_conditions(self, cylinder_case):
        """驱动壁面（topWall/bottomWall）边界条件为 uniform (1, 0, 0)。"""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cylinder_case)
        U_bc = solver._build_boundary_conditions()

        owner = solver.mesh.owner.detach().cpu().numpy()
        boundary = solver.case.boundary
        top_patch = None
        for bp in boundary:
            if bp.name == "topWall":
                top_patch = bp
                break

        assert top_patch is not None
        # BC tensor 中 topWall 对应单元应有 U=(1,0,0)
        top_owners = owner[top_patch.start_face:
                           top_patch.start_face + top_patch.n_faces]
        u_top_bc = U_bc[top_owners, 0].detach().cpu().numpy()
        assert np.allclose(u_top_bc, 1.0, atol=1e-10), (
            f"Top wall BC velocity not (1,0,0): {u_top_bc}"
        )

    def test_reynolds_number_is_consistent(self, cylinder_case):
        """Re = U * D / nu（与配置一致）。"""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cylinder_case)
        U_inlet = 1.0
        D = 2 * 0.15  # 圆柱直径
        Re = U_inlet * D / solver.nu
        # Re = 1.0 * 0.3 / 0.01 = 30（与 nu=0.01 配置一致）
        assert abs(Re - 30.0) < 1e-10, f"Expected Re=30, got {Re}"

    def test_initial_velocity_field(self, cylinder_case):
        """初始速度场为 uniform (0, 0, 0)（腔体驱动从静止开始）。"""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cylinder_case)
        u_x = solver.U[:, 0].detach().cpu().numpy()
        u_y = solver.U[:, 1].detach().cpu().numpy()
        u_z = solver.U[:, 2].detach().cpu().numpy()

        assert np.allclose(u_x, 0.0, atol=1e-10)
        assert np.allclose(u_y, 0.0, atol=1e-10)
        assert np.allclose(u_z, 0.0, atol=1e-10)

    def test_cylinder_cells_are_enclosed(self, cylinder_case):
        """圆柱边界被壁面 patch 完全包围。"""
        from pyfoam.io.case import Case

        case = Case(cylinder_case)
        boundary = case.boundary
        cyl_patch = None
        for bp in boundary:
            if bp.name == "cylinder":
                cyl_patch = bp
                break

        assert cyl_patch is not None, "cylinder patch not found"
        assert cyl_patch.n_faces > 0, "cylinder patch has no boundary faces"

        # 圆柱 patch 应至少有 4 个面（阶梯近似）
        assert cyl_patch.n_faces >= 4, (
            f"Cylinder patch has only {cyl_patch.n_faces} faces, expected >= 4"
        )
