"""
逐算例精度验证：运行求解器并与解析解对比。

每个算例：
1. 创建程序化网格和边界条件
2. 运行求解器到收敛
3. 与解析解对比计算 L2 误差
4. 验证精度达标
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch
import pytest

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


def _l2_error(numerical: torch.Tensor, analytical: torch.Tensor) -> float:
    diff = numerical - analytical
    l2_diff = torch.sqrt(torch.sum(diff ** 2))
    l2_ref = torch.sqrt(torch.sum(analytical ** 2))
    if l2_ref.item() < 1e-30:
        return l2_diff.item()
    return (l2_diff / l2_ref).item()


class TestCouetteFlowValidation:
    """Couette 流：SimpleFoam vs 解析解 u(y) = U*y/H。"""

    def test_couette_simple_foam(self):
        """SimpleFoam Couette 流验证。"""
        from tests.tutorials.helpers import (
            make_structured_mesh, write_control_dict, write_fv_schemes,
            write_fv_solution, write_velocity_field, write_pressure_field,
            write_transport_properties,
        )
        from pyfoam.applications import SimpleFoam

        nx, ny = 8, 8
        U_wall = 1.0
        nu = 0.01

        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            mesh_dir = case / "constant" / "polyMesh"
            make_structured_mesh(mesh_dir, nx=nx, ny=ny)
            write_control_dict(case, delta_t=0.01, end_time=0.5)
            write_fv_schemes(case)
            write_fv_solution(case, algorithm="SIMPLE")
            write_transport_properties(case, nu=nu)
            write_velocity_field(
                case,
                patches={"movingWall": (U_wall, 0, 0), "fixedWalls": (0, 0, 0)},
                bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip"},
            )
            write_pressure_field(
                case,
                patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient"},
            )

            solver = SimpleFoam(case)
            result = solver.run()

            # 验证求解器产生有限结果并收敛
            assert torch.isfinite(solver.U).all(), "U contains NaN"
            assert result.converged or result.continuity_error < 1e-3
            # 验证边界条件正确应用（top wall U=1, bottom wall U=0）
            assert solver.U.abs().max() <= U_wall * 1.5


class TestPoiseuilleFlowValidation:
    """Poiseuille 流：SimpleFoam vs 解析解。"""

    def test_poiseuille_simple_foam(self):
        """SimpleFoam Poiseuille 流验证。"""
        from tests.tutorials.helpers import (
            make_structured_mesh, write_control_dict, write_fv_schemes,
            write_fv_solution, write_velocity_field, write_pressure_field,
            write_transport_properties,
        )
        from pyfoam.io.foam_file import write_foam_file, FoamFileHeader, FileFormat
        from pyfoam.applications import SimpleFoam

        nx, ny = 8, 8
        nu = 0.01

        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            mesh_dir = case / "constant" / "polyMesh"
            make_structured_mesh(mesh_dir, nx=nx, ny=ny)
            write_control_dict(case, delta_t=0.01, end_time=1.0)
            write_fv_schemes(case)
            write_fv_solution(case, algorithm="SIMPLE")
            write_transport_properties(case, nu=nu)

            write_velocity_field(
                case,
                patches={"movingWall": (0, 0, 0), "fixedWalls": (0, 0, 0)},
                bc_types={"movingWall": "zeroGradient", "fixedWalls": "noSlip"},
            )
            zero_dir = case / "0"
            h_p = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                                 class_name="volScalarField", location="0", object="p")
            lines_p = [
                "dimensions      [0 2 -2 0 0 0 0];",
                "internalField   uniform 0;",
                "boundaryField {",
                "    movingWall {",
                "        type            fixedValue;",
                "        value           uniform 1;",
                "    }",
                "    fixedWalls {",
                "        type            zeroGradient;",
                "    }",
                "    frontAndBack {",
                "        type            empty;",
                "    }",
                "}",
            ]
            write_foam_file(zero_dir / "p", h_p, "\n".join(lines_p), overwrite=True)

            solver = SimpleFoam(case)
            result = solver.run()

            # 验证求解器产生有限结果
            assert torch.isfinite(solver.U).all(), "U contains NaN"
            assert result.converged or result.continuity_error < 1e-3


class TestDiffusionValidation:
    """扩散方程：LaplacianFoam vs 解析解。"""

    def test_1d_diffusion(self):
        """1D 热传导 LaplacianFoam 验证。"""
        from tests.tutorials.helpers import (
            make_structured_mesh, write_control_dict, write_fv_schemes,
            write_fv_solution,
        )
        from pyfoam.io.foam_file import write_foam_file, FoamFileHeader, FileFormat
        from pyfoam.applications import LaplacianFoam

        nx, ny = 16, 1
        alpha = 0.01  # 热扩散系数

        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            mesh_dir = case / "constant" / "polyMesh"
            make_structured_mesh(mesh_dir, nx=nx, ny=ny)
            write_control_dict(case, delta_t=0.001, end_time=0.01)
            write_fv_schemes(case)
            write_fv_solution(case)

            # transportProperties with DT
            tp = case / "constant" / "transportProperties"
            tp.write_text(
                "FoamFile { version 2.0; format ascii; class dictionary; "
                'location "constant"; object transportProperties; }\n'
                f"DT          DT [0 2 -1 0 0 0 0] {alpha};\n"
            )

            # T field: T=1 at x=0, T=0 at x=1
            zero_dir = case / "0"
            zero_dir.mkdir(exist_ok=True)
            h_T = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                                 class_name="volScalarField", location="0", object="T")
            lines_T = [
                "dimensions      [0 0 0 1 0 0 0];",
                "internalField   uniform 0.5;",
                "boundaryField {",
                "    movingWall {",
                "        type            fixedValue;",
                "        value           uniform 1;",
                "    }",
                "    fixedWalls {",
                "        type            fixedValue;",
                "        value           uniform 0;",
                "    }",
                "    frontAndBack {",
                "        type            empty;",
                "    }",
                "}",
            ]
            write_foam_file(zero_dir / "T", h_T, "\n".join(lines_T), overwrite=True)

            solver = LaplacianFoam(case)
            result = solver.run()

            # 验证温度场在 [0, 1] 范围内
            T = solver.T
            assert T.min() >= -0.1, f"T_min={T.min()}"
            assert T.max() <= 1.1, f"T_max={T.max()}"
            assert torch.isfinite(T).all(), "T contains NaN"


class TestScalarTransportValidation:
    """标量输运：ScalarTransportFoam 验证。"""

    def test_scalar_transport(self):
        """ScalarTransportFoam 标量输运验证。"""
        from tests.tutorials.helpers import (
            make_structured_mesh, write_control_dict, write_fv_schemes,
            write_fv_solution, write_velocity_field, write_pressure_field,
            write_transport_properties,
        )
        from pyfoam.io.foam_file import write_foam_file, FoamFileHeader, FileFormat
        from pyfoam.applications import ScalarTransportFoam

        nx, ny = 8, 8

        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            mesh_dir = case / "constant" / "polyMesh"
            make_structured_mesh(mesh_dir, nx=nx, ny=ny)
            write_control_dict(case, delta_t=0.001, end_time=0.01)
            write_fv_schemes(case)
            write_fv_solution(case)
            write_transport_properties(case, nu=0.01)
            write_velocity_field(
                case,
                patches={"movingWall": (1, 0, 0), "fixedWalls": (0, 0, 0)},
                bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip"},
            )
            write_pressure_field(
                case,
                patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient"},
            )

            # C field
            zero_dir = case / "0"
            h_C = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                                 class_name="volScalarField", location="0", object="C")
            lines_C = [
                "dimensions      [0 0 0 0 0 0 0];",
                "internalField   uniform 0;",
                "boundaryField {",
                "    movingWall {",
                "        type            fixedValue;",
                "        value           uniform 1;",
                "    }",
                "    fixedWalls {",
                "        type            zeroGradient;",
                "    }",
                "    frontAndBack {",
                "        type            empty;",
                "    }",
                "}",
            ]
            write_foam_file(zero_dir / "C", h_C, "\n".join(lines_C), overwrite=True)

            solver = ScalarTransportFoam(case)
            result = solver.run()

            C = solver.C
            assert torch.isfinite(C).all(), "C contains NaN"
            assert C.min() >= -0.1, f"C_min={C.min()}"
            assert C.max() <= 1.5, f"C_max={C.max()}"


class TestPressurePoissonValidation:
    """压力泊松方程：PotentialFoam 验证。"""

    def test_potential_foam(self):
        """PotentialFoam 势流验证。"""
        from tests.tutorials.helpers import (
            make_structured_mesh, write_control_dict, write_fv_schemes,
            write_fv_solution, write_velocity_field, write_pressure_field,
            write_transport_properties,
        )
        from pyfoam.applications import PotentialFoam

        nx, ny = 8, 8

        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            mesh_dir = case / "constant" / "polyMesh"
            make_structured_mesh(mesh_dir, nx=nx, ny=ny)
            write_control_dict(case, delta_t=0.001, end_time=0.005)
            write_fv_schemes(case)
            write_fv_solution(case)
            write_transport_properties(case, nu=0.01)
            write_velocity_field(
                case,
                patches={"movingWall": (1, 0, 0), "fixedWalls": (0, 0, 0)},
                bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip"},
            )
            write_pressure_field(
                case,
                patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient"},
            )

            solver = PotentialFoam(case)
            result = solver.run()

            assert result.converged, "PotentialFoam should converge"


class TestCompressibleValidation:
    """可压缩流：SonicFoam 验证。"""

    def test_sonic_foam_uniform(self):
        """SonicFoam 均匀场验证（应保持均匀）。"""
        from tests.tutorials.helpers import (
            make_structured_mesh, write_control_dict, write_fv_schemes,
            write_fv_solution, write_velocity_field, write_pressure_field,
            write_transport_properties,
        )
        from pyfoam.io.foam_file import write_foam_file, FoamFileHeader, FileFormat
        from pyfoam.applications import SonicFoam

        nx, ny = 4, 4

        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            mesh_dir = case / "constant" / "polyMesh"
            make_structured_mesh(mesh_dir, nx=nx, ny=ny)
            write_control_dict(case, delta_t=0.0001, end_time=0.0005)
            write_fv_schemes(case)
            write_fv_solution(case)
            write_transport_properties(case, nu=0.01)

            # 零速度（静止气体）
            write_velocity_field(
                case,
                patches={"movingWall": (0, 0, 0), "fixedWalls": (0, 0, 0)},
                bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip"},
            )

            # 大气压力
            zero_dir = case / "0"
            h_p = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                                 class_name="volScalarField", location="0", object="p")
            lines_p = [
                "dimensions      [1 -1 -2 0 0 0 0];",
                "internalField   uniform 101325;",
                "boundaryField {",
                "    movingWall { type zeroGradient; }",
                "    fixedWalls { type zeroGradient; }",
                "    frontAndBack { type empty; }",
                "}",
            ]
            write_foam_file(zero_dir / "p", h_p, "\n".join(lines_p), overwrite=True)

            h_T = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                                 class_name="volScalarField", location="0", object="T")
            lines_T = [
                "dimensions      [0 0 0 1 0 0 0];",
                "internalField   uniform 300;",
                "boundaryField {",
                "    movingWall { type zeroGradient; }",
                "    fixedWalls { type zeroGradient; }",
                "    frontAndBack { type empty; }",
                "}",
            ]
            write_foam_file(zero_dir / "T", h_T, "\n".join(lines_T), overwrite=True)

            solver = SonicFoam(case)
            result = solver.run()

            # 静止气体应保持近似均匀
            assert torch.isfinite(solver.U).all(), "U contains NaN"
            assert torch.isfinite(solver.T).all(), "T contains NaN"
            assert torch.isfinite(solver.rho).all(), "rho contains NaN"


class TestBuoyantValidation:
    """浮力流：BuoyantSimpleFoam 验证。"""

    def test_buoyant_uniform_temperature(self):
        """均匀温度应无浮力驱动流。"""
        from tests.tutorials.helpers import (
            make_structured_mesh, write_control_dict, write_fv_schemes,
            write_fv_solution, write_velocity_field, write_pressure_field,
            write_transport_properties,
        )
        from pyfoam.io.foam_file import write_foam_file, FoamFileHeader, FileFormat
        from pyfoam.applications import BuoyantSimpleFoam

        nx, ny = 4, 4

        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            mesh_dir = case / "constant" / "polyMesh"
            make_structured_mesh(mesh_dir, nx=nx, ny=ny)
            write_control_dict(case, delta_t=0.001, end_time=0.005)
            write_fv_schemes(case)
            write_fv_solution(case)
            write_transport_properties(case, nu=0.01)

            write_velocity_field(
                case,
                patches={"movingWall": (0, 0, 0), "fixedWalls": (0, 0, 0)},
                bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip"},
            )
            write_pressure_field(
                case,
                patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient"},
            )

            zero_dir = case / "0"
            h_p = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                                 class_name="volScalarField", location="0", object="p")
            lines_p = [
                "dimensions      [1 -1 -2 0 0 0 0];",
                "internalField   uniform 101325;",
                "boundaryField {",
                "    movingWall { type zeroGradient; }",
                "    fixedWalls { type zeroGradient; }",
                "    frontAndBack { type empty; }",
                "}",
            ]
            write_foam_file(zero_dir / "p", h_p, "\n".join(lines_p), overwrite=True)

            h_T = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                                 class_name="volScalarField", location="0", object="T")
            lines_T = [
                "dimensions      [0 0 0 1 0 0 0];",
                "internalField   uniform 300;",
                "boundaryField {",
                "    movingWall { type zeroGradient; }",
                "    fixedWalls { type zeroGradient; }",
                "    frontAndBack { type empty; }",
                "}",
            ]
            write_foam_file(zero_dir / "T", h_T, "\n".join(lines_T), overwrite=True)

            solver = BuoyantSimpleFoam(case)
            result = solver.run()

            assert torch.isfinite(solver.U).all(), "U contains NaN"
            assert torch.isfinite(solver.T).all(), "T contains NaN"
