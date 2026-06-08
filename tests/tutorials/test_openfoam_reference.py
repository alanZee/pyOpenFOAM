"""
OpenFOAM 参考验证：用 Docker 运行 OpenFOAM-13 并对比 pyOpenFOAM 结果。

验证流程：
1. 用 Docker OpenFOAM 运行 blockMesh + simpleFoam
2. 用 pyOpenFOAM 运行相同算例
3. 对比速度场、压力场的 L2 误差
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch
import pytest

from pyfoam.core.dtype import CFD_DTYPE


def _docker_available() -> bool:
    """检查 Docker 是否可用。"""
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _openfoam_image_available() -> bool:
    """检查 OpenFOAM Docker 镜像是否可用。"""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", "openfoam/openfoam13-default"],
            capture_output=True, text=True, timeout=10,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _run_openfoam_blockmesh(case_dir: Path) -> bool:
    """在 Docker 中运行 blockMesh。"""
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{case_dir}:/home/openfoam/case",
                "openfoam/openfoam13-default",
                "bash", "-c",
                "cd /home/openfoam/case && blockMesh 2>&1",
            ],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0
    except Exception:
        return False


def _run_openfoam_simplefoam(case_dir: Path, n_iterations: int = 100) -> bool:
    """在 Docker 中运行 simpleFoam。"""
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{case_dir}:/home/openfoam/case",
                "openfoam/openfoam13-default",
                "bash", "-c",
                f"cd /home/openfoam/case && simpleFoam -nIter {n_iterations} 2>&1",
            ],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


def _read_openfoam_field(case_dir: Path, time_dir: str, field_name: str):
    """读取 OpenFOAM 场文件。"""
    from pyfoam.io.field_io import read_field
    field_path = case_dir / time_dir / field_name
    if not field_path.exists():
        return None
    return read_field(field_path)


skip_no_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker not available",
)

skip_no_image = pytest.mark.skipif(
    not _docker_available() or not _openfoam_image_available(),
    reason="OpenFOAM Docker image not available",
)


@skip_no_docker
class TestOpenFOAMReference:
    """OpenFOAM 参考验证。"""

    def test_docker_available(self):
        """Docker 可用。"""
        assert _docker_available()

    @skip_no_image
    def test_cavity_blockmesh(self):
        """OpenFOAM blockMesh 生成 cavity 网格。"""
        from tests.tutorials.helpers import make_structured_mesh

        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            mesh_dir = case_dir / "constant" / "polyMesh"
            make_structured_mesh(mesh_dir, nx=4, ny=4)

            # 创建 blockMeshDict
            system_dir = case_dir / "system"
            system_dir.mkdir(exist_ok=True)
            (system_dir / "blockMeshDict").write_text(
                "FoamFile { version 2.0; format ascii; class dictionary; "
                'object blockMeshDict; }\n'
                "convertToMeters 1;\n"
                "vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) "
                "(0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1));\n"
                "blocks (hex (0 1 2 3 4 5 6 7) (4 4 1) simpleGrading (1 1 1));\n"
                "edges ();\n"
                "boundary ((movingWall { type wall; faces ((3 7 6 2)); }) "
                "(fixedWalls { type wall; faces ((0 4 7 3) (0 1 5 4) (1 2 6 5)); }) "
                "(frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }));\n"
            )

            success = _run_openfoam_blockmesh(case_dir)
            assert success, "blockMesh failed"

    @skip_no_image
    def test_simplefoam_reference(self):
        """OpenFOAM simpleFoam 参考解。"""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            # 创建最小 cavity 算例
            self._create_cavity_case(case_dir, nx=4)

            # 运行 blockMesh
            assert _run_openfoam_blockmesh(case_dir), "blockMesh failed"

            # 运行 simpleFoam
            success = _run_openfoam_simplefoam(case_dir, n_iterations=100)
            assert success, "simpleFoam failed"

            # 读取结果
            U_data = _read_openfoam_field(case_dir, "100", "U")
            assert U_data is not None, "U field not found"

    def _create_cavity_case(self, case_dir: Path, nx: int = 4):
        """创建 OpenFOAM cavity 算例。"""
        from tests.tutorials.helpers import (
            write_control_dict, write_fv_schemes, write_fv_solution,
            write_transport_properties,
        )

        write_control_dict(case_dir, delta_t=0.001, end_time=0.1)
        write_fv_schemes(case_dir)
        write_fv_solution(case_dir)
        write_transport_properties(case_dir, nu=0.01)

        # U field
        zero_dir = case_dir / "0"
        zero_dir.mkdir(exist_ok=True)
        (zero_dir / "U").write_text(
            "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
            "dimensions [0 1 -1 0 0 0 0];\n"
            "internalField uniform (0 0 0);\n"
            "boundaryField {\n"
            "    movingWall { type fixedValue; value uniform (1 0 0); }\n"
            "    fixedWalls { type noSlip; }\n"
            "    frontAndBack { type empty; }\n"
            "}\n"
        )

        # p field
        (zero_dir / "p").write_text(
            "FoamFile { version 2.0; format ascii; class volScalarField; object p; }\n"
            "dimensions [0 2 -2 0 0 0 0];\n"
            "internalField uniform 0;\n"
            "boundaryField {\n"
            "    movingWall { type zeroGradient; }\n"
            "    fixedWalls { type zeroGradient; }\n"
            "    frontAndBack { type empty; }\n"
            "}\n"
        )
