"""
批量 Tutorial 验证运行器。

系统性运行 OpenFOAM-13 tutorial 算例并记录结果。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.mesh_io import read_mesh
from pyfoam.mesh.fv_mesh import FvMesh
from tests.tutorials.helpers import (
    make_structured_mesh,
    write_control_dict,
    write_fv_schemes,
    write_fv_solution,
    write_pressure_field,
    write_transport_properties,
    write_velocity_field,
)


@dataclass
class TutorialResult:
    """单个 tutorial 运行结果。"""
    name: str
    solver: str
    category: str
    status: str  # "pass", "fail", "skip", "error"
    duration: float = 0.0
    n_cells: int = 0
    max_velocity: float = 0.0
    max_pressure: float = 0.0
    error_message: str = ""


# ── 算例定义 ─────────────────────────────────────────────────────────

TUTORIAL_CASES = [
    # 不可压缩流
    ("cavity_10x10", "SimpleFoam", "incompressibleFluid", 10, 10, 0.01, 0.005, 0.5),
    ("cavity_20x20", "SimpleFoam", "incompressibleFluid", 20, 20, 0.01, 0.005, 0.5),
    ("channel_10x5", "SimpleFoam", "incompressibleFluid", 10, 5, 0.01, 0.01, 1.0),
    ("channel_20x10", "SimpleFoam", "incompressibleFluid", 20, 10, 0.01, 0.01, 1.0),
    ("pipe_10x10", "PisoFoam", "incompressibleFluid", 10, 10, 0.01, 0.01, 1.0),
    ("step_20x10", "SimpleFoam", "incompressibleFluid", 20, 10, 1e-5, 0.001, 0.5),
    # 求解器变体
    ("piso_5x5", "PisoFoam", "incompressibleFluid", 5, 5, 0.01, 0.01, 0.05),
    ("pimple_5x5", "PimpleFoam", "incompressibleFluid", 5, 5, 0.01, 0.01, 0.05),
    ("ico_5x5", "IcoFoam", "incompressibleFluid", 5, 5, 0.01, 0.01, 0.05),
    # 网格细化
    ("cavity_3x3", "SimpleFoam", "incompressibleFluid", 3, 3, 0.01, 0.01, 0.1),
    ("cavity_5x5", "SimpleFoam", "incompressibleFluid", 5, 5, 0.01, 0.01, 0.1),
    ("cavity_50x50", "SimpleFoam", "incompressibleFluid", 50, 50, 0.01, 0.001, 0.1),
    # 不同雷诺数
    ("cavity_re1", "SimpleFoam", "incompressibleFluid", 10, 10, 1.0, 0.01, 0.5),
    ("cavity_re100", "SimpleFoam", "incompressibleFluid", 10, 10, 0.01, 0.005, 0.5),
    ("cavity_re1000", "SimpleFoam", "incompressibleFluid", 20, 20, 0.001, 0.001, 0.5),
]


def run_single_tutorial(
    tmp_dir: Path,
    name: str,
    solver_name: str,
    category: str,
    nx: int,
    ny: int,
    nu: float,
    dt: float,
    end_time: float,
) -> TutorialResult:
    """运行单个 tutorial 算例。"""
    result = TutorialResult(name=name, solver=solver_name, category=category, status="skip")

    try:
        case = tmp_dir / name
        mesh_dir = case / "constant" / "polyMesh"
        make_structured_mesh(mesh_dir, nx=nx, ny=ny)
        write_transport_properties(case, nu=nu)
        write_control_dict(case, delta_t=dt, end_time=end_time, write_interval=100)
        write_fv_schemes(case)
        write_fv_solution(case)
        write_velocity_field(
            case,
            patches={"movingWall": (1, 0, 0), "fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
            bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip", "frontAndBack": "empty"},
        )
        write_pressure_field(
            case,
            patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient", "frontAndBack": "empty"},
        )

        # 动态导入求解器
        solver_map = {
            "SimpleFoam": "simple_foam",
            "PisoFoam": "piso_foam",
            "PimpleFoam": "pimple_foam",
            "IcoFoam": "ico_foam",
        }
        module_name = solver_map.get(solver_name)
        if not module_name:
            result.error_message = f"Unknown solver: {solver_name}"
            return result

        import importlib
        module = importlib.import_module(f"pyfoam.applications.{module_name}")
        solver_class = getattr(module, solver_name)

        start = time.time()
        solver = solver_class(case)
        solver.run()
        duration = time.time() - start

        if torch.isfinite(solver.U).all() and torch.isfinite(solver.p).all():
            result.status = "pass"
            result.duration = duration
            result.n_cells = solver.mesh.n_cells
            result.max_velocity = solver.U.abs().max().item()
            result.max_pressure = solver.p.abs().max().item()
        else:
            result.status = "fail"
            result.error_message = "NaN/Inf detected"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)[:200]

    return result


def run_batch(
    tmp_dir: Path,
    cases: Optional[List[tuple]] = None,
) -> List[TutorialResult]:
    """批量运行 tutorial 算例。"""
    if cases is None:
        cases = TUTORIAL_CASES

    results = []
    for case_def in cases:
        name, solver, category, nx, ny, nu, dt, end_time = case_def
        result = run_single_tutorial(tmp_dir, name, solver, category, nx, ny, nu, dt, end_time)
        results.append(result)
        status_icon = {"pass": "[PASS]", "fail": "[FAIL]", "error": "[ERR]", "skip": "[SKIP]"}
        print(f"  {status_icon.get(result.status, '?')} {name} ({result.duration:.1f}s)")

    return results


def save_results(results: List[TutorialResult], output_path: Path) -> None:
    """保存结果到 JSON。"""
    data = [asdict(r) for r in results]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_report(results: List[TutorialResult]) -> str:
    """生成验证报告。"""
    lines = ["# Tutorial 验证报告\n"]
    lines.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    n_pass = sum(1 for r in results if r.status == "pass")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_error = sum(1 for r in results if r.status == "error")
    n_skip = sum(1 for r in results if r.status == "skip")

    lines.append(f"## 汇总\n")
    lines.append(f"- 通过: {n_pass}")
    lines.append(f"- 失败: {n_fail}")
    lines.append(f"- 错误: {n_error}")
    lines.append(f"- 跳过: {n_skip}")
    lines.append(f"- 总计: {len(results)}\n")

    lines.append(f"## 逐算例结果\n")
    lines.append("| 算例 | 求解器 | 类别 | 状态 | 耗时(s) | 单元数 | 最大速度 | 最大压力 |")
    lines.append("|------|--------|------|------|---------|--------|----------|----------|")
    for r in results:
        status_icon = {"pass": "PASS", "fail": "FAIL", "error": "ERR", "skip": "SKIP"}
        lines.append(
            f"| {r.name} | {r.solver} | {r.category} | {status_icon.get(r.status, '?')} | "
            f"{r.duration:.1f} | {r.n_cells} | {r.max_velocity:.3f} | {r.max_pressure:.3f} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    tmp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/tutorial_validation")
    output_dir = Path("validation/results")

    print("Running tutorial validation...")
    results = run_batch(tmp_dir)

    save_results(results, output_dir / "tutorial_results.json")
    report = generate_report(results)
    (output_dir / "tutorial_report.md").write_text(report, encoding="utf-8")

    print(f"\nResults saved to {output_dir}")
    print(report)
