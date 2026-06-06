"""
Tutorial validation: OpenFOAM tutorial batch runner.

批量运行 OpenFOAM tutorial 算例并记录结果。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.tutorials.tutorial_parser import TutorialCase, scan_tutorial_case


@dataclass
class TutorialResult:
    """Tutorial 运行结果。"""
    name: str
    solver: str
    status: str  # "pass", "fail", "skip", "error"
    duration: float = 0.0
    error_message: str = ""
    n_cells: int = 0
    max_velocity: float = 0.0
    max_pressure: float = 0.0


def run_tutorial_validation(
    case_dir: Path,
    timeout: float = 300.0,
) -> TutorialResult:
    """运行单个 tutorial 算例验证。

    Args:
        case_dir: Tutorial 算例目录。
        timeout: 超时时间（秒）。

    Returns:
        TutorialResult 对象。
    """
    case = scan_tutorial_case(case_dir)
    result = TutorialResult(name=case.name, solver=case.solver, status="skip")

    if not case.solver:
        result.error_message = "No solver specified"
        return result

    # 映射求解器名称到 pyfoam 模块和类名
    solver_map = {
        "incompressibleFluid": ("simple_foam", "SimpleFoam"),
        "isothermalFluid": ("isothermal_fluid_foam", "IsothermalFluidFoam"),
        "fluid": ("fluid_foam", "FluidFoam"),
        "multicomponentFluid": ("multicomponent_fluid_foam", "MulticomponentFluidFoam"),
        "shockFluid": ("sonic_foam", "SonicFoam"),
        "incompressibleVoF": ("inter_foam", "InterFoam"),
        "compressibleVoF": ("compressible_vof_foam", "CompressibleVoFFoam"),
        "incompressibleMultiphaseVoF": ("incompressible_vof_foam", "IncompressibleVoFFoam"),
        "compressibleMultiphaseVoF": ("compressible_vof_foam", "CompressibleVoFFoam"),
        "multiphaseEuler": ("multiphase_euler_foam", "MultiphaseEulerFoam"),
        "incompressibleDriftFlux": ("incompressible_drift_flux_foam", "IncompressibleDriftFluxFoam"),
        "incompressibleDenseParticleFluid": ("dense_particle_fluid", "DenseParticleFluidFoam"),
        "potentialFoam": ("potential_foam", "PotentialFoam"),
        "solidDisplacement": ("solid_displacement_foam", "SolidDisplacementFoam"),
    }

    solver_info = solver_map.get(case.solver)
    if not solver_info:
        result.error_message = f"Unknown solver: {case.solver}"
        return result

    module_name, solver_name = solver_info

    try:
        # 动态导入求解器
        import importlib
        module = importlib.import_module(f"pyfoam.applications.{module_name}")
        solver_class = getattr(module, solver_name)

        start_time = time.time()
        solver = solver_class(case_dir)
        solver.run()
        duration = time.time() - start_time

        # 检查结果
        if torch.isfinite(solver.U).all() and torch.isfinite(solver.p).all():
            result.status = "pass"
            result.duration = duration
            result.max_velocity = solver.U.abs().max().item()
            result.max_pressure = solver.p.abs().max().item()
        else:
            result.status = "fail"
            result.error_message = "NaN/Inf detected in solution"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)[:200]

    return result


def run_batch_validation(
    tutorials_dir: Path,
    categories: Optional[List[str]] = None,
    max_cases_per_category: int = 5,
    timeout: float = 300.0,
) -> Dict[str, List[TutorialResult]]:
    """批量运行 tutorial 验证。

    Args:
        tutorials_dir: Tutorial 根目录。
        categories: 要验证的类别列表（None 表示全部）。
        max_cases_per_category: 每个类别的最大算例数。
        timeout: 每个算例的超时时间。

    Returns:
        按类别分组的结果字典。
    """
    results: Dict[str, List[TutorialResult]] = {}

    for category_dir in sorted(tutorials_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        if categories and category_dir.name not in categories:
            continue

        category_results = []
        case_count = 0

        for case_dir in sorted(category_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            if not (case_dir / "system").exists():
                continue

            if case_count >= max_cases_per_category:
                break

            result = run_tutorial_validation(case_dir, timeout)
            category_results.append(result)
            case_count += 1

        if category_results:
            results[category_dir.name] = category_results

    return results


def save_results(results: Dict[str, List[TutorialResult]], output_path: Path) -> None:
    """保存验证结果到 JSON 文件。"""
    data = {}
    for category, category_results in results.items():
        data[category] = [asdict(r) for r in category_results]

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_report(results: Dict[str, List[TutorialResult]]) -> str:
    """生成验证报告。"""
    lines = ["# Tutorial 验证报告\n"]
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    total_pass = 0
    total_fail = 0
    total_skip = 0
    total_error = 0

    for category, category_results in sorted(results.items()):
        lines.append(f"\n## {category}\n")
        lines.append("| 算例 | 求解器 | 状态 | 耗时(s) | 备注 |")
        lines.append("|------|--------|------|---------|------|")

        for r in category_results:
            status_icon = {
                "pass": "[PASS]",
                "fail": "[FAIL]",
                "skip": "[SKIP]",
                "error": "[ERR]",
            }.get(r.status, "?")

            lines.append(
                f"| {r.name} | {r.solver} | {status_icon} {r.status} | "
                f"{r.duration:.1f} | {r.error_message[:50]} |"
            )

            if r.status == "pass":
                total_pass += 1
            elif r.status == "fail":
                total_fail += 1
            elif r.status == "skip":
                total_skip += 1
            elif r.status == "error":
                total_error += 1

    lines.append(f"\n## 汇总\n")
    lines.append(f"- 通过: {total_pass}")
    lines.append(f"- 失败: {total_fail}")
    lines.append(f"- 跳过: {total_skip}")
    lines.append(f"- 错误: {total_error}")
    lines.append(f"- 总计: {total_pass + total_fail + total_skip + total_error}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    tutorials_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".reference/OpenFOAM-13/tutorials")
    output_dir = Path("validation/results")

    print("Running tutorial validation...")
    results = run_batch_validation(
        tutorials_dir,
        categories=["incompressibleFluid", "potentialFoam"],
        max_cases_per_category=3,
        timeout=120.0,
    )

    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, output_dir / "tutorial_results.json")

    # 生成报告
    report = generate_report(results)
    (output_dir / "tutorial_report.md").write_text(report, encoding="utf-8")

    print(f"Results saved to {output_dir}")
    print(report)
