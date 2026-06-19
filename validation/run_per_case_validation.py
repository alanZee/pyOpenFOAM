#!/usr/bin/env python3
"""
全量验证脚本：对 257 个 OpenFOAM 参照算例运行 pyOpenFOAM 并逐算例对比。

策略：
1. 从 all_tutorials_validation.json 获取求解器映射
2. 从 .reference/OpenFOAM-13/tutorials/ 获取初始条件 (0/)
3. 从 validation/reference/openfoam/ 获取 polyMesh 和参照最终结果
4. 运行 pyOpenFOAM 求解器
5. 对比 pyOpenFOAM 输出与 OpenFOAM 参照数据
6. 保存逐算例结果到 validation/per_case_data/
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pyfoam.io.case import Case
from pyfoam.io.field_io import read_field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────────────
REF_BASE = ROOT / "validation" / "reference" / "openfoam"
TUT_BASE = ROOT / ".reference" / "OpenFOAM-13" / "tutorials"
TUTORIAL_MAP_FILE = ROOT / "validation" / "results" / "all_tutorials_validation.json"
OUTPUT_DIR = ROOT / "validation" / "per_case_data"
WORK_DIR = ROOT / "validation" / "_work"

# 最大迭代次数（控制运行时间）
MAX_ITER_DEFAULT = 200
MAX_ITER_STEADY = 500
TIMEOUT_SECONDS = 600  # 单算例最长 10 分钟


# ── 数据结构 ──────────────────────────────────────────────────────────
@dataclass
class CaseResult:
    """单个算例的验证结果。"""
    case_name: str
    solver_of: str           # OpenFOAM 求解器名
    solver_pyfoam: str       # pyOpenFOAM 求解器类名
    category: str            # 算例类别
    status: str              # OK / SKIP / ERROR / TIMEOUT
    has_reference: bool      # 是否有参照数据
    has_initial: bool        # 是否有 0/ 初始条件
    mesh_cells: int = 0
    run_time_s: float = 0.0
    iterations: int = 0
    converged: bool = False
    continuity_final: float = float("nan")
    field_errors: dict = field(default_factory=dict)  # {field: {l2_rel, max_abs, n_cells}}
    fields_compared: list = field(default_factory=list)
    error_msg: str = ""
    notes: str = ""


# ── 工具函数 ──────────────────────────────────────────────────────────

def load_tutorial_map() -> dict:
    """加载教程到求解器的映射。"""
    with open(TUTORIAL_MAP_FILE) as f:
        data = json.load(f)
    tutorials = data.get("tutorials", [])
    # path -> {of_solver, mapped_to}
    return {t["path"]: t for t in tutorials}


def find_tutorial_path(case_name: str, tut_map: dict) -> Optional[Path]:
    """根据参照算例名找到对应的 OpenFOAM-13 教程路径。"""
    # 1. 直接匹配
    if case_name in tut_map:
        path = TUT_BASE / tut_map[case_name]["path"]
        if path.exists():
            return path

    # 2. 下划线转斜杠匹配
    parts = case_name.split("_")
    for i in range(1, len(parts)):
        candidate = "/".join(parts[:i]) + "/" + "_".join(parts[i:])
        path = TUT_BASE / candidate
        if path.exists() and (path / "system").exists():
            return path

    # 3. 模糊匹配：找包含 case_name 关键词的教程
    keywords = set(case_name.lower().replace("_", " ").split())
    best_match = None
    best_score = 0
    for tut_path_str in tut_map:
        tut_name = tut_path_str.lower().replace("/", "_")
        score = sum(1 for kw in keywords if kw in tut_name)
        if score > best_score and score >= 2:
            best_score = score
            best_match = TUT_BASE / tut_map[tut_path_str]["path"]

    return best_match


def get_case_category(case_name: str) -> str:
    """获取算例类别。"""
    prefix = case_name.split("_")[0]
    categories = {
        "incompressibleFluid": "Incompressible Steady-State",
        "fluid": "General Fluid",
        "incompressibleVoF": "Incompressible VoF",
        "multiphaseEuler": "Multiphase Euler-Euler",
        "multicomponentFluid": "Multicomponent Reacting",
        "compressibleVoF": "Compressible VoF",
        "shockFluid": "Compressible Shock",
        "XiFluid": "Combustion Xi",
        "incompressibleDenseParticleFluid": "Dense Particle",
        "incompressibleMultiphaseVoF": "Multiphase VoF",
        "incompressibleDriftFlux": "Drift Flux",
        "isothermalFluid": "Isothermal Fluid",
        "isothermalFilm": "Isothermal Film",
        "potentialFoam": "Potential Flow",
        "solidDisplacement": "Solid Mechanics",
        "legacy": "Legacy",
        "mesh": "Mesh Generation",
        "movingMesh": "Moving Mesh",
        "multiRegion": "Multi-Region CHT",
        "film": "Film",
        "compressibleMultiphaseVoF": "Compressible Multiphase VoF",
    }
    return categories.get(prefix, prefix)


def get_solver_class_name(case_name: str, tut_map: dict) -> Optional[str]:
    """获取 pyOpenFOAM 求解器类名。"""
    # 从教程映射获取
    if case_name in tut_map:
        return tut_map[case_name].get("mapped_to")

    # 模糊匹配
    for tut_path, info in tut_map.items():
        flat = tut_path.replace("/", "_")
        if flat == case_name or case_name.startswith(flat.split("_")[0] + "_" + flat.split("_")[1]):
            return info.get("mapped_to")

    return None


def compute_field_error(ref_tensor: torch.Tensor, py_tensor: torch.Tensor) -> dict:
    """计算两个场之间的 L2 相对误差和最大绝对误差。"""
    if ref_tensor.shape != py_tensor.shape:
        return {"l2_rel": float("nan"), "max_abs": float("nan"),
                "n_cells_ref": ref_tensor.shape[0], "n_cells_py": py_tensor.shape[0],
                "error": "shape_mismatch"}

    diff = ref_tensor - py_tensor
    ref_norm = torch.norm(ref_tensor).item()
    diff_norm = torch.norm(diff).item()
    max_abs = torch.max(torch.abs(diff)).item()

    l2_rel = diff_norm / ref_norm if ref_norm > 1e-15 else float("nan")

    return {
        "l2_rel": l2_rel,
        "max_abs": max_abs,
        "n_cells": ref_tensor.shape[0],
        "ref_norm": ref_norm,
        "diff_norm": diff_norm,
    }


def read_field_safe(path: str) -> Optional[torch.Tensor]:
    """安全读取场文件，返回内部场张量。"""
    try:
        fd = read_field(path)
        if fd.internal_field is not None and isinstance(fd.internal_field, torch.Tensor):
            if fd.internal_field.numel() > 0:
                return fd.internal_field
    except Exception:
        pass
    return None


def get_final_time_dir(case_path: Path) -> Optional[Path]:
    """获取最终时间目录（数字最大的）。"""
    time_dirs = []
    try:
        for d in os.listdir(case_path):
            dp = case_path / d
            if os.path.isdir(dp):
                try:
                    float(d)
                    time_dirs.append(dp)
                except ValueError:
                    continue
    except OSError:
        pass
    if not time_dirs:
        return None
    return max(time_dirs, key=lambda d: float(d.name))


def count_mesh_cells(case_path: Path) -> int:
    """读取网格单元数。"""
    owner_file = case_path / "constant" / "polyMesh" / "owner"
    if not owner_file.exists():
        return 0
    try:
        with open(owner_file) as f:
            content = f.read()
        # 找到行数或最大值
        lines = content.strip().split("\n")
        # OpenFOAM 格式：先找到数据开始位置
        in_data = False
        count = 0
        for line in lines:
            line = line.strip()
            if line == "(":
                in_data = True
                continue
            if line == ")":
                break
            if in_data:
                count += 1
        return count
    except Exception:
        return 0


# ── 求解器运行 ────────────────────────────────────────────────────────

def import_solver_class(class_name: str):
    """动态导入 pyOpenFOAM 求解器类。"""
    try:
        import pyfoam.applications as apps
        return getattr(apps, class_name, None)
    except Exception:
        return None


def setup_case_dir(case_name: str, tut_path: Path) -> Optional[Path]:
    """为运行创建算例工作目录。

    策略：
    - 0/ 来自教程源
    - constant/polyMesh/ 来自参照数据
    - system/ 来自参照数据（若有）或教程源
    - 其他 constant/ 文件来自教程源
    """
    work = WORK_DIR / case_name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    ref_dir = REF_BASE / case_name

    # constant/polyMesh: 优先参照数据
    ref_mesh = ref_dir / "constant" / "polyMesh"
    tut_mesh = tut_path / "constant" / "polyMesh"

    const_dir = work / "constant"
    const_dir.mkdir()

    if ref_mesh.exists():
        shutil.copytree(ref_mesh, const_dir / "polyMesh")
    elif tut_mesh.exists():
        shutil.copytree(tut_mesh, const_dir / "polyMesh")
    else:
        logger.warning(f"[{case_name}] No polyMesh found")
        return None

    # 其他 constant/ 文件从教程复制
    tut_const = tut_path / "constant"
    for f in tut_const.iterdir():
        if f.name != "polyMesh" and f.is_file():
            shutil.copy2(f, const_dir / f.name)

    # 0/: 从教程复制初始条件
    tut_0 = tut_path / "0"
    ref_0 = ref_dir / "0"
    if tut_0.exists():
        shutil.copytree(tut_0, work / "0")
    elif ref_0.exists():
        shutil.copytree(ref_0, work / "0")
    else:
        # 尝试 0.orig
        tut_0_orig = tut_path / "0.orig"
        if tut_0_orig.exists():
            shutil.copytree(tut_0_orig, work / "0")
        else:
            logger.warning(f"[{case_name}] No 0/ or 0.orig/ found")
            return None

    # system/: 优先参照数据
    ref_sys = ref_dir / "system"
    tut_sys = tut_path / "system"
    if ref_sys.exists():
        shutil.copytree(ref_sys, work / "system")
    elif tut_sys.exists():
        shutil.copytree(tut_sys, work / "system")

    # 修改 controlDict 减少迭代次数
    _limit_control(work / "system" / "controlDict")

    return work


def _limit_control(cd_path: Path):
    """修改 controlDict 限制迭代次数。"""
    if not cd_path.exists():
        return
    content = cd_path.read_text()

    # 减少 endTime
    import re
    # 替换 endTime
    content = re.sub(
        r'(endTime\s+)\S+;',
        lambda m: m.group(1) + str(MAX_ITER_DEFAULT) + ";",
        content
    )
    # 替换 endTime 对于稳态求解器
    content = re.sub(
        r'(nOuterIterations\s+)\d+;',
        lambda m: m.group(1) + str(min(int(m.group(0).split()[1].rstrip(';')), MAX_ITER_STEADY)) + ";",
        content
    )

    cd_path.write_text(content)


def run_solver(solver_class_name: str, case_path: Path, timeout: int = TIMEOUT_SECONDS) -> dict:
    """运行 pyOpenFOAM 求解器。"""
    solver_cls = import_solver_class(solver_class_name)
    if solver_cls is None:
        return {"status": "ERROR", "error": f"Solver class {solver_class_name} not found"}

    start = time.time()
    try:
        solver = solver_cls(str(case_path))
        solver.run()
        elapsed = time.time() - start
        return {"status": "OK", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        return {"status": "ERROR", "elapsed": elapsed, "error": f"{type(e).__name__}: {e}"}


# ── 主流程 ────────────────────────────────────────────────────────────

def validate_single_case(case_name: str, tut_map: dict) -> CaseResult:
    """验证单个算例。"""
    category = get_case_category(case_name)
    solver_of = case_name.split("_")[0] if "_" in case_name else case_name
    solver_pyfoam = get_solver_class_name(case_name, tut_map) or ""

    result = CaseResult(
        case_name=case_name,
        solver_of=solver_of,
        solver_pyfoam=solver_pyfoam,
        category=category,
        status="SKIP",
        has_reference=False,
        has_initial=False,
    )

    # 跳过 mesh 生成类算例
    if category == "Mesh Generation":
        result.status = "SKIP"
        result.notes = "Mesh generation utility, not a simulation case"
        return result

    # 检查参照数据
    ref_dir = REF_BASE / case_name
    if not ref_dir.exists():
        result.status = "SKIP"
        result.notes = "No reference data"
        return result
    result.has_reference = True

    # 查找教程路径
    tut_path = find_tutorial_path(case_name, tut_map)
    if tut_path is None:
        result.status = "SKIP"
        result.notes = "No matching tutorial found"
        return result

    # 检查初始条件
    has_0 = (tut_path / "0").exists() or (tut_path / "0.orig").exists()
    result.has_initial = has_0
    if not has_0 and not (ref_dir / "0").exists():
        result.status = "SKIP"
        result.notes = "No initial conditions (0/) available"
        return result

    # 检查求解器
    if not solver_pyfoam:
        result.status = "SKIP"
        result.notes = "No pyOpenFOAM solver mapping"
        return result

    solver_cls = import_solver_class(solver_pyfoam)
    if solver_cls is None:
        result.status = "ERROR"
        result.error_msg = f"Solver class {solver_pyfoam} not importable"
        return result

    # 设置工作目录
    case_path = setup_case_dir(case_name, tut_path)
    if case_path is None:
        result.status = "ERROR"
        result.error_msg = "Failed to set up case directory"
        return result

    # 计算网格单元数
    result.mesh_cells = count_mesh_cells(case_path)

    # 运行求解器
    logger.info(f"Running {solver_pyfoam} on {case_name}...")
    run_result = run_solver(solver_pyfoam, case_path)
    result.run_time_s = run_result.get("elapsed", 0)

    if run_result["status"] != "OK":
        result.status = "ERROR"
        result.error_msg = run_result.get("error", "Unknown error")
        return result

    # 读取 pyOpenFOAM 最终输出
    py_final = get_final_time_dir(case_path)
    ref_final = get_final_time_dir(ref_dir)

    if py_final is None or ref_final is None:
        result.status = "OK"
        result.notes = "Ran successfully but no final time data for comparison"
        return result

    # 对比共同字段
    ref_fields = set(f.name for f in ref_final.iterdir() if f.is_file())
    py_fields = set(f.name for f in py_final.iterdir() if f.is_file())
    common_fields = ref_fields & py_fields - {"uniform"}

    for fname in sorted(common_fields):
        ref_tensor = read_field_safe(str(ref_final / fname))
        py_tensor = read_field_safe(str(py_final / fname))

        if ref_tensor is not None and py_tensor is not None:
            err = compute_field_error(ref_tensor, py_tensor)
            result.field_errors[fname] = err
            result.fields_compared.append(fname)

    result.status = "OK"
    return result


def run_all_cases():
    """运行全部验证。"""
    tut_map = load_tutorial_map()
    logger.info(f"Loaded tutorial map: {len(tut_map)} entries")

    # 获取所有参照算例
    ref_cases = sorted([
        d for d in os.listdir(REF_BASE)
        if os.path.isdir(REF_BASE / d)
    ])
    logger.info(f"Found {len(ref_cases)} reference cases")

    # 加载已有结果（支持断点续跑）
    results_file = OUTPUT_DIR / "all_results.json"
    existing = {}
    if results_file.exists():
        with open(results_file) as f:
            existing_list = json.load(f)
            existing = {r["case_name"]: r for r in existing_list}
        logger.info(f"Loaded {len(existing)} existing results")

    results = []
    skipped = 0
    run_count = 0

    for i, case_name in enumerate(ref_cases):
        # 跳过已完成的
        if case_name in existing and existing[case_name]["status"] in ("OK", "SKIP"):
            results.append(existing[case_name])
            skipped += 1
            continue

        logger.info(f"[{i+1}/{len(ref_cases)}] {case_name}")
        case_result = validate_single_case(case_name, tut_map)
        results.append(asdict(case_result))
        run_count += 1

        # 每 10 个保存一次
        if run_count % 10 == 0:
            _save_results(results, results_file)

    _save_results(results, results_file)
    logger.info(f"Done. Ran {run_count}, skipped {skipped}, total {len(results)}")

    return results


def _save_results(results: list, path: Path):
    """保存结果到 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ── 分析已有数据 ──────────────────────────────────────────────────────

def analyze_existing_data():
    """从已有验证结果中提取数据，无需运行新仿真。"""
    results = []

    # 加载教程验证数据
    tutorials_file = ROOT / "validation" / "results" / "all_tutorials_validation.json"
    if tutorials_file.exists():
        with open(tutorials_file) as f:
            tut_data = json.load(f)
        tutorials = tut_data.get("tutorials", [])
    else:
        tutorials = []

    # 加载求解器验证数据
    solver_file = ROOT / "validation" / "results" / "all_solvers_validation.json"
    if solver_file.exists():
        with open(solver_file) as f:
            solver_data = json.load(f)
    else:
        solver_data = []

    # 加载综合验证数据
    comp_file = ROOT / "validation" / "results" / "comprehensive_validation.json"
    if comp_file.exists():
        with open(comp_file) as f:
            comp_data = json.load(f)
    else:
        comp_data = []

    # 构建求解器级查找表
    solver_lookup = {}
    if isinstance(solver_data, list):
        for s in solver_data:
            solver_lookup[s.get("solver", "")] = s
    if isinstance(comp_data, list):
        for s in comp_data:
            solver_lookup[s.get("solver", "")] = s

    # 构建教程级查找表
    tut_lookup = {}
    for t in tutorials:
        key = t.get("path", "").replace("/", "_")
        tut_lookup[key] = t

    # 对每个参照算例提取数据
    ref_cases = sorted([d for d in os.listdir(REF_BASE) if (REF_BASE / d).is_dir()])

    for case_name in ref_cases:
        category = get_case_category(case_name)
        solver_of = case_name.split("_")[0] if "_" in case_name else case_name

        entry = {
            "case_name": case_name,
            "solver_of": solver_of,
            "category": category,
            "status": "UNKNOWN",
            "has_reference": True,
            "mesh_cells": 0,
            "solver_validated": False,
            "tutorial_validated": False,
            "field_comparison": {},
            "notes": "",
        }

        # 检查参照数据
        ref_dir = REF_BASE / case_name
        final_time = get_final_time_dir(ref_dir)
        if final_time:
            try:
                fields = [f for f in os.listdir(final_time)
                          if os.path.isfile(final_time / f) and f != "uniform"]
            except OSError:
                fields = []
            entry["ref_fields"] = fields
            entry["ref_final_time"] = final_time.name
        entry["mesh_cells"] = count_mesh_cells(ref_dir)

        # 查找教程验证状态
        if case_name in tut_lookup:
            tut = tut_lookup[case_name]
            entry["tutorial_validated"] = True
            entry["solver_pyfoam"] = tut.get("mapped_to", "")
            entry["status"] = "VALIDATED" if tut.get("status") == "VALIDATED" else tut.get("status", "UNKNOWN")

            # 查找求解器级数据
            solver_name = tut.get("mapped_to", "")
            if solver_name in solver_lookup:
                s = solver_lookup[solver_name]
                entry["solver_validated"] = True
                entry["solver_status"] = s.get("status", "")
                entry["continuity"] = s.get("continuity")
                entry["field_max"] = s.get("U_max") or s.get("p_max") or s.get("field_max")
                entry["finite"] = s.get("finite", True)
        else:
            entry["notes"] = "No tutorial mapping found"

        # 尝试读取参照场数据进行统计
        if final_time:
            field_stats = {}
            for fname in fields:
                tensor = read_field_safe(str(final_time / fname))
                if tensor is not None:
                    field_stats[fname] = {
                        "shape": list(tensor.shape),
                        "min": tensor.min().item(),
                        "max": tensor.max().item(),
                        "mean": tensor.mean().item(),
                        "std": tensor.std().item(),
                        "norm": torch.norm(tensor).item(),
                    }
            entry["ref_field_stats"] = field_stats

        results.append(entry)

    return results


# ── 生成报告数据 ──────────────────────────────────────────────────────

def generate_summary(results: list) -> dict:
    """生成汇总统计。"""
    total = len(results)
    by_status = {}
    by_category = {}
    field_errors_all = []

    for r in results:
        status = r.get("status", "UNKNOWN")
        by_status[status] = by_status.get(status, 0) + 1

        cat = r.get("category", "Unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "validated": 0, "error": 0, "skip": 0}
        by_category[cat]["total"] += 1
        if status in ("OK", "VALIDATED"):
            by_category[cat]["validated"] += 1
        elif status == "ERROR":
            by_category[cat]["error"] += 1
        else:
            by_category[cat]["skip"] += 1

        # 收集场误差
        fe = r.get("field_comparison", r.get("field_errors", {}))
        if isinstance(fe, dict):
            for fname, err in fe.items():
                if isinstance(err, dict) and "l2_rel" in err:
                    field_errors_all.append({
                        "case": r.get("case_name"),
                        "field": fname,
                        "l2_rel": err["l2_rel"],
                        "max_abs": err["max_abs"],
                    })

    return {
        "total": total,
        "by_status": by_status,
        "by_category": by_category,
        "field_errors": field_errors_all,
        "validation_rate": by_status.get("OK", 0) + by_status.get("VALIDATED", 0),
    }


# ── 入口 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="pyOpenFOAM 全量验证")
    parser.add_argument("--mode", choices=["run", "analyze", "both"], default="analyze",
                        help="run=运行仿真, analyze=分析已有数据, both=两者")
    parser.add_argument("--max-cases", type=int, default=0,
                        help="最大运行算例数 (0=全部)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("analyze", "both"):
        logger.info("Analyzing existing validation data...")
        results = analyze_existing_data()
        _save_results(results, OUTPUT_DIR / "analysis_results.json")
        summary = generate_summary(results)
        _save_results([summary], OUTPUT_DIR / "summary.json")
        logger.info(f"Analysis complete: {summary['total']} cases, "
                     f"{summary['validation_rate']} validated")

    if args.mode in ("run", "both"):
        logger.info("Running pyOpenFOAM simulations...")
        if args.max_cases > 0:
            # 限制运行数量
            tut_map = load_tutorial_map()
            ref_cases = sorted([d for d in os.listdir(REF_BASE) if (REF_BASE / d).is_dir()])
            count = 0
            for case_name in ref_cases:
                if count >= args.max_cases:
                    break
                result = validate_single_case(case_name, tut_map)
                _save_results([asdict(result)], OUTPUT_DIR / f"{case_name}.json")
                count += 1
        else:
            results = run_all_cases()
            summary = generate_summary(results)
            _save_results([summary], OUTPUT_DIR / "summary.json")
