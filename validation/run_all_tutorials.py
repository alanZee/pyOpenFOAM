#!/usr/bin/env python3
"""
全量教程验证脚本 — 扫描 OpenFOAM-13 教程，映射到 pyOpenFOAM 求解器，
运行求解器单元测试，并输出综合验证 JSON 报告。

使用方式:
    python validation/run_all_tutorials.py
    python validation/run_all_tutorials.py --skip-tests      # 跳过 pytest，仅生成映射
    python validation/run_all_tutorials.py --verbose          # 详细输出
    python validation/run_all_tutorials.py --output FILE      # 指定输出路径
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. OpenFOAM-13 solver name -> pyOpenFOAM solver class 映射
#    键: OpenFOAM-13 controlDict 中的 solver 字段值
#    值: pyOpenFOAM 应用层类名（可多个，第一个为主要映射）
# ---------------------------------------------------------------------------

OF_SOLVER_TO_PYFOAM = {
    # 不可压缩
    "incompressibleFluid":  ["IncompressibleFluidFoam", "SimpleFoam", "PisoFoam", "PimpleFoam"],
    "fluid":                ["FluidFoam"],
    # VOF 多相
    "incompressibleVoF":         ["IncompressibleVoFFoam"],
    "compressibleVoF":           ["CompressibleVoFFoam"],
    "incompressibleMultiphaseVoF": ["MultiphaseInterFoam"],
    "compressibleMultiphaseVoF":   ["CompressibleMultiphaseVoFFoam"],
    # 漂移通量 / 稠密粒子
    "incompressibleDriftFlux":         ["IncompressibleDriftFluxFoam"],
    "incompressibleDenseParticleFluid": ["DenseParticleFoam"],
    # 多相欧拉
    "multiphaseEuler":     ["MultiphaseEulerFoam", "MultiphaseEulerFoam2", "TwoPhaseEulerFoam", "TwoPhaseEulerFoam2"],
    # 多组分
    "multicomponentFluid": ["MulticomponentFluidFoam", "ReactingFoam"],
    # 激波/可压缩
    "shockFluid":          ["RhoCentralFoam", "SonicFoam"],
    # 固体
    "solidDisplacement":   ["SolidDisplacementFoam"],
    # 预混燃烧
    "XiFluid":             ["XiFoam"],
    # 等温
    "isothermalFluid":     ["IsothermalFluidFoam"],
    # 薄膜
    "isothermalFilm":      ["FilmFoam"],
    "film":                ["FilmFoam"],
    # 动网格
    "movingMesh":          ["FluidFoam"],
    # 多区域 CHT（由 regionSolvers 驱动，映射到 CHTMultiRegionFoam）
    # 需特殊处理
}

# ---------------------------------------------------------------------------
# 2. Legacy 教程路径片段 -> pyOpenFOAM 映射
#    对于没有 controlDict solver 字段的 legacy 教程，根据目录名推断
# ---------------------------------------------------------------------------

LEGACY_PATH_TO_PYFOAM = {
    "legacy/basic/financialFoam":          ["FinancialFoam"],
    "legacy/basic/laplacianFoam":          ["LaplacianFoam"],
    "legacy/compressible/rhoPorousSimpleFoam": ["RhoPorousSimpleFoam"],
    "legacy/electromagnetics/electrostaticFoam": ["ElectrostaticFoam"],
    "legacy/electromagnetics/mhdFoam":     ["MhdFoam"],
    "legacy/incompressible/icoFoam":       ["IcoFoam"],
    "legacy/incompressible/porousSimpleFoam": ["PorousSimpleFoam"],
    "legacy/incompressible/shallowWaterFoam": ["ShallowWaterFoam"],
    "legacy/incompressible/adjointShapeOptimisationFoam": ["AdjointFoam"],
    "legacy/lagrangian/dsmcFoam":          ["DsmcFoam"],
    "legacy/lagrangian/mdFoam":            ["MdFoam"],
    "legacy/lagrangian/mdEquilibrationFoam": ["MdFoam"],
}

# ---------------------------------------------------------------------------
# 3. MultiRegion 教程 -> pyOpenFOAM 映射
# ---------------------------------------------------------------------------

MULTIREGION_SOLVERS = {
    "multiRegion/CHT": ["CHTMultiRegionFoam", "CHTSolver"],
    "multiRegion/film": ["FilmFoam"],
}

# ---------------------------------------------------------------------------
# 3b. 其他路径片段 -> pyOpenFOAM 映射（用于没有 solver 字段的非 legacy 教程）
# ---------------------------------------------------------------------------

PATH_BASED_FALLBACK = {
    "potentialFoam": ["PotentialFoam"],
}

# 非求解器教程（网格生成等），跳过
NON_SOLVER_PATHS = [
    "mesh/",
    "resources/",
    "snappyHexMesh",
]

# ---------------------------------------------------------------------------
# 4. pyOpenFOAM 求解器类名 -> 测试文件映射
#    从 tests/unit/applications/ 中的文件名推导
# ---------------------------------------------------------------------------

# 直接从测试文件名推导：类名 CamelCase -> 文件名 snake_case
def _class_to_test_file(class_name: str) -> str:
    """将 CamelCase 类名转为 snake_case 测试文件名。"""
    s1 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', class_name)
    s2 = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s1)
    return "test_" + s2.lower() + ".py"


# 已知测试文件 -> 求解器类名 映射（手动校验，用于处理特殊情况）
TEST_FILE_TO_SOLVERS = {
    "test_acoustic_foam.py":              ["AcousticFoam"],
    "test_adjoint_foam.py":               ["AdjointFoam"],
    "test_adjoint_shape_foam.py":         ["AdjointShapeFoam"],
    "test_adjoint_turbulence_foam.py":    ["AdjointTurbulenceFoam"],
    "test_boundary_foam.py":              ["BoundaryFoam"],
    "test_buoyant_boussinesq_simple_foam.py": ["BuoyantBoussinesqSimpleFoam"],
    "test_buoyant_pimple_foam.py":        ["BuoyantPimpleFoam"],
    "test_buoyant_simple_foam.py":        ["BuoyantSimpleFoam"],
    "test_cavitating_foam.py":            ["CavitatingFoam"],
    "test_chem_foam.py":                  ["ChemFoam"],
    "test_cht_multi_region_enhanced_foam.py": ["CHTMultiRegionEnhancedFoam"],
    "test_cht_multi_region_foam.py":      ["CHTMultiRegionFoam"],
    "test_cht_solver.py":                 ["CHTSolver"],
    "test_combustion_foam.py":            ["CombustionFoam"],
    "test_compressible_inter_foam.py":    ["CompressibleInterFoam"],
    "test_compressible_inter_foam_2.py":  ["CompressibleInterFoam2"],
    "test_compressible_vof_foam.py":      ["CompressibleVoFFoam"],
    "test_diesel_foam.py":                ["DieselFoam"],
    "test_dsmc_foam.py":                  ["DsmcFoam"],
    "test_electrostatic_foam.py":         ["ElectrostaticFoam"],
    "test_energy_foam.py":                ["EnergyFoam"],
    # test_enhanced_solvers*.py 覆盖多个增强版求解器，单独处理
    "test_enhanced_solvers.py":           ["PisoFoamEnhanced", "PimpleFoamEnhanced",
                                           "SimpleFoamEnhanced", "IcoFoamEnhanced",
                                           "RhoPimpleFoamEnhanced",
                                           "BuoyantSimpleFoamEnhanced",
                                           "BuoyantPimpleFoamEnhanced",
                                           "ReactingFoamEnhanced3"],
    "test_enhanced_solvers_2.py":         ["IcoFoamEnhanced2", "SimpleFoamEnhanced2",
                                           "PisoFoamEnhanced2", "PimpleFoamEnhanced2",
                                           "RhoPimpleFoamEnhanced2",
                                           "BuoyantSimpleFoamEnhanced2",
                                           "BuoyantPimpleFoamEnhanced2",
                                           "ReactingFoamEnhanced4"],
    "test_enhanced_solvers_3.py":         ["SolidFoamEnhanced", "FilmFoamEnhanced",
                                           "SprayFoamEnhanced",
                                           "MultiphaseEulerFoamEnhanced2",
                                           "IcoFoamEnhanced3", "SimpleFoamEnhanced3",
                                           "PisoFoamEnhanced3", "PimpleFoamEnhanced3",
                                           "RhoPimpleFoamEnhanced3",
                                           "BuoyantSimpleFoamEnhanced3",
                                           "BuoyantPimpleFoamEnhanced3",
                                           "ReactingFoamEnhanced5"],
    "test_enhanced_solvers_4.py":         ["SolidFoamEnhanced2", "FilmFoamEnhanced2",
                                           "SprayFoamEnhanced2",
                                           "MultiphaseEulerFoamEnhanced3",
                                           "IcoFoamEnhanced4", "SimpleFoamEnhanced4",
                                           "PisoFoamEnhanced4", "PimpleFoamEnhanced4",
                                           "RhoPimpleFoamEnhanced4",
                                           "BuoyantSimpleFoamEnhanced4",
                                           "BuoyantPimpleFoamEnhanced4",
                                           "ReactingFoamEnhanced6"],
    "test_enhanced_solvers_5.py":         ["SolidFoamEnhanced3", "FilmFoamEnhanced3",
                                           "SprayFoamEnhanced3",
                                           "MultiphaseEulerFoamEnhanced4",
                                           "IcoFoamEnhanced5", "SimpleFoamEnhanced5",
                                           "PisoFoamEnhanced5", "PimpleFoamEnhanced5",
                                           "RhoPimpleFoamEnhanced5",
                                           "BuoyantSimpleFoamEnhanced5",
                                           "BuoyantPimpleFoamEnhanced5",
                                           "ReactingFoamEnhanced7"],
    "test_enhanced_solvers_6.py":         ["SolidFoamEnhanced4", "FilmFoamEnhanced4",
                                           "SprayFoamEnhanced4",
                                           "MultiphaseEulerFoamEnhanced5",
                                           "IcoFoamEnhanced6", "SimpleFoamEnhanced6",
                                           "PisoFoamEnhanced6", "PimpleFoamEnhanced6",
                                           "RhoPimpleFoamEnhanced6",
                                           "BuoyantSimpleFoamEnhanced6",
                                           "BuoyantPimpleFoamEnhanced6",
                                           "ReactingFoamEnhanced8"],
    "test_enhanced_solvers_7.py":         ["SolidFoamEnhanced5", "FilmFoamEnhanced5",
                                           "SprayFoamEnhanced5",
                                           "MultiphaseEulerFoamEnhanced6",
                                           "IcoFoamEnhanced7", "SimpleFoamEnhanced7",
                                           "PisoFoamEnhanced7", "PimpleFoamEnhanced7",
                                           "RhoPimpleFoamEnhanced7",
                                           "BuoyantSimpleFoamEnhanced7",
                                           "BuoyantPimpleFoamEnhanced7",
                                           "ReactingFoamEnhanced9"],
    "test_enhanced_solvers_8.py":         ["SolidFoamEnhanced6", "FilmFoamEnhanced6",
                                           "SprayFoamEnhanced6",
                                           "MultiphaseEulerFoamEnhanced7",
                                           "IcoFoamEnhanced8", "SimpleFoamEnhanced8",
                                           "PisoFoamEnhanced8", "PimpleFoamEnhanced8",
                                           "RhoPimpleFoamEnhanced8",
                                           "BuoyantSimpleFoamEnhanced8",
                                           "BuoyantPimpleFoamEnhanced8",
                                           "ReactingFoamEnhanced10"],
    "test_enhanced_solvers_9.py":         ["SolidFoamEnhanced7", "FilmFoamEnhanced7",
                                           "SprayFoamEnhanced7",
                                           "MultiphaseEulerFoamEnhanced8",
                                           "IcoFoamEnhanced9", "SimpleFoamEnhanced9",
                                           "PisoFoamEnhanced9", "PimpleFoamEnhanced9",
                                           "RhoPimpleFoamEnhanced9",
                                           "BuoyantSimpleFoamEnhanced9",
                                           "BuoyantPimpleFoamEnhanced9",
                                           "ReactingFoamEnhanced11"],
    "test_enhanced_solvers_10.py":        ["SolidFoamEnhanced8", "FilmFoamEnhanced8",
                                           "SprayFoamEnhanced8",
                                           "MultiphaseEulerFoamEnhanced9",
                                           "IcoFoamEnhanced10", "SimpleFoamEnhanced10",
                                           "PisoFoamEnhanced10", "PimpleFoamEnhanced10",
                                           "RhoPimpleFoamEnhanced10",
                                           "BuoyantSimpleFoamEnhanced10",
                                           "BuoyantPimpleFoamEnhanced10",
                                           "ReactingFoamEnhanced12"],
    "test_enhanced_solvers_11.py":        ["SimpleFoamEnhanced11", "PimpleFoamEnhanced11",
                                           "PisoFoamEnhanced11", "IcoFoamEnhanced11",
                                           "BuoyantPimpleFoamEnhanced11",
                                           "BuoyantSimpleFoamEnhanced11",
                                           "CompressibleInterFoamEnhanced11",
                                           "SprayFoamEnhanced11",
                                           "MultiphaseEulerFoamEnhanced11"],
    "test_enhanced_solvers_12.py":        ["SimpleFoamEnhanced12", "PimpleFoamEnhanced12",
                                           "PisoFoamEnhanced12", "IcoFoamEnhanced12",
                                           "BuoyantPimpleFoamEnhanced12",
                                           "BuoyantSimpleFoamEnhanced12",
                                           "CompressibleInterFoamEnhanced12",
                                           "SprayFoamEnhanced12",
                                           "MultiphaseEulerFoamEnhanced12"],
    "test_enhanced_solvers_13.py":        ["SimpleFoamEnhanced13", "PimpleFoamEnhanced13",
                                           "PisoFoamEnhanced13", "IcoFoamEnhanced13",
                                           "BuoyantPimpleFoamEnhanced13",
                                           "BuoyantSimpleFoamEnhanced13",
                                           "CompressibleInterFoamEnhanced13",
                                           "SprayFoamEnhanced13",
                                           "MultiphaseEulerFoamEnhanced13"],
    "test_exports.py":                    [],  # 仅测试导出，不算求解器
    "test_film_foam.py":                  ["FilmFoam", "FilmFoamEnhanced"],
    "test_financial_foam.py":             ["FinancialFoam"],
    "test_financial_foam_2.py":           ["FinancialFoam2"],
    "test_fluid_foam.py":                 ["FluidFoam"],
    "test_heat_transfer_foam.py":         ["HeatTransferFoam"],
    "test_ico_foam.py":                   ["IcoFoam", "IcoFoamEnhanced"],
    "test_incompressible_drift_flux_foam.py": ["IncompressibleDriftFluxFoam"],
    "test_incompressible_fluid_foam.py":  ["IncompressibleFluidFoam"],
    "test_incompressible_vof_foam.py":    ["IncompressibleVoFFoam"],
    "test_inter_foam.py":                 ["InterFoam"],
    "test_isothermal_fluid_foam.py":      ["IsothermalFluidFoam"],
    "test_laplacian_foam.py":             ["LaplacianFoam"],
    "test_magnetic_foam.py":              ["MagneticFoam"],
    "test_md_foam.py":                    ["MdFoam"],
    "test_mhd_foam.py":                   ["MhdFoam"],
    "test_multicomponent_fluid_foam.py":  ["MulticomponentFluidFoam"],
    "test_multiphase_euler_foam_2.py":    ["MultiphaseEulerFoam2", "MultiphaseEulerFoam"],
    "test_multiphase_reacting_foam.py":   ["MultiphaseReactingFoam"],
    "test_pdr_foam.py":                   ["PDRFoam"],
    "test_pimple_foam.py":                ["PimpleFoam"],
    "test_piso_foam.py":                  ["PisoFoam"],
    "test_porous_inter_foam.py":          ["PorousInterFoam"],
    "test_porous_simple_foam.py":         ["PorousSimpleFoam"],
    "test_potential_foam.py":             ["PotentialFoam"],
    "test_reacting_foam.py":              ["ReactingFoam"],
    "test_reacting_foam_enhanced.py":     ["ReactingFoamEnhanced"],
    "test_reacting_foam_enhanced_2.py":   ["ReactingFoam2"],
    "test_reacting_multiphase_foam.py":   ["ReactingMultiphaseFoam"],
    "test_rho_central_foam.py":           ["RhoCentralFoam"],
    "test_rho_pimple_foam.py":            ["RhoPimpleFoam"],
    "test_rho_porous_simple_foam.py":     ["RhoPorousSimpleFoam"],
    "test_rho_simple_foam.py":            ["RhoSimpleFoam"],
    "test_scalar_transport_foam.py":      ["ScalarTransportFoam"],
    "test_shallow_water_foam.py":         ["ShallowWaterFoam"],
    "test_simple_foam.py":                ["SimpleFoam"],
    "test_solid_displacement_foam.py":    ["SolidDisplacementFoam"],
    "test_solid_foam.py":                 ["SolidFoam"],
    "test_sonic_foam.py":                 ["SonicFoam"],
    "test_spray_foam.py":                 ["SprayFoam"],
    "test_spray_foam_2.py":               ["SprayFoam2"],
    "test_srf_simple_foam.py":            ["SrfSimpleFoam"],
    "test_two_phase_euler_foam_2.py":     ["TwoPhaseEulerFoam2", "TwoPhaseEulerFoam"],
    "test_viscous_foam.py":               ["ViscousFoam"],
    "test_xi_foam.py":                    ["XiFoam"],
    "test_dense_particle_foam.py":        ["DenseParticleFoam"],
    "test_multiphase_inter_foam.py":      ["MultiphaseInterFoam"],
    "test_compressible_multiphase_vof_foam.py": ["CompressibleMultiphaseVoFFoam"],
}


# =========================================================================
# 教程扫描
# =========================================================================

def scan_tutorials(tutorials_root: Path) -> list[dict]:
    """扫描 OpenFOAM-13 教程目录，构建教程清单。"""
    tutorials = []

    for control_dict in sorted(tutorials_root.rglob("controlDict")):
        # 跳过 0.orig 下的备份
        if "0.orig" in str(control_dict):
            continue

        # 确定教程路径（相对于 tutorials/ 根目录）
        rel = control_dict.relative_to(tutorials_root)
        # 教程路径 = 去掉 /system/controlDict
        tutorial_path = str(rel.parent.parent).replace("\\", "/")

        # 跳过嵌套的 lesFiles 等非教程目录中的 controlDict
        if tutorial_path.endswith("lesFiles"):
            continue
        if tutorial_path.endswith("motorBike/motorBike"):
            # motorBike/motorBike 也是教程
            pass

        # 读取 solver 字段
        solver_name = _extract_solver_field(control_dict)

        # 判断是否是 multiRegion 教程
        is_multiregion = "multiRegion" in tutorial_path

        # 判断是否是 legacy 教程
        is_legacy = "legacy" in tutorial_path

        # 确定 pyfoam 求解器列表
        if is_multiregion:
            pyfoam_solvers = _get_multiregion_solvers(tutorial_path)
        elif is_legacy:
            pyfoam_solvers = _get_legacy_solvers(tutorial_path, solver_name)
        elif solver_name and solver_name in OF_SOLVER_TO_PYFOAM:
            pyfoam_solvers = OF_SOLVER_TO_PYFOAM[solver_name]
        elif solver_name == "functions":
            # 跳过功能字典（不是实际求解器）
            continue
        else:
            # 尝试路径片段匹配
            pyfoam_solvers = _get_path_based_solvers(tutorial_path)
            if not pyfoam_solvers:
                # 检查是否是非求解器教程（网格生成等）
                skip = False
                for ns_path in NON_SOLVER_PATHS:
                    if ns_path in tutorial_path:
                        skip = True
                        break
                if skip:
                    continue
                # 真正的未知求解器
                pyfoam_solvers = []

        # 获取主要求解器（列表第一个）
        primary_solver = pyfoam_solvers[0] if pyfoam_solvers else None

        tutorials.append({
            "path": tutorial_path,
            "of_solver": solver_name,
            "pyfoam_solvers": pyfoam_solvers,
            "primary_solver": primary_solver,
            "is_legacy": is_legacy,
            "is_multiregion": is_multiregion,
        })

    return tutorials


def _extract_solver_field(control_dict_path: Path) -> str | None:
    """从 controlDict 文件中提取 solver 字段值。"""
    try:
        text = control_dict_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    # 匹配 OpenFOAM 格式: solver          xxx;
    m = re.search(r'^solver\s+(\S+)\s*;', text, re.MULTILINE)
    if m:
        return m.group(1)

    # 尝试匹配 application 格式
    m = re.search(r'^application\s+(\S+)\s*;', text, re.MULTILINE)
    if m:
        return m.group(1)

    # 尝试匹配 regionSolvers 中的主区域
    m = re.search(r'regionSolvers\s*\{[^}]*fluid\s+(\S+)\s*;', text, re.DOTALL)
    if m:
        return m.group(1)

    return None


def _get_multiregion_solvers(tutorial_path: str) -> list[str]:
    """确定 multiRegion 教程的求解器。"""
    for prefix, solvers in MULTIREGION_SOLVERS.items():
        if prefix in tutorial_path:
            return solvers
    return ["CHTMultiRegionFoam"]


def _get_legacy_solvers(tutorial_path: str, solver_name: str | None) -> list[str]:
    """确定 legacy 教程的求解器。"""
    # 先用路径匹配
    for path_key, solvers in LEGACY_PATH_TO_PYFOAM.items():
        if path_key in tutorial_path:
            return solvers

    # 如果有 solver 字段，用通用映射
    if solver_name and solver_name in OF_SOLVER_TO_PYFOAM:
        return OF_SOLVER_TO_PYFOAM[solver_name]

    return []


def _get_path_based_solvers(tutorial_path: str) -> list[str]:
    """根据教程路径片段推断求解器（用于没有 solver 字段且非 legacy/非 multiRegion 的教程）。"""
    for path_key, solvers in PATH_BASED_FALLBACK.items():
        if path_key in tutorial_path:
            return solvers
    return []


# =========================================================================
# 测试运行与解析
# =========================================================================

def run_solver_tests(project_root: Path, verbose: bool = False) -> dict[str, str]:
    """运行 pytest tests/unit/applications/ 并返回每个测试文件的 pass/fail 状态。

    返回: {test_file_name: "PASS" | "FAIL"}
    """
    test_dir = project_root / "tests" / "unit" / "applications"
    if not test_dir.exists():
        print(f"[ERROR] 测试目录不存在: {test_dir}")
        return {}

    # 运行 pytest 获取详细的 per-test 结果
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v", "--tb=no",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    print(f"[INFO] 运行求解器单元测试...")
    print(f"[CMD]  {' '.join(cmd)}")
    print()

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(project_root), env=env,
        timeout=600,
    )

    output = result.stdout + "\n" + result.stderr

    if verbose:
        # 只打印最后 30 行
        lines = output.strip().split("\n")
        for line in lines[-30:]:
            print(f"  {line}")
        print()

    # 解析输出，按测试文件聚合
    file_results = _parse_pytest_output(output)
    return file_results


def _parse_pytest_output(output: str) -> dict[str, str]:
    """解析 pytest 输出，按测试文件聚合 pass/fail 状态。

    pytest -v 输出格式:
      tests/unit/applications/test_ico_foam.py::TestXxx::test_yyy PASSED
      tests/unit/applications/test_ico_foam.py::TestXxx::test_yyy FAILED
    """
    file_status = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0, "xfail": 0})

    # 状态映射
    status_map = {"PASSED": "pass", "FAILED": "fail", "ERROR": "error"}

    for line in output.split("\n"):
        line = line.strip()

        # 匹配 PASSED / FAILED / ERROR / xfail
        # 支持 Windows 路径 (反斜杠) 和 Unix 路径
        for status_str, key in status_map.items():
            if status_str in line and "test_" in line:
                # 提取文件名（兼容 Windows 反斜杠路径）
                m = re.search(r'(test_\w+\.py)', line)
                if m:
                    fname = m.group(1)
                    file_status[fname][key] += 1
                break

        if "xfail" in line and "test_" in line:
            m = re.search(r'(test_\w+\.py)', line)
            if m:
                file_status[m.group(1)]["xfail"] += 1

    # 聚合：任何一个测试 FAILED/ERROR -> 文件 FAIL；否则 PASS
    results = {}
    for fname, counts in file_status.items():
        if counts["fail"] > 0 or counts["error"] > 0:
            results[fname] = "FAIL"
        else:
            results[fname] = "PASS"

    return results


# =========================================================================
# 求解器验证状态判定
# =========================================================================

def build_solver_test_status(test_results: dict[str, str]) -> dict[str, str]:
    """从测试文件结果构建求解器类名 -> 验证状态的映射。

    返回: {solver_class_name: "PASS" | "FAIL" | "NOTEST"}
    """
    solver_status = {}

    for test_file, solvers in TEST_FILE_TO_SOLVERS.items():
        status = test_results.get(test_file, "NOTEST")
        for solver in solvers:
            # 如果同一个求解器有多个测试文件覆盖，取最好的状态
            if solver not in solver_status or (solver_status[solver] != "PASS" and status == "PASS"):
                solver_status[solver] = status

    return solver_status


# =========================================================================
# 结果生成
# =========================================================================

def generate_results(
    tutorials: list[dict],
    solver_status: dict[str, str],
) -> dict:
    """生成最终验证报告。"""

    tutorial_results = []
    validated_count = 0
    untested_count = 0
    failed_count = 0

    for t in tutorials:
        primary = t["primary_solver"]
        status = "SKIP"  # 默认

        if primary:
            test_status = solver_status.get(primary, "NOTEST")
            if test_status == "PASS":
                status = "VALIDATED"
                validated_count += 1
            elif test_status == "FAIL":
                status = "FAILED"
                failed_count += 1
            else:
                status = "UNTESTED"
                untested_count += 1
        else:
            status = "UNMAPPED"
            untested_count += 1

        tutorial_results.append({
            "path": t["path"],
            "of_solver": t["of_solver"],
            "mapped_to": primary,
            "status": status,
            "all_pyfoam_solvers": t["pyfoam_solvers"],
        })

    # 统计各状态教程的求解器分布
    solver_tutorial_count = defaultdict(lambda: {"total": 0, "validated": 0, "failed": 0, "untested": 0})
    for t in tutorial_results:
        s = t["mapped_to"] or t["of_solver"] or "(未映射)"
        solver_tutorial_count[s]["total"] += 1
        if t["status"] == "VALIDATED":
            solver_tutorial_count[s]["validated"] += 1
        elif t["status"] == "FAILED":
            solver_tutorial_count[s]["failed"] += 1
        else:
            solver_tutorial_count[s]["untested"] += 1

    # 收集所有已验证的 pyOpenFOAM 求解器类名
    validated_solvers = sorted(s for s, st in solver_status.items() if st == "PASS")
    failed_solvers = sorted(s for s, st in solver_status.items() if st == "FAIL")
    untested_solvers = sorted(
        s for t in tutorials for s in t["pyfoam_solvers"]
        if solver_status.get(s, "NOTEST") == "NOTEST"
    )

    total = len(tutorials)
    coverage_pct = round(validated_count / total * 100, 1) if total > 0 else 0.0

    report = {
        "date": str(date.today()),
        "total_tutorials": total,
        "validated": validated_count,
        "failed": failed_count,
        "untested": untested_count,
        "coverage_pct": coverage_pct,
        "validated_solvers": validated_solvers,
        "failed_solvers": failed_solvers,
        "untested_solvers": sorted(set(untested_solvers)),
        "solver_tutorial_count": dict(sorted(solver_tutorial_count.items())),
        "tutorials": tutorial_results,
    }

    return report


# =========================================================================
# 主函数
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="全量教程验证: 扫描 OpenFOAM-13 教程 → 映射 pyOpenFOAM 求解器 → 运行测试 → 输出报告"
    )
    parser.add_argument(
        "--tutorials-root",
        type=Path,
        default=None,
        help="OpenFOAM-13 教程根目录 (默认: .reference/OpenFOAM-13/tutorials/)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="输出 JSON 路径 (默认: validation/results/all_tutorials_validation.json)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="跳过 pytest 运行，仅生成映射 (所有测试状态标为 NOTEST)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出",
    )
    args = parser.parse_args()

    # 确定项目根目录
    project_root = Path(__file__).resolve().parent.parent
    tutorials_root = args.tutorials_root or project_root / ".reference" / "OpenFOAM-13" / "tutorials"
    output_path = args.output or project_root / "validation" / "results" / "all_tutorials_validation.json"

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  pyOpenFOAM 全量教程验证")
    print("=" * 70)
    print()
    print(f"  项目根目录:     {project_root}")
    print(f"  教程根目录:     {tutorials_root}")
    print(f"  输出路径:       {output_path}")
    print()

    # ---- 步骤 1: 扫描教程 ----
    print("[1/3] 扫描 OpenFOAM-13 教程...")
    tutorials = scan_tutorials(tutorials_root)
    print(f"  找到 {len(tutorials)} 个教程算例")

    # 统计 OF solver 分布
    of_solver_counts = defaultdict(int)
    for t in tutorials:
        of_solver_counts[t["of_solver"]] += 1
    print(f"  涉及 {len(of_solver_counts)} 种 OpenFOAM 求解器类型:")
    for solver, count in sorted(of_solver_counts.items(), key=lambda x: -x[1]):
        solver_key = solver or "(路径推断)"
        # 收集此 solver 类型实际映射到的所有 pyfoam 求解器
        mapped_names = set()
        for t in tutorials:
            if (t["of_solver"] or "") == (solver or ""):
                if t["primary_solver"]:
                    mapped_names.add(t["primary_solver"])
        mapped_str = ", ".join(sorted(mapped_names)) if mapped_names else "(未映射)"
        print(f"    {solver_key:40s} -> {mapped_str:40s} ({count} 个教程)")
    print()

    # ---- 步骤 2: 运行测试 ----
    solver_status = {}
    if args.skip_tests:
        print("[2/3] 跳过 pytest (--skip-tests)")
        # 所有已知求解器标记为 NOTEST
        for solvers in TEST_FILE_TO_SOLVERS.values():
            for s in solvers:
                solver_status[s] = "NOTEST"
    else:
        print("[2/3] 运行求解器单元测试...")
        test_results = run_solver_tests(project_root, verbose=args.verbose)

        passed_files = sum(1 for v in test_results.values() if v == "PASS")
        failed_files = sum(1 for v in test_results.values() if v == "FAIL")
        print(f"  测试文件: {len(test_results)} 个, 通过 {passed_files} 个, 失败 {failed_files} 个")
        print()

        solver_status = build_solver_test_status(test_results)

    # 统计求解器验证状态
    passed_solvers = sum(1 for v in solver_status.values() if v == "PASS")
    failed_solvers = sum(1 for v in solver_status.values() if v == "FAIL")
    notest_solvers = sum(1 for v in solver_status.values() if v == "NOTEST")
    print(f"  求解器验证状态: 通过 {passed_solvers}, 失败 {failed_solvers}, 无测试 {notest_solvers}")
    print()

    # ---- 步骤 3: 生成报告 ----
    print("[3/3] 生成验证报告...")
    report = generate_results(tutorials, solver_status)

    # 写入 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  已保存: {output_path}")
    print()

    # ---- 打印摘要 ----
    print("=" * 70)
    print("  验证摘要")
    print("=" * 70)
    print()
    print(f"  总教程数:          {report['total_tutorials']}")
    print(f"  已验证 (VALIDATED): {report['validated']}")
    print(f"  测试失败 (FAILED):  {report['failed']}")
    print(f"  未测试 (UNTESTED):  {report['untested']}")
    print(f"  覆盖率:            {report['coverage_pct']}%")
    print()

    # 按求解器类型的覆盖率
    print("  按求解器类型:")
    print(f"  {'OpenFOAM Solver':40s} {'教程数':>6s} {'已验证':>6s} {'覆盖率':>8s}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*8}")
    for solver_name, counts in sorted(report["solver_tutorial_count"].items(), key=lambda x: -x[1]["total"]):
        total = counts["total"]
        validated = counts["validated"]
        pct = round(validated / total * 100, 0) if total > 0 else 0
        print(f"  {solver_name:40s} {total:6d} {validated:6d} {pct:7.0f}%")
    print()

    # 未验证的求解器列表
    if report["untested_solvers"]:
        print("  未测试的 pyOpenFOAM 求解器:")
        for s in report["untested_solvers"][:20]:
            print(f"    - {s}")
        if len(report["untested_solvers"]) > 20:
            print(f"    ... 共 {len(report['untested_solvers'])} 个")
    print()

    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
