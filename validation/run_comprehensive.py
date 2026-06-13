#!/usr/bin/env python3
"""
综合求解器验证脚本 — 对 pyOpenFOAM 所有主要求解器进行端到端验证。

对每个求解器：
1. 使用其测试用例创建器构建最小 OpenFOAM 算例
2. 运行求解器
3. 记录状态、有限性、连续性误差等指标
4. 汇总保存到 validation/results/comprehensive_validation.json
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# 强制 CPU 运行
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ============================================================================
# 求解器配置列表：(名称, 类别, 模块路径, 类名, 用例创建器模块, 创建器函数, 创建器参数)
# ============================================================================

SOLVER_CONFIGS = [
    # ---- 1. 不可压缩稳态 (SIMPLE) ----
    {
        "name": "SimpleFoam",
        "category": "incompressible_steady",
        "class_module": "pyfoam.applications.simple_foam",
        "class_name": "SimpleFoam",
        "case_module": "tests.unit.applications.test_simple_foam",
        "case_func": "_make_cavity_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, nu=0.01, end_time=5,
                            write_interval=5, max_outer_iterations=10),
    },
    {
        "name": "PorousSimpleFoam",
        "category": "incompressible_steady",
        "class_module": "pyfoam.applications.porous_simple_foam",
        "class_name": "PorousSimpleFoam",
        "case_module": "tests.unit.applications.test_porous_simple_foam",
        "case_func": "_make_porous_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, nu=0.01, end_time=5,
                            write_interval=5, max_outer_iterations=10,
                            d_coeff=(1e2, 1e2, 1e2), f_coeff=(0, 0, 0)),
    },
    {
        "name": "SrfSimpleFoam",
        "category": "incompressible_steady",
        "class_module": "pyfoam.applications.srf_simple_foam",
        "class_name": "SrfSimpleFoam",
        "case_module": "tests.unit.applications.test_srf_simple_foam",
        "case_func": "_make_srf_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, nu=0.01, end_time=5,
                            write_interval=5, max_outer_iterations=10,
                            omega=1.0, axis=(0, 0, 1), origin=(0.5, 0.5, 0)),
    },

    # ---- 2. 不可压缩瞬态 (PISO/PIMPLE) ----
    {
        "name": "IcoFoam",
        "category": "incompressible_transient",
        "class_module": "pyfoam.applications.ico_foam",
        "class_name": "IcoFoam",
        "case_module": "tests.unit.applications.test_ico_foam",
        "case_func": "_make_cavity_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, nu=0.01, delta_t=0.001,
                            end_time=0.01, n_piso_correctors=2),
    },
    {
        "name": "PisoFoam",
        "category": "incompressible_transient",
        "class_module": "pyfoam.applications.piso_foam",
        "class_name": "PisoFoam",
        "case_module": "tests.unit.applications.test_piso_foam",
        "case_func": "_make_cavity_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, nu=0.01, delta_t=0.001,
                            end_time=0.01, n_piso_correctors=2),
    },
    {
        "name": "PimpleFoam",
        "category": "incompressible_transient",
        "class_module": "pyfoam.applications.pimple_foam",
        "class_name": "PimpleFoam",
        "case_module": "tests.unit.applications.test_pimple_foam",
        "case_func": "_make_cavity_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, nu=0.01, delta_t=0.001,
                            end_time=0.01, n_outer_correctors=2, n_correctors=2),
    },

    # ---- 3. 可压缩 ----
    {
        "name": "SonicFoam",
        "category": "compressible",
        "class_module": "pyfoam.applications.sonic_foam",
        "class_name": "SonicFoam",
        "case_module": "tests.unit.applications.test_sonic_foam",
        "case_func": "_make_sod_shock_tube",
        "case_kwargs": dict(n_cells=50, length=1.0, delta_t=1e-4,
                            end_time=5e-4, n_piso_correctors=2, tvd_limiter="vanLeer"),
    },
    {
        "name": "RhoSimpleFoam",
        "category": "compressible",
        "class_module": "pyfoam.applications.rho_simple_foam",
        "class_name": "RhoSimpleFoam",
        "case_module": "tests.unit.applications.test_rho_simple_foam",
        "case_func": "_make_compressible_cavity_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, T_init=300, T_top=310,
                            p_init=101325, end_time=5, write_interval=5,
                            max_outer_iterations=10),
    },
    {
        "name": "RhoPimpleFoam",
        "category": "compressible",
        "class_module": "pyfoam.applications.rho_pimple_foam",
        "class_name": "RhoPimpleFoam",
        "case_module": "tests.unit.applications.test_rho_pimple_foam",
        "case_func": "_make_heated_cavity_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, delta_t=1e-5,
                            end_time=5e-5, n_outer_correctors=2, n_correctors=2,
                            T_hot=350),
    },
    {
        "name": "RhoCentralFoam",
        "category": "compressible",
        "class_module": "pyfoam.applications.rho_central_foam",
        "class_name": "RhoCentralFoam",
        "case_module": "tests.unit.applications.test_rho_central_foam",
        "case_func": "_make_sod_shock_tube",
        "case_kwargs": dict(n_cells=50, length=1.0, delta_t=1e-5,
                            end_time=5e-5, limiter="vanLeer", CFL=0.5),
    },

    # ---- 4. 多相流 ----
    {
        "name": "InterFoam",
        "category": "multiphase",
        "class_module": "pyfoam.applications.inter_foam",
        "class_name": "InterFoam",
        "case_module": "tests.unit.applications.test_inter_foam",
        "case_func": "_make_dam_break_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, dx=1.0, dy=1.0,
                            delta_t=0.001, end_time=0.005, n_outer=2, n_correctors=2),
    },
    {
        "name": "IncompressibleVoFFoam",
        "category": "multiphase",
        "class_module": "pyfoam.applications.incompressible_vof_foam",
        "class_name": "IncompressibleVoFFoam",
        "case_module": "tests.unit.applications.test_incompressible_vof_foam",
        "case_func": "_make_dam_break_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, dx=1.0, dy=1.0,
                            delta_t=0.001, end_time=0.005, n_outer=2, n_correctors=2),
    },
    {
        "name": "CavitatingFoam",
        "category": "multiphase",
        "class_module": "pyfoam.applications.cavitating_foam",
        "class_name": "CavitatingFoam",
        "case_module": "tests.unit.applications.test_cavitating_foam",
        "case_func": "_make_cavitation_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, dx=1.0, dy=1.0,
                            delta_t=0.001, end_time=0.005, n_outer=2, n_correctors=2),
    },

    # ---- 5. 传热 ----
    {
        "name": "LaplacianFoam",
        "category": "heat_transfer",
        "class_module": "pyfoam.applications.laplacian_foam",
        "class_name": "LaplacianFoam",
        "case_module": "tests.unit.applications.test_laplacian_foam",
        "case_func": "_make_diffusion_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, T_init=300, T_hot=400,
                            T_cold=200, D=1.0, end_time=1, delta_t=0.01,
                            write_interval=100),
    },
    {
        "name": "BuoyantSimpleFoam",
        "category": "heat_transfer",
        "class_module": "pyfoam.applications.buoyant_simple_foam",
        "class_name": "BuoyantSimpleFoam",
        "case_module": "tests.unit.applications.test_buoyant_simple_foam",
        "case_func": "_make_buoyant_cavity_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, T_init=300, T_hot=310,
                            T_cold=290, p_init=101325, end_time=2,
                            write_interval=2, max_outer_iterations=10),
    },
    {
        "name": "BuoyantPimpleFoam",
        "category": "heat_transfer",
        "class_module": "pyfoam.applications.buoyant_pimple_foam",
        "class_name": "BuoyantPimpleFoam",
        "case_module": "tests.unit.applications.test_buoyant_pimple_foam",
        "case_func": "_make_buoyant_pimple_cavity_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, T_init=300, T_hot=310,
                            T_cold=290, p_init=101325, delta_t=1e-4,
                            end_time=5e-4, n_outer_correctors=2, n_correctors=2),
    },

    # ---- 6. 燃烧/化学 ----
    {
        "name": "ReactingFoam",
        "category": "combustion",
        "class_module": "pyfoam.applications.reacting_foam",
        "class_name": "ReactingFoam",
        "case_module": "tests.unit.applications.test_reacting_foam",
        "case_func": "_make_reacting_case",
        "case_kwargs": dict(n_cells=3, L=1.0, end_time=2, delta_t=0.1,
                            write_interval=100, T_init=300, Y_A_init=1.0),
    },
    {
        "name": "ChemFoam",
        "category": "combustion",
        "class_module": "pyfoam.applications.chem_foam",
        "class_name": "ChemFoam",
        "case_module": "tests.unit.applications.test_chem_foam",
        "case_func": "_make_chem_case",
        "case_kwargs": dict(n_cells=1, end_time=1, delta_t=0.001,
                            write_interval=100, T_init=1000, Y_A_init=1.0),
    },

    # ---- 7. 特殊求解器 ----
    {
        "name": "PotentialFoam",
        "category": "special",
        "class_module": "pyfoam.applications.potential_foam",
        "class_name": "PotentialFoam",
        "case_module": "tests.unit.applications.test_potential_foam",
        "case_func": "_make_potential_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, L=1.0, end_time=50,
                            write_interval=100, phi_inlet=1.0),
    },
    {
        "name": "ScalarTransportFoam",
        "category": "special",
        "class_module": "pyfoam.applications.scalar_transport_foam",
        "class_name": "ScalarTransportFoam",
        "case_module": "tests.unit.applications.test_scalar_transport_foam",
        "case_func": "_make_scalar_case",
        "case_kwargs": dict(n_cells=5, L=1.0, D=0.01, C_inlet=1.0,
                            end_time=10, delta_t=1.0, write_interval=100),
    },
    {
        "name": "DsmcFoam",
        "category": "special",
        "class_module": "pyfoam.applications.dsmc_foam",
        "class_name": "DsmcFoam",
        "case_module": "tests.unit.applications.test_dsmc_foam",
        "case_func": "_make_dsmc_case",
        "case_kwargs": dict(n_cells=4, L=1.0, end_time=2, delta_t=0.01,
                            T_init=300),
    },
    {
        "name": "AcousticFoam",
        "category": "special",
        "class_module": "pyfoam.applications.acoustic_foam",
        "class_name": "AcousticFoam",
        "case_module": "tests.unit.applications.test_acoustic_foam",
        "case_func": "_make_acoustic_case",
        "case_kwargs": dict(n_cells_x=8, n_cells_y=8, p_init=0, p_source=1.0,
                            end_time=50, delta_t=0.001, write_interval=100),
    },
    {
        "name": "MagneticFoam",
        "category": "special",
        "class_module": "pyfoam.applications.magnetic_foam",
        "class_name": "MagneticFoam",
        "case_module": "tests.unit.applications.test_magnetic_foam",
        "case_func": "_make_magnetic_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, mu0=1.0, end_time=50,
                            write_interval=100),
    },
    {
        "name": "MhdFoam",
        "category": "special",
        "class_module": "pyfoam.applications.mhd_foam",
        "class_name": "MhdFoam",
        "case_module": "tests.unit.applications.test_mhd_foam",
        "case_func": "_make_mhd_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, nu=1.0, mu0=1.0,
                            sigma=1.0, end_time=1, delta_t=0.01,
                            write_interval=1, U_inlet=1.0, B_applied=0.1),
    },
    {
        "name": "ShallowWaterFoam",
        "category": "special",
        "class_module": "pyfoam.applications.shallow_water_foam",
        "class_name": "ShallowWaterFoam",
        "case_module": "tests.unit.applications.test_shallow_water_foam",
        "case_func": "_make_shallow_water_case",
        "case_kwargs": dict(n_cells_x=8, n_cells_y=4, g=9.81, f=0, Cf=0,
                            delta_t=0.001, end_time=0.005,
                            n_piso_correctors=2, h_left=2.0, h_right=1.0),
    },
    {
        "name": "FinancialFoam",
        "category": "special",
        "class_module": "pyfoam.applications.financial_foam",
        "class_name": "FinancialFoam",
        "case_module": "tests.unit.applications.test_financial_foam",
        "case_func": "_make_financial_case",
        "case_kwargs": dict(n_cells=10, end_time=1, delta_t=0.01),
    },
    {
        "name": "MdFoam",
        "category": "special",
        "class_module": "pyfoam.applications.md_foam",
        "class_name": "MdFoam",
        "case_module": "tests.unit.applications.test_md_foam",
        "case_func": "_make_md_case",
        "case_kwargs": dict(n_cells=2, end_time=1, delta_t=0.001),
    },
    {
        "name": "BoundaryFoam",
        "category": "special",
        "class_module": "pyfoam.applications.boundary_foam",
        "class_name": "BoundaryFoam",
        "case_module": "tests.unit.applications.test_boundary_foam",
        "case_func": "_make_bl_case",
        "case_kwargs": dict(n_cells=5, y_max=1.0, nu=0.01, dp_dx=1.0,
                            end_time=5, write_interval=5),
    },

    # ---- 8. 额外的基础求解器 ----
    {
        "name": "IncompressibleFluidFoam",
        "category": "unified",
        "class_module": "pyfoam.applications.incompressible_fluid_foam",
        "class_name": "IncompressibleFluidFoam",
        "case_module": "tests.unit.applications.test_incompressible_fluid_foam",
        "case_func": "_make_cavity_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, nu=0.01, delta_t=1.0,
                            end_time=2, write_interval=2),
    },
    {
        "name": "ElectrostaticFoam",
        "category": "special",
        "class_module": "pyfoam.applications.electrostatic_foam",
        "class_name": "ElectrostaticFoam",
        "case_module": "tests.unit.applications.test_electrostatic_foam",
        "case_func": "_make_electrostatic_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, end_time=50, write_interval=100),
    },
    {
        "name": "CompressibleInterFoam",
        "category": "multiphase",
        "class_module": "pyfoam.applications.compressible_inter_foam",
        "class_name": "CompressibleInterFoam",
        "case_module": "tests.unit.applications.test_compressible_inter_foam",
        "case_func": "_make_dam_break_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, dx=1.0, dy=1.0,
                            delta_t=0.001, end_time=0.005, n_outer=2, n_correctors=2),
    },
    {
        "name": "CompressibleVoFFoam",
        "category": "multiphase",
        "class_module": "pyfoam.applications.compressible_vof_foam",
        "class_name": "CompressibleVoFFoam",
        "case_module": "tests.unit.applications.test_compressible_vof_foam",
        "case_func": "_make_dam_break_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, dx=1.0, dy=1.0,
                            delta_t=0.001, end_time=0.005, n_outer=2, n_correctors=2),
    },
    {
        "name": "TwoPhaseEulerFoam",
        "category": "multiphase",
        "class_module": "pyfoam.applications.two_phase_euler_foam",
        "class_name": "TwoPhaseEulerFoam",
        "case_module": "tests.unit.applications.test_two_phase_euler_foam_2",
        "case_func": "_skip",
        "case_kwargs": dict(),
        "_skip": True,
        "_skip_reason": "No case creator in test_two_phase_euler_foam_2 (component-level tests only)",
    },
    {
        "name": "PDRFoam",
        "category": "combustion",
        "class_module": "pyfoam.applications.pdr_foam",
        "class_name": "PDRFoam",
        "case_module": "tests.unit.applications.test_pdr_foam",
        "case_func": "_make_pdr_case",
        "case_kwargs": dict(n_cells=3, L=1.0, end_time=1e-4, delta_t=1e-5,
                            n_outer_correctors=2, n_correctors=2,
                            p_init=101325, b_init=0),
    },
    {
        "name": "FluidFoam",
        "category": "compressible",
        "class_module": "pyfoam.applications.fluid_foam",
        "class_name": "FluidFoam",
        "case_module": "tests.unit.applications.test_fluid_foam",
        "case_func": "_make_fluid_case",
        "case_kwargs": dict(n_cells=3, L=1.0, end_time=1e-4, delta_t=1e-5),
    },
    {
        "name": "IsothermalFluidFoam",
        "category": "compressible",
        "class_module": "pyfoam.applications.isothermal_fluid_foam",
        "class_name": "IsothermalFluidFoam",
        "case_module": "tests.unit.applications.test_isothermal_fluid_foam",
        "case_func": "_make_isothermal_case",
        "case_kwargs": dict(n_cells=3, L=1.0, end_time=1, delta_t=0.01),
    },
    {
        "name": "IncompressibleDriftFluxFoam",
        "category": "multiphase",
        "class_module": "pyfoam.applications.incompressible_drift_flux_foam",
        "class_name": "IncompressibleDriftFluxFoam",
        "case_module": "tests.unit.applications.test_incompressible_drift_flux_foam",
        "case_func": "_make_settling_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, dx=0.01, dy=0.01,
                            delta_t=1e-4, end_time=5e-4),
    },
    {
        "name": "SolidDisplacementFoam",
        "category": "solid",
        "class_module": "pyfoam.applications.solid_displacement_foam",
        "class_name": "SolidDisplacementFoam",
        "case_module": "tests.unit.applications.test_solid_displacement_foam",
        "case_func": "_make_solid_case",
        "case_kwargs": dict(n_cells=3, L=1.0, E=1e9, nu=0.3, end_time=50,
                            write_interval=100),
    },
    {
        "name": "MulticomponentFluidFoam",
        "category": "combustion",
        "class_module": "pyfoam.applications.multicomponent_fluid_foam",
        "class_name": "MulticomponentFluidFoam",
        "case_module": "tests.unit.applications.test_multicomponent_fluid_foam",
        "case_func": "_make_multicomponent_case",
        "case_kwargs": dict(n_cells=3, L=1.0, end_time=3e-5, delta_t=1e-5),
    },
    {
        "name": "ViscousFoam",
        "category": "special",
        "class_module": "pyfoam.applications.viscous_foam",
        "class_name": "ViscousFoam",
        "case_module": "tests.unit.applications.test_viscous_foam",
        "case_func": "_make_viscous_case",
        "case_kwargs": dict(n_cells=3, end_time=10, delta_t=1.0, write_interval=100),
    },
    {
        "name": "AdjointFoam",
        "category": "special",
        "class_module": "pyfoam.applications.adjoint_foam",
        "class_name": "AdjointFoam",
        "case_module": "tests.unit.applications.test_adjoint_foam",
        "case_func": "_make_adjoint_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, nu=0.01, end_time=2,
                            write_interval=2),
    },
    {
        "name": "PorousInterFoam",
        "category": "multiphase",
        "class_module": "pyfoam.applications.porous_inter_foam",
        "class_name": "PorousInterFoam",
        "case_module": "tests.unit.applications.test_porous_inter_foam",
        "case_func": "_make_porous_inter_case",
        "case_kwargs": dict(n_cells_x=4, n_cells_y=4, end_time=0.01, write_interval=0.01),
    },
]


def _resolve_case_creator(cfg: dict):
    """动态导入求解器的用例创建器和求解器类。"""
    # 导入求解器类
    import importlib
    solver_mod = importlib.import_module(cfg["class_module"])
    solver_cls = getattr(solver_mod, cfg["class_name"])

    # 导入用例创建器
    case_mod = importlib.import_module(cfg["case_module"])
    case_func = getattr(case_mod, cfg["case_func"])

    return solver_cls, case_func


def _extract_results(solver) -> dict:
    """从求解器实例中提取关键指标。"""
    results = {}
    all_finite = True

    # U (速度场) — 部分求解器无 U 场 (如 LaplacianFoam, ChemFoam)
    if hasattr(solver, "U") and solver.U is not None:
        U = solver.U
        u_finite = bool(torch.isfinite(U).all().item())
        all_finite = all_finite and u_finite
        results["U_max"] = float(U.abs().max().item()) if U.numel() > 0 else 0.0
        results["U_shape"] = list(U.shape)
    else:
        results["U_max"] = None
        results["U_shape"] = None

    # p (压力场) — 部分求解器无 p 场
    if hasattr(solver, "p") and solver.p is not None:
        p = solver.p
        p_finite = bool(torch.isfinite(p).all().item())
        all_finite = all_finite and p_finite
        results["p_max"] = float(p.abs().max().item()) if p.numel() > 0 else 0.0
        results["p_shape"] = list(p.shape)
    else:
        results["p_max"] = None
        results["p_shape"] = None

    # phi (通量场)
    if hasattr(solver, "phi") and solver.phi is not None:
        phi = solver.phi
        phi_finite = bool(torch.isfinite(phi).all().item())
        all_finite = all_finite and phi_finite
        results["phi_max"] = float(phi.abs().max().item()) if phi.numel() > 0 else 0.0
    else:
        results["phi_max"] = None

    # T (温度场) — 传热/可压缩求解器
    if hasattr(solver, "T") and solver.T is not None:
        T = solver.T
        t_finite = bool(torch.isfinite(T).all().item())
        all_finite = all_finite and t_finite
        results["T_max"] = float(T.abs().max().item()) if T.numel() > 0 else 0.0
    else:
        results["T_max"] = None

    # alpha (体积分数) — 多相求解器
    for attr_name in ("alpha", "alpha1", "alpha_water"):
        if hasattr(solver, attr_name) and getattr(solver, attr_name) is not None:
            alpha = getattr(solver, attr_name)
            a_finite = bool(torch.isfinite(alpha).all().item())
            all_finite = all_finite and a_finite
            results["alpha_max"] = float(alpha.max().item()) if alpha.numel() > 0 else 0.0
            results["alpha_min"] = float(alpha.min().item()) if alpha.numel() > 0 else 0.0
            break
    else:
        results["alpha_max"] = None
        results["alpha_min"] = None

    # 连续性误差 (如有)
    results["continuity"] = None
    results["finite"] = all_finite

    return results


def run_single_solver(cfg: dict) -> dict:
    """运行单个求解器并返回结果。"""
    result = {
        "solver": cfg["name"],
        "category": cfg["category"],
        "status": "OK",
        "finite": None,
        "continuity": None,
        "U_max": None,
        "p_max": None,
        "phi_max": None,
        "T_max": None,
        "alpha_max": None,
        "alpha_min": None,
        "error": None,
    }

    case_dir = None
    try:
        # 导入求解器类和用例创建器
        solver_cls, case_func = _resolve_case_creator(cfg)

        # 创建临时目录
        case_dir = Path(tempfile.mkdtemp(prefix=f"val_{cfg['name']}_"))

        # 创建算例
        case_func(case_dir, **cfg["case_kwargs"])

        # 初始化求解器
        solver = solver_cls(case_dir)

        # 运行求解器
        conv = solver.run()

        # 提取连续性误差
        if conv is not None and hasattr(conv, "continuity_error"):
            result["continuity"] = float(conv.continuity_error) if conv.continuity_error is not None else None

        # 提取结果指标
        metrics = _extract_results(solver)
        result["finite"] = metrics["finite"]
        result["U_max"] = metrics["U_max"]
        result["p_max"] = metrics["p_max"]
        result["phi_max"] = metrics["phi_max"]
        result["T_max"] = metrics.get("T_max")
        result["alpha_max"] = metrics.get("alpha_max")
        result["alpha_min"] = metrics.get("alpha_min")

        # 检查 NaN
        if result["finite"] is False:
            result["status"] = "NaN"

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        import traceback as tb
        tb.print_exc()

    finally:
        # 清理临时目录
        if case_dir and case_dir.exists():
            try:
                shutil.rmtree(case_dir, ignore_errors=True)
            except Exception:
                pass

    return result


def _find_case_func_name(test_module_path: str) -> str | None:
    """从测试模块中找到第一个 _make_ 开头的函数名。"""
    try:
        import importlib
        mod = importlib.import_module(test_module_path)
        for name in dir(mod):
            if name.startswith("_make_"):
                return name
    except Exception:
        pass
    return None


def _resolve_configs_with_fallback(configs: list[dict]) -> list[dict]:
    """对于 case_kwargs 为空且 case_func 为 '_make_case' 的配置，自动发现正确的函数名。"""
    import importlib
    resolved = []
    for cfg in configs:
        new_cfg = dict(cfg)
        if cfg["case_func"] == "_make_case" and not cfg["case_kwargs"]:
            func_name = _find_case_func_name(cfg["case_module"])
            if func_name:
                new_cfg["case_func"] = func_name
            else:
                new_cfg["_skip"] = True
                new_cfg["_skip_reason"] = f"No case creator found in {cfg['case_module']}"
        resolved.append(new_cfg)
    return resolved


def main():
    """运行所有求解器验证并保存结果。"""
    print("=" * 70)
    print("pyOpenFOAM 综合求解器验证")
    print("=" * 70)

    # 解析配置
    configs = _resolve_configs_with_fallback(SOLVER_CONFIGS)

    results = []
    ok_count = 0
    fail_count = 0
    nan_count = 0
    skip_count = 0

    total = len(configs)

    for i, cfg in enumerate(configs, 1):
        name = cfg["name"]
        print(f"\n[{i}/{total}] {name} ({cfg['category']})")
        print("-" * 50)

        if cfg.get("_skip"):
            print(f"  SKIP: {cfg['_skip_reason']}")
            results.append({
                "solver": name,
                "category": cfg["category"],
                "status": "SKIP",
                "finite": None,
                "continuity": None,
                "U_max": None,
                "p_max": None,
                "phi_max": None,
                "error": cfg["_skip_reason"],
            })
            skip_count += 1
            continue

        result = run_single_solver(cfg)
        results.append(result)

        status = result["status"]
        if status == "OK":
            ok_count += 1
            u_str = f"{result['U_max']:.6e}" if result['U_max'] is not None else "N/A"
            p_str = f"{result['p_max']:.6e}" if result['p_max'] is not None else "N/A"
            t_str = f"{result['T_max']:.6e}" if result.get('T_max') is not None else ""
            cont_str = f"{result['continuity']:.6e}" if result['continuity'] is not None else "N/A"
            extra = f"  T_max={t_str}" if t_str else ""
            print(f"  status=OK  finite={result['finite']}  "
                  f"U_max={u_str}  p_max={p_str}{extra}  "
                  f"continuity={cont_str}")
        elif status == "NaN":
            nan_count += 1
            u_str = f"{result['U_max']:.6e}" if result['U_max'] is not None else "N/A"
            p_str = f"{result['p_max']:.6e}" if result['p_max'] is not None else "N/A"
            print(f"  status=NaN  finite={result['finite']}  "
                  f"U_max={u_str}  p_max={p_str}")
        elif status == "SKIP":
            skip_count += 1
            print(f"  status=SKIP")
        else:
            fail_count += 1
            print(f"  status=FAIL  error={result['error']}")

    # 汇总
    print("\n" + "=" * 70)
    print("验证汇总")
    print("=" * 70)
    print(f"  总计: {total}")
    print(f"  OK:   {ok_count}")
    print(f"  NaN:  {nan_count}")
    print(f"  FAIL: {fail_count}")
    print(f"  SKIP: {skip_count}")
    print("=" * 70)

    # 保存结果
    output_dir = ROOT / "validation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comprehensive_validation.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_path}")

    # 打印分类统计
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "ok": 0, "fail": 0, "nan": 0, "skip": 0}
        categories[cat]["total"] += 1
        if r["status"] == "OK":
            categories[cat]["ok"] += 1
        elif r["status"] == "NaN":
            categories[cat]["nan"] += 1
        elif r["status"] == "SKIP":
            categories[cat]["skip"] += 1
        else:
            categories[cat]["fail"] += 1

    print("\n分类统计:")
    for cat, stats in categories.items():
        print(f"  {cat:30s}: {stats['ok']}/{stats['total']} OK, "
              f"{stats['fail']} FAIL, {stats['nan']} NaN, {stats['skip']} SKIP")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
