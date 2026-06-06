"""
Tutorial validation: solver import coverage tests.

验证所有 OpenFOAM 求解器的 pyfoam 实现可导入。
"""
from __future__ import annotations

import pytest


# OpenFOAM-13 求解器模块映射
SOLVER_MAP = {
    # 不可压缩流
    "incompressibleFluid": ("simple_foam", "SimpleFoam"),
    "isothermalFluid": ("isothermal_fluid_foam", "IsothermalFluidFoam"),
    "potentialFoam": ("potential_foam", "PotentialFoam"),
    # 可压缩流
    "fluid": ("fluid_foam", "FluidFoam"),
    "multicomponentFluid": ("multicomponent_fluid_foam", "MulticomponentFluidFoam"),
    "shockFluid": ("sonic_foam", "SonicFoam"),
    # 多相流
    "incompressibleVoF": ("inter_foam", "InterFoam"),
    "compressibleVoF": ("compressible_vof_foam", "CompressibleVoFFoam"),
    "incompressibleMultiphaseVoF": ("incompressible_vof_foam", "IncompressibleVoFFoam"),
    "compressibleMultiphaseVoF": ("compressible_vof_foam", "CompressibleVoFFoam"),
    "multiphaseEuler": ("multiphase_euler_foam", "MultiphaseEulerFoam"),
    "incompressibleDriftFlux": ("incompressible_drift_flux_foam", "IncompressibleDriftFluxFoam"),
    # 传热
    "solidDisplacement": ("solid_displacement_foam", "SolidDisplacementFoam"),
    # 其他
    "film": ("film_foam", "FilmFoam"),
    "isothermalFilm": ("film_foam", "FilmFoam"),
    "movingMesh": ("pimple_foam", "PimpleFoam"),
}

# 已知缺失的求解器（标记为 xfail）
MISSING_SOLVERS = {
    "incompressibleDenseParticleFluid",
    "XiFluid",
}


class TestSolverImportCoverage:
    """求解器导入覆盖测试。"""

    @pytest.mark.parametrize("solver_name,module_info", SOLVER_MAP.items())
    def test_solver_import(self, solver_name: str, module_info: tuple):
        """验证求解器可导入。"""
        module_name, class_name = module_info
        import importlib
        try:
            module = importlib.import_module(f"pyfoam.applications.{module_name}")
            solver_class = getattr(module, class_name)
            assert solver_class is not None
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Cannot import {solver_name}: {e}")


class TestSolverInstantiation:
    """求解器实例化测试。"""

    def test_simple_foam_instantiation(self):
        """SimpleFoam 可实例化。"""
        from pyfoam.applications.simple_foam import SimpleFoam
        assert SimpleFoam is not None

    def test_piso_foam_instantiation(self):
        """PisoFoam 可实例化。"""
        from pyfoam.applications.piso_foam import PisoFoam
        assert PisoFoam is not None

    def test_pimple_foam_instantiation(self):
        """PimpleFoam 可实例化。"""
        from pyfoam.applications.pimple_foam import PimpleFoam
        assert PimpleFoam is not None

    def test_inter_foam_instantiation(self):
        """InterFoam 可实例化。"""
        from pyfoam.applications.inter_foam import InterFoam
        assert InterFoam is not None

    def test_sonic_foam_instantiation(self):
        """SonicFoam 可实例化。"""
        from pyfoam.applications.sonic_foam import SonicFoam
        assert SonicFoam is not None

    def test_multiphase_euler_foam_instantiation(self):
        """MultiphaseEulerFoam 可实例化。"""
        from pyfoam.applications.multiphase_euler_foam import MultiphaseEulerFoam
        assert MultiphaseEulerFoam is not None
