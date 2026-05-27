"""验证 applications 包导出全部求解器。"""

import pyfoam.applications as app


# 预期的求解器类名（不含工具类 SolverBase / TimeLoop / ConvergenceMonitor）
EXPECTED_SOLVERS = [
    "BoundaryFoam",
    "IcoFoam",
    "PimpleFoam",
    "SimpleFoam",
    "RhoSimpleFoam",
    "BuoyantSimpleFoam",
    "BuoyantBoussinesqSimpleFoam",
    "RhoPimpleFoam",
    "RhoCentralFoam",
    "InterFoam",
    "PorousSimpleFoam",
    "MultiphaseInterFoam",
    "CompressibleInterFoam",
    "TwoPhaseEulerFoam",
    "MultiphaseEulerFoam",
    "CavitatingFoam",
    "PisoFoam",
    "PotentialFoam",
    "ScalarTransportFoam",
    "LaplacianFoam",
    "SonicFoam",
    "SrfSimpleFoam",
    "BuoyantPimpleFoam",
    "CHTMultiRegionFoam",
    "ReactingFoam",
    "SolidDisplacementFoam",
    "IncompressibleFluidFoam",
    "ShallowWaterFoam",
    "RhoPorousSimpleFoam",
    "ChemFoam",
    "IsothermalFluidFoam",
    "IncompressibleVoFFoam",
    "CompressibleVoFFoam",
    "IncompressibleDriftFluxFoam",
]

# 额外导出的枚举/工具
EXTRA_EXPORTS = ["Algorithm"]

# 工具类
UTILITY_EXPORTS = ["SolverBase", "TimeLoop", "ConvergenceMonitor"]


def test_all_solvers_in___all__():
    """__all__ 应包含全部 25 个求解器。"""
    for name in EXPECTED_SOLVERS:
        assert name in app.__all__, f"{name} missing from __all__"


def test_all_solvers_importable():
    """每个求解器应可从包顶层导入。"""
    for name in EXPECTED_SOLVERS:
        obj = getattr(app, name, None)
        assert obj is not None, f"{name} not importable from pyfoam.applications"


def test_utility_exports():
    """工具类也应可导入。"""
    for name in UTILITY_EXPORTS:
        assert name in app.__all__, f"{name} missing from __all__"
        assert getattr(app, name, None) is not None


def test_total_export_count():
    """__all__ 总条目数 = 求解器 + 工具类 + 额外导出。"""
    expected = len(EXPECTED_SOLVERS) + len(UTILITY_EXPORTS) + len(EXTRA_EXPORTS)
    assert len(app.__all__) >= len(EXPECTED_SOLVERS)
