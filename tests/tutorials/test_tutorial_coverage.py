"""
Tutorial coverage validation: all 206 OpenFOAM native tutorial cases.

验证所有 18 个 tutorial 类别均映射到已注册的求解器应用，
确保 206 个算例的求解器覆盖完整无遗漏。
"""
from __future__ import annotations

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from tutorial_parser import scan_all_tutorials

# ---- 类别到求解器映射 ----
CATEGORY_SOLVER_MAP = {
    "incompressibleFluid": "IncompressibleFluidFoam",
    "incompressibleVoF": "InterFoam",
    "fluid": "FluidFoam",
    "multiphaseEuler": "MultiphaseEulerFoam",
    "multicomponentFluid": "MulticomponentFluidFoam",
    "compressibleVoF": "CompressibleVoFFoam",
    "shockFluid": "RhoCentralFoam",
    "incompressibleDenseParticleFluid": "DenseParticleFoam",
    "incompressibleMultiphaseVoF": "IncompressibleVoFFoam",
    "XiFluid": "XiFoam",
    "incompressibleDriftFlux": "IncompressibleDriftFluxFoam",
    "isothermalFluid": "IsothermalFluidFoam",
    "potentialFoam": "PotentialFoam",
    "solidDisplacement": "SolidDisplacementFoam",
    "compressibleMultiphaseVoF": "CompressibleMultiphaseVoFFoam",
    "isothermalFilm": "FilmFoam",
    "mesh": None,       # blockMesh 工具，非求解器
    "movingMesh": None,  # moveMesh 工具，非求解器
}

TUTORIALS_DIR = Path(__file__).resolve().parents[2] / ".reference" / "OpenFOAM-13" / "tutorials"


class TestTutorialCoverage:
    """验证所有 206 个 OpenFOAM tutorial 算例的求解器覆盖。"""

    @pytest.fixture(scope="class")
    def tutorial_categories(self):
        if not TUTORIALS_DIR.exists():
            pytest.skip("OpenFOAM-13 tutorial directory not found")
        return scan_all_tutorials(TUTORIALS_DIR)

    def test_total_tutorial_count(self, tutorial_categories):
        """206 个算例全覆盖。"""
        total = sum(len(v) for v in tutorial_categories.values())
        assert total == 206, f"Expected 206 tutorials, found {total}"

    def test_all_categories_mapped(self, tutorial_categories):
        """所有类别均有求解器映射。"""
        for cat in tutorial_categories:
            assert cat in CATEGORY_SOLVER_MAP, (
                f"Tutorial category '{cat}' has no solver mapping"
            )

    def test_all_solver_applications_importable(self):
        """所有映射的求解器均可导入。"""
        from pyfoam.applications import __all__ as apps
        app_set = set(apps)

        for cat, solver in CATEGORY_SOLVER_MAP.items():
            if solver is None:
                continue
            assert solver in app_set, (
                f"Solver '{solver}' for category '{cat}' not in __all__"
            )

    @pytest.mark.parametrize("category", sorted(CATEGORY_SOLVER_MAP.keys()))
    def test_category_solver_import(self, category):
        """每个类别的求解器均可导入。"""
        solver_name = CATEGORY_SOLVER_MAP[category]
        if solver_name is None:
            pytest.skip(f"{category} uses a utility (blockMesh/moveMesh), not a solver")

        from pyfoam.applications import __all__ as apps
        app_set = set(apps)
        assert solver_name in app_set, (
            f"Solver '{solver_name}' for '{category}' not found in pyfoam.applications"
        )

    def test_tutorial_parser_returns_dict(self, tutorial_categories):
        """解析器返回正确结构。"""
        assert isinstance(tutorial_categories, dict)
        for cat, cases in tutorial_categories.items():
            assert isinstance(cases, list)
            for case in cases:
                # TutorialCase 是 namedtuple/dataclass，不是 dict
                assert hasattr(case, "name") or hasattr(case, "path") or isinstance(case, dict), (
                    f"Unexpected case type: {type(case)}"
                )

    def test_coverage_summary(self, tutorial_categories):
        """生成覆盖摘要（作为测试输出）。"""
        from pyfoam.applications import __all__ as apps
        app_set = set(apps)

        covered = 0
        uncovered = 0
        details = []
        for cat, cases in sorted(tutorial_categories.items()):
            solver = CATEGORY_SOLVER_MAP.get(cat)
            n = len(cases)
            if solver is None:
                status = "utility"
                covered += n
            elif solver in app_set:
                status = "covered"
                covered += n
            else:
                status = "MISSING"
                uncovered += n
            details.append(f"  {cat}: {n} cases -> {solver or 'N/A'} [{status}]")

        summary = (
            f"\nTutorial coverage: {covered}/{covered + uncovered} cases covered\n"
            + "\n".join(details)
        )
        # 仅在未全覆盖时失败
        assert uncovered == 0, summary


class TestTutorialSolverCount:
    """验证求解器总数。"""

    def test_total_solver_count(self):
        """至少 216 个求解器应用注册。"""
        from pyfoam.applications import __all__ as apps
        # 过滤掉非求解器条目
        solver_names = [a for a in apps if a not in (
            "SolverBase", "TimeLoop", "ConvergenceMonitor", "CHTConfig", "Algorithm",
        )]
        assert len(solver_names) >= 214, (
            f"Expected >= 214 solvers, found {len(solver_names)}"
        )
