"""
Tutorial validation: tutorial parser tests.

验证 tutorial parser 能正确解析 OpenFOAM tutorial 算例。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.tutorials.tutorial_parser import (
    TutorialCase,
    parse_control_dict,
    parse_foam_file_header,
    scan_tutorial_case,
    scan_tutorial_category,
    scan_all_tutorials,
    count_tutorials,
)


TUTORIALS_DIR = Path(".reference/OpenFOAM-13/tutorials")


class TestTutorialParser:
    """Tutorial parser 测试。"""

    def test_parse_control_dict(self):
        """解析 controlDict。"""
        path = TUTORIALS_DIR / "incompressibleFluid" / "cavity" / "system" / "controlDict"
        if not path.exists():
            pytest.skip("Tutorial not found")
        cd = parse_control_dict(path)
        assert "solver" in cd
        assert "deltaT" in cd
        assert "endTime" in cd

    def test_parse_foam_file_header(self):
        """解析 FoamFile 头。"""
        path = TUTORIALS_DIR / "incompressibleFluid" / "cavity" / "0" / "U"
        if not path.exists():
            pytest.skip("Tutorial not found")
        header = parse_foam_file_header(path)
        assert "class" in header
        assert "object" in header

    def test_scan_tutorial_case(self):
        """扫描单个 tutorial 算例。"""
        case_dir = TUTORIALS_DIR / "incompressibleFluid" / "cavity"
        if not case_dir.exists():
            pytest.skip("Tutorial not found")
        case = scan_tutorial_case(case_dir)
        assert case.name == "cavity"
        assert case.solver == "incompressibleFluid"
        # Mesh 可能在 system/blockMeshDict 中定义（需要生成）
        assert case.has_mesh or (case_dir / "system" / "blockMeshDict").exists()
        assert case.has_fields is True
        assert len(case.field_files) > 0

    def test_scan_tutorial_category(self):
        """扫描 tutorial 类别。"""
        category_dir = TUTORIALS_DIR / "incompressibleFluid"
        if not category_dir.exists():
            pytest.skip("Tutorial category not found")
        cases = scan_tutorial_category(category_dir)
        assert len(cases) > 0
        assert all(isinstance(c, TutorialCase) for c in cases)

    def test_scan_all_tutorials(self):
        """扫描所有 tutorial。"""
        if not TUTORIALS_DIR.exists():
            pytest.skip("Tutorials directory not found")
        all_cases = scan_all_tutorials(TUTORIALS_DIR)
        assert len(all_cases) > 0
        total = sum(len(cases) for cases in all_cases.values())
        assert total > 100  # 至少 100 个 tutorial

    def test_count_tutorials(self):
        """统计 tutorial 数量。"""
        if not TUTORIALS_DIR.exists():
            pytest.skip("Tutorials directory not found")
        counts = count_tutorials(TUTORIALS_DIR)
        assert len(counts) > 0
        total = sum(counts.values())
        assert total > 100
        # 检查主要类别存在
        assert "incompressibleFluid" in counts
        assert counts["incompressibleFluid"] > 0

    def test_tutorial_case_dataclass(self):
        """TutorialCase 数据类。"""
        case = TutorialCase(name="test", path=Path("/tmp"))
        assert case.name == "test"
        assert case.solver == ""
        assert case.has_mesh is False
        assert case.field_files == []
        assert case.bc_types == {}


class TestTutorialCoverage:
    """Tutorial 覆盖度分析。"""

    def test_total_tutorial_count(self):
        """总 tutorial 数量。"""
        if not TUTORIALS_DIR.exists():
            pytest.skip("Tutorials directory not found")
        counts = count_tutorials(TUTORIALS_DIR)
        total = sum(counts.values())
        # OpenFOAM-13 应有 ~200+ 个 tutorial
        assert total >= 200, f"Expected >=200 tutorials, got {total}"

    def test_major_categories_present(self):
        """主要类别存在。"""
        if not TUTORIALS_DIR.exists():
            pytest.skip("Tutorials directory not found")
        counts = count_tutorials(TUTORIALS_DIR)
        expected_categories = [
            "incompressibleFluid",
            "incompressibleVoF",
            "fluid",
            "multiphaseEuler",
        ]
        for cat in expected_categories:
            assert cat in counts, f"Missing category: {cat}"
            assert counts[cat] > 0, f"Empty category: {cat}"

    def test_tutorial_solver_coverage(self):
        """Tutorial 求解器覆盖。"""
        if not TUTORIALS_DIR.exists():
            pytest.skip("Tutorials directory not found")
        all_cases = scan_all_tutorials(TUTORIALS_DIR)
        solvers = set()
        for cases in all_cases.values():
            for case in cases:
                if case.solver:
                    solvers.add(case.solver)
        # 应该有多种求解器
        assert len(solvers) >= 5, f"Expected >=5 solvers, got {len(solvers)}: {solvers}"
