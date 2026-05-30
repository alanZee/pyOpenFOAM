"""Tests for renumber_mesh_enhanced_2."""
from __future__ import annotations
import pytest
from pyfoam.tools.renumber_mesh_enhanced_2 import RenumberEnhanced2Result, NestedDissectionResult, MultiLevelOrderingResult, renumber_mesh_enhanced_2


class TestRenumberEnhanced2Result:
    def test_returns_result(self):
        r = renumber_mesh_enhanced_2()
        assert isinstance(r, RenumberEnhanced2Result)

    def test_nested_dissection(self):
        r = renumber_mesh_enhanced_2(enable_nested_dissection=True)
        assert isinstance(r.nested_dissection, NestedDissectionResult)
        assert r.nested_dissection.enabled is True

    def test_multi_level(self):
        r = renumber_mesh_enhanced_2(enable_multi_level=True)
        assert isinstance(r.multi_level, MultiLevelOrderingResult)
        assert r.multi_level.enabled is True
