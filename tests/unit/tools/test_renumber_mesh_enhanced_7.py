"""Tests for renumber_mesh_enhanced_7."""
from __future__ import annotations
import pytest
from pyfoam.tools.renumber_mesh_enhanced_7 import RenumberEnhanced7Result, FillInResult, BlockOrderingResult, renumber_mesh_enhanced_7


class TestRenumberEnhanced7Result:
    def test_returns_result(self):
        r = renumber_mesh_enhanced_7()
        assert isinstance(r, RenumberEnhanced7Result)

    def test_fill_in(self):
        r = renumber_mesh_enhanced_7(enable_fill_in=True)
        assert isinstance(r.fill_in, FillInResult)
        assert r.fill_in.enabled is True

    def test_block(self):
        r = renumber_mesh_enhanced_7(enable_block=True)
        assert isinstance(r.block, BlockOrderingResult)
        assert r.block.enabled is True
