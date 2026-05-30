"""Tests for create_patch_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.create_patch_enhanced_10 import PatchEnhanced10Result, PatchCouplingInterface, MappedPatchResult, create_patch_enhanced_10


class TestPatchEnhanced10Result:
    def test_returns_result(self):
        r = create_patch_enhanced_10()
        assert isinstance(r, PatchEnhanced10Result)

    def test_coupling(self):
        r = create_patch_enhanced_10(enable_coupling=True)
        assert isinstance(r.coupling, PatchCouplingInterface)
        assert r.coupling.enabled is True

    def test_mapped(self):
        r = create_patch_enhanced_10(enable_mapped=True)
        assert isinstance(r.mapped, MappedPatchResult)
        assert r.mapped.enabled is True
