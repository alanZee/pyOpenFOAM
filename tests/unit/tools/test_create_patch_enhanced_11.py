"""Tests for create_patch_enhanced_11."""
from __future__ import annotations
import pytest
from pyfoam.tools.create_patch_enhanced_11 import PatchEnhanced11Result, CyclicAMIPatchResult, OversetPatchResult, create_patch_enhanced_11


class TestPatchEnhanced11Result:
    def test_returns_result(self):
        r = create_patch_enhanced_11()
        assert isinstance(r, PatchEnhanced11Result)

    def test_cyclic_ami(self):
        r = create_patch_enhanced_11(enable_cyclic_ami=True)
        assert isinstance(r.cyclic_ami, CyclicAMIPatchResult)
        assert r.cyclic_ami.enabled is True

    def test_overset(self):
        r = create_patch_enhanced_11(enable_overset=True)
        assert isinstance(r.overset, OversetPatchResult)
        assert r.overset.enabled is True
