"""Tests for foam_dictionary_enhanced_4."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_dictionary_enhanced_4 import FoamDictEnhanced4Result, VersionControlResult, AuditLogResult, foam_dictionary_enhanced_4


class TestFoamDictEnhanced4Result:
    def test_returns_result(self):
        r = foam_dictionary_enhanced_4()
        assert isinstance(r, FoamDictEnhanced4Result)

    def test_vcs(self):
        r = foam_dictionary_enhanced_4(enable_vcs=True)
        assert isinstance(r.vcs, VersionControlResult)
        assert r.vcs.enabled is True

    def test_audit(self):
        r = foam_dictionary_enhanced_4(enable_audit=True)
        assert isinstance(r.audit, AuditLogResult)
        assert r.audit.enabled is True
