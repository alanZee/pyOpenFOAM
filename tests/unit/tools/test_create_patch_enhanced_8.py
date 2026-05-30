"""Tests for create_patch_enhanced_8 — enhanced patch creation v8."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_patch_enhanced_8 import (
    PatchEnhanced8Result, CompatibilityReport, BCValidation, NamingConvention,
    create_patch_enhanced_8,
)


def _single_hex():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[{"name": "all", "type": "wall", "startFace": 0, "nFaces": 6}],
               validate=False)
    m.compute_geometry()
    return m


class TestPatchEnhanced8:
    def test_returns_result_type(self):
        m = _single_hex()
        r = create_patch_enhanced_8(m)
        assert isinstance(r, PatchEnhanced8Result)

    def test_bc_validation(self):
        m = _single_hex()
        r = create_patch_enhanced_8(m, validate_bc=True, patch_type="wall")
        assert isinstance(r.bc_validation, BCValidation)

    def test_compatibility(self):
        m = _single_hex()
        r = create_patch_enhanced_8(m, check_compatibility=True)
        assert isinstance(r.compatibility, CompatibilityReport)

    def test_naming_convention(self):
        m = _single_hex()
        r = create_patch_enhanced_8(m, enforce_naming=True, patch_name="valid_name")
        assert isinstance(r.naming, NamingConvention)
        assert r.naming.is_valid is True

    def test_naming_invalid_chars(self):
        m = _single_hex()
        r = create_patch_enhanced_8(m, enforce_naming=True, patch_name="bad-name!")
        assert r.naming.sanitised_name != "bad-name!"
