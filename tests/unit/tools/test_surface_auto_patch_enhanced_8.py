"""Tests for surface_auto_patch_enhanced_8 — enhanced auto-patching v8."""
from __future__ import annotations
import pytest
import numpy as np
from pyfoam.tools.surface_auto_patch_enhanced_8 import (
    SurfaceAutoPatchEnhanced8Result, InheritedPatch, BoundaryLayerPatch,
    surface_auto_patch_enhanced_8,
)


class TestSurfaceAutoPatch8:
    def test_returns_result_type(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_auto_patch_enhanced_8(vertices=verts, faces=faces)
        assert isinstance(r, SurfaceAutoPatchEnhanced8Result)

    def test_inheritance(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        ref_ids = np.array([0, 0], dtype=np.int32)
        r = surface_auto_patch_enhanced_8(
            vertices=verts, faces=faces,
            inherit_patches=True, reference_patch_ids=ref_ids,
        )
        assert isinstance(r.inherited_patches, list)

    def test_bl_aware(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_auto_patch_enhanced_8(
            vertices=verts, faces=faces, bl_aware=True,
        )
        assert isinstance(r.bl_patches, list)

    def test_boundary_dict(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_auto_patch_enhanced_8(
            vertices=verts, faces=faces, generate_boundary_dict=True,
        )
        # May be None if no patches detected, but should not raise
        assert r.boundary_dict is None or isinstance(r.boundary_dict, str)
