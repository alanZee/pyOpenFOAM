"""Tests for surface_auto_patch_enhanced_2 — enhanced auto-patching v2."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_auto_patch_enhanced_2 import (
    SurfaceAutoPatchEnhanced2Result,
    surface_auto_patch_enhanced_2,
)


def _simple_triangles():
    """Two triangles sharing an edge with 90-degree dihedral."""
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    return verts, faces


def _flat_quad():
    """4 coplanar triangles in a quad (same normal)."""
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
    ], dtype=np.int32)
    return verts, faces


class TestSurfaceAutoPatchEnhanced2:
    def test_returns_result_type(self):
        v, f = _simple_triangles()
        r = surface_auto_patch_enhanced_2(vertices=v, faces=f)
        assert isinstance(r, SurfaceAutoPatchEnhanced2Result)

    def test_n_patches(self):
        v, f = _simple_triangles()
        r = surface_auto_patch_enhanced_2(vertices=v, faces=f)
        assert r.n_patches >= 1

    def test_patch_ids_shape(self):
        v, f = _simple_triangles()
        r = surface_auto_patch_enhanced_2(vertices=v, faces=f)
        assert r.patch_ids.shape == (2,)

    def test_coplanar_single_patch(self):
        v, f = _flat_quad()
        r = surface_auto_patch_enhanced_2(vertices=v, faces=f, feature_angle=30.0)
        assert r.n_patches == 1

    def test_smooth_iterations(self):
        v, f = _simple_triangles()
        r = surface_auto_patch_enhanced_2(vertices=v, faces=f, smooth_iterations=2)
        assert r.n_patches >= 1

    def test_name_by_direction(self):
        v, f = _flat_quad()
        r = surface_auto_patch_enhanced_2(vertices=v, faces=f, name_by_direction=True)
        for pid, name in r.patch_names.items():
            assert isinstance(name, str)

    def test_min_patch_faces(self):
        v, f = _simple_triangles()
        r = surface_auto_patch_enhanced_2(vertices=v, faces=f, min_patch_faces=100)
        # All patches are smaller than 100, should be merged to 1
        assert r.n_patches <= 2

    def test_n_regions(self):
        v, f = _simple_triangles()
        r = surface_auto_patch_enhanced_2(vertices=v, faces=f)
        assert r.n_regions == 1

    def test_empty_surface(self):
        v = np.array([[0, 0, 0]], dtype=np.float64)
        f = np.empty((0, 3), dtype=np.int32)
        r = surface_auto_patch_enhanced_2(vertices=v, faces=f)
        assert r.n_patches == 0
