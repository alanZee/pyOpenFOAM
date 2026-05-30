"""Tests for foam_to_ensight_enhanced_8 — enhanced EnSight export v8."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pathlib import Path
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.foam_to_ensight_enhanced_8 import EnSightV8Result, AnimationKeyframe, VariableMapping, foam_to_ensight_enhanced_8


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


class TestEnSight8:
    def test_no_case_raises(self):
        """Nonexistent case directory should raise FileNotFoundError."""
        m = _single_hex()
        p = np.zeros(1, dtype=np.float64)
        with pytest.raises(FileNotFoundError):
            foam_to_ensight_enhanced_8("nonexistent_case_xyz", mesh=m, fields={"p": p})

    def test_no_mesh_raises(self):
        """Missing mesh should raise ValueError from v7."""
        with pytest.raises((ValueError, FileNotFoundError)):
            foam_to_ensight_enhanced_8("nonexistent_case_xyz")

    def test_result_type(self):
        """Verify result dataclass instantiation."""
        r = EnSightV8Result()
        assert isinstance(r, EnSightV8Result)

    def test_variable_mapping_type(self):
        assert VariableMapping().foam_name == ""
        assert VariableMapping().ensight_name == ""

    def test_animation_keyframe_type(self):
        kf = AnimationKeyframe(time=1.0, camera_position=(2, 2, 2))
        assert kf.time == 1.0
        assert kf.camera_position == (2, 2, 2)

    def test_default_result_fields(self):
        r = EnSightV8Result()
        assert r.n_keyframes == 0
        assert r.animation_file is None
        assert r.template_file is None
        assert isinstance(r.variable_mappings, list)
