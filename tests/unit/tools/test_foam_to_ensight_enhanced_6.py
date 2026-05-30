"""Tests for foam_to_ensight_enhanced_6 — enhanced EnSight v6."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.foam_to_ensight_enhanced_6 import EnSightV6Result, foam_to_ensight_enhanced_6


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


def _make_case_dir():
    d = Path(tempfile.mkdtemp()) / "test_case"
    d.mkdir()
    return d


class TestFoamToEnsightEnhanced6:
    def test_returns_result_type(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_6(case, mesh=m)
        assert isinstance(r, EnSightV6Result)

    def test_case_file_created(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_6(case, mesh=m)
        assert r.case_file.exists()

    def test_recovery_mode(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_6(case, mesh=m, recover=True)
        assert isinstance(r.n_recovered, int)

    def test_time_interpolation(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_6(
            case, mesh=m,
            time_range=[0.0, 1.0],
            n_interpolation_steps=2,
        )
        assert r.n_interpolated >= 0

    def test_no_mesh_raises(self):
        case = _make_case_dir()
        with pytest.raises(ValueError, match="No mesh"):
            foam_to_ensight_enhanced_6(case)

    def test_parallel_flag(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_6(case, mesh=m, parallel_write=True)
        assert isinstance(r.parallel, bool)
