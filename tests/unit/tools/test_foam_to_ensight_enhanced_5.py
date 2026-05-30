"""Tests for foam_to_ensight_enhanced_5 — enhanced EnSight v5."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.foam_to_ensight_enhanced_5 import EnSightV5Result, foam_to_ensight_enhanced_5


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


class TestFoamToEnsightEnhanced5:
    def test_returns_result_type(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_5(case, mesh=m)
        assert isinstance(r, EnSightV5Result)

    def test_case_file_created(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_5(case, mesh=m)
        assert r.case_file.exists()

    def test_multi_resolution(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_5(case, mesh=m, multi_resolution=True)
        assert r.coarse_geometry_file is not None
        assert r.n_coarse_cells >= 1

    def test_stream_mode(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_5(case, mesh=m, binary=True, stream_mode=True)
        assert r.streamed is True
        assert r.case_file.exists()

    def test_adaptive_compression(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_5(
            case, mesh=m, adaptive_compression=True,
        )
        assert r.compression_ratio >= 0

    def test_fields_export(self):
        case = _make_case_dir()
        m = _single_hex()
        p = np.ones(1) * 1e5
        U = np.ones((1, 3)) * 10.0
        r = foam_to_ensight_enhanced_5(case, mesh=m, fields={"p": p, "U": U})
        assert r.n_variables == 2

    def test_no_mesh_raises(self):
        case = _make_case_dir()
        with pytest.raises(ValueError, match="No mesh"):
            foam_to_ensight_enhanced_5(case)
