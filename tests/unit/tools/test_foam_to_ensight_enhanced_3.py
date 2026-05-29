"""Tests for foam_to_ensight_enhanced_3 — enhanced EnSight v3."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.foam_to_ensight_enhanced_3 import EnSightV3Result, foam_to_ensight_enhanced_3


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


class TestFoamToEnsightEnhanced3:
    def test_returns_result_type(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_3(case, mesh=m)
        assert isinstance(r, EnSightV3Result)

    def test_case_file_created(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_3(case, mesh=m)
        assert r.case_file.exists()

    def test_geometry_file_created(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_3(case, mesh=m)
        assert len(r.geometry_files) >= 1
        assert r.geometry_files[0].exists()

    def test_binary_export(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_3(case, mesh=m, binary=True)
        assert r.binary is True
        assert r.case_file.exists()

    def test_fields_export(self):
        case = _make_case_dir()
        m = _single_hex()
        p = np.ones(1) * 1e5
        U = np.ones((1, 3)) * 10.0
        fields = {"p": p, "U": U}
        r = foam_to_ensight_enhanced_3(case, mesh=m, fields=fields)
        assert r.n_variables == 2

    def test_selective_export(self):
        case = _make_case_dir()
        m = _single_hex()
        p = np.ones(1) * 1e5
        U = np.ones((1, 3)) * 10.0
        fields = {"p": p, "U": U}
        r = foam_to_ensight_enhanced_3(
            case, mesh=m, fields=fields, export_variables={"p"},
        )
        assert r.n_variables == 1

    def test_geometry_deduplication(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_3(
            case, mesh=m, time_range=[0.0, 1.0, 2.0], deduplicate_geometry=True,
        )
        assert r.geometry_reused == 2
        assert len(r.geometry_files) == 1

    def test_boundary_parts(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_3(case, mesh=m, write_boundary_parts=True)
        assert r.n_parts == 2  # volume + 1 boundary

    def test_no_boundary_parts(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_3(case, mesh=m, write_boundary_parts=False)
        assert r.n_parts == 1

    def test_n_times(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_3(case, mesh=m, time_range=[0.0, 0.5, 1.0])
        assert r.n_times == 3

    def test_no_mesh_raises(self):
        case = _make_case_dir()
        with pytest.raises(ValueError, match="No mesh"):
            foam_to_ensight_enhanced_3(case)

    def test_tensor_export(self):
        case = _make_case_dir()
        m = _single_hex()
        tau = np.ones((1, 6)) * 100.0
        r = foam_to_ensight_enhanced_3(case, mesh=m, fields={"tau": tau})
        assert r.n_variables == 1
