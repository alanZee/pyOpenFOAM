"""Tests for foam_to_ensight_enhanced_7 — enhanced EnSight v7."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.foam_to_ensight_enhanced_7 import EnSightV7Result, foam_to_ensight_enhanced_7


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


class TestFoamToEnsightEnhanced7:
    def test_returns_result_type(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_7(case, mesh=m)
        assert isinstance(r, EnSightV7Result)

    def test_case_file_created(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_7(case, mesh=m)
        assert r.case_file.exists()

    def test_derived_fields(self):
        case = _make_case_dir()
        m = _single_hex()
        p = np.ones(1) * 1e5
        U = np.ones((1, 3)) * 10.0
        derived = {"pSquared": lambda fields: fields.get("p", np.zeros(1)) ** 2}
        r = foam_to_ensight_enhanced_7(
            case, mesh=m,
            fields={"p": p, "U": U},
            derived_fields=derived,
        )
        assert r.n_derived_fields >= 0

    def test_component_export(self):
        case = _make_case_dir()
        m = _single_hex()
        U = np.ones((1, 3)) * 10.0
        r = foam_to_ensight_enhanced_7(
            case, mesh=m,
            fields={"U": U},
            component_export={"U": ["x", "magnitude"]},
        )
        assert r.n_components_exported >= 0

    def test_generate_config(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_7(
            case, mesh=m, generate_config=True,
        )
        assert isinstance(r.config_file, (Path, type(None)))
        if r.config_file is not None:
            assert r.config_file.exists()

    def test_case_comments(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_7(
            case, mesh=m, case_comments=["Test comment"],
        )
        assert len(r.case_comments) >= 1

    def test_no_mesh_raises(self):
        case = _make_case_dir()
        with pytest.raises(ValueError, match="No mesh"):
            foam_to_ensight_enhanced_7(case)

    def test_recovery_mode(self):
        case = _make_case_dir()
        m = _single_hex()
        r = foam_to_ensight_enhanced_7(case, mesh=m, recover=True)
        assert isinstance(r.n_recovered, int)
