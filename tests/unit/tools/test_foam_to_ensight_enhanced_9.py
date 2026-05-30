"""Tests for foam_to_ensight_enhanced_9 — enhanced EnSight export v9."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pathlib import Path
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.foam_to_ensight_enhanced_9 import (
    EnSightV9Result, ExportStatistics, SelectiveExport, BatchCaseResult,
    foam_to_ensight_enhanced_9,
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


class TestEnSight9:
    def test_no_case_raises(self):
        """Nonexistent case directory should raise FileNotFoundError."""
        m = _single_hex()
        p = np.zeros(1, dtype=np.float64)
        with pytest.raises(FileNotFoundError):
            foam_to_ensight_enhanced_9("nonexistent_case_xyz", mesh=m, fields={"p": p})

    def test_result_type(self):
        """Verify result dataclass instantiation."""
        r = EnSightV9Result()
        assert isinstance(r, EnSightV9Result)

    def test_export_statistics_type(self):
        s = ExportStatistics()
        assert s.bytes_per_second == 0.0
        assert s.n_io_operations == 0

    def test_selective_export_type(self):
        se = SelectiveExport(field_name="U", component="x", n_values=10)
        assert se.field_name == "U"
        assert se.bytes_written == 0  # default

    def test_batch_case_result_type(self):
        b = BatchCaseResult(case_name="test", success=True)
        assert b.success is True

    def test_default_result_fields(self):
        r = EnSightV9Result()
        assert r.n_batch_cases == 0
        assert r.n_batch_failures == 0
        assert r.n_selective_exports == 0
        assert isinstance(r.statistics, ExportStatistics)
