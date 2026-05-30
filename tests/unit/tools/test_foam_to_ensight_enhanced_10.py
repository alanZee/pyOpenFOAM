"""Tests for foam_to_ensight_enhanced_10 — enhanced EnSight export v10."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.foam_to_ensight_enhanced_10 import (
    EnSightV10Result, AnimationPipeline, CaseMetadata, DistributedPartition,
    foam_to_ensight_enhanced_10,
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


class TestEnSight10:
    def test_no_case_raises(self):
        """Nonexistent case directory should raise FileNotFoundError."""
        m = _single_hex()
        p = np.zeros(1, dtype=np.float64)
        with pytest.raises(FileNotFoundError):
            foam_to_ensight_enhanced_10("nonexistent_case_xyz_10", mesh=m, fields={"p": p})

    def test_result_type(self):
        """Verify result dataclass instantiation."""
        r = EnSightV10Result()
        assert isinstance(r, EnSightV10Result)

    def test_animation_pipeline_type(self):
        ap = AnimationPipeline(n_keyframes=5, frame_rate=30.0)
        assert ap.n_keyframes == 5
        assert ap.duration_seconds == 0.0  # default

    def test_case_metadata_type(self):
        m = CaseMetadata(solver_name="pisoFoam", case_title="test", n_cells=1000)
        assert m.solver_name == "pisoFoam"
        assert m.n_cells == 1000

    def test_distributed_partition_type(self):
        p = DistributedPartition(partition_id=0, output_dir="/tmp/p0", n_fields=3)
        assert p.partition_id == 0
        assert p.success is True

    def test_default_result_fields(self):
        r = EnSightV10Result()
        assert r.n_partitions == 0
        assert r.animation_pipeline is None
        assert r.case_metadata is None
        assert isinstance(r.partitions, list)
