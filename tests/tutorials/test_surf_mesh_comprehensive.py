"""
Tutorial validation: solver surf mesh comprehensive tests.

全面验证求解器表面网格模型。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestSurfMeshComprehensive:
    """全面表面网格测试。"""

    def test_surf_mesh_import(self):
        """SurfMesh 可导入。"""
        from pyfoam.surf_mesh import SurfMesh
        assert SurfMesh is not None

    def test_surf_zone_import(self):
        """SurfZone 可导入。"""
        from pyfoam.surf_mesh import SurfZone
        assert SurfZone is not None

    def test_surf_scalar_field_import(self):
        """SurfScalarField 可导入。"""
        from pyfoam.surf_mesh import SurfScalarField
        assert SurfScalarField is not None

    def test_surf_vector_field_import(self):
        """SurfVectorField 可导入。"""
        from pyfoam.surf_mesh import SurfVectorField
        assert SurfVectorField is not None

    def test_surf_mesh_basic(self):
        """SurfMesh 基本操作。"""
        from pyfoam.surf_mesh import SurfMesh
        pts = torch.tensor([[0,0,0],[1,0,0],[0.5,1,0]], dtype=CFD_DTYPE)
        faces = [torch.tensor([0, 1, 2])]
        mesh = SurfMesh(points=pts, faces=faces)
        assert mesh.n_points == 3
        assert mesh.n_faces == 1

    def test_surf_mesh_face_centres(self):
        """SurfMesh 面心计算。"""
        from pyfoam.surf_mesh import SurfMesh
        pts = torch.tensor([[0,0,0],[1,0,0],[0.5,1,0]], dtype=CFD_DTYPE)
        faces = [torch.tensor([0, 1, 2])]
        mesh = SurfMesh(points=pts, faces=faces)
        centres = mesh.face_centres()
        assert centres.shape == (1, 3)

    def test_surf_mesh_face_areas(self):
        """SurfMesh 面积计算。"""
        from pyfoam.surf_mesh import SurfMesh
        pts = torch.tensor([[0,0,0],[1,0,0],[0.5,1,0]], dtype=CFD_DTYPE)
        faces = [torch.tensor([0, 1, 2])]
        mesh = SurfMesh(points=pts, faces=faces)
        areas = mesh.face_areas()
        assert areas.shape == (1, 3)
        assert areas.norm().item() > 0
