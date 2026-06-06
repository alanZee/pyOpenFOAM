"""
Tutorial validation: surface mesh and format smoke tests.

验证表面网格和格式转换的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestSurfaceMeshSmoke:
    """表面网格 smoke 测试。"""

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


class TestFormatConversionSmoke:
    """格式转换 smoke 测试。"""

    def test_foam_to_vtk_import(self):
        """foamToVTK 可导入。"""
        from pyfoam.tools import foam_to_vtk
        assert foam_to_vtk is not None

    def test_foam_to_ensight_import(self):
        """foamToEnsight 可导入。"""
        from pyfoam.tools import foam_to_ensight
        assert foam_to_ensight is not None

    def test_foam_to_fluent_import(self):
        """foamToFluent 可导入。"""
        from pyfoam.tools import foam_to_fluent
        assert foam_to_fluent is not None

    def test_surface_convert_import(self):
        """surfaceConvert 可导入。"""
        from pyfoam.tools import surface_convert
        assert surface_convert is not None

    def test_surface_features_import(self):
        """surfaceFeatures 可导入。"""
        from pyfoam.tools import surface_features
        assert surface_features is not None
