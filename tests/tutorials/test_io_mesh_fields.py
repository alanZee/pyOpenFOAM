"""
Tutorial validation: IO and mesh smoke tests.

验证 IO 和网格模块的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestIOSmoke:
    """IO 模块 smoke 测试。"""

    def test_case_import(self):
        """Case 类可导入。"""
        from pyfoam.io import Case
        assert Case is not None

    def test_foam_file_import(self):
        """FoamFile 可导入。"""
        from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file
        assert FoamFileHeader is not None
        assert FileFormat is not None
        assert write_foam_file is not None

    def test_mesh_io_import(self):
        """mesh_io 可导入。"""
        from pyfoam.io.mesh_io import read_mesh
        assert read_mesh is not None

    def test_dictionary_import(self):
        """dictionary 可导入。"""
        from pyfoam.io.dictionary import parse_dict, FoamDict
        assert parse_dict is not None
        assert FoamDict is not None


class TestMeshSmoke:
    """网格模块 smoke 测试。"""

    def test_poly_mesh_import(self):
        """PolyMesh 可导入。"""
        from pyfoam.mesh.poly_mesh import PolyMesh
        assert PolyMesh is not None

    def test_fv_mesh_import(self):
        """FvMesh 可导入。"""
        from pyfoam.mesh.fv_mesh import FvMesh
        assert FvMesh is not None


class TestFieldSmoke:
    """场模块 smoke 测试。"""

    def test_geometric_field_import(self):
        """GeometricField 可导入。"""
        from pyfoam.fields import GeometricField
        assert GeometricField is not None

    def test_vol_scalar_field_import(self):
        """VolScalarField 可导入。"""
        from pyfoam.fields import volScalarField
        assert volScalarField is not None

    def test_vol_vector_field_import(self):
        """VolVectorField 可导入。"""
        from pyfoam.fields import volVectorField
        assert volVectorField is not None

    def test_surface_scalar_field_import(self):
        """SurfaceScalarField 可导入。"""
        from pyfoam.fields import surfaceScalarField
        assert surfaceScalarField is not None

    def test_dimension_set_import(self):
        """DimensionSet 可导入。"""
        from pyfoam.fields import DimensionSet
        assert DimensionSet is not None


class TestCoreSmoke:
    """核心模块 smoke 测试。"""

    def test_device_import(self):
        """设备管理可导入。"""
        from pyfoam.core import get_device, get_default_dtype
        assert get_device is not None
        assert get_default_dtype is not None

    def test_ldu_matrix_import(self):
        """LduMatrix 可导入。"""
        from pyfoam.core import LduMatrix
        assert LduMatrix is not None

    def test_fv_matrix_import(self):
        """FvMatrix 可导入。"""
        from pyfoam.core import FvMatrix
        assert FvMatrix is not None

    def test_dtype_import(self):
        """dtype 可导入。"""
        from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
        assert CFD_DTYPE is not None
        assert INDEX_DTYPE is not None
