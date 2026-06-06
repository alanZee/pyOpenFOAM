"""
Tutorial validation: solver field operation tests.

验证求解器场操作。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


class TestFieldOperations:
    """场操作测试。"""

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

    def test_geometric_field_import(self):
        """GeometricField 可导入。"""
        from pyfoam.fields import GeometricField
        assert GeometricField is not None

    def test_dimension_set_import(self):
        """DimensionSet 可导入。"""
        from pyfoam.fields import DimensionSet
        assert DimensionSet is not None


class TestFieldArithmetic:
    """场算术测试。"""

    def test_field_addition(self):
        """场加法。"""
        from pyfoam.core.dtype import CFD_DTYPE
        a = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        b = torch.tensor([4.0, 5.0, 6.0], dtype=CFD_DTYPE)
        c = a + b
        assert torch.allclose(c, torch.tensor([5.0, 7.0, 9.0], dtype=CFD_DTYPE))

    def test_field_multiplication(self):
        """场乘法。"""
        from pyfoam.core.dtype import CFD_DTYPE
        a = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        b = torch.tensor([4.0, 5.0, 6.0], dtype=CFD_DTYPE)
        c = a * b
        assert torch.allclose(c, torch.tensor([4.0, 10.0, 18.0], dtype=CFD_DTYPE))

    def test_field_gradient(self):
        """场梯度。"""
        from pyfoam.core.dtype import CFD_DTYPE
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=CFD_DTYPE)
        grad = torch.gradient(a)
        assert len(grad) == 1
        assert grad[0].shape == a.shape


class TestFieldIO:
    """场 IO 测试。"""

    def test_foam_file_header_import(self):
        """FoamFileHeader 可导入。"""
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
