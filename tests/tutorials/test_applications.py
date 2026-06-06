"""
Tutorial validation: application smoke tests.

验证所有求解器应用的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestApplicationSmoke:
    """求解器应用 smoke 测试。"""

    def test_simple_foam_import(self):
        """SimpleFoam 可导入。"""
        from pyfoam.applications import SimpleFoam
        assert SimpleFoam is not None

    def test_piso_foam_import(self):
        """PisoFoam 可导入。"""
        from pyfoam.applications import PisoFoam
        assert PisoFoam is not None

    def test_pimple_foam_import(self):
        """PimpleFoam 可导入。"""
        from pyfoam.applications import PimpleFoam
        assert PimpleFoam is not None

    def test_ico_foam_import(self):
        """IcoFoam 可导入。"""
        from pyfoam.applications import IcoFoam
        assert IcoFoam is not None

    def test_inter_foam_import(self):
        """InterFoam 可导入。"""
        from pyfoam.applications import InterFoam
        assert InterFoam is not None

    def test_buoyant_simple_foam_import(self):
        """BuoyantSimpleFoam 可导入。"""
        from pyfoam.applications import BuoyantSimpleFoam
        assert BuoyantSimpleFoam is not None

    def test_buoyant_pimple_foam_import(self):
        """BuoyantPimpleFoam 可导入。"""
        from pyfoam.applications import BuoyantPimpleFoam
        assert BuoyantPimpleFoam is not None

    def test_sonic_foam_import(self):
        """SonicFoam 可导入。"""
        from pyfoam.applications import SonicFoam
        assert SonicFoam is not None

    def test_rho_central_foam_import(self):
        """RhoCentralFoam 可导入。"""
        from pyfoam.applications import RhoCentralFoam
        assert RhoCentralFoam is not None

    def test_laplacian_foam_import(self):
        """LaplacianFoam 可导入。"""
        from pyfoam.applications import LaplacianFoam
        assert LaplacianFoam is not None

    def test_scalar_transport_foam_import(self):
        """ScalarTransportFoam 可导入。"""
        from pyfoam.applications import ScalarTransportFoam
        assert ScalarTransportFoam is not None

    def test_potential_foam_import(self):
        """PotentialFoam 可导入。"""
        from pyfoam.applications import PotentialFoam
        assert PotentialFoam is not None

    def test_multiphase_euler_foam_import(self):
        """MultiphaseEulerFoam 可导入。"""
        from pyfoam.applications import MultiphaseEulerFoam
        assert MultiphaseEulerFoam is not None

    def test_compressible_inter_foam_import(self):
        """CompressibleInterFoam 可导入。"""
        from pyfoam.applications import CompressibleInterFoam
        assert CompressibleInterFoam is not None

    def test_cht_multi_region_foam_import(self):
        """ChtMultiRegionFoam 可导入。"""
        from pyfoam.applications import CHTMultiRegionFoam
        assert CHTMultiRegionFoam is not None
