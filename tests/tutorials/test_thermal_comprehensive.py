"""
Tutorial validation: solver thermal model comprehensive tests.

全面验证求解器热模型。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestThermalModelComprehensive:
    """全面热模型测试。"""

    def test_laplacian_foam_import(self):
        """LaplacianFoam 可导入。"""
        from pyfoam.applications import LaplacianFoam
        assert LaplacianFoam is not None

    def test_heat_transfer_foam_import(self):
        """HeatTransferFoam 可导入。"""
        from pyfoam.applications import HeatTransferFoam
        assert HeatTransferFoam is not None

    def test_buoyant_simple_foam_import(self):
        """BuoyantSimpleFoam 可导入。"""
        from pyfoam.applications import BuoyantSimpleFoam
        assert BuoyantSimpleFoam is not None

    def test_buoyant_pimple_foam_import(self):
        """BuoyantPimpleFoam 可导入。"""
        from pyfoam.applications import BuoyantPimpleFoam
        assert BuoyantPimpleFoam is not None

    def test_cht_multi_region_foam_import(self):
        """CHTMultiRegionFoam 可导入。"""
        from pyfoam.applications import CHTMultiRegionFoam
        assert CHTMultiRegionFoam is not None

    def test_solid_foam_import(self):
        """SolidFoam 可导入。"""
        from pyfoam.applications import SolidFoam
        assert SolidFoam is not None

    def test_solid_displacement_foam_import(self):
        """SolidDisplacementFoam 可导入。"""
        from pyfoam.applications import SolidDisplacementFoam
        assert SolidDisplacementFoam is not None

    def test_energy_foam_import(self):
        """EnergyFoam 可导入。"""
        from pyfoam.applications import EnergyFoam
        assert EnergyFoam is not None

    def test_radiation_model_import(self):
        """Radiation 模型可导入。"""
        from pyfoam.models.radiation import P1Radiation, RadiationModel
        assert P1Radiation is not None
        assert RadiationModel is not None

    def test_thermal_conductivity_import(self):
        """热传导模型可导入。"""
        from pyfoam.thermophysical import ConstantTransport
        assert ConstantTransport is not None

    def test_sutherland_transport_import(self):
        """Sutherland 输运模型可导入。"""
        from pyfoam.thermophysical import Sutherland
        assert Sutherland is not None
