"""
Tutorial validation: physical properties and specie transfer smoke tests.

验证物性参数和组分传输的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestPhysicalPropertiesSmoke:
    """物性参数 smoke 测试。"""

    def test_physical_properties_import(self):
        """PhysicalProperties 可导入。"""
        from pyfoam.physical_properties import PhysicalProperties
        assert PhysicalProperties is not None

    def test_constant_viscosity_import(self):
        """ConstantViscosity 可导入。"""
        from pyfoam.physical_properties import ConstantViscosity
        assert ConstantViscosity is not None

    def test_polynomial_viscosity_import(self):
        """PolynomialViscosity 可导入。"""
        from pyfoam.physical_properties import PolynomialViscosity
        assert PolynomialViscosity is not None


class TestSpecieTransferSmoke:
    """组分传输 smoke 测试。"""

    def test_specie_transfer_model_import(self):
        """SpecieTransferModel 可导入。"""
        from pyfoam.specie_transfer import SpecieTransferModel
        assert SpecieTransferModel is not None

    def test_simple_diffusion_import(self):
        """SimpleDiffusionModel 可导入。"""
        from pyfoam.specie_transfer import SimpleDiffusionModel
        assert SimpleDiffusionModel is not None


class TestPolyTopoChangeSmoke:
    """拓扑修改 smoke 测试。"""

    def test_poly_topo_change_import(self):
        """PolyTopoChange 可导入。"""
        from pyfoam.poly_topo_change import PolyTopoChange
        assert PolyTopoChange is not None

    def test_topo_set_import(self):
        """TopoSet 可导入。"""
        from pyfoam.poly_topo_change import TopoSet
        assert TopoSet is not None

    def test_box_to_cell_import(self):
        """BoxToCell 可导入。"""
        from pyfoam.poly_topo_change import BoxToCell
        assert BoxToCell is not None

    def test_cylinder_to_cell_import(self):
        """CylinderToCell 可导入。"""
        from pyfoam.poly_topo_change import CylinderToCell
        assert CylinderToCell is not None
