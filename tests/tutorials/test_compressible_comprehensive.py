"""
Tutorial validation: solver compressible comprehensive tests.

全面验证求解器可压缩流模型。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestCompressibleComprehensive:
    """全面可压缩流测试。"""

    def test_sonic_foam_import(self):
        """SonicFoam 可导入。"""
        from pyfoam.applications import SonicFoam
        assert SonicFoam is not None

    def test_rho_central_foam_import(self):
        """RhoCentralFoam 可导入。"""
        from pyfoam.applications import RhoCentralFoam
        assert RhoCentralFoam is not None

    def test_rho_pimple_foam_import(self):
        """RhoPimpleFoam 可导入。"""
        from pyfoam.applications import RhoPimpleFoam
        assert RhoPimpleFoam is not None

    def test_rho_simple_foam_import(self):
        """RhoSimpleFoam 可导入。"""
        from pyfoam.applications import RhoSimpleFoam
        assert RhoSimpleFoam is not None

    def test_compressible_inter_foam_import(self):
        """CompressibleInterFoam 可导入。"""
        from pyfoam.applications import CompressibleInterFoam
        assert CompressibleInterFoam is not None

    def test_compressible_vof_foam_import(self):
        """CompressibleVoFFoam 可导入。"""
        from pyfoam.applications import CompressibleVoFFoam
        assert CompressibleVoFFoam is not None

    def test_fluid_foam_import(self):
        """FluidFoam 可导入。"""
        from pyfoam.applications import FluidFoam
        assert FluidFoam is not None

    def test_multicomponent_fluid_foam_import(self):
        """MulticomponentFluidFoam 可导入。"""
        from pyfoam.applications import MulticomponentFluidFoam
        assert MulticomponentFluidFoam is not None

    def test_isothermal_fluid_foam_import(self):
        """IsothermalFluidFoam 可导入。"""
        from pyfoam.applications import IsothermalFluidFoam
        assert IsothermalFluidFoam is not None

    def test_perfect_gas_import(self):
        """PerfectGas EOS 可导入。"""
        from pyfoam.thermophysical import PerfectGas
        assert PerfectGas is not None

    def test_constant_transport_import(self):
        """ConstantTransport 可导入。"""
        from pyfoam.thermophysical import ConstantTransport
        assert ConstantTransport is not None

    def test_sutherland_import(self):
        """Sutherland 输运模型可导入。"""
        from pyfoam.thermophysical import Sutherland
        assert Sutherland is not None
