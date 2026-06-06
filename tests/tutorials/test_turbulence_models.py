"""
Tutorial validation: turbulence model tests.

验证湍流模型的基本功能。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


class TestTurbulenceModelSmoke:
    """湍流模型 smoke 测试。"""

    def test_k_epsilon_import(self):
        """k-epsilon 模型可导入。"""
        from pyfoam.turbulence.k_epsilon import KEpsilonModel
        assert KEpsilonModel is not None

    def test_k_omega_sst_import(self):
        """k-omega SST 模型可导入。"""
        from pyfoam.turbulence.k_omega_sst import KOmegaSSTModel
        assert KOmegaSSTModel is not None

    def test_spalart_allmaras_import(self):
        """Spalart-Allmaras 模型可导入。"""
        from pyfoam.turbulence.spalart_allmaras import SpalartAllmarasModel
        assert SpalartAllmarasModel is not None

    def test_buoyant_kepsilon_import(self):
        """浮力 k-epsilon 模型可导入。"""
        from pyfoam.turbulence.buoyant_kepsilon import BuoyantKEpsilon
        assert BuoyantKEpsilon is not None

    def test_komega_sst_sato_import(self):
        """Sato 气泡诱导湍流模型可导入。"""
        from pyfoam.turbulence.komega_sst_sato import KOmegaSSTSato
        assert KOmegaSSTSato is not None

    def test_lahey_kepsilon_import(self):
        """Lahey k-epsilon 模型可导入。"""
        from pyfoam.turbulence.lahey_kepsilon import LaheyKEpsilon
        assert LaheyKEpsilon is not None

    def test_smagorinsky_import(self):
        """Smagorinsky LES 模型可导入。"""
        from pyfoam.turbulence.smagorinsky import SmagorinskyModel
        assert SmagorinskyModel is not None

    def test_wale_import(self):
        """WALE LES 模型可导入。"""
        from pyfoam.turbulence.wale import WALEModel
        assert WALEModel is not None

    def test_dynamic_smagorinsky_import(self):
        """Dynamic Smagorinsky LES 模型可导入。"""
        from pyfoam.turbulence.dynamic_smagorinsky import DynamicSmagorinskyModel
        assert DynamicSmagorinskyModel is not None


class TestViscoelasticModelSmoke:
    """粘弹性模型 smoke 测试。"""

    def test_maxwell_import(self):
        """Maxwell 模型可导入。"""
        from pyfoam.turbulence.viscoelastic_models import MaxwellModel
        assert MaxwellModel is not None

    def test_giesekus_import(self):
        """Giesekus 模型可导入。"""
        from pyfoam.turbulence.viscoelastic_models import GiesekusModel
        assert GiesekusModel is not None

    def test_ptt_import(self):
        """PTT 模型可导入。"""
        from pyfoam.turbulence.viscoelastic_models import PTTModel
        assert PTTModel is not None


class TestGeneralizedNewtonianSmoke:
    """广义牛顿模型 smoke 测试。"""

    def test_bird_carreau_import(self):
        """Bird-Carreau 模型可导入。"""
        from pyfoam.turbulence.generalized_newtonian_v2 import BirdCarreauModel
        assert BirdCarreauModel is not None

    def test_herschel_bulkley_import(self):
        """Herschel-Bulkley 模型可导入。"""
        from pyfoam.turbulence.generalized_newtonian_v2 import HerschelBulkleyModel
        assert HerschelBulkleyModel is not None

    def test_cross_power_law_import(self):
        """Cross 幂律模型可导入。"""
        from pyfoam.turbulence.generalized_newtonian_v2 import CrossPowerLawModel
        assert CrossPowerLawModel is not None

    def test_casson_import(self):
        """Casson 模型可导入。"""
        from pyfoam.turbulence.generalized_newtonian_v2 import CassonModel
        assert CassonModel is not None
