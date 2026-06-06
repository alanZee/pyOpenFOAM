"""
Tutorial validation: postprocessing smoke tests.

验证后处理功能的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestPostprocessingSmoke:
    """后处理 smoke 测试。"""

    def test_forces_import(self):
        """forces 后处理可导入。"""
        from pyfoam.postprocessing.forces import Forces
        assert Forces is not None

    def test_field_min_max_import(self):
        """fieldMinMax 后处理可导入。"""
        from pyfoam.postprocessing.field_min_max import FieldMinMax
        assert FieldMinMax is not None

    def test_noise_import(self):
        """noise 后处理可导入。"""
        from pyfoam.postprocessing.noise import Noise
        assert Noise is not None

    def test_sample_import(self):
        """sample 后处理可导入。"""
        from pyfoam.postprocessing.sampling import Probes, LineSample, SurfaceSample
        assert Probes is not None
        assert LineSample is not None
        assert SurfaceSample is not None

    def test_function_object_import(self):
        """FunctionObject 框架可导入。"""
        from pyfoam.postprocessing.function_object import FunctionObject
        assert FunctionObject is not None


class TestWavesSmoke:
    """波浪模型 smoke 测试。"""

    def test_airy_import(self):
        """Airy 波浪模型可导入。"""
        from pyfoam.waves import AiryWave
        assert AiryWave is not None

    def test_stokes_import(self):
        """Stokes 波浪模型可导入。"""
        from pyfoam.waves import StokesWave
        assert StokesWave is not None

    def test_cnoidal_import(self):
        """Cnoidal 波浪模型可导入。"""
        from pyfoam.waves import CnoidalWave
        assert CnoidalWave is not None


class TestRigidBodySmoke:
    """刚体运动 smoke 测试。"""

    def test_rigid_body_import(self):
        """RigidBody 可导入。"""
        from pyfoam.rigid_body import RigidBodySolver, SixDoFSolver, Joint
        assert RigidBodySolver is not None
        assert SixDoFSolver is not None
        assert Joint is not None


class TestStructuralSmoke:
    """结构力学 smoke 测试。"""

    def test_elastic_model_import(self):
        """ElasticModel 可导入。"""
        from pyfoam.structural import LinearElasticModel, DisplacementSolver, StressSolver
        assert LinearElasticModel is not None
        assert DisplacementSolver is not None
        assert StressSolver is not None


class TestParallelSmoke:
    """并行计算 smoke 测试。"""

    def test_decomposition_import(self):
        """Decomposition 可导入。"""
        from pyfoam.parallel import Decomposition
        assert Decomposition is not None

    def test_halo_exchange_import(self):
        """HaloExchange 可导入。"""
        from pyfoam.parallel import HaloExchange
        assert HaloExchange is not None
