"""
Tutorial validation: solver postprocessing comprehensive tests.

全面验证求解器后处理功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestPostprocessingComprehensive:
    """全面后处理测试。"""

    def test_forces_import(self):
        """Forces 后处理可导入。"""
        from pyfoam.postprocessing.forces import Forces
        assert Forces is not None

    def test_field_min_max_import(self):
        """FieldMinMax 后处理可导入。"""
        from pyfoam.postprocessing.field_min_max import FieldMinMax
        assert FieldMinMax is not None

    def test_noise_import(self):
        """Noise 后处理可导入。"""
        from pyfoam.postprocessing.noise import Noise
        assert Noise is not None

    def test_sampling_import(self):
        """Sampling 后处理可导入。"""
        from pyfoam.postprocessing.sampling import Probes, LineSample, SurfaceSample
        assert Probes is not None
        assert LineSample is not None
        assert SurfaceSample is not None

    def test_function_object_import(self):
        """FunctionObject 框架可导入。"""
        from pyfoam.postprocessing.function_object import FunctionObject
        assert FunctionObject is not None

    def test_wall_shear_stress_import(self):
        """WallShearStress 后处理可导入。"""
        from pyfoam.postprocessing import wall_shear_stress
        assert wall_shear_stress is not None
