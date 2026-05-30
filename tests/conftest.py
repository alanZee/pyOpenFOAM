"""全局测试配置：强制使用 CPU 设备，避免 CUDA/CPU 设备不匹配问题。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch


@pytest.fixture(autouse=True)
def _force_cpu():
    """所有测试自动使用 CPU 设备。"""
    from pyfoam.core.device import _tensor_config
    _tensor_config.device = torch.device("cpu")
    yield
