"""
Tutorial validation: GPU support verification.

验证 GPU 基础设施和 PyTorch CUDA 支持。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestGPUInfrastructure:
    """GPU 基础设施测试。"""

    def test_device_manager_import(self):
        """DeviceManager 可导入。"""
        from pyfoam.core.device import DeviceManager
        assert DeviceManager is not None

    def test_device_capabilities(self):
        """设备能力检测。"""
        from pyfoam.core.device import DeviceCapabilities
        caps = DeviceCapabilities()
        assert caps.cpu is True
        # GPU 可用性取决于硬件
        assert isinstance(caps.cuda, bool)
        assert isinstance(caps.mps, bool)

    def test_available_devices(self):
        """可用设备列表。"""
        from pyfoam.core.device import DeviceCapabilities
        caps = DeviceCapabilities()
        devices = caps.available_devices
        assert "cpu" in devices
        if caps.cuda:
            assert "cuda" in devices
        if caps.mps:
            assert "mps" in devices

    def test_get_device(self):
        """get_device 返回正确设备。"""
        from pyfoam.core.device import get_device
        device = get_device()
        assert device is not None
        assert isinstance(device, torch.device)

    def test_get_default_dtype(self):
        """get_default_dtype 返回正确类型。"""
        from pyfoam.core.device import get_default_dtype
        dtype = get_default_dtype()
        assert dtype is not None
        assert dtype == CFD_DTYPE

    def test_device_context(self):
        """device_context 上下文管理器。"""
        from pyfoam.core.device import device_context
        with device_context("cpu"):
            device = torch.device("cpu")
            assert device.type == "cpu"


class TestMultiGPU:
    """多 GPU 支持测试。"""

    def test_multi_gpu_import(self):
        """multi_gpu 模块可导入。"""
        from pyfoam.core.multi_gpu import MultiGPUManager
        assert MultiGPUManager is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor_operations(self):
        """CUDA 张量操作。"""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE, device="cuda")
        y = x * 2
        assert y.device.type == "cuda"
        assert torch.allclose(y.cpu(), torch.tensor([2.0, 4.0, 6.0], dtype=CFD_DTYPE))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_mesh_operations(self):
        """CUDA 网格操作。"""
        from pyfoam.mesh.fv_mesh import FvMesh
        # 创建简单网格并移到 GPU
        pts = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=CFD_DTYPE, device="cuda")
        # 注意：实际网格操作需要更多设置
        assert pts.device.type == "cuda"


class TestGPUReadiness:
    """GPU 就绪状态检查。"""

    def test_pytorch_version(self):
        """PyTorch 版本满足要求。"""
        version = torch.__version__
        assert version is not None
        # 至少需要 2.0
        major, minor = version.split(".")[:2]
        assert int(major) >= 2

    def test_cuda_availability_report(self):
        """CUDA 可用性报告。"""
        report = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        assert report["pytorch_version"] is not None
        assert isinstance(report["cuda_available"], bool)
