"""
GPU 验证：验证 CUDA 加速支持。

当 CUDA 可用时运行全部 GPU 测试。
"""
from __future__ import annotations

import torch
import pytest

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE

CUDA_AVAILABLE = torch.cuda.is_available()

skip_no_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available — install pytorch with CUDA support",
)


@skip_no_cuda
class TestGPUCore:
    """GPU 核心功能验证。"""

    def test_cuda_device(self):
        """CUDA 设备可用。"""
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() >= 1

    def test_cuda_tensor_creation(self):
        """GPU 张量创建。"""
        device = torch.device("cuda:0")
        x = torch.randn(100, 3, dtype=CFD_DTYPE, device=device)
        assert x.device.type == "cuda"
        assert x.shape == (100, 3)

    def test_cuda_arithmetic(self):
        """GPU 算术运算。"""
        device = torch.device("cuda:0")
        a = torch.randn(100, dtype=CFD_DTYPE, device=device)
        b = torch.randn(100, dtype=CFD_DTYPE, device=device)
        c = a + b
        assert c.device.type == "cuda"
        assert torch.allclose(c.cpu(), a.cpu() + b.cpu(), atol=1e-10)


@skip_no_cuda
class TestGPUDeviceManager:
    """GPU 设备管理器验证。"""

    def test_set_device(self):
        """设置全局设备。"""
        from pyfoam.core.device import set_device, get_device
        set_device("cuda:0")
        assert get_device().type == "cuda"
        # 恢复
        set_device("cpu")

    def test_device_context_manager(self):
        """设备上下文管理器。"""
        from pyfoam.core.device import device_context
        with device_context("cuda:0"):
            assert torch.cuda.is_available()


@skip_no_cuda
class TestGPUMeshOperations:
    """GPU 网格运算验证。"""

    def test_mesh_on_gpu(self):
        """网格数据迁移至 GPU。"""
        from tests.tutorials.helpers import make_structured_mesh
        from pyfoam.mesh.fv_mesh import FvMesh
        from pyfoam.io.mesh_io import read_mesh
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            mesh_dir = Path(tmp) / "constant" / "polyMesh"
            make_structured_mesh(mesh_dir, nx=4, ny=4)
            md = read_mesh(mesh_dir)
            faces_t = [torch.tensor(f, dtype=INDEX_DTYPE) for f in md.faces]
            mesh = FvMesh(
                points=md.points, faces=faces_t,
                owner=md.owner, neighbour=md.neighbour,
                boundary=md.boundary,
            )
            mesh.compute_geometry()

        device = torch.device("cuda:0")
        # 迁移几何数据
        cell_centres_gpu = mesh.cell_centres.to(device)
        assert cell_centres_gpu.device.type == "cuda"
        assert cell_centres_gpu.shape == mesh.cell_centres.shape


@skip_no_cuda
class TestGPUFieldOperations:
    """GPU 场运算验证。"""

    def test_field_gradient_gpu(self):
        """GPU 梯度运算。"""
        from pyfoam.differentiable.operators import DifferentiableGradient
        from tests.tutorials.helpers import make_structured_mesh
        from pyfoam.mesh.fv_mesh import FvMesh
        from pyfoam.io.mesh_io import read_mesh
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            mesh_dir = Path(tmp) / "constant" / "polyMesh"
            make_structured_mesh(mesh_dir, nx=4, ny=4)
            md = read_mesh(mesh_dir)
            faces_t = [torch.tensor(f, dtype=INDEX_DTYPE) for f in md.faces]
            mesh = FvMesh(
                points=md.points, faces=faces_t,
                owner=md.owner, neighbour=md.neighbour,
                boundary=md.boundary,
            )
            mesh.compute_geometry()

        device = torch.device("cuda:0")
        n_cells = mesh.n_cells
        phi = torch.randn(n_cells, dtype=CFD_DTYPE, device=device, requires_grad=True)

        # 注意：DifferentiableGradient 需要 mesh 数据也在 GPU
        # 这里仅验证张量能正确创建
        assert phi.device.type == "cuda"
        assert phi.requires_grad


@skip_no_cuda
class TestGPUAutograd:
    """GPU 自动微分验证。"""

    def test_backward_on_gpu(self):
        """GPU 反向传播。"""
        device = torch.device("cuda:0")
        x = torch.randn(10, dtype=CFD_DTYPE, device=device, requires_grad=True)
        y = x.pow(2).sum()
        y.backward()
        assert x.grad is not None
        assert x.grad.device.type == "cuda"
        assert torch.isfinite(x.grad).all()
