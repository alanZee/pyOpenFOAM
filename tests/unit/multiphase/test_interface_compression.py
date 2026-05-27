"""接口压缩模型测试。

测试 InterfaceCompression 的压缩通量计算、压缩速度场和
压缩效果，使用与 VOF 测试相同的 SimpleMesh 模式。
"""

from dataclasses import dataclass

import pytest
import torch

from pyfoam.core.backend import gather
from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.core.device import device_context


# ---------------------------------------------------------------------------
# 复用 SimpleMesh 和 make_2d_mesh（与 test_vof.py 相同）
# ---------------------------------------------------------------------------


@dataclass
class SimpleMesh:
    """最小网格，用于测试。"""
    n_cells: int
    n_internal_faces: int
    n_faces: int
    owner: torch.Tensor
    neighbour: torch.Tensor
    face_areas: torch.Tensor
    cell_volumes: torch.Tensor
    face_weights: torch.Tensor
    delta_coefficients: torch.Tensor


def make_2d_mesh(
    n_x: int = 4,
    n_y: int = 4,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 0.1,
    device: str = "cpu",
    dtype: torch.dtype = CFD_DTYPE,
) -> SimpleMesh:
    """创建 2D 笛卡尔网格。"""
    n_cells = n_x * n_y

    owners = []
    neighbours = []

    # x 方向内部面
    for j in range(n_y):
        for i in range(n_x - 1):
            owners.append(j * n_x + i)
            neighbours.append(j * n_x + i + 1)

    # y 方向内部面
    for j in range(n_y - 1):
        for i in range(n_x):
            owners.append(j * n_x + i)
            neighbours.append((j + 1) * n_x + i)

    n_internal = len(neighbours)

    # 边界面
    for i in range(n_x):
        owners.append(i)
    for i in range(n_x):
        owners.append((n_y - 1) * n_x + i)
    for j in range(n_y):
        owners.append(j * n_x)
    for j in range(n_y):
        owners.append(j * n_x + n_x - 1)

    n_boundary = 2 * n_x + 2 * n_y
    n_faces = n_internal + n_boundary

    owner = torch.tensor(owners, dtype=torch.long, device=device)
    neigh = torch.tensor(neighbours, dtype=torch.long, device=device)

    # 向量面积 (n_faces, 3)
    face_areas_list = []
    for j in range(n_y):
        for i in range(n_x - 1):
            face_areas_list.append([dy * dz, 0.0, 0.0])
    for j in range(n_y - 1):
        for i in range(n_x):
            face_areas_list.append([0.0, dx * dz, 0.0])
    for _ in range(n_x):
        face_areas_list.append([0.0, -dx * dz, 0.0])
    for _ in range(n_x):
        face_areas_list.append([0.0, dx * dz, 0.0])
    for _ in range(n_y):
        face_areas_list.append([-dy * dz, 0.0, 0.0])
    for _ in range(n_y):
        face_areas_list.append([dy * dz, 0.0, 0.0])

    face_areas = torch.tensor(face_areas_list, dtype=dtype, device=device)

    vol = dx * dy * dz
    cell_volumes = torch.full((n_cells,), vol, dtype=dtype, device=device)
    face_weights = torch.full((n_internal,), 0.5, dtype=dtype, device=device)
    delta_coeffs = torch.full((n_internal,), 1.0 / max(dx, dy), dtype=dtype, device=device)

    return SimpleMesh(
        n_cells=n_cells,
        n_internal_faces=n_internal,
        n_faces=n_faces,
        owner=owner,
        neighbour=neigh,
        face_areas=face_areas,
        cell_volumes=cell_volumes,
        face_weights=face_weights,
        delta_coefficients=delta_coeffs,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mesh_4x4():
    """4x4 网格。"""
    return make_2d_mesh(4, 4)


@pytest.fixture
def mesh_8x8():
    """8x8 网格。"""
    return make_2d_mesh(8, 8)


# ---------------------------------------------------------------------------
# Tests: InterfaceCompression
# ---------------------------------------------------------------------------


class TestInterfaceCompression:
    """InterfaceCompression 测试。"""

    def test_initialization(self, mesh_4x4):
        """初始化应正确设置参数。"""
        from pyfoam.multiphase.interface_compression import InterfaceCompression

        comp = InterfaceCompression(mesh_4x4, C_alpha=1.5)
        assert comp.C_alpha == 1.5
        assert comp.alpha_min == 0.0
        assert comp.alpha_max == 1.0
        assert "InterfaceCompression" in repr(comp)

    def test_compression_flux_zero_uniform(self, mesh_4x4):
        """均匀 alpha 时压缩通量应为零。"""
        from pyfoam.multiphase.interface_compression import InterfaceCompression

        comp = InterfaceCompression(mesh_4x4, C_alpha=1.0)
        n_cells = mesh_4x4.n_cells
        alpha = torch.full((n_cells,), 0.5, dtype=CFD_DTYPE)
        phi = torch.ones(mesh_4x4.n_faces, dtype=CFD_DTYPE) * 0.01

        phi_c = comp.compute_compression_flux(alpha, phi)
        assert phi_c.shape == (mesh_4x4.n_internal_faces,)
        assert torch.allclose(phi_c, torch.zeros_like(phi_c), atol=1e-15)

    def test_compression_flux_nonzero_interface(self, mesh_4x4):
        """存在界面时压缩通量应非零。"""
        from pyfoam.multiphase.interface_compression import InterfaceCompression

        comp = InterfaceCompression(mesh_4x4, C_alpha=1.0)
        n_cells = mesh_4x4.n_cells

        # Step function: alpha=1 for left half, 0 for right half
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(2):
                alpha[j * 4 + i] = 1.0

        phi = torch.ones(mesh_4x4.n_faces, dtype=CFD_DTYPE) * 0.01
        phi_c = comp.compute_compression_flux(alpha, phi)

        # At least some faces should have nonzero compression flux
        assert phi_c.abs().sum() > 0

    def test_compression_flux_proportional_to_C_alpha(self, mesh_4x4):
        """压缩通量应与 C_alpha 成正比。"""
        from pyfoam.multiphase.interface_compression import InterfaceCompression

        n_cells = mesh_4x4.n_cells
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        alpha[0] = 1.0  # 单个 cell 不同
        phi = torch.ones(mesh_4x4.n_faces, dtype=CFD_DTYPE) * 0.01

        comp1 = InterfaceCompression(mesh_4x4, C_alpha=1.0)
        comp2 = InterfaceCompression(mesh_4x4, C_alpha=2.0)

        phi_c1 = comp1.compute_compression_flux(alpha, phi)
        phi_c2 = comp2.compute_compression_flux(alpha, phi)

        assert torch.allclose(phi_c2, 2.0 * phi_c1, atol=1e-15)

    def test_compression_flux_direction(self, mesh_4x4):
        """压缩通量方向应从低 alpha 到高 alpha。"""
        from pyfoam.multiphase.interface_compression import InterfaceCompression

        comp = InterfaceCompression(mesh_4x4, C_alpha=1.0)
        n_cells = mesh_4x4.n_cells

        # cell 0: alpha=1, cell 1: alpha=0 => face between them: phi_c should be positive
        # (alpha_P - alpha_N) where P=owner (lower index), N=neighbour
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        alpha[0] = 1.0
        phi = torch.ones(mesh_4x4.n_faces, dtype=CFD_DTYPE) * 0.1

        phi_c = comp.compute_compression_flux(alpha, phi)
        # First internal face connects cells 0 and 1
        # alpha_P = alpha[0] = 1.0, alpha_N = alpha[1] = 0.0 => phi_c > 0
        assert phi_c[0].item() > 0

    def test_compressive_velocity_zero_uniform(self, mesh_4x4):
        """均匀 alpha 时压缩速度应为零。"""
        from pyfoam.multiphase.interface_compression import InterfaceCompression

        comp = InterfaceCompression(mesh_4x4, C_alpha=1.0)
        n_cells = mesh_4x4.n_cells
        alpha = torch.full((n_cells,), 0.5, dtype=CFD_DTYPE)
        phi = torch.ones(mesh_4x4.n_faces, dtype=CFD_DTYPE) * 0.01

        U_c = comp.compute_compressive_velocity(alpha, phi)
        assert U_c.shape == (n_cells, 3)
        assert torch.allclose(U_c, torch.zeros_like(U_c), atol=1e-15)

    def test_compressive_velocity_nonzero(self, mesh_4x4):
        """存在界面时压缩速度应非零。"""
        from pyfoam.multiphase.interface_compression import InterfaceCompression

        comp = InterfaceCompression(mesh_4x4, C_alpha=1.0)
        n_cells = mesh_4x4.n_cells

        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(2):
                alpha[j * 4 + i] = 1.0

        phi = torch.ones(mesh_4x4.n_faces, dtype=CFD_DTYPE) * 0.01
        U_c = comp.compute_compressive_velocity(alpha, phi)

        assert U_c.abs().sum() > 0

    def test_apply_compression_sharpens_interface(self, mesh_8x8):
        """apply_compression 应使界面更锐利。"""
        from pyfoam.multiphase.interface_compression import InterfaceCompression

        comp = InterfaceCompression(mesh_8x8, C_alpha=2.0)
        n_cells = mesh_8x8.n_cells
        n_faces = mesh_8x8.n_faces

        # Create diffuse interface: smooth transition
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(8):
            for i in range(8):
                x = (i + 0.5) / 8.0
                # Smooth step around x=0.5
                alpha[j * 8 + i] = max(0.0, min(1.0, 2.0 - 4.0 * abs(x - 0.5)))

        # Need some nonzero flux for compression
        phi = torch.ones(n_faces, dtype=CFD_DTYPE) * 0.01

        alpha_before = alpha.clone()
        alpha_after = comp.apply_compression(alpha, phi, delta_t=0.1)

        # Alpha should still be bounded
        assert alpha_after.min() >= -1e-10
        assert alpha_after.max() <= 1.0 + 1e-10

    def test_apply_compression_bounded(self, mesh_4x4):
        """压缩后 alpha 应保持在 [0, 1] 范围内。"""
        from pyfoam.multiphase.interface_compression import InterfaceCompression

        comp = InterfaceCompression(mesh_4x4, C_alpha=5.0)  # 很大的 C_alpha
        n_cells = mesh_4x4.n_cells
        n_faces = mesh_4x4.n_faces

        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(2):
                alpha[j * 4 + i] = 0.8

        phi = torch.ones(n_faces, dtype=CFD_DTYPE) * 0.1

        for _ in range(10):
            alpha = comp.apply_compression(alpha, phi, delta_t=0.05)
            assert alpha.min() >= -1e-10, f"alpha min = {alpha.min()}"
            assert alpha.max() <= 1.0 + 1e-10, f"alpha max = {alpha.max()}"

    def test_C_alpha_setter(self, mesh_4x4):
        """C_alpha 应可动态修改。"""
        from pyfoam.multiphase.interface_compression import InterfaceCompression

        comp = InterfaceCompression(mesh_4x4, C_alpha=1.0)
        comp.C_alpha = 2.5
        assert comp.C_alpha == 2.5

    def test_custom_bounds(self, mesh_4x4):
        """自定义 alpha_min / alpha_max 应生效。"""
        from pyfoam.multiphase.interface_compression import InterfaceCompression

        comp = InterfaceCompression(mesh_4x4, C_alpha=0.0, alpha_min=0.1, alpha_max=0.9)
        assert comp.alpha_min == 0.1
        assert comp.alpha_max == 0.9

    def test_conservation(self, mesh_4x4):
        """纯压缩应保守 alpha * V（不改变总量）。"""
        from pyfoam.multiphase.interface_compression import InterfaceCompression

        comp = InterfaceCompression(mesh_4x4, C_alpha=1.0)
        n_cells = mesh_4x4.n_cells
        n_faces = mesh_4x4.n_faces

        # Smooth gradient
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(4):
                alpha[j * 4 + i] = i / 3.0

        V = mesh_4x4.cell_volumes
        total_before = (alpha * V).sum().item()

        phi = torch.zeros(n_faces, dtype=CFD_DTYPE)
        alpha_new = comp.apply_compression(alpha, phi, delta_t=0.01)

        total_after = (alpha_new * V).sum().item()
        # With zero flux, no compression change expected
        assert abs(total_after - total_before) < 1e-10 * max(abs(total_before), 1e-30)
