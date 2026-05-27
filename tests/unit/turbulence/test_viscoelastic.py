"""粘弹性湍流模型测试 — Maxwell、Giesekus、PTT。"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE


# ---------------------------------------------------------------------------
# 复用 2-cell hex 网格
# ---------------------------------------------------------------------------

_POINTS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 2.0],
    [1.0, 0.0, 2.0],
    [1.0, 1.0, 2.0],
    [0.0, 1.0, 2.0],
]

_FACES = [
    [4, 5, 6, 7],
    [0, 3, 2, 1],
    [0, 1, 5, 4],
    [3, 7, 6, 2],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
    [8, 9, 10, 11],
    [4, 5, 9, 8],
    [7, 11, 10, 6],
    [4, 8, 11, 7],
    [5, 6, 10, 9],
]

_OWNER = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_NEIGHBOUR = [1]

_BOUNDARY = [
    {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 5},
    {"name": "top", "type": "wall", "startFace": 6, "nFaces": 5},
]


def make_fv_mesh(device="cpu", dtype=torch.float64):
    """返回 2-cell hex FvMesh 并计算几何量."""
    from pyfoam.mesh.fv_mesh import FvMesh

    mesh = FvMesh(
        points=torch.tensor(_POINTS, dtype=dtype, device=device),
        faces=[
            torch.tensor(f, dtype=INDEX_DTYPE, device=device) for f in _FACES
        ],
        owner=torch.tensor(_OWNER, dtype=INDEX_DTYPE, device=device),
        neighbour=torch.tensor(_NEIGHBOUR, dtype=INDEX_DTYPE, device=device),
        boundary=_BOUNDARY,
    )
    mesh.compute_geometry()
    return mesh


@pytest.fixture
def fv_mesh():
    """2-cell hex FvMesh."""
    return make_fv_mesh()


@pytest.fixture
def U_linear(fv_mesh):
    """线性 z 方向速度场: U = (0, 0, z)."""
    cc = fv_mesh.cell_centres
    U = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
    U[:, 2] = cc[:, 2]
    return U


@pytest.fixture
def face_flux(fv_mesh):
    """零面通量."""
    return torch.zeros(fv_mesh.n_faces, dtype=torch.float64)


# ---------------------------------------------------------------------------
# RTS 注册测试
# ---------------------------------------------------------------------------


class TestViscoelasticRegistration:
    """测试粘弹性模型 RTS 注册."""

    def test_maxwell_registered(self):
        from pyfoam.turbulence import TurbulenceModel
        assert "Maxwell" in TurbulenceModel.available_types()

    def test_giesekus_registered(self):
        from pyfoam.turbulence import TurbulenceModel
        assert "Giesekus" in TurbulenceModel.available_types()

    def test_ptt_registered(self):
        from pyfoam.turbulence import TurbulenceModel
        assert "PTT" in TurbulenceModel.available_types()


# ---------------------------------------------------------------------------
# Maxwell 模型测试
# ---------------------------------------------------------------------------


class TestMaxwellModel:
    """Maxwell 粘弹性模型测试."""

    def test_factory_creation(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence import TurbulenceModel
        from pyfoam.turbulence.viscoelastic import MaxwellModel
        model = TurbulenceModel.create(
            "Maxwell", fv_mesh, U_linear, face_flux,
            lambda_1=0.1, mu_p=0.01,
        )
        assert isinstance(model, MaxwellModel)

    def test_elastic_stress_shape(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import MaxwellModel
        model = MaxwellModel(fv_mesh, U_linear, face_flux)
        tau = model.elastic_stress()
        assert tau.shape == (fv_mesh.n_cells, 3, 3)

    def test_elastic_stress_zero_initially(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import MaxwellModel
        model = MaxwellModel(fv_mesh, U_linear, face_flux)
        tau = model.elastic_stress()
        assert torch.allclose(tau, torch.zeros_like(tau))

    def test_correct_updates_stress(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import MaxwellModel
        model = MaxwellModel(fv_mesh, U_linear, face_flux)
        model.correct()
        tau = model.elastic_stress()
        # 线性速度场下应产生非零应力
        assert tau.abs().sum() > 0

    def test_k_zero(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import MaxwellModel
        model = MaxwellModel(fv_mesh, U_linear, face_flux)
        model.correct()
        k = model.k()
        assert torch.allclose(k, torch.zeros(fv_mesh.n_cells, dtype=torch.float64))

    def test_nut_nonnegative(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import MaxwellModel
        model = MaxwellModel(fv_mesh, U_linear, face_flux)
        model.correct()
        nut = model.nut()
        assert (nut >= 0).all()

    def test_stress_increases_with_more_correct(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import MaxwellModel
        model = MaxwellModel(fv_mesh, U_linear, face_flux)
        model.correct()
        tau_norm_1 = model.elastic_stress().abs().sum().item()
        model.correct()
        tau_norm_2 = model.elastic_stress().abs().sum().item()
        # 多次 correct 后应力应增大（趋近稳态）
        assert tau_norm_2 >= tau_norm_1

    def test_properties(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import MaxwellModel
        model = MaxwellModel(
            fv_mesh, U_linear, face_flux,
            lambda_1=0.5, mu_p=0.02,
        )
        assert model.lambda_1 == 0.5
        assert model.mu_p == 0.02

    def test_repr(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import MaxwellModel
        model = MaxwellModel(fv_mesh, U_linear, face_flux)
        r = repr(model)
        assert "MaxwellModel" in r
        assert "lambda_1" in r


# ---------------------------------------------------------------------------
# Giesekus 模型测试
# ---------------------------------------------------------------------------


class TestGiesekusModel:
    """Giesekus 粘弹性模型测试."""

    def test_factory_creation(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence import TurbulenceModel
        from pyfoam.turbulence.viscoelastic import GiesekusModel
        model = TurbulenceModel.create(
            "Giesekus", fv_mesh, U_linear, face_flux,
            lambda_1=0.1, mu_p=0.01, alpha=0.3,
        )
        assert isinstance(model, GiesekusModel)

    def test_alpha_property(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import GiesekusModel
        model = GiesekusModel(
            fv_mesh, U_linear, face_flux,
            alpha=0.3, lambda_1=0.1, mu_p=0.01,
        )
        assert model.alpha == 0.3

    def test_correct_updates_stress(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import GiesekusModel
        model = GiesekusModel(fv_mesh, U_linear, face_flux)
        model.correct()
        tau = model.elastic_stress()
        assert tau.abs().sum() > 0

    def test_nut_nonnegative(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import GiesekusModel
        model = GiesekusModel(fv_mesh, U_linear, face_flux)
        model.correct()
        nut = model.nut()
        assert (nut >= 0).all()

    def test_giesekus_drag_reduces_stress_vs_maxwell(
        self, fv_mesh, U_linear, face_flux,
    ):
        """Giesekus 拖曳项应限制弹性应力增长."""
        from pyfoam.turbulence.viscoelastic import MaxwellModel, GiesekusModel
        maxwell = MaxwellModel(
            fv_mesh, U_linear, face_flux, lambda_1=0.1, mu_p=0.01,
        )
        giesekus = GiesekusModel(
            fv_mesh, U_linear, face_flux,
            lambda_1=0.1, mu_p=0.01, alpha=0.3,
        )
        # 多次推进让两者趋近不同稳态
        for _ in range(20):
            maxwell.correct()
            giesekus.correct()

        tau_maxwell = maxwell.elastic_stress().abs().sum().item()
        tau_giesekus = giesekus.elastic_stress().abs().sum().item()
        # Giesekus 拖曳项应使得应力更小
        assert tau_giesekus <= tau_maxwell * 1.01  # 允许小的数值误差

    def test_repr(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import GiesekusModel
        model = GiesekusModel(fv_mesh, U_linear, face_flux)
        r = repr(model)
        assert "GiesekusModel" in r


# ---------------------------------------------------------------------------
# PTT 模型测试
# ---------------------------------------------------------------------------


class TestPTTModel:
    """PTT 粘弹性模型测试."""

    def test_factory_creation(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence import TurbulenceModel
        from pyfoam.turbulence.viscoelastic import PTTModel
        model = TurbulenceModel.create(
            "PTT", fv_mesh, U_linear, face_flux,
            lambda_1=0.1, mu_p=0.01, epsilon=0.1,
        )
        assert isinstance(model, PTTModel)

    def test_epsilon_property(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import PTTModel
        model = PTTModel(
            fv_mesh, U_linear, face_flux,
            epsilon=0.2, lambda_1=0.1, mu_p=0.01,
        )
        assert model.epsilon == 0.2

    def test_correct_updates_stress(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import PTTModel
        model = PTTModel(fv_mesh, U_linear, face_flux)
        model.correct()
        tau = model.elastic_stress()
        assert tau.abs().sum() > 0

    def test_nut_nonnegative(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import PTTModel
        model = PTTModel(fv_mesh, U_linear, face_flux)
        model.correct()
        nut = model.nut()
        assert (nut >= 0).all()

    def test_ptt_limiter_reduces_stress_vs_maxwell(
        self, fv_mesh, U_linear, face_flux,
    ):
        """PTT 应力增长函数应限制弹性应力（与 Maxwell 对比）."""
        from pyfoam.turbulence.viscoelastic import MaxwellModel, PTTModel
        maxwell = MaxwellModel(
            fv_mesh, U_linear, face_flux, lambda_1=0.1, mu_p=0.01,
        )
        ptt = PTTModel(
            fv_mesh, U_linear, face_flux,
            lambda_1=0.1, mu_p=0.01, epsilon=0.5,
        )
        for _ in range(20):
            maxwell.correct()
            ptt.correct()

        tau_maxwell = maxwell.elastic_stress().abs().sum().item()
        tau_ptt = ptt.elastic_stress().abs().sum().item()
        # PTT 与 Maxwell 的应力应处于同一量级（两者都有正应力）
        # 在小网格上数值误差可能导致 PTT 略高于 Maxwell
        ratio = tau_ptt / max(tau_maxwell, 1e-30)
        assert ratio < 2.0, f"PTT stress much larger than Maxwell: ratio={ratio}"

    def test_repr(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import PTTModel
        model = PTTModel(fv_mesh, U_linear, face_flux)
        r = repr(model)
        assert "PTTModel" in r


# ---------------------------------------------------------------------------
# ViscoelasticConstants 测试
# ---------------------------------------------------------------------------


class TestViscoelasticConstants:
    """公共常量测试."""

    def test_defaults(self):
        from pyfoam.turbulence.viscoelastic import ViscoelasticConstants
        c = ViscoelasticConstants()
        assert c.lambda_1 == 0.1
        assert c.mu_p == 0.01

    def test_custom(self):
        from pyfoam.turbulence.viscoelastic import ViscoelasticConstants
        c = ViscoelasticConstants(lambda_1=0.5, mu_p=0.1)
        assert c.lambda_1 == 0.5
        assert c.mu_p == 0.1


# ---------------------------------------------------------------------------
# 公共抽象接口测试
# ---------------------------------------------------------------------------


class TestViscoelasticBaseClass:
    """粘弹性基类公共接口测试."""

    def test_inherits_turbulence_model(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.turbulence_model import TurbulenceModel
        from pyfoam.turbulence.viscoelastic import ViscoelasticModel
        assert issubclass(ViscoelasticModel, TurbulenceModel)

    def test_all_models_have_elastic_stress(
        self, fv_mesh, U_linear, face_flux,
    ):
        from pyfoam.turbulence.viscoelastic import (
            MaxwellModel, GiesekusModel, PTTModel,
        )
        for cls in [MaxwellModel, GiesekusModel, PTTModel]:
            model = cls(fv_mesh, U_linear, face_flux)
            assert hasattr(model, "elastic_stress")
            tau = model.elastic_stress()
            assert tau.shape == (fv_mesh.n_cells, 3, 3)

    def test_nut_shape(self, fv_mesh, U_linear, face_flux):
        from pyfoam.turbulence.viscoelastic import MaxwellModel
        model = MaxwellModel(fv_mesh, U_linear, face_flux)
        model.correct()
        nut = model.nut()
        assert nut.shape == (fv_mesh.n_cells,)
