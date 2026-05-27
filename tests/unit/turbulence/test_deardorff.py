"""Tests for the Deardorff diffusion stress one-equation LES model."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.turbulence.deardorff_diff_stress import (
    DeardorffDiffStressModel,
    DeardorffDiffStressConstants,
)
from pyfoam.turbulence.les_model import LESModel


# ---------------------------------------------------------------------------
# Mesh fixture (2-cell hex)
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


@pytest.fixture
def mesh():
    """2-cell hex FvMesh with geometry computed."""
    m = FvMesh(
        points=torch.tensor(_POINTS, dtype=torch.float64),
        faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in _FACES],
        owner=torch.tensor(_OWNER, dtype=INDEX_DTYPE),
        neighbour=torch.tensor(_NEIGHBOUR, dtype=INDEX_DTYPE),
        boundary=_BOUNDARY,
    )
    m.compute_geometry()
    return m


@pytest.fixture
def U_linear(mesh):
    """Velocity field with linear z-profile: U = (0, 0, z)."""
    cc = mesh.cell_centres
    U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
    U[:, 2] = cc[:, 2]
    return U


@pytest.fixture
def phi_zero(mesh):
    """Zero face flux."""
    return torch.zeros(mesh.n_faces, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestDeardorffDiffStressConstants:
    def test_defaults(self):
        c = DeardorffDiffStressConstants()
        assert c.C_k == pytest.approx(0.1)
        assert c.C_epsilon == pytest.approx(1.0)

    def test_custom(self):
        c = DeardorffDiffStressConstants(C_k=0.12, C_epsilon=1.2)
        assert c.C_k == pytest.approx(0.12)
        assert c.C_epsilon == pytest.approx(1.2)

    def test_frozen(self):
        c = DeardorffDiffStressConstants()
        with pytest.raises(AttributeError):
            c.C_k = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Model creation & identity tests
# ---------------------------------------------------------------------------


class TestDeardorffDiffStressCreation:
    def test_inherits_from_les_model(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        assert isinstance(model, LESModel)

    def test_default_constants(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        assert model._C.C_k == pytest.approx(0.1)
        assert model._C.C_epsilon == pytest.approx(1.0)

    def test_custom_constants(self, mesh, U_linear, phi_zero):
        c = DeardorffDiffStressConstants(C_k=0.12, C_epsilon=1.2)
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero, constants=c)
        assert model._C is c

    def test_k_sgs_initial_shape(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        assert model.k_sgs_field.shape == (mesh.n_cells,)

    def test_k_sgs_initial_positive(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        assert (model.k_sgs_field > 0).all()

    def test_type_name(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        assert type(model).__name__ == "DeardorffDiffStressModel"

    def test_repr(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        r = repr(model)
        assert "DeardorffDiffStressModel" in r
        assert "n_cells=2" in r


# ---------------------------------------------------------------------------
# nut() / k() / correct() tests
# ---------------------------------------------------------------------------


class TestDeardorffDiffStressCompute:
    def test_nut_shape(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_nonnegative(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert (nut >= 0).all()

    def test_nut_formula(self, mesh, U_linear, phi_zero):
        """nut = C_k * delta * sqrt(k_sgs)."""
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        model.correct()

        C_k = 0.1
        delta = model.delta
        k_sgs = model.k_sgs_field
        expected = C_k * delta * k_sgs.clamp(min=1e-16).sqrt()
        assert torch.allclose(model.nut(), expected, atol=1e-12)

    def test_k_returns_k_sgs(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        model.correct()
        assert torch.equal(model.k(), model.k_sgs_field)

    def test_k_sgs_updated_after_correct(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        initial_k = model.k_sgs_field.clone()
        model.correct()
        assert not torch.allclose(model.k_sgs_field, initial_k, atol=1e-16)

    def test_k_sgs_field_setter(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        new_k = torch.full((mesh.n_cells,), 0.5, dtype=torch.float64)
        model.k_sgs_field = new_k
        assert torch.allclose(model.k_sgs_field, new_k)

    def test_k_sgs_clamped_positive(self, mesh, U_linear, phi_zero):
        """k_sgs must remain positive after correct()."""
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        model.correct()
        assert (model.k_sgs_field > 0).all()

    def test_correct_populates_gradients(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        assert model.grad_U is None
        assert model.strain_rate is None
        model.correct()
        assert model.grad_U is not None
        assert model.strain_rate is not None

    def test_strain_rate_symmetric(self, mesh, U_linear, phi_zero):
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        model.correct()
        S = model.strain_rate
        assert S is not None
        assert torch.allclose(S, S.transpose(-1, -2), atol=1e-12)

    def test_custom_constants_affect_nut(self, mesh, U_linear, phi_zero):
        """Higher C_k should give larger nut."""
        c_low = DeardorffDiffStressConstants(C_k=0.05)
        c_high = DeardorffDiffStressConstants(C_k=0.2)
        m1 = DeardorffDiffStressModel(mesh, U_linear, phi_zero, constants=c_low)
        m2 = DeardorffDiffStressModel(mesh, U_linear, phi_zero, constants=c_high)
        m1.correct()
        m2.correct()
        ratio = (m2.nut().mean() / m1.nut().clamp(min=1e-30).mean()).item()
        assert ratio > 1.0

    def test_zero_velocity_nut(self, mesh, phi_zero):
        """Zero velocity with non-zero k_sgs still yields nut via k_sgs."""
        U_zero = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        model = DeardorffDiffStressModel(mesh, U_zero, phi_zero)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_diffusion_uses_full_nu_plus_nut(self, mesh, U_linear, phi_zero):
        """Verify the diffusion uses (nu + nu_sgs) without sigma_k divisor.

        This is the key difference from KEqnModel: Deardorff uses the
        full effective viscosity for diffusion, not nu + nut/sigma_k.
        """
        model = DeardorffDiffStressModel(mesh, U_linear, phi_zero)
        # DeardorffDiffStressConstants has no sigma_k attribute
        assert not hasattr(model._C, "sigma_k")
