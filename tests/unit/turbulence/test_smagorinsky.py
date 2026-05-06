"""Tests for the Smagorinsky SGS model."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.turbulence.smagorinsky import SmagorinskyModel, DEFAULT_CS
from pyfoam.turbulence.filter_width import compute_filter_width
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
# Filter width tests
# ---------------------------------------------------------------------------


class TestFilterWidth:
    def test_filter_width_shape(self, mesh):
        delta = compute_filter_width(mesh)
        assert delta.shape == (mesh.n_cells,)

    def test_filter_width_unit_cube(self, mesh):
        """For unit cube cells, delta should be 1.0."""
        delta = compute_filter_width(mesh)
        assert torch.allclose(delta, torch.ones(2, dtype=torch.float64), atol=1e-10)

    def test_filter_width_positive(self, mesh):
        delta = compute_filter_width(mesh)
        assert (delta > 0).all()

    def test_filter_width_is_cube_root_of_volume(self, mesh):
        delta = compute_filter_width(mesh)
        expected = mesh.cell_volumes.pow(1.0 / 3.0)
        assert torch.allclose(delta, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Smagorinsky model tests
# ---------------------------------------------------------------------------


class TestSmagorinskyModel:
    def test_inherits_from_les_model(self, mesh, U_linear, phi_zero):
        model = SmagorinskyModel(mesh, U_linear, phi_zero)
        assert isinstance(model, LESModel)

    def test_default_cs(self, mesh, U_linear, phi_zero):
        model = SmagorinskyModel(mesh, U_linear, phi_zero)
        assert model.Cs == DEFAULT_CS
        assert model.Cs == pytest.approx(0.17)

    def test_custom_cs(self, mesh, U_linear, phi_zero):
        model = SmagorinskyModel(mesh, U_linear, phi_zero, Cs=0.1)
        assert model.Cs == pytest.approx(0.1)

    def test_cs_setter(self, mesh, U_linear, phi_zero):
        model = SmagorinskyModel(mesh, U_linear, phi_zero)
        model.Cs = 0.2
        assert model.Cs == pytest.approx(0.2)

    def test_nut_without_correct_raises(self, mesh, U_linear, phi_zero):
        model = SmagorinskyModel(mesh, U_linear, phi_zero)
        with pytest.raises(RuntimeError, match="correct\\(\\) must be called"):
            model.nut()

    def test_nut_shape(self, mesh, U_linear, phi_zero):
        model = SmagorinskyModel(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_nonnegative(self, mesh, U_linear, phi_zero):
        """SGS viscosity must be non-negative."""
        model = SmagorinskyModel(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert (nut >= 0).all()

    def test_nut_zero_for_zero_velocity(self, mesh, phi_zero):
        """Zero velocity → zero strain rate → zero SGS viscosity."""
        U_zero = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        model = SmagorinskyModel(mesh, U_zero, phi_zero)
        model.correct()
        nut = model.nut()
        assert torch.allclose(nut, torch.zeros(2, dtype=torch.float64), atol=1e-15)

    def test_nut_formula(self, mesh, U_linear, phi_zero):
        """Verify nut = (Cs * delta)^2 * |S|."""
        Cs = 0.17
        model = SmagorinskyModel(mesh, U_linear, phi_zero, Cs=Cs)
        model.correct()

        delta = model.delta
        mag_S = model.mag_strain_rate

        expected = (Cs * delta).pow(2) * mag_S
        nut = model.nut()
        assert torch.allclose(nut, expected, atol=1e-12)

    def test_strain_rate_tensor_symmetric(self, mesh, U_linear, phi_zero):
        """Strain rate tensor must be symmetric: S_ij = S_ji."""
        model = SmagorinskyModel(mesh, U_linear, phi_zero)
        model.correct()
        S = model.strain_rate
        assert S is not None
        assert torch.allclose(S, S.transpose(-1, -2), atol=1e-12)

    def test_strain_rate_magnitude_positive(self, mesh, U_linear, phi_zero):
        model = SmagorinskyModel(mesh, U_linear, phi_zero)
        model.correct()
        mag_S = model.mag_strain_rate
        assert mag_S is not None
        assert (mag_S >= 0).all()

    def test_velocity_gradient_shape(self, mesh, U_linear, phi_zero):
        model = SmagorinskyModel(mesh, U_linear, phi_zero)
        model.correct()
        grad_U = model.grad_U
        assert grad_U is not None
        assert grad_U.shape == (mesh.n_cells, 3, 3)

    def test_correct_updates_tensors(self, mesh, U_linear, phi_zero):
        model = SmagorinskyModel(mesh, U_linear, phi_zero)
        assert model.grad_U is None
        assert model.strain_rate is None
        model.correct()
        assert model.grad_U is not None
        assert model.strain_rate is not None

    def test_repr(self, mesh, U_linear, phi_zero):
        model = SmagorinskyModel(mesh, U_linear, phi_zero)
        r = repr(model)
        assert "SmagorinskyModel" in r
        assert "n_cells=2" in r

    def test_nut_scales_with_cs(self, mesh, U_linear, phi_zero):
        """nut should scale as Cs^2."""
        model1 = SmagorinskyModel(mesh, U_linear, phi_zero, Cs=0.1)
        model2 = SmagorinskyModel(mesh, U_linear, phi_zero, Cs=0.2)
        model1.correct()
        model2.correct()
        ratio = (model2.nut() / model1.nut().clamp(min=1e-30)).mean()
        assert torch.allclose(ratio, torch.tensor(4.0, dtype=torch.float64), atol=0.1)

    def test_constant_velocity_zero_nut(self, mesh, phi_zero):
        """Uniform velocity → zero strain → zero nut."""
        U_const = torch.ones(mesh.n_cells, 3, dtype=torch.float64)
        model = SmagorinskyModel(mesh, U_const, phi_zero)
        model.correct()
        nut = model.nut()
        assert torch.allclose(nut, torch.zeros(2, dtype=torch.float64), atol=1e-12)
