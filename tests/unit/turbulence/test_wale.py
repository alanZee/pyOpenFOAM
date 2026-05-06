"""Tests for the WALE SGS model."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.turbulence.wale import WALEModel, DEFAULT_CW
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
# WALE model tests
# ---------------------------------------------------------------------------


class TestWALEModel:
    def test_inherits_from_les_model(self, mesh, U_linear, phi_zero):
        model = WALEModel(mesh, U_linear, phi_zero)
        assert isinstance(model, LESModel)

    def test_default_cw(self, mesh, U_linear, phi_zero):
        model = WALEModel(mesh, U_linear, phi_zero)
        assert model.Cw == DEFAULT_CW
        assert model.Cw == pytest.approx(0.325)

    def test_custom_cw(self, mesh, U_linear, phi_zero):
        model = WALEModel(mesh, U_linear, phi_zero, Cw=0.5)
        assert model.Cw == pytest.approx(0.5)

    def test_cw_setter(self, mesh, U_linear, phi_zero):
        model = WALEModel(mesh, U_linear, phi_zero)
        model.Cw = 0.4
        assert model.Cw == pytest.approx(0.4)

    def test_nut_without_correct_raises(self, mesh, U_linear, phi_zero):
        model = WALEModel(mesh, U_linear, phi_zero)
        with pytest.raises(RuntimeError, match="correct\\(\\) must be called"):
            model.nut()

    def test_nut_shape(self, mesh, U_linear, phi_zero):
        model = WALEModel(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_nonnegative(self, mesh, U_linear, phi_zero):
        """SGS viscosity must be non-negative."""
        model = WALEModel(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert (nut >= 0).all()

    def test_nut_finite(self, mesh, U_linear, phi_zero):
        """SGS viscosity must be finite (no NaN or Inf)."""
        model = WALEModel(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert torch.isfinite(nut).all()

    def test_nut_zero_for_zero_velocity(self, mesh, phi_zero):
        """Zero velocity → zero strain rate → zero SGS viscosity."""
        U_zero = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        model = WALEModel(mesh, U_zero, phi_zero)
        model.correct()
        nut = model.nut()
        assert torch.allclose(nut, torch.zeros(2, dtype=torch.float64), atol=1e-15)

    def test_sd_tensor_shape(self, mesh, U_linear, phi_zero):
        model = WALEModel(mesh, U_linear, phi_zero)
        model.correct()
        Sd = model.Sd
        assert Sd is not None
        assert Sd.shape == (mesh.n_cells, 3, 3)

    def test_sd_tensor_symmetric(self, mesh, U_linear, phi_zero):
        """Sd tensor must be symmetric: Sd_ij = Sd_ji."""
        model = WALEModel(mesh, U_linear, phi_zero)
        model.correct()
        Sd = model.Sd
        assert Sd is not None
        assert torch.allclose(Sd, Sd.transpose(-1, -2), atol=1e-12)

    def test_sd_tensor_traceless(self, mesh, U_linear, phi_zero):
        """Sd tensor must be traceless: Sd_kk = 0."""
        model = WALEModel(mesh, U_linear, phi_zero)
        model.correct()
        Sd = model.Sd
        assert Sd is not None
        trace = Sd.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        assert torch.allclose(trace, torch.zeros(2, dtype=torch.float64), atol=1e-12)

    def test_mag_Sd_sq_nonnegative(self, mesh, U_linear, phi_zero):
        """Sd_ij * Sd_ij must be non-negative."""
        model = WALEModel(mesh, U_linear, phi_zero)
        model.correct()
        mag_Sd_sq = model.mag_Sd_sq
        assert mag_Sd_sq is not None
        assert (mag_Sd_sq >= 0).all()

    def test_correct_updates_tensors(self, mesh, U_linear, phi_zero):
        model = WALEModel(mesh, U_linear, phi_zero)
        assert model.Sd is None
        assert model.mag_Sd_sq is None
        model.correct()
        assert model.Sd is not None
        assert model.mag_Sd_sq is not None
        assert model.grad_U is not None
        assert model.strain_rate is not None

    def test_repr(self, mesh, U_linear, phi_zero):
        model = WALEModel(mesh, U_linear, phi_zero)
        r = repr(model)
        assert "WALEModel" in r
        assert "n_cells=2" in r

    def test_constant_velocity_zero_nut(self, mesh, phi_zero):
        """Uniform velocity → zero strain → zero nut."""
        U_const = torch.ones(mesh.n_cells, 3, dtype=torch.float64)
        model = WALEModel(mesh, U_const, phi_zero)
        model.correct()
        nut = model.nut()
        assert torch.allclose(nut, torch.zeros(2, dtype=torch.float64), atol=1e-12)

    def test_wale_vs_smagorinsky_nonzero(self, mesh, U_linear, phi_zero):
        """Both models should give non-zero nut for non-uniform flow."""
        from pyfoam.turbulence.smagorinsky import SmagorinskyModel

        smag = SmagorinskyModel(mesh, U_linear, phi_zero)
        wale = WALEModel(mesh, U_linear, phi_zero)
        smag.correct()
        wale.correct()
        assert smag.nut().abs().sum() > 0
        assert wale.nut().abs().sum() > 0

    def test_wale_nut_formula(self, mesh, U_linear, phi_zero):
        """Verify the WALE formula manually."""
        Cw = 0.325
        model = WALEModel(mesh, U_linear, phi_zero, Cw=Cw)
        model.correct()
        nut = model.nut()

        delta = model.delta
        mag_S = model.mag_strain_rate
        mag_Sd_sq = model.mag_Sd_sq

        coeff = (Cw * delta).pow(2)
        numerator = mag_Sd_sq.pow(1.5)
        S_ij_S_ij = (mag_S.pow(2) / 2.0).clamp(min=1e-30)
        denominator = S_ij_S_ij.pow(2.5) + mag_Sd_sq.pow(1.25) + 1e-30
        expected = coeff * numerator / denominator

        assert torch.allclose(nut, expected, atol=1e-10)
