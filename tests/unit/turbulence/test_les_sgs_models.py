"""Tests for SGS (Subgrid-Scale) turbulence models.

Tests cover:
- SGSModel abstract base class
- DynamicSmagorinskySGS model
- WALE_SGS model
- compute_eddy_viscosity interface
- Model constants and properties
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.turbulence.les_model import LESModel
from pyfoam.turbulence.les_sgs_models import (
    SGSModel,
    DynamicSmagorinskySGS,
    WALE_SGS,
    _DEFAULT_CS_MIN,
    _DEFAULT_CS_MAX,
    _DEFAULT_CW,
)


# ---------------------------------------------------------------------------
# Shared mesh fixture
# ---------------------------------------------------------------------------

_POINTS = [
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0],
]

_FACES = [
    [4, 5, 6, 7],
    [0, 3, 2, 1], [0, 1, 5, 4], [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5],
    [8, 9, 10, 11], [4, 5, 9, 8], [7, 11, 10, 6], [4, 8, 11, 7], [5, 6, 10, 9],
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


# ===========================================================================
# SGSModel abstract base tests
# ===========================================================================


class TestSGSModelAbstractBase:
    """Tests for the SGSModel abstract base class."""

    def test_cannot_instantiate_directly(self, mesh, U_linear, phi_zero):
        """SGSModel is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            SGSModel(mesh, U_linear, phi_zero)

    def test_inherits_from_les_model(self):
        """SGSModel inherits from LESModel."""
        assert issubclass(SGSModel, LESModel)

    def test_has_compute_eddy_viscosity_method(self):
        """SGSModel defines compute_eddy_viscosity()."""
        assert hasattr(SGSModel, "compute_eddy_viscosity")

    def test_subclass_must_implement_nut(self, mesh, U_linear, phi_zero):
        """Subclass must implement nut()."""
        class IncompleteSGS(SGSModel):
            def correct(self):
                pass

        with pytest.raises(TypeError):
            IncompleteSGS(mesh, U_linear, phi_zero)

    def test_subclass_must_implement_correct(self, mesh, U_linear, phi_zero):
        """Subclass must implement correct()."""
        class IncompleteSGS(SGSModel):
            def nut(self):
                pass

        with pytest.raises(TypeError):
            IncompleteSGS(mesh, U_linear, phi_zero)


# ===========================================================================
# DynamicSmagorinskySGS tests
# ===========================================================================


class TestDynamicSmagorinskySGS:
    """Tests for the DynamicSmagorinskySGS model."""

    def test_inherits_from_sgs_model(self, mesh, U_linear, phi_zero):
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        assert isinstance(model, SGSModel)
        assert isinstance(model, LESModel)

    def test_default_constants(self, mesh, U_linear, phi_zero):
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        assert model.Cs_min == pytest.approx(0.0)
        assert model.Cs_max == pytest.approx(0.5)

    def test_custom_constants(self, mesh, U_linear, phi_zero):
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero, Cs_min=0.01, Cs_max=0.3)
        assert model.Cs_min == pytest.approx(0.01)
        assert model.Cs_max == pytest.approx(0.3)

    def test_default_module_constants(self):
        assert _DEFAULT_CS_MIN == pytest.approx(0.0)
        assert _DEFAULT_CS_MAX == pytest.approx(0.5)

    def test_Cs_initially_none(self, mesh, U_linear, phi_zero):
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        assert model.Cs is None
        assert model.Cs2 is None

    def test_compute_eddy_viscosity_before_correct_raises(self, mesh, U_linear, phi_zero):
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        with pytest.raises(RuntimeError, match="correct\\(\\) must be called"):
            model.compute_eddy_viscosity()

    def test_nut_without_correct_raises(self, mesh, U_linear, phi_zero):
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        with pytest.raises(RuntimeError, match="correct\\(\\) must be called"):
            model.nut()

    def test_nut_shape(self, mesh, U_linear, phi_zero):
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_nonnegative(self, mesh, U_linear, phi_zero):
        """SGS viscosity must be non-negative."""
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert (nut >= 0).all()

    def test_nut_finite(self, mesh, U_linear, phi_zero):
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert torch.isfinite(nut).all()

    def test_nut_zero_for_zero_velocity(self, mesh, phi_zero):
        U_zero = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        model = DynamicSmagorinskySGS(mesh, U_zero, phi_zero)
        model.correct()
        nut = model.nut()
        assert torch.allclose(nut, torch.zeros(2, dtype=torch.float64), atol=1e-15)

    def test_Cs_after_correct(self, mesh, U_linear, phi_zero):
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        model.correct()
        Cs = model.Cs
        assert Cs is not None
        assert Cs.shape == (mesh.n_cells,)
        assert (Cs >= 0).all()

    def test_Cs2_after_correct(self, mesh, U_linear, phi_zero):
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        model.correct()
        Cs2 = model.Cs2
        assert Cs2 is not None
        assert Cs2.shape == (mesh.n_cells,)
        assert (Cs2 >= 0).all()
        assert (Cs2 <= 0.5).all()

    def test_compute_eddy_viscosity_matches_nut(self, mesh, U_linear, phi_zero):
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        eddy_visc = model.compute_eddy_viscosity()
        assert torch.allclose(nut, eddy_visc)

    def test_repr(self, mesh, U_linear, phi_zero):
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        r = repr(model)
        assert "DynamicSmagorinskySGS" in r
        assert "n_cells=2" in r

    def test_constant_velocity_zero_nut(self, mesh, phi_zero):
        U_const = torch.ones(mesh.n_cells, 3, dtype=torch.float64)
        model = DynamicSmagorinskySGS(mesh, U_const, phi_zero)
        model.correct()
        nut = model.nut()
        assert torch.allclose(nut, torch.zeros(2, dtype=torch.float64), atol=1e-12)

    def test_dynamic_coefficient_formula(self, mesh, U_linear, phi_zero):
        """Verify the dynamic coefficient matches manual computation."""
        model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)
        model.correct()

        # Manual computation
        S = model.strain_rate
        mag_S = model.mag_strain_rate
        delta = model.delta
        delta_hat = 2.0 * delta

        L = (
            delta.pow(2).unsqueeze(-1).unsqueeze(-1) * mag_S.unsqueeze(-1).unsqueeze(-1) * S
            - delta_hat.pow(2).unsqueeze(-1).unsqueeze(-1) * mag_S.unsqueeze(-1).unsqueeze(-1) * S
        )
        M = 2.0 * L
        L_dot_M = (L * M).sum(dim=(-2, -1))
        M_dot_M = (M * M).sum(dim=(-2, -1)).clamp(min=1e-30)
        expected_Cs2 = (L_dot_M / M_dot_M).clamp(min=0.0, max=0.5)

        assert torch.allclose(model.Cs2, expected_Cs2, atol=1e-12)


# ===========================================================================
# WALE_SGS tests
# ===========================================================================


class TestWALE_SGS:
    """Tests for the WALE_SGS model."""

    def test_inherits_from_sgs_model(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        assert isinstance(model, SGSModel)
        assert isinstance(model, LESModel)

    def test_default_Cw(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        assert model.Cw == pytest.approx(_DEFAULT_CW)
        assert model.Cw == pytest.approx(0.325)

    def test_custom_Cw(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero, Cw=0.5)
        assert model.Cw == pytest.approx(0.5)

    def test_Cw_setter(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        model.Cw = 0.4
        assert model.Cw == pytest.approx(0.4)

    def test_Sd_initially_none(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        assert model.Sd is None
        assert model.mag_Sd_sq is None

    def test_compute_eddy_viscosity_before_correct_raises(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        with pytest.raises(RuntimeError, match="correct\\(\\) must be called"):
            model.compute_eddy_viscosity()

    def test_nut_without_correct_raises(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        with pytest.raises(RuntimeError, match="correct\\(\\) must be called"):
            model.nut()

    def test_nut_shape(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_nonnegative(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert (nut >= 0).all()

    def test_nut_finite(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        assert torch.isfinite(nut).all()

    def test_nut_zero_for_zero_velocity(self, mesh, phi_zero):
        U_zero = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        model = WALE_SGS(mesh, U_zero, phi_zero)
        model.correct()
        nut = model.nut()
        assert torch.allclose(nut, torch.zeros(2, dtype=torch.float64), atol=1e-15)

    def test_sd_tensor_after_correct(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        model.correct()
        Sd = model.Sd
        assert Sd is not None
        assert Sd.shape == (mesh.n_cells, 3, 3)

    def test_sd_tensor_symmetric(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        model.correct()
        Sd = model.Sd
        assert torch.allclose(Sd, Sd.transpose(-1, -2), atol=1e-12)

    def test_sd_tensor_traceless(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        model.correct()
        Sd = model.Sd
        trace = Sd.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        assert torch.allclose(trace, torch.zeros(2, dtype=torch.float64), atol=1e-12)

    def test_mag_Sd_sq_nonnegative(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        model.correct()
        mag_Sd_sq = model.mag_Sd_sq
        assert mag_Sd_sq is not None
        assert (mag_Sd_sq >= 0).all()

    def test_compute_eddy_viscosity_matches_nut(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        model.correct()
        nut = model.nut()
        eddy_visc = model.compute_eddy_viscosity()
        assert torch.allclose(nut, eddy_visc)

    def test_repr(self, mesh, U_linear, phi_zero):
        model = WALE_SGS(mesh, U_linear, phi_zero)
        r = repr(model)
        assert "WALE_SGS" in r
        assert "n_cells=2" in r

    def test_constant_velocity_zero_nut(self, mesh, phi_zero):
        U_const = torch.ones(mesh.n_cells, 3, dtype=torch.float64)
        model = WALE_SGS(mesh, U_const, phi_zero)
        model.correct()
        nut = model.nut()
        assert torch.allclose(nut, torch.zeros(2, dtype=torch.float64), atol=1e-12)

    def test_wale_sgs_nut_formula(self, mesh, U_linear, phi_zero):
        """Verify the WALE formula manually."""
        Cw = 0.325
        model = WALE_SGS(mesh, U_linear, phi_zero, Cw=Cw)
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

    def test_wale_sgs_vs_wale_model(self, mesh, U_linear, phi_zero):
        """WALE_SGS should give same results as WALEModel."""
        from pyfoam.turbulence.wale import WALEModel

        wale_model = WALEModel(mesh, U_linear, phi_zero)
        sgs_model = WALE_SGS(mesh, U_linear, phi_zero)

        wale_model.correct()
        sgs_model.correct()

        assert torch.allclose(wale_model.nut(), sgs_model.nut(), atol=1e-12)

    def test_dynamic_sgs_vs_dynamic_smagorinsky(self, mesh, U_linear, phi_zero):
        """DynamicSmagorinskySGS should give same results as DynamicSmagorinskyModel."""
        from pyfoam.turbulence.dynamic_smagorinsky import DynamicSmagorinskyModel

        ds_model = DynamicSmagorinskyModel(mesh, U_linear, phi_zero)
        sgs_model = DynamicSmagorinskySGS(mesh, U_linear, phi_zero)

        ds_model.correct()
        sgs_model.correct()

        assert torch.allclose(ds_model.nut(), sgs_model.nut(), atol=1e-12)

    def test_wale_sgs_natural_wall_scaling(self, mesh, phi_zero):
        """WALE should give non-zero nut for non-uniform flow near wall.

        The WALE model naturally recovers y^3 scaling without damping.
        """
        # Create velocity with strong gradient (simulating near-wall)
        U_grad = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U_grad[0] = torch.tensor([0.0, 0.0, 0.1], dtype=torch.float64)
        U_grad[1] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

        model = WALE_SGS(mesh, U_grad, phi_zero)
        model.correct()
        nut = model.nut()

        # Should be non-zero for non-uniform velocity
        assert nut.abs().sum() > 0
