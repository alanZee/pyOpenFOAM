"""Tests for Phase 7 fvModels: ActuationDiskModel and HeatExchangerModel."""

import pytest
import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel
from pyfoam.fv.actuation_disk import ActuationDiskModel
from pyfoam.fv.heat_exchanger import HeatExchangerModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_matrix(n_cells: int = 5) -> FvMatrix:
    """Create a minimal FvMatrix for testing (no internal faces)."""
    owner = torch.tensor([], dtype=torch.long)
    neighbour = torch.tensor([], dtype=torch.long)
    return FvMatrix(n_cells, owner, neighbour)


# ---------------------------------------------------------------------------
# RTS registry
# ---------------------------------------------------------------------------


class TestPhase7Registry:
    """Test that the new models are registered in the RTS registry."""

    def test_actuation_disk_registered(self):
        """ActuationDiskModel is registered as 'actuationDisk'."""
        assert "actuationDisk" in FvModel.available_types()

    def test_heat_exchanger_registered(self):
        """HeatExchangerModel is registered as 'heatExchanger'."""
        assert "heatExchanger" in FvModel.available_types()

    def test_create_actuation_disk_via_factory(self):
        """ActuationDiskModel created via factory."""
        m = FvModel.create("actuationDisk", Ct=0.8, A_disk=50.0, U_inf=10.0, V_disk=10.0)
        assert isinstance(m, ActuationDiskModel)

    def test_create_heat_exchanger_via_factory(self):
        """HeatExchangerModel created via factory."""
        m = FvModel.create("heatExchanger", h_vol=5000.0, T_coolant=300.0)
        assert isinstance(m, HeatExchangerModel)


# ---------------------------------------------------------------------------
# ActuationDiskModel
# ---------------------------------------------------------------------------


class TestActuationDiskModel:
    """Test ActuationDiskModel (wind turbine thrust source)."""

    def test_basic_thrust_force(self):
        """Thrust force computed correctly from parameters."""
        model = ActuationDiskModel(
            Ct=0.8, rho=1.225, A_disk=50.0, U_inf=10.0, V_disk=10.0,
        )
        # F = 0.5 * 0.8 * 1.225 * 50.0 * 100.0 = 2450.0
        assert abs(model.thrust_force - 2450.0) < 1e-6

    def test_volumetric_force(self):
        """Volumetric force = total thrust / disk volume."""
        model = ActuationDiskModel(
            Ct=0.8, rho=1.225, A_disk=50.0, U_inf=10.0, V_disk=10.0,
        )
        # F_vol = 2450.0 / 10.0 = 245.0
        assert abs(model.volumetric_force - 245.0) < 1e-6

    def test_apply_uniform_distribution(self):
        """Force distributed uniformly across disk cells."""
        model = ActuationDiskModel(
            Ct=0.8, rho=1.225, A_disk=50.0, U_inf=10.0, V_disk=10.0,
            cells=[0, 1, 2, 3, 4],
        )
        matrix = _make_matrix(5)
        field = torch.ones(5, dtype=torch.float64) * 10.0
        model.apply(matrix, field)

        # Total thrust = 2450 N
        # Per cell = -2450 / 5 / 10.0 = -49 N/m^3 each
        # Wait, the model computes F_vol = thrust/V_disk = 245, then f_cell = -245/5 = -49
        expected_su = torch.full((5,), -49.0, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su, atol=1e-10)

    def test_apply_partial_cells(self):
        """Force applied only to specified cells."""
        model = ActuationDiskModel(
            Ct=0.8, rho=1.225, A_disk=50.0, U_inf=10.0, V_disk=10.0,
            cells=[1, 3],
        )
        matrix = _make_matrix(5)
        field = torch.ones(5, dtype=torch.float64) * 10.0
        model.apply(matrix, field)

        # f_cell = -245 / 2 = -122.5 per disk cell
        expected = torch.tensor([0.0, -122.5, 0.0, -122.5, 0.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected, atol=1e-10)

    def test_apply_all_cells_when_none_specified(self):
        """cells=None distributes across all cells."""
        model = ActuationDiskModel(
            Ct=0.8, rho=1.225, A_disk=50.0, U_inf=10.0, V_disk=10.0,
            cells=None,
        )
        matrix = _make_matrix(5)
        field = torch.ones(5, dtype=torch.float64) * 10.0
        model.apply(matrix, field)

        # f_cell = -245 / 5 = -49
        expected = torch.full((5,), -49.0, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected, atol=1e-10)

    def test_implicit_sp_contribution(self):
        """Sp contribution added to diagonal when specified."""
        model = ActuationDiskModel(
            Ct=0.8, rho=1.225, A_disk=50.0, U_inf=10.0, V_disk=10.0,
            cells=[0, 1], Sp=-0.5,
        )
        matrix = _make_matrix(5)
        field = torch.ones(5, dtype=torch.float64) * 10.0
        model.apply(matrix, field)

        expected_sp = torch.tensor([-0.5, -0.5, 0.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-10)

    def test_negative_ct_raises(self):
        """Negative Ct raises ValueError."""
        with pytest.raises(ValueError, match="Ct"):
            ActuationDiskModel(Ct=-0.5)

    def test_negative_rho_raises(self):
        """Non-positive rho raises ValueError."""
        with pytest.raises(ValueError, match="rho"):
            ActuationDiskModel(rho=0.0)
        with pytest.raises(ValueError, match="rho"):
            ActuationDiskModel(rho=-1.0)

    def test_negative_V_disk_raises(self):
        """Non-positive V_disk raises ValueError."""
        with pytest.raises(ValueError, match="V_disk"):
            ActuationDiskModel(V_disk=0.0)

    def test_inactive_does_nothing(self):
        """Inactive model does not modify matrix."""
        model = ActuationDiskModel(
            Ct=0.8, rho=1.225, A_disk=50.0, U_inf=10.0, V_disk=10.0,
        )
        model.active = False
        matrix = _make_matrix(5)
        orig_src = matrix.source.clone()
        orig_diag = matrix.diag.clone()
        model.apply(matrix, torch.ones(5, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig_src)
        assert torch.allclose(matrix.diag, orig_diag)

    def test_properties(self):
        """Properties return correct values."""
        model = ActuationDiskModel(
            Ct=0.9, rho=1.0, A_disk=100.0, U_inf=12.0, V_disk=20.0,
        )
        assert model.Ct == 0.9
        assert model.rho == 1.0
        assert model.A_disk == 100.0
        assert model.U_inf == 12.0
        assert model.V_disk == 20.0

    def test_repr(self):
        """repr includes class and key parameters."""
        model = ActuationDiskModel(
            Ct=0.8, rho=1.225, A_disk=50.0, U_inf=10.0, V_disk=10.0,
        )
        r = repr(model)
        assert "ActuationDiskModel" in r
        assert "Ct=0.8" in r

    def test_type_name(self):
        """type_name returns 'actuationDisk'."""
        model = ActuationDiskModel()
        assert model.type_name == "actuationDisk"


# ---------------------------------------------------------------------------
# HeatExchangerModel
# ---------------------------------------------------------------------------


class TestHeatExchangerModel:
    """Test HeatExchangerModel (temperature-dependent heat exchange)."""

    def test_cooling_source(self):
        """Cooling source (h_vol > 0) drives temperature down."""
        model = HeatExchangerModel(h_vol=5000.0, T_coolant=300.0, cells=[0, 1, 2])
        matrix = _make_matrix(3)
        T = torch.tensor([350.0, 400.0, 450.0], dtype=torch.float64)
        model.apply(matrix, T)

        # Su = h_vol * T_coolant = 5000 * 300 = 1.5e6
        expected_su = torch.full((3,), 5000.0 * 300.0, dtype=torch.float64)
        # Sp = -h_vol = -5000
        expected_sp = torch.full((3,), -5000.0, dtype=torch.float64)

        assert torch.allclose(matrix.source, expected_su, atol=1e-6)
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-6)

    def test_heating_source(self):
        """Heating source (h_vol < 0, T_coolant > T_fluid) works."""
        model = HeatExchangerModel(h_vol=-3000.0, T_coolant=500.0, cells=[0, 1])
        matrix = _make_matrix(2)
        T = torch.tensor([300.0, 350.0], dtype=torch.float64)
        model.apply(matrix, T)

        # Su = -3000 * 500 = -1.5e6
        expected_su = torch.full((2,), -3000.0 * 500.0, dtype=torch.float64)
        # Sp = 3000
        expected_sp = torch.full((2,), 3000.0, dtype=torch.float64)

        assert torch.allclose(matrix.source, expected_su, atol=1e-6)
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-6)

    def test_all_cells_when_none_specified(self):
        """cells=None applies to all cells."""
        model = HeatExchangerModel(h_vol=1000.0, T_coolant=293.15)
        matrix = _make_matrix(4)
        T = torch.ones(4, dtype=torch.float64) * 350.0
        model.apply(matrix, T)

        expected_su = torch.full((4,), 1000.0 * 293.15, dtype=torch.float64)
        expected_sp = torch.full((4,), -1000.0, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su, atol=1e-6)
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-6)

    def test_cell_restriction(self):
        """Source restricted to specified cells."""
        model = HeatExchangerModel(
            h_vol=2000.0, T_coolant=300.0, cells=[1, 3],
        )
        matrix = _make_matrix(5)
        T = torch.ones(5, dtype=torch.float64) * 350.0
        model.apply(matrix, T)

        expected_su = torch.tensor(
            [0.0, 2000.0 * 300.0, 0.0, 2000.0 * 300.0, 0.0],
            dtype=torch.float64,
        )
        expected_sp = torch.tensor(
            [0.0, -2000.0, 0.0, -2000.0, 0.0],
            dtype=torch.float64,
        )
        assert torch.allclose(matrix.source, expected_su, atol=1e-6)
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-6)

    def test_zero_h_vol_no_effect(self):
        """h_vol=0 produces no source contribution."""
        model = HeatExchangerModel(h_vol=0.0, T_coolant=300.0)
        matrix = _make_matrix(3)
        orig_src = matrix.source.clone()
        orig_diag = matrix.diag.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64) * 350.0)
        assert torch.allclose(matrix.source, orig_src)
        assert torch.allclose(matrix.diag, orig_diag)

    def test_inactive_does_nothing(self):
        """Inactive model does not modify matrix."""
        model = HeatExchangerModel(h_vol=5000.0, T_coolant=300.0)
        model.active = False
        matrix = _make_matrix(3)
        orig_src = matrix.source.clone()
        orig_diag = matrix.diag.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64) * 350.0)
        assert torch.allclose(matrix.source, orig_src)
        assert torch.allclose(matrix.diag, orig_diag)

    def test_properties(self):
        """Properties return correct values."""
        model = HeatExchangerModel(h_vol=1234.0, T_coolant=293.15)
        assert model.h_vol == 1234.0
        assert model.T_coolant == 293.15

    def test_repr(self):
        """repr includes class and key parameters."""
        model = HeatExchangerModel(h_vol=5000.0, T_coolant=300.0)
        r = repr(model)
        assert "HeatExchangerModel" in r
        assert "h_vol=5000.0" in r

    def test_type_name(self):
        """type_name returns 'heatExchanger'."""
        model = HeatExchangerModel()
        assert model.type_name == "heatExchanger"

    def test_default_T_coolant(self):
        """Default T_coolant is 300.0 K."""
        model = HeatExchangerModel(h_vol=1000.0)
        assert model.T_coolant == 300.0
