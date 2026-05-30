"""Tests for v6 enhanced turbulent kinetic energy inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_kinetic_energy_inlet_6 import TurbulentKineticEnergyInlet6BC


class TestTurbulentKineticEnergyInlet6BC:
    """Test the turbulentKineticEnergyInlet6 boundary condition."""

    def test_registration(self):
        assert "turbulentKineticEnergyInlet6" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentKineticEnergyInlet6", simple_patch,
            {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentKineticEnergyInlet6BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentKineticEnergyInlet6BC(simple_patch)
        assert bc.type_name == "turbulentKineticEnergyInlet6"

    def test_default_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet6BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.alpha == pytest.approx(0.8)
        assert bc.beta == pytest.approx(0.05)
        assert bc.Re_t_ref == pytest.approx(100.0)
        assert bc.k_min == pytest.approx(1e-10)
        assert bc.k_max == pytest.approx(100.0)
        assert bc.beta_thermal == pytest.approx(0.0034)
        assert bc.Richardson == pytest.approx(0.0)
        assert bc.C_buoyancy == pytest.approx(0.1)
        assert bc.C_production_limit == pytest.approx(2.0)
        assert bc.C_thermal == pytest.approx(0.1)
        assert bc.gravity_mag == pytest.approx(9.81)
        assert bc.delta_T == pytest.approx(0.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet6BC(simple_patch, {
            "intensity": 0.10, "Cmu": 0.1, "Richardson": 0.5,
            "Cbuoyancy": 0.2, "CproductionLimit": 3.0,
            "Cthermal": 0.2, "gravityMag": 9.81, "deltaT": 50.0,
        })
        assert bc.C_buoyancy == pytest.approx(0.2)
        assert bc.C_production_limit == pytest.approx(3.0)
        assert bc.C_thermal == pytest.approx(0.2)
        assert bc.delta_T == pytest.approx(50.0)

    def test_apply_with_velocity_and_epsilon(self, simple_patch):
        bc = TurbulentKineticEnergyInlet6BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=epsilon)

        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_buoyancy_production(self, simple_patch):
        """Buoyancy production should increase k."""
        bc_no_buoy = TurbulentKineticEnergyInlet6BC(simple_patch, {"Richardson": 0.0})
        bc_buoy = TurbulentKineticEnergyInlet6BC(simple_patch, {
            "Richardson": 0.5, "Cbuoyancy": 0.2,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field_no = torch.zeros(15, dtype=torch.float64)
        field_yes = torch.zeros(15, dtype=torch.float64)

        bc_no_buoy.apply(field_no, velocity=velocity, epsilon=epsilon)
        bc_buoy.apply(field_yes, velocity=velocity, epsilon=epsilon)

        # Buoyancy should add positive contribution
        assert torch.all(field_yes[10:13] >= field_no[10:13])

    def test_apply_with_thermal_fluctuation(self, simple_patch):
        """Thermal fluctuation energy should increase k."""
        bc_base = TurbulentKineticEnergyInlet6BC(simple_patch, {"deltaT": 0.0})
        bc_thermal = TurbulentKineticEnergyInlet6BC(simple_patch, {
            "deltaT": 100.0, "Cthermal": 0.1,
            "betaThermal": 0.0034, "gravityMag": 9.81,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field_base = torch.zeros(15, dtype=torch.float64)
        field_thermal = torch.zeros(15, dtype=torch.float64)

        bc_base.apply(field_base, velocity=velocity, epsilon=epsilon)
        bc_thermal.apply(field_thermal, velocity=velocity, epsilon=epsilon)

        assert torch.all(field_thermal[10:13] >= field_base[10:13])

    def test_apply_without_velocity(self, simple_patch):
        bc = TurbulentKineticEnergyInlet6BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentKineticEnergyInlet6BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, velocity=velocity)
        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentKineticEnergyInlet6BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
