"""Tests for enhanced pressure inlet/outlet velocity boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.pressure_inlet_outlet_velocity_2 import PressureInletOutletVelocity2BC


class TestPressureInletOutletVelocity2BC:
    """Test the pressureInletOutletVelocity2 boundary condition."""

    def test_registration(self):
        assert "pressureInletOutletVelocity2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "pressureInletOutletVelocity2", simple_patch,
            {"inletDir": [1, 0, 0]},
        )
        assert isinstance(bc, PressureInletOutletVelocity2BC)

    def test_type_name(self, simple_patch):
        bc = PressureInletOutletVelocity2BC(simple_patch)
        assert bc.type_name == "pressureInletOutletVelocity2"

    def test_default_properties(self, simple_patch):
        bc = PressureInletOutletVelocity2BC(simple_patch)
        assert torch.allclose(bc.inlet_dir, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        assert bc.blending == pytest.approx(0.1)

    def test_custom_properties(self, simple_patch):
        bc = PressureInletOutletVelocity2BC(simple_patch, {
            "inletDir": [0, 1, 0], "blending": 0.5,
        })
        expected_dir = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        assert torch.allclose(bc.inlet_dir, expected_dir)
        assert bc.blending == pytest.approx(0.5)

    def test_inlet_dir_normalized(self, simple_patch):
        """Inlet direction should be normalised."""
        bc = PressureInletOutletVelocity2BC(simple_patch, {
            "inletDir": [3, 4, 0],
        })
        norm = torch.norm(bc.inlet_dir)
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_apply_outflow_zero_gradient(self, simple_patch):
        """Outflow (positive flux) should give approximately zero-gradient (owner values)."""
        bc = PressureInletOutletVelocity2BC(simple_patch, {"blending": 0.01})
        field = torch.zeros((15, 3), dtype=torch.float64)
        field[0] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        field[1] = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        field[2] = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64)

        flux = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        bc.apply(field, flux=flux)

        # Strong outflow -> sigmoid(-10/0.01) ≈ 0 -> zero-gradient
        assert torch.allclose(field[10], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), atol=1e-6)
        assert torch.allclose(field[11], torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64), atol=1e-6)
        assert torch.allclose(field[12], torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64), atol=1e-6)

    def test_apply_inflow_blends_direction(self, simple_patch):
        """Inflow (negative flux) should blend with prescribed direction."""
        bc = PressureInletOutletVelocity2BC(simple_patch, {
            "inletDir": [1, 0, 0], "blending": 0.01,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        field[0] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        field[1] = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        field[2] = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64)

        flux = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
        bc.apply(field, flux=flux)

        # Strong inflow -> velocity should be mostly in inlet direction
        # with magnitude from interior
        # face[10]: owner U = (1,2,3), |U| = sqrt(14) ~ 3.74
        # inlet vel = (|U|, 0, 0)
        assert field[10, 0] > 0.0  # x-component positive
        # y and z should be close to zero (blended toward pure inlet dir)
        assert abs(field[10, 1]) < 2.0  # reduced from 2.0

    def test_apply_no_flux_uses_zero_gradient(self, simple_patch):
        """Without flux, defaults to outflow (zero-gradient)."""
        bc = PressureInletOutletVelocity2BC(simple_patch)
        field = torch.zeros((15, 3), dtype=torch.float64)
        field[0] = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)

        bc.apply(field)

        # alpha = 0 (default outflow) -> pure zero_gradient = owner values
        assert field[10, 0] == pytest.approx(10.0, rel=1e-6)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PressureInletOutletVelocity2BC(simple_patch)
        field = torch.zeros((20, 3), dtype=torch.float64)
        field[0] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        flux = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        bc.apply(field, patch_idx=5, flux=flux)

        assert torch.allclose(field[5], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        bc = PressureInletOutletVelocity2BC(simple_patch)
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)

    def test_matrix_contributions_with_outflow(self, simple_patch):
        """Outflow should contribute less to matrix."""
        bc = PressureInletOutletVelocity2BC(simple_patch)
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        flux = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        diag_out, source_out = bc.matrix_contributions(field, n_cells, flux=flux)

        # Strong outflow -> near-zero contribution
        assert diag_out.abs().max() < 0.1
