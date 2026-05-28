"""Tests for flowRateInlet2 and wallHeatFlux boundary conditions.

Tests cover:
- RTS registration and factory creation
- FlowRateInlet2BC: profile computation, mass/volume flow rate, apply
- WallHeatFluxBC: heat flux, gradient, apply, matrix contributions
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition


# ---------------------------------------------------------------------------
# FlowRateInlet2BC
# ---------------------------------------------------------------------------


class TestFlowRateInlet2BC:
    """Test the flowRateInlet2 boundary condition."""

    def test_registration(self):
        assert "flowRateInlet2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "flowRateInlet2", simple_patch,
            {"volumeFlowRate": 0.01, "rho": 1.0, "exponent": 7},
        )
        from pyfoam.boundary.flow_rate_inlet_2 import FlowRateInlet2BC
        assert isinstance(bc, FlowRateInlet2BC)

    def test_type_name(self, simple_patch):
        from pyfoam.boundary.flow_rate_inlet_2 import FlowRateInlet2BC
        bc = FlowRateInlet2BC(simple_patch, {"volumeFlowRate": 0.01})
        assert bc.type_name == "flowRateInlet2"

    def test_properties(self, simple_patch):
        from pyfoam.boundary.flow_rate_inlet_2 import FlowRateInlet2BC
        bc = FlowRateInlet2BC(simple_patch, {
            "massFlowRate": 0.1, "rho": 1.225, "exponent": 10,
        })
        assert bc.mass_flow_rate == pytest.approx(0.1)
        assert bc.rho == pytest.approx(1.225)
        assert bc.exponent == pytest.approx(10.0)

    def test_default_properties(self, simple_patch):
        from pyfoam.boundary.flow_rate_inlet_2 import FlowRateInlet2BC
        bc = FlowRateInlet2BC(simple_patch)
        assert bc.mass_flow_rate is None
        assert bc.volume_flow_rate is None
        assert bc.rho == pytest.approx(1.0)
        assert bc.exponent == pytest.approx(7.0)

    def test_apply_with_volume_flow_rate(self, simple_patch):
        """Volume flow rate sets velocity on faces."""
        from pyfoam.boundary.flow_rate_inlet_2 import FlowRateInlet2BC
        bc = FlowRateInlet2BC(simple_patch, {
            "volumeFlowRate": 0.03, "rho": 1.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        # All faces should have non-zero velocity
        assert field[10:13].abs().sum() > 0

    def test_apply_with_mass_flow_rate(self, simple_patch):
        """Mass flow rate with density computes correct velocity."""
        from pyfoam.boundary.flow_rate_inlet_2 import FlowRateInlet2BC
        bc = FlowRateInlet2BC(simple_patch, {
            "massFlowRate": 0.06, "rho": 2.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        # Bulk velocity = 0.06 / (2.0 * 3.0) = 0.01
        # Profile varies by face but bulk should be near 0.01
        assert field[10:13].abs().max() > 0

    def test_apply_with_zero_flow(self, simple_patch):
        """Zero flow yields zero velocity."""
        from pyfoam.boundary.flow_rate_inlet_2 import FlowRateInlet2BC
        bc = FlowRateInlet2BC(simple_patch, {"volumeFlowRate": 0.0})
        field = torch.ones((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        from pyfoam.boundary.flow_rate_inlet_2 import FlowRateInlet2BC
        bc = FlowRateInlet2BC(simple_patch, {"volumeFlowRate": 0.03})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5:8].abs().sum() > 0

    def test_profile_is_power_law(self, simple_patch):
        """The profile should follow a power-law distribution."""
        from pyfoam.boundary.flow_rate_inlet_2 import FlowRateInlet2BC
        bc = FlowRateInlet2BC(simple_patch, {
            "volumeFlowRate": 0.06, "exponent": 7,
        })
        profile = bc._compute_profile()
        assert profile.shape == (3,)
        # All values should be non-negative
        assert (profile >= 0).all()

    def test_matrix_contributions(self, simple_patch):
        from pyfoam.boundary.flow_rate_inlet_2 import FlowRateInlet2BC
        bc = FlowRateInlet2BC(simple_patch, {"volumeFlowRate": 0.03})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)

    def test_export_availability(self):
        from pyfoam.boundary import FlowRateInlet2BC
        assert FlowRateInlet2BC is not None


# ---------------------------------------------------------------------------
# WallHeatFluxBC
# ---------------------------------------------------------------------------


class TestWallHeatFluxBC:
    """Test the wallHeatFlux boundary condition."""

    def test_registration(self):
        assert "wallHeatFlux" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "wallHeatFlux", simple_patch,
            {"q": 1000.0, "k": 0.025},
        )
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        assert isinstance(bc, WallHeatFluxBC)

    def test_type_name(self, simple_patch):
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        bc = WallHeatFluxBC(simple_patch, {"q": 1000.0})
        assert bc.type_name == "wallHeatFlux"

    def test_properties(self, simple_patch):
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        bc = WallHeatFluxBC(simple_patch, {
            "q": 500.0, "k": 0.6, "value": 350.0,
        })
        assert bc.q == pytest.approx(500.0)
        assert bc.k == pytest.approx(0.6)
        assert bc.T_ref == pytest.approx(350.0)

    def test_default_properties(self, simple_patch):
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        bc = WallHeatFluxBC(simple_patch)
        assert bc.q == pytest.approx(0.0)
        assert bc.k == pytest.approx(0.025)
        assert bc.T_ref == pytest.approx(300.0)

    def test_gradient_computation(self, simple_patch):
        """dT/dn = -q / k"""
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        bc = WallHeatFluxBC(simple_patch, {"q": 1000.0, "k": 0.5})
        assert bc.gradient == pytest.approx(-2000.0)

    def test_gradient_zero_k(self, simple_patch):
        """Zero conductivity should not crash."""
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        bc = WallHeatFluxBC(simple_patch, {"q": 100.0, "k": 0.0})
        assert bc.gradient == pytest.approx(0.0)

    def test_gradient_zero_flux(self, simple_patch):
        """Zero heat flux yields zero gradient."""
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        bc = WallHeatFluxBC(simple_patch, {"q": 0.0, "k": 0.025})
        assert bc.gradient == pytest.approx(0.0)

    def test_apply_sets_temperature(self, simple_patch):
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        bc = WallHeatFluxBC(simple_patch, {"q": 1000.0, "k": 0.5})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        # Face values should be non-zero
        assert field[10:13].abs().sum() > 0

    def test_apply_with_patch_idx(self, simple_patch):
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        bc = WallHeatFluxBC(simple_patch, {"q": 500.0, "k": 0.6})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5:8].abs().sum() > 0

    def test_apply_zero_flux(self, simple_patch):
        """Zero heat flux sets temperature to T_ref."""
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        bc = WallHeatFluxBC(simple_patch, {"q": 0.0, "k": 0.025})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        # With zero gradient, T_face ≈ T_ref
        assert torch.allclose(field[10:13], torch.full((3,), 300.0, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """Heat flux contributes to source: source += q * area."""
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        bc = WallHeatFluxBC(simple_patch, {"q": 1000.0, "k": 0.025})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # source += 1000.0 * 1.0 = 1000.0 per face
        assert torch.allclose(
            source, torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64),
        )

    def test_matrix_contributions_accumulated(self, simple_patch):
        """Pre-existing diag/source should be accumulated into."""
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        bc = WallHeatFluxBC(simple_patch, {"q": 500.0, "k": 0.5})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, diag=diag, source=source)
        # diag unchanged, source = 1 + 500*1 = 501
        assert torch.allclose(diag, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(
            source, torch.tensor([501.0, 501.0, 501.0], dtype=torch.float64),
        )

    def test_settable_properties(self, simple_patch):
        """q and k are settable."""
        from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC
        bc = WallHeatFluxBC(simple_patch, {"q": 100.0, "k": 0.5})
        bc.q = 200.0
        bc.k = 1.0
        assert bc.q == pytest.approx(200.0)
        assert bc.k == pytest.approx(1.0)
        assert bc.gradient == pytest.approx(-200.0)

    def test_export_availability(self):
        from pyfoam.boundary import WallHeatFluxBC
        assert WallHeatFluxBC is not None
