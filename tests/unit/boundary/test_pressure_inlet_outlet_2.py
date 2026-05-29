"""Tests for enhanced pressure inlet/outlet boundary condition (version 2).

Tests cover PressureInletOutlet2BC:
- RTS registration
- Factory creation
- Property access (p0, k_field, relaxation)
- Turbulence-aware inlet pressure computation
- apply() with and without TKE correction
- matrix_contributions with flow direction detection
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.pressure_inlet_outlet_2 import PressureInletOutlet2BC


class TestPressureInletOutlet2BC:
    """pressureInletOutlet2 boundary condition tests."""

    def test_registration(self):
        """pressureInletOutlet2 is registered in RTS."""
        assert "pressureInletOutlet2" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = PressureInletOutlet2BC(simple_patch)
        assert bc.type_name == "pressureInletOutlet2"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "pressureInletOutlet2", simple_patch,
            {"p0": 101325.0},
        )
        assert isinstance(bc, PressureInletOutlet2BC)

    def test_default_coefficients(self, simple_patch):
        bc = PressureInletOutlet2BC(simple_patch)
        assert bc.p0 == pytest.approx(0.0)
        assert bc.k_field == "k"
        assert bc.relaxation == pytest.approx(1.0)

    def test_custom_coefficients(self, simple_patch):
        bc = PressureInletOutlet2BC(
            simple_patch,
            {"p0": 101325.0, "k_field": "k.air", "relaxation": 0.7},
        )
        assert bc.p0 == pytest.approx(101325.0)
        assert bc.k_field == "k.air"
        assert bc.relaxation == pytest.approx(0.7)

    def test_compute_inlet_pressure_no_k(self, simple_patch):
        """Without TKE: p_inlet = p0."""
        bc = PressureInletOutlet2BC(simple_patch, {"p0": 101325.0})
        p = bc.compute_inlet_pressure()
        assert p.shape == (3,)
        assert torch.allclose(p, torch.full((3,), 101325.0, dtype=p.dtype))

    def test_compute_inlet_pressure_with_k_scalar(self, simple_patch):
        """With scalar TKE: p_inlet = p0 + 2/3 * k."""
        bc = PressureInletOutlet2BC(simple_patch, {"p0": 100000.0})
        p = bc.compute_inlet_pressure(k=300.0)
        # 100000 + 200 = 100200
        assert torch.allclose(p, torch.full((3,), 100200.0, dtype=p.dtype))

    def test_compute_inlet_pressure_with_k_tensor(self, simple_patch):
        """With per-face TKE tensor."""
        bc = PressureInletOutlet2BC(simple_patch, {"p0": 100000.0})
        k = torch.tensor([300.0, 600.0, 900.0], dtype=torch.float64)
        p = bc.compute_inlet_pressure(k=k)
        expected = torch.tensor([100200.0, 100400.0, 100600.0], dtype=torch.float64)
        assert torch.allclose(p, expected)

    def test_apply_outflow_zero_gradient(self, simple_patch):
        """Outflow (all positive flux): copies owner values."""
        bc = PressureInletOutlet2BC(simple_patch, {"p0": 101325.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 50000.0
        field[1] = 60000.0
        field[2] = 70000.0
        flux = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field = bc.apply(field, flux=flux)
        assert field[10] == pytest.approx(50000.0)
        assert field[11] == pytest.approx(60000.0)
        assert field[12] == pytest.approx(70000.0)

    def test_apply_inflow_uses_p0(self, simple_patch):
        """Inflow (all negative flux): applies p0."""
        bc = PressureInletOutlet2BC(simple_patch, {"p0": 101325.0})
        field = torch.zeros(15, dtype=torch.float64)
        flux = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
        field = bc.apply(field, flux=flux)
        assert field[10] == pytest.approx(101325.0)
        assert field[11] == pytest.approx(101325.0)
        assert field[12] == pytest.approx(101325.0)

    def test_apply_inflow_with_tke(self, simple_patch):
        """Inflow with TKE: applies p0 + 2/3 * k."""
        bc = PressureInletOutlet2BC(simple_patch, {"p0": 100000.0})
        field = torch.zeros(15, dtype=torch.float64)
        flux = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
        k = torch.tensor([300.0, 600.0, 900.0], dtype=torch.float64)
        field = bc.apply(field, flux=flux, k=k)
        assert field[10] == pytest.approx(100200.0)
        assert field[11] == pytest.approx(100400.0)
        assert field[12] == pytest.approx(100600.0)

    def test_apply_mixed_flow(self, simple_patch):
        """Mixed inflow/outflow: correct treatment per face."""
        bc = PressureInletOutlet2BC(simple_patch, {"p0": 101325.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 50000.0
        field[1] = 60000.0
        field[2] = 70000.0
        # face 0 inflow, face 1 outflow, face 2 inflow
        flux = torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float64)
        field = bc.apply(field, flux=flux)
        assert field[10] == pytest.approx(101325.0)  # inflow → p0
        assert field[11] == pytest.approx(60000.0)   # outflow → owner
        assert field[12] == pytest.approx(101325.0)  # inflow → p0

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PressureInletOutlet2BC(simple_patch, {"p0": 101325.0})
        field = torch.zeros(20, dtype=torch.float64)
        flux = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
        field = bc.apply(field, patch_idx=5, flux=flux)
        assert field[5] == pytest.approx(101325.0)
        assert field[6] == pytest.approx(101325.0)
        assert field[7] == pytest.approx(101325.0)

    def test_apply_no_flux_or_velocity_assumes_outflow(self, simple_patch):
        """Without flux or velocity: assumes outflow (zero gradient)."""
        bc = PressureInletOutlet2BC(simple_patch, {"p0": 101325.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 50000.0
        field[1] = 60000.0
        field[2] = 70000.0
        field = bc.apply(field)
        assert field[10] == pytest.approx(50000.0)
        assert field[11] == pytest.approx(60000.0)
        assert field[12] == pytest.approx(70000.0)

    def test_matrix_contributions_inflow(self, simple_patch):
        """Inflow faces contribute to matrix; outflow faces do not."""
        bc = PressureInletOutlet2BC(simple_patch, {"p0": 101325.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        flux = torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, flux=flux)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # At least some contribution from inflow faces
        assert (diag > 0).any()

    def test_matrix_contributions_all_outflow(self, simple_patch):
        """All outflow: no matrix contribution."""
        bc = PressureInletOutlet2BC(simple_patch, {"p0": 101325.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        flux = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, flux=flux)
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_matrix_contributions_with_tke(self, simple_patch):
        """TKE contribution modifies source term."""
        bc = PressureInletOutlet2BC(simple_patch, {"p0": 100000.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        flux = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)

        _, source_no_k = bc.matrix_contributions(field, n_cells, flux=flux)
        _, source_with_k = bc.matrix_contributions(field, n_cells, flux=flux, k=300.0)

        # TKE increases effective inlet pressure → larger source
        assert (source_with_k >= source_no_k).any()
