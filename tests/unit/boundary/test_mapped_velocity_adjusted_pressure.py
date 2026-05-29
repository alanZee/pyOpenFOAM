"""Tests for mapped velocity adjusted pressure boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_velocity_adjusted_pressure import MappedVelocityAdjustedPressureBC


class TestMappedVelocityAdjustedPressureBC:
    """Test the mappedVelocityAdjustedPressure boundary condition."""

    def test_registration(self):
        assert "mappedVelocityAdjustedPressure" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "mappedVelocityAdjustedPressure", simple_patch,
            {"pRef": 101325.0, "rho": 1.0},
        )
        assert isinstance(bc, MappedVelocityAdjustedPressureBC)

    def test_type_name(self, simple_patch):
        bc = MappedVelocityAdjustedPressureBC(simple_patch)
        assert bc.type_name == "mappedVelocityAdjustedPressure"

    def test_default_properties(self, simple_patch):
        bc = MappedVelocityAdjustedPressureBC(simple_patch)
        assert bc.p_ref == pytest.approx(101325.0)
        assert bc.rho == pytest.approx(1.0)

    def test_custom_properties(self, simple_patch):
        bc = MappedVelocityAdjustedPressureBC(simple_patch, {
            "pRef": 100000.0, "rho": 1.225,
        })
        assert bc.p_ref == pytest.approx(100000.0)
        assert bc.rho == pytest.approx(1.225)

    def test_mapped_pressure_default_none(self, simple_patch):
        bc = MappedVelocityAdjustedPressureBC(simple_patch)
        assert bc._mapped_pressure is None

    def test_set_mapped_pressure(self, simple_patch):
        bc = MappedVelocityAdjustedPressureBC(simple_patch)
        p = torch.tensor([102000.0, 103000.0, 104000.0], dtype=torch.float64)
        bc.set_mapped_pressure(p)
        assert bc._mapped_pressure is not None
        assert torch.allclose(bc._mapped_pressure, p)

    def test_apply_without_mapped_pressure(self, simple_patch):
        """Without mapped pressure, velocity should be zero."""
        bc = MappedVelocityAdjustedPressureBC(simple_patch)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        assert torch.allclose(field[10], torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(field[11], torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(field[12], torch.zeros(3, dtype=torch.float64))

    def test_apply_with_higher_pressure(self, simple_patch):
        """Higher mapped pressure than p_ref should produce velocity."""
        bc = MappedVelocityAdjustedPressureBC(simple_patch, {
            "pRef": 100000.0, "rho": 1.0,
        })
        # Mapped pressure > p_ref -> positive dp -> velocity
        p = torch.tensor([100500.0, 101000.0, 102000.0], dtype=torch.float64)
        bc.set_mapped_pressure(p)

        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # Velocity should be along -normal (inward), so x-component negative
        assert field[10, 0] < 0.0
        assert field[11, 0] < 0.0
        assert field[12, 0] < 0.0

    def test_apply_with_lower_pressure_clamped(self, simple_patch):
        """Lower mapped pressure should produce zero velocity (clamped)."""
        bc = MappedVelocityAdjustedPressureBC(simple_patch, {
            "pRef": 100000.0, "rho": 1.0,
        })
        p = torch.tensor([99000.0, 98000.0, 97000.0], dtype=torch.float64)
        bc.set_mapped_pressure(p)

        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # dp < 0 -> clamped to zero -> no velocity
        assert torch.allclose(field[10], torch.zeros(3, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MappedVelocityAdjustedPressureBC(simple_patch, {"pRef": 0.0, "rho": 1.0})
        p = torch.tensor([500.0, 500.0, 500.0], dtype=torch.float64)
        bc.set_mapped_pressure(p)

        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        # u = sqrt(2*500/1) = sqrt(1000) ~ 31.62, direction = -normal = (-1,0,0)
        expected_mag = (2.0 * 500.0) ** 0.5
        assert field[5, 0] == pytest.approx(-expected_mag, rel=1e-6)
        assert field[6, 0] == pytest.approx(-expected_mag, rel=1e-6)
        assert field[7, 0] == pytest.approx(-expected_mag, rel=1e-6)

    def test_matrix_contributions(self, simple_patch):
        bc = MappedVelocityAdjustedPressureBC(simple_patch, {"pRef": 0.0, "rho": 1.0})
        p = torch.tensor([500.0, 500.0, 500.0], dtype=torch.float64)
        bc.set_mapped_pressure(p)

        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))

    def test_matrix_contributions_no_mapped(self, simple_patch):
        bc = MappedVelocityAdjustedPressureBC(simple_patch)
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        # diag should still have penalty coefficients
        assert (diag > 0).all()
        # source should be zero (no velocity)
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_neighbour_patch(self, simple_patch):
        bc = MappedVelocityAdjustedPressureBC(
            simple_patch, {"neighbourPatch": "outletPatch"},
        )
        assert bc.neighbour_patch_name == "outletPatch"
