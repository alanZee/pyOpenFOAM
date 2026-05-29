"""Tests for transmissive pressure boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.pressure_transmissive import PressureTransmissiveBC


class TestPressureTransmissiveBC:
    """Test the pressureTransmissive boundary condition."""

    def test_registration(self):
        assert "pressureTransmissive" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "pressureTransmissive", simple_patch,
            {"fieldInf": 101325.0, "lInf": 1.0},
        )
        assert isinstance(bc, PressureTransmissiveBC)

    def test_type_name(self, simple_patch):
        bc = PressureTransmissiveBC(simple_patch)
        assert bc.type_name == "pressureTransmissive"

    def test_default_properties(self, simple_patch):
        bc = PressureTransmissiveBC(simple_patch)
        assert bc.field_inf == pytest.approx(101325.0)
        assert bc.l_inf == pytest.approx(1.0)
        assert bc.gamma == pytest.approx(1.4)

    def test_custom_properties(self, simple_patch):
        bc = PressureTransmissiveBC(simple_patch, {
            "fieldInf": 100000.0,
            "lInf": 0.5,
            "gamma": 1.667,
        })
        assert bc.field_inf == pytest.approx(100000.0)
        assert bc.l_inf == pytest.approx(0.5)
        assert bc.gamma == pytest.approx(1.667)

    def test_apply_without_velocity_fallback(self, simple_patch):
        """Without velocity, falls back to zero-gradient (owner values)."""
        bc = PressureTransmissiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        bc.apply(field)

        assert field[10] == pytest.approx(100.0)
        assert field[11] == pytest.approx(200.0)
        assert field[12] == pytest.approx(300.0)

    def test_apply_with_velocity(self, simple_patch):
        """With velocity, applies characteristic correction."""
        bc = PressureTransmissiveBC(simple_patch, {"fieldInf": 0.0, "lInf": 1.0})

        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0

        # Outward velocity (along face normal +x)
        velocity = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64)
        rho = 1.0

        bc.apply(field, velocity=velocity, rho=rho, c=343.0)

        # Outward velocity should reduce interior pressure at boundary
        assert field[10] < 100.0
        assert field[11] < 200.0
        assert field[12] < 300.0

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PressureTransmissiveBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 50.0
        field[1] = 60.0
        field[2] = 70.0
        bc.apply(field, patch_idx=10)

        assert field[10] == pytest.approx(50.0)
        assert field[11] == pytest.approx(60.0)
        assert field[12] == pytest.approx(70.0)

    def test_apply_zero_velocity_no_correction(self, simple_patch):
        """Zero velocity should give zero-gradient result."""
        bc = PressureTransmissiveBC(simple_patch, {"fieldInf": 0.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0

        velocity = torch.zeros(3, 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity, rho=1.0, c=343.0)

        # Zero normal velocity → zero correction → owner values
        assert field[10] == pytest.approx(100.0)
        assert field[11] == pytest.approx(200.0)
        assert field[12] == pytest.approx(300.0)

    def test_matrix_contributions(self, simple_patch):
        """Relaxation term in matrix."""
        bc = PressureTransmissiveBC(simple_patch, {"fieldInf": 100000.0, "lInf": 1.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Relaxation coefficient: rho*c*A / l_inf = 1.225*343*1.0/1.0 = 420.175
        expected_coeff = 1.225 * 343.0 * 1.0 / 1.0  # per face
        assert torch.allclose(
            diag,
            torch.full((n_cells,), expected_coeff, dtype=torch.float64),
            rtol=1e-6,
        )
        # source = relax_coeff * field_inf
        assert torch.allclose(
            source,
            torch.full((n_cells,), expected_coeff * 100000.0, dtype=torch.float64),
            rtol=1e-6,
        )

    def test_matrix_contributions_zero(self, simple_patch):
        """Matrix contributions are non-zero (relaxation)."""
        bc = PressureTransmissiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        # Should have non-zero relaxation contribution
        assert (diag > 0).all()
        assert (source > 0).all()
