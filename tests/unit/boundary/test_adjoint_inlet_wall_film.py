"""Tests for adjointInlet and wallFilm boundary conditions."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.adjoint_inlet import AdjointInletBC
from pyfoam.boundary.wall_film import WallFilmBC


# ======================================================================
# AdjointInletBC
# ======================================================================


class TestAdjointInletBC:
    """Test the adjointInlet boundary condition."""

    def test_registration(self):
        """adjointInlet is registered in the RTS registry."""
        assert "adjointInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("adjointInlet", simple_patch)
        assert isinstance(bc, AdjointInletBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = AdjointInletBC(simple_patch)
        assert bc.type_name == "adjointInlet"

    def test_default_coeffs(self, simple_patch):
        """Default coefficients."""
        bc = AdjointInletBC(simple_patch)
        assert bc.ua_name == "Ua"
        assert bc.scale == 1.0
        assert bc.base_velocity == (0.0, 0.0, 0.0)

    def test_custom_coeffs(self, simple_patch):
        """Custom coefficients are parsed."""
        bc = AdjointInletBC(simple_patch, coeffs={
            "UaName": "Uadjoint", "scale": 0.5, "value": (1.0, 0.0, 0.0),
        })
        assert bc.ua_name == "Uadjoint"
        assert bc.scale == 0.5
        assert bc.base_velocity == (1.0, 0.0, 0.0)

    def test_apply_base_only(self, simple_patch):
        """apply() with no adjoint field sets base velocity."""
        bc = AdjointInletBC(simple_patch, coeffs={"value": (2.0, 0.0, 0.0)})
        field = torch.zeros(15, 3, dtype=torch.float64)
        bc.apply(field)
        # Faces 10-12 should have base velocity
        assert torch.allclose(field[10], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))

    def test_apply_with_adjoint(self, simple_patch):
        """apply() combines base velocity with adjoint correction."""
        bc = AdjointInletBC(simple_patch, coeffs={
            "value": (1.0, 0.0, 0.0), "scale": 2.0,
        })
        field = torch.zeros(15, 3, dtype=torch.float64)
        # Adjoint velocity at 3 faces
        Ua = torch.tensor([
            [0.5, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, Ua=Ua)
        # U = base + scale * Ua = 1.0 + 2.0 * [0.5, 0.3, 0.1]
        assert torch.allclose(field[10, 0], torch.tensor(2.0, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(field[11, 0], torch.tensor(1.6, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(field[12, 0], torch.tensor(1.2, dtype=torch.float64), atol=1e-10)

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx writes to contiguous slice."""
        bc = AdjointInletBC(simple_patch, coeffs={"value": (3.0, 0.0, 0.0)})
        field = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5], torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64))

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = AdjointInletBC(simple_patch)
        field = torch.zeros(15, 3, dtype=torch.float64)
        field[3] = torch.tensor([5.0, 6.0, 7.0])
        bc.apply(field)
        assert torch.allclose(field[3], torch.tensor([5.0, 6.0, 7.0], dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """Penalty method contributes to diagonal and source."""
        bc = AdjointInletBC(simple_patch, coeffs={"value": (10.0, 0.0, 0.0)})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # U_x = 10.0, source = 2.0 * 10.0 = 20.0 per face
        assert torch.allclose(
            source, torch.tensor([20.0, 20.0, 20.0], dtype=torch.float64), atol=1e-10,
        )

    def test_matrix_contributions_with_adjoint(self, simple_patch):
        """Penalty method includes adjoint correction."""
        bc = AdjointInletBC(simple_patch, coeffs={
            "value": (5.0, 0.0, 0.0), "scale": 1.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        Ua = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, Ua=Ua)
        # U = base + scale * Ua = [6, 7, 8]
        # source = coeff * U_x = 2.0 * [6, 7, 8]
        assert torch.allclose(
            source, torch.tensor([12.0, 14.0, 16.0], dtype=torch.float64), atol=1e-10,
        )


# ======================================================================
# WallFilmBC
# ======================================================================


class TestWallFilmBC:
    """Test the wallFilm boundary condition."""

    def test_registration(self):
        """wallFilm is registered in the RTS registry."""
        assert "wallFilm" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("wallFilm", simple_patch)
        assert isinstance(bc, WallFilmBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = WallFilmBC(simple_patch)
        assert bc.type_name == "wallFilm"

    def test_default_coeffs(self, simple_patch):
        """Default coefficients."""
        bc = WallFilmBC(simple_patch)
        assert bc.delta_init == 0.0
        assert bc.Tf_init == 300.0
        assert bc.rho_f == 1000.0
        assert bc.mu_f == 1e-3
        assert bc.sigma == 0.07

    def test_custom_coeffs(self, simple_patch):
        """Custom coefficients are parsed."""
        bc = WallFilmBC(simple_patch, coeffs={
            "delta": 1e-4, "Tf": 350.0, "rho_f": 800.0,
        })
        assert bc.delta_init == 1e-4
        assert bc.Tf_init == 350.0
        assert bc.rho_f == 800.0

    def test_apply_sets_film_thickness(self, simple_patch):
        """apply() sets scalar field to initial film thickness."""
        bc = WallFilmBC(simple_patch, coeffs={"delta": 1e-4})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        # Faces 10-12 should have film thickness
        assert torch.allclose(field[10], torch.tensor(1e-4, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(1e-4, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx writes to contiguous slice."""
        bc = WallFilmBC(simple_patch, coeffs={"delta": 5e-3})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5], torch.tensor(5e-3, dtype=torch.float64))

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = WallFilmBC(simple_patch, coeffs={"delta": 1e-4})
        field = torch.zeros(15, dtype=torch.float64)
        field[3] = 42.0
        bc.apply(field)
        assert field[3] == 42.0

    def test_compute_film_velocity_shape(self, simple_patch):
        """compute_film_velocity returns correct shape."""
        bc = WallFilmBC(simple_patch, coeffs={"delta": 1e-4})
        tau_w = torch.ones(3, dtype=torch.float64)
        velocity = bc.compute_film_velocity(tau_w)
        assert velocity.shape == (3, 3)

    def test_compute_film_velocity_positive_shear(self, simple_patch):
        """Positive wall shear produces non-zero film velocity."""
        bc = WallFilmBC(simple_patch, coeffs={"delta": 1e-4, "mu_f": 1e-3})
        tau_w = torch.full((3,), 0.1, dtype=torch.float64)
        velocity = bc.compute_film_velocity(tau_w)
        # Should have non-zero tangential component
        assert velocity.norm(dim=1).sum() > 0

    def test_compute_film_velocity_zero_shear(self, simple_patch):
        """Zero wall shear (no gravity) produces zero velocity."""
        bc = WallFilmBC(simple_patch)
        tau_w = torch.zeros(3, dtype=torch.float64)
        velocity = bc.compute_film_velocity(tau_w)
        assert torch.allclose(velocity, torch.zeros(3, 3, dtype=torch.float64), atol=1e-10)

    def test_compute_film_velocity_with_gravity(self, simple_patch):
        """Gravity adds a component to the film velocity."""
        bc = WallFilmBC(simple_patch, coeffs={"delta": 1e-4})
        tau_w = torch.zeros(3, dtype=torch.float64)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        velocity = bc.compute_film_velocity(tau_w, gravity=gravity)
        # Should have non-zero velocity from gravity-driven flow
        assert velocity.norm(dim=1).sum() > 0

    def test_matrix_contributions(self, simple_patch):
        """Penalty method contributes to diagonal and source."""
        bc = WallFilmBC(simple_patch, coeffs={"delta": 1e-4})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = 2.0 * 1e-4 = 2e-4 per face
        expected_source = torch.full((3,), 2e-4, dtype=torch.float64)
        assert torch.allclose(source, expected_source, atol=1e-8)
