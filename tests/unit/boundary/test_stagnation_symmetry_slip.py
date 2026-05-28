"""Tests for stagnationInlet and symmetrySlip boundary conditions."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.stagnation_inlet import StagnationInletBC
from pyfoam.boundary.symmetry_slip import SymmetrySlipBC


# ======================================================================
# StagnationInletBC
# ======================================================================


class TestStagnationInletBC:
    """Test the stagnationInlet boundary condition."""

    def test_registration(self):
        """stagnationInlet is registered in the RTS registry."""
        assert "stagnationInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("stagnationInlet", simple_patch)
        assert isinstance(bc, StagnationInletBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = StagnationInletBC(simple_patch)
        assert bc.type_name == "stagnationInlet"

    def test_default_coeffs(self, simple_patch):
        """Default p0 = 101325, rho = 1.0."""
        bc = StagnationInletBC(simple_patch)
        assert bc.p0 == 101325.0
        assert bc.rho == 1.0

    def test_custom_coeffs(self, simple_patch):
        """Coefficients parsed from coeffs dict."""
        bc = StagnationInletBC(simple_patch, coeffs={"p0": 200000.0, "rho": 1.225})
        assert bc.p0 == 200000.0
        assert bc.rho == 1.225

    def test_compute_velocity_zero_pressure_diff(self, simple_patch):
        """Zero pressure difference yields zero velocity."""
        bc = StagnationInletBC(simple_patch, coeffs={"p0": 101325.0, "rho": 1.0})
        p = torch.tensor([101325.0, 101325.0, 101325.0], dtype=torch.float64)
        u = bc.compute_velocity_magnitude(p)
        assert torch.allclose(u, torch.zeros(3, dtype=torch.float64), atol=1e-10)

    def test_compute_velocity_positive_pressure_diff(self, simple_patch):
        """U = sqrt(2*(p0-p)/rho)."""
        bc = StagnationInletBC(simple_patch, coeffs={"p0": 101325.0, "rho": 1.225})
        p = torch.tensor([101225.0, 101225.0, 101225.0], dtype=torch.float64)
        u = bc.compute_velocity_magnitude(p)
        expected = (2.0 * 100.0 / 1.225) ** 0.5
        assert torch.allclose(u, torch.full((3,), expected, dtype=torch.float64), atol=1e-6)

    def test_compute_velocity_clamped_negative_diff(self, simple_patch):
        """Negative pressure difference (p > p0) is clamped to zero."""
        bc = StagnationInletBC(simple_patch, coeffs={"p0": 100000.0, "rho": 1.0})
        p = torch.tensor([200000.0, 200000.0, 200000.0], dtype=torch.float64)
        u = bc.compute_velocity_magnitude(p)
        assert torch.allclose(u, torch.zeros(3, dtype=torch.float64), atol=1e-10)

    def test_apply_with_pressure(self, simple_patch):
        """apply() sets velocity = U_mag * face_normal."""
        bc = StagnationInletBC(
            simple_patch, coeffs={"p0": 101325.0, "rho": 1.0}
        )
        # Velocity field (n_total=15, 3)
        field = torch.zeros(15, 3, dtype=torch.float64)
        # p0 - p = 50 for all faces => U = sqrt(100) = 10
        p = torch.tensor([101275.0, 101275.0, 101275.0], dtype=torch.float64)
        bc.apply(field, p=p)
        # normals face +x => velocity = (10, 0, 0)
        assert torch.allclose(field[10, 0], torch.tensor(10.0, dtype=torch.float64), atol=1e-6)
        assert torch.allclose(field[10, 1], torch.tensor(0.0, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(field[11, 0], torch.tensor(10.0, dtype=torch.float64), atol=1e-6)

    def test_apply_no_pressure_yields_zero_velocity(self, simple_patch):
        """Without pressure argument, velocity defaults to zero (p=p0)."""
        bc = StagnationInletBC(simple_patch, coeffs={"p0": 101325.0, "rho": 1.0})
        field = torch.ones(15, 3, dtype=torch.float64)
        bc.apply(field)
        # p defaults to p0, so dp=0 => velocity=0
        assert torch.allclose(field[10], torch.zeros(3, dtype=torch.float64), atol=1e-10)

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx writes to contiguous slice."""
        bc = StagnationInletBC(simple_patch, coeffs={"p0": 101325.0, "rho": 1.0})
        field = torch.zeros(20, 3, dtype=torch.float64)
        p = torch.tensor([101275.0, 101275.0, 101275.0], dtype=torch.float64)
        bc.apply(field, patch_idx=5, p=p)
        assert torch.allclose(field[5, 0], torch.tensor(10.0, dtype=torch.float64), atol=1e-6)

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = StagnationInletBC(simple_patch, coeffs={"p0": 101325.0, "rho": 1.0})
        field = torch.zeros(15, 3, dtype=torch.float64)
        # Set some internal cell values that should not change
        field[3] = torch.tensor([5.0, 6.0, 7.0])
        field[8] = torch.tensor([8.0, 9.0, 10.0])
        p = torch.tensor([101275.0, 101275.0, 101275.0], dtype=torch.float64)
        bc.apply(field, p=p)
        # Cells 3, 8 (not adjacent to patch) are untouched
        assert torch.allclose(field[3], torch.tensor([5.0, 6.0, 7.0], dtype=torch.float64))
        assert torch.allclose(field[8], torch.tensor([8.0, 9.0, 10.0], dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """Penalty method contributes to diagonal and source."""
        bc = StagnationInletBC(simple_patch, coeffs={"p0": 101325.0, "rho": 1.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        p = torch.tensor([101275.0, 101275.0, 101275.0], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, p=p)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # U_x = 10.0, source = 2.0 * 10.0 = 20.0 per face
        assert torch.allclose(
            source, torch.tensor([20.0, 20.0, 20.0], dtype=torch.float64), atol=1e-4
        )

    def test_matrix_contributions_preserves_existing(self, simple_patch):
        """matrix_contributions accumulates into pre-existing diag/source."""
        bc = StagnationInletBC(simple_patch, coeffs={"p0": 101325.0, "rho": 1.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64) * 5.0
        p = torch.tensor([101275.0, 101275.0, 101275.0], dtype=torch.float64)
        new_diag, new_source = bc.matrix_contributions(
            field, n_cells, diag=diag, source=source, p=p,
        )
        # diag = 1 + 2 = 3
        assert torch.allclose(new_diag, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
        # source = 5 + 20 = 25
        assert torch.allclose(
            new_source, torch.tensor([25.0, 25.0, 25.0], dtype=torch.float64), atol=1e-4,
        )


# ======================================================================
# SymmetrySlipBC
# ======================================================================


class TestSymmetrySlipBC:
    """Test the symmetrySlip boundary condition."""

    def test_registration(self):
        """symmetrySlip is registered in the RTS registry."""
        assert "symmetrySlip" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("symmetrySlip", simple_patch)
        assert isinstance(bc, SymmetrySlipBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = SymmetrySlipBC(simple_patch)
        assert bc.type_name == "symmetrySlip"

    def test_apply_scalar_copies_owner_values(self, simple_patch):
        """For scalar fields, apply() behaves as zeroGradient."""
        bc = SymmetrySlipBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(20.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(30.0, dtype=torch.float64))

    def test_apply_vector_removes_normal_component(self, simple_patch):
        """For vector fields, apply() removes the normal component (free slip)."""
        bc = SymmetrySlipBC(simple_patch)
        field = torch.zeros(15, 3, dtype=torch.float64)
        # Normals face +x
        field[0] = torch.tensor([5.0, 3.0, 0.0])
        field[1] = torch.tensor([10.0, -2.0, 1.0])
        field[2] = torch.tensor([1.0, 0.0, 4.0])
        bc.apply(field)
        # Normal removed, tangential preserved
        assert torch.allclose(field[10], torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([0.0, -2.0, 1.0], dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx writes to contiguous slice."""
        bc = SymmetrySlipBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 6.0
        field[2] = 7.0
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5], torch.tensor(5.0, dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor(6.0, dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor(7.0, dtype=torch.float64))

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = SymmetrySlipBC(simple_patch)
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_matrix_contributions_zero(self, simple_patch):
        """Free-slip has zero matrix contribution (no flux coupling)."""
        bc = SymmetrySlipBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_matrix_contributions_preserves_existing(self, simple_patch):
        """matrix_contributions preserves pre-existing diag/source values."""
        bc = SymmetrySlipBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 5
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64) * 2.0
        new_diag, new_source = bc.matrix_contributions(
            field, n_cells, diag=diag, source=source,
        )
        assert torch.allclose(new_diag, torch.ones(n_cells, dtype=torch.float64))
        assert torch.allclose(new_source, torch.ones(n_cells, dtype=torch.float64) * 2.0)

    def test_symmetry_plane_validation_logs_warning(self, caplog):
        """Non-planar patch should log a warning."""
        import logging

        # Create a patch with non-parallel normals
        from pyfoam.boundary.boundary_condition import Patch

        non_planar_patch = Patch(
            name="nonPlanar",
            face_indices=torch.tensor([0, 1, 2]),
            face_normals=torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=torch.float64),
            face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
            delta_coeffs=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
            owner_cells=torch.tensor([0, 1, 2]),
        )

        with caplog.at_level(logging.WARNING):
            bc = SymmetrySlipBC(non_planar_patch)

        assert "not parallel" in caplog.text or "symmetry plane" in caplog.text

    def test_planar_patch_no_warning(self, simple_patch, caplog):
        """Planar patch (all normals parallel) should not log a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            SymmetrySlipBC(simple_patch)

        assert "not parallel" not in caplog.text
