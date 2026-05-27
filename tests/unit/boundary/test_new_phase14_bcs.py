"""Tests for matchedFlowRateOutlet and fixedShearStress boundary conditions."""

import dataclasses

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.matched_flow_rate import MatchedFlowRateOutletBC
from pyfoam.boundary.fixed_shear_stress import FixedShearStressBC


# ---------------------------------------------------------------------------
# MatchedFlowRateOutletBC
# ---------------------------------------------------------------------------


class TestMatchedFlowRateOutletBC:
    """Test the matchedFlowRateOutlet boundary condition."""

    def test_registration(self):
        assert "matchedFlowRateOutlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "matchedFlowRateOutlet", simple_patch,
            {"massFlowRate": 0.6, "rho": 1.0},
        )
        assert isinstance(bc, MatchedFlowRateOutletBC)

    def test_type_name(self, simple_patch):
        bc = MatchedFlowRateOutletBC(
            simple_patch, {"massFlowRate": 0.6, "rho": 1.0}
        )
        assert bc.type_name == "matchedFlowRateOutlet"

    def test_properties(self, simple_patch):
        bc = MatchedFlowRateOutletBC(
            simple_patch, {"massFlowRate": 1.5, "rho": 2.0}
        )
        assert bc.mass_flow_rate == pytest.approx(1.5)
        assert bc.rho == pytest.approx(2.0)

    def test_default_properties(self, simple_patch):
        """Default: massFlowRate=0, rho=1."""
        bc = MatchedFlowRateOutletBC(simple_patch)
        assert bc.mass_flow_rate == pytest.approx(0.0)
        assert bc.rho == pytest.approx(1.0)

    def test_apply_sets_outlet_velocity(self, simple_patch):
        """simple_patch: A_total = 3.0, rho = 1.0, massFlowRate = 0.6
        u_mag = 0.6 / (1.0 * 3.0) = 0.2
        velocity = u_mag * face_normal = (0.2, 0, 0) per face
        """
        bc = MatchedFlowRateOutletBC(
            simple_patch, {"massFlowRate": 0.6, "rho": 1.0}
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)
        assert torch.allclose(field[11], expected)
        assert torch.allclose(field[12], expected)

    def test_apply_with_rho(self, simple_patch):
        """massFlowRate = 0.6, rho = 2.0 => u_mag = 0.6 / (2.0 * 3.0) = 0.1"""
        bc = MatchedFlowRateOutletBC(
            simple_patch, {"massFlowRate": 0.6, "rho": 2.0}
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MatchedFlowRateOutletBC(
            simple_patch, {"massFlowRate": 0.6, "rho": 1.0}
        )
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        expected = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[5], expected)

    def test_apply_zero_flow(self, simple_patch):
        """Zero mass flow rate yields zero velocity."""
        bc = MatchedFlowRateOutletBC(
            simple_patch, {"massFlowRate": 0.0, "rho": 1.0}
        )
        field = torch.ones((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_zero_area(self, simple_patch):
        """Zero area yields zero velocity without error."""
        patch_zero = dataclasses.replace(
            simple_patch,
            face_areas=torch.zeros(3, dtype=torch.float64),
        )
        bc = MatchedFlowRateOutletBC(
            patch_zero, {"massFlowRate": 1.0, "rho": 1.0}
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """Penalty method: diag += coeff, source += coeff * velocity_x."""
        bc = MatchedFlowRateOutletBC(
            simple_patch, {"massFlowRate": 0.6, "rho": 1.0}
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # velocity_x = 0.2 => source = 2.0 * 0.2 = 0.4 per face
        assert torch.allclose(
            source, torch.tensor([0.4, 0.4, 0.4], dtype=torch.float64)
        )

    def test_accumulated_matrix_contributions(self, simple_patch):
        """Pre-existing diag/source should be accumulated into."""
        bc = MatchedFlowRateOutletBC(
            simple_patch, {"massFlowRate": 0.6, "rho": 1.0}
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64) * 0.5

        diag, source = bc.matrix_contributions(field, n_cells, diag=diag, source=source)
        # Pre-existing values + new contributions
        assert torch.allclose(diag, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
        assert torch.allclose(
            source, torch.tensor([0.9, 0.9, 0.9], dtype=torch.float64)
        )


# ---------------------------------------------------------------------------
# FixedShearStressBC
# ---------------------------------------------------------------------------


class TestFixedShearStressBC:
    """Test the fixedShearStress boundary condition."""

    def test_registration(self):
        assert "fixedShearStress" in BoundaryCondition.available_types()

    def test_factory_creation(self, wall_patch):
        bc = BoundaryCondition.create(
            "fixedShearStress", wall_patch,
            {"tau0": [0.1, 0.0, 0.0], "rho": 1.0},
        )
        assert isinstance(bc, FixedShearStressBC)

    def test_type_name(self, wall_patch):
        bc = FixedShearStressBC(wall_patch, {"tau0": 0.1})
        assert bc.type_name == "fixedShearStress"

    def test_properties_vector_tau(self, wall_patch):
        bc = FixedShearStressBC(
            wall_patch, {"tau0": [0.1, 0.2, 0.3], "rho": 1.5}
        )
        assert torch.allclose(
            bc.tau0, torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        )
        assert bc.rho == pytest.approx(1.5)

    def test_properties_scalar_tau(self, wall_patch):
        """Scalar tau0 is applied in the x-direction."""
        bc = FixedShearStressBC(wall_patch, {"tau0": 0.5})
        assert torch.allclose(
            bc.tau0, torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        )

    def test_properties_tensor_tau(self, wall_patch):
        tau = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc = FixedShearStressBC(wall_patch, {"tau0": tau})
        assert torch.allclose(bc.tau0, tau)

    def test_default_properties(self, wall_patch):
        """Default tau0 = (0, 0, 0), rho = 1.0."""
        bc = FixedShearStressBC(wall_patch)
        assert torch.allclose(
            bc.tau0, torch.zeros(3, dtype=torch.float64)
        )
        assert bc.rho == pytest.approx(1.0)

    def test_invalid_tau_raises(self, wall_patch):
        """tau0 with wrong size should raise ValueError."""
        with pytest.raises(ValueError, match="3-element vector"):
            FixedShearStressBC(wall_patch, {"tau0": [0.1, 0.2]})
        with pytest.raises(ValueError, match="3-element vector"):
            FixedShearStressBC(wall_patch, {"tau0": [0.1, 0.2, 0.3, 0.4]})

    def test_apply_sets_shear_velocity(self, wall_patch):
        """wall_patch: deltaCoeff=100 => delta=0.01, rho=1.0, tau_x=1.0
        u_x = tau_x / (rho * deltaCoeff) = 1.0 / (1.0 * 100) = 0.01
        All faces have normals (0, -1, 0), so tangential is in (x, z) plane.
        The BC applies tau * inv_rho_delta regardless of normal direction.
        """
        bc = FixedShearStressBC(
            wall_patch, {"tau0": [1.0, 0.0, 0.0], "rho": 1.0}
        )
        field = torch.zeros((35, 3), dtype=torch.float64)
        bc.apply(field)

        # velocity = tau * inv_rho_delta = (1.0, 0, 0) * (1 / (1*100))
        expected = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[30], expected)
        assert torch.allclose(field[31], expected)
        assert torch.allclose(field[32], expected)

    def test_apply_with_vector_tau(self, wall_patch):
        """Vector tau0 produces velocity in all three directions."""
        bc = FixedShearStressBC(
            wall_patch, {"tau0": [1.0, 2.0, 3.0], "rho": 1.0}
        )
        field = torch.zeros((35, 3), dtype=torch.float64)
        bc.apply(field)

        # inv_rho_delta = 1 / (1.0 * 100) = 0.01
        expected = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        assert torch.allclose(field[30], expected)

    def test_apply_with_patch_idx(self, wall_patch):
        bc = FixedShearStressBC(
            wall_patch, {"tau0": [1.0, 0.0, 0.0], "rho": 1.0}
        )
        field = torch.zeros((40, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        expected = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[5], expected)

    def test_apply_with_different_rho(self, wall_patch):
        """rho=2.0 => inv_rho_delta = 1/(2*100) = 0.005."""
        bc = FixedShearStressBC(
            wall_patch, {"tau0": [1.0, 0.0, 0.0], "rho": 2.0}
        )
        field = torch.zeros((35, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([0.005, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[30], expected)

    def test_apply_zero_tau(self, wall_patch):
        """Zero shear stress yields zero velocity."""
        bc = FixedShearStressBC(wall_patch, {"tau0": [0.0, 0.0, 0.0]})
        field = torch.ones((35, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(
            field[30:33], torch.zeros(3, 3, dtype=torch.float64)
        )

    def test_matrix_contributions(self, wall_patch):
        """Penalty method: diag += coeff, source += coeff * velocity_x."""
        bc = FixedShearStressBC(
            wall_patch, {"tau0": [1.0, 0.0, 0.0], "rho": 1.0}
        )
        field = torch.zeros((35, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 100.0 * 1.0 = 100.0 per face
        assert torch.allclose(
            diag, torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        )
        # velocity_x = 1.0 / (1.0 * 100.0) = 0.01
        # source = 100.0 * 0.01 = 1.0 per face
        assert torch.allclose(
            source, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        )

    def test_accumulated_matrix_contributions(self, wall_patch):
        """Pre-existing diag/source should be accumulated into."""
        bc = FixedShearStressBC(
            wall_patch, {"tau0": [1.0, 0.0, 0.0], "rho": 1.0}
        )
        field = torch.zeros((35, 3), dtype=torch.float64)
        n_cells = 3
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64) * 0.5

        diag, source = bc.matrix_contributions(field, n_cells, diag=diag, source=source)
        # Pre-existing values + new contributions
        assert torch.allclose(
            diag, torch.tensor([101.0, 101.0, 101.0], dtype=torch.float64)
        )
        assert torch.allclose(
            source, torch.tensor([1.5, 1.5, 1.5], dtype=torch.float64)
        )
