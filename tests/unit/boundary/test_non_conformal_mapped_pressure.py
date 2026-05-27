"""Tests for non-conformal couple and mapped pressure inlet boundary conditions."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.non_conformal_couple import NonConformalCoupleBC
from pyfoam.boundary.mapped_pressure_inlet import MappedPressureInletBC


# ---------------------------------------------------------------------------
# NonConformalCoupleBC tests
# ---------------------------------------------------------------------------

class TestNonConformalCoupleBC:
    """Test the non-conformal couple boundary condition."""

    def test_registration(self):
        """nonConformalCouple is registered in the RTS registry."""
        assert "nonConformalCouple" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create(
            "nonConformalCouple", simple_patch, {"neighbourPatch": "otherPatch"}
        )
        assert isinstance(bc, NonConformalCoupleBC)

    def test_neighbour_patch_name(self, simple_patch):
        """neighbourPatch coefficient is stored."""
        bc = NonConformalCoupleBC(simple_patch, {"neighbourPatch": "outlet"})
        assert bc.neighbour_patch_name == "outlet"

    def test_neighbour_patch_fallback(self, simple_patch):
        """Falls back to Patch.neighbour_patch if coeff not set."""
        bc = NonConformalCoupleBC(simple_patch)
        assert bc.neighbour_patch_name is None

    def test_transform_default(self, simple_patch):
        """Default transform is 'none'."""
        bc = NonConformalCoupleBC(simple_patch)
        assert bc.transform == "none"

    def test_transform_from_coeffs(self, simple_patch):
        """Transform can be set via coefficients."""
        bc = NonConformalCoupleBC(simple_patch, {"transform": "rotational"})
        assert bc.transform == "rotational"

    def test_apply_with_mapping(self, simple_patch):
        """apply() interpolates between owner and neighbour values."""
        bc = NonConformalCoupleBC(simple_patch, {"neighbourPatch": "other"})

        # 4 neighbour face values
        nbr_values = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
        # Weight = 0.5 for each face, indices into nbr_values
        weights = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        nbr_indices = torch.tensor([0, 1, 2])
        bc.set_mapping(nbr_values, weights, nbr_indices)

        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0

        bc.apply(field)
        # value = 0.5 * nbr + 0.5 * owner
        expected = torch.tensor([55.0, 110.0, 165.0], dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_with_full_weight(self, simple_patch):
        """Weight=1.0 gives pure neighbour value."""
        bc = NonConformalCoupleBC(simple_patch)
        nbr_values = torch.tensor([7.0, 8.0, 9.0, 10.0], dtype=torch.float64)
        weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        nbr_indices = torch.tensor([0, 1, 2])
        bc.set_mapping(nbr_values, weights, nbr_indices)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))

    def test_apply_with_zero_weight(self, simple_patch):
        """Weight=0.0 gives pure owner value."""
        bc = NonConformalCoupleBC(simple_patch)
        nbr_values = torch.tensor([7.0, 8.0, 9.0, 10.0], dtype=torch.float64)
        weights = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        nbr_indices = torch.tensor([0, 1, 2])
        bc.set_mapping(nbr_values, weights, nbr_indices)

        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64))

    def test_apply_without_mapping_fallback(self, simple_patch):
        """Without mapping, falls back to zero-gradient (owner values)."""
        bc = NonConformalCoupleBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx sets values at offset."""
        bc = NonConformalCoupleBC(simple_patch)
        nbr_values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        weights = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float64)
        nbr_indices = torch.tensor([0, 1, 2])
        bc.set_mapping(nbr_values, weights, nbr_indices)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5] > 0
        assert field[6] > 0
        assert field[7] > 0

    def test_matrix_contributions_with_mapping(self, simple_patch):
        """Matrix contributions use weighted penalty method."""
        bc = NonConformalCoupleBC(simple_patch)
        nbr_values = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
        weights = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        nbr_indices = torch.tensor([0, 1, 2])
        bc.set_mapping(nbr_values, weights, nbr_indices)

        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)

        # coeff = delta * area = 2.0 * 1.0 = 2.0
        # weighted_coeff = 0.5 * 2.0 = 1.0
        # diag = [1.0, 1.0, 1.0]
        # source = 1.0 * [10, 20, 30] = [10, 20, 30]
        assert torch.allclose(diag, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))

    def test_matrix_contributions_without_mapping(self, simple_patch):
        """Without mapping, uses unweighted penalty (like zeroGradient+penalty)."""
        bc = NonConformalCoupleBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)

        # coeff = delta * area = 2.0 * 1.0 = 2.0 (unweighted)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_matrix_contributions_accumulate(self, simple_patch):
        """Matrix contributions accumulate into pre-existing tensors."""
        bc = NonConformalCoupleBC(simple_patch)
        nbr_values = torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float64)
        weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        nbr_indices = torch.tensor([0, 1, 2])
        bc.set_mapping(nbr_values, weights, nbr_indices)

        field = torch.zeros(15, dtype=torch.float64)
        diag = torch.ones(3, dtype=torch.float64)
        source = torch.ones(3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, diag=diag, source=source)
        assert diag[0] > 1.0
        assert source[0] > 1.0

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = NonConformalCoupleBC(simple_patch)
        assert bc.type_name == "nonConformalCouple"


# ---------------------------------------------------------------------------
# MappedPressureInletBC tests
# ---------------------------------------------------------------------------

class TestMappedPressureInletBC:
    """Test the mapped pressure inlet boundary condition."""

    def test_registration(self):
        """mappedPressureInlet is registered in the RTS registry."""
        assert "mappedPressureInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create(
            "mappedPressureInlet", simple_patch, {"p0": 100000.0}
        )
        assert isinstance(bc, MappedPressureInletBC)

    def test_p0_default(self, simple_patch):
        """Default total pressure is 101325 Pa."""
        bc = MappedPressureInletBC(simple_patch)
        assert bc.p0 == 101325.0

    def test_p0_from_coeffs(self, simple_patch):
        """Total pressure from coefficients."""
        bc = MappedPressureInletBC(simple_patch, {"p0": 200000.0})
        assert bc.p0 == 200000.0

    def test_rho_default(self, simple_patch):
        """Default density is 1.0."""
        bc = MappedPressureInletBC(simple_patch)
        assert bc.rho == 1.0

    def test_rho_from_coeffs(self, simple_patch):
        """Density from coefficients."""
        bc = MappedPressureInletBC(simple_patch, {"rho": 1.225})
        assert bc.rho == 1.225

    def test_neighbour_patch_name(self, simple_patch):
        """neighbourPatch coefficient is stored."""
        bc = MappedPressureInletBC(simple_patch, {"neighbourPatch": "outlet"})
        assert bc.neighbour_patch_name == "outlet"

    def test_apply_bernoulli(self, simple_patch):
        """apply() computes pressure from Bernoulli: p = p0 - 0.5*rho*|U|²."""
        bc = MappedPressureInletBC(simple_patch, {"p0": 100000.0, "rho": 1.0})

        # 3 faces, each with velocity in x-direction
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.set_mapped_velocity(velocity)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        # p = 100000 - 0.5 * 1.0 * |U|²
        # face 0: 100000 - 0.5*100  = 99950
        # face 1: 100000 - 0.5*400  = 99800
        # face 2: 100000 - 0.5*900  = 99550
        expected = torch.tensor([99950.0, 99800.0, 99550.0], dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx."""
        bc = MappedPressureInletBC(simple_patch, {"p0": 100000.0, "rho": 1.0})
        velocity = torch.tensor([
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.set_mapped_velocity(velocity)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        # face 0: 100000 - 0.5*25 = 99987.5
        assert torch.allclose(field[5], torch.tensor(99987.5, dtype=torch.float64))

    def test_apply_without_velocity(self, simple_patch):
        """Without mapped velocity, uses total pressure directly."""
        bc = MappedPressureInletBC(simple_patch, {"p0": 101325.0})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(
            field[10:13],
            torch.full((3,), 101325.0, dtype=torch.float64),
        )

    def test_apply_zero_velocity(self, simple_patch):
        """Zero velocity gives p = p0 (stagnation point)."""
        bc = MappedPressureInletBC(simple_patch, {"p0": 101325.0, "rho": 1.0})
        velocity = torch.zeros((3, 3), dtype=torch.float64)
        bc.set_mapped_velocity(velocity)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(
            field[10:13],
            torch.full((3,), 101325.0, dtype=torch.float64),
        )

    def test_apply_vector_magnitude(self, simple_patch):
        """Bernoulli uses full velocity magnitude, not just x-component."""
        bc = MappedPressureInletBC(simple_patch, {"p0": 100000.0, "rho": 2.0})
        # |U|² = 3² + 4² + 0² = 25
        velocity = torch.tensor([
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.set_mapped_velocity(velocity)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        # p = 100000 - 0.5 * 2.0 * 25 = 99975.0
        assert torch.allclose(field[10], torch.tensor(99975.0, dtype=torch.float64))

    def test_matrix_contributions_with_velocity(self, simple_patch):
        """Matrix contributions use Bernoulli pressure."""
        bc = MappedPressureInletBC(simple_patch, {"p0": 100000.0, "rho": 1.0})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.set_mapped_velocity(velocity)

        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)

        # coeff = delta * area = 2.0 * 1.0 = 2.0
        # diag = [2.0, 2.0, 2.0]
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))

        # source = coeff * p_bernoulli
        # p[0] = 99950, p[1] = 99800, p[2] = 99550
        expected_source = torch.tensor(
            [2.0 * 99950.0, 2.0 * 99800.0, 2.0 * 99550.0], dtype=torch.float64
        )
        assert torch.allclose(source, expected_source)

    def test_matrix_contributions_without_velocity(self, simple_patch):
        """Without velocity, source uses p0."""
        bc = MappedPressureInletBC(simple_patch, {"p0": 101325.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)

        expected_source = torch.tensor(
            [2.0 * 101325.0, 2.0 * 101325.0, 2.0 * 101325.0], dtype=torch.float64
        )
        assert torch.allclose(source, expected_source)

    def test_matrix_contributions_accumulate(self, simple_patch):
        """Matrix contributions accumulate into pre-existing tensors."""
        bc = MappedPressureInletBC(simple_patch, {"p0": 100000.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag = torch.ones(3, dtype=torch.float64)
        source = torch.ones(3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, diag=diag, source=source)
        assert diag[0] > 1.0
        assert source[0] > 1.0

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = MappedPressureInletBC(simple_patch)
        assert bc.type_name == "mappedPressureInlet"
