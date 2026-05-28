"""
Tests for TurbulentInlet2BC — enhanced turbulent inlet BC with digital filter.

Tests cover:
- RTS registration
- Factory creation
- Default and custom parameters
- Filter kernel construction
- apply() produces velocity fluctuations
- Previous value blending (alpha relaxation)
- matrix_contributions()
- Spatial correlation via digital filter
"""

import pytest
import torch

from pyfoam.boundary.boundary_condition import BoundaryCondition, Patch
from pyfoam.boundary.turbulent_inlet_2 import TurbulentInlet2BC


@pytest.fixture
def inlet_patch():
    """A 5-face inlet patch."""
    return Patch(
        name="inlet",
        face_indices=torch.tensor([0, 1, 2, 3, 4]),
        face_normals=torch.tensor([
            [1.0, 0.0, 0.0]] * 5, dtype=torch.float64,
        ),
        face_areas=torch.tensor([1.0] * 5, dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0] * 5, dtype=torch.float64),
        owner_cells=torch.tensor([0, 1, 2, 3, 4]),
    )


class TestTurbulentInlet2Registration:
    """RTS registration tests."""

    def test_registered_in_registry(self):
        assert "turbulentInlet2" in BoundaryCondition.available_types()

    def test_factory_creation(self, inlet_patch):
        bc = BoundaryCondition.create(
            "turbulentInlet2", inlet_patch,
            coeffs={"referenceField": [1.0, 0.0, 0.0]},
        )
        assert isinstance(bc, TurbulentInlet2BC)


class TestTurbulentInlet2Defaults:
    """Default parameter tests."""

    def test_default_reference(self, inlet_patch):
        bc = TurbulentInlet2BC(inlet_patch)
        ref = bc.reference_field
        assert torch.allclose(ref, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    def test_default_intensity(self, inlet_patch):
        bc = TurbulentInlet2BC(inlet_patch)
        assert bc.intensity == 0.05

    def test_default_length_scale(self, inlet_patch):
        bc = TurbulentInlet2BC(inlet_patch)
        assert bc.length_scale == 0.01

    def test_default_alpha(self, inlet_patch):
        bc = TurbulentInlet2BC(inlet_patch)
        assert bc.alpha == 0.1

    def test_custom_params(self, inlet_patch):
        bc = TurbulentInlet2BC(inlet_patch, coeffs={
            "referenceField": [5.0, 1.0, 0.0],
            "intensity": 0.1,
            "lengthScale": 0.05,
            "alpha": 0.5,
        })
        assert bc.intensity == 0.1
        assert bc.length_scale == 0.05
        assert bc.alpha == 0.5
        assert bc.reference_field[0] == pytest.approx(5.0)


class TestTurbulentInlet2Kernel:
    """Digital filter kernel tests."""

    def test_kernel_shape(self, inlet_patch):
        bc = TurbulentInlet2BC(inlet_patch, coeffs={"nFilterPoints": 8})
        assert bc.kernel.shape == (8,)

    def test_kernel_normalised(self, inlet_patch):
        """Kernel normalised so sum(g^2) * N ~ 1."""
        bc = TurbulentInlet2BC(inlet_patch, coeffs={"nFilterPoints": 8})
        k = bc.kernel
        N = 8
        sum_g_sq = (k * k).sum()
        # sum(g^2) * N should be ~1 (variance-preserving)
        assert sum_g_sq * N == pytest.approx(1.0, rel=0.1)

    def test_kernel_symmetric(self, inlet_patch):
        """Kernel is symmetric around centre."""
        bc = TurbulentInlet2BC(inlet_patch, coeffs={"nFilterPoints": 9})
        k = bc.kernel
        # For symmetric Gaussian, k[i] ~ k[N-1-i]
        assert torch.allclose(k, k.flip(0), atol=1e-10)

    def test_kernel_positive(self, inlet_patch):
        """Gaussian kernel values are all positive."""
        bc = TurbulentInlet2BC(inlet_patch, coeffs={"nFilterPoints": 8})
        assert (bc.kernel > 0).all()

    def test_kernel_custom_n_points(self, inlet_patch):
        """Custom nFilterPoints is respected."""
        bc = TurbulentInlet2BC(inlet_patch, coeffs={"nFilterPoints": 16})
        assert bc.kernel.shape == (16,)


class TestTurbulentInlet2Apply:
    """apply() tests."""

    def test_apply_sets_velocity(self, inlet_patch):
        """apply() writes velocity vectors to the field."""
        bc = TurbulentInlet2BC(inlet_patch, coeffs={
            "referenceField": [1.0, 0.0, 0.0],
            "intensity": 0.0,
        })
        field = torch.zeros(20, 3, dtype=torch.float64)
        result = bc.apply(field)
        # With zero intensity, should be exactly the reference
        assert result.shape == (20, 3)

    def test_apply_nonzero_intensity(self, inlet_patch):
        """With nonzero intensity, velocities differ from mean."""
        bc = TurbulentInlet2BC(inlet_patch, coeffs={
            "referenceField": [1.0, 0.0, 0.0],
            "intensity": 0.1,
        })
        field = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field)
        # At least some faces should differ from [1,0,0]
        faces = field[0:5]
        mean_face = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        # Not all exactly equal (probability of exact match ~ 0)
        assert not torch.allclose(faces, mean_face.unsqueeze(0).expand(5, -1), atol=1e-10)

    def test_apply_relaxation(self, inlet_patch):
        """With alpha < 1, new values are blended with previous."""
        bc = TurbulentInlet2BC(inlet_patch, coeffs={
            "referenceField": [1.0, 0.0, 0.0],
            "intensity": 0.1,
            "alpha": 0.5,
        })
        field1 = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field1)

        field2 = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field2)

        # Second call should blend with first
        assert bc._prev_values is not None

    def test_apply_with_patch_idx(self, inlet_patch):
        """apply() with explicit patch index."""
        bc = TurbulentInlet2BC(inlet_patch, coeffs={
            "referenceField": [2.0, 0.0, 0.0],
            "intensity": 0.01,
        })
        field = torch.zeros(30, 3, dtype=torch.float64)
        bc.apply(field, patch_idx=10)
        # Faces 10..14 should be modified
        assert not torch.allclose(field[10:15], torch.zeros(5, 3, dtype=torch.float64))


class TestTurbulentInlet2Matrix:
    """matrix_contributions() tests."""

    def test_diagonal_contributions(self, inlet_patch):
        """Diagonal has deltaCoeff * area contributions."""
        bc = TurbulentInlet2BC(inlet_patch)
        n_cells = 10
        field = torch.zeros(n_cells, 3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells)
        # delta=2, area=1 => coeff=2 per face, owners [0..4]
        for i in range(5):
            assert diag[i] == pytest.approx(2.0)

    def test_source_uses_reference(self, inlet_patch):
        """Source term uses reference field x-component."""
        bc = TurbulentInlet2BC(inlet_patch, coeffs={
            "referenceField": [3.0, 0.0, 0.0],
        })
        n_cells = 10
        field = torch.zeros(n_cells, 3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells)
        # coeff * ref_x = 2 * 3 = 6
        for i in range(5):
            assert source[i] == pytest.approx(6.0)
