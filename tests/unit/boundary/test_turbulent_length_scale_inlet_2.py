"""Tests for enhanced turbulent length scale inlet boundary condition (v2)."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_length_scale_inlet_2 import TurbulentLengthScaleInlet2BC


class TestTurbulentLengthScaleInlet2BC:
    """Test the turbulentLengthScaleInlet2 boundary condition."""

    def test_registration(self):
        assert "turbulentLengthScaleInlet2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentLengthScaleInlet2", simple_patch,
            {"intensity": 0.05, "lengthScale": 0.01},
        )
        assert isinstance(bc, TurbulentLengthScaleInlet2BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentLengthScaleInlet2BC(simple_patch)
        assert bc.type_name == "turbulentLengthScaleInlet2"

    def test_default_properties(self, simple_patch):
        bc = TurbulentLengthScaleInlet2BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.length_scale_min == pytest.approx(1e-6)
        assert bc.length_scale_fraction == pytest.approx(0.07)
        assert bc.hydraulic_diameter == pytest.approx(0.1)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentLengthScaleInlet2BC(simple_patch, {
            "lengthScaleMin": 1e-8,
            "lengthScaleFraction": 0.1,
            "hydraulicDiameter": 0.5,
        })
        assert bc.length_scale_min == pytest.approx(1e-8)
        assert bc.length_scale_fraction == pytest.approx(0.1)
        assert bc.hydraulic_diameter == pytest.approx(0.5)

    def test_apply_with_k_epsilon(self, simple_patch):
        bc = TurbulentLengthScaleInlet2BC(simple_patch, {
            "Cmu": 0.09,
            "hydraulicDiameter": 10.0,  # l_max = 0.07 * 10 = 0.7
        })
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        epsilon = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)

        # l_mix = C_mu^0.75 * k^1.5 / epsilon = 0.09^0.75 * 0.1^1.5 / 1.0
        expected = (0.09 ** 0.75) * (0.1 ** 1.5) / 1.0
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_clamping_max(self, simple_patch):
        """Length scale should be clamped by hydraulic diameter."""
        bc = TurbulentLengthScaleInlet2BC(simple_patch, {
            "lengthScaleFraction": 0.07,
            "hydraulicDiameter": 0.1,  # l_max = 0.007
            "lengthScaleMin": 1e-6,
        })

        # Very large k / very small epsilon -> large l_mix
        k = torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64)
        epsilon = torch.tensor([1e-10, 1e-10, 1e-10], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)

        # Should be clamped to l_max = 0.07 * 0.1 = 0.007
        assert field[10] <= 0.007 + 1e-10

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentLengthScaleInlet2BC(simple_patch, {
            "lengthScale": 0.005,  # below default l_max = 0.007
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        assert field[10] == pytest.approx(0.005)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentLengthScaleInlet2BC(simple_patch, {
            "hydraulicDiameter": 10.0,  # l_max = 0.7, no clamping
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        assert field[10] > 0

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentLengthScaleInlet2BC(simple_patch, {
            "lengthScale": 0.005,  # below default l_max = 0.007
        })
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5] == pytest.approx(0.005)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentLengthScaleInlet2BC(simple_patch, {"lengthScale": 0.005})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64))
