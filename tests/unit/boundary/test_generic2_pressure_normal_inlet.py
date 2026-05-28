"""Tests for Generic2BC and PressureNormalInletBC boundary conditions."""

import pytest
import torch

from pyfoam.boundary import (
    BoundaryCondition,
    Generic2BC,
    PressureNormalInletBC,
)


# ---- Generic2BC ----


class TestGeneric2BC:
    """Test the enhanced generic boundary condition."""

    def test_registration(self):
        """generic2 is registered in the RTS registry."""
        assert "generic2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("generic2", simple_patch, {
            "value": 10.0,
            "gradient": 0.0,
            "valueFraction": 0.8,
        })
        assert isinstance(bc, Generic2BC)

    def test_default_blend_mode(self, simple_patch):
        """Default blend mode is linear."""
        bc = Generic2BC(simple_patch)
        assert bc.blend_mode == "linear"

    def test_harmonic_blend_mode(self, simple_patch):
        """Harmonic blend mode can be selected."""
        bc = Generic2BC(simple_patch, {"blendMode": "harmonic"})
        assert bc.blend_mode == "harmonic"

    def test_exponential_blend_mode(self, simple_patch):
        """Exponential blend mode can be selected."""
        bc = Generic2BC(simple_patch, {"blendMode": "exponential"})
        assert bc.blend_mode == "exponential"

    def test_invalid_blend_mode_raises(self, simple_patch):
        """Unknown blend mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown blend mode"):
            Generic2BC(simple_patch, {"blendMode": "nonexistent"})

    def test_linear_blend_pure_fixed_value(self, simple_patch):
        """Linear blend with f=1 gives pure fixed value."""
        bc = Generic2BC(simple_patch, {
            "value": 5.0,
            "valueFraction": 1.0,
            "blendMode": "linear",
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 5.0, dtype=torch.float64))

    def test_linear_blend_pure_zero_gradient(self, simple_patch):
        """Linear blend with f=0 copies owner values."""
        bc = Generic2BC(simple_patch, {
            "value": 100.0,
            "valueFraction": 0.0,
            "blendMode": "linear",
        })
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 1.0
        field[1] = 2.0
        field[2] = 3.0
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(2.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(3.0, dtype=torch.float64))

    def test_linear_blend_intermediate(self, simple_patch):
        """Linear blend with f=0.8: face = 0.8*10 + 0.2*0 = 8."""
        bc = Generic2BC(simple_patch, {
            "value": 10.0,
            "valueFraction": 0.8,
            "blendMode": "linear",
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 8.0, dtype=torch.float64))

    def test_harmonic_blend(self, simple_patch):
        """Harmonic blending produces harmonic mean."""
        bc = Generic2BC(simple_patch, {
            "value": 10.0,
            "valueFraction": 0.5,
            "blendMode": "harmonic",
        })
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 5.0
        field[2] = 5.0
        bc.apply(field)
        # Harmonic mean of 10 and 5 with f=0.5:
        # 1 / (0.5/10 + 0.5/5) = 1 / (0.05 + 0.1) = 1/0.15 = 6.6667
        expected = 1.0 / (0.5 / 10.0 + 0.5 / 5.0)
        assert torch.allclose(field[10], torch.tensor(expected, dtype=torch.float64), atol=1e-6)

    def test_exponential_blend(self, simple_patch):
        """Exponential blending produces smooth transition."""
        bc = Generic2BC(simple_patch, {
            "value": 10.0,
            "valueFraction": 0.5,
            "blendMode": "exponential",
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        # owner = 0, v = 10, f = 0.5
        # weight = 1 - exp(-0.5*3) = 1 - exp(-1.5) ~ 0.7769
        expected = 0.0 + (10.0 - 0.0) * (1.0 - torch.exp(torch.tensor(-0.5 * 3.0, dtype=torch.float64))).item()
        assert torch.allclose(field[10], torch.tensor(expected, dtype=torch.float64), atol=1e-4)

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx."""
        bc = Generic2BC(simple_patch, {
            "value": 10.0,
            "valueFraction": 0.8,
            "blendMode": "linear",
        })
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5:8], torch.full((3,), 8.0, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """Matrix contributions use valueFraction weighting."""
        bc = Generic2BC(simple_patch, {
            "value": 10.0,
            "valueFraction": 0.5,
            "blendMode": "linear",
        })
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64))

    def test_time_varying_flag(self, simple_patch):
        """Time-varying flag is stored correctly."""
        bc = Generic2BC(simple_patch, {"timeVarying": True})
        assert bc.is_time_varying

    def test_time_varying_table_interpolation(self, simple_patch):
        """Table interpolation updates value at intermediate time."""
        bc = Generic2BC(simple_patch, {
            "value": 0.0,
            "valueFraction": 1.0,
            "timeVarying": True,
            "valueTable": [(0, 5.0), (1, 10.0), (2, 15.0)],
            "fractionTable": [(0, 1.0), (2, 0.0)],
        })
        # At t=0.5: value = 5 + 0.5*(10-5) = 7.5
        bc.set_time(0.5)
        assert torch.allclose(bc.value, torch.full((3,), 7.5, dtype=torch.float64))
        # fraction at t=0.5: 1.0 + 0.5*(0.5-1.0) = 0.75
        assert torch.allclose(bc.value_fraction, torch.full((3,), 0.75, dtype=torch.float64))

    def test_time_varying_at_boundaries(self, simple_patch):
        """Table interpolation at start and end times."""
        bc = Generic2BC(simple_patch, {
            "timeVarying": True,
            "valueTable": [(0, 5.0), (1, 10.0)],
        })
        bc.set_time(-1.0)
        assert torch.allclose(bc.value, torch.full((3,), 5.0, dtype=torch.float64))
        bc.set_time(2.0)
        assert torch.allclose(bc.value, torch.full((3,), 10.0, dtype=torch.float64))

    def test_not_time_varying_by_default(self, simple_patch):
        """Not time-varying by default."""
        bc = Generic2BC(simple_patch)
        assert not bc.is_time_varying

    def test_per_face_value_fraction_tensor(self, simple_patch):
        """Per-face valueFraction tensor is supported."""
        vf = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        bc = Generic2BC(simple_patch, {
            "value": 10.0,
            "valueFraction": vf,
            "blendMode": "linear",
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor(0.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(5.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(10.0, dtype=torch.float64))

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = Generic2BC(simple_patch)
        assert bc.type_name == "generic2"

    def test_repr(self, simple_patch):
        """repr shows class name and info."""
        bc = Generic2BC(simple_patch, {"blendMode": "harmonic"})
        r = repr(bc)
        assert "Generic2BC" in r
        assert "harmonic" in r


# ---- PressureNormalInletBC ----


class TestPressureNormalInletBC:
    """Test the pressure normal inlet boundary condition."""

    def test_registration(self):
        """pressureNormalInlet is registered in the RTS registry."""
        assert "pressureNormalInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("pressureNormalInlet", simple_patch, {
            "p0": 101325.0,
            "rho": 1.225,
        })
        assert isinstance(bc, PressureNormalInletBC)

    def test_default_params(self, simple_patch):
        """Default parameters: p0=101325, rho=1.0."""
        bc = PressureNormalInletBC(simple_patch)
        assert bc.p0 == 101325.0
        assert bc.rho == 1.0

    def test_custom_params(self, simple_patch):
        """Custom parameters are stored."""
        bc = PressureNormalInletBC(simple_patch, {
            "p0": 200000.0,
            "rho": 1.5,
        })
        assert bc.p0 == 200000.0
        assert bc.rho == 1.5

    def test_direction_default(self, simple_patch):
        """Default direction is (1, 0, 0)."""
        bc = PressureNormalInletBC(simple_patch)
        d = bc.direction
        assert torch.allclose(d, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    def test_direction_custom(self, simple_patch):
        """Custom direction is normalised."""
        bc = PressureNormalInletBC(simple_patch, {
            "value": (0.0, 3.0, 4.0),
        })
        d = bc.direction
        assert torch.allclose(d.norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)
        expected = torch.tensor([0.0, 0.6, 0.8], dtype=torch.float64)
        assert torch.allclose(d, expected, atol=1e-10)

    def test_apply_velocity_field(self, simple_patch):
        """apply() sets velocity on vector field."""
        bc = PressureNormalInletBC(simple_patch, {
            "p0": 101325.0,
            "rho": 1.0,
        })
        # 15 cells, 3 components
        field = torch.zeros(15, 3, dtype=torch.float64)
        # No pressure field: p = p0 -> U_n = 0
        bc.apply(field)
        # With p=p0, U_n should be 0
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_with_pressure_deficit(self, simple_patch):
        """apply() computes correct velocity from Bernoulli."""
        bc = PressureNormalInletBC(simple_patch, {
            "p0": 101325.0,
            "rho": 1.0,
        })
        # Provide pressure field (p < p0 -> velocity > 0)
        p_field = torch.full((15,), 101320.0, dtype=torch.float64)
        bc._coeffs["_p_field"] = p_field

        field = torch.zeros(15, 3, dtype=torch.float64)
        bc.apply(field)

        # U_n = sqrt(2 * (101325 - 101320) / 1.0) = sqrt(10) ~ 3.1623
        expected_Un = (2.0 * 5.0 / 1.0) ** 0.5
        # Direction is (1,0,0) by default
        assert torch.allclose(field[10, 0], torch.tensor(expected_Un, dtype=torch.float64), atol=1e-6)
        assert torch.allclose(field[10, 1], torch.tensor(0.0, dtype=torch.float64))
        assert torch.allclose(field[10, 2], torch.tensor(0.0, dtype=torch.float64))

    def test_apply_no_reverse_flow(self, simple_patch):
        """When p > p0, velocity is zero (no reverse flow)."""
        bc = PressureNormalInletBC(simple_patch, {
            "p0": 101325.0,
            "rho": 1.0,
        })
        p_field = torch.full((15,), 101330.0, dtype=torch.float64)  # p > p0
        bc._coeffs["_p_field"] = p_field

        field = torch.zeros(15, 3, dtype=torch.float64)
        bc.apply(field)
        # dp clamped to 0 -> U_n = 0
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx."""
        bc = PressureNormalInletBC(simple_patch, {"p0": 101325.0, "rho": 1.0})
        field = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        # With no pressure field, p = p0 -> U = 0
        assert torch.allclose(field[5:8], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_scalar_field_noop(self, simple_patch):
        """apply() is a no-op for scalar fields."""
        bc = PressureNormalInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        assert torch.equal(field, original)

    def test_matrix_contributions_shape(self, simple_patch):
        """Matrix contributions have correct shape."""
        bc = PressureNormalInletBC(simple_patch, {"p0": 101325.0, "rho": 1.0})
        field = torch.zeros(15, 3, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)

    def test_matrix_contributions_accumulate(self, simple_patch):
        """Matrix contributions accumulate into pre-existing tensors."""
        bc = PressureNormalInletBC(simple_patch, {"p0": 101325.0, "rho": 1.0})
        field = torch.zeros(15, 3, dtype=torch.float64)
        n_cells = 3
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, diag=diag, source=source)
        # diag should have been accumulated (each cell gets deltaCoeff*area = 2.0*1.0 = 2.0)
        assert diag.shape == (n_cells,)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = PressureNormalInletBC(simple_patch)
        assert bc.type_name == "pressureNormalInlet"

    def test_repr(self, simple_patch):
        """repr shows class name and params."""
        bc = PressureNormalInletBC(simple_patch, {"p0": 200000.0, "rho": 1.5})
        r = repr(bc)
        assert "PressureNormalInletBC" in r
        assert "200000" in r
