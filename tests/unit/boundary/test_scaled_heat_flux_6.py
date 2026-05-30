"""Tests for v6 enhanced scaled heat flux boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.scaled_heat_flux_6 import ScaledHeatFlux6BC


class TestScaledHeatFlux6BC:
    """Test the scaledHeatFlux6 boundary condition."""

    def test_registration(self):
        assert "scaledHeatFlux6" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "scaledHeatFlux6", simple_patch,
            {"scale": 2.0, "q_ref": 500.0},
        )
        assert isinstance(bc, ScaledHeatFlux6BC)

    def test_type_name(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch)
        assert bc.type_name == "scaledHeatFlux6"

    def test_default_properties(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch)
        assert bc.scale == pytest.approx(1.0)
        assert bc.q_ref == pytest.approx(0.0)
        assert bc.k == pytest.approx(0.025)
        assert bc.density == pytest.approx(1000.0)
        assert bc.Cp == pytest.approx(4186.0)
        assert bc.thickness == pytest.approx(0.001)
        assert bc.thermal_inertia_coeff == pytest.approx(0.0)
        assert bc.spatial_decay == pytest.approx(1.0)
        assert bc.spatial_scale == pytest.approx(1.0)

    def test_custom_properties(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch, {
            "density": 2000.0, "Cp": 5000.0, "thickness": 0.005,
            "thermalInertiaCoeff": 0.1, "spatialDecay": 2.0, "spatialScale": 0.5,
        })
        assert bc.density == pytest.approx(2000.0)
        assert bc.Cp == pytest.approx(5000.0)
        assert bc.thickness == pytest.approx(0.005)
        assert bc.thermal_inertia_coeff == pytest.approx(0.1)
        assert bc.spatial_decay == pytest.approx(2.0)

    def test_apply_basic(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch, {"scale": 2.0, "q_ref": 500.0, "k": 0.025})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_temperature(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch, {
            "scale": 2.0, "q_ref": 500.0, "k": 0.025, "alphaT": 0.001,
        })
        field = torch.zeros(15, dtype=torch.float64)
        T_field = torch.tensor([350.0, 400.0, 300.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_radiation(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 0.025,
            "epsilonSigma": 0.9, "Tamb": 300.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        T_field = torch.tensor([350.0, 400.0, 500.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_conjugate_coupling(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 0.025,
            "hConv": 100.0, "Tfluid": 300.0, "blendCoeff": 0.5,
        })
        field = torch.zeros(15, dtype=torch.float64)
        T_field = torch.tensor([350.0, 350.0, 350.0], dtype=torch.float64)
        T_interior = torch.tensor([400.0, 400.0, 400.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field, T_interior=T_interior)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_thermal_inertia(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 0.025,
            "thermalInertiaCoeff": 0.1, "density": 1000.0, "Cp": 4186.0, "thickness": 0.001,
        })
        field = torch.zeros(15, dtype=torch.float64)
        T_field = torch.tensor([350.0, 350.0, 350.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field, dt=1e-3)
        assert torch.all(torch.isfinite(field[10:13]))
        # Second call with different temperature to activate dT/dt
        T_field2 = torch.tensor([360.0, 360.0, 360.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field2, dt=1e-3)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_contact_resistance(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 0.025,
            "contactResistance": 0.01, "contactCoeff": 0.5,
        })
        field = torch.zeros(15, dtype=torch.float64)
        T_field = torch.tensor([350.0, 350.0, 350.0], dtype=torch.float64)
        T_interior = torch.tensor([400.0, 400.0, 400.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field, T_interior=T_interior)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_without_args(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch, {"scale": 1.0, "q_ref": 500.0})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch, {"scale": 1.0, "q_ref": 500.0})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.isfinite(field[5])

    def test_matrix_contributions(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch, {"scale": 1.0, "q_ref": 500.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.all(source != 0)

    def test_gradient_property(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch, {"scale": 2.0, "q_ref": 500.0, "k": 0.025})
        grad = bc.gradient
        assert isinstance(grad, float)
        assert grad != 0.0

    def test_q_property(self, simple_patch):
        bc = ScaledHeatFlux6BC(simple_patch, {
            "scale": 2.0, "q_ref": 500.0, "timeModulation": 0.5, "spatialWeight": 0.8,
        })
        assert bc.q == pytest.approx(2.0 * 500.0 * 0.5 * 0.8)
