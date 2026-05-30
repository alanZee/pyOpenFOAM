"""Tests for EnhancedStressSolver7 -- v7 enhanced stress solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced_6 import EnhancedStressSolver6
from pyfoam.structural.stress_solver_enhanced_7 import (
    EnhancedStressSolver7,
    XFEMResult,
    ThermalStressResult,
    HomogenisationResult,
)


class TestXFEMResult:
    """Test XFEMResult dataclass."""

    def test_defaults(self):
        result = XFEMResult()
        assert result.crack_tip_stress == 0.0
        assert result.enrichment_dof == 0


class TestThermalStressResult:
    """Test ThermalStressResult dataclass."""

    def test_defaults(self):
        result = ThermalStressResult()
        assert result.max_principal == 0.0
        assert result.temperature_gradient_effect == 0.0


class TestHomogenisationResult:
    """Test HomogenisationResult dataclass."""

    def test_defaults(self):
        result = HomogenisationResult()
        assert result.volume_fraction == 0.0
        assert result.rve_error == 0.0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v6(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver7(model)
        assert isinstance(solver, EnhancedStressSolver6)


class TestLevelSet:
    """Test level set computation."""

    def test_level_set_above_crack(self):
        coords = torch.tensor([[0.0, 0.1], [1.0, 0.1]], dtype=torch.float64)
        tip = torch.tensor([0.5, 0.0], dtype=torch.float64)
        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
        phi = EnhancedStressSolver7.compute_level_set(coords, tip, direction)
        assert phi[0].item() > 0  # Above crack
        assert phi[1].item() > 0


class TestXFEMStress:
    """Test XFEM stress analysis."""

    def test_returns_xfem_result(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver7(model)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        crack_nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        result = solver.xfem_stress_analysis(strain, crack_nodes)
        assert isinstance(result, XFEMResult)
        assert result.enriched_stress.numel() == 6


class TestCoupledThermalStress:
    """Test coupled thermal stress analysis."""

    def test_thermal_stress_at_reference(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver7(model, thermal_expansion=12e-6)
        strain = torch.zeros(6, dtype=torch.float64)
        result = solver.coupled_thermal_stress(strain, 293.15)
        # At reference temperature, thermal stress should be zero
        assert result.thermal_stress.norm().item() < 1e-6

    def test_thermal_stress_at_elevated_temp(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver7(model, thermal_expansion=12e-6)
        strain = torch.zeros(6, dtype=torch.float64)
        result = solver.coupled_thermal_stress(strain, 393.15)
        # Should have non-zero thermal stress
        assert result.thermal_stress.norm().item() > 0

    def test_returns_correct_type(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver7(model)
        strain = torch.zeros(6, dtype=torch.float64)
        result = solver.coupled_thermal_stress(strain, 500.0)
        assert isinstance(result, ThermalStressResult)


class TestHomogenisation:
    """Test multi-scale homogenisation."""

    def test_rule_of_mixtures(self):
        C_matrix = torch.eye(6, dtype=torch.float64) * 10e9
        C_fibre = torch.eye(6, dtype=torch.float64) * 200e9
        C_eff = EnhancedStressSolver7.rule_of_mixtures_stiffness(
            C_matrix, C_fibre, 0.3
        )
        expected_diag = 0.7 * 10e9 + 0.3 * 200e9
        assert C_eff[0, 0].item() == pytest.approx(expected_diag)

    def test_zero_volume_fraction(self):
        C_matrix = torch.eye(6, dtype=torch.float64) * 10e9
        C_fibre = torch.eye(6, dtype=torch.float64) * 200e9
        C_eff = EnhancedStressSolver7.rule_of_mixtures_stiffness(
            C_matrix, C_fibre, 0.0
        )
        assert torch.allclose(C_eff, C_matrix)

    def test_homogenise_returns_result(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver7(model)
        C_m = torch.eye(6, dtype=torch.float64) * 10e9
        C_f = torch.eye(6, dtype=torch.float64) * 200e9
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.homogenise(C_m, C_f, 0.3, strain)
        assert isinstance(result, HomogenisationResult)
        assert result.effective_stiffness.shape == (6, 6)


class TestRepr:
    """Test string representations."""

    def test_repr(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver7(model)
        r = repr(solver)
        assert "EnhancedStressSolver7" in r
