"""Tests for EnhancedStressSolver5 -- v5 enhanced stress solver."""

import pytest
import torch
import math

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.stress_solver_enhanced_4 import EnhancedStressSolver4
from pyfoam.structural.stress_solver_enhanced_5 import (
    EnhancedStressSolver5,
    FailureAssessment,
    StressInvariants,
)


class TestStressInvariants:
    """Test StressInvariants dataclass."""

    def test_defaults(self):
        inv = StressInvariants()
        assert inv.I1 == 0.0
        assert inv.von_mises == 0.0


class TestFailureAssessment:
    """Test FailureAssessment dataclass."""

    def test_defaults(self):
        fa = FailureAssessment()
        assert fa.is_yielding is False
        assert fa.triaxiality == 0.0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v4(self):
        assert issubclass(EnhancedStressSolver5, EnhancedStressSolver4)


class TestComputeInvariants:
    """Test stress invariant computation."""

    def test_hydrostatic_stress(self):
        """Hydrostatic stress: all normal components equal."""
        stress = torch.tensor([100.0, 100.0, 100.0, 0, 0, 0], dtype=torch.float64)
        inv = EnhancedStressSolver5.compute_invariants(stress)
        assert inv.I1 == pytest.approx(300.0)
        assert inv.J2 == pytest.approx(0.0)
        assert inv.von_mises == pytest.approx(0.0)

    def test_uniaxial_stress(self):
        """Uniaxial tension: sigma_11 = sigma, others = 0."""
        sigma = 200.0
        stress = torch.tensor([sigma, 0, 0, 0, 0, 0], dtype=torch.float64)
        inv = EnhancedStressSolver5.compute_invariants(stress)
        assert inv.I1 == pytest.approx(sigma)
        assert inv.von_mises == pytest.approx(sigma)

    def test_shear_stress(self):
        """Pure shear: only tau_12 nonzero."""
        tau = 50.0
        stress = torch.tensor([0, 0, 0, 0, 0, tau], dtype=torch.float64)
        inv = EnhancedStressSolver5.compute_invariants(stress)
        # Von Mises for pure shear = sqrt(3) * tau
        assert inv.von_mises == pytest.approx(math.sqrt(3.0) * tau)


class TestTriaxialityAndLode:
    """Test triaxiality and Lode angle computation."""

    def test_hydrostatic_triaxiality(self):
        """Hydrostatic stress has triaxiality = 1/sqrt(3) (infinite VM -> 0)."""
        stress = torch.tensor([100.0, 100.0, 100.0, 0, 0, 0], dtype=torch.float64)
        triax = EnhancedStressSolver5.compute_triaxiality(stress)
        # VM = 0, so triaxiality = 0 (division protected)
        assert triax == 0.0

    def test_uniaxial_triaxiality(self):
        """Uniaxial tension: triaxiality = 1/3."""
        stress = torch.tensor([300.0, 0, 0, 0, 0, 0], dtype=torch.float64)
        triax = EnhancedStressSolver5.compute_triaxiality(stress)
        assert triax == pytest.approx(1.0 / 3.0, rel=1e-3)

    def test_lode_angle_range(self):
        """Lode angle should be in [0, pi/3]."""
        stress = torch.tensor([100.0, 50.0, 25.0, 10.0, 5.0, 2.0], dtype=torch.float64)
        lode = EnhancedStressSolver5.compute_lode_angle(stress)
        assert 0.0 <= lode <= math.pi / 3.0 + 1e-10


class TestAssessFailure:
    """Test multi-criteria failure assessment."""

    def test_no_failure_small_strain(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver5(model, yield_stress=250e6)
        strain = torch.tensor([0.0001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.assess_failure(strain, yield_stress=250e6)
        assert isinstance(result, FailureAssessment)
        assert result.von_mises_ratio < 1.0
        assert result.is_yielding is False

    def test_yielding_large_strain(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver5(model, yield_stress=250e6)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.assess_failure(strain, yield_stress=250e6)
        assert result.is_yielding is True
        assert result.von_mises_ratio > 1.0

    def test_assessment_has_triaxiality(self):
        model = LinearElasticModel()
        solver = EnhancedStressSolver5(model)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.assess_failure(strain)
        assert result.triaxiality == pytest.approx(1.0 / 3.0, rel=1e-2)

    def test_repr(self):
        model = LinearElasticModel()
        solver = EnhancedStressSolver5(model)
        r = repr(solver)
        assert "EnhancedStressSolver5" in r
