"""Tests for stress solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver import StressSolver


class TestStressSolver:
    """Test the stress solver."""

    def setup_method(self):
        """Set up a steel-like solver for each test."""
        self.model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        self.yield_criterion = VonMisesYield(yield_stress=250e6)
        self.solver = StressSolver(self.model, self.yield_criterion)

    def test_solve_returns_stress(self):
        """solve() returns stress from strain."""
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = self.solver.solve(strain)
        assert stress.shape == (6,)
        # Stress should be nonzero
        assert stress.norm().item() > 0

    def test_solve_consistency(self):
        """solve() result matches model.stress()."""
        strain = torch.tensor([0.001, -0.0003, -0.0003, 0.0001, 0, 0], dtype=torch.float64)
        stress_solver = self.solver.solve(strain)
        stress_model = self.model.stress(strain)
        assert torch.allclose(stress_solver, stress_model)

    def test_solve_full_has_all_keys(self):
        """solve_full() returns all expected diagnostic keys."""
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = self.solver.solve_full(strain)
        assert "stress" in result
        assert "von_mises" in result
        assert "is_yielding" in result
        assert "safety_factor" in result

    def test_solve_full_no_yield(self):
        """solve_full() without yield criterion omits yield keys."""
        solver = StressSolver(self.model)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = solver.solve_full(strain)
        assert "stress" in result
        assert "von_mises" not in result

    def test_principal_stresses(self):
        """Principal stresses computed correctly for uniaxial stress."""
        # Uniaxial stress
        stress = torch.tensor([100e6, 0, 0, 0, 0, 0], dtype=torch.float64)
        principals = self.solver.principal_stresses(stress)
        assert principals.shape == (3,)
        # Largest principal should be ~100e6
        assert abs(principals[0].item() - 100e6) < 1e3

    def test_principal_stresses_hydrostatic(self):
        """Hydrostatic stress: all principal stresses equal."""
        p = 50e6
        stress = torch.tensor([p, p, p, 0, 0, 0], dtype=torch.float64)
        principals = self.solver.principal_stresses(stress)
        assert torch.allclose(principals, torch.full((3,), p, dtype=torch.float64), atol=1e3)

    def test_yielding_detection(self):
        """Solver detects yielding correctly."""
        # Low strain: safe
        strain_safe = torch.tensor([0.0001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result_safe = self.solver.solve_full(strain_safe)
        assert not result_safe["is_yielding"].item()

    def test_repr(self):
        """__repr__ includes model info."""
        r = repr(self.solver)
        assert "StressSolver" in r
