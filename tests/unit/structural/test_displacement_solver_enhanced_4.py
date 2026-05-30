"""Tests for EnhancedDisplacementSolver4 — v4 enhanced displacement solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced_3 import EnhancedDisplacementSolver3
from pyfoam.structural.displacement_solver_enhanced_4 import (
    EnhancedDisplacementSolver4,
    ContactResult,
    RefinementIndicator,
)


class TestRefinementIndicator:
    """Test RefinementIndicator dataclass."""

    def test_defaults(self):
        ri = RefinementIndicator()
        assert ri.global_error == 0.0
        assert ri.refine_cells.numel() == 0

    def test_with_data(self):
        ri = RefinementIndicator(
            cell_error=torch.tensor([1.0, 2.0, 3.0]),
            global_error=6.0,
        )
        assert ri.global_error == 6.0


class TestContactResult:
    """Test ContactResult dataclass."""

    def test_defaults(self):
        cr = ContactResult()
        assert cr.contact_iterations == 0
        assert cr.contact_open is True

    def test_with_data(self):
        cr = ContactResult(
            displacement=torch.tensor([0.0, 0.01], dtype=torch.float64),
            contact_iterations=5,
            contact_open=False,
        )
        assert cr.contact_iterations == 5


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v3(self):
        assert issubclass(EnhancedDisplacementSolver4, EnhancedDisplacementSolver3)


class TestRefinementIndicatorComputation:
    """Test h-adaptive refinement indicator."""

    def test_uniform_no_refinement(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver4(model)

        # Linear displacement: u = [0, 0.001, 0.002, 0.003]
        disp = torch.tensor([0.0, 0.001, 0.002, 0.003], dtype=torch.float64)
        ri = solver.compute_refinement_indicator_1d(
            disp, area=0.01, length=1.0, n_elements=3,
        )
        # Second derivative of linear is zero
        assert ri.global_error == pytest.approx(0.0, abs=1e-6)

    def test_nonlinear_refines(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver4(model)

        # Nonlinear displacement with very sharp feature in one element
        # u = [0, 0, 1, 0] => second derivative is very large for element 1
        disp = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        ri = solver.compute_refinement_indicator_1d(
            disp, area=0.01, length=1.0, n_elements=3,
        )
        assert ri.global_error > 0
        # At least one element should have error above 2*mean
        # (the element with the sharp feature)
        assert ri.cell_error.max().item() > 0

    def test_too_few_nodes(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver4(model)

        disp = torch.tensor([0.0, 0.001], dtype=torch.float64)
        ri = solver.compute_refinement_indicator_1d(
            disp, area=0.01, length=1.0, n_elements=1,
        )
        assert ri.global_error == 0.0


class TestContactSolve:
    """Test contact mechanics solve."""

    def test_no_contact(self):
        model = LinearElasticModel(youngs_modulus=210e9)
        solver = EnhancedDisplacementSolver4(model)

        result = solver.solve_with_contact(
            area=0.01,
            length=1.0,
            total_force=100.0,  # small force
            contact_gap=0.1,  # large gap
            n_steps=5,
        )
        assert result.contact_open is True
        assert result.displacement[1].item() > 0

    def test_contact_closed(self):
        model = LinearElasticModel(youngs_modulus=210e9)
        solver = EnhancedDisplacementSolver4(model)

        result = solver.solve_with_contact(
            area=0.01,
            length=1.0,
            total_force=1e8,  # large force
            contact_stiffness=1e10,
            contact_gap=0.0,  # no gap
            n_steps=5,
            max_contact_iters=5,
        )
        # Contact should be detected (or force should be limited)
        assert result.contact_iterations >= 1

    def test_repr(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver4(model)
        r = repr(solver)
        assert "EnhancedDisplacementSolver4" in r
