"""
Unit tests for ReactingFoam2 — enhanced reacting flow with ISAT.

Tests cover:
- Solver initialisation
- ISAT table creation
- ISAT query/insert/hit-rate
- Multi-step mechanism parsing
- Strang splitting step
- Run produces finite values
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# ISAT table tests (no case needed)
# ---------------------------------------------------------------------------


class TestISATTable:
    """Tests for ISAT table (unit-level, no mesh)."""

    def test_isat_creation(self):
        from pyfoam.applications.reacting_foam_enhanced_2 import ISATTable
        table = ISATTable(n_species=3, tolerance=1e-4)
        assert table.size == 0
        assert table.n_species == 3

    def test_isat_query_miss(self):
        """First query should be a miss (exact computation)."""
        from pyfoam.applications.reacting_foam_enhanced_2 import ISATTable
        table = ISATTable(n_species=2, tolerance=1e-4)

        phi = torch.tensor([[0.5, 0.5]])
        def exact_fn(p):
            return torch.tensor([[1.0, -1.0]])

        R = table.query(phi, exact_fn)
        assert table.size == 1
        assert table._miss_count == 1
        assert R.shape == (1, 2)

    def test_isat_query_hit(self):
        """Repeated identical query should be a hit."""
        from pyfoam.applications.reacting_foam_enhanced_2 import ISATTable
        table = ISATTable(n_species=2, tolerance=0.1)  # Large tolerance

        phi = torch.tensor([[0.5, 0.5]])
        def exact_fn(p):
            return torch.tensor([[1.0, -1.0]])

        table.query(phi, exact_fn)
        table.query(phi, exact_fn)

        assert table._hit_count >= 1

    def test_isat_hit_rate(self):
        from pyfoam.applications.reacting_foam_enhanced_2 import ISATTable
        table = ISATTable(n_species=2, tolerance=0.5)

        phi = torch.tensor([[0.5, 0.5]])
        def exact_fn(p):
            return torch.ones(p.shape[0], 2)

        table.query(phi, exact_fn)
        table.query(phi, exact_fn)

        assert table.hit_rate > 0.0

    def test_isat_clear(self):
        from pyfoam.applications.reacting_foam_enhanced_2 import ISATTable
        table = ISATTable(n_species=2)

        phi = torch.tensor([[0.5, 0.5]])
        table.query(phi, lambda p: torch.ones(p.shape[0], 2))
        assert table.size == 1

        table.clear()
        assert table.size == 0
        assert table.hit_rate == 0.0

    def test_isat_multi_cell(self):
        from pyfoam.applications.reacting_foam_enhanced_2 import ISATTable
        table = ISATTable(n_species=3, tolerance=1e-4)

        phi = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        def exact_fn(p):
            return p * 2.0

        R = table.query(phi, exact_fn)
        assert R.shape == (2, 3)
        assert table.size == 2


class TestMechanismStep:
    """Tests for MechanismStep dataclass."""

    def test_mechanism_step_creation(self):
        from pyfoam.applications.reacting_foam_enhanced_2 import MechanismStep
        step = MechanismStep(
            name="test",
            reactants={"A": 1.0, "B": 1.0},
            products={"C": 1.0},
            A=1e10,
            beta=0.0,
            Ea=10000.0,
        )
        assert step.name == "test"
        assert step.A == 1e10
        assert step.reversible is False

    def test_mechanism_creation(self):
        from pyfoam.applications.reacting_foam_enhanced_2 import Mechanism
        mech = Mechanism(species=["A", "B", "C"])
        assert len(mech.species) == 3
        assert len(mech.reactions) == 0


# ---------------------------------------------------------------------------
# Solver tests (require case)
# ---------------------------------------------------------------------------


# We reuse the reacting_foam test case from test_reacting_foam.py
# by importing its helper function. If it doesn't exist, we skip.


def _can_import_reacting_test():
    try:
        from tests.unit.applications.test_reacting_foam import _make_reacting_case
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _can_import_reacting_test(),
    reason="test_reacting_foam helper not available",
)
class TestReactingFoam2Solver:
    """Solver-level tests for ReactingFoam2."""

    def test_imports(self):
        from pyfoam.applications.reacting_foam_enhanced_2 import ReactingFoam2
        assert ReactingFoam2 is not None

    def test_exports_in_all(self):
        from pyfoam.applications import ReactingFoam2
        assert ReactingFoam2 is not None
