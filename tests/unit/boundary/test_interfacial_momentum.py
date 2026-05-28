"""Tests for interfacial momentum transfer BC.

Tests cover InterfacialMomentumBC:
- RTS registration
- Factory creation
- Property access
- apply() and matrix_contributions
- Combined drag + lift + wall lubrication contributions
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.interfacial_momentum import InterfacialMomentumBC


class TestInterfacialMomentumBC:
    """interfacialMomentum boundary condition tests."""

    def test_registration(self):
        """interfacialMomentum is registered in RTS."""
        assert "interfacialMomentum" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = InterfacialMomentumBC(simple_patch)
        assert bc.type_name == "interfacialMomentum"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "interfacialMomentum", simple_patch,
            {"K": 0.5, "CL": 0.1, "Cvm": 0.5},
        )
        assert isinstance(bc, InterfacialMomentumBC)

    def test_default_coefficients(self, simple_patch):
        bc = InterfacialMomentumBC(simple_patch)
        assert bc.K == pytest.approx(0.5)
        assert bc.CL == pytest.approx(0.1)
        assert bc.Cvm == pytest.approx(0.5)
        assert bc.Cw == pytest.approx(0.0)
        assert bc.Dp == pytest.approx(0.003)

    def test_custom_coefficients(self, simple_patch):
        bc = InterfacialMomentumBC(
            simple_patch,
            {"K": 1.0, "CL": 0.2, "Cvm": 1.0, "Cw": 0.05, "Dp": 0.005},
        )
        assert bc.K == pytest.approx(1.0)
        assert bc.CL == pytest.approx(0.2)
        assert bc.Cvm == pytest.approx(1.0)
        assert bc.Cw == pytest.approx(0.05)
        assert bc.Dp == pytest.approx(0.005)

    def test_alpha_name_default(self, simple_patch):
        bc = InterfacialMomentumBC(simple_patch)
        assert bc.alpha_name == "alpha.d"

    def test_rho_name_custom(self, simple_patch):
        bc = InterfacialMomentumBC(
            simple_patch, {"alpha": "alpha.air", "rho": "rho.air"},
        )
        assert bc.alpha_name == "alpha.air"
        assert bc.rho_name == "rho.air"

    def test_apply_sets_zero_gradient(self, simple_patch):
        """apply() sets BC face values to owner cell values."""
        bc = InterfacialMomentumBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0

        field = bc.apply(field)

        assert field[10] == pytest.approx(100.0)
        assert field[11] == pytest.approx(200.0)
        assert field[12] == pytest.approx(300.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = InterfacialMomentumBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 50.0
        field[1] = 60.0
        field[2] = 70.0

        field = bc.apply(field, patch_idx=5)

        assert field[5] == pytest.approx(50.0)
        assert field[6] == pytest.approx(60.0)
        assert field[7] == pytest.approx(70.0)

    def test_matrix_contributions_drag_only(self, simple_patch):
        """Drag-only contributions (CL=0, Cw=0)."""
        bc = InterfacialMomentumBC(
            simple_patch, {"K": 1.0, "CL": 0.0, "Cvm": 0.5, "Cw": 0.0},
        )
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Drag adds to diagonal: diag[c] += K * area = 1.0 * 1.0 = 1.0
        # Each owner cell [0,1,2] gets 1 contribution of K*area
        assert (diag > 0).all()
        # No lift or wall lubrication → source = 0
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_matrix_contributions_with_lift(self, simple_patch):
        """With lift coefficient > 0, source should be positive."""
        bc = InterfacialMomentumBC(
            simple_patch, {"K": 1.0, "CL": 0.1, "Cvm": 0.5, "Cw": 0.0},
        )
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        _, source = bc.matrix_contributions(field, n_cells, alpha=0.2, rho=1000.0)

        # Lift: CL * rho * alpha * area = 0.1 * 1000 * 0.2 * 1.0 = 20.0
        assert (source > 0).all()

    def test_matrix_contributions_with_wall_lubrication(self, simple_patch):
        """With Cw > 0, additional wall lubrication source."""
        bc = InterfacialMomentumBC(
            simple_patch, {"K": 1.0, "CL": 0.0, "Cvm": 0.5, "Cw": 0.05, "Dp": 0.003},
        )
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        _, source = bc.matrix_contributions(field, n_cells, alpha=0.1, rho=1000.0)

        # Wall lubrication: Cw * rho * alpha * area / Dp = 0.05 * 1000 * 0.1 * 1 / 0.003
        assert (source > 0).all()

    def test_matrix_contributions_tensor_alpha(self, simple_patch):
        """Matrix contributions with per-cell tensor alpha."""
        bc = InterfacialMomentumBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        alpha = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells, alpha=alpha)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)

    def test_matrix_contributions_zero_all_coeffs(self, simple_patch):
        """Zero all coefficients → only diagonal drag remains."""
        bc = InterfacialMomentumBC(
            simple_patch, {"K": 0.0, "CL": 0.0, "Cvm": 0.0, "Cw": 0.0},
        )
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        # K=0 → no diagonal contribution
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))
