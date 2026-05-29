"""Tests for interfacial heat transfer boundary condition.

Tests cover:
- RTS registration
- Factory creation
- Property access
- Interfacial area factor
- Heat transfer rate computation
- apply() and matrix_contributions
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.interfacial_heat_transfer import InterfacialHeatTransferBC


class TestInterfacialHeatTransferBC:
    """interfacialHeatTransfer boundary condition tests."""

    def test_registration(self):
        assert "interfacialHeatTransfer" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = InterfacialHeatTransferBC(simple_patch)
        assert bc.type_name == "interfacialHeatTransfer"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "interfacialHeatTransfer", simple_patch,
            {"hi": 5000.0, "Tsat": 373.15},
        )
        assert isinstance(bc, InterfacialHeatTransferBC)

    def test_default_coefficients(self, simple_patch):
        bc = InterfacialHeatTransferBC(simple_patch)
        assert bc.hi == pytest.approx(10000.0)
        assert bc.Tsat == pytest.approx(373.15)
        assert bc.L == pytest.approx(2.26e6)
        assert bc.alphaMax == pytest.approx(0.8)

    def test_custom_coefficients(self, simple_patch):
        bc = InterfacialHeatTransferBC(
            simple_patch,
            {"hi": 5000.0, "Tsat": 400.0, "L": 1e6, "alphaMax": 0.5},
        )
        assert bc.hi == pytest.approx(5000.0)
        assert bc.Tsat == pytest.approx(400.0)
        assert bc.L == pytest.approx(1e6)
        assert bc.alphaMax == pytest.approx(0.5)

    def test_interfacial_area_factor_below_max(self, simple_patch):
        bc = InterfacialHeatTransferBC(simple_patch, {"alphaMax": 0.8})
        alpha = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
        A_i = bc.interfacial_area_factor(alpha)
        assert torch.allclose(A_i, alpha)

    def test_interfacial_area_factor_above_max(self, simple_patch):
        bc = InterfacialHeatTransferBC(simple_patch, {"alphaMax": 0.8})
        alpha = torch.tensor([0.9, 0.95, 1.0], dtype=torch.float64)
        A_i = bc.interfacial_area_factor(alpha)
        assert torch.allclose(A_i, torch.full((3,), 0.8, dtype=torch.float64))

    def test_heat_transfer_rate_condensation(self, simple_patch):
        """T < Tsat: heating (positive Q)."""
        bc = InterfacialHeatTransferBC(
            simple_patch, {"hi": 10000.0, "Tsat": 373.15},
        )
        T = torch.tensor([350.0, 360.0, 370.0], dtype=torch.float64)
        alpha = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        Q = bc.heat_transfer_rate(T, alpha)
        assert (Q > 0).all()  # Positive = heating

    def test_heat_transfer_rate_evaporation(self, simple_patch):
        """T > Tsat: cooling (negative Q)."""
        bc = InterfacialHeatTransferBC(
            simple_patch, {"hi": 10000.0, "Tsat": 373.15},
        )
        T = torch.tensor([380.0, 390.0, 400.0], dtype=torch.float64)
        alpha = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        Q = bc.heat_transfer_rate(T, alpha)
        assert (Q < 0).all()

    def test_heat_transfer_rate_at_saturation(self, simple_patch):
        """T == Tsat: zero heat transfer."""
        bc = InterfacialHeatTransferBC(
            simple_patch, {"hi": 10000.0, "Tsat": 373.15},
        )
        T = torch.full((3,), 373.15, dtype=torch.float64)
        alpha = torch.full((3,), 0.3, dtype=torch.float64)
        Q = bc.heat_transfer_rate(T, alpha)
        assert torch.allclose(Q, torch.zeros(3, dtype=torch.float64), atol=1e-10)

    def test_apply_sets_zero_gradient(self, simple_patch):
        bc = InterfacialHeatTransferBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 350.0
        field[1] = 360.0
        field[2] = 370.0
        field = bc.apply(field)
        assert field[10] == pytest.approx(350.0)
        assert field[11] == pytest.approx(360.0)
        assert field[12] == pytest.approx(370.0)

    def test_matrix_contributions_shape(self, simple_patch):
        bc = InterfacialHeatTransferBC(simple_patch)
        field = torch.full((15,), 360.0, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, alpha=0.2)
        assert diag.shape == (3,)
        assert source.shape == (3,)

    def test_matrix_contributions_robin_type(self, simple_patch):
        """Robin-type linearisation: diag and source are both positive."""
        bc = InterfacialHeatTransferBC(
            simple_patch, {"hi": 10000.0, "Tsat": 373.15},
        )
        field = torch.full((15,), 360.0, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, alpha=0.3)
        assert (diag > 0).all()
        assert (source > 0).all()
