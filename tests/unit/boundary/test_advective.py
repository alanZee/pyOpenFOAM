"""Tests for advective boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.advective import AdvectiveBC


class TestAdvectiveBC:
    """Test the advective boundary condition."""

    def test_registration(self):
        assert "advective" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("advective", simple_patch)
        assert isinstance(bc, AdvectiveBC)

    def test_type_name(self, simple_patch):
        bc = AdvectiveBC(simple_patch)
        assert bc.type_name == "advective"

    def test_apply_no_phi_zero_gradient(self, simple_patch):
        """Without flux info, behaves as zero-gradient (copy owner values)."""
        bc = AdvectiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)

        assert field[10].item() == pytest.approx(10.0)
        assert field[11].item() == pytest.approx(20.0)
        assert field[12].item() == pytest.approx(30.0)

    def test_apply_with_phi_outflow(self, simple_patch):
        """With positive flux (outflow), copies owner values."""
        bc = AdvectiveBC(simple_patch)
        phi = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 10.0
        field[2] = 15.0
        bc.apply(field, phi=phi, dt=0.1)

        assert field[10].item() == pytest.approx(5.0)
        assert field[11].item() == pytest.approx(10.0)
        assert field[12].item() == pytest.approx(15.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = AdvectiveBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 7.0
        field[1] = 8.0
        field[2] = 9.0
        bc.apply(field, patch_idx=5)

        assert field[5].item() == pytest.approx(7.0)
        assert field[6].item() == pytest.approx(8.0)
        assert field[7].item() == pytest.approx(9.0)

    def test_matrix_contributions_no_phi(self, simple_patch):
        """Without flux, zero matrix contribution."""
        bc = AdvectiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)

        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_matrix_contributions_with_outflow_phi(self, simple_patch):
        """Outflow flux contributes to source term."""
        bc = AdvectiveBC(simple_patch)
        phi = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, phi=phi)

        # source += phi for outflow faces
        assert torch.allclose(source, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    def test_matrix_contributions_with_inflow_phi(self, simple_patch):
        """Influx (negative phi) does not contribute."""
        bc = AdvectiveBC(simple_patch)
        phi = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, phi=phi)

        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))
