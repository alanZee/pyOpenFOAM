"""Tests for mappedConvectiveHeatTransfer boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_convective_heat_transfer import MappedConvectiveHeatTransferBC


class TestMappedConvectiveHeatTransferBC:
    """Test the mappedConvectiveHeatTransfer boundary condition."""

    def test_registration(self):
        assert "mappedConvectiveHeatTransfer" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "mappedConvectiveHeatTransfer", simple_patch,
            {"h": 100.0, "Tinf": 350.0, "k": 0.6},
        )
        assert isinstance(bc, MappedConvectiveHeatTransferBC)

    def test_type_name(self, simple_patch):
        bc = MappedConvectiveHeatTransferBC(simple_patch)
        assert bc.type_name == "mappedConvectiveHeatTransfer"

    def test_default_coefficients(self, simple_patch):
        bc = MappedConvectiveHeatTransferBC(simple_patch)
        assert bc.h == pytest.approx(100.0)
        assert bc.Tinf == pytest.approx(300.0)
        assert bc.k == pytest.approx(0.6)
        assert bc.neighbour_region == ""
        assert bc.neighbour_patch == ""

    def test_custom_coefficients(self, simple_patch):
        bc = MappedConvectiveHeatTransferBC(simple_patch, {
            "h": 500.0, "Tinf": 350.0, "k": 45.0,
            "neighbourRegion": "fluid", "neighbourPatch": "interface",
        })
        assert bc.h == pytest.approx(500.0)
        assert bc.Tinf == pytest.approx(350.0)
        assert bc.k == pytest.approx(45.0)
        assert bc.neighbour_region == "fluid"
        assert bc.neighbour_patch == "interface"

    def test_h_setter(self, simple_patch):
        bc = MappedConvectiveHeatTransferBC(simple_patch)
        bc.h = 250.0
        assert bc.h == pytest.approx(250.0)

    def test_Tinf_setter(self, simple_patch):
        bc = MappedConvectiveHeatTransferBC(simple_patch)
        bc.Tinf = 400.0
        assert bc.Tinf == pytest.approx(400.0)

    def test_mapped_T_default_none(self, simple_patch):
        bc = MappedConvectiveHeatTransferBC(simple_patch)
        assert bc.mapped_T is None

    def test_mapped_T_setter(self, simple_patch):
        bc = MappedConvectiveHeatTransferBC(simple_patch)
        T = torch.tensor([310.0, 320.0, 330.0], dtype=torch.float64)
        bc.mapped_T = T
        assert bc.mapped_T is not None
        assert torch.allclose(bc.mapped_T, T)

    def test_apply_without_mapped_T(self, simple_patch):
        """Without mapped T, uses Tinf."""
        bc = MappedConvectiveHeatTransferBC(simple_patch, {"Tinf": 350.0})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        for i in range(3):
            assert field[10 + i].item() == pytest.approx(350.0)

    def test_apply_with_mapped_T(self, simple_patch):
        """With mapped T, uses the mapped values."""
        bc = MappedConvectiveHeatTransferBC(simple_patch, {"Tinf": 350.0})
        mapped = torch.tensor([310.0, 320.0, 330.0], dtype=torch.float64)
        bc.mapped_T = mapped

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        for i in range(3):
            assert field[10 + i].item() == pytest.approx(310.0 + 10.0 * i)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MappedConvectiveHeatTransferBC(simple_patch, {"Tinf": 350.0})
        mapped = torch.tensor([310.0, 320.0, 330.0], dtype=torch.float64)
        bc.mapped_T = mapped

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        for i in range(3):
            assert field[5 + i].item() == pytest.approx(310.0 + 10.0 * i)

    def test_matrix_contributions_without_mapped_T(self, simple_patch):
        """Robin BC: diag += h*A, source += h*A*Tinf."""
        h_val = 100.0
        Tinf = 350.0
        bc = MappedConvectiveHeatTransferBC(simple_patch, {
            "h": h_val, "Tinf": Tinf,
        })

        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)

        assert diag.shape == (3,)
        assert source.shape == (3,)
        # h*A = 100 * 1.0 = 100.0 per face
        expected_diag = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        assert torch.allclose(diag, expected_diag)
        # h*A*Tinf = 100 * 1.0 * 350 = 35000 per face
        expected_source = torch.tensor([35000.0, 35000.0, 35000.0], dtype=torch.float64)
        assert torch.allclose(source, expected_source)

    def test_matrix_contributions_with_mapped_T(self, simple_patch):
        """Robin BC with mapped T: source uses mapped T instead of Tinf."""
        h_val = 200.0
        bc = MappedConvectiveHeatTransferBC(simple_patch, {
            "h": h_val, "Tinf": 300.0,
        })
        mapped = torch.tensor([310.0, 320.0, 330.0], dtype=torch.float64)
        bc.mapped_T = mapped

        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)

        # h*A = 200 * 1.0 = 200 per face
        expected_diag = torch.tensor([200.0, 200.0, 200.0], dtype=torch.float64)
        assert torch.allclose(diag, expected_diag)
        # source = h*A*T_mapped
        expected_source = torch.tensor(
            [200 * 310.0, 200 * 320.0, 200 * 330.0], dtype=torch.float64,
        )
        assert torch.allclose(source, expected_source)
