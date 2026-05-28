"""
Tests for HeatExchangerBC — heat exchanger boundary condition.

Tests cover:
- RTS registration
- Factory creation
- Default and custom parameters
- Multi-zone face distribution
- Zone temperature update (effectiveness-NTU)
- apply() sets face values
- matrix_contributions() diagonal and source
- Single-zone and multi-zone configurations
"""

import pytest
import torch

from pyfoam.boundary.boundary_condition import BoundaryCondition, Patch
from pyfoam.boundary.heat_exchanger_bc import HeatExchangerBC


@pytest.fixture
def heat_exchanger_patch():
    """A 6-face patch for heat exchanger testing."""
    return Patch(
        name="hexPatch",
        face_indices=torch.tensor([20, 21, 22, 23, 24, 25]),
        face_normals=torch.tensor([
            [1.0, 0.0, 0.0]] * 6, dtype=torch.float64,
        ),
        face_areas=torch.tensor([1.0] * 6, dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0] * 6, dtype=torch.float64),
        owner_cells=torch.tensor([0, 1, 2, 3, 4, 5]),
    )


class TestHeatExchangerRegistration:
    """RTS registration tests."""

    def test_registered_in_registry(self):
        """heatExchanger is in the BC registry."""
        assert "heatExchanger" in BoundaryCondition.available_types()

    def test_factory_creation(self, heat_exchanger_patch):
        """Can create via factory method."""
        bc = BoundaryCondition.create(
            "heatExchanger", heat_exchanger_patch,
            coeffs={"h": 100.0, "Treservoir": 350.0},
        )
        assert isinstance(bc, HeatExchangerBC)


class TestHeatExchangerDefaults:
    """Default parameter tests."""

    def test_default_h(self, heat_exchanger_patch):
        bc = HeatExchangerBC(heat_exchanger_patch)
        assert bc.h == 100.0

    def test_default_Treservoir(self, heat_exchanger_patch):
        bc = HeatExchangerBC(heat_exchanger_patch)
        assert bc.T_reservoir == 350.0

    def test_default_effectiveness(self, heat_exchanger_patch):
        bc = HeatExchangerBC(heat_exchanger_patch)
        assert bc.effectiveness == 0.8

    def test_default_Cmin(self, heat_exchanger_patch):
        bc = HeatExchangerBC(heat_exchanger_patch)
        assert bc.Cmin == 500.0

    def test_default_n_zones(self, heat_exchanger_patch):
        bc = HeatExchangerBC(heat_exchanger_patch)
        assert bc.n_zones == 1

    def test_custom_params(self, heat_exchanger_patch):
        bc = HeatExchangerBC(
            heat_exchanger_patch,
            coeffs={"h": 200.0, "Treservoir": 400.0, "effectiveness": 0.9, "nZones": 3},
        )
        assert bc.h == 200.0
        assert bc.T_reservoir == 400.0
        assert bc.effectiveness == 0.9
        assert bc.n_zones == 3


class TestHeatExchangerMultiZone:
    """Multi-zone configuration tests."""

    def test_zone_face_distribution(self, heat_exchanger_patch):
        """Faces are distributed evenly across zones."""
        bc = HeatExchangerBC(heat_exchanger_patch, coeffs={"nZones": 3})
        assert bc.zone_T_out.shape == (3,)
        # 6 faces / 3 zones = 2 faces each
        for z in range(3):
            start, end = bc._get_zone_face_range(z)
            assert end - start == 2

    def test_zone_face_distribution_uneven(self, heat_exchanger_patch):
        """Uneven face distribution when not divisible."""
        bc = HeatExchangerBC(heat_exchanger_patch, coeffs={"nZones": 4})
        # 6 faces / 4 zones: [2, 2, 1, 1]
        counts = bc._zone_face_counts.tolist()
        assert sum(counts) == 6
        assert counts[0] == 2
        assert counts[1] == 2

    def test_zone_T_out_initialised_to_reservoir(self, heat_exchanger_patch):
        """Zone outlet temperatures start at reservoir temperature."""
        bc = HeatExchangerBC(heat_exchanger_patch, coeffs={"Treservoir": 400.0, "nZones": 2})
        assert torch.allclose(bc.zone_T_out, torch.tensor([400.0, 400.0], dtype=torch.float64))


class TestHeatExchangerUpdate:
    """Zone temperature update tests."""

    def test_update_cools_when_wall_hot(self, heat_exchanger_patch):
        """When wall is hotter than reservoir, outlet approaches wall."""
        bc = HeatExchangerBC(heat_exchanger_patch, coeffs={
            "Treservoir": 300.0, "effectiveness": 0.8, "Cmin": 100.0,
        })
        T_wall = torch.full((6,), 350.0, dtype=torch.float64)
        bc.update_zone_temperatures(T_wall)
        # Q = 0.8 * 100 * (300 - 350) = -4000
        # T_out = 300 - (-4000)/100 = 340
        assert bc.zone_T_out[0] == pytest.approx(340.0)

    def test_update_heats_when_wall_cold(self, heat_exchanger_patch):
        """When wall is colder than reservoir, outlet is cooled."""
        bc = HeatExchangerBC(heat_exchanger_patch, coeffs={
            "Treservoir": 400.0, "effectiveness": 0.5, "Cmin": 200.0,
        })
        T_wall = torch.full((6,), 300.0, dtype=torch.float64)
        bc.update_zone_temperatures(T_wall)
        # Q = 0.5 * 200 * (400 - 300) = 10000
        # T_out = 400 - 10000/200 = 350
        assert bc.zone_T_out[0] == pytest.approx(350.0)

    def test_multi_zone_update(self, heat_exchanger_patch):
        """Different zones can have different wall temperatures."""
        bc = HeatExchangerBC(heat_exchanger_patch, coeffs={
            "Treservoir": 300.0, "effectiveness": 1.0, "Cmin": 100.0, "nZones": 2,
        })
        # Zone 0 (faces 0-2): wall at 300, Zone 1 (faces 3-5): wall at 200
        T_wall = torch.tensor([300.0, 300.0, 300.0, 200.0, 200.0, 200.0], dtype=torch.float64)
        bc.update_zone_temperatures(T_wall)
        # Zone 0: Q = 1.0 * 100 * (300 - 300) = 0, T_out = 300
        # Zone 1: Q = 1.0 * 100 * (300 - 200) = 10000, T_out = 300 - 10000/100 = 200
        assert bc.zone_T_out[0] == pytest.approx(300.0)
        assert bc.zone_T_out[1] == pytest.approx(200.0)


class TestHeatExchangerApply:
    """apply() tests."""

    def test_apply_sets_face_values(self, heat_exchanger_patch):
        """apply() writes zone temperatures to face field."""
        bc = HeatExchangerBC(heat_exchanger_patch, coeffs={"Treservoir": 350.0})
        field = torch.zeros(30, dtype=torch.float64)
        bc.apply(field)
        # All 6 faces should be 350 (reservoir temp)
        assert torch.allclose(field[20:26], torch.full((6,), 350.0, dtype=torch.float64))

    def test_apply_multi_zone(self, heat_exchanger_patch):
        """apply() sets per-zone temperatures correctly."""
        bc = HeatExchangerBC(heat_exchanger_patch, coeffs={
            "Treservoir": 300.0, "nZones": 3,
        })
        # Manually set zone temperatures
        bc._zone_T_out = torch.tensor([310.0, 320.0, 330.0], dtype=torch.float64)
        field = torch.zeros(30, dtype=torch.float64)
        bc.apply(field)
        # Zone 0: faces [20, 21], Zone 1: faces [22, 23], Zone 2: faces [24, 25]
        assert torch.allclose(field[20:22], torch.tensor([310.0, 310.0], dtype=torch.float64))
        assert torch.allclose(field[22:24], torch.tensor([320.0, 320.0], dtype=torch.float64))
        assert torch.allclose(field[24:26], torch.tensor([330.0, 330.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, heat_exchanger_patch):
        """apply() with explicit patch index."""
        bc = HeatExchangerBC(heat_exchanger_patch, coeffs={"Treservoir": 400.0})
        field = torch.zeros(30, dtype=torch.float64)
        bc.apply(field, patch_idx=10)
        assert torch.allclose(field[10:16], torch.full((6,), 400.0, dtype=torch.float64))


class TestHeatExchangerMatrix:
    """matrix_contributions() tests."""

    def test_matrix_diagonal(self, heat_exchanger_patch):
        """Diagonal has h*A contributions."""
        bc = HeatExchangerBC(heat_exchanger_patch, coeffs={"h": 100.0})
        n_cells = 10
        field = torch.zeros(n_cells, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells)
        # 6 faces, each with A=1, h=100 => each owner gets +100
        # owners [0,1,2,3,4,5] each once => diag[i] = 100 for i in 0..5
        for i in range(6):
            assert diag[i] == pytest.approx(100.0)
        for i in range(6, 10):
            assert diag[i] == pytest.approx(0.0)

    def test_matrix_source(self, heat_exchanger_patch):
        """Source has h*A*T_HEX contributions."""
        bc = HeatExchangerBC(heat_exchanger_patch, coeffs={"h": 50.0, "Treservoir": 400.0})
        n_cells = 10
        field = torch.zeros(n_cells, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells)
        # h*A*T_HEX = 50 * 1 * 400 = 20000 per face
        for i in range(6):
            assert source[i] == pytest.approx(20000.0)

    def test_matrix_accumulates(self, heat_exchanger_patch):
        """Multiple calls accumulate contributions."""
        bc = HeatExchangerBC(heat_exchanger_patch, coeffs={"h": 10.0, "Treservoir": 300.0})
        n_cells = 10
        field = torch.zeros(n_cells, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells)
        diag2, source2 = bc.matrix_contributions(field, n_cells, diag=diag, source=source)
        # Should be double
        assert diag2[0] == pytest.approx(20.0)
        assert source2[0] == pytest.approx(6000.0)
