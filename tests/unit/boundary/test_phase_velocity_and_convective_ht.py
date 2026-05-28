"""Tests for phaseMeanVelocity and convectiveHeatTransfer boundary conditions."""

from __future__ import annotations

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.boundary_condition import Patch
from pyfoam.core.dtype import CFD_DTYPE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def inlet_patch() -> Patch:
    """An inlet patch with 3 faces."""
    return Patch(
        name="inlet",
        face_indices=torch.tensor([0, 1, 2]),
        face_normals=torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1, 2]),
    )


@pytest.fixture
def wall_patch() -> Patch:
    """A wall patch with 3 faces."""
    return Patch(
        name="wall",
        face_indices=torch.tensor([30, 31, 32]),
        face_normals=torch.tensor([
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1, 2]),
    )


# ===========================================================================
# PhaseMeanVelocityBC tests
# ===========================================================================


class TestPhaseMeanVelocityBC:
    """Tests for phaseMeanVelocity boundary condition."""

    def test_registration(self):
        """phaseMeanVelocity is registered in the RTS registry."""
        assert "phaseMeanVelocity" in BoundaryCondition.available_types()

    def test_factory_creation(self, inlet_patch):
        """BC can be created via the factory method."""
        from pyfoam.boundary.phase_mean_velocity import PhaseMeanVelocityBC

        bc = BoundaryCondition.create(
            "phaseMeanVelocity", inlet_patch,
            coeffs={"Umean": [1.0, 0.0, 0.0], "phaseName": "gas"},
        )
        assert isinstance(bc, PhaseMeanVelocityBC)

    def test_Umean_property(self, inlet_patch):
        """Umean returns the prescribed velocity."""
        from pyfoam.boundary.phase_mean_velocity import PhaseMeanVelocityBC

        bc = PhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"Umean": [2.0, 0.5, 0.0]},
        )
        expected = torch.tensor([2.0, 0.5, 0.0], dtype=torch.float64)
        assert torch.allclose(bc.Umean, expected)

    def test_phase_name_property(self, inlet_patch):
        """Phase name is stored correctly."""
        from pyfoam.boundary.phase_mean_velocity import PhaseMeanVelocityBC

        bc = PhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"phaseName": "gas"},
        )
        assert bc.phase_name == "gas"

    def test_apply_sets_velocity(self, inlet_patch):
        """apply() sets face velocities to Umean."""
        from pyfoam.boundary.phase_mean_velocity import PhaseMeanVelocityBC

        bc = PhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"Umean": [1.0, 0.0, 0.0]},
        )
        field = torch.zeros(35, 3, dtype=torch.float64)
        bc.apply(field)
        for i in range(3):
            assert torch.allclose(field[i], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    def test_apply_with_alpha(self, inlet_patch):
        """apply() with alpha divides velocity by alpha."""
        from pyfoam.boundary.phase_mean_velocity import PhaseMeanVelocityBC

        bc = PhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"Umean": [1.0, 0.0, 0.0]},
        )
        field = torch.zeros(35, 3, dtype=torch.float64)
        alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        bc.apply(field, alpha=alpha)
        # U_phase = Umean / alpha = [2, 0, 0] for alpha=0.5
        for i in range(3):
            assert torch.allclose(field[i], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))

    def test_matrix_contributions(self, inlet_patch):
        """Matrix contributions are non-zero."""
        from pyfoam.boundary.phase_mean_velocity import PhaseMeanVelocityBC

        bc = PhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"Umean": [1.0, 0.0, 0.0]},
        )
        field = torch.zeros(3, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        assert (diag > 0).any()


# ===========================================================================
# ConvectiveHeatTransferBC tests
# ===========================================================================


class TestConvectiveHeatTransferBC:
    """Tests for convectiveHeatTransfer boundary condition."""

    def test_registration(self):
        """convectiveHeatTransfer is registered in the RTS registry."""
        assert "convectiveHeatTransfer" in BoundaryCondition.available_types()

    def test_factory_creation(self, wall_patch):
        """BC can be created via the factory method."""
        from pyfoam.boundary.convective_heat_transfer import ConvectiveHeatTransferBC

        bc = BoundaryCondition.create(
            "convectiveHeatTransfer", wall_patch,
            coeffs={"h": 10.0, "Tinf": 300.0, "k": 0.025},
        )
        assert isinstance(bc, ConvectiveHeatTransferBC)

    def test_properties(self, wall_patch):
        """Properties are stored correctly."""
        from pyfoam.boundary.convective_heat_transfer import ConvectiveHeatTransferBC

        bc = ConvectiveHeatTransferBC(
            wall_patch,
            coeffs={"h": 50.0, "Tinf": 350.0, "k": 0.05},
        )
        assert bc.h == 50.0
        assert bc.Tinf == 350.0
        assert bc.k == 0.05

    def test_biot_number(self, wall_patch):
        """Biot number is computed correctly."""
        from pyfoam.boundary.convective_heat_transfer import ConvectiveHeatTransferBC

        bc = ConvectiveHeatTransferBC(
            wall_patch,
            coeffs={"h": 10.0, "Tinf": 300.0, "k": 0.025},
        )
        assert bc.Biot == pytest.approx(10.0 / 0.025)

    def test_apply_sets_Tinf(self, wall_patch):
        """apply() sets face temperature to Tinf."""
        from pyfoam.boundary.convective_heat_transfer import ConvectiveHeatTransferBC

        bc = ConvectiveHeatTransferBC(
            wall_patch,
            coeffs={"h": 10.0, "Tinf": 350.0, "k": 0.025},
        )
        field = torch.zeros(35, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[30:33], torch.full((3,), 350.0, dtype=torch.float64))

    def test_matrix_contributions_robin(self, wall_patch):
        """Matrix contributions implement Robin BC correctly."""
        from pyfoam.boundary.convective_heat_transfer import ConvectiveHeatTransferBC

        h = 10.0
        Tinf = 350.0
        bc = ConvectiveHeatTransferBC(
            wall_patch,
            coeffs={"h": h, "Tinf": Tinf, "k": 0.025},
        )
        field = torch.zeros(3, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        # Diagonal should have h*A contribution
        assert (diag > 0).all()

        # Source should have h*A*Tinf contribution
        assert (source > 0).all()

        # Check ratio: source/diag should be ~Tinf
        nonzero = diag > 0
        ratio = source[nonzero] / diag[nonzero]
        assert torch.allclose(ratio, torch.full_like(ratio, Tinf), rtol=1e-10)

    def test_h_setter(self, wall_patch):
        """h can be updated via setter."""
        from pyfoam.boundary.convective_heat_transfer import ConvectiveHeatTransferBC

        bc = ConvectiveHeatTransferBC(
            wall_patch,
            coeffs={"h": 10.0, "Tinf": 300.0},
        )
        bc.h = 25.0
        assert bc.h == 25.0
