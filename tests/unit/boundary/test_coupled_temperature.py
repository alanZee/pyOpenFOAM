"""Tests for coupledTemperature boundary condition (CHT interface)."""

import pytest
import torch

from pyfoam.boundary.coupled_temperature import CoupledTemperatureBC, create_coupled_bc


class TestCoupledTemperatureBC:
    """Test the coupled temperature boundary condition."""

    def test_construction(self):
        """BC can be constructed with coupled region data."""
        coupled_field = torch.tensor([300.0, 350.0, 400.0], dtype=torch.float64)
        coupled_owner = torch.tensor([0, 1, 2], dtype=torch.long)
        coupled_face_indices = torch.tensor([0, 2], dtype=torch.long)

        bc = CoupledTemperatureBC(
            name="interface",
            coupled_field=coupled_field,
            coupled_owner=coupled_owner,
            coupled_face_indices=coupled_face_indices,
        )
        assert bc.name == "interface"
        assert torch.allclose(bc.coupled_field, coupled_field)
        assert torch.allclose(bc.coupled_owner, coupled_owner)
        assert torch.allclose(bc.coupled_face_indices, coupled_face_indices)

    def test_value_returns_coupled_temperatures(self):
        """value() reads temperatures from coupled region owner cells."""
        coupled_field = torch.tensor([300.0, 350.0, 400.0], dtype=torch.float64)
        coupled_owner = torch.tensor([0, 1, 2], dtype=torch.long)
        # Map boundary faces to coupled face indices 0 and 2
        coupled_face_indices = torch.tensor([0, 2], dtype=torch.long)

        bc = CoupledTemperatureBC(
            name="interface",
            coupled_field=coupled_field,
            coupled_owner=coupled_owner,
            coupled_face_indices=coupled_face_indices,
        )
        vals = bc.value()
        # coupled_owner[0]=0 -> field[0]=300, coupled_owner[2]=2 -> field[2]=400
        assert vals.shape == (2,)
        assert torch.allclose(vals, torch.tensor([300.0, 400.0], dtype=torch.float64))

    def test_value_with_indirect_owner_mapping(self):
        """value() correctly follows owner -> field indirection."""
        coupled_field = torch.tensor([100.0, 200.0, 300.0, 400.0], dtype=torch.float64)
        # Face 0 owned by cell 3, face 1 owned by cell 1, face 2 owned by cell 0
        coupled_owner = torch.tensor([3, 1, 0], dtype=torch.long)
        coupled_face_indices = torch.tensor([0, 1, 2], dtype=torch.long)

        bc = CoupledTemperatureBC(
            name="htInterface",
            coupled_field=coupled_field,
            coupled_owner=coupled_owner,
            coupled_face_indices=coupled_face_indices,
        )
        vals = bc.value()
        assert torch.allclose(vals, torch.tensor([400.0, 200.0, 100.0], dtype=torch.float64))

    def test_repr(self):
        """repr shows class name, patch name, and coupled face count."""
        coupled_field = torch.tensor([300.0], dtype=torch.float64)
        coupled_owner = torch.tensor([0], dtype=torch.long)
        coupled_face_indices = torch.tensor([0], dtype=torch.long)

        bc = CoupledTemperatureBC(
            name="interface",
            coupled_field=coupled_field,
            coupled_owner=coupled_owner,
            coupled_face_indices=coupled_face_indices,
        )
        r = repr(bc)
        assert "CoupledTemperatureBC" in r
        assert "interface" in r
        assert "n_coupled=1" in r


class TestCreateCoupledBC:
    """Test the create_coupled_bc factory function."""

    def test_factory_creates_bc(self):
        """Factory creates a CoupledTemperatureBC from mesh-like objects."""
        # Simulate minimal mesh objects with face_centres and owner attributes
        class FakeMesh:
            def __init__(self, face_centres, owner):
                self.face_centres = face_centres
                self.owner = owner

        # 3 fluid interface faces at z=0, y=0, x=[0, 1, 2]
        fluid_centres = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=torch.float64)
        # 3 solid interface faces at z=0.1, y=0, x=[0, 1, 2] (offset in z)
        solid_centres = torch.tensor([
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.1],
            [2.0, 0.0, 0.1],
        ], dtype=torch.float64)

        fluid_mesh = FakeMesh(
            face_centres=fluid_centres,
            owner=torch.tensor([0, 1, 2], dtype=torch.long),
        )
        # Owner cells for solid faces — indices into the full solid T field
        solid_mesh = FakeMesh(
            face_centres=solid_centres,
            owner=torch.tensor([0, 1, 2], dtype=torch.long),
        )

        # Full solid temperature field (one value per cell in the solid region)
        T_solid = torch.tensor([500.0, 600.0, 700.0], dtype=torch.float64)
        interface_faces_fluid = torch.tensor([0, 1, 2], dtype=torch.long)
        interface_faces_solid = torch.tensor([0, 1, 2], dtype=torch.long)

        bc = create_coupled_bc(
            patch_name="chtInterface",
            fluid_mesh=fluid_mesh,
            solid_mesh=solid_mesh,
            T_solid=T_solid,
            interface_faces_fluid=interface_faces_fluid,
            interface_faces_solid=interface_faces_solid,
        )

        assert isinstance(bc, CoupledTemperatureBC)
        assert bc.name == "chtInterface"
        # Nearest-neighbour mapping should match 1:1 since faces are aligned
        vals = bc.value()
        assert vals.shape == (3,)
        assert torch.allclose(vals, torch.tensor([500.0, 600.0, 700.0], dtype=torch.float64))

    def test_factory_nearest_neighbour_mapping(self):
        """Factory maps fluid faces to nearest solid faces correctly."""
        class FakeMesh:
            def __init__(self, face_centres, owner):
                self.face_centres = face_centres
                self.owner = owner

        # Fluid faces at x=0, x=2 (2 faces)
        fluid_centres = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=torch.float64)
        # Solid faces at x=0.1, x=0.9, x=1.9 (3 faces — non-trivial mapping)
        solid_centres = torch.tensor([
            [0.1, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.9, 0.0, 0.0],
        ], dtype=torch.float64)

        fluid_mesh = FakeMesh(
            face_centres=fluid_centres,
            owner=torch.tensor([0, 1], dtype=torch.long),
        )
        # Owner cells for solid faces — indices into the full solid T field
        solid_mesh = FakeMesh(
            face_centres=solid_centres,
            owner=torch.tensor([0, 1, 2], dtype=torch.long),
        )

        # Full solid temperature field
        T_solid = torch.tensor([500.0, 600.0, 700.0], dtype=torch.float64)
        interface_faces_fluid = torch.tensor([0, 1], dtype=torch.long)
        interface_faces_solid = torch.tensor([0, 1, 2], dtype=torch.long)

        bc = create_coupled_bc(
            patch_name="chtFace",
            fluid_mesh=fluid_mesh,
            solid_mesh=solid_mesh,
            T_solid=T_solid,
            interface_faces_fluid=interface_faces_fluid,
            interface_faces_solid=interface_faces_solid,
        )

        vals = bc.value()
        # Fluid face 0 (x=0) -> nearest solid face 0 (x=0.1) -> owner 0 -> T=500
        # Fluid face 1 (x=2) -> nearest solid face 2 (x=1.9) -> owner 2 -> T=700
        assert vals.shape == (2,)
        assert torch.allclose(vals, torch.tensor([500.0, 700.0], dtype=torch.float64))
