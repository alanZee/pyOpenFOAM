"""Tests for SprayFoam — Lagrangian spray solver."""

import pytest
import torch

from pyfoam.lagrangian.cloud import KinematicCloud
from pyfoam.lagrangian.particle import Particle


class TestLagrangianCoupling:
    """Test the LagrangianCoupling helper class."""

    def _make_coupling(self, fv_mesh):
        """Create a LagrangianCoupling with a simple cloud."""
        from pyfoam.applications.spray_foam import LagrangianCoupling

        cloud = KinematicCloud(
            fluid_velocity=[1.0, 0.0, 0.0],
            fluid_density=1.225,
            fluid_viscosity=1.8e-5,
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[10.0, 10.0, 10.0],
        )
        return LagrangianCoupling(mesh=fv_mesh, cloud=cloud)

    def test_momentum_source_no_particles(self, fv_mesh):
        """With no particles, momentum source is zero."""
        coupling = self._make_coupling(fv_mesh)
        S = coupling.momentum_source()
        assert S.shape == (fv_mesh.n_cells, 3)
        assert torch.allclose(S, torch.zeros_like(S))

    def test_momentum_source_with_particle(self, fv_mesh):
        """Particle in a cell produces non-zero momentum source."""
        coupling = self._make_coupling(fv_mesh)
        # Place particle near first cell centre
        centres = fv_mesh.cell_centres
        pos = centres[0].tolist()
        p = Particle(
            position=pos,
            velocity=[2.0, 0.0, 0.0],
            diameter=1e-4,
            density=800.0,
        )
        coupling.cloud.add_particle(p)
        S = coupling.momentum_source()
        assert S.shape == (fv_mesh.n_cells, 3)
        # Should have non-zero source in at least one cell
        assert S.abs().sum() > 0.0

    def test_heat_source_no_particles(self, fv_mesh):
        """With no particles, heat source is zero."""
        coupling = self._make_coupling(fv_mesh)
        T = torch.full((fv_mesh.n_cells,), 300.0, dtype=torch.float64)
        Q = coupling.heat_source(T)
        assert Q.shape == (fv_mesh.n_cells,)
        assert torch.allclose(Q, torch.zeros_like(Q))

    def test_heat_source_hot_gas(self, fv_mesh):
        """Hot gas drives heat transfer to cold droplet."""
        coupling = self._make_coupling(fv_mesh)
        centres = fv_mesh.cell_centres
        pos = centres[0].tolist()
        p = Particle(
            position=pos,
            velocity=[0.0, 0.0, 0.0],
            diameter=1e-4,
            density=800.0,
        )
        # Set particle temperature lower than gas
        p.temperature = 300.0
        coupling.cloud.add_particle(p)

        T = torch.full((fv_mesh.n_cells,), 500.0, dtype=torch.float64)
        Q = coupling.heat_source(T)
        # Heat flows from gas to particle → Q > 0 in that cell
        assert Q[0].item() > 0.0

    def test_mass_source_no_particles(self, fv_mesh):
        """With no particles, mass source is zero."""
        coupling = self._make_coupling(fv_mesh)
        S, evap = coupling.mass_source(dt=1e-4)
        assert S.shape == (fv_mesh.n_cells,)
        assert torch.allclose(S, torch.zeros_like(S))

    def test_mass_source_evaporation(self, fv_mesh):
        """Evaporating particle produces positive mass source."""
        coupling = self._make_coupling(fv_mesh)
        centres = fv_mesh.cell_centres
        pos = centres[0].tolist()
        p = Particle(
            position=pos,
            velocity=[1.0, 0.0, 0.0],
            diameter=1e-4,
            density=800.0,
        )
        coupling.cloud.add_particle(p)
        S, evap = coupling.mass_source(dt=1e-4)
        assert S[0].item() > 0.0

    def test_locate_cell(self, fv_mesh):
        """_locate_cell returns valid cell index."""
        coupling = self._make_coupling(fv_mesh)
        centres = fv_mesh.cell_centres
        pos = centres[0].tolist()
        ci = coupling._locate_cell(pos)
        assert 0 <= ci < fv_mesh.n_cells

    def test_particle_reynolds(self, fv_mesh):
        """Particle Reynolds number is non-negative."""
        coupling = self._make_coupling(fv_mesh)
        p = Particle(
            position=[0.0, 0.0, 0.0],
            velocity=[2.0, 0.0, 0.0],
            diameter=1e-4,
            density=800.0,
        )
        Re = coupling._particle_reynolds(p)
        assert Re >= 0.0


class TestSprayFoamInit:
    """Test SprayFoam initialisation (requires case directory)."""

    def test_import(self):
        """SprayFoam can be imported."""
        from pyfoam.applications.spray_foam import SprayFoam
        assert SprayFoam is not None

    def test_lagrangian_coupling_import(self):
        """LagrangianCoupling can be imported."""
        from pyfoam.applications.spray_foam import LagrangianCoupling
        assert LagrangianCoupling is not None

    def test_cloud_default(self, fv_mesh):
        """Default cloud has zero particles."""
        cloud = KinematicCloud()
        assert cloud.n_particles == 0

    def test_cloud_with_particles(self, fv_mesh):
        """Cloud with particles reports correct count."""
        cloud = KinematicCloud()
        for i in range(5):
            cloud.add_particle(
                Particle(
                    position=[float(i), 0.0, 0.0],
                    velocity=[1.0, 0.0, 0.0],
                    diameter=1e-4,
                    density=800.0,
                )
            )
        assert cloud.n_particles == 5
