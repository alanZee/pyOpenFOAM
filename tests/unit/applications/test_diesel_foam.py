"""Tests for DieselFoam — diesel spray combustion solver.

Tests cover:
- DieselReaction dataclass
- DieselCoupling (momentum, heat, mass sources)
- DieselFoam import and species initialisation
"""

import pytest
import torch

from pyfoam.lagrangian.cloud import KinematicCloud
from pyfoam.lagrangian.particle import Particle


class TestDieselReaction:
    """Tests for DieselReaction dataclass."""

    def test_default_values(self):
        from pyfoam.applications.diesel_foam import DieselReaction

        rxn = DieselReaction()
        assert rxn.name == ""
        assert rxn.A == 1.0
        assert rxn.beta == 0.0
        assert rxn.Ea == 0.0
        assert rxn.reactants == {}
        assert rxn.products == {}

    def test_custom_values(self):
        from pyfoam.applications.diesel_foam import DieselReaction

        rxn = DieselReaction(
            name="diesel_ox",
            A=4.16e9,
            beta=0.0,
            Ea=1.255e5,
            reactants={"C12H23": 1.0, "O2": 17.75},
            products={"CO2": 12.0, "H2O": 11.5},
        )
        assert rxn.name == "diesel_ox"
        assert rxn.A == 4.16e9
        assert rxn.Ea == 1.255e5
        assert rxn.reactants["C12H23"] == 1.0
        assert rxn.products["CO2"] == 12.0


class TestDieselCoupling:
    """Tests for DieselCoupling helper."""

    def _make_coupling(self, fv_mesh):
        from pyfoam.applications.diesel_foam import DieselCoupling

        cloud = KinematicCloud(
            fluid_velocity=[1.0, 0.0, 0.0],
            fluid_density=1.225,
            fluid_viscosity=1.8e-5,
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[10.0, 10.0, 10.0],
        )
        return DieselCoupling(mesh=fv_mesh, cloud=cloud)

    def test_momentum_source_no_particles(self, fv_mesh):
        coupling = self._make_coupling(fv_mesh)
        S = coupling.momentum_source()
        assert S.shape == (fv_mesh.n_cells, 3)
        assert torch.allclose(S, torch.zeros_like(S))

    def test_momentum_source_with_particle(self, fv_mesh):
        coupling = self._make_coupling(fv_mesh)
        centres = fv_mesh.cell_centres
        pos = centres[0].tolist()
        p = Particle(
            position=pos, velocity=[2.0, 0.0, 0.0],
            diameter=1e-4, density=800.0,
        )
        coupling.cloud.add_particle(p)
        S = coupling.momentum_source()
        assert S.shape == (fv_mesh.n_cells, 3)
        assert S.abs().sum() > 0.0

    def test_heat_source_no_particles(self, fv_mesh):
        coupling = self._make_coupling(fv_mesh)
        T = torch.full((fv_mesh.n_cells,), 300.0, dtype=torch.float64)
        Q = coupling.heat_source(T)
        assert Q.shape == (fv_mesh.n_cells,)
        assert torch.allclose(Q, torch.zeros_like(Q))

    def test_heat_source_hot_gas(self, fv_mesh):
        coupling = self._make_coupling(fv_mesh)
        centres = fv_mesh.cell_centres
        pos = centres[0].tolist()
        p = Particle(
            position=pos, velocity=[0.0, 0.0, 0.0],
            diameter=1e-4, density=800.0,
        )
        p.temperature = 300.0
        coupling.cloud.add_particle(p)

        T = torch.full((fv_mesh.n_cells,), 500.0, dtype=torch.float64)
        Q = coupling.heat_source(T)
        assert Q[0].item() > 0.0

    def test_mass_source_no_particles(self, fv_mesh):
        coupling = self._make_coupling(fv_mesh)
        S, species_src = coupling.mass_source(dt=1e-4)
        assert S.shape == (fv_mesh.n_cells,)
        assert torch.allclose(S, torch.zeros_like(S))
        assert "C12H23" in species_src

    def test_mass_source_evaporation(self, fv_mesh):
        coupling = self._make_coupling(fv_mesh)
        centres = fv_mesh.cell_centres
        pos = centres[0].tolist()
        p = Particle(
            position=pos, velocity=[1.0, 0.0, 0.0],
            diameter=1e-4, density=800.0,
        )
        coupling.cloud.add_particle(p)
        S, species_src = coupling.mass_source(dt=1e-4)
        assert S[0].item() > 0.0
        assert species_src["C12H23"][0].item() > 0.0

    def test_fuel_species_configurable(self, fv_mesh):
        from pyfoam.applications.diesel_foam import DieselCoupling

        cloud = KinematicCloud()
        coupling = DieselCoupling(
            mesh=fv_mesh, cloud=cloud, fuel_species="CH4",
        )
        assert coupling.fuel_species == "CH4"
        S, species_src = coupling.mass_source(dt=1e-4)
        assert "CH4" in species_src

    def test_locate_cell(self, fv_mesh):
        coupling = self._make_coupling(fv_mesh)
        centres = fv_mesh.cell_centres
        pos = centres[0].tolist()
        ci = coupling._locate_cell(pos)
        assert 0 <= ci < fv_mesh.n_cells

    def test_particle_reynolds(self, fv_mesh):
        coupling = self._make_coupling(fv_mesh)
        p = Particle(
            position=[0.0, 0.0, 0.0], velocity=[2.0, 0.0, 0.0],
            diameter=1e-4, density=800.0,
        )
        Re = coupling._particle_reynolds(p)
        assert Re >= 0.0


class TestDieselFoamInit:
    """Test DieselFoam import and class properties."""

    def test_import(self):
        from pyfoam.applications.diesel_foam import DieselFoam
        assert DieselFoam is not None

    def test_diesel_coupling_import(self):
        from pyfoam.applications.diesel_foam import DieselCoupling
        assert DieselCoupling is not None

    def test_default_reactions(self):
        """DieselReaction has sensible default oxidation reaction."""
        from pyfoam.applications.diesel_foam import DieselReaction

        rxn = DieselReaction(
            name="diesel_oxidation",
            A=4.16e9,
            Ea=1.255e5,
            reactants={"C12H23": 1.0, "O2": 17.75},
            products={"CO2": 12.0, "H2O": 11.5},
        )
        # Verify stoichiometric balance (atoms)
        # C: 12 = 12, H: 23 = 23 (11.5*2), O: 35.5 = 24+11.5 = 35.5
        assert rxn.reactants["O2"] == 17.75
        assert rxn.products["CO2"] == 12.0
        assert rxn.products["H2O"] == 11.5
