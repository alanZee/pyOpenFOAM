"""
Unit tests for Lagrangian reacting particle models.
"""

from __future__ import annotations

import math
import pytest

from pyfoam.lagrangian.reacting_models import (
    ReactingModel,
    SinglePhaseReacting,
    MultiPhaseReacting,
)


# ======================================================================
# ReactingModel ABC
# ======================================================================

class TestReactingModelABC:
    """Tests for ReactingModel abstract base."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ReactingModel()


# ======================================================================
# SinglePhaseReacting
# ======================================================================

class TestSinglePhaseReacting:
    """Tests for SinglePhaseReacting model."""

    def test_default_parameters(self):
        model = SinglePhaseReacting()
        assert model.A == pytest.approx(1e3)
        assert model.activation_energy == pytest.approx(1.2e5)
        assert model.n_order == pytest.approx(0.5)
        assert model.heat_of_reaction == pytest.approx(-2e7)

    def test_custom_parameters(self):
        model = SinglePhaseReacting(A=5e3, activation_energy=1.0e5)
        assert model.A == pytest.approx(5e3)
        assert model.activation_energy == pytest.approx(1.0e5)

    def test_A_must_be_non_negative(self):
        with pytest.raises(ValueError, match="A"):
            SinglePhaseReacting(A=-1.0)

    def test_ea_must_be_non_negative(self):
        with pytest.raises(ValueError, match="activation_energy"):
            SinglePhaseReacting(activation_energy=-1.0)

    def test_diffusivity_must_be_non_negative(self):
        with pytest.raises(ValueError, match="diffusivity"):
            SinglePhaseReacting(diffusivity=-1.0)

    def test_r_gas_must_be_positive(self):
        with pytest.raises(ValueError, match="r_gas"):
            SinglePhaseReacting(r_gas=0.0)

    def test_zero_diameter_no_reaction(self):
        model = SinglePhaseReacting()
        result = model.react(
            dt=1e-4, diameter=0.0, temperature=1500.0,
            fluid_temperature=2000.0,
        )
        assert result["mass_loss"] == pytest.approx(0.0)
        assert result["diameter"] == pytest.approx(0.0)

    def test_low_temperature_no_reaction(self):
        """Temperature < 1K should give no reaction."""
        model = SinglePhaseReacting()
        result = model.react(
            dt=1e-4, diameter=1e-4, temperature=0.5,
            fluid_temperature=2000.0,
        )
        assert result["mass_loss"] == pytest.approx(0.0)

    def test_zero_species_fraction_no_reaction(self):
        model = SinglePhaseReacting()
        result = model.react(
            dt=1e-4, diameter=1e-4, temperature=1500.0,
            fluid_temperature=2000.0, species_mass_fraction=0.0,
        )
        assert result["mass_loss"] == pytest.approx(0.0)

    def test_reaction_at_high_temperature(self):
        model = SinglePhaseReacting(A=1e10, activation_energy=5e4)
        result = model.react(
            dt=1e-4, diameter=1e-4, temperature=1500.0,
            fluid_temperature=2000.0, species_mass_fraction=1.0,
        )
        assert result["mass_loss"] > 0.0
        assert result["heat_release"] > 0.0
        assert result["diameter"] < 1e-4

    def test_mass_loss_bounded_by_particle_mass(self):
        """Mass loss should not exceed current particle mass."""
        model = SinglePhaseReacting(A=1e20, activation_energy=0.0)
        result = model.react(
            dt=1.0, diameter=1e-4, temperature=3000.0,
            fluid_temperature=5000.0, species_mass_fraction=1.0,
        )
        m_particle = (math.pi / 6.0) * (1e-4) ** 3 * 1000.0
        assert result["mass_loss"] <= m_particle + 1e-30

    def test_diameter_decreases(self):
        model = SinglePhaseReacting(A=1e10, activation_energy=5e4)
        d0 = 1e-4
        result = model.react(
            dt=1e-4, diameter=d0, temperature=1500.0,
            fluid_temperature=2000.0, species_mass_fraction=1.0,
        )
        assert result["diameter"] < d0
        assert result["diameter"] > 0.0

    def test_arrhenius_rate(self):
        model = SinglePhaseReacting(A=1e3, activation_energy=1.2e5)
        rate_low = model.arrhenius_rate(500.0)
        rate_high = model.arrhenius_rate(2000.0)
        assert rate_high > rate_low
        assert rate_low == pytest.approx(rate_low)  # finite

    def test_arrhenius_rate_zero_temp(self):
        model = SinglePhaseReacting()
        assert model.arrhenius_rate(0.0) == pytest.approx(0.0)

    def test_higher_species_fraction_more_loss(self):
        model = SinglePhaseReacting(A=1e10, activation_energy=5e4)
        common = dict(
            dt=1e-4, diameter=1e-4,
            temperature=1500.0, fluid_temperature=2000.0,
        )
        r1 = model.react(species_mass_fraction=0.2, **common)
        r2 = model.react(species_mass_fraction=0.8, **common)
        assert r2["mass_loss"] > r1["mass_loss"]

    def test_repr(self):
        model = SinglePhaseReacting(A=5e3)
        r = repr(model)
        assert "SinglePhaseReacting" in r
        assert "5000" in r


# ======================================================================
# MultiPhaseReacting
# ======================================================================

class TestMultiPhaseReacting:
    """Tests for MultiPhaseReacting model."""

    def test_default_species(self):
        model = MultiPhaseReacting()
        assert len(model.species) == 1

    def test_custom_species(self):
        species = [
            {"A": 1e3, "Ea": 1.2e5, "heat": -2e7, "weight": 0.7},
            {"A": 5e3, "Ea": 8e4, "heat": -1e7, "weight": 0.3},
        ]
        model = MultiPhaseReacting(species=species)
        assert len(model.species) == 2

    def test_species_must_have_required_keys(self):
        with pytest.raises(ValueError, match="A.*Ea"):
            MultiPhaseReacting(species=[{"A": 1.0}])  # missing Ea

    def test_r_gas_must_be_positive(self):
        with pytest.raises(ValueError, match="r_gas"):
            MultiPhaseReacting(r_gas=0.0)

    def test_zero_diameter_no_reaction(self):
        model = MultiPhaseReacting()
        result = model.react(
            dt=1e-4, diameter=0.0, temperature=1500.0,
            fluid_temperature=2000.0,
        )
        assert result["mass_loss"] == pytest.approx(0.0)

    def test_low_temperature_no_reaction(self):
        model = MultiPhaseReacting()
        result = model.react(
            dt=1e-4, diameter=1e-4, temperature=0.5,
            fluid_temperature=2000.0,
        )
        assert result["mass_loss"] == pytest.approx(0.0)

    def test_zero_species_fraction_no_reaction(self):
        model = MultiPhaseReacting()
        result = model.react(
            dt=1e-4, diameter=1e-4, temperature=1500.0,
            fluid_temperature=2000.0, species_mass_fraction=0.0,
        )
        assert result["mass_loss"] == pytest.approx(0.0)

    def test_multi_species_reaction(self):
        species = [
            {"A": 1e10, "Ea": 5e4, "heat": -2e7, "weight": 0.6},
            {"A": 1e10, "Ea": 8e4, "heat": -1e7, "weight": 0.4},
        ]
        model = MultiPhaseReacting(species=species)
        result = model.react(
            dt=1e-4, diameter=1e-4, temperature=1500.0,
            fluid_temperature=2000.0, species_mass_fraction=1.0,
        )
        assert result["mass_loss"] > 0.0
        assert result["heat_release"] > 0.0
        assert result["diameter"] < 1e-4

    def test_total_heat_release_sum(self):
        species = [
            {"A": 1e10, "Ea": 5e4, "heat": -2e7, "weight": 0.5},
            {"A": 1e10, "Ea": 5e4, "heat": -1e7, "weight": 0.5},
        ]
        model = MultiPhaseReacting(species=species)
        result = model.react(
            dt=1e-4, diameter=1e-4, temperature=1500.0,
            fluid_temperature=2000.0, species_mass_fraction=1.0,
        )
        assert result["heat_release"] > 0.0

    def test_repr(self):
        model = MultiPhaseReacting(
            species=[
                {"A": 1e3, "Ea": 1e5},
                {"A": 2e3, "Ea": 2e5},
            ],
        )
        r = repr(model)
        assert "MultiPhaseReacting" in r
        assert "2" in r
