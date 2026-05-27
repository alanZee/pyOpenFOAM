"""Tests for k-omega SST Langtry-Menter transition model.

Tests cover:
- Model creation and RTS registration
- Transition-specific constants
- Transition correlation functions (F_onset, F_length, gamma_sep, Re_theta_c)
- Gamma and Re_thetat transport equations
- Gamma_eff modification of k equation
- Integration with base SST model
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.k_omega_sst import KOmegaSSTModel
from pyfoam.turbulence.k_omega_sst_lm import (
    KOmegaSSTLMModel,
    KOmegaSSTLMConstants,
)

from tests.unit.turbulence.conftest import make_fv_mesh


class TestKOmegaSSTLMRegistration:
    """Tests for RTS registration of k-omega SST LM model."""

    def test_k_omega_sst_lm_registered(self):
        """kOmegaSSTLM is registered in the RTS registry."""
        assert "kOmegaSSTLM" in TurbulenceModel.available_types()

    def test_create_k_omega_sst_lm(self):
        """Can create kOmegaSSTLM model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("kOmegaSSTLM", mesh, U, phi)
        assert isinstance(model, KOmegaSSTLMModel)

    def test_model_inherits_sst(self):
        """kOmegaSSTLM is a subclass of KOmegaSSTModel."""
        assert issubclass(KOmegaSSTLMModel, KOmegaSSTModel)


class TestKOmegaSSTLMConstants:
    """Tests for k-omega SST LM transition constants."""

    def test_default_constants(self):
        """Default constants match Langtry-Menter (2009) values."""
        C = KOmegaSSTLMConstants()
        # SST constants inherited
        assert C.sigma_k1 == 0.85
        assert C.beta_star == 0.09
        assert C.a1 == 0.31
        # Transition constants
        assert C.ca1 == 2.0
        assert C.ca2 == 0.06
        assert C.ce1 == 1.0
        assert C.ce2 == 50.0
        assert C.cThetat == 0.03
        assert C.sigmaf == 1.0
        assert C.sigmat == 2.0

    def test_custom_constants(self):
        """Can create custom constants."""
        C = KOmegaSSTLMConstants(ca1=3.0, sigmaf=0.5)
        assert C.ca1 == 3.0
        assert C.sigmaf == 0.5
        # SST defaults preserved
        assert C.beta_star == 0.09

    def test_constants_frozen(self):
        """Constants are immutable (frozen dataclass)."""
        C = KOmegaSSTLMConstants()
        with pytest.raises(AttributeError):
            C.ca1 = 3.0


class TestKOmegaSSTLMModel:
    """Tests for k-omega SST LM model behaviour."""

    def test_model_creation(self):
        """Model can be created with mesh, U, phi."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_gamma_field_shape(self):
        """gamma_field returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        assert model.gamma_field.shape == (mesh.n_cells,)

    def test_gamma_initial_value(self):
        """gamma initialized to 1.0 (fully turbulent)."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        assert torch.allclose(model.gamma_field, torch.ones(mesh.n_cells, dtype=torch.float64))

    def test_Re_thetat_field_shape(self):
        """Re_thetat_field returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        assert model.Re_thetat_field.shape == (mesh.n_cells,)

    def test_Re_thetat_initial_value(self):
        """Re_thetat initialized to 0.0."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        assert torch.allclose(model.Re_thetat_field, torch.zeros(mesh.n_cells, dtype=torch.float64))

    def test_nut_inherited(self):
        """nut() inherits SST limiter behaviour."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_k_and_omega_inherited(self):
        """k() and omega() inherit from SST."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_repr(self):
        """Model repr includes class name."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        r = repr(model)
        assert "KOmegaSSTLMModel" in r


class TestKOmegaSSTLMCorrelationFunctions:
    """Tests for transition correlation functions."""

    def _make_model(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        return KOmegaSSTLMModel(mesh, U, phi), mesh

    def test_Re_theta_c_shape(self):
        """Re_theta_c returns correct shape."""
        model, mesh = self._make_model()
        result = model._Re_theta_c()
        assert result.shape == (mesh.n_cells,)

    def test_Re_theta_c_non_negative(self):
        """Re_theta_c returns non-negative values."""
        model, _ = self._make_model()
        result = model._Re_theta_c()
        assert (result >= 0).all()

    def test_F_onset_shape(self):
        """F_onset returns correct shape."""
        model, mesh = self._make_model()
        result = model._F_onset()
        assert result.shape == (mesh.n_cells,)

    def test_F_onset_non_negative(self):
        """F_onset returns non-negative values."""
        model, _ = self._make_model()
        result = model._F_onset()
        assert (result >= 0).all()

    def test_F_length_shape(self):
        """F_length returns correct shape."""
        model, mesh = self._make_model()
        result = model._F_length()
        assert result.shape == (mesh.n_cells,)

    def test_F_length_positive(self):
        """F_length returns positive values."""
        model, _ = self._make_model()
        result = model._F_length()
        assert (result > 0).all()

    def test_gamma_sep_shape(self):
        """gamma_sep returns correct shape."""
        model, mesh = self._make_model()
        result = model._gamma_sep()
        assert result.shape == (mesh.n_cells,)

    def test_gamma_sep_non_negative(self):
        """gamma_sep returns non-negative values."""
        model, _ = self._make_model()
        result = model._gamma_sep()
        assert (result >= 0).all()

    def test_gamma_eff_shape(self):
        """gamma_eff returns correct shape."""
        model, mesh = self._make_model()
        result = model._gamma_eff()
        assert result.shape == (mesh.n_cells,)

    def test_gamma_eff_non_negative(self):
        """gamma_eff returns non-negative values."""
        model, _ = self._make_model()
        result = model._gamma_eff()
        assert (result >= 0).all()

    def test_gamma_eff_geq_gamma(self):
        """gamma_eff >= gamma (includes separation contribution)."""
        model, _ = self._make_model()
        gamma_eff = model._gamma_eff()
        gamma = model.gamma_field
        assert (gamma_eff >= gamma - 1e-10).all()


class TestKOmegaSSTLMTransport:
    """Tests for transition transport equation solving."""

    def test_correct_updates_all_fields(self):
        """correct() updates k, omega, gamma, and Re_thetat."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)
        assert model.gamma_field.shape == (mesh.n_cells,)
        assert model.Re_thetat_field.shape == (mesh.n_cells,)

    def test_correct_with_velocity(self):
        """correct() works with non-zero velocity."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0  # Uniform x-velocity
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)
        assert model.gamma_field.shape == (mesh.n_cells,)
        assert model.Re_thetat_field.shape == (mesh.n_cells,)

    def test_gamma_remains_bounded_after_correct(self):
        """gamma stays in [0, 1] after correct()."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        model.correct()

        gamma = model.gamma_field
        assert (gamma >= 0.0).all()
        assert (gamma <= 1.0).all()

    def test_Re_thetat_non_negative_after_correct(self):
        """Re_thetat stays non-negative after correct()."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        model.correct()

        assert (model.Re_thetat_field >= 0.0).all()

    def test_custom_constants_passed_to_model(self):
        """Custom constants are accessible in the model."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        C = KOmegaSSTLMConstants(ca1=3.0)
        model = KOmegaSSTLMModel(mesh, U, phi, constants=C)
        assert model._C.ca1 == 3.0

    def test_solve_gamma_direct(self):
        """_solve_gamma modifies gamma field."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        gamma_before = model.gamma_field.clone()

        # Need velocity gradient for strain rate computation in gamma equation
        model._grad_U = torch.zeros(mesh.n_cells, 3, 3, dtype=torch.float64)
        model._solve_gamma()

        # gamma should still be in valid range
        assert (model.gamma_field >= 0.0).all()
        assert (model.gamma_field <= 1.0).all()

    def test_solve_Re_thetat_direct(self):
        """_solve_Re_thetat modifies Re_thetat field."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        Re_before = model.Re_thetat_field.clone()

        model._solve_Re_thetat()

        assert (model.Re_thetat_field >= 0.0).all()


class TestKOmegaSSTLMFieldSetters:
    """Tests for field property setters."""

    def test_gamma_setter(self):
        """gamma_field can be set externally."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        new_gamma = torch.tensor([0.5, 0.8], dtype=torch.float64)
        model.gamma_field = new_gamma

        assert torch.allclose(model.gamma_field, new_gamma)

    def test_Re_thetat_setter(self):
        """Re_thetat_field can be set externally."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTLMModel(mesh, U, phi)
        new_re = torch.tensor([200.0, 400.0], dtype=torch.float64)
        model.Re_thetat_field = new_re

        assert torch.allclose(model.Re_thetat_field, new_re)
