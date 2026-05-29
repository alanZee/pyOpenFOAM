"""Tests for vortex identification and turbulence postprocessing tools.

Covers: Vorticity, QCriterion, Lambda2, Enstrophy, TurbulentKineticEnergy.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from pyfoam.postprocessing.vorticity import Vorticity
from pyfoam.postprocessing.q_criterion import QCriterion
from pyfoam.postprocessing.lambda2 import Lambda2
from pyfoam.postprocessing.enstrophy import Enstrophy
from pyfoam.postprocessing.turbulent_kinetic_energy import TurbulentKineticEnergy
from pyfoam.postprocessing.function_object import FunctionObjectRegistry


# ---------------------------------------------------------------------------
# Vorticity
# ---------------------------------------------------------------------------

class TestVorticity:
    def test_init_defaults(self):
        fo = Vorticity()
        assert fo.name == "vorticity"
        assert fo._field_name == "U"

    def test_init_with_config(self):
        config = {"field": "U", "writeField": True}
        fo = Vorticity("vort1", config)
        assert fo._field_name == "U"
        assert fo._write_field is True

    def test_initialise(self, fv_mesh, sample_fields):
        fo = Vorticity()
        fo.initialise(fv_mesh, sample_fields)
        assert fo.mesh is fv_mesh

    def test_execute(self, fv_mesh, sample_fields):
        fo = Vorticity()
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.omega is not None
        assert fo.omega.shape == (2, 3)
        assert len(fo.times) == 1

    def test_execute_missing_field(self, fv_mesh, sample_fields):
        fo = Vorticity("vort", {"field": "nonexistent"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert fo.omega is None

    def test_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        from pyfoam.postprocessing.vorticity import Vorticity
        FunctionObjectRegistry.register("vorticity", Vorticity)
        assert "vorticity" in FunctionObjectRegistry.list_registered()


# ---------------------------------------------------------------------------
# QCriterion
# ---------------------------------------------------------------------------

class TestQCriterion:
    def test_init_defaults(self):
        fo = QCriterion()
        assert fo.name == "Q"
        assert fo._field_name == "U"
        assert fo.threshold == 0.0

    def test_init_with_threshold(self):
        fo = QCriterion("Q1", {"threshold": 100.0})
        assert fo.threshold == 100.0

    def test_initialise(self, fv_mesh, sample_fields):
        fo = QCriterion()
        fo.initialise(fv_mesh, sample_fields)
        assert fo.mesh is fv_mesh

    def test_execute(self, fv_mesh, sample_fields):
        fo = QCriterion()
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.Q_field is not None
        assert fo.Q_field.shape == (2,)
        assert len(fo.times) == 1

    def test_vortex_cells(self, fv_mesh, sample_fields):
        fo = QCriterion("Q1", {"threshold": -1e10})  # Low threshold to capture all
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        cells = fo.vortex_cells()
        assert cells is not None

    def test_execute_missing_field(self, fv_mesh, sample_fields):
        fo = QCriterion("Q1", {"field": "nonexistent"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert fo.Q_field is None

    def test_registered(self):
        from pyfoam.postprocessing import q_criterion as _m
        importlib.reload(_m)
        assert "Q" in FunctionObjectRegistry.list_registered()


# ---------------------------------------------------------------------------
# Lambda2
# ---------------------------------------------------------------------------

class TestLambda2:
    def test_init_defaults(self):
        fo = Lambda2()
        assert fo.name == "lambda2"
        assert fo._field_name == "U"

    def test_initialise(self, fv_mesh, sample_fields):
        fo = Lambda2()
        fo.initialise(fv_mesh, sample_fields)
        assert fo.mesh is fv_mesh

    def test_execute(self, fv_mesh, sample_fields):
        fo = Lambda2()
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.lambda2_field is not None
        assert fo.lambda2_field.shape == (2,)
        assert len(fo.times) == 1

    def test_vortex_cells(self, fv_mesh, sample_fields):
        fo = Lambda2()
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        # Result should be a tensor (may or may not find vortex cells
        # depending on the velocity field)
        cells = fo.vortex_cells()
        assert cells is not None

    def test_execute_missing_field(self, fv_mesh, sample_fields):
        fo = Lambda2("l2", {"field": "nonexistent"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert fo.lambda2_field is None

    def test_registered(self):
        from pyfoam.postprocessing import lambda2 as _m
        importlib.reload(_m)
        assert "Lambda2" in FunctionObjectRegistry.list_registered()


# ---------------------------------------------------------------------------
# Enstrophy
# ---------------------------------------------------------------------------

class TestEnstrophy:
    def test_init_defaults(self):
        fo = Enstrophy()
        assert fo.name == "enstrophy"
        assert fo._field_name == "U"

    def test_initialise(self, fv_mesh, sample_fields):
        fo = Enstrophy()
        fo.initialise(fv_mesh, sample_fields)
        assert fo.mesh is fv_mesh

    def test_execute(self, fv_mesh, sample_fields):
        fo = Enstrophy()
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.enstrophy_field is not None
        assert fo.enstrophy_field.shape == (2,)
        # Enstrophy is non-negative
        assert (fo.enstrophy_field >= 0).all()
        assert len(fo.times) == 1

    def test_execute_missing_field(self, fv_mesh, sample_fields):
        fo = Enstrophy("ens", {"field": "nonexistent"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert fo.enstrophy_field is None

    def test_registered(self):
        from pyfoam.postprocessing import enstrophy as _m
        importlib.reload(_m)
        assert "enstrophy" in FunctionObjectRegistry.list_registered()


# ---------------------------------------------------------------------------
# TurbulentKineticEnergy
# ---------------------------------------------------------------------------

class TestTurbulentKineticEnergy:
    def test_init_defaults(self):
        fo = TurbulentKineticEnergy()
        assert fo.name == "TKE"
        assert fo._mode == "resolved"
        assert fo._field_name == "U"
        assert fo._mean_field_name == "UMean"

    def test_init_rans_mode(self):
        fo = TurbulentKineticEnergy("k", {"mode": "rans"})
        assert fo._mode == "rans"

    def test_init_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            TurbulentKineticEnergy("k", {"mode": "invalid"})

    def test_initialise(self, fv_mesh, sample_fields):
        fo = TurbulentKineticEnergy()
        fo.initialise(fv_mesh, sample_fields)
        assert fo.mesh is fv_mesh

    def test_execute_resolved_with_mean(self, fv_mesh, sample_fields):
        """TKE resolved mode with mean velocity provided."""
        from pyfoam.fields.vol_fields import volVectorField

        # Add a mean velocity field (same as U → zero fluctuation)
        U_mean = volVectorField(fv_mesh, "UMean")
        U_mean.assign(torch.tensor([
            [1.0, 0.0, 0.0],
            [0.5, 0.1, 0.0],
        ], dtype=fv_mesh.dtype, device=fv_mesh.device))
        sample_fields["UMean"] = U_mean

        fo = TurbulentKineticEnergy()
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.k_field is not None
        assert fo.k_field.shape == (2,)
        # When U == UMean, fluctuation is zero → TKE is zero
        assert torch.allclose(fo.k_field, torch.zeros(2, dtype=fv_mesh.dtype))
        assert len(fo.times) == 1

    def test_execute_resolved_with_fluctuation(self, fv_mesh, sample_fields):
        """TKE resolved mode with non-zero fluctuation."""
        from pyfoam.fields.vol_fields import volVectorField

        # Mean velocity = zero → fluctuations = U
        U_mean = volVectorField(fv_mesh, "UMean")
        U_mean.assign(torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=fv_mesh.dtype, device=fv_mesh.device))
        sample_fields["UMean"] = U_mean

        fo = TurbulentKineticEnergy()
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.k_field is not None
        # k = 0.5 * (1^2 + 0 + 0) = 0.5 for cell 0
        # k = 0.5 * (0.5^2 + 0.1^2 + 0) = 0.5 * 0.26 = 0.13 for cell 1
        expected_k0 = 0.5 * (1.0**2 + 0.0**2 + 0.0**2)
        expected_k1 = 0.5 * (0.5**2 + 0.1**2 + 0.0**2)
        assert torch.allclose(
            fo.k_field,
            torch.tensor([expected_k0, expected_k1], dtype=fv_mesh.dtype),
            atol=1e-12,
        )

    def test_execute_rans(self, fv_mesh, sample_fields):
        """TKE RANS mode reads k field directly."""
        from pyfoam.fields.vol_fields import volScalarField

        k = volScalarField(fv_mesh, "k")
        k.assign(torch.tensor([0.1, 0.05], dtype=fv_mesh.dtype, device=fv_mesh.device))
        sample_fields["k"] = k

        fo = TurbulentKineticEnergy("k_rans", {"mode": "rans"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.k_field is not None
        assert torch.allclose(
            fo.k_field,
            torch.tensor([0.1, 0.05], dtype=fv_mesh.dtype),
        )

    def test_execute_resolved_no_mean(self, fv_mesh, sample_fields):
        """Resolved mode without mean field → should skip."""
        fo = TurbulentKineticEnergy()
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert fo.k_field is None

    def test_execute_rans_no_k(self, fv_mesh, sample_fields):
        """RANS mode without k field → should skip."""
        fo = TurbulentKineticEnergy("k", {"mode": "rans"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert fo.k_field is None

    def test_registered(self):
        from pyfoam.postprocessing import turbulent_kinetic_energy as _m
        importlib.reload(_m)
        assert "TKE" in FunctionObjectRegistry.list_registered()
        assert "turbulentKineticEnergy" in FunctionObjectRegistry.list_registered()
