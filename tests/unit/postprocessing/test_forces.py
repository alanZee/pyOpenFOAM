"""Tests for Forces and ForceCoeffs function objects."""

from __future__ import annotations

import pytest
import torch

from pyfoam.postprocessing.forces import Forces, ForceCoeffs
from pyfoam.postprocessing.function_object import FunctionObjectRegistry


class TestForces:
    def test_init_defaults(self):
        forces = Forces()
        assert forces.name == "forces"
        assert forces._patches == []
        assert forces._rho_inf == 1.0

    def test_init_with_config(self):
        config = {
            "patches": ["movingWall"],
            "rhoInf": 1.225,
            "CofR": [0.5, 0.0, 0.0],
        }
        forces = Forces("forces1", config)
        assert forces.name == "forces1"
        assert forces._patches == ["movingWall"]
        assert forces._rho_inf == 1.225
        assert forces._cofr[0] == 0.5

    def test_initialise(self, fv_mesh, sample_fields):
        forces = Forces("forces", {"patches": ["bottom", "top"]})
        forces.initialise(fv_mesh, sample_fields)

        assert forces.mesh is fv_mesh
        assert forces.fields == sample_fields

    def test_execute(self, fv_mesh, sample_fields):
        forces = Forces("forces", {"patches": ["bottom", "top"]})
        forces.initialise(fv_mesh, sample_fields)

        forces.execute(0.0)
        assert len(forces.times) == 1
        assert len(forces.force_pressure) == 1
        assert len(forces.force_viscous) == 1
        assert len(forces.force_total) == 1
        assert len(forces.moment_history) == 1

    def test_execute_multiple_times(self, fv_mesh, sample_fields):
        forces = Forces("forces", {"patches": ["bottom"]})
        forces.initialise(fv_mesh, sample_fields)

        for t in [0.0, 0.1, 0.2]:
            forces.execute(t)

        assert len(forces.times) == 3
        assert forces.times == [0.0, 0.1, 0.2]

    def test_force_pressure_shape(self, fv_mesh, sample_fields):
        forces = Forces("forces", {"patches": ["bottom"]})
        forces.initialise(fv_mesh, sample_fields)
        forces.execute(0.0)

        assert forces.force_pressure[0].shape == (3,)

    def test_force_total_is_sum(self, fv_mesh, sample_fields):
        forces = Forces("forces", {"patches": ["bottom"]})
        forces.initialise(fv_mesh, sample_fields)
        forces.execute(0.0)

        fp = forces.force_pressure[0]
        fv = forces.force_viscous[0]
        ft = forces.force_total[0]
        assert torch.allclose(ft, fp + fv)

    def test_write(self, fv_mesh, sample_fields, tmp_path):
        forces = Forces("forces", {"patches": ["bottom"]})
        forces.set_output_path(tmp_path)
        forces.initialise(fv_mesh, sample_fields)
        forces.execute(0.0)
        forces.write()

        forces_file = tmp_path / "forces.dat"
        assert forces_file.exists()

    def test_disabled(self, fv_mesh, sample_fields):
        forces = Forces("forces", {"enabled": False})
        forces.initialise(fv_mesh, sample_fields)
        forces.execute(0.0)

        assert len(forces.times) == 0


class TestForceCoeffs:
    def test_init_defaults(self):
        fc = ForceCoeffs()
        assert fc.name == "forceCoeffs"
        assert fc._u_ref == 1.0
        assert fc._a_ref == 1.0

    def test_init_with_config(self):
        config = {
            "patches": ["aerofoil"],
            "rhoInf": 1.225,
            "Uref": 30.0,
            "Aref": 0.1,
            "lRef": 0.1,
        }
        fc = ForceCoeffs("fc1", config)
        assert fc._u_ref == 30.0
        assert fc._a_ref == 0.1

    def test_execute(self, fv_mesh, sample_fields):
        fc = ForceCoeffs("fc", {"patches": ["bottom"]})
        fc.initialise(fv_mesh, sample_fields)

        fc.execute(0.0)
        assert len(fc.cd) == 1
        assert len(fc.cl) == 1
        assert len(fc.cm) == 1

    def test_coefficients_are_floats(self, fv_mesh, sample_fields):
        fc = ForceCoeffs("fc", {"patches": ["bottom"]})
        fc.initialise(fv_mesh, sample_fields)
        fc.execute(0.0)

        assert isinstance(fc.cd[0], float)
        assert isinstance(fc.cl[0], float)
        assert isinstance(fc.cm[0], float)

    def test_write(self, fv_mesh, sample_fields, tmp_path):
        fc = ForceCoeffs("fc", {"patches": ["bottom"]})
        fc.set_output_path(tmp_path)
        fc.initialise(fv_mesh, sample_fields)
        fc.execute(0.0)
        fc.write()

        coeffs_file = tmp_path / "forceCoeffs.dat"
        assert coeffs_file.exists()


class TestForcesRegistration:
    @pytest.fixture(autouse=True)
    def register(self):
        FunctionObjectRegistry.register("forces", Forces)
        FunctionObjectRegistry.register("forceCoeffs", ForceCoeffs)
        yield
        FunctionObjectRegistry.clear()

    def test_forces_registered(self):
        assert "forces" in FunctionObjectRegistry.list_registered()

    def test_force_coeffs_registered(self):
        assert "forceCoeffs" in FunctionObjectRegistry.list_registered()

    def test_create_forces(self):
        fo = FunctionObjectRegistry.create("forces", {"name": "f1", "patches": ["wall"]})
        assert isinstance(fo, Forces)
        assert fo.name == "f1"
