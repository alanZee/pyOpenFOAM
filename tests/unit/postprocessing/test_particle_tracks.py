"""
Unit tests for ParticleTracks — particle track generation.

Tests cover:
- Init with default and custom config
- Scheme validation
- Particle seeding (random and explicit)
- Single-step advection (RK1, RK2, RK4)
- Track history collection
- Velocity sampling
- Active particle masking
- Finalisation
- Writing VTK and CSV output
- Registration in FunctionObjectRegistry
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.postprocessing.particle_tracks import ParticleTracks


class TestParticleTracksInit:
    """Tests for ParticleTracks initialisation."""

    def test_init_defaults(self):
        pt = ParticleTracks()
        assert pt.name == "particleTracks"
        assert pt._n_particles == 100
        assert pt._track_length == 1000
        assert pt._scheme == "RK4"
        assert pt._cloud_name == "particleTracks"

    def test_init_with_config(self):
        config = {
            "nParticleTracks": 50,
            "trackLength": 500,
            "integrationScheme": "RK2",
            "cloudName": "myCloud",
            "lifeTime": 10.0,
        }
        pt = ParticleTracks("pt1", config)
        assert pt.name == "pt1"
        assert pt._n_particles == 50
        assert pt._track_length == 500
        assert pt._scheme == "RK2"
        assert pt._cloud_name == "myCloud"
        assert pt._life_time == 10.0

    def test_init_with_seed_points(self):
        config = {
            "seedPoints": [[0.1, 0.1, 0.05], [0.5, 0.5, 0.05]],
        }
        pt = ParticleTracks("pt", config)
        assert pt._seed_points == [[0.1, 0.1, 0.05], [0.5, 0.5, 0.05]]

    def test_init_invalid_scheme(self):
        with pytest.raises(ValueError, match="Unknown integration scheme"):
            ParticleTracks("test", {"integrationScheme": "RK3"})

    def test_valid_schemes(self):
        for scheme in ["RK1", "RK2", "RK4"]:
            pt = ParticleTracks("test", {"integrationScheme": scheme})
            assert pt._scheme == scheme


class TestParticleTracksSeeding:
    """Tests for particle seed generation."""

    def test_random_seeds_created(self, fv_mesh, sample_fields):
        """Random seeds are generated inside the mesh bounding box."""
        pt = ParticleTracks("pt", {"nParticleTracks": 20})
        pt.initialise(fv_mesh, sample_fields)

        assert pt.positions is not None
        assert pt.positions.shape == (20, 3)
        assert pt.n_particles == 20

    def test_explicit_seeds(self, fv_mesh, sample_fields):
        """Explicit seed points are used when provided."""
        seeds = [[0.25, 0.25, 0.5], [0.75, 0.75, 1.5]]
        pt = ParticleTracks("pt", {"seedPoints": seeds})
        pt.initialise(fv_mesh, sample_fields)

        assert pt.positions is not None
        assert pt.positions.shape == (2, 3)
        assert pt.n_particles == 2

    def test_seeds_within_bounds(self, fv_mesh, sample_fields):
        """Random seeds are within the mesh bounding box."""
        pt = ParticleTracks("pt", {"nParticleTracks": 50})
        pt.initialise(fv_mesh, sample_fields)

        cc = fv_mesh.cell_centres
        bb_min = cc.min(dim=0).values
        bb_max = cc.max(dim=0).values

        # All positions should be within bounds (with small tolerance)
        assert (pt.positions >= bb_min - 0.01).all()
        assert (pt.positions <= bb_max + 0.01).all()

    def test_tracks_initialised(self, fv_mesh, sample_fields):
        """Each particle has a track starting with its initial position."""
        pt = ParticleTracks("pt", {"nParticleTracks": 5})
        pt.initialise(fv_mesh, sample_fields)

        assert len(pt.tracks) == 5
        for i in range(5):
            assert len(pt.tracks[i]) == 1
            assert torch.allclose(
                pt.tracks[i][0], pt.positions[i].cpu()
            )


class TestParticleTracksAdvection:
    """Tests for particle advection schemes."""

    def test_rk1_single_step(self, fv_mesh, sample_fields):
        """RK1 advances particles by one step."""
        pt = ParticleTracks("pt", {
            "nParticleTracks": 2,
            "integrationScheme": "RK1",
        })
        pt.initialise(fv_mesh, sample_fields)
        pos_before = pt.positions.clone()

        pt.execute(0.001)

        # Positions should have changed
        assert not torch.allclose(pt.positions, pos_before)
        assert pt.n_steps == 1

    def test_rk2_single_step(self, fv_mesh, sample_fields):
        """RK2 advances particles by one step."""
        pt = ParticleTracks("pt", {
            "nParticleTracks": 2,
            "integrationScheme": "RK2",
        })
        pt.initialise(fv_mesh, sample_fields)
        pos_before = pt.positions.clone()

        pt.execute(0.001)

        assert not torch.allclose(pt.positions, pos_before)
        assert pt.n_steps == 1

    def test_rk4_single_step(self, fv_mesh, sample_fields):
        """RK4 advances particles by one step."""
        pt = ParticleTracks("pt", {
            "nParticleTracks": 2,
            "integrationScheme": "RK4",
        })
        pt.initialise(fv_mesh, sample_fields)
        pos_before = pt.positions.clone()

        pt.execute(0.001)

        assert not torch.allclose(pt.positions, pos_before)
        assert pt.n_steps == 1

    def test_multiple_steps(self, fv_mesh, sample_fields):
        """Multiple execute calls advance particles multiple steps."""
        pt = ParticleTracks("pt", {
            "nParticleTracks": 3,
            "integrationScheme": "RK1",
        })
        pt.initialise(fv_mesh, sample_fields)

        for i in range(5):
            pt.execute(0.001 * (i + 1))

        assert pt.n_steps == 5
        # Each track should have initial + 5 steps = 6 points
        for track in pt.tracks:
            assert len(track) == 6

    def test_tracks_grow(self, fv_mesh, sample_fields):
        """Track history grows with each execute call."""
        pt = ParticleTracks("pt", {"nParticleTracks": 2})
        pt.initialise(fv_mesh, sample_fields)

        initial_lengths = [len(t) for t in pt.tracks]

        pt.execute(0.001)

        for i, track in enumerate(pt.tracks):
            assert len(track) == initial_lengths[i] + 1


class TestParticleTracksVelocitySampling:
    """Tests for velocity interpolation at particle positions."""

    def test_sample_velocity_shape(self, fv_mesh, sample_fields):
        """Velocity sampling returns correct shape."""
        pt = ParticleTracks("pt", {"nParticleTracks": 3, "fields": "U"})
        pt.initialise(fv_mesh, sample_fields)

        # Trigger U field caching
        pt.execute(0.0)

        vel = pt._sample_velocity(
            pt.positions, pt.positions.dtype, pt.positions.device,
        )
        assert vel.shape == (3, 3)

    def test_sample_velocity_nonzero(self, fv_mesh, sample_fields):
        """Velocity is non-zero at some positions (U has non-zero values)."""
        pt = ParticleTracks("pt", {"nParticleTracks": 1, "fields": "U"})
        pt.initialise(fv_mesh, sample_fields)
        pt.execute(0.0)

        vel = pt._sample_velocity(
            pt.positions, pt.positions.dtype, pt.positions.device,
        )
        # At least one velocity component should be non-zero
        assert vel.abs().sum() > 0


class TestParticleTracksActiveMask:
    """Tests for active particle management."""

    def test_all_active_initially(self, fv_mesh, sample_fields):
        """All particles are active after initialisation."""
        pt = ParticleTracks("pt", {"nParticleTracks": 5})
        pt.initialise(fv_mesh, sample_fields)

        assert pt.active is not None
        assert pt.active.all()

    def test_finalise_deactivates(self, fv_mesh, sample_fields):
        """finalise deactivates all particles."""
        pt = ParticleTracks("pt", {"nParticleTracks": 5})
        pt.initialise(fv_mesh, sample_fields)

        pt.finalise()

        assert not pt.active.any()

    def test_disabled_no_execute(self, fv_mesh, sample_fields):
        """Disabled function object does not advance particles."""
        pt = ParticleTracks("pt", {"enabled": False, "nParticleTracks": 3})
        pt.initialise(fv_mesh, sample_fields)
        pt.execute(0.001)

        assert pt.n_steps == 0


class TestParticleTracksWrite:
    """Tests for output file writing."""

    def test_write_vtk(self, fv_mesh, sample_fields, tmp_path):
        """Writing produces VTK and CSV files."""
        pt = ParticleTracks("pt", {
            "nParticleTracks": 3,
            "cloudName": "testCloud",
        })
        pt.set_output_path(tmp_path)
        pt.initialise(fv_mesh, sample_fields)

        # Advance a few steps
        for i in range(3):
            pt.execute(0.001 * (i + 1))

        pt.write()

        vtk_file = tmp_path / "testCloud.vtk"
        csv_file = tmp_path / "testCloud_tracks.csv"
        assert vtk_file.exists()
        assert csv_file.exists()

        # Check VTK header
        with open(vtk_file) as f:
            header = f.readline()
            assert "vtk" in header.lower()

        # Check CSV header
        with open(csv_file) as f:
            header = f.readline()
            assert "track_id" in header
            assert "x" in header

    def test_write_no_data(self, tmp_path):
        """Writing with no tracks is skipped gracefully."""
        pt = ParticleTracks("pt")
        pt.set_output_path(tmp_path)
        pt.write()

        # Files should not exist (positions is None)
        assert not (tmp_path / "particleTracks.vtk").exists()


class TestParticleTracksRegistration:
    """Tests for FunctionObjectRegistry registration."""

    def test_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        from pyfoam.postprocessing import particle_tracks  # noqa: F401
        assert "particleTracks" in FunctionObjectRegistry.list_registered()

    def test_create_from_registry(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        from pyfoam.postprocessing import particle_tracks  # noqa: F401
        fo = FunctionObjectRegistry.create("particleTracks", {"name": "pt1"})
        assert isinstance(fo, ParticleTracks)
        assert fo.name == "pt1"
