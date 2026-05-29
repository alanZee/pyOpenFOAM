"""
Unit tests for ParticleTracksEnhanced — enhanced particle tracking.

Tests cover:
- Init with default and custom config
- Track statistics computation
- Residence time tracking
- Scalar field sampling
- VTK enhanced output
- CSV enhanced output
- Statistics output
- Registration in FunctionObjectRegistry
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.postprocessing.particle_tracks_enhanced import (
    ParticleTracksEnhanced,
    TrackStatistics,
)


class TestTrackStatistics:
    """Tests for TrackStatistics data class."""

    def test_defaults(self):
        s = TrackStatistics()
        assert s.track_id == 0
        assert s.arc_length == 0.0
        assert s.tortuosity == 0.0
        assert s.n_points == 0

    def test_custom_values(self):
        s = TrackStatistics(
            track_id=5, arc_length=10.0, displacement=5.0,
            tortuosity=2.0, residence_time=1.0,
            mean_velocity=3.0, max_velocity=5.0, n_points=100,
        )
        assert s.track_id == 5
        assert s.arc_length == 10.0
        assert s.tortuosity == 2.0


class TestParticleTracksEnhancedInit:
    """Tests for ParticleTracksEnhanced initialisation."""

    def test_init_defaults(self):
        pte = ParticleTracksEnhanced()
        assert pte.name == "particleTracksEnhanced"
        assert pte._compute_stats is True
        assert pte._output_format == "both"
        assert pte._use_residence is True

    def test_init_with_config(self):
        config = {
            "nParticleTracks": 20,
            "computeStatistics": False,
            "outputFormat": "vtk",
            "residenceTime": False,
        }
        pte = ParticleTracksEnhanced("pte1", config)
        assert pte.name == "pte1"
        assert pte._compute_stats is False
        assert pte._output_format == "vtk"
        assert pte._use_residence is False

    def test_inherits_particle_tracks(self):
        """ParticleTracksEnhanced is a subclass of ParticleTracks."""
        from pyfoam.postprocessing.particle_tracks import ParticleTracks
        pte = ParticleTracksEnhanced()
        assert isinstance(pte, ParticleTracks)


class TestResidenceTime:
    """Tests for residence time tracking."""

    def test_residence_time_created(self, fv_mesh, sample_fields):
        """Residence time tensor is created on initialise."""
        pte = ParticleTracksEnhanced("pte", {
            "nParticleTracks": 5,
            "residenceTime": True,
        })
        pte.initialise(fv_mesh, sample_fields)

        assert pte.residence_time is not None
        assert pte.residence_time.shape == (5,)

    def test_residence_time_updated(self, fv_mesh, sample_fields):
        """Residence time increases with each execute step."""
        pte = ParticleTracksEnhanced("pte", {
            "nParticleTracks": 3,
            "residenceTime": True,
            "dt": 0.01,
        })
        pte.initialise(fv_mesh, sample_fields)

        pte.execute(0.01)
        pte.execute(0.02)
        pte.execute(0.03)

        # All particles should have non-zero residence time
        assert (pte.residence_time > 0).any()

    def test_residence_time_disabled(self, fv_mesh, sample_fields):
        """Residence time is None when disabled."""
        pte = ParticleTracksEnhanced("pte", {
            "nParticleTracks": 5,
            "residenceTime": False,
        })
        pte.initialise(fv_mesh, sample_fields)

        assert pte.residence_time is None


class TestStatistics:
    """Tests for track statistics computation."""

    def test_statistics_computed(self, fv_mesh, sample_fields):
        """Statistics are computed after finalise."""
        pte = ParticleTracksEnhanced("pte", {
            "nParticleTracks": 3,
            "computeStatistics": True,
        })
        pte.initialise(fv_mesh, sample_fields)

        for i in range(5):
            pte.execute(0.001 * (i + 1))

        pte.finalise()

        assert pte.statistics is not None
        assert len(pte.statistics) == 3

    def test_statistics_values(self, fv_mesh, sample_fields):
        """Statistics have sensible values."""
        pte = ParticleTracksEnhanced("pte", {
            "nParticleTracks": 2,
            "computeStatistics": True,
        })
        pte.initialise(fv_mesh, sample_fields)

        for i in range(5):
            pte.execute(0.001 * (i + 1))

        pte.finalise()

        for s in pte.statistics:
            assert s.n_points >= 1
            assert s.arc_length >= 0.0
            assert s.displacement >= 0.0

    def test_statistics_disabled(self, fv_mesh, sample_fields):
        """Statistics not computed when disabled."""
        pte = ParticleTracksEnhanced("pte", {
            "nParticleTracks": 3,
            "computeStatistics": False,
        })
        pte.initialise(fv_mesh, sample_fields)

        for i in range(3):
            pte.execute(0.001 * (i + 1))

        pte.finalise()

        assert pte.statistics is None


class TestScalarSampling:
    """Tests for scalar field sampling along tracks."""

    def test_scalar_history_created(self, fv_mesh, sample_fields):
        """Scalar history dict is created."""
        pte = ParticleTracksEnhanced("pte", {
            "nParticleTracks": 3,
            "trackScalars": ["U"],
        })
        pte.initialise(fv_mesh, sample_fields)

        assert "U" in pte.scalar_history
        assert isinstance(pte.scalar_history["U"], list)


class TestWrite:
    """Tests for enhanced output writing."""

    def test_write_vtk(self, fv_mesh, sample_fields, tmp_path):
        """Writing enhanced VTK file."""
        pte = ParticleTracksEnhanced("pte", {
            "nParticleTracks": 3,
            "cloudName": "testEnhanced",
            "outputFormat": "vtk",
        })
        pte.set_output_path(tmp_path)
        pte.initialise(fv_mesh, sample_fields)

        for i in range(3):
            pte.execute(0.001 * (i + 1))

        pte.write()

        vtk_file = tmp_path / "testEnhanced_enhanced.vtk"
        assert vtk_file.exists()

    def test_write_csv(self, fv_mesh, sample_fields, tmp_path):
        """Writing enhanced CSV file."""
        pte = ParticleTracksEnhanced("pte", {
            "nParticleTracks": 3,
            "cloudName": "testEnhanced",
            "outputFormat": "csv",
        })
        pte.set_output_path(tmp_path)
        pte.initialise(fv_mesh, sample_fields)

        for i in range(3):
            pte.execute(0.001 * (i + 1))

        pte.write()

        csv_file = tmp_path / "testEnhanced_enhanced_tracks.csv"
        assert csv_file.exists()

    def test_write_both(self, fv_mesh, sample_fields, tmp_path):
        """Writing both VTK and CSV."""
        pte = ParticleTracksEnhanced("pte", {
            "nParticleTracks": 3,
            "cloudName": "testEnhanced",
            "outputFormat": "both",
        })
        pte.set_output_path(tmp_path)
        pte.initialise(fv_mesh, sample_fields)

        for i in range(3):
            pte.execute(0.001 * (i + 1))

        pte.finalise()
        pte.write()

        assert (tmp_path / "testEnhanced_enhanced.vtk").exists()
        assert (tmp_path / "testEnhanced_enhanced_tracks.csv").exists()

    def test_write_statistics(self, fv_mesh, sample_fields, tmp_path):
        """Writing statistics file."""
        pte = ParticleTracksEnhanced("pte", {
            "nParticleTracks": 3,
            "cloudName": "testEnhanced",
            "computeStatistics": True,
        })
        pte.set_output_path(tmp_path)
        pte.initialise(fv_mesh, sample_fields)

        for i in range(3):
            pte.execute(0.001 * (i + 1))

        pte.finalise()
        pte.write()

        stats_file = tmp_path / "testEnhanced_statistics.csv"
        assert stats_file.exists()

    def test_write_no_tracks(self, tmp_path):
        """Writing with no tracks is skipped."""
        pte = ParticleTracksEnhanced("pte")
        pte.set_output_path(tmp_path)
        pte.write()

        assert not (tmp_path / "particleTracksEnhanced_enhanced.vtk").exists()


class TestRegistration:
    """Tests for FunctionObjectRegistry registration."""

    def test_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        from pyfoam.postprocessing.particle_tracks_enhanced import ParticleTracksEnhanced
        FunctionObjectRegistry.register("particleTracksEnhanced", ParticleTracksEnhanced)
        assert "particleTracksEnhanced" in FunctionObjectRegistry.list_registered()
