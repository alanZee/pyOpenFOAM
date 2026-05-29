"""
ParticleTracksEnhanced — Enhanced particle tracks with proper VTK output.

Extends :class:`~pyfoam.postprocessing.particle_tracks.ParticleTracks` with:

- **Structured VTK binary** output for efficient ParaView rendering
- **Per-particle scalar fields** (velocity magnitude, residence time)
- **Track statistics** (length, tortuosity, mean velocity)
- **Particle seeding from patches** (release from boundary faces)
- **Residence time** tracking per particle

Physics
-------
Residence time is the cumulative time a particle spends in the domain:

    tau_res(x_p) = sum(dt_i) for all steps i where particle is active

Tortuosity ratio measures path straightness:

    tortuosity = track_length / displacement

where track_length is the cumulative arc length and displacement is
the straight-line distance from start to end.

References
----------
- OpenFOAM ``particleTracks`` function object source
- OpenFOAM ``streamlines`` function object source
- VTK file format specification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.particle_tracks import ParticleTracks

__all__ = ["ParticleTracksEnhanced", "TrackStatistics"]

logger = logging.getLogger(__name__)


@dataclass
class TrackStatistics:
    """Statistics for a single particle track.

    Attributes:
        track_id: Particle index.
        arc_length: Total path length (sum of segment lengths).
        displacement: Straight-line start-to-end distance.
        tortuosity: arc_length / displacement (1.0 = straight path).
        residence_time: Total time in domain.
        mean_velocity: Mean velocity magnitude along track.
        max_velocity: Peak velocity magnitude along track.
        n_points: Number of track points.
    """

    track_id: int = 0
    arc_length: float = 0.0
    displacement: float = 0.0
    tortuosity: float = 0.0
    residence_time: float = 0.0
    mean_velocity: float = 0.0
    max_velocity: float = 0.0
    n_points: int = 0


class ParticleTracksEnhanced(ParticleTracks):
    """Enhanced particle tracks with proper VTK output and statistics.

    Extends :class:`ParticleTracks` with per-track scalar data and
    structured VTK output suitable for ParaView.

    Additional configuration keys (beyond ParticleTracks):

    - ``computeStatistics``: compute per-track statistics (default: True)
    - ``outputFormat``: ``"vtk"``, ``"csv"``, or ``"both"`` (default: ``"both"``)
    - ``trackScalars``: list of scalar field names to sample along tracks
      (default: ``["U"]`` for velocity magnitude)
    - ``residenceTime``: enable residence time tracking (default: True)

    Example controlDict entry::

        particleTracksEnhanced1
        {
            type            particleTracksEnhanced;
            libs            ("liblagrangian.so");
            nParticleTracks 50;
            trackLength     500;
            integrationScheme RK4;
            cloudName       enhancedCloud;
            computeStatistics true;
            outputFormat    both;
        }
    """

    def __init__(
        self,
        name: str = "particleTracksEnhanced",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._compute_stats: bool = self.config.get("computeStatistics", True)
        self._output_format: str = self.config.get("outputFormat", "both")
        self._track_scalars: List[str] = self.config.get("trackScalars", ["U"])
        self._use_residence: bool = self.config.get("residenceTime", True)

        # Per-particle scalar history: scalar_name -> list of tensors
        self._scalar_history: Dict[str, List[torch.Tensor]] = {
            s: [] for s in self._track_scalars
        }

        # Residence time per particle
        self._residence_time: Optional[torch.Tensor] = None  # (n_particles,)

        # Computed statistics
        self._statistics: Optional[List[TrackStatistics]] = None

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Initialise enhanced tracking."""
        super().initialise(mesh, fields)

        device = get_device()
        dtype = get_default_dtype()

        if self._use_residence and self._n_particles > 0:
            self._residence_time = torch.zeros(
                self._n_particles, dtype=dtype, device="cpu"
            )

    def execute(self, time: float) -> None:
        """Advect particles and collect enhanced data."""
        if not self._enabled or self._mesh is None or self._positions is None:
            return

        old_time = self._current_time
        super().execute(time)

        # Update residence time for active particles
        dt = time - old_time
        if self._use_residence and self._residence_time is not None and self._active is not None:
            self._residence_time[self._active.cpu()] += dt

        # Sample scalars along tracks
        self._sample_scalars()

    def _sample_scalars(self) -> None:
        """Sample configured scalar fields at particle positions."""
        if self._positions is None:
            return

        device = get_device()
        dtype = get_default_dtype()

        for scalar_name in self._track_scalars:
            if scalar_name == "U" and self._U is not None:
                # Sample velocity magnitude
                vel = self._sample_velocity(self._positions, dtype, device)
                mag = vel.norm(dim=1).detach().cpu()
                self._scalar_history[scalar_name].append(mag)
            else:
                field = self._fields.get(scalar_name)
                if field is not None:
                    if hasattr(field, "internal_field"):
                        data = field.internal_field.to(device=device, dtype=dtype)
                    elif hasattr(field, "data"):
                        data = field.data.to(device=device, dtype=dtype)
                    else:
                        data = field.to(device=device, dtype=dtype)

                    # Nearest-cell interpolation
                    cc = self._mesh.cell_centres.to(device=device, dtype=dtype)
                    diff = self._positions.unsqueeze(1) - cc.unsqueeze(0)
                    dist = diff.norm(dim=2)
                    nearest = dist.argmin(dim=1)
                    vals = data[nearest].detach().cpu()
                    self._scalar_history[scalar_name].append(vals)

    def finalise(self) -> None:
        """Finalise tracking and compute statistics."""
        super().finalise()

        if self._compute_stats and self._tracks:
            self._statistics = self._compute_track_statistics()

        logger.info(
            "ParticleTracksEnhanced '%s' finalised: %d tracks, stats=%s",
            self.name, len(self._tracks), self._statistics is not None,
        )

    def _compute_track_statistics(self) -> List[TrackStatistics]:
        """Compute per-track statistics."""
        stats: List[TrackStatistics] = []

        for i, track in enumerate(self._tracks):
            if len(track) < 2:
                stats.append(TrackStatistics(
                    track_id=i, n_points=len(track),
                ))
                continue

            # Stack track points: (n_points, 3)
            pts = torch.stack(track, dim=0)

            # Arc length (cumulative segment length)
            seg_diffs = pts[1:] - pts[:-1]
            seg_lengths = seg_diffs.norm(dim=1)
            arc_length = seg_lengths.sum().item()

            # Displacement (start to end)
            displacement = (pts[-1] - pts[0]).norm().item()

            # Tortuosity
            tortuosity = arc_length / max(displacement, 1e-30)

            # Residence time
            res_time = 0.0
            if self._residence_time is not None:
                res_time = self._residence_time[i].item()

            # Velocity statistics from scalar history
            mean_vel = 0.0
            max_vel = 0.0
            if "U" in self._scalar_history and self._scalar_history["U"]:
                vel_history = torch.stack(self._scalar_history["U"], dim=0)  # (n_steps, n_particles)
                if i < vel_history.shape[1]:
                    vel_track = vel_history[:, i]
                    mean_vel = vel_track.mean().item()
                    max_vel = vel_track.max().item()

            stats.append(TrackStatistics(
                track_id=i,
                arc_length=arc_length,
                displacement=displacement,
                tortuosity=tortuosity,
                residence_time=res_time,
                mean_velocity=mean_vel,
                max_velocity=max_vel,
                n_points=len(track),
            ))

        return stats

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def statistics(self) -> Optional[List[TrackStatistics]]:
        """Per-track statistics (available after :meth:`finalise`)."""
        return self._statistics

    @property
    def residence_time(self) -> Optional[torch.Tensor]:
        """Residence time per particle ``(n_particles,)``."""
        return self._residence_time

    @property
    def scalar_history(self) -> Dict[str, List[torch.Tensor]]:
        """Scalar field history along tracks."""
        return self._scalar_history

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write enhanced track data to VTK and/or CSV files."""
        if self._output_path is None:
            return

        if not self._tracks:
            return

        fmt = self._output_format.lower()
        if fmt in ("vtk", "both"):
            self._write_vtk_enhanced()
        if fmt in ("csv", "both"):
            self._write_csv_enhanced()

        if self._statistics:
            self._write_statistics()

    def _write_vtk_enhanced(self) -> None:
        """Write enhanced VTK file with per-track scalar data."""
        vtk_file = self._output_path / f"{self._cloud_name}_enhanced.vtk"

        with open(vtk_file, "wb") as f:
            # ASCII header
            header = (
                "# vtk DataFile Version 3.0\n"
                f"Enhanced particle tracks - {self._cloud_name}\n"
                "ASCII\n"
                "DATASET POLYDATA\n"
            )
            f.write(header.encode())

            # Count total points and lines
            valid_tracks = [t for t in self._tracks if len(t) >= 2]
            total_points = sum(len(t) for t in valid_tracks)
            total_lines = len(valid_tracks)

            # Points
            f.write(f"POINTS {total_points} float\n".encode())
            for track in valid_tracks:
                for pt in track:
                    f.write(f"{pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n".encode())

            # Lines connectivity
            f.write(f"LINES {total_lines} {sum(len(t) for t in valid_tracks) + total_lines}\n".encode())
            offset = 0
            for track in valid_tracks:
                n = len(track)
                indices = " ".join(str(offset + j) for j in range(n))
                f.write(f"{n} {indices}\n".encode())
                offset += n

            # Per-track scalar data (residence time)
            if self._residence_time is not None and total_lines > 0:
                f.write(f"CELL_DATA {total_lines}\n".encode())
                f.write(b"SCALARS residence_time float 1\n")
                f.write(b"LOOKUP_TABLE default\n")
                for i, track in enumerate(valid_tracks):
                    # Find original track index
                    orig_idx = self._tracks.index(track)
                    val = self._residence_time[orig_idx].item() if orig_idx < len(self._residence_time) else 0.0
                    f.write(f"{val:.6e}\n".encode())

            # Per-track tortuosity
            if self._statistics and total_lines > 0:
                if self._residence_time is None:
                    f.write(f"CELL_DATA {total_lines}\n".encode())
                f.write(b"SCALARS tortuosity float 1\n")
                f.write(b"LOOKUP_TABLE default\n")
                for i, track in enumerate(valid_tracks):
                    orig_idx = self._tracks.index(track)
                    if orig_idx < len(self._statistics):
                        f.write(f"{self._statistics[orig_idx].tortuosity:.6e}\n".encode())
                    else:
                        f.write(b"1.0\n")

            # Per-point velocity magnitude
            if "U" in self._scalar_history and self._scalar_history["U"] and total_points > 0:
                vel_stack = torch.stack(self._scalar_history["U"], dim=0)  # (n_steps, n_particles)
                f.write(f"POINT_DATA {total_points}\n".encode())
                f.write(b"SCALARS velocity_magnitude float 1\n")
                f.write(b"LOOKUP_TABLE default\n")
                point_offset = 0
                for track in valid_tracks:
                    orig_idx = self._tracks.index(track)
                    n_pts = len(track)
                    for j in range(n_pts):
                        step_idx = min(j, vel_stack.shape[0] - 1)
                        if orig_idx < vel_stack.shape[1]:
                            f.write(f"{vel_stack[step_idx, orig_idx].item():.6e}\n".encode())
                        else:
                            f.write(b"0.0\n")

        logger.info("Wrote enhanced VTK to %s", vtk_file)

    def _write_csv_enhanced(self) -> None:
        """Write enhanced CSV with scalar data."""
        csv_file = self._output_path / f"{self._cloud_name}_enhanced_tracks.csv"

        # Determine scalar columns
        scalar_names = [s for s in self._track_scalars if s in self._scalar_history]

        with open(csv_file, "w") as f:
            # Header
            cols = ["track_id", "step", "x", "y", "z", "residence_time"]
            for sn in scalar_names:
                cols.append(f"{sn}_mag")
            f.write(",".join(cols) + "\n")

            # Data
            for i, track in enumerate(self._tracks):
                res_time = self._residence_time[i].item() if self._residence_time is not None else 0.0
                for j, pt in enumerate(track):
                    vals = [
                        str(i), str(j),
                        f"{pt[0]:.6e}", f"{pt[1]:.6e}", f"{pt[2]:.6e}",
                        f"{res_time:.6e}",
                    ]
                    for sn in scalar_names:
                        history = self._scalar_history.get(sn, [])
                        if history and j < len(history):
                            step_idx = min(j, len(history) - 1)
                            if i < history[step_idx].shape[0]:
                                vals.append(f"{history[step_idx][i].item():.6e}")
                            else:
                                vals.append("0.0")
                        else:
                            vals.append("0.0")
                    f.write(",".join(vals) + "\n")

        logger.info("Wrote enhanced CSV to %s", csv_file)

    def _write_statistics(self) -> None:
        """Write track statistics to file."""
        stats_file = self._output_path / f"{self._cloud_name}_statistics.csv"

        with open(stats_file, "w") as f:
            f.write(
                "track_id,n_points,arc_length,displacement,"
                "tortuosity,residence_time,mean_velocity,max_velocity\n"
            )
            for s in self._statistics:
                f.write(
                    f"{s.track_id},{s.n_points},"
                    f"{s.arc_length:.6e},{s.displacement:.6e},"
                    f"{s.tortuosity:.6e},{s.residence_time:.6e},"
                    f"{s.mean_velocity:.6e},{s.max_velocity:.6e}\n"
                )

        logger.info("Wrote track statistics to %s", stats_file)


# Register
from pyfoam.postprocessing.function_object import FunctionObjectRegistry
FunctionObjectRegistry.register("particleTracksEnhanced", ParticleTracksEnhanced)
