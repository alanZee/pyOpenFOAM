"""
TemporalInterpolate — interpolate fields between time steps.

Provides function objects for temporally interpolating solution fields
to arbitrary times between computed time steps, mirroring OpenFOAM's
``temporalInterpolate`` utility.

Use cases:

- Generating smooth animations at arbitrary frame rates
- Computing fields at times between written output intervals
- Post-processing data at precise time values for comparison with
  experimental data

Interpolation methods:

- ``linear``: Linear interpolation between bounding time steps
- ``cubic``: Cubic (Hermite) interpolation using four surrounding steps

References
----------
- OpenFOAM ``postProcess -func temporalInterpolate`` source
"""

from __future__ import annotations

import logging
import bisect
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["TemporalInterpolate"]

logger = logging.getLogger(__name__)


class TemporalInterpolate(FunctionObject):
    """Interpolate fields between time steps.

    Collects field snapshots over time, then interpolates to any
    requested intermediate time on demand.

    Configuration keys:

    - ``fields``: list of field names to interpolate
    - ``interpolationScheme``: ``"linear"`` (default) or ``"cubic"``
    - ``writeFields``: if True, write interpolated fields (default: False)

    Example controlDict entry::

        temporalInterp1
        {
            type            temporalInterpolate;
            libs            ("libpostProcessing.so");
            fields          (U p);
            interpolationScheme linear;
            writeFields     true;
        }
    """

    SCHEMES = {"linear", "cubic"}

    def __init__(self, name: str = "temporalInterpolate", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._field_names: List[str] = self.config.get("fields", [])
        self._scheme: str = self.config.get("interpolationScheme", "linear")
        self._write_fields: bool = self.config.get("writeFields", False)

        if self._scheme not in self.SCHEMES:
            raise ValueError(
                f"Unknown interpolation scheme '{self._scheme}'. "
                f"Available: {self.SCHEMES}"
            )

        # Storage: {field_name: [(time, tensor), ...]}
        self._snapshots: Dict[str, List[tuple[float, torch.Tensor]]] = {}

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and fields."""
        self._mesh = mesh
        self._fields = fields

        # If no fields specified, use all available
        if not self._field_names:
            self._field_names = list(fields.keys())

        for fname in self._field_names:
            self._snapshots[fname] = []

        logger.info(
            "TemporalInterpolate '%s' initialised: fields=%s, scheme=%s",
            self.name, self._field_names, self._scheme,
        )

    def execute(self, time: float) -> None:
        """Store field snapshots at current time."""
        if not self._enabled or self._mesh is None:
            return

        for fname in self._field_names:
            field = self._fields.get(fname)
            if field is None:
                continue

            device = get_device()
            dtype = get_default_dtype()

            if hasattr(field, "internal_field"):
                data = field.internal_field.to(device=device, dtype=dtype)
            elif hasattr(field, "data"):
                data = field.data.to(device=device, dtype=dtype)
            else:
                data = field.to(device=device, dtype=dtype)

            self._snapshots[fname].append((time, data.detach().cpu().clone()))

        self._log.info(
            "t=%g  stored snapshots for %d fields",
            time, len(self._field_names),
        )

    def interpolate(self, field_name: str, time: float) -> Optional[torch.Tensor]:
        """Interpolate a field to a specific time.

        Parameters
        ----------
        field_name : str
            Name of the field to interpolate.
        time : float
            Target time value.

        Returns
        -------
        torch.Tensor | None
            Interpolated field data, or None if insufficient data.
        """
        snapshots = self._snapshots.get(field_name)
        if snapshots is None or len(snapshots) < 2:
            logger.warning(
                "Cannot interpolate '%s': need >= 2 snapshots, have %d",
                field_name, len(snapshots) if snapshots else 0,
            )
            return None

        times = [s[0] for s in snapshots]

        # Clamp to range
        if time <= times[0]:
            return snapshots[0][1].clone()
        if time >= times[-1]:
            return snapshots[-1][1].clone()

        if self._scheme == "linear":
            return self._interpolate_linear(snapshots, times, time)
        elif self._scheme == "cubic":
            return self._interpolate_cubic(snapshots, times, time)
        else:
            return self._interpolate_linear(snapshots, times, time)

    def interpolate_all(self, time: float) -> Dict[str, torch.Tensor]:
        """Interpolate all tracked fields to a specific time.

        Parameters
        ----------
        time : float
            Target time value.

        Returns
        -------
        dict
            Mapping of field name to interpolated tensor.
        """
        result = {}
        for fname in self._field_names:
            val = self.interpolate(fname, time)
            if val is not None:
                result[fname] = val
        return result

    def _interpolate_linear(
        self,
        snapshots: List[tuple[float, torch.Tensor]],
        times: List[float],
        t: float,
    ) -> torch.Tensor:
        """Linear interpolation between bounding time steps.

        Parameters
        ----------
        snapshots : list
            List of ``(time, data)`` tuples.
        times : list
            Sorted list of time values.
        t : float
            Target time.

        Returns
        -------
        torch.Tensor
            Linearly interpolated data.
        """
        idx = bisect.bisect_left(times, t)
        if idx == 0:
            return snapshots[0][1].clone()
        if idx >= len(times):
            return snapshots[-1][1].clone()

        t0, data0 = snapshots[idx - 1]
        t1, data1 = snapshots[idx]

        alpha = (t - t0) / max(t1 - t0, 1e-30)
        return (1.0 - alpha) * data0 + alpha * data1

    def _interpolate_cubic(
        self,
        snapshots: List[tuple[float, torch.Tensor]],
        times: List[float],
        t: float,
    ) -> torch.Tensor:
        """Cubic (Hermite) interpolation using Catmull-Rom splines.

        Uses 4 surrounding points for smooth interpolation with
        C1 continuity.

        Parameters
        ----------
        snapshots : list
            List of ``(time, data)`` tuples.
        times : list
            Sorted list of time values.
        t : float
            Target time.

        Returns
        -------
        torch.Tensor
            Cubically interpolated data.
        """
        n = len(times)
        idx = bisect.bisect_left(times, t)

        # Clamp indices to valid range for 4-point stencil
        i1 = max(1, min(idx, n - 2))
        i0 = i1 - 1
        i2 = min(i1 + 1, n - 1)
        im1 = max(i0 - 1, 0)

        t0 = times[i0]
        t1 = times[i1]
        dt = max(t1 - t0, 1e-30)
        s = (t - t0) / dt

        p0 = snapshots[im1][1]
        p1 = snapshots[i0][1]
        p2 = snapshots[i1][1]
        p3 = snapshots[i2][1]

        # Catmull-Rom basis functions
        s2 = s * s
        s3 = s2 * s

        # Tangents (central differences, scaled)
        m1 = 0.5 * (p2 - p0)  # tangent at p1
        m2 = 0.5 * (p3 - p1)  # tangent at p2

        # Hermite interpolation
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2

        return h00 * p1 + h10 * dt * m1 + h01 * p2 + h11 * dt * m2

    def write(self) -> None:
        """Write snapshot count info."""
        if self._output_path is None:
            return

        info_file = self._output_path / "temporalInterpolate.info"
        with open(info_file, "w") as f:
            f.write("# TemporalInterpolate results\n")
            f.write(f"# Scheme: {self._scheme}\n")
            for fname in self._field_names:
                n = len(self._snapshots.get(fname, []))
                f.write(f"# {fname}: {n} snapshots\n")
                if n > 0:
                    times = [s[0] for s in self._snapshots[fname]]
                    f.write(f"#   t_range: [{times[0]:.6e}, {times[-1]:.6e}]\n")
        logger.info("Wrote temporal interpolation info to %s", info_file)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def field_names(self) -> List[str]:
        """List of tracked field names."""
        return list(self._field_names)

    @property
    def snapshots(self) -> Dict[str, List[tuple[float, torch.Tensor]]]:
        """Raw snapshot storage: ``{field_name: [(time, data), ...]}``."""
        return self._snapshots

    def get_times(self, field_name: str) -> List[float]:
        """Get sorted time values for a field.

        Parameters
        ----------
        field_name : str
            Field name.

        Returns
        -------
        list[float]
            Sorted time values.
        """
        return [s[0] for s in self._snapshots.get(field_name, [])]

    def get_snapshot_count(self, field_name: str) -> int:
        """Get number of stored snapshots for a field."""
        return len(self._snapshots.get(field_name, []))


# Register
FunctionObjectRegistry.register("temporalInterpolate", TemporalInterpolate)
