"""
FieldAverage — time-averaged field statistics function object.

Computes running time averages, mean squares, and higher-order statistics
of fields.  Supports ``simple`` (cumulative) and ``windowed`` averaging.

This mirrors OpenFOAM's ``fieldAverage`` function object.

Physics
-------
For a field quantity ``q(t)``:

- Mean:  ``<q> = (1/T) * integral(q dt)``
- Mean square:  ``<q^2> = (1/T) * integral(q^2 dt)``
- Variance:  ``var(q) = <q^2> - <q>^2``
- RMS:  ``rms(q) = sqrt(<q^2>)``

Configuration keys:

- ``fields``: list of field names to average
- ``windowType``: ``"simple"`` (cumulative) or ``"windowed"``
- ``window``: averaging window duration (for windowed mode)
- ``mean``: if True, compute mean
- ``mean2``: if True, compute mean square
- ``prime2Mean``: if True, compute variance (mean of squared fluctuations)

Usage::

    fa = FieldAverage("fieldAvg1", {
        "fields": ["p", "U"],
        "windowType": "simple",
        "mean": True,
        "mean2": True,
    })
    fa.initialise(mesh, fields)
    fa.execute(0.001)   # call each time step
    fa.execute(0.002)
    fa.write()

References
----------
- OpenFOAM ``fieldAverage`` function object source
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["FieldAverage"]

logger = logging.getLogger(__name__)


class FieldAverage(FunctionObject):
    """Compute time-averaged field statistics.

    Configuration keys:

    - ``fields``: list of field names to average (required)
    - ``windowType``: ``"simple"`` or ``"windowed"`` (default: ``"simple"``)
    - ``window``: window duration for windowed mode (default: 1.0)
    - ``mean``: compute running mean (default: True)
    - ``mean2``: compute running mean square (default: False)
    - ``prime2Mean``: compute variance (default: False)
    """

    def __init__(self, name: str = "fieldAverage", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._field_names: List[str] = self.config.get("fields", [])
        self._window_type: str = self.config.get("windowType", "simple")
        self._window_duration: float = float(self.config.get("window", 1.0))
        self._compute_mean: bool = self.config.get("mean", True)
        self._compute_mean2: bool = self.config.get("mean2", False)
        self._compute_prime2: bool = self.config.get("prime2Mean", False)

        # Running statistics per field
        self._n_samples: int = 0
        self._total_time: float = 0.0
        self._start_time: Optional[float] = None
        self._last_time: Optional[float] = None

        # Accumulators: field_name -> tensor
        self._mean_accum: Dict[str, torch.Tensor] = {}
        self._mean2_accum: Dict[str, torch.Tensor] = {}
        self._prime2_accum: Dict[str, torch.Tensor] = {}

        # Results snapshot
        self._mean_fields: Dict[str, torch.Tensor] = {}
        self._mean2_fields: Dict[str, torch.Tensor] = {}
        self._prime2_fields: Dict[str, torch.Tensor] = {}

        # Time history for windowed mode
        self._time_history: List[float] = []
        self._field_history: Dict[str, List[torch.Tensor]] = {}

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and fields, initialise accumulators."""
        self._mesh = mesh
        self._fields = fields

        # Validate field names
        for fname in self._field_names:
            if fname not in fields:
                logger.warning("Field '%s' not found during initialisation.", fname)

        logger.info(
            "FieldAverage '%s' initialised: fields=%s, windowType=%s",
            self.name, self._field_names, self._window_type,
        )

    def execute(self, time: float) -> None:
        """Accumulate statistics at the current time step."""
        if not self._enabled:
            return

        device = get_device()
        dtype = get_default_dtype()

        # Time step weight (dt)
        if self._last_time is None:
            dt = 0.0
            self._start_time = time
        else:
            dt = time - self._last_time

        self._last_time = time

        if dt <= 0 and self._n_samples > 0:
            logger.warning("Non-positive time step dt=%g at t=%g. Skipping.", dt, time)
            return

        self._n_samples += 1
        self._total_time += dt if dt > 0 else 1.0  # first sample gets weight 1

        for fname in self._field_names:
            field = self._fields.get(fname)
            if field is None:
                continue

            data = self._get_field_data(field, device, dtype)

            weight = dt if dt > 0 else 1.0

            # Mean accumulation
            if self._compute_mean:
                if fname not in self._mean_accum:
                    self._mean_accum[fname] = torch.zeros_like(data)
                if self._n_samples == 1:
                    self._mean_accum[fname] = data.clone()
                else:
                    # Incremental update: mean_new = mean_old + (data - mean_old) / n
                    alpha = weight / self._total_time
                    self._mean_accum[fname] += alpha * (data - self._mean_accum[fname])
                self._mean_fields[fname] = self._mean_accum[fname].clone()

            # Mean square accumulation
            if self._compute_mean2:
                if fname not in self._mean2_accum:
                    self._mean2_accum[fname] = torch.zeros_like(data)
                data2 = data * data
                if self._n_samples == 1:
                    self._mean2_accum[fname] = data2.clone()
                else:
                    alpha = weight / self._total_time
                    self._mean2_accum[fname] += alpha * (data2 - self._mean2_accum[fname])
                self._mean2_fields[fname] = self._mean2_accum[fname].clone()

            # Prime-squared mean (variance)
            if self._compute_prime2:
                if fname in self._mean_accum:
                    fluct = data - self._mean_accum[fname]
                    fluct2 = fluct * fluct
                    if fname not in self._prime2_accum:
                        self._prime2_accum[fname] = torch.zeros_like(data)
                    if self._n_samples <= 1:
                        self._prime2_accum[fname] = fluct2.clone()
                    else:
                        alpha = weight / self._total_time
                        self._prime2_accum[fname] += alpha * (fluct2 - self._prime2_accum[fname])
                    self._prime2_fields[fname] = self._prime2_accum[fname].clone()

        self._log.info(
            "t=%g  n_samples=%d  total_time=%.6g",
            time, self._n_samples, self._total_time,
        )

    def _get_field_data(self, field, device, dtype) -> torch.Tensor:
        """Extract raw tensor data from a field."""
        if hasattr(field, "internal_field"):
            return field.internal_field.to(device=device, dtype=dtype)
        return field.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Results access
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of accumulated samples."""
        return self._n_samples

    @property
    def mean_fields(self) -> Dict[str, torch.Tensor]:
        """Computed mean fields."""
        return self._mean_fields

    @property
    def mean2_fields(self) -> Dict[str, torch.Tensor]:
        """Computed mean-square fields."""
        return self._mean2_fields

    @property
    def prime2_fields(self) -> Dict[str, torch.Tensor]:
        """Computed variance fields (prime-squared mean)."""
        return self._prime2_fields

    def get_rms(self, field_name: str) -> Optional[torch.Tensor]:
        """Get the RMS of a field.

        Args:
            field_name: Name of the field.

        Returns:
            RMS tensor, or None if mean-square not computed.
        """
        if field_name in self._mean2_fields:
            return torch.sqrt(self._mean2_fields[field_name].clamp(min=0.0))
        return None

    def get_variance(self, field_name: str) -> Optional[torch.Tensor]:
        """Get the variance of a field (if prime2Mean computed).

        Args:
            field_name: Name of the field.

        Returns:
            Variance tensor, or None.
        """
        if field_name in self._prime2_fields:
            return self._prime2_fields[field_name]
        return None

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write averaged field data."""
        if self._output_path is None:
            return

        for fname in self._field_names:
            if fname in self._mean_fields:
                data = self._mean_fields[fname]
                self._write_field_file(self._output_path / f"{fname}_mean", fname, data)

            if fname in self._mean2_fields:
                data = self._mean2_fields[fname]
                self._write_field_file(self._output_path / f"{fname}_mean2", fname, data)

            if fname in self._prime2_fields:
                data = self._prime2_fields[fname]
                self._write_field_file(self._output_path / f"{fname}_prime2Mean", fname, data)

        # Write summary
        summary_file = self._output_path / "fieldAverage_summary.dat"
        with open(summary_file, "w") as f:
            f.write(f"# FieldAverage summary\n")
            f.write(f"# n_samples: {self._n_samples}\n")
            f.write(f"# total_time: {self._total_time:.6g}\n")
            f.write(f"# window_type: {self._window_type}\n")
            for fname in self._field_names:
                if fname in self._mean_fields:
                    mean = self._mean_fields[fname]
                    f.write(f"# {fname}_mean: avg={mean.mean():.10g}  min={mean.min():.10g}  max={mean.max():.10g}\n")

        logger.info("Wrote FieldAverage results to %s", self._output_path)

    def _write_field_file(self, filepath: Path, field_name: str, data: torch.Tensor) -> None:
        """Write an averaged field to a file."""
        n = data.shape[0]
        with open(filepath, "w") as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       volScalarField;\n")
            f.write(f'    location    "postProcessing/{self.name}";\n')
            f.write(f"    object      {field_name};\n")
            f.write("}\n\n")
            f.write("dimensions      [0 0 0 0 0 0 0];\n\n")
            f.write(f"internalField nonuniform {n}\n(\n")
            vals = data.cpu().numpy()
            for i in range(n):
                f.write(f"{vals[i]:.10g}\n")
            f.write(");\n\n")
            f.write("boundaryField\n{\n}\n")

    def finalise(self) -> None:
        """Reset accumulators."""
        self._mean_accum.clear()
        self._mean2_accum.clear()
        self._prime2_accum.clear()


# Register
FunctionObjectRegistry.register("fieldAverage", FieldAverage)
