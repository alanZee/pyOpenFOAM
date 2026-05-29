"""
FieldMinMax — field minimum/maximum analysis function object.

Computes the global and per-patch min/max of scalar and vector fields
at each time step.  Reports extrema locations (cell indices) when
available.

Configuration keys:

- ``field``: field name to analyse
- ``writeInterval``: interval for writing results
- ``log``: if True, log results each time step

Usage::

    fmm = FieldMinMax("fieldMinMax1", {"field": "p"})
    fmm.initialise(mesh, {"p": p_field})
    fmm.execute(1.0)

References
----------
- OpenFOAM ``fieldMinMax`` function object source
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["FieldMinMax", "MinMaxResult"]

logger = logging.getLogger(__name__)


@dataclass
class MinMaxResult:
    """Result of a min/max analysis at a single time step.

    Attributes:
        time: Simulation time.
        field_name: Name of the analysed field.
        min_value: Global minimum value (scalar) or magnitude minimum.
        max_value: Global maximum value (scalar) or magnitude maximum.
        min_location: Cell index of minimum (if available).
        max_location: Cell index of maximum (if available).
    """

    time: float = 0.0
    field_name: str = ""
    min_value: float = 0.0
    max_value: float = 0.0
    min_location: int = -1
    max_location: int = -1


class FieldMinMax(FunctionObject):
    """Compute min/max of a field at each time step.

    Configuration keys:

    - ``field``: field name to analyse (required)
    - ``log``: if True, print min/max to log each step (default: True)
    """

    def __init__(self, name: str = "fieldMinMax", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._field_name: str = self.config.get("field", "")
        self._do_log: bool = self.config.get("log", True)
        self._results: List[MinMaxResult] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and fields."""
        self._mesh = mesh
        self._fields = fields
        logger.info("FieldMinMax '%s' initialised for field '%s'", self.name, self._field_name)

    def execute(self, time: float) -> None:
        """Compute min/max at the current time step."""
        if not self._enabled:
            return

        field = self._fields.get(self._field_name)
        if field is None:
            logger.warning("Field '%s' not found. Skipping.", self._field_name)
            return

        result = self._compute_min_max(field, time)
        self._results.append(result)

        if self._do_log:
            self._log.info(
                "t=%g  field='%s'  min=%.6g (cell %d)  max=%.6g (cell %d)",
                result.time, result.field_name,
                result.min_value, result.min_location,
                result.max_value, result.max_location,
            )

    def _compute_min_max(self, field, time: float) -> MinMaxResult:
        """Compute min/max of a field."""
        device = get_device()
        dtype = get_default_dtype()

        if hasattr(field, "internal_field"):
            data = field.internal_field.to(device=device, dtype=dtype)
        else:
            data = field.to(device=device, dtype=dtype)

        if data.dim() == 1:
            # Scalar field
            min_val = data.min()
            max_val = data.max()
            min_loc = int(data.argmin().item())
            max_loc = int(data.argmax().item())
        elif data.dim() == 2 and data.shape[1] == 3:
            # Vector field: use magnitude
            mag = data.norm(dim=1)
            min_val = mag.min()
            max_val = mag.max()
            min_loc = int(mag.argmin().item())
            max_loc = int(mag.argmax().item())
        else:
            min_val = data.min()
            max_val = data.max()
            min_loc = -1
            max_loc = -1

        return MinMaxResult(
            time=time,
            field_name=self._field_name,
            min_value=min_val.item(),
            max_value=max_val.item(),
            min_location=min_loc,
            max_location=max_loc,
        )

    @property
    def results(self) -> List[MinMaxResult]:
        """All computed min/max results."""
        return self._results

    def write(self) -> None:
        """Write min/max history to a data file."""
        if self._output_path is None or not self._results:
            return

        outfile = self._output_path / "fieldMinMax.dat"
        with open(outfile, "w") as f:
            f.write("# Time  min  minCell  max  maxCell\n")
            for r in self._results:
                f.write(
                    f"{r.time:.6e}  {r.min_value:.10g}  {r.min_location}  "
                    f"{r.max_value:.10g}  {r.max_location}\n"
                )
        logger.info("Wrote FieldMinMax to %s", outfile)


# Register
FunctionObjectRegistry.register("fieldMinMax", FieldMinMax)
