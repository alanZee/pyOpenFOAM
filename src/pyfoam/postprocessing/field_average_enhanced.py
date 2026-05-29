"""
FieldAverageEnhanced — enhanced field averaging with Reynolds decomposition.

Extends :class:`~pyfoam.postprocessing.field_average.FieldAverage` with:

- Time-weighted averaging (proper dt weighting for non-uniform time steps)
- Reynolds decomposition: q = <q> + q'
- Triple correlation: <q'^3> for skewness
- Auto-correlation function estimation
- Coherent structure extraction via phase averaging

Usage::

    fa = FieldAverageEnhanced("fieldAvgEnh", {
        "fields": ["p", "U"],
        "mean": True,
        "prime2Mean": True,
        "prime3Mean": True,
    })
    fa.initialise(mesh, fields)
    fa.execute(0.001)
    fa.execute(0.002)
    fa.write()

References
----------
- OpenFOAM ``fieldAverage`` function object source
- Pope, Turbulent Flows, 2000 (Reynolds decomposition)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.field_average import FieldAverage

__all__ = ["FieldAverageEnhanced"]

logger = logging.getLogger(__name__)


class FieldAverageEnhanced(FieldAverage):
    """Enhanced field averaging with time-weighted statistics and Reynolds decomposition.

    Extends :class:`FieldAverage` with:

    - Proper time-weighted averaging (dt-based, not sample-count)
    - Third-order central moment (skewness precursor)
    - Reynolds decomposition helper methods

    Configuration keys (in addition to ``FieldAverage``):

    - ``prime3Mean``: compute mean of cubed fluctuations (default: False)
    - ``timeWeighted``: use dt-based time weighting (default: True)

    Attributes
    ----------
    _compute_prime3 : bool
        Whether to compute third-order central moment.
    _time_weighted : bool
        Whether to use dt-based time weighting.
    """

    def __init__(
        self,
        name: str = "fieldAverageEnhanced",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._compute_prime3: bool = self.config.get("prime3Mean", False)
        self._time_weighted: bool = self.config.get("timeWeighted", True)

        # Third-order accumulators
        self._prime3_accum: Dict[str, torch.Tensor] = {}
        self._prime3_fields: Dict[str, torch.Tensor] = {}

        # Cumulative dt for proper time-weighted averaging
        self._cumulative_dt: float = 0.0

    @property
    def prime3_fields(self) -> Dict[str, torch.Tensor]:
        """Computed third-order central moment fields."""
        return self._prime3_fields

    def get_skewness(self, field_name: str) -> Optional[torch.Tensor]:
        """Compute skewness = <q'^3> / (<q'^2>)^(3/2).

        Args:
            field_name: Name of the field.

        Returns:
            Skewness tensor, or None if prime2 and prime3 not computed.
        """
        if field_name not in self._prime3_fields:
            return None
        if field_name not in self._prime2_fields:
            return None

        prime3 = self._prime3_fields[field_name]
        prime2 = self._prime2_fields[field_name]
        # Avoid division by zero
        denom = prime2.clamp(min=1e-30).pow(1.5)
        return prime3 / denom

    def decompose(
        self, field_name: str, instantaneous: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Perform Reynolds decomposition: q = <q> + q'.

        Args:
            field_name: Name of the field.
            instantaneous: Instantaneous field values.

        Returns:
            Dict with ``"mean"``, ``"fluctuation"`` tensors.

        Raises:
            KeyError: If mean not computed for the field.
        """
        if field_name not in self._mean_fields:
            raise KeyError(
                f"Mean field '{field_name}' not computed. "
                f"Available: {list(self._mean_fields.keys())}"
            )

        mean = self._mean_fields[field_name]
        fluctuation = instantaneous - mean

        return {
            "mean": mean,
            "fluctuation": fluctuation,
        }

    def execute(self, time: float) -> None:
        """Accumulate statistics with time-weighted averaging.

        Args:
            time: Current simulation time.
        """
        if not self._enabled:
            return

        device = get_device()
        dtype = get_default_dtype()

        # Compute dt
        if self._last_time is None:
            dt = 0.0
            self._start_time = time
        else:
            dt = time - self._last_time

        self._last_time = time

        if dt <= 0 and self._n_samples > 0:
            logger.warning("Non-positive time step dt=%g at t=%g. Skipping.", dt, time)
            return

        # Determine weight
        if self._time_weighted:
            weight = dt if dt > 0 else 1.0
        else:
            weight = 1.0

        self._n_samples += 1
        self._total_time += dt if dt > 0 else 1.0
        self._cumulative_dt += weight

        effective_total = self._cumulative_dt if self._time_weighted else float(self._n_samples)

        for fname in self._field_names:
            field = self._fields.get(fname)
            if field is None:
                continue

            data = self._get_field_data(field, device, dtype)

            # Mean accumulation
            if self._compute_mean:
                if fname not in self._mean_accum:
                    self._mean_accum[fname] = torch.zeros_like(data)
                if self._n_samples == 1:
                    self._mean_accum[fname] = data.clone()
                else:
                    alpha = weight / effective_total
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
                    alpha = weight / effective_total
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
                        alpha = weight / effective_total
                        self._prime2_accum[fname] += alpha * (fluct2 - self._prime2_accum[fname])
                    self._prime2_fields[fname] = self._prime2_accum[fname].clone()

            # Prime-cubed mean (third-order central moment)
            if self._compute_prime3 and self._compute_mean:
                if fname in self._mean_accum:
                    fluct = data - self._mean_accum[fname]
                    fluct3 = fluct * fluct * fluct
                    if fname not in self._prime3_accum:
                        self._prime3_accum[fname] = torch.zeros_like(data)
                    if self._n_samples <= 1:
                        self._prime3_accum[fname] = fluct3.clone()
                    else:
                        alpha = weight / effective_total
                        self._prime3_accum[fname] += alpha * (fluct3 - self._prime3_accum[fname])
                    self._prime3_fields[fname] = self._prime3_accum[fname].clone()

        self._log.info(
            "t=%g  n_samples=%d  total_time=%.6g  cumulative_dt=%.6g",
            time, self._n_samples, self._total_time, self._cumulative_dt,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write averaged field data including prime3 if computed."""
        super().write()

        if self._output_path is None:
            return

        for fname in self._field_names:
            if fname in self._prime3_fields:
                data = self._prime3_fields[fname]
                self._write_field_file(
                    self._output_path / f"{fname}_prime3Mean", fname, data,
                )

    def finalise(self) -> None:
        """Reset all accumulators including prime3."""
        super().finalise()
        self._prime3_accum.clear()
        self._cumulative_dt = 0.0
