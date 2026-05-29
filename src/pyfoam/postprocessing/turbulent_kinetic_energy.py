"""
TurbulentKineticEnergy — TKE computation function object.

Computes the turbulent kinetic energy k from fluctuating velocity
components or from the Reynolds stress tensor.

Physics
-------
For resolved (DNS/LES) fields, TKE is computed from velocity fluctuations:

    k = 0.5 * (u'² + v'² + w'²)

where u', v', w' are fluctuations about the mean. The mean can be
provided as a pre-computed field or computed via a running average.

For RANS models that directly provide k, this function object can
read and output the k field.

References
----------
- Pope, "Turbulent Flows", Cambridge University Press, 2000
- OpenFOAM ``fieldAverage`` function object (used for computing
  mean fields needed for fluctuation extraction)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["TurbulentKineticEnergy"]

logger = logging.getLogger(__name__)


class TurbulentKineticEnergy(FunctionObject):
    """Compute turbulent kinetic energy.

    Two modes:

    1. **Resolved mode** (DNS/LES): compute from velocity fluctuations
       ``k = 0.5 * (u'^2 + v'^2 + w'^2)``. Requires a mean velocity
       field ``UMean`` or ``U_avg``.

    2. **RANS mode**: read k directly from a ``k`` field (e.g. k-epsilon
       or k-omega model output).

    Configuration keys:

    - ``mode``: ``"resolved"`` or ``"rans"`` (default: ``"resolved"``)
    - ``field``: velocity field name (default: ``"U"``)
    - ``meanField``: mean velocity field name (default: ``"UMean"``)
    - ``writeField``: if True, write k to time directories (default: False)

    Example controlDict entry::

        TKE1
        {
            type            TurbulentKineticEnergy;
            libs            ("libfieldFunctionObjects.so");
            mode            resolved;
            field           U;
            meanField       UMean;
            writeField      true;
        }

    Output shape: ``(n_cells,)`` — one scalar TKE value per cell.
    """

    def __init__(self, name: str = "TKE", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._mode: str = self.config.get("mode", "resolved")
        self._field_name: str = self.config.get("field", "U")
        self._mean_field_name: str = self.config.get("meanField", "UMean")
        self._write_field: bool = self.config.get("writeField", False)

        if self._mode not in ("resolved", "rans"):
            raise ValueError(f"Unknown mode '{self._mode}'. Use 'resolved' or 'rans'.")

        # Results
        self._k: Optional[torch.Tensor] = None
        self._times: List[float] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and fields."""
        self._mesh = mesh
        self._fields = fields
        logger.info(
            "TurbulentKineticEnergy '%s' initialised: mode=%s field=%s",
            self.name, self._mode, self._field_name
        )

    def execute(self, time: float) -> None:
        """Compute TKE at current time step."""
        if not self._enabled or self._mesh is None:
            return

        if self._mode == "rans":
            k = self._compute_rans()
        else:
            k = self._compute_resolved()

        if k is None:
            return

        self._k = k.detach().cpu()
        self._times.append(time)

        self._log.info(
            "t=%g  TKE computed  k_avg=%.6e  k_max=%.6e",
            time, k.mean().item(), k.max().item()
        )

    def _compute_resolved(self) -> Optional[torch.Tensor]:
        """Compute TKE from velocity fluctuations (DNS/LES).

        Returns:
            ``(n_cells,)`` TKE field, or None if fields missing.
        """
        device = get_device()
        dtype = get_default_dtype()

        U = self._fields.get(self._field_name)
        if U is None:
            logger.warning("Field '%s' required for TKE. Skipping.", self._field_name)
            return None

        U_mean = self._fields.get(self._mean_field_name)
        if U_mean is None:
            logger.warning(
                "Mean field '%s' not found. Cannot compute fluctuations. "
                "Available: %s",
                self._mean_field_name, list(self._fields.keys())
            )
            return None

        if hasattr(U, "internal_field"):
            U_data = U.internal_field.to(device=device, dtype=dtype)
        else:
            U_data = U.to(device=device, dtype=dtype)

        if hasattr(U_mean, "internal_field"):
            U_mean_data = U_mean.internal_field.to(device=device, dtype=dtype)
        else:
            U_mean_data = U_mean.to(device=device, dtype=dtype)

        # Fluctuations
        u_prime = U_data - U_mean_data

        # k = 0.5 * (u'^2 + v'^2 + w'^2)
        k = 0.5 * (u_prime * u_prime).sum(dim=1)

        return k

    def _compute_rans(self) -> Optional[torch.Tensor]:
        """Read k from RANS model field.

        Returns:
            ``(n_cells,)`` k field, or None if not found.
        """
        device = get_device()
        dtype = get_default_dtype()

        k_field = self._fields.get("k")
        if k_field is None:
            logger.warning("RANS k field not found. Available: %s", list(self._fields.keys()))
            return None

        if hasattr(k_field, "internal_field"):
            k_data = k_field.internal_field.to(device=device, dtype=dtype)
        else:
            k_data = k_field.to(device=device, dtype=dtype)

        return k_data

    @property
    def k_field(self) -> Optional[torch.Tensor]:
        """Computed TKE field ``(n_cells,)``."""
        return self._k

    @property
    def mode(self) -> str:
        """Computation mode (``"resolved"`` or ``"rans"``)."""
        return self._mode

    @property
    def times(self) -> List[float]:
        """Time values."""
        return self._times

    def write(self) -> None:
        """Write TKE data to output files."""
        if self._output_path is None or self._k is None:
            return

        info_file = self._output_path / "TKE.info"
        with open(info_file, "w") as f:
            f.write("# Turbulent Kinetic Energy (TKE)\n")
            f.write(f"# Mode: {self._mode}\n")
            f.write(f"# Field: {self._field_name}\n")
            f.write(f"# Shape: {self._k.shape}\n")
            f.write(f"# k_avg: {self._k.mean().item():.6e}\n")
            f.write(f"# k_max: {self._k.max().item():.6e}\n")
            f.write(f"# Times computed: {len(self._times)}\n")
        logger.info("Wrote TKE info to %s", info_file)


# Register
FunctionObjectRegistry.register("TKE", TurbulentKineticEnergy)
FunctionObjectRegistry.register("turbulentKineticEnergy", TurbulentKineticEnergy)
