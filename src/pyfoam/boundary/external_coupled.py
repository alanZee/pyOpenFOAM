"""
External coupled boundary condition.

Couples the boundary with an external solver via file-based data exchange.
The boundary reads field values from a file written by the external solver,
and writes boundary values to a file read by the external solver.

This pattern is used for co-simulation scenarios (e.g., FSI, CHt with
external codes) where the two solvers run alternately and communicate
through the filesystem.

Data exchange files:
    - ``<dir>/<patchName>_T`` — temperature written / read
    - ``<dir>/<patchName>_q`` — heat flux written / read
    (File format: OpenFOAM boundary field format, one value per line)

In OpenFOAM syntax::

    type            externalCoupled;
    commsDir        "$FOAM_CASE/externalCoupling";
    transformModel  linear;
    nSample         100;           // number of sample points
    positions       ((0 0 0)(1 0 0));
    value           uniform 300;

Usage::

    bc = BoundaryCondition.create("externalCoupled", patch, coeffs={
        "commsDir": "/tmp/externalCoupling",
        "nSample": 100,
    })
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ExternalCoupledBC"]


@BoundaryCondition.register("externalCoupled")
class ExternalCoupledBC(BoundaryCondition):
    """External coupled boundary condition for co-simulation.

    Communicates boundary field values with an external solver via
    file-based data exchange.  The BC reads values from a file
    (written by the external solver) and writes values to a file
    (read by the external solver).

    Coefficients:
        - ``commsDir``: Communication directory (default: ``"/tmp/externalCoupling"``).
        - ``nSample``: Number of sample points (informational).  Default 0.
        - ``positions``: Sample positions (informational).
        - ``transformModel``: Coordinate transform model (informational).
        - ``value``: Initial / fallback value (default: 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._comms_dir = self._coeffs.get("commsDir", "/tmp/externalCoupling")
        self._n_sample = int(self._coeffs.get("nSample", 0))
        self._positions = self._coeffs.get("positions", None)
        self._transform_model = self._coeffs.get("transformModel", "linear")

        # Internal buffer: populated by load(), consumed by apply()
        self._external_values: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def comms_dir(self) -> str:
        """Communication directory path."""
        return self._comms_dir

    @property
    def n_sample(self) -> int:
        """Number of sample points."""
        return self._n_sample

    @property
    def transform_model(self) -> str:
        """Coordinate transform model name."""
        return self._transform_model

    @property
    def external_values(self) -> torch.Tensor | None:
        """Buffered external field values, or ``None`` if not loaded."""
        return self._external_values

    # ------------------------------------------------------------------
    # Data exchange
    # ------------------------------------------------------------------

    def load(self, path: Path | str | None = None) -> torch.Tensor:
        """Load boundary values from the external solver file.

        Reads one scalar value per line (OpenFOAM boundary data format).
        Falls back to the ``value`` coefficient if the file does not exist.

        Args:
            path: Override file path.  When ``None`` the default path
                  ``<commsDir>/<patchName>`` is used.

        Returns:
            Loaded values tensor ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        n = self._patch.n_faces

        if path is None:
            path = Path(self._comms_dir) / self._patch.name
        else:
            path = Path(path)

        if path.exists():
            values: list[float] = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("(") or line.startswith(")"):
                        continue
                    try:
                        values.append(float(line))
                    except ValueError:
                        continue

            if len(values) >= n:
                self._external_values = torch.tensor(
                    values[:n], dtype=dtype, device=device,
                )
            else:
                # Pad with fallback value
                fallback = float(self._coeffs.get("value", 0.0))
                padded = values + [fallback] * (n - len(values))
                self._external_values = torch.tensor(
                    padded, dtype=dtype, device=device,
                )
        else:
            # File not found: use fallback value
            fallback = float(self._coeffs.get("value", 0.0))
            self._external_values = torch.full(
                (n,), fallback, dtype=dtype, device=device,
            )

        return self._external_values

    def save(self, field: torch.Tensor, path: Path | str | None = None) -> None:
        """Write boundary values to file for the external solver.

        Args:
            field: Full field tensor.
            path: Override file path.  When ``None`` the default path
                  ``<commsDir>/<patchName>`` is used.
        """
        if path is None:
            path = Path(self._comms_dir) / self._patch.name
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        vals = field[self._patch.face_indices].detach().cpu()
        with open(path, "w") as f:
            for v in vals:
                f.write(f"{v.item():.10e}\n")

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply external coupled BC.

        If external values have been loaded (via :meth:`load`), writes
        them to the boundary faces.  Otherwise, falls back to zero-gradient
        (copies owner cell values).

        Args:
            field: Field tensor.
            patch_idx: Optional start index into *field*.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if self._external_values is not None:
            vals = self._external_values[:n].to(device=device, dtype=dtype)
        else:
            # Zero-gradient fallback
            owners = self._patch.owner_cells.to(device=device)
            vals = field[owners]

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = vals
        else:
            field[self._patch.face_indices] = vals
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method when external values are available.

        When external values are loaded: large penalty towards the
        external value.  Otherwise: zero contribution (zero-gradient).
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        if self._external_values is None:
            return diag, source

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)

        ext_vals = self._external_values[: self._patch.n_faces].to(
            device=device, dtype=dtype,
        )
        source.scatter_add_(0, owners, coeff * ext_vals)

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
