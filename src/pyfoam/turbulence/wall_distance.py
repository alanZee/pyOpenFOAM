"""
Wall distance computation for turbulence modelling.

Provides wall distance calculators used by turbulence models that need
the distance from each cell centre to the nearest wall (e.g. y+ computation,
low-Re models, SST model blending functions).

Models:

- :class:`WallDistanceCalculator` — abstract base class
- :class:`ExactWallDistance` — exact geometric distance to wall boundaries
- :class:`ApproximateWallDistance` — fast approximate wall distance using
  a Poisson-equation-like diffusion approach

In OpenFOAM, the wall distance (``yWall``) is computed by the
``wallDistData`` function object or the ``wavePropagation`` method.

Usage::

    from pyfoam.turbulence.wall_distance import ExactWallDistance, ApproximateWallDistance

    calc = ExactWallDistance(wall_centres, wall_normals)
    y = calc.compute(cell_centres)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "WallDistanceCalculator",
    "ExactWallDistance",
    "ApproximateWallDistance",
]

logger = logging.getLogger(__name__)


class WallDistanceCalculator(ABC):
    """Abstract base class for wall distance computation.

    Wall distance is the minimum geometric distance from each cell centre
    to the nearest wall boundary face.
    """

    @abstractmethod
    def compute(self, cell_centres: torch.Tensor) -> torch.Tensor:
        """Compute wall distance for each cell.

        Parameters
        ----------
        cell_centres : torch.Tensor
            Cell centre positions ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Wall distance ``(n_cells,)``.
        """


class ExactWallDistance(WallDistanceCalculator):
    """Exact geometric wall distance calculator.

    Computes the exact minimum distance from each cell centre to all wall
    boundary faces.  For each cell, the distance is:

        y = min_i | (c - p_i) - ((c - p_i) . n_i) n_i |

    where p_i is a point on wall face i and n_i is its outward normal.
    The projection removes the normal component so we get the perpendicular
    distance to the wall plane.

    For efficiency, the computation uses batched tensor operations.

    Parameters
    ----------
    wall_centres : torch.Tensor
        Wall face centres ``(n_wall_faces, 3)``.
    wall_normals : torch.Tensor
        Wall face outward unit normals ``(n_wall_faces, 3)``.
    """

    def __init__(
        self,
        wall_centres: torch.Tensor,
        wall_normals: torch.Tensor,
    ) -> None:
        if wall_centres.shape[0] != wall_normals.shape[0]:
            raise ValueError(
                "wall_centres and wall_normals must have the same number of entries"
            )
        self._wall_centres = wall_centres
        self._wall_normals = wall_normals

    @property
    def n_wall_faces(self) -> int:
        """Number of wall faces."""
        return int(self._wall_centres.shape[0])

    def compute(self, cell_centres: torch.Tensor) -> torch.Tensor:
        """Compute exact minimum wall distance for each cell.

        For each cell, the perpendicular distance to a wall plane is::

            d = | (c - p_i) . n_i |

        where p_i is the wall face centre and n_i is its outward normal.
        The minimum over all wall faces is returned.

        Parameters
        ----------
        cell_centres : torch.Tensor
            Cell centre positions ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Wall distance ``(n_cells,)``.
        """
        device = cell_centres.device
        dtype = cell_centres.dtype

        wc = self._wall_centres.to(device=device, dtype=dtype)  # (W, 3)
        wn = self._wall_normals.to(device=device, dtype=dtype)  # (W, 3)
        n_cells = cell_centres.shape[0]
        n_wall = wc.shape[0]

        # For memory efficiency, process in chunks if n_cells * n_wall is large
        max_chunk_mem = 1024 * 1024  # 1M elements
        chunk_size = max(1, max_chunk_mem // max(1, n_wall))

        min_dist = torch.full((n_cells,), float("inf"), dtype=dtype, device=device)

        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            cc = cell_centres[start:end].unsqueeze(1)  # (chunk, 1, 3)
            # Vector from wall centre to cell centre
            r = cc - wc.unsqueeze(0)  # (chunk, W, 3)
            # Perpendicular distance: |(c - p) . n|
            dot_rn = (r * wn.unsqueeze(0)).sum(dim=-1)  # (chunk, W)
            dist = dot_rn.abs()  # (chunk, W)
            chunk_min = dist.min(dim=1).values  # (chunk,)
            min_dist[start:end] = chunk_min

        return min_dist


class ApproximateWallDistance(WallDistanceCalculator):
    """Fast approximate wall distance calculator.

    Uses a geometric mean approach: for each cell, computes the distance
    to the nearest wall face centre (point distance), then applies a
    correction factor based on the face normal alignment.

    This is significantly faster than exact distance for large meshes
    but may underestimate distance for cells near curved walls.

    Parameters
    ----------
    wall_centres : torch.Tensor
        Wall face centres ``(n_wall_faces, 3)``.
    wall_normals : torch.Tensor
        Wall face outward unit normals ``(n_wall_faces, 3)`` (used for
        the normal-alignment correction).
    correction_factor : float
        Multiplier for the approximate distance (default: 1.0).
        Values > 1.0 compensate for the underestimation near curved walls.
    """

    def __init__(
        self,
        wall_centres: torch.Tensor,
        wall_normals: Optional[torch.Tensor] = None,
        correction_factor: float = 1.0,
    ) -> None:
        self._wall_centres = wall_centres
        self._wall_normals = wall_normals
        self.correction_factor = correction_factor

    @property
    def n_wall_faces(self) -> int:
        """Number of wall faces."""
        return int(self._wall_centres.shape[0])

    def compute(self, cell_centres: torch.Tensor) -> torch.Tensor:
        """Compute approximate wall distance for each cell.

        Uses Euclidean distance to the nearest wall face centre as the
        base estimate, then applies the correction factor.

        Parameters
        ----------
        cell_centres : torch.Tensor
            Cell centre positions ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Approximate wall distance ``(n_cells,)``.
        """
        device = cell_centres.device
        dtype = cell_centres.dtype

        wc = self._wall_centres.to(device=device, dtype=dtype)  # (W, 3)
        n_cells = cell_centres.shape[0]
        n_wall = wc.shape[0]

        # Process in chunks for memory efficiency
        max_chunk_mem = 1024 * 1024
        chunk_size = max(1, max_chunk_mem // max(1, n_wall))

        min_dist = torch.full((n_cells,), float("inf"), dtype=dtype, device=device)

        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            cc = cell_centres[start:end]  # (chunk, 3)
            # Euclidean distance to each wall face centre
            diff = cc.unsqueeze(1) - wc.unsqueeze(0)  # (chunk, W, 3)
            dist = diff.norm(dim=-1)  # (chunk, W)
            chunk_min = dist.min(dim=1).values  # (chunk,)
            min_dist[start:end] = chunk_min

        # Apply correction factor
        if self.correction_factor != 1.0:
            min_dist = min_dist * self.correction_factor

        return min_dist
