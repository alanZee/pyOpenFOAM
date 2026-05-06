"""
FvMesh — finite volume mesh with computed geometric quantities.

Extends :class:`~pyfoam.mesh.poly_mesh.PolyMesh` with the geometric data
needed for FVM discretisation:

- Cell centres ``(n_cells, 3)``
- Cell volumes ``(n_cells,)``
- Face centres ``(n_faces, 3)``
- Face area vectors ``(n_faces, 3)`` (normal with magnitude = area)
- Face weights ``(n_faces,)`` — interpolation weights
- Delta coefficients ``(n_faces,)`` — diffusion distance factors

All quantities are computed lazily on first access and cached.
"""

from __future__ import annotations

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.mesh.poly_mesh import PolyMesh
from pyfoam.mesh.mesh_geometry import (
    compute_face_centres,
    compute_face_area_vectors,
    compute_cell_volumes_and_centres,
    compute_face_weights,
    compute_delta_coefficients,
)

__all__ = ["FvMesh"]


class FvMesh(PolyMesh):
    """Finite volume mesh extending :class:`PolyMesh` with geometric data.

    Geometric quantities are computed lazily on first access and cached.
    Call :meth:`compute_geometry` to pre-compute everything at once.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to :class:`PolyMesh`.

    Attributes
    ----------
    cell_centres : torch.Tensor
        ``(n_cells, 3)`` cell centre positions.
    cell_volumes : torch.Tensor
        ``(n_cells,)`` cell volumes.
    face_centres : torch.Tensor
        ``(n_faces, 3)`` face centre positions.
    face_areas : torch.Tensor
        ``(n_faces, 3)`` face area vectors (normal × area).
    face_weights : torch.Tensor
        ``(n_faces,)`` linear interpolation weights.
    delta_coefficients : torch.Tensor
        ``(n_faces,)`` diffusion delta coefficients.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Lazy-computed caches
        self._face_centres: torch.Tensor | None = None
        self._face_area_vectors: torch.Tensor | None = None
        self._cell_centres: torch.Tensor | None = None
        self._cell_volumes: torch.Tensor | None = None
        self._face_weights: torch.Tensor | None = None
        self._delta_coefficients: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Geometry computation
    # ------------------------------------------------------------------

    def compute_geometry(self) -> None:
        """Pre-compute all geometric quantities.

        This is equivalent to accessing every geometric property, but
        called explicitly it ensures everything is ready before the
        simulation loop starts.
        """
        self._compute_face_geometry()
        self._compute_cell_geometry()
        self._compute_interpolation_geometry()

    def _compute_face_geometry(self) -> None:
        """Compute face centres and area vectors."""
        if self._face_centres is None:
            self._face_centres = compute_face_centres(
                self._points, self._faces,
                device=self.device, dtype=self.dtype,
            )
        if self._face_area_vectors is None:
            self._face_area_vectors = compute_face_area_vectors(
                self._points, self._faces,
                device=self.device, dtype=self.dtype,
            )

    def _compute_cell_geometry(self) -> None:
        """Compute cell volumes and centres."""
        if self._cell_volumes is not None and self._cell_centres is not None:
            return
        self._compute_face_geometry()
        volumes, centres = compute_cell_volumes_and_centres(
            self._points,
            self._faces,
            self._owner,
            self._neighbour,
            self._n_cells,
            self._n_internal_faces,
            face_centres=self._face_centres,
            face_area_vectors=self._face_area_vectors,
            device=self.device,
            dtype=self.dtype,
        )
        self._cell_volumes = volumes
        self._cell_centres = centres

    def _compute_interpolation_geometry(self) -> None:
        """Compute face weights and delta coefficients."""
        self._compute_face_geometry()
        self._compute_cell_geometry()

        if self._face_weights is None:
            self._face_weights = compute_face_weights(
                self._cell_centres,
                self._face_centres,
                self._owner,
                self._neighbour,
                self._n_internal_faces,
                device=self.device,
                dtype=self.dtype,
            )
        if self._delta_coefficients is None:
            self._delta_coefficients = compute_delta_coefficients(
                self._cell_centres,
                self._face_centres,
                self._face_area_vectors,
                self._owner,
                self._neighbour,
                self._n_internal_faces,
                device=self.device,
                dtype=self.dtype,
            )

    # ------------------------------------------------------------------
    # Properties (lazy)
    # ------------------------------------------------------------------

    @property
    def face_centres(self) -> torch.Tensor:
        """Face centre positions ``(n_faces, 3)``."""
        if self._face_centres is None:
            self._compute_face_geometry()
        return self._face_centres  # type: ignore[return-value]

    @property
    def face_areas(self) -> torch.Tensor:
        """Face area vectors ``(n_faces, 3)`` (normal × area)."""
        if self._face_area_vectors is None:
            self._compute_face_geometry()
        return self._face_area_vectors  # type: ignore[return-value]

    @property
    def cell_centres(self) -> torch.Tensor:
        """Cell centre positions ``(n_cells, 3)``."""
        if self._cell_centres is None:
            self._compute_cell_geometry()
        return self._cell_centres  # type: ignore[return-value]

    @property
    def cell_volumes(self) -> torch.Tensor:
        """Cell volumes ``(n_cells,)``."""
        if self._cell_volumes is None:
            self._compute_cell_geometry()
        return self._cell_volumes  # type: ignore[return-value]

    @property
    def face_weights(self) -> torch.Tensor:
        """Face interpolation weights ``(n_faces,)``."""
        if self._face_weights is None:
            self._compute_interpolation_geometry()
        return self._face_weights  # type: ignore[return-value]

    @property
    def delta_coefficients(self) -> torch.Tensor:
        """Face delta coefficients ``(n_faces,)``."""
        if self._delta_coefficients is None:
            self._compute_interpolation_geometry()
        return self._delta_coefficients  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def face_areas_magnitude(self) -> torch.Tensor:
        """Face area magnitudes ``(n_faces,)``."""
        return self.face_areas.norm(dim=1)

    @property
    def face_normals(self) -> torch.Tensor:
        """Unit face normals ``(n_faces, 3)``."""
        area_vecs = self.face_areas
        mag = area_vecs.norm(dim=1, keepdim=True)
        safe_mag = torch.where(mag > 1e-30, mag, torch.ones_like(mag))
        return area_vecs / safe_mag

    @property
    def total_volume(self) -> torch.Tensor:
        """Sum of all cell volumes (scalar tensor)."""
        return self.cell_volumes.sum()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_poly_mesh(cls, mesh: PolyMesh) -> "FvMesh":
        """Create an FvMesh from an existing PolyMesh.

        Copies the topology and computes geometric quantities.

        Args:
            mesh: Source :class:`PolyMesh`.

        Returns:
            New :class:`FvMesh` with geometry computed.
        """
        fv = cls(
            points=mesh.points,
            faces=mesh.faces,
            owner=mesh.owner,
            neighbour=mesh.neighbour,
            boundary=mesh.boundary,
            validate=False,  # already validated
        )
        fv.compute_geometry()
        return fv

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FvMesh(n_points={self._n_points}, n_faces={self._n_faces}, "
            f"n_cells={self._n_cells}, n_internal_faces={self._n_internal_faces}, "
            f"n_patches={len(self._boundary)}, "
            f"device={self.device}, dtype={self.dtype})"
        )
