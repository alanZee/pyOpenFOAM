"""
PolyMesh — the fundamental mesh representation.

Stores the raw topological data of a polyhedral mesh:

- **points** — vertex positions ``(n_points, 3)``
- **faces** — list of point-index tensors (one per face)
- **owner** — owner cell index per face ``(n_faces,)``
- **neighbour** — neighbour cell index per internal face ``(n_internal_faces,)``
- **boundary** — list of boundary patch descriptors

This mirrors OpenFOAM's ``polyMesh`` class.  The class does **not** compute
any geometric quantities — use :class:`~pyfoam.mesh.fv_mesh.FvMesh` for that.

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core`.
"""

from __future__ import annotations

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.topology import validate_owner_neighbour

__all__ = ["PolyMesh"]


class PolyMesh:
    """Polyhedral mesh storing raw topology.

    Parameters
    ----------
    points : torch.Tensor
        ``(n_points, 3)`` vertex positions.
    faces : list[torch.Tensor]
        Each element is a 1-D int tensor of point indices for one face.
    owner : torch.Tensor
        ``(n_faces,)`` owner cell index for each face.
    neighbour : torch.Tensor
        ``(n_internal_faces,)`` neighbour cell index for each internal face.
    boundary : list[dict]
        Boundary patch descriptors.  Each dict has keys:
        ``name`` (str), ``type`` (str), ``startFace`` (int), ``nFaces`` (int).
    validate : bool
        If ``True`` (default), validate owner/neighbour conventions on
        construction.

    Attributes
    ----------
    n_points : int
        Number of vertices.
    n_faces : int
        Total number of faces (internal + boundary).
    n_cells : int
        Number of cells (inferred from owner array).
    n_internal_faces : int
        Number of internal faces (length of neighbour array).
    """

    def __init__(
        self,
        points: torch.Tensor,
        faces: list[torch.Tensor],
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        boundary: list[dict] | None = None,
        *,
        validate: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device = torch.device(device) if device is not None else get_device()
        dtype = dtype or get_default_dtype()

        # Store tensors on the configured device
        self._points = points.to(device=device, dtype=dtype)
        self._faces = [f.to(device=device, dtype=INDEX_DTYPE) for f in faces]
        self._owner = owner.to(device=device, dtype=INDEX_DTYPE)
        self._neighbour = neighbour.to(device=device, dtype=INDEX_DTYPE)
        self._boundary = boundary if boundary is not None else []

        # Derived counts
        self._n_points = self._points.shape[0]
        self._n_faces = len(self._faces)
        self._n_internal_faces = self._neighbour.shape[0]
        # n_cells = max(owner) + 1 (cells are 0-indexed)
        if self._owner.numel() > 0:
            self._n_cells = int(self._owner.max().item()) + 1
        else:
            self._n_cells = 0

        # Validate
        if validate:
            validate_owner_neighbour(
                self._owner,
                self._neighbour,
                self._n_cells,
                self._n_internal_faces,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def points(self) -> torch.Tensor:
        """Vertex positions ``(n_points, 3)``."""
        return self._points

    @property
    def faces(self) -> list[torch.Tensor]:
        """List of face-vertex index tensors."""
        return self._faces

    @property
    def owner(self) -> torch.Tensor:
        """Owner cell index per face ``(n_faces,)``."""
        return self._owner

    @property
    def neighbour(self) -> torch.Tensor:
        """Neighbour cell index per internal face ``(n_internal_faces,)``."""
        return self._neighbour

    @property
    def boundary(self) -> list[dict]:
        """Boundary patch descriptors."""
        return self._boundary

    @property
    def n_points(self) -> int:
        """Number of vertices."""
        return self._n_points

    @property
    def n_faces(self) -> int:
        """Total number of faces."""
        return self._n_faces

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self._n_cells

    @property
    def n_internal_faces(self) -> int:
        """Number of internal faces."""
        return self._n_internal_faces

    @property
    def device(self) -> torch.device:
        """Device tensors reside on."""
        return self._points.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype for geometric data."""
        return self._points.dtype

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def face_points(self, face_idx: int) -> torch.Tensor:
        """Return vertex positions for face *face_idx*.

        Returns:
            ``(n_vertices_in_face, 3)`` tensor.
        """
        return self._points[self._faces[face_idx]]

    def is_boundary_face(self, face_idx: int) -> bool:
        """Return ``True`` if *face_idx* is a boundary face."""
        return face_idx >= self._n_internal_faces

    def patch_faces(self, patch_idx: int) -> range:
        """Return the range of face indices for boundary patch *patch_idx*.

        Args:
            patch_idx: Index into :attr:`boundary`.

        Returns:
            ``range(startFace, startFace + nFaces)``.
        """
        patch = self._boundary[patch_idx]
        start = patch["startFace"]
        n = patch["nFaces"]
        return range(start, start + n)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_raw(
        cls,
        points: list[list[float]],
        faces: list[list[int]],
        owner: list[int],
        neighbour: list[int],
        boundary: list[dict] | None = None,
        **kwargs,
    ) -> "PolyMesh":
        """Construct a PolyMesh from plain Python lists.

        Convenience factory for testing and manual mesh construction.
        Converts lists to tensors on the configured device.

        Args:
            points: ``[[x, y, z], ...]`` vertex positions.
            faces: ``[[p0, p1, p2, ...], ...]`` face-vertex indices.
            owner: ``[cell0, cell1, ...]`` owner per face.
            neighbour: ``[cell0, cell1, ...]`` neighbour per internal face.
            boundary: Optional boundary patch descriptors.
            **kwargs: Forwarded to ``__init__``.

        Returns:
            A new :class:`PolyMesh`.
        """
        device = get_device()
        dtype = get_default_dtype()
        pts = torch.tensor(points, dtype=dtype, device=device)
        face_tensors = [
            torch.tensor(f, dtype=INDEX_DTYPE, device=device) for f in faces
        ]
        own = torch.tensor(owner, dtype=INDEX_DTYPE, device=device)
        nbr = torch.tensor(neighbour, dtype=INDEX_DTYPE, device=device)
        return cls(pts, face_tensors, own, nbr, boundary, **kwargs)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PolyMesh(n_points={self._n_points}, n_faces={self._n_faces}, "
            f"n_cells={self._n_cells}, n_internal_faces={self._n_internal_faces}, "
            f"n_patches={len(self._boundary)}, "
            f"device={self.device}, dtype={self.dtype})"
        )
