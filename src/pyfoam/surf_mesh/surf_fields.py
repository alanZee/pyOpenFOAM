"""
surfFields — 表面场类型。

对应 OpenFOAM-13 的 surfMesh/surfFields/。
为 SurfMesh 的面和点提供标量/向量/张量场。
"""
from __future__ import annotations

from typing import Optional, Union

import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.surf_mesh.surf_mesh import SurfMesh


class _SurfField:
    """表面场基类。"""

    def __init__(self, mesh: SurfMesh, name: str = "", data: Optional[torch.Tensor] = None):
        self._mesh = mesh
        self._name = name
        self._data = data

    @property
    def mesh(self) -> SurfMesh:
        return self._mesh

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> torch.Tensor:
        return self._data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}', size={len(self)})"


class SurfScalarField(_SurfField):
    """表面标量场（每面一个值）。"""

    def __init__(self, mesh: SurfMesh, name: str = "", data: Optional[torch.Tensor] = None):
        if data is None:
            data = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        super().__init__(mesh, name, data)

    def __len__(self) -> int:
        return self._mesh.n_faces

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, value):
        self._data[idx] = value


class SurfVectorField(_SurfField):
    """表面向量场（每面一个 3D 向量）。"""

    def __init__(self, mesh: SurfMesh, name: str = "", data: Optional[torch.Tensor] = None):
        if data is None:
            data = torch.zeros(mesh.n_faces, 3, dtype=CFD_DTYPE)
        super().__init__(mesh, name, data)

    def __len__(self) -> int:
        return self._mesh.n_faces

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, value):
        self._data[idx] = value


class SurfTensorField(_SurfField):
    """表面张量场（每面一个 3x3 张量）。"""

    def __init__(self, mesh: SurfMesh, name: str = "", data: Optional[torch.Tensor] = None):
        if data is None:
            data = torch.zeros(mesh.n_faces, 3, 3, dtype=CFD_DTYPE)
        super().__init__(mesh, name, data)

    def __len__(self) -> int:
        return self._mesh.n_faces


class SurfPointScalarField(_SurfField):
    """表面点标量场（每点一个值）。"""

    def __init__(self, mesh: SurfMesh, name: str = "", data: Optional[torch.Tensor] = None):
        if data is None:
            data = torch.zeros(mesh.n_points, dtype=CFD_DTYPE)
        super().__init__(mesh, name, data)

    def __len__(self) -> int:
        return self._mesh.n_points


class SurfPointVectorField(_SurfField):
    """表面点向量场（每点一个 3D 向量）。"""

    def __init__(self, mesh: SurfMesh, name: str = "", data: Optional[torch.Tensor] = None):
        if data is None:
            data = torch.zeros(mesh.n_points, 3, dtype=CFD_DTYPE)
        super().__init__(mesh, name, data)

    def __len__(self) -> int:
        return self._mesh.n_points
