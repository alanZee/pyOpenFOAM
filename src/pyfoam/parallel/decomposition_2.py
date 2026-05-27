"""
独立域分解策略类 — Scotch 和 Simple。

提供两种域分解策略的独立实现，每种策略作为一个单独的类，
支持 RTS (Run-Time Selection) 注册表用于运行时选择。

- **SimpleDecomposition**: 基于几何坐标的简单分解。沿最长轴排序后
  均匀分配。速度快，适用于任意网格。
- **ScotchDecomposition**: 基于图的 Scotch 库分解。最小化通信面。
  若 Scotch 未安装，回退到 Simple 分解。

所有策略通过 ``@DecompositionStrategy.register(name)`` 注册到 RTS 表。

Usage::

    from pyfoam.parallel.decomposition_2 import DecompositionStrategy

    # 直接创建
    strategy = DecompositionStrategy.create("simple", mesh, n_processors=4)
    assignment = strategy.decompose()

    # 或使用具体类
    from pyfoam.parallel.decomposition_2 import ScotchDecomposition

    strategy = ScotchDecomposition(mesh, n_processors=4)
    assignment = strategy.decompose()
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.poly_mesh import PolyMesh
from pyfoam.mesh.fv_mesh import FvMesh

__all__ = [
    "DecompositionStrategy",
    "SimpleDecomposition",
    "ScotchDecomposition",
]

_EPS = 1e-30


# ---------------------------------------------------------------------------
# 抽象基类 + RTS 注册表
# ---------------------------------------------------------------------------


class DecompositionStrategy(ABC):
    """域分解策略的抽象基类。

    子类实现 :meth:`decompose` 返回处理器分配张量。

    RTS 注册表允许按名称运行时选择::

        strategy = DecompositionStrategy.create("simple", mesh, n_processors=4)
        assignment = strategy.decompose()  # (n_cells,) int tensor

    Parameters
    ----------
    mesh : PolyMesh | FvMesh
        要分解的全局网格.
    n_processors : int
        处理器（子域）数量.
    """

    # RTS 注册表: name -> class
    _registry: ClassVar[dict[str, Type[DecompositionStrategy]]] = {}

    def __init__(
        self,
        mesh: PolyMesh | FvMesh,
        n_processors: int,
    ) -> None:
        if n_processors < 1:
            raise ValueError(f"n_processors must be >= 1, got {n_processors}")
        if n_processors > mesh.n_cells:
            raise ValueError(
                f"n_processors ({n_processors}) > n_cells ({mesh.n_cells})"
            )

        self._mesh = mesh
        self._n_processors = n_processors
        self._device = get_device()
        self._dtype = get_default_dtype()

    # ------------------------------------------------------------------
    # RTS 注册表
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """装饰器：注册域分解策略.

        Usage::

            @DecompositionStrategy.register("simple")
            class SimpleDecomposition(DecompositionStrategy):
                ...
        """

        def decorator(strategy_cls: Type[DecompositionStrategy]) -> Type[DecompositionStrategy]:
            if name in cls._registry:
                raise ValueError(
                    f"Decomposition strategy '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = strategy_cls
            return strategy_cls

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        mesh: PolyMesh | FvMesh,
        n_processors: int,
        **kwargs: Any,
    ) -> DecompositionStrategy:
        """工厂方法：按名称创建分解策略.

        Args:
            name: 注册策略名称 (``"simple"`` 或 ``"scotch"``).
            mesh: 全局网格.
            n_processors: 处理器数量.
            **kwargs: 额外参数.

        Returns:
            分解策略实例.

        Raises:
            KeyError: 名称未注册.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown decomposition strategy '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](mesh, n_processors, **kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """返回已注册策略名称列表."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def mesh(self) -> PolyMesh | FvMesh:
        """全局网格."""
        return self._mesh

    @property
    def n_processors(self) -> int:
        """处理器数量."""
        return self._n_processors

    # ------------------------------------------------------------------
    # 抽象接口
    # ------------------------------------------------------------------

    @abstractmethod
    def decompose(self) -> torch.Tensor:
        """执行域分解.

        Returns:
            ``(n_cells,)`` int 张量，每个单元格的处理器 ID。
        """

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _estimate_cell_centres(self) -> torch.Tensor:
        """估算单元格中心点（用于 PolyMesh）.

        对每个单元格，取其所有面中心的平均值。

        Returns:
            ``(n_cells, 3)`` 单元格中心坐标.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells

        if isinstance(mesh, FvMesh):
            return mesh.cell_centres

        face_centres = torch.zeros(
            mesh.n_faces, 3, device=self._device, dtype=self._dtype,
        )
        for f_idx in range(mesh.n_faces):
            pts = mesh.points[mesh.faces[f_idx]]
            face_centres[f_idx] = pts.mean(dim=0)

        cell_sum = torch.zeros(
            n_cells, 3, device=self._device, dtype=self._dtype,
        )
        cell_count = torch.zeros(
            n_cells, device=self._device, dtype=INDEX_DTYPE,
        )

        for f_idx in range(mesh.n_faces):
            owner_cell = mesh.owner[f_idx].item()
            cell_sum[owner_cell] += face_centres[f_idx]
            cell_count[owner_cell] += 1

        for f_idx in range(mesh.n_internal_faces):
            nbr_cell = mesh.neighbour[f_idx].item()
            cell_sum[nbr_cell] += face_centres[f_idx]
            cell_count[nbr_cell] += 1

        safe_count = cell_count.float().clamp(min=1)
        return cell_sum / safe_count.unsqueeze(1)


# ---------------------------------------------------------------------------
# Simple 几何分解
# ---------------------------------------------------------------------------


@DecompositionStrategy.register("simple")
class SimpleDecomposition(DecompositionStrategy):
    """简单几何域分解.

    沿网格包围盒最长轴，按位置排序后均匀分配到各处理器。
    使用分位数分割保证各处理器负载大致均衡。

    适用场景：任意网格，不需要外部依赖。

    Parameters
    ----------
    mesh : PolyMesh | FvMesh
        全局网格.
    n_processors : int
        处理器数量.
    """

    def decompose(self) -> torch.Tensor:
        """执行简单几何分解.

        Returns:
            ``(n_cells,)`` int 张量，处理器 ID.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_procs = self._n_processors

        centres = self._estimate_cell_centres()

        # 找最长轴
        mins = centres.min(dim=0).values
        maxs = centres.max(dim=0).values
        extents = maxs - mins
        longest_axis = int(extents.argmax().item())

        # 按最长轴坐标排序
        coords = centres[:, longest_axis]

        # 分位数分割
        sorted_coords, _ = torch.sort(coords)
        boundaries = torch.zeros(
            n_procs + 1, device=self._device, dtype=self._dtype,
        )
        boundaries[0] = sorted_coords[0] - 1e-10
        boundaries[-1] = sorted_coords[-1] + 1e-10
        for i in range(1, n_procs):
            idx = int(i * n_cells / n_procs)
            boundaries[i] = sorted_coords[idx]

        # 分配处理器 ID
        assignment = torch.zeros(
            n_cells, dtype=INDEX_DTYPE, device=self._device,
        )
        for i in range(n_procs):
            mask = (coords >= boundaries[i]) & (coords < boundaries[i + 1])
            assignment[mask] = i

        # 确保最后一个处理器获得边界单元
        assignment[coords >= boundaries[-2]] = n_procs - 1

        return assignment

    def __repr__(self) -> str:
        return (
            f"SimpleDecomposition("
            f"n_processors={self._n_processors}, "
            f"n_cells={self._mesh.n_cells})"
        )


# ---------------------------------------------------------------------------
# Scotch 图分解
# ---------------------------------------------------------------------------


@DecompositionStrategy.register("scotch")
class ScotchDecomposition(DecompositionStrategy):
    """基于 Scotch 库的图分解.

    利用网格拓扑构建邻接图，使用 Scotch 库进行图划分。
    目标是最小化处理器间的通信面。

    若 Scotch 未安装，自动回退到 SimpleDecomposition。

    Parameters
    ----------
    mesh : PolyMesh | FvMesh
        全局网格.
    n_processors : int
        处理器数量.
    """

    def decompose(self) -> torch.Tensor:
        """执行 Scotch 图分解.

        Returns:
            ``(n_cells,)`` int 张量，处理器 ID.
        """
        try:
            import scotch  # type: ignore[import-untyped]
        except ImportError:
            warnings.warn(
                "scotch library not installed; falling back to simple decomposition",
                stacklevel=2,
            )
            fallback = SimpleDecomposition(
                self._mesh, self._n_processors,
            )
            return fallback.decompose()

        mesh = self._mesh
        n_cells = mesh.n_cells

        # 构建邻接图
        neighbours_per_cell: list[list[int]] = [[] for _ in range(n_cells)]
        for f_idx in range(mesh.n_internal_faces):
            own = mesh.owner[f_idx].item()
            nbr = mesh.neighbour[f_idx].item()
            neighbours_per_cell[own].append(nbr)
            neighbours_per_cell[nbr].append(own)

        # CSR 格式
        adj_offsets = torch.zeros(n_cells + 1, dtype=INDEX_DTYPE, device="cpu")
        adj_indices: list[int] = []
        offset = 0
        for c in range(n_cells):
            adj_offsets[c] = offset
            unique_nbrs = sorted(set(neighbours_per_cell[c]))
            adj_indices.extend(unique_nbrs)
            offset += len(unique_nbrs)
        adj_offsets[n_cells] = offset

        adj_indices_t = torch.tensor(adj_indices, dtype=torch.int32, device="cpu")
        adj_offsets_np = adj_offsets.numpy().astype("int32")
        adj_indices_np = adj_indices_t.numpy().astype("int32")

        # Scotch 划分
        strat = scotch.Strat()
        strat.archCmpltWgt(0)

        graph = scotch.Graph()
        graph.init()
        graph.build(0, adj_offsets_np, None, None, None, None, adj_indices_np)

        part = torch.zeros(n_cells, dtype=torch.int32, device="cpu")
        part_np = part.numpy()

        graph.part(self._n_processors, strat, part_np)

        return torch.from_numpy(part_np).to(
            dtype=INDEX_DTYPE, device=self._device,
        )

    def __repr__(self) -> str:
        return (
            f"ScotchDecomposition("
            f"n_processors={self._n_processors}, "
            f"n_cells={self._mesh.n_cells})"
        )
