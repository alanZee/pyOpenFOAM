"""
renumberMesh — renumber cells using the Reverse Cuthill-McKee algorithm.

Mirrors OpenFOAM's ``renumberMesh`` utility.  RCM reordering reduces the
bandwidth of the cell-adjacency graph, improving cache locality and
preconditioner performance in linear solvers.

The adjacency graph is constructed from the internal faces of an
:class:`~pyfoam.mesh.fv_mesh.FvMesh`.  The algorithm operates on a
sparse adjacency matrix built with ``torch.sparse``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RenumberResult", "renumber_mesh"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RenumberResult:
    """Result from :func:`renumber_mesh`.

    Attributes
    ----------
    permutation : torch.Tensor
        ``(n_cells,)`` int tensor.  ``permutation[new_idx] = old_idx`` —
        maps new cell index to original cell index.
    inverse_permutation : torch.Tensor
        ``(n_cells,)`` int tensor.  ``inverse_permutation[old_idx] = new_idx`` —
        maps original cell index to new cell index.
    original_bandwidth : int
        Profile bandwidth of the original ordering.
    renumbered_bandwidth : int
        Profile bandwidth after RCM reordering.
    """

    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    original_bandwidth: int
    renumbered_bandwidth: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def renumber_mesh(mesh: "FvMesh") -> RenumberResult:
    """Renumber cells using the Reverse Cuthill-McKee algorithm.

    Builds a cell-adjacency graph from internal faces, applies RCM, and
    returns the permutation tensors.  The mesh object itself is **not**
    modified — the caller is responsible for applying the permutation to
    all relevant arrays.

    Args:
        mesh: Finite volume mesh with topology data.

    Returns:
        :class:`RenumberResult` with permutation tensors and bandwidth info.
    """
    n_cells = mesh.n_cells
    n_internal = mesh.n_internal_faces
    owner = mesh.owner
    neighbour = mesh.neighbour

    # 构建稀疏邻接矩阵（对称）
    adj = _build_adjacency(owner, neighbour, n_cells, n_internal)

    # 计算 RCM 排列
    perm = _reverse_cuthill_mckee(adj, n_cells)
    inv_perm = _invert_permutation(perm)

    # 计算带宽
    bw_orig = _bandwidth(adj, n_cells)
    # 重排后的新邻接矩阵：P A P^T
    adj_new = _apply_permutation_to_adj(adj, perm, n_cells)
    bw_new = _bandwidth(adj_new, n_cells)

    return RenumberResult(
        permutation=perm,
        inverse_permutation=inv_perm,
        original_bandwidth=bw_orig,
        renumbered_bandwidth=bw_new,
    )


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _build_adjacency(
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_cells: int,
    n_internal: int,
) -> torch.Tensor:
    """构建对称稀疏邻接矩阵。

    从内部面的 owner-neighbour 关系构造 COO 格式的稀疏邻接矩阵。

    Args:
        owner: ``(n_faces,)`` 每个面的 owner 单元索引。
        neighbour: ``(n_internal,)`` 每个内部面的 neighbour 单元索引。
        n_cells: 单元总数。
        n_internal: 内部面数。

    Returns:
        ``(n_cells, n_cells)`` 稀疏 COO 张量（值全为 1）。
    """
    if n_internal == 0:
        # 无内部面 → 空邻接矩阵
        indices = torch.zeros((2, 0), dtype=INDEX_DTYPE)
        return torch.sparse_coo_tensor(indices, torch.ones(0), (n_cells, n_cells))

    own_int = owner[:n_internal]
    nbr_int = neighbour[:n_internal]

    # 对称：(owner, neighbour) 和 (neighbour, owner)
    rows = torch.cat([own_int, nbr_int])
    cols = torch.cat([nbr_int, own_int])
    indices = torch.stack([rows, cols])
    values = torch.ones(indices.shape[1], dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, (n_cells, n_cells)).coalesce()


def _degree(adj: torch.Tensor, n: int) -> torch.Tensor:
    """计算每个节点的度数。

    Args:
        adj: 稀疏邻接矩阵。
        n: 节点数。

    Returns:
        ``(n,)`` int 张量，每个节点的邻居数。
    """
    if adj._nnz() == 0:
        return torch.zeros(n, dtype=INDEX_DTYPE)
    rows = adj.indices()[0]
    # 对每行计数（对称矩阵中等价于度数）
    deg = torch.zeros(n, dtype=INDEX_DTYPE)
    ones = torch.ones(rows.shape[0], dtype=INDEX_DTYPE)
    deg.scatter_add_(0, rows, ones)
    return deg


def _reverse_cuthill_mckee(adj: torch.Tensor, n: int) -> torch.Tensor:
    """Reverse Cuthill-McKee 排序算法。

    步骤：
    1. 选择度最小的节点作为起始节点。
    2. BFS 遍历，按度数升序排列同层邻居。
    3. 反转排列以获得 RCM 序。

    Args:
        adj: 稀疏邻接矩阵。
        n: 节点数。

    Returns:
        ``(n,)`` 排列张量 ``perm[new] = old``。
    """
    if n == 0:
        return torch.zeros(0, dtype=INDEX_DTYPE)

    # 转为 dense 进行 BFS（矩阵通常不会太大）
    if adj._nnz() == 0:
        return torch.arange(n, dtype=INDEX_DTYPE)

    deg = _degree(adj, n)

    # 构建邻居列表（仅上三角信息 + 下三角信息 = 全部邻居）
    indices = adj.indices()
    rows, cols = indices[0], indices[1]
    # 去重（coalesce 已处理）
    # 按行分组
    neighbor_list: list[list[int]] = [[] for _ in range(n)]
    for i in range(rows.shape[0]):
        r, c = rows[i].item(), cols[i].item()
        if r != c:  # 排除自环
            neighbor_list[r].append(c)

    # 对每个节点的邻居按度数排序
    for i in range(n):
        neighbor_list[i].sort(key=lambda x: deg[x].item())

    visited = torch.zeros(n, dtype=torch.bool)
    perm_list: list[int] = []

    while len(perm_list) < n:
        # 选择未访问的度最小节点
        if len(perm_list) == 0:
            candidates = torch.arange(n)
            unvisited_mask = ~visited
            candidates = candidates[unvisited_mask]
            if len(candidates) == 0:
                break
            # 度最小
            cand_deg = deg[candidates]
            start_idx = candidates[cand_deg.argmin()].item()
        else:
            # 如果 BFS 队列耗尽但还有未访问节点（不连通图）
            candidates = torch.arange(n)
            unvisited_mask = ~visited
            candidates = candidates[unvisited_mask]
            if len(candidates) == 0:
                break
            cand_deg = deg[candidates]
            start_idx = candidates[cand_deg.argmin()].item()

        # BFS
        queue = [start_idx]
        visited[start_idx] = True
        while queue:
            node = queue.pop(0)
            perm_list.append(node)
            for nb in neighbor_list[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

    # Reverse → RCM
    perm_list.reverse()
    return torch.tensor(perm_list, dtype=INDEX_DTYPE)


def _invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    """计算逆排列。

    若 ``perm[new] = old``，则 ``inv_perm[old] = new``。

    Args:
        perm: ``(n,)`` 排列张量。

    Returns:
        ``(n,)`` 逆排列张量。
    """
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.shape[0], dtype=perm.dtype)
    return inv


def _bandwidth(adj: torch.Tensor, n: int) -> int:
    """计算稀疏矩阵的半带宽。

    带宽 = max(|i - j|) 对所有非零元素 (i, j)。

    Args:
        adj: 稀疏邻接矩阵。
        n: 节点数。

    Returns:
        半带宽（整数）。
    """
    if adj._nnz() == 0:
        return 0
    indices = adj.indices()
    rows, cols = indices[0], indices[1]
    diff = (rows - cols).abs()
    return int(diff.max().item())


def _apply_permutation_to_adj(
    adj: torch.Tensor, perm: torch.Tensor, n: int
) -> torch.Tensor:
    """对邻接矩阵应用 P A P^T 重排。

    Args:
        adj: 原始稀疏邻接矩阵。
        perm: ``(n,)`` 排列张量 ``perm[new] = old``。
        n: 节点数。

    Returns:
        重排后的稀疏邻接矩阵。
    """
    if adj._nnz() == 0:
        return adj

    indices = adj.indices()
    values = adj.values()
    old_rows = indices[0]
    old_cols = indices[1]

    new_rows = perm[old_rows]
    new_cols = perm[old_cols]
    new_indices = torch.stack([new_rows, new_cols])

    return torch.sparse_coo_tensor(new_indices, values, (n, n)).coalesce()
