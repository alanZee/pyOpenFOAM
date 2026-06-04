"""
icoFoam 定量验证算例：Couette 流和 Poiseuille 流。

使用 icoFoam (PISO) 求解器求解，与解析解进行 L2 误差对比。
与 validation/cases/ 中的 Jacobi 松弛求解器不同，这些算例验证
的是完整的 PISO 算法精度。

解析解：
- Couette: u(y) = U_top * y / H
- Poiseuille: u(y) = u_max * 4 * y * (H - y) / H²
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def run_icofoam_couette(
    n_cells_x: int = 8,
    n_cells_y: int = 16,
    length: float = 1.0,
    height: float = 0.5,
    nu: float = 0.01,
    U_top: float = 1.0,
    end_time: float = 200.0,
    delta_t: float = 0.01,
) -> dict:
    """运行 icoFoam Couette 流算例并计算 L2 误差。

    Returns:
        dict with keys: U_computed, U_reference, l2_error, max_error,
        converged, n_cells.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "validation"))
    from test_couette_flow import _make_couette_case
    from pyfoam.applications.ico_foam import IcoFoam

    with tempfile.TemporaryDirectory() as tmp:
        case_dir = Path(tmp) / "couette"
        _make_couette_case(
            case_dir,
            n_cells_x=n_cells_x,
            n_cells_y=n_cells_y,
            length=length,
            height=height,
            nu=nu,
            U_top=U_top,
        )

        solver = IcoFoam(case_dir)
        conv = solver.run()

        # 计算解析解
        centres = solver.mesh.cell_centres.detach().cpu()
        dy = height / n_cells_y
        U_ref = torch.zeros(solver.mesh.n_cells, 3)
        for j in range(n_cells_y):
            y = (j + 0.5) * dy
            u_exact = U_top * y / height
            for i in range(n_cells_x):
                idx = j * n_cells_x + i
                U_ref[idx, 0] = u_exact

        U_computed = solver.U.detach().cpu()
        diff = U_computed - U_ref
        l2_err = diff.norm() / U_ref.norm()
        max_err = diff.abs().max().item()

        return {
            "U_computed": U_computed,
            "U_reference": U_ref,
            "l2_error": l2_err.item(),
            "max_error": max_err,
            "converged": conv.converged if conv else False,
            "n_cells": solver.mesh.n_cells,
        }


def run_icofoam_poiseuille(
    n_cells_x: int = 8,
    n_cells_y: int = 16,
    length: float = 1.0,
    height: float = 0.5,
    nu: float = 0.01,
    u_mean: float = 1.0,
    end_time: float = 200.0,
    delta_t: float = 0.01,
) -> dict:
    """运行 icoFoam Poiseuille 流算例并计算 L2 误差。

    Returns:
        dict with keys: U_computed, U_reference, l2_error, max_error,
        converged, n_cells.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "validation"))
    from test_poiseuille_flow import _make_poiseuille_case
    from pyfoam.applications.ico_foam import IcoFoam

    with tempfile.TemporaryDirectory() as tmp:
        case_dir = Path(tmp) / "poiseuille"
        _make_poiseuille_case(
            case_dir,
            n_cells_x=n_cells_x,
            n_cells_y=n_cells_y,
            length=length,
            height=height,
            nu=nu,
            u_inlet=u_mean,
        )

        solver = IcoFoam(case_dir)
        conv = solver.run()

        # 计算解析解 (充分发展 Poiseuille 流)
        centres = solver.mesh.cell_centres.detach().cpu()
        dy = height / n_cells_y
        u_max = 1.5 * u_mean
        U_ref = torch.zeros(solver.mesh.n_cells, 3)
        for j in range(n_cells_y):
            y = (j + 0.5) * dy
            u_exact = u_max * 4.0 * y * (height - y) / (height ** 2)
            for i in range(n_cells_x):
                idx = j * n_cells_x + i
                U_ref[idx, 0] = u_exact

        U_computed = solver.U.detach().cpu()
        diff = U_computed - U_ref
        l2_err = diff.norm() / U_ref.norm()
        max_err = diff.abs().max().item()

        return {
            "U_computed": U_computed,
            "U_reference": U_ref,
            "l2_error": l2_err.item(),
            "max_error": max_err,
            "converged": conv.converged if conv else False,
            "n_cells": solver.mesh.n_cells,
        }
