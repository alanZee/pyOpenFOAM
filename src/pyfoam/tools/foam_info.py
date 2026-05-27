"""
foamInfo — print OpenFOAM case information summary.

Mirrors the ``foamInfo`` concept: scans a case directory and returns
a structured summary with mesh statistics, field names, and time range.

Usage::

    from pyfoam.tools.foam_info import foam_info

    info = foam_info("path/to/case")
    print(info)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = ["foam_info"]


# 匹配纯数字目录名（如 ``0``, ``0.005``, ``1``, ``100``）
_TIME_DIR_RE = re.compile(r"^\d+(?:\.\d+)?$")


def foam_info(
    case_path: Union[str, Path],
    time: Optional[Union[str, float, int]] = None,
) -> Dict[str, Any]:
    """Collect and return OpenFOAM case information summary.

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    time : str, float, int, or None, optional
        Time directory to inspect for fields.
        - ``None`` — use time ``0``.
        - ``"latestTime"`` — use the largest time directory.
        - numeric value — use the specified time.

    Returns
    -------
    dict
        Case summary with keys:

        - ``case_name``: Name of the case directory.
        - ``case_path``: Absolute path to the case directory.
        - ``has_mesh``: Whether the mesh directory exists with required files.
        - ``mesh_stats``: Dictionary with mesh statistics (n_points, n_cells,
          n_internal_faces, n_boundary_faces, boundary_patches).
        - ``field_names``: List of field file names in the time directory.
        - ``time_dirs``: Sorted list of all time directory values.
        - ``time_range``: Tuple (start, end) of time range, or None.
        - ``application``: Application name from controlDict (or empty string).

    Raises
    ------
    FileNotFoundError
        If *case_path* does not exist or is not a directory.
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    result: Dict[str, Any] = {
        "case_name": case_dir.name,
        "case_path": str(case_dir),
    }

    # --- 时间目录 ---
    time_dirs = _scan_time_dirs(case_dir)
    result["time_dirs"] = time_dirs

    if time_dirs:
        result["time_range"] = (time_dirs[0], time_dirs[-1])
    else:
        result["time_range"] = None

    # --- 网格信息 ---
    mesh_dir = case_dir / "constant" / "polyMesh"
    required_files = ["points", "faces", "owner", "neighbour", "boundary"]
    has_mesh = mesh_dir.is_dir() and all(
        (mesh_dir / f).exists() for f in required_files
    )
    result["has_mesh"] = has_mesh

    if has_mesh:
        result["mesh_stats"] = _read_mesh_stats(mesh_dir)
    else:
        result["mesh_stats"] = {}

    # --- 场文件 ---
    time_dir = _resolve_time_dir(case_dir, time_dirs, time)
    if time_dir is not None and time_dir.is_dir():
        result["field_names"] = _list_fields(time_dir)
    else:
        result["field_names"] = []

    # --- application ---
    result["application"] = _read_application(case_dir)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scan_time_dirs(case_dir: Path) -> List[float]:
    """扫描所有数字时间目录，返回排序后的列表。"""
    times: List[float] = []
    for entry in case_dir.iterdir():
        if entry.is_dir() and _TIME_DIR_RE.match(entry.name):
            try:
                times.append(float(entry.name))
            except ValueError:
                continue
    times.sort()
    return times


def _resolve_time_dir(
    case_dir: Path,
    time_dirs: List[float],
    time: Optional[Union[str, float, int]],
) -> Optional[Path]:
    """解析时间选择器，返回对应的时间目录路径。"""
    if time is None:
        target = 0.0
    elif isinstance(time, str) and time == "latestTime":
        if not time_dirs:
            return None
        target = time_dirs[-1]
    else:
        target = float(time)

    # 查找精确匹配或最接近的时间目录
    for t in time_dirs:
        if abs(t - target) < 1e-12:
            name = str(int(t)) if t == int(t) else str(t)
            return case_dir / name

    # 回退到字符串形式
    time_path = case_dir / str(target)
    if time_path.is_dir():
        return time_path
    return None


def _list_fields(time_dir: Path) -> List[str]:
    """列出时间目录中的场文件名。"""
    fields: List[str] = []
    for p in time_dir.iterdir():
        if p.is_file() and not p.name.startswith("."):
            fields.append(p.name)
    return sorted(fields)


def _read_mesh_stats(mesh_dir: Path) -> Dict[str, Any]:
    """读取网格统计信息。"""
    stats: Dict[str, Any] = {}

    # n_points
    points_file = mesh_dir / "points"
    if points_file.exists():
        content = points_file.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(\d+)\s*\(", content)
        if match:
            stats["n_points"] = int(match.group(1))

    # n_cells (from owner file: max index + 1)
    owner_file = mesh_dir / "owner"
    if owner_file.exists():
        content = owner_file.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(\d+)\s*\(", content)
        if match:
            n_faces = int(match.group(1))
            # 找到最大 cell index
            nums = re.findall(r"^\s*(\d+)", content[match.end():], re.MULTILINE)
            if nums:
                max_idx = max(int(n) for n in nums)
                stats["n_cells"] = max_idx + 1
            stats["n_faces"] = n_faces

    # n_internal_faces (from neighbour file)
    neighbour_file = mesh_dir / "neighbour"
    if neighbour_file.exists():
        content = neighbour_file.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(\d+)\s*\(", content)
        if match:
            stats["n_internal_faces"] = int(match.group(1))

    # 从 n_faces 和 n_internal_faces 计算边界表面数
    if "n_faces" in stats and "n_internal_faces" in stats:
        stats["n_boundary_faces"] = stats["n_faces"] - stats["n_internal_faces"]

    # boundary patches
    boundary_file = mesh_dir / "boundary"
    if boundary_file.exists():
        patches = _parse_boundary_patches(boundary_file)
        stats["boundary_patches"] = patches
        stats["n_patches"] = len(patches)

    return stats


def _parse_boundary_patches(boundary_file: Path) -> List[Dict[str, Any]]:
    """解析 boundary 文件，提取 patch 信息。"""
    content = boundary_file.read_text(encoding="utf-8", errors="replace")
    patches: List[Dict[str, Any]] = []

    # 匹配顶层 patch 块
    # 模式: patchName { ... nFaces N; startFace S; type T; ... }
    for match in re.finditer(r"(\w+)\s*\{([^}]*)\}", content, re.DOTALL):
        name = match.group(1)
        block = match.group(2)

        # 跳过 FoamFile 头部块
        if name == "FoamFile":
            continue

        patch_info: Dict[str, Any] = {"name": name}
        has_nfaces = False

        for kv in re.finditer(r"(\w+)\s+(.+?)\s*;", block):
            key, val = kv.group(1), kv.group(2).strip()
            if key == "nFaces":
                patch_info["n_faces"] = int(val)
                has_nfaces = True
            elif key == "startFace":
                patch_info["start_face"] = int(val)
            elif key == "type":
                patch_info["type"] = val
            elif key == "nGroups":
                patch_info["n_groups"] = int(val)

        # 仅保留有 nFaces 的有效 patch
        if has_nfaces:
            patches.append(patch_info)

    return patches


def _read_application(case_dir: Path) -> str:
    """从 system/controlDict 读取 application 名称。"""
    control_dict = case_dir / "system" / "controlDict"
    if not control_dict.exists():
        return ""
    content = control_dict.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"application\s+(.+?)\s*;", content)
    return match.group(1).strip() if match else ""
