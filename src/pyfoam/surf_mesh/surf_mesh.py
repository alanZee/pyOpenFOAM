"""
surfMesh — 表面网格数据结构。

对应 OpenFOAM-13 的 surfMesh/surfMesh/surfMesh.H。
存储三角形/多边形表面网格的点、面和区域信息。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


@dataclass
class SurfZone:
    """表面网格区域定义。

    对应 OpenFOAM-13 的 surfZone/surfZone/surfZone.H。

    Attributes:
        name: 区域名称。
        start_face: 起始面索引。
        n_faces: 区域内面数。
    """
    name: str
    start_face: int
    n_faces: int


class SurfMesh:
    """表面网格数据结构。

    对应 OpenFOAM-13 的 surfMesh/surfMesh/surfMesh.H。
    存储表面网格的点坐标、面顶点列表和区域划分。

    支持三角形和多边形面。

    Examples:
        >>> import torch
        >>> pts = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=torch.float64)
        >>> faces = [torch.tensor([0,1,2,3])]
        >>> mesh = SurfMesh(points=pts, faces=faces)
        >>> mesh.n_points
        4
        >>> mesh.n_faces
        1
    """

    def __init__(
        self,
        points: torch.Tensor,
        faces: Sequence[torch.Tensor],
        zones: Optional[List[SurfZone]] = None,
    ):
        """初始化表面网格。

        Args:
            points: 顶点坐标，形状 ``(n_points, 3)``。
            faces: 面顶点索引列表，每个元素为一维整数张量。
            zones: 区域定义列表。若为 None 则创建单一区域。
        """
        self._points = points.to(dtype=CFD_DTYPE)
        self._faces = [f.to(dtype=INDEX_DTYPE) for f in faces]

        if zones is None:
            self._zones = [
                SurfZone(name="default", start_face=0, n_faces=len(self._faces))
            ]
        else:
            self._zones = zones

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def points(self) -> torch.Tensor:
        """顶点坐标，形状 ``(n_points, 3)``。"""
        return self._points

    @property
    def faces(self) -> List[torch.Tensor]:
        """面顶点索引列表。"""
        return self._faces

    @property
    def zones(self) -> List[SurfZone]:
        """区域列表。"""
        return self._zones

    @property
    def n_points(self) -> int:
        """顶点数。"""
        return self._points.shape[0]

    @property
    def n_faces(self) -> int:
        """面数。"""
        return len(self._faces)

    @property
    def n_zones(self) -> int:
        """区域数。"""
        return len(self._zones)

    # ── 几何计算 ──────────────────────────────────────────────────────

    def face_centres(self) -> torch.Tensor:
        """计算面心坐标。

        Returns:
            形状 ``(n_faces, 3)`` 的面心坐标张量。
        """
        centres = torch.zeros(self.n_faces, 3, dtype=CFD_DTYPE)
        for i, face in enumerate(self._faces):
            verts = self._points[face]
            centres[i] = verts.mean(dim=0)
        return centres

    def face_areas(self) -> torch.Tensor:
        """计算面面积向量（带方向）。

        对于三角形面使用叉积公式；对于多边形面使用扇形三角化。

        Returns:
            形状 ``(n_faces, 3)`` 的面积向量张量。
        """
        areas = torch.zeros(self.n_faces, 3, dtype=CFD_DTYPE)
        for i, face in enumerate(self._faces):
            verts = self._points[face]
            n = verts.shape[0]
            if n == 3:
                # 三角形叉积
                areas[i] = 0.5 * torch.cross(verts[1] - verts[0], verts[2] - verts[0])
            else:
                # 多边形扇形三角化
                centre = verts.mean(dim=0)
                for j in range(n):
                    v0 = verts[j] - centre
                    v1 = verts[(j + 1) % n] - centre
                    areas[i] += 0.5 * torch.cross(v0, v1)
        return areas

    def face_mag_areas(self) -> torch.Tensor:
        """计算面面积标量（面积大小）。

        Returns:
            形状 ``(n_faces,)`` 的面积标量张量。
        """
        return self.face_areas().norm(dim=1)

    # ── 面类型统计 ────────────────────────────────────────────────────

    def face_type_counts(self) -> Dict[int, int]:
        """统计不同顶点数的面数量。

        Returns:
            字典：{顶点数: 面数}。
        """
        counts: Dict[int, int] = {}
        for face in self._faces:
            n = face.shape[0]
            counts[n] = counts.get(n, 0) + 1
        return counts

    # ── I/O ───────────────────────────────────────────────────────────

    @classmethod
    def from_stl(cls, path: Union[str, Path]) -> "SurfMesh":
        """从 STL 文件读取表面网格。

        支持 ASCII 和二进制 STL 格式。

        Args:
            path: STL 文件路径。

        Returns:
            SurfMesh 实例。
        """
        path = Path(path)
        content = path.read_bytes()

        # 检测格式：ASCII STL 以 "solid" 开头
        if content[:5].lower().startswith(b"solid"):
            return cls._read_stl_ascii(path)
        else:
            return cls._read_stl_binary(content)

    @classmethod
    def _read_stl_ascii(cls, path: Path) -> "SurfMesh":
        """读取 ASCII STL。"""
        text = path.read_text(encoding="utf-8", errors="replace")
        points_list: List[List[float]] = []
        faces_list: List[torch.Tensor] = []
        vertex_map: Dict[Tuple[float, float, float], int] = {}
        current_verts: List[int] = []

        for line in text.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "vertex" and len(parts) >= 4:
                xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
                if xyz not in vertex_map:
                    vertex_map[xyz] = len(points_list)
                    points_list.append(list(xyz))
                current_verts.append(vertex_map[xyz])
            elif parts[0] == "endfacet" and len(current_verts) == 3:
                faces_list.append(torch.tensor(current_verts, dtype=INDEX_DTYPE))
                current_verts = []

        points = torch.tensor(points_list, dtype=CFD_DTYPE)
        return cls(points=points, faces=faces_list)

    @classmethod
    def _read_stl_binary(cls, content: bytes) -> "SurfMesh":
        """读取二进制 STL。"""
        import struct

        n_triangles = struct.unpack_from("<I", content, 80)[0]
        points_list: List[List[float]] = []
        faces_list: List[torch.Tensor] = []
        vertex_map: Dict[Tuple[float, float, float], int] = {}

        offset = 84
        for _ in range(n_triangles):
            # 跳过法线 (12 bytes)
            offset += 12
            current_verts: List[int] = []
            for _ in range(3):
                x, y, z = struct.unpack_from("<fff", content, offset)
                offset += 12
                xyz = (round(x, 6), round(y, 6), round(z, 6))
                if xyz not in vertex_map:
                    vertex_map[xyz] = len(points_list)
                    points_list.append([x, y, z])
                current_verts.append(vertex_map[xyz])
            faces_list.append(torch.tensor(current_verts, dtype=INDEX_DTYPE))
            offset += 2  # attribute byte count

        points = torch.tensor(points_list, dtype=CFD_DTYPE)
        return cls(points=points, faces=faces_list)

    def write_stl(self, path: Union[str, Path], binary: bool = True) -> None:
        """写入 STL 文件。

        Args:
            path: 输出文件路径。
            binary: 是否写入二进制格式（默认 True）。
        """
        path = Path(path)
        if binary:
            self._write_stl_binary(path)
        else:
            self._write_stl_ascii(path)

    def _write_stl_ascii(self, path: Path) -> None:
        """写入 ASCII STL。"""
        lines = ["solid surfMesh"]
        areas = self.face_areas()
        for i, face in enumerate(self._faces):
            verts = self._points[face]
            normal = areas[i]
            norm_mag = normal.norm()
            if norm_mag > 0:
                normal = normal / norm_mag
            lines.append(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}")
            lines.append("    outer loop")
            for v in verts:
                lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append("endsolid surfMesh")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_stl_binary(self, path: Path) -> None:
        """写入二进制 STL。"""
        import struct

        areas = self.face_areas()
        data = bytearray()
        # Header (80 bytes)
        data.extend(b"\0" * 80)
        # Triangle count
        data.extend(struct.pack("<I", self.n_faces))

        for i, face in enumerate(self._faces):
            verts = self._points[face]
            normal = areas[i]
            norm_mag = normal.norm()
            if norm_mag > 0:
                normal = normal / norm_mag
            # Normal
            data.extend(struct.pack("<fff", *normal.tolist()))
            # Vertices
            for v in verts:
                data.extend(struct.pack("<fff", *v.tolist()))
            # Attribute byte count
            data.extend(struct.pack("<H", 0))

        path.write_bytes(bytes(data))

    # ── 表示 ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SurfMesh(n_points={self.n_points}, n_faces={self.n_faces}, "
            f"n_zones={self.n_zones})"
        )
