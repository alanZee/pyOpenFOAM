"""foam_to_fluent 导出工具测试。

测试 Fluent ASCII 格式导出功能，包括：
- 输出目录创建
- .msh 文件生成与格式验证
- .dat 文件生成与场数据写入
- 多时间步导出
- 错误处理
"""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.foam_to_fluent import foam_to_fluent


class TestFoamToFluent:
    """foam_to_fluent 函数测试。"""

    def test_export_creates_output_dir(self, fv_mesh, tmp_path):
        """导出应创建 Fluent 输出目录。"""
        result = foam_to_fluent(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Fluent"),
        )
        assert Path(result).is_dir()

    def test_export_creates_msh_file(self, fv_mesh, tmp_path):
        """导出应生成 .msh 网格文件。"""
        foam_to_fluent(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Fluent"),
        )
        fluent_dir = tmp_path / "Fluent"
        msh_files = list(fluent_dir.glob("*.msh"))
        assert len(msh_files) == 1
        assert "mesh.msh" == msh_files[0].name

    def test_msh_has_header(self, fv_mesh, tmp_path):
        """Fluent .msh 文件应包含标题行。"""
        foam_to_fluent(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Fluent"),
        )
        msh_path = tmp_path / "Fluent" / "mesh.msh"
        content = msh_path.read_text()
        assert "(0 " in content  # Comment header
        assert "(2 3)" in content  # 3D dimensions

    def test_msh_has_nodes(self, fv_mesh, tmp_path):
        """Fluent .msh 文件应包含节点坐标数据。"""
        foam_to_fluent(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Fluent"),
        )
        msh_path = tmp_path / "Fluent" / "mesh.msh"
        content = msh_path.read_text()
        # Node section header (zone 1)
        assert "(10 " in content
        # Nodes should have floating point values
        assert "E+" in content or "E-" in content

    def test_msh_has_cells_and_faces(self, fv_mesh, tmp_path):
        """Fluent .msh 文件应包含单元和面数据。"""
        foam_to_fluent(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Fluent"),
        )
        msh_path = tmp_path / "Fluent" / "mesh.msh"
        content = msh_path.read_text()
        # Cell section header
        assert "(12 " in content
        # Face section header
        assert "(13 " in content

    def test_export_creates_dat_file(self, fv_mesh, tmp_path):
        """导出应生成 .dat 场数据文件。"""
        foam_to_fluent(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Fluent"),
        )
        fluent_dir = tmp_path / "Fluent"
        dat_files = list(fluent_dir.glob("*.dat"))
        assert len(dat_files) == 1
        assert "0.dat" in dat_files[0].name

    def test_export_with_scalar_field(self, fv_mesh, tmp_path):
        """标量场数据应写入 .dat 文件。"""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells) * 101325.0
        foam_to_fluent(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": pressure},
            output_dir=str(tmp_path / "Fluent"),
        )
        dat_path = tmp_path / "Fluent" / "0.dat"
        content = dat_path.read_text()
        assert "p" in content
        assert "1.01325" in content  # scientific notation

    def test_export_with_vector_field(self, fv_mesh, tmp_path):
        """矢量场数据应分量写入。"""
        n_cells = fv_mesh.n_cells
        velocity = np.zeros((n_cells, 3))
        velocity[:, 0] = 1.0
        foam_to_fluent(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"U": velocity},
            output_dir=str(tmp_path / "Fluent"),
        )
        dat_path = tmp_path / "Fluent" / "0.dat"
        content = dat_path.read_text()
        assert "U_x" in content
        assert "U_y" in content
        assert "U_z" in content

    def test_export_multiple_times(self, fv_mesh, tmp_path):
        """多时间步导出应生成对应的 .dat 文件。"""
        foam_to_fluent(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 0.5, 1.0],
            output_dir=str(tmp_path / "Fluent"),
        )
        fluent_dir = tmp_path / "Fluent"
        dat_files = list(fluent_dir.glob("*.dat"))
        assert len(dat_files) == 3

    def test_nonexistent_case_path_raises(self, tmp_path):
        """不存在的路径应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            foam_to_fluent(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
            )

    def test_no_mesh_raises(self, tmp_path):
        """未提供网格时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="No mesh provided"):
            foam_to_fluent(
                case_path=str(tmp_path),
                mesh=None,
                time_range=[0.0],
            )
