"""Tests for foam_info — case information summary tool."""

import pytest
from pathlib import Path

from pyfoam.tools.foam_info import foam_info


class TestFoamInfoBasic:
    """基本功能测试。"""

    @pytest.fixture
    def simple_case(self, tmp_path):
        """创建一个简单的 OpenFOAM case 目录结构。"""
        # system/
        system_dir = tmp_path / "system"
        system_dir.mkdir()
        control_dict = system_dir / "controlDict"
        control_dict.write_text(
            "FoamFile\n"
            "{\n"
            '    version     2.0;\n'
            "    format      ascii;\n"
            "    class       dictionary;\n"
            '    object      controlDict;\n'
            "}\n"
            "\n"
            "application     simpleFoam;\n"
            "startTime       0;\n"
            "endTime         100;\n"
            "deltaT          1;\n",
            encoding="utf-8",
        )

        # constant/polyMesh/
        mesh_dir = tmp_path / "constant" / "polyMesh"
        mesh_dir.mkdir(parents=True)
        # points
        (mesh_dir / "points").write_text(
            "FoamFile\n{ version 2.0; format ascii; class vectorField; object points; }\n"
            "8\n(\n(0 0 0)\n(1 0 0)\n(1 1 0)\n(0 1 0)\n"
            "(0 0 1)\n(1 0 1)\n(1 1 1)\n(0 1 1)\n)\n",
            encoding="utf-8",
        )
        # faces
        (mesh_dir / "faces").write_text(
            "FoamFile\n{ version 2.0; format ascii; class faceList; object faces; }\n"
            "6\n(\n(0 3 2 1)\n(4 5 6 7)\n(0 1 5 4)\n"
            "(2 3 7 6)\n(0 4 7 3)\n(1 2 6 5)\n)\n",
            encoding="utf-8",
        )
        # owner (6 faces, all belong to cell 0)
        (mesh_dir / "owner").write_text(
            "FoamFile\n{ version 2.0; format ascii; class labelList; object owner; }\n"
            "6\n(\n0\n0\n0\n0\n0\n0\n)\n",
            encoding="utf-8",
        )
        # neighbour (0 internal faces)
        (mesh_dir / "neighbour").write_text(
            "FoamFile\n{ version 2.0; format ascii; class labelList; object neighbour; }\n"
            "0\n(\n)\n",
            encoding="utf-8",
        )
        # boundary
        (mesh_dir / "boundary").write_text(
            "FoamFile\n{ version 2.0; format ascii; class polyBoundaryMesh; object boundary; }\n"
            "6\n(\n"
            "bottom\n{ type wall; nFaces 1; startFace 0; }\n"
            "top\n{ type wall; nFaces 1; startFace 1; }\n"
            "front\n{ type wall; nFaces 1; startFace 2; }\n"
            "back\n{ type wall; nFaces 1; startFace 3; }\n"
            "left\n{ type wall; nFaces 1; startFace 4; }\n"
            "right\n{ type wall; nFaces 1; startFace 5; }\n"
            ")\n",
            encoding="utf-8",
        )

        # 0/ — time directory with fields
        time_dir = tmp_path / "0"
        time_dir.mkdir()
        (time_dir / "U").write_text(
            "FoamFile\n{ version 2.0; format ascii; class volVectorField; object U; }\n"
            "1\n(\n(0 0 0)\n)\n",
            encoding="utf-8",
        )
        (time_dir / "p").write_text(
            "FoamFile\n{ version 2.0; format ascii; class volScalarField; object p; }\n"
            "1\n(\n0\n)\n",
            encoding="utf-8",
        )

        # 1/ — second time directory
        time_dir2 = tmp_path / "1"
        time_dir2.mkdir()
        (time_dir2 / "U").write_text(
            "FoamFile\n{ version 2.0; format ascii; class volVectorField; object U; }\n"
            "1\n(\n(1 0 0)\n)\n",
            encoding="utf-8",
        )

        return tmp_path

    def test_returns_dict(self, simple_case):
        info = foam_info(simple_case)
        assert isinstance(info, dict)

    def test_case_name(self, simple_case):
        info = foam_info(simple_case)
        assert info["case_name"] == simple_case.name

    def test_case_path(self, simple_case):
        info = foam_info(simple_case)
        assert info["case_path"] == str(simple_case.resolve())

    def test_has_mesh(self, simple_case):
        info = foam_info(simple_case)
        assert info["has_mesh"] is True

    def test_mesh_stats_present(self, simple_case):
        info = foam_info(simple_case)
        stats = info["mesh_stats"]
        assert stats["n_points"] == 8
        assert stats["n_cells"] == 1
        assert stats["n_internal_faces"] == 0
        assert stats["n_patches"] == 6

    def test_boundary_patches(self, simple_case):
        info = foam_info(simple_case)
        patches = info["mesh_stats"]["boundary_patches"]
        patch_names = [p["name"] for p in patches]
        assert "bottom" in patch_names
        assert "top" in patch_names
        assert all(p.get("n_faces") == 1 for p in patches)

    def test_time_dirs(self, simple_case):
        info = foam_info(simple_case)
        assert sorted(info["time_dirs"]) == [0.0, 1.0]

    def test_time_range(self, simple_case):
        info = foam_info(simple_case)
        assert info["time_range"] == (0.0, 1.0)

    def test_field_names_default_time(self, simple_case):
        info = foam_info(simple_case, time=0)
        assert sorted(info["field_names"]) == ["U", "p"]

    def test_field_names_latest_time(self, simple_case):
        info = foam_info(simple_case, time="latestTime")
        assert info["field_names"] == ["U"]

    def test_application(self, simple_case):
        info = foam_info(simple_case)
        assert info["application"] == "simpleFoam"


class TestFoamInfoEdgeCases:
    """边界情况测试。"""

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            foam_info(tmp_path / "nonexistent_case")

    def test_empty_case(self, tmp_path):
        """无网格、无时间目录的空 case。"""
        tmp_path.mkdir(exist_ok=True)
        info = foam_info(tmp_path)
        assert info["has_mesh"] is False
        assert info["mesh_stats"] == {}
        assert info["time_dirs"] == []
        assert info["time_range"] is None
        assert info["field_names"] == []
        assert info["application"] == ""

    def test_case_without_control_dict(self, tmp_path):
        """无 controlDict 时 application 为空字符串。"""
        tmp_path.mkdir(exist_ok=True)
        info = foam_info(tmp_path)
        assert info["application"] == ""
