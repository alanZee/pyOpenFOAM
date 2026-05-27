"""Tests for foam_dictionary — OpenFOAM dictionary query and modify tool."""

from __future__ import annotations

import pytest
from pathlib import Path
import tempfile
import shutil

from pyfoam.tools.foam_dictionary import foam_dictionary
from pyfoam.io.dictionary import FoamDict, FoamList


class TestFoamDictionaryQuery:
    """Query mode: read values from dictionary files."""

    @pytest.fixture
    def case_dir(self, tmp_path):
        """创建一个临时 case 目录，包含 system/controlDict。"""
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
            "startFrom       startTime;\n"
            "startTime       0;\n"
            "stopAt          endTime;\n"
            "endTime         1000;\n"
            "deltaT          1;\n"
            "\n"
            "functions\n"
            "{\n"
            "    fieldAverage\n"
            "    {\n"
            "        type            fieldAverage;\n"
            "        libs            (\"libfieldFunctionObjects.so\");\n"
            "    }\n"
            "}\n",
            encoding="utf-8",
        )
        return tmp_path

    def test_returns_full_dict(self, case_dir):
        """不指定 key 时返回完整 FoamDict。"""
        d = foam_dictionary(case_dir, "system/controlDict")
        assert isinstance(d, FoamDict)
        assert "application" in d
        assert d["application"] == "simpleFoam"

    def test_query_simple_key(self, case_dir):
        """查询顶层 key。"""
        val = foam_dictionary(case_dir, "system/controlDict", key="application")
        assert val == "simpleFoam"

    def test_query_numeric_key(self, case_dir):
        """查询数值型 key。"""
        val = foam_dictionary(case_dir, "system/controlDict", key="endTime")
        assert val == 1000

    def test_query_float_key(self, case_dir):
        """查询浮点型 key。"""
        val = foam_dictionary(case_dir, "system/controlDict", key="deltaT")
        assert val == 1.0

    def test_query_nested_key(self, case_dir):
        """查询嵌套路径 key（subDict/entry）。"""
        val = foam_dictionary(
            case_dir, "system/controlDict", key="functions/fieldAverage/type"
        )
        assert val == "fieldAverage"

    def test_query_missing_key_raises(self, case_dir):
        """查询不存在的 key 时抛出 KeyError。"""
        with pytest.raises(KeyError, match="nonExistentKey"):
            foam_dictionary(case_dir, "system/controlDict", key="nonExistentKey")

    def test_missing_file_raises(self, case_dir):
        """文件不存在时抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            foam_dictionary(case_dir, "system/nonExistentDict")


class TestFoamDictionaryWrite:
    """Write mode: modify values and write back to disk."""

    @pytest.fixture
    def case_dir(self, tmp_path):
        """创建一个简单的 controlDict。"""
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
            "endTime         1000;\n"
            "deltaT          1;\n",
            encoding="utf-8",
        )
        return tmp_path

    def test_modify_existing_key(self, case_dir):
        """修改已存在的 key 并写回。"""
        foam_dictionary(case_dir, "system/controlDict", key="endTime", value=2000)

        # 重新读取验证
        val = foam_dictionary(case_dir, "system/controlDict", key="endTime")
        assert val == 2000

    def test_modify_preserves_header(self, case_dir):
        """修改后保留 FoamFile header。"""
        foam_dictionary(case_dir, "system/controlDict", key="deltaT", value=0.5)

        content = (case_dir / "system/controlDict").read_text(encoding="utf-8")
        assert "FoamFile" in content
        assert "version" in content

    def test_add_new_key(self, case_dir):
        """添加新的 key-value 对。"""
        foam_dictionary(case_dir, "system/controlDict", key="writeInterval", value=100)

        val = foam_dictionary(case_dir, "system/controlDict", key="writeInterval")
        assert val == 100

    def test_modify_nested_key_creates_subdict(self, case_dir):
        """在嵌套路径中写入，自动创建子字典。"""
        foam_dictionary(
            case_dir,
            "system/controlDict",
            key="functions/probes/type",
            value="probes",
        )

        val = foam_dictionary(
            case_dir, "system/controlDict", key="functions/probes/type"
        )
        assert val == "probes"

    def test_write_returns_value(self, case_dir):
        """write 模式返回传入的 value。"""
        result = foam_dictionary(
            case_dir, "system/controlDict", key="endTime", value=5000
        )
        assert result == 5000


class TestFoamDictionaryStringValues:
    """字符串值处理：包含空格的字符串应被引号包裹。"""

    @pytest.fixture
    def case_dir(self, tmp_path):
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
            "application     simpleFoam;\n",
            encoding="utf-8",
        )
        return tmp_path

    def test_write_string_with_spaces(self, case_dir):
        """包含空格的字符串写入后应加引号。"""
        foam_dictionary(
            case_dir, "system/controlDict", key="note", value="test value"
        )

        content = (case_dir / "system/controlDict").read_text(encoding="utf-8")
        assert '"test value"' in content

    def test_roundtrip_string_value(self, case_dir):
        """字符串写入后可正确读回。"""
        foam_dictionary(
            case_dir, "system/controlDict", key="note", value="some note text"
        )
        val = foam_dictionary(case_dir, "system/controlDict", key="note")
        assert val == "some note text"
