"""
Tutorial validation: OpenFOAM tutorial case parser.

解析 OpenFOAM tutorial 算例目录结构并提取关键参数。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass, field


@dataclass
class TutorialCase:
    """OpenFOAM tutorial 算例描述。"""
    name: str
    path: Path
    solver: str = ""
    delta_t: float = 0.0
    end_time: float = 0.0
    n_cells: int = 0
    has_mesh: bool = False
    has_fields: bool = False
    field_files: List[str] = field(default_factory=list)
    bc_types: Dict[str, List[str]] = field(default_factory=dict)


def parse_control_dict(path: Path) -> Dict[str, str]:
    """解析 controlDict 文件。"""
    result = {}
    if not path.exists():
        return result

    content = path.read_text(encoding="utf-8", errors="replace")
    # 移除注释
    content = re.sub(r'//.*?\n', '\n', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # 解析 key-value 对
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('{') or line.startswith('}'):
            continue
        if ';' in line:
            line = line.replace(';', '').strip()
            parts = line.split(None, 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]

    return result


def parse_foam_file_header(path: Path) -> Dict[str, str]:
    """解析 OpenFOAM 文件头。"""
    result = {}
    if not path.exists():
        return result

    content = path.read_text(encoding="utf-8", errors="replace")

    # 查找 FoamFile 块
    match = re.search(r'FoamFile\s*\{([^}]+)\}', content, re.DOTALL)
    if match:
        block = match.group(1)
        for line in block.split('\n'):
            line = line.strip()
            if ';' in line:
                line = line.replace(';', '').strip()
                parts = line.split(None, 1)
                if len(parts) == 2:
                    result[parts[0]] = parts[1].strip('"')

    return result


def scan_tutorial_case(case_dir: Path) -> TutorialCase:
    """扫描 tutorial 算例目录。"""
    case = TutorialCase(name=case_dir.name, path=case_dir)

    # 解析 controlDict
    control_dict_path = case_dir / "system" / "controlDict"
    if control_dict_path.exists():
        cd = parse_control_dict(control_dict_path)
        case.solver = cd.get("solver", "")
        case.delta_t = float(cd.get("deltaT", "0"))
        case.end_time = float(cd.get("endTime", "0"))

    # 检查网格
    mesh_dir = case_dir / "constant" / "polyMesh"
    case.has_mesh = mesh_dir.exists() and (mesh_dir / "points").exists()

    # 检查场文件
    zero_dir = case_dir / "0"
    if zero_dir.exists():
        case.field_files = [f.name for f in zero_dir.iterdir() if f.is_file()]
        case.has_fields = len(case.field_files) > 0

    # 解析边界条件
    if zero_dir.exists():
        for field_file in zero_dir.iterdir():
            if field_file.is_file():
                content = field_file.read_text(encoding="utf-8", errors="replace")
                # 提取 patch 名称和类型
                patches = re.findall(r'(\w+)\s*\{[^}]*type\s+(\w+)', content, re.DOTALL)
                if patches:
                    case.bc_types[field_file.name] = [(p[0], p[1]) for p in patches]

    return case


def scan_tutorial_category(category_dir: Path) -> List[TutorialCase]:
    """扫描 tutorial 类别目录。"""
    cases = []
    for item in sorted(category_dir.iterdir()):
        if item.is_dir() and (item / "system").exists():
            cases.append(scan_tutorial_case(item))
    return cases


def scan_all_tutorials(tutorials_dir: Path) -> Dict[str, List[TutorialCase]]:
    """扫描所有 tutorial。"""
    result = {}
    for category_dir in sorted(tutorials_dir.iterdir()):
        if category_dir.is_dir() and category_dir.name not in ("resources", "Allrun", "Allclean"):
            cases = scan_tutorial_category(category_dir)
            if cases:
                result[category_dir.name] = cases
    return result


def count_tutorials(tutorials_dir: Path) -> Dict[str, int]:
    """统计每个类别的 tutorial 数量。"""
    all_cases = scan_all_tutorials(tutorials_dir)
    return {cat: len(cases) for cat, cases in all_cases.items()}


if __name__ == "__main__":
    import sys
    tutorials_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".reference/OpenFOAM-13/tutorials")
    counts = count_tutorials(tutorials_dir)
    total = sum(counts.values())
    print(f"Total tutorials: {total}")
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
