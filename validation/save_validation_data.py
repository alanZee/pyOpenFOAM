"""
运行全部验证测试并保存结果数据。

仅保留最后一次成功运行的数据。
输出目录: validation/results/<case_name>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


# 验证案例及其参考数据
VALIDATION_CASES = {
    "lid_driven_cavity": {
        "description": "盖驱动方腔流 Re=100 (Ghia et al. 1982)",
        "solver": "icoFoam",
        "reference": {
            "source": "Ghia et al. 1982, J. Comput. Phys. 48, 387-411",
            "Re": 100,
            "mesh": "32x32",
            "u_centreline_y": {
                # y/H vs u/U_lid 沿中心线 (x=0.5)
                "y": [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                       0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                       0.9688, 0.9766, 1.0],
                "u": [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                       -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                       0.68717, 0.73722, 0.78871, 0.84123, 1.0],
            },
            "tolerance": {"l2_relative": 0.20, "max_absolute": 0.15},
        },
    },
    "poiseuille_flow": {
        "description": "Poiseuille 管道流 (解析解)",
        "solver": "icoFoam",
        "reference": {
            "source": "Hagen-Poiseuille 解析解",
            "formula": "u(y) = U_max * (1 - (2y/H)^2)",
            "tolerance": {"l2_relative": 0.05},
        },
    },
    "couette_flow": {
        "description": "Couette 流 (解析解)",
        "solver": "icoFoam",
        "reference": {
            "source": "Couette 解析解",
            "formula": "u(y) = U_wall * y/H",
            "tolerance": {"l2_relative": 0.05},
        },
    },
    "taylor_green_vortex": {
        "description": "Taylor-Green 涡衰减 (解析解)",
        "solver": "icoFoam",
        "reference": {
            "source": "Taylor & Green 1937",
            "formula": "E(t) = E(0) * exp(-4*nu*k^2*t)",
            "tolerance": {"energy_decay_relative": 0.20},
        },
    },
    "backward_facing_step": {
        "description": "后向台阶流 (Driver & Seegmiller 1985)",
        "solver": "simpleFoam",
        "reference": {
            "source": "Driver & Seegmiller 1985, AIAA J. 23(2)",
            "Re_h": 28000,
            "reattachment_length": {"expected_ratio": 6.0, "tolerance": 3.0},
        },
    },
    "sod_shock_tube": {
        "description": "Sod 激波管 (Sod 1978)",
        "solver": "rhoCentralFoam",
        "reference": {
            "source": "Sod 1978, J. Comput. Phys. 27, 1-31",
            "shock_position": {"x_shock": 0.48, "tolerance": 0.05},
            "tolerance": {"density_l2": 0.10, "pressure_l2": 0.10},
        },
    },
    "natural_convection": {
        "description": "自然对流方腔 (de Vahl Davis 1983)",
        "solver": "buoyantBoussinesqSimpleFoam",
        "reference": {
            "source": "de Vahl Davis 1983, Int. J. Numer. Methods Fluids 3, 249-264",
            "Ra": 1e5,
            "Nu_expected": 4.519,
            "tolerance": {"Nu_relative": 0.30},
        },
    },
    "dam_break": {
        "description": "溃坝 (Martin & Moyce 1952)",
        "solver": "interFoam",
        "reference": {
            "source": "Martin & Moyce 1952, Phil. Trans. R. Soc. A 244, 312-324",
            "tolerance": {"mass_conservation": 0.10},
        },
    },
    "turbulent_channel": {
        "description": "湍流通道 Re_tau=180 (Moser, Kim & Mansour 1999)",
        "solver": "simpleFoam+kOmegaSST",
        "reference": {
            "source": "Moser, Kim & Mansour 1999, Phys. Fluids 11(4), 943-945",
            "Re_tau": 180,
            "tolerance": {"velocity_profile_l2": 0.20},
        },
    },
    "compressible_nozzle": {
        "description": "收缩-扩张喷管等熵流",
        "solver": "rhoCentralFoam",
        "reference": {
            "source": "等熵喷管流动理论解",
            "tolerance": {"Mach_number_relative": 0.10},
        },
    },
    "laminar_cylinder": {
        "description": "层流圆柱绕流 Re=20 (Dennis & Chang 1970)",
        "solver": "icoFoam",
        "reference": {
            "source": "Dennis & Chang 1970, J. Fluid Mech. 42, 471-489",
            "Re": 20,
            "Cd_expected": 2.045,
            "tolerance": {"Cd_relative": 0.30},
        },
    },
    "cylinder_flow": {
        "description": "圆柱绕流 Re=100 (Williamson 1996)",
        "solver": "pisoFoam",
        "reference": {
            "source": "Williamson 1996, Annu. Rev. Fluid Mech. 28, 477-539",
            "Re": 100,
            "St_expected": 0.164,
            "tolerance": {"St_relative": 0.20},
        },
    },
    "turbulent_duct": {
        "description": "湍流方管 Re=10000 (Petukhov 摩擦系数)",
        "solver": "simpleFoam+kOmegaSST",
        "reference": {
            "source": "Petukhov 1970",
            "Re": 10000,
            "f_expected": 0.0315,
            "tolerance": {"f_relative": 0.30},
        },
    },
}


def save_validation_results():
    """运行验证并保存结果（仅保存元数据和参考数据）。"""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pyfoam_version": "0.2.0",
        "total_cases": len(VALIDATION_CASES),
        "cases": {},
    }

    for case_name, case_info in VALIDATION_CASES.items():
        result = {
            "description": case_info["description"],
            "solver": case_info["solver"],
            "reference": case_info["reference"],
            "status": "pending",
        }
        summary["cases"][case_name] = result

        # 保存单个案例结果
        case_file = results_dir / f"{case_name}.json"
        with open(case_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # 保存汇总
    summary_file = results_dir / "validation_summary_v2.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"验证结果模板已保存到 {results_dir}/")
    print(f"共 {len(VALIDATION_CASES)} 个验证案例")
    for name, info in VALIDATION_CASES.items():
        print(f"  - {name}: {info['description']}")

    return summary


if __name__ == "__main__":
    save_validation_results()
