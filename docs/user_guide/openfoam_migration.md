# OpenFOAM 案例迁移速查

本文档提供从 OpenFOAM 迁移到 pyOpenFOAM 的快速参考。详细流程见 [完整迁移指南](migration_guide.md)。

---

## 目录

- [案例目录结构转换](#案例目录结构转换)
- [controlDict 语法差异](#controldict-语法差异)
- [场文件差异](#场文件差异)
- [边界条件对照](#边界条件对照)
- [后处理迁移](#后处理迁移)
- [常见迁移陷阱](#常见迁移陷阱)

---

## 案例目录结构转换

pyOpenFOAM 直接读取标准 OpenFOAM 目录：

```
myCase/
├── 0/                  # 初始条件（无需修改）
│   ├── U
│   └── p
├── constant/
│   ├── transportProperties
│   └── polyMesh/       # 网格（ASCII 格式可直接读取）
├── system/
│   ├── controlDict
│   ├── fvSchemes
│   └── fvSolution
└── Allrun              # 替换为 Python 脚本
```

### 替换 Allrun

```bash
# OpenFOAM
./Allrun

# pyOpenFOAM
python run_case.py
```

```python
# run_case.py
from pyfoam.applications import SimpleFoam

solver = SimpleFoam("myCase")
result = solver.run()
```

---

## controlDict 语法差异

### OpenFOAM 格式

```
application     simpleFoam;
startFrom       latestTime;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
```

### pyOpenFOAM 对应

```python
# pyOpenFOAM 使用 Python 字典配置
config = {
    "application": "simpleFoam",
    "startTime": 0,
    "endTime": 1000,
    "deltaT": 1.0,
    "writeInterval": 100,
}
```

> **要点**: pyOpenFOAM 的求解器类可直接读取 OpenFOAM 的 `controlDict` 文件，无需手动转换。只在需要 Python 化配置时才需要改写。

---

## 场文件差异

### 格式兼容性

| 格式 | OpenFOAM | pyOpenFOAM |
|------|----------|------------|
| ASCII | 原生支持 | 直接读取 |
| binary | 原生支持 | **需先转换** (`foamToASCII`) |
| HDF5 | 可选 | 不支持 |

### 二进制文件处理

```bash
# 转换所有二进制场为 ASCII
foamToASCII -case myCase
```

### 张量布局

```python
# OpenFOAM (C++) 中的 VectorField 存储
#   U[0].x()  U[0].y()  U[0].z()
#   U[1].x()  U[1].y()  U[1].z()

# pyOpenFOAM (PyTorch) 中的存储 — 行主序
#   U[0, 0]=Ux  U[0, 1]=Uy  U[0, 2]=Uz
#   U[1, 0]=Ux  U[1, 1]=Uy  U[1, 2]=Uz
import torch
U = torch.tensor(...)  # shape: (n_cells, 3)
```

---

## 边界条件对照

| OpenFOAM `type` | pyOpenFOAM | 使用方式 |
|-----------------|-----------|----------|
| `fixedValue uniform (1 0 0)` | `FixedValueBC` | `coeffs={"value": [1, 0, 0]}` |
| `zeroGradient` | `ZeroGradientBC` | 无额外配置 |
| `noSlip` | `NoSlipBC` | 无额外配置 |
| `calculated` | 自动处理 | 由求解器内部计算 |
| `cyclic` | `CyclicBC` | 需指定配对 patch |
| `inletOutlet` | `InletOutletBC` | `coeffs={"inletValue": ...}` |
| `totalPressure` | `TotalPressureBC` | `coeffs={"p0": 101325}` |

---

## 后处理迁移

### 函数对象对照

| OpenFOAM functionObject | pyOpenFOAM 类 |
|------------------------|--------------|
| `forces` | `Forces` |
| `wallShearStress` | `WallShearStress` |
| `yPlus` | `YPlus` |
| `fieldAverage` | `FieldAverage` |
| `noise` | `Noise` |
| `vorticity` | `Vorticity` |
| `Q` | `QCriterion` |
| `Lambda2` | `Lambda2` |
| `enstrophy` | `Enstrophy` |
| `TKE` | `TurbulentKineticEnergy` |

### OpenFOAM 内联后处理 → pyOpenFOAM

```
# OpenFOAM controlDict
functions
{
    forces1
    {
        type        forces;
        libs        ("libforces.so");
        patches     (wall);
        rho         rhoInf;
        rhoInf      1.0;
        CofR        (0 0 0);
    }
}
```

```python
# pyOpenFOAM 等价
from pyfoam.postprocessing import Forces

fo = Forces("forces1", {
    "patches": ["wall"],
    "rho": 1.0,
    "CofR": [0, 0, 0],
})
fo.initialise(mesh, {"U": U, "p": p})
fo.execute(time=1.0)
fo.write()
```

---

## 常见迁移陷阱

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 二进制场文件读取失败 | pyOpenFOAM 仅支持 ASCII | `foamToASCII` 转换 |
| 残差行为不同 | 浮点精度差异 | 确认 `float64`；调整残差容差 |
| 边界条件类型不支持 | pyOpenFOAM 尚未实现 | 查看 [边界条件对照](#边界条件对照) |
| `blockMesh` 不可用 | pyOpenFOAM 不含网格生成 | 用 OpenFOAM 先生成网格 |
| 并行计算缺失 | pyOpenFOAM 无 MPI 分区 | 使用 GPU 加速替代 |

---

## 进一步参考

- [完整迁移指南](migration_guide.md) — 分步详细流程
- [性能优化](performance.md) — GPU 加速和内存优化
- [快速入门](getting_started.md) — 安装和第一个案例
