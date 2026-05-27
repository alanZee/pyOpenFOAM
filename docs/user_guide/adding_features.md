# 开发者指南：扩展 pyOpenFOAM

本文档介绍如何向 pyOpenFOAM 添加新功能，包括求解器、边界条件和湍流模型。

## 目录

- [项目结构](#项目结构)
- [添加新求解器](#添加新求解器)
- [添加新边界条件](#添加新边界条件)
- [添加新湍流模型](#添加新湍流模型)
- [测试](#测试)
- [代码规范](#代码规范)

---

## 项目结构

```
src/pyfoam/
├── core/               # 基础层：设备管理、LDU 矩阵、稀疏运算
├── mesh/               # 网格表示与几何计算
│   └── generation/     # blockMesh, snappyHexMesh
├── fields/             # 场类（volScalar, volVector, surfaceScalar, ...）
├── boundary/           # 边界条件（RTS 注册机制）
├── io/                 # OpenFOAM 文件格式 I/O
├── discretisation/     # 离散格式（插值、ddt、梯度、法向梯度）
│   └── schemes/        # 具体格式实现
├── solvers/            # 线性求解器 + SIMPLE/PISO/PIMPLE
├── turbulence/         # RANS/LES/DES 湍流模型
├── thermophysical/     # 状态方程、输运模型、热力学
├── multiphase/         # VOF、MULES、相间力、空化
├── parallel/           # MPI 域分解与并行 I/O
├── applications/       # 应用级求解器
├── postprocessing/     # 函数对象框架
├── differentiable/     # 可微分算子（torch.autograd）
├── models/             # 物理模型（辐射等）
├── ode/                # ODE 时间积分器
├── fv/                 # fvConstraints + fvModels
├── lagrangian/         # 拉格朗日粒子追踪
├── waves/              # 波浪模型
└── tools/              # 命令行工具
```

### 架构层次

```
applications/      应用层：读取案例 -> 构建网格/场 -> 运行算法
    |
solvers/           算法层：SIMPLE/PISO/PIMPLE + 线性求解器
    |
fields/ + boundary/ + discretisation/    物理层：场、BC、离散格式
    |
core/ + mesh/ + io/    基础层：张量后端、网格、I/O
```

---

## 添加新求解器

### 步骤概述

1. 创建求解器文件 `src/pyfoam/applications/my_solver.py`
2. 继承 `SolverBase`
3. 实现 `run()` 方法
4. 注册到 `__init__.py`
5. 编写测试

### 详细步骤

#### 第 1 步：创建求解器文件

在 `src/pyfoam/applications/` 下创建新文件：

```python
"""
my_solver.py — 我的自定义求解器。

求解器描述：方程、算法、适用场景。

用法::

    from pyfoam.applications.my_solver import MySolver

    solver = MySolver("path/to/case")
    solver.run()
"""

import logging
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.turbulence import RASModel, RASConfig

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["MySolver"]
logger = logging.getLogger(__name__)
```

#### 第 2 步：继承 SolverBase

```python
class MySolver(SolverBase):
    """我的自定义求解器。

    Parameters
    ----------
    case_path : str | Path
        OpenFOAM 案例目录路径。
    """

    def __init__(self, case_path):
        super().__init__(case_path)

        # SolverBase 已完成：
        # - self.case: Case 对象
        # - self.mesh: FvMesh
        # - self.controlDict, self.fvSchemes, self.fvSolution

        # 读取场（SolverBase 提供 _read_field 辅助方法）
        self.U = self._init_vector_field("U")
        self.p = self._init_scalar_field("p")

        # 初始化湍流模型（可选）
        self.turbulence = self._init_turbulence()

    def _init_turbulence(self):
        """从 turbulenceProperties 初始化湍流模型。"""
        turb_props = self.case.constant_dir / "turbulenceProperties"
        if not turb_props.exists():
            return None
        # 解析 turbulenceProperties 并创建模型
        config = RASConfig(model_name="kOmegaSST", nu=self.nu)
        return RASModel(self.mesh, self.U.internal_field,
                       self.phi.internal_field if hasattr(self, 'phi') else None,
                       config)
```

#### 第 3 步：实现 run() 方法

```python
    def run(self):
        """运行求解器主循环。"""
        device = get_device()
        dtype = get_default_dtype()

        # 从 controlDict 读取参数
        end_time = self.case.get_end_time()
        delta_t = self.case.get_delta_t()

        # 从 fvSolution 读取求解器配置
        fv_sol = self.case.fvSolution
        simple_config = self._read_simple_config(fv_sol)

        # 创建耦合求解器
        simple_solver = SIMPLESolver(self.mesh, simple_config)

        # 收敛监控
        monitor = ConvergenceMonitor()

        # 时间循环
        time_loop = TimeLoop(0.0, end_time, delta_t)
        for t, step in time_loop:
            logger.info("Time = %.6f, step = %d", t, step)

            # 湍流更新
            if self.turbulence is not None:
                self.turbulence.correct()

            # SIMPLE 求解
            U_out, p_out, phi_out, convergence = simple_solver.solve(
                self.U.internal_field,
                self.p.internal_field,
                self.phi.internal_field,
                max_outer_iterations=100,
                tolerance=1e-4,
            )

            # 更新场
            self.U.assign(U_out)
            self.p.assign(p_out)
            self.phi.assign(phi_out)

            # 监控收敛
            monitor.update(convergence)

            # 按 writeInterval 输出
            if time_loop.should_write():
                self._write_fields(t)

        logger.info("求解完成。")
```

#### 第 4 步：注册到 `__init__.py`

在 `src/pyfoam/applications/__init__.py` 中添加导入：

```python
# 已有导入行之后添加
from pyfoam.applications.my_solver import MySolver

# 在 __all__ 列表中添加
__all__ = [
    # ... 已有条目 ...
    "MySolver",
]
```

#### 第 5 步：编写测试

```python
# tests/unit/applications/test_my_solver.py
import pytest
import torch
from pyfoam.applications.my_solver import MySolver


class TestMySolver:
    """MySolver 单元测试。"""

    def test_init(self, tmp_path, simple_cavity_case):
        """测试求解器初始化。"""
        solver = MySolver(simple_cavity_case)
        assert solver.mesh is not None
        assert solver.U is not None
        assert solver.p is not None

    def test_run_converges(self, simple_cavity_case):
        """测试求解器收敛。"""
        solver = MySolver(simple_cavity_case)
        solver.run()
        # 验证输出文件存在
```

---

## 添加新边界条件

### 步骤概述

1. 创建 BC 文件 `src/pyfoam/boundary/my_bc.py`
2. 继承 `BoundaryCondition`
3. 使用 `@BoundaryCondition.register` 装饰器注册
4. 实现 `apply()` 和 `matrix_contributions()`
5. 在 `__init__.py` 中添加导入
6. 编写测试

### 详细步骤

#### 第 1 步：创建 BC 文件

```python
"""
my_bc.py — 自定义边界条件。

描述边界条件的物理含义和数学公式。
"""

import torch
from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MyCustomBC"]
```

#### 第 2 步：实现边界条件

```python
@BoundaryCondition.register("myCustom")
class MyCustomBC(BoundaryCondition):
    """自定义边界条件。

    物理描述...

    Parameters
    ----------
    patch : Patch
        边界面片描述符。
    coeffs : dict
        系数字典，支持键:
        - ``value``: 基准值 (float 或 tensor)
        - ``exponent``: 指数 (float, 默认 1.0)
    """

    def __init__(self, patch: Patch, coeffs: dict | None = None):
        super().__init__(patch, coeffs)
        self.value = self.coeffs.get("value", 0.0)
        self.exponent = self.coeffs.get("exponent", 1.0)

    def apply(self, field: torch.Tensor, patch_idx: int = 0) -> torch.Tensor:
        """修改边界面值。

        Parameters
        ----------
        field : torch.Tensor
            场数据（形状取决于场类型）。
        patch_idx : int
            面片索引。

        Returns
        -------
        torch.Tensor
            修改后的场数据。
        """
        face_indices = self.patch.face_indices
        if field.dim() == 1:
            # 标量场
            field[face_indices] = self.value ** self.exponent
        else:
            # 矢量场
            val = self.value if isinstance(self.value, torch.Tensor) \
                else torch.full((len(face_indices), field.shape[1]), self.value,
                               device=field.device, dtype=field.dtype)
            field[face_indices] = val ** self.exponent
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 FVM 矩阵贡献。

        使用惩罚法将边界值耦合到矩阵中：
        - diag[owner] += deltaCoeff * area
        - source[owner] += deltaCoeff * area * value

        Parameters
        ----------
        field : torch.Tensor
            当前场值。
        n_cells : int
            单元数。
        diag : torch.Tensor
            对角系数 (n_cells,)。
        source : torch.Tensor
            右端项 (n_cells,)。

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            更新后的 (diag, source)。
        """
        owners = self.patch.owner_cells
        delta = self.patch.delta_coeffs
        areas = self.patch.face_areas
        value = self.value ** self.exponent

        penalty = delta * areas
        diag.index_add_(0, owners, penalty)
        source.index_add_(0, owners, penalty * value)

        return diag, source
```

#### 第 3 步：注册

在 `src/pyfoam/boundary/__init__.py` 中添加：

```python
from pyfoam.boundary.my_bc import MyCustomBC

__all__ = [
    # ... 已有条目 ...
    "MyCustomBC",
]
```

### 关键接口说明

| 方法 | 说明 |
|------|------|
| `apply(field, patch_idx)` | 修改边界面值（后处理用） |
| `matrix_contributions(field, n_cells, diag, source)` | 向 FVM 矩阵添加边界贡献（求解用） |

`Patch` 提供以下属性：

| 属性 | 形状 | 说明 |
|------|------|------|
| `face_indices` | `(n_faces,)` | 面索引 |
| `face_normals` | `(n_faces, 3)` | 外法向单位向量 |
| `face_areas` | `(n_faces,)` | 面积 |
| `delta_coeffs` | `(n_faces,)` | 1/距离系数 |
| `owner_cells` | `(n_faces,)` | 相邻单元索引 |

---

## 添加新湍流模型

### 步骤概述

1. 创建模型文件 `src/pyfoam/turbulence/my_model.py`
2. 继承 `TurbulenceModel`
3. 使用 `@TurbulenceModel.register` 装饰器注册
4. 实现 `correct()` 和 `nut()` 方法
5. 在 `__init__.py` 中添加导入
6. 编写测试

### 详细步骤

#### 第 1 步：创建模型文件

```python
"""
my_model.py — 自定义 RANS 湍流模型。

模型描述：方程组、常数、适用场景。
"""

import torch
from pyfoam.turbulence.turbulence_model import TurbulenceModel

__all__ = ["MyTurbModel", "MyTurbConstants"]
```

#### 第 2 步：定义常数和实现模型

```python
class MyTurbConstants:
    """模型常数。"""
    Cmu: float = 0.09
    C1: float = 1.44
    C2: float = 1.92
    sigma_k: float = 1.0
    sigma_epsilon: float = 1.3


@TurbulenceModel.register("myTurb")
class MyTurbModel(TurbulenceModel):
    """自定义 RANS 湍流模型。

    求解 k 和 epsilon 的输运方程：
        d(k)/dt + div(U*k) - div(Dk * grad(k)) = Pk - epsilon
        d(eps)/dt + div(U*eps) - div(De * grad(eps))
            = C1 * Pk * eps/k - C2 * eps^2/k

    Parameters
    ----------
    mesh : FvMesh
        有限体积网格。
    U : torch.Tensor
        速度场 (n_cells, 3)。
    phi : torch.Tensor
        面通量 (n_faces,)。
    """

    def __init__(self, mesh, U, phi, constants=None):
        super().__init__(mesh, U, phi)
        self.C = constants or MyTurbConstants()

        # 湍流量初始化
        n_cells = mesh.n_cells
        self._k = torch.full((n_cells,), 1e-3, device=U.device, dtype=U.dtype)
        self._epsilon = torch.full((n_cells,), 1e-4, device=U.device, dtype=U.dtype)
        self._nut = torch.zeros(n_cells, device=U.device, dtype=U.dtype)

    def correct(self):
        """执行一步湍流量更新。

        1. 计算涡粘性 nut = Cmu * k^2 / epsilon
        2. 计算生成项 Pk = nut * S^2（S 为应变率）
        3. 组装并求解 k 方程
        4. 组装并求解 epsilon 方程
        5. 更新边界条件
        """
        C = self.C

        # 限制小值
        k = torch.clamp(self._k, min=1e-10)
        eps = torch.clamp(self._epsilon, min=1e-10)

        # 涡粘性
        self._nut = C.Cmu * k ** 2 / eps

        # 生成项（简化：实际需要计算应变率张量）
        # Pk = nut * 2 * Sij * Sij
        # ... 组装 FvMatrix 并求解 ...

        # 边界更新（调用壁面函数等）
        self._update_boundaries()

    def nut(self) -> torch.Tensor:
        """返回涡粘性场 (n_cells,)。"""
        return self._nut

    def k(self) -> torch.Tensor:
        """返回湍动能场 (n_cells,)。"""
        return self._k

    def epsilon(self) -> torch.Tensor:
        """返回耗散率场 (n_cells,)。"""
        return self._epsilon

    def _update_boundaries(self):
        """更新壁面函数等边界条件。"""
        pass  # 实际实现需遍历 boundary patches
```

#### 第 3 步：注册

在 `src/pyfoam/turbulence/__init__.py` 中添加：

```python
from pyfoam.turbulence.my_model import MyTurbModel, MyTurbConstants

__all__ = [
    # ... 已有条目 ...
    "MyTurbModel",
    "MyTurbConstants",
]
```

### 使用新模型

```python
from pyfoam.turbulence import TurbulenceModel

# RTS 创建（自动可用，无需额外注册代码）
model = TurbulenceModel.create("myTurb", mesh, U, phi)
model.correct()
nut = model.nut()
```

### TurbulenceModel 基类接口

| 方法 | 说明 |
|------|------|
| `correct()` | 更新一步湍流量（抽象，必须实现） |
| `nut()` | 返回涡粘性（抽象，必须实现） |
| `k()` | 返回湍动能（可选覆盖） |
| `epsilon()` / `omega()` | 返回耗散率/比耗散率（可选覆盖） |

---

## 测试

### 测试目录结构

```
tests/
├── unit/
│   ├── core/
│   ├── mesh/
│   ├── fields/
│   ├── boundary/
│   ├── solvers/
│   ├── turbulence/
│   ├── applications/
│   └── ...
└── conftest.py  # 共享 fixtures
```

### 运行测试

```bash
# 全部测试
pytest tests/unit/ -q

# 特定模块
pytest tests/unit/boundary/test_my_bc.py -q

# 带覆盖率
pytest tests/unit/ --cov=pyfoam --cov-report=term-missing
```

### 测试规范

- 每个新模块都需要对应的单元测试
- 使用 `pytest.fixture` 提供共享测试数据
- 边界条件测试需验证 `apply()` 和 `matrix_contributions()` 的正确性
- 湍流模型测试需验证 `correct()` 不发散且 `nut()` 返回正值

---

## 代码规范

### 通用规范

- **类型注解**：所有公共函数/方法需有类型注解
- **文档字符串**：使用 NumPy 风格 docstring
- **日志**：使用 `logging.getLogger(__name__)` 记录日志
- **设备/类型**：通过 `get_device()` / `get_default_dtype()` 获取全局配置

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `MySolver`, `FixedValueBC` |
| 函数/方法 | snake_case | `compute_face_centres` |
| 常数 | UPPER_SNAKE_CASE | `CFD_DTYPE`, `INDEX_DTYPE` |
| 文件名 | snake_case | `my_solver.py`, `fixed_value.py` |
| RTS 注册名 | camelCase | `"fixedValue"`, `"kOmegaSST"` |

### RTS（运行时选择）模式

pyOpenFOAM 大量使用 RTS 模式实现可扩展性：

```python
# 基类定义注册机制
class Base:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(klass):
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, name, *args, **kwargs):
        if name not in cls._registry:
            raise KeyError(f"Unknown type: {name}")
        return cls._registry[name](*args, **kwargs)

# 子类注册
@Base.register("myType")
class MyType(Base):
    pass

# 运行时创建
instance = Base.create("myType")
```

此模式用于：
- `BoundaryCondition.register("name")` — 边界条件
- `TurbulenceModel.register("name")` — 湍流模型
- `ODESolver.register("name")` — ODE 求解器
- `FvConstraint.register("name")` — fvConstraint
- `FvModel.register("name")` — fvModel
- `WaveModel.register("name")` — 波浪模型

添加新的可选类型时，只需用 `@register` 装饰器标注即可自动注册，无需修改基类代码。
