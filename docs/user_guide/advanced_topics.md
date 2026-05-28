# 高级主题

本文档涵盖 pyOpenFOAM 的高级用法，包括自定义模型开发、可微分 CFD 和 GPU 加速深入配置。

---

## 目录

- [自定义湍流模型开发](#自定义湍流模型开发)
- [自定义边界条件开发](#自定义边界条件开发)
- [可微分 CFD 用法](#可微分-cfd-用法)
- [GPU 加速深入](#gpu-加速深入)

---

## 自定义湍流模型开发

### 架构概述

pyOpenFOAM 的湍流模型框架位于 `pyfoam.models` 模块，采用分层设计：

```
pyfoam.models/
├── turbulence_model.py        # 基类 TurbulenceModel
├── mixing_length.py           # 混合长度模型
├── spalart_allmaras.py        # SA 单方程模型
├── k_epsilon.py               # k-ε 双方程模型
├── k_omega_sst.py             # k-ω SST 双方程模型
└── les/                       # 大涡模拟模型
    ├── smagorinsky.py
    └── wale.py
```

### 湍流模型基类

所有自定义湍流模型需继承 `TurbulenceModel`：

```python
from pyfoam.models.turbulence_model import TurbulenceModel

class MyTurbulenceModel(TurbulenceModel):
    """自定义湍流模型。

    Parameters
    ----------
    mesh : Mesh
        计算网格。
    U : torch.Tensor
        ``(n_cells, 3)`` 速度场。
    """

    def __init__(self, mesh, U, **kwargs):
        super().__init__(mesh, U, **kwargs)
        n_cells = mesh.n_cells
        device = U.device
        dtype = U.dtype

        # 初始化湍流量
        self.k = torch.zeros(n_cells, dtype=dtype, device=device)
        self.epsilon = torch.full((n_cells,), 1e-10, dtype=dtype, device=device)

    def compute_eddy_viscosity(self):
        """计算湍流涡粘性 nut。

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` 涡粘性场。
        """
        # 实现模型方程
        ...

    def solve(self, dt):
        """求解一个时间步的湍流方程。

        Parameters
        ----------
        dt : float
            时间步长。
        """
        # 求解 k 和 epsilon 的输运方程
        ...
```

### 注册与使用自定义模型

通过 `model_registry` 注册自定义模型后，可在求解器中使用：

```python
from pyfoam.models.registry import model_registry

# 注册
model_registry.register("myModel", MyTurbulenceModel)

# 在求解器中使用
solver = SimpleFoam("case/", turbulence_model="myModel")
solver.run()
```

### 湍流模型校准流程

1. **单元测试**：在均匀剪切流中验证涡粘性计算
2. **基准验证**：与湍流平板/通道流基准对比
3. **敏感性分析**：测试模型常数对结果的影响

---

## 自定义边界条件开发

### 边界条件架构

pyOpenFOAM 的边界条件通过 `pyfoam.boundary` 模块实现：

```
pyfoam.boundary/
├── boundary_condition.py      # 基类 BoundaryCondition
├── fixed_value.py             # fixedValue
├── zero_gradient.py           # zeroGradient
├── fixed_gradient.py          # fixedGradient
├── mixed.py                   # mixed (通用 Robin BC)
└── custom/                    # 用户自定义 BC 目录
```

### 边界条件基类

```python
from pyfoam.boundary.boundary_condition import BoundaryCondition

class OscillatingVelocityBC(BoundaryCondition):
    """振荡速度边界条件: U(t) = U_mean + U_amp * sin(omega * t).

    Parameters
    ----------
    patch_name : str
        边界面名称。
    U_mean : tuple[float, float, float]
        平均速度。
    U_amp : tuple[float, float, float]
        振幅。
    omega : float
        角频率 (rad/s)。
    """

    def __init__(self, patch_name, U_mean, U_amp, omega):
        super().__init__(patch_name, bc_type="fixedValue")
        self.U_mean = U_mean
        self.U_amp = U_amp
        self.omega = omega

    def apply(self, field, t, mesh):
        """在时间 t 更新边界面上的场值。

        Parameters
        ----------
        field : torch.Tensor
            被修改的场变量。
        t : float
            当前模拟时间。
        mesh : Mesh
            计算网格（用于获取面索引）。
        """
        import math
        value = tuple(
            m + a * math.sin(self.omega * t)
            for m, a in zip(self.U_mean, self.U_amp)
        )
        # 获取边界面单元并设置值
        patch_faces = mesh.get_patch_faces(self.patch_name)
        for face_idx in patch_faces:
            owner_cell = mesh.owner[face_idx]
            field[owner_cell] = torch.tensor(value, dtype=field.dtype)
```

### 使用自定义边界条件

```python
from pyfoam.boundary.custom.oscillating_velocity import OscillatingVelocityBC

# 定义振荡入口
bc_inlet = OscillatingVelocityBC(
    patch_name="inlet",
    U_mean=(1.0, 0.0, 0.0),
    U_amp=(0.2, 0.0, 0.0),
    omega=6.28,  # 1 Hz
)

# 在求解器中附加
solver = IcoFoam("case/", boundary_conditions={"inlet": bc_inlet})
solver.run()
```

---

## 可微分 CFD 用法

### 概述

pyOpenFOAM 的可微分 CFD 功能基于 PyTorch 的自动微分（autograd）引擎，
支持**物理信息神经网络 (PINN)**、**灵敏度分析**和**逆向设计**。

模块位于 `pyfoam.differentiable`：

```
pyfoam.differentiable/
├── linear_solver.py    # 可微分线性求解器
├── operators.py        # 可微分算子（梯度、散度、拉普拉斯）
└── simple.py           # 可微分 SIMPLE 算法
```

### 灵敏度分析

计算目标函数对设计变量的梯度：

```python
import torch
from pyfoam.differentiable.operators import DifferentiableOperators

# 创建可微分算子
ops = DifferentiableSolver(mesh)

# 压力场需要梯度
p = torch.zeros(n_cells, dtype=torch.float64, requires_grad=True)
U = torch.zeros((n_cells, 3), dtype=torch.float64, requires_grad=True)

# 运行求解器（前向计算）
U_out, p_out = ops.forward_step(U, p, dt=0.01)

# 定义目标函数（例如：出口平均速度）
J = U_out[exit_cells, 0].mean()

# 反向传播：计算 dJ/dp_initial
J.backward()

# 设计灵敏度
print(f"dJ/dp shape: {p.grad.shape}")
print(f"最大灵敏度: {p.grad.abs().max():.6e}")
```

### PINN 集成示例

```python
import torch
import torch.nn as nn

class NSNet(nn.Module):
    """物理信息神经网络：逼近 N-S 方程的解。"""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 3),  # (u, v, p)
        )

    def forward(self, xy):
        return self.net(xy)

def physics_loss(model, xy, nu):
    """N-S 方程残差作为损失函数。

    Parameters
    ----------
    model : NSNet
        神经网络模型。
    xy : torch.Tensor
        ``(N, 2)`` 采样点坐标（需要梯度）。
    nu : float
        运动粘性。
    """
    xy.requires_grad_(True)
    out = model(xy)
    u, v, p = out[:, 0], out[:, 1], out[:, 2]

    # 计算导数
    grad_u = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    grad_v = torch.autograd.grad(v, xy, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    grad_p = torch.autograd.grad(p, xy, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    u_x, u_y = grad_u[:, 0], grad_u[:, 1]
    v_x, v_y = grad_v[:, 0], grad_v[:, 1]
    p_x, p_y = grad_p[:, 0], grad_p[:, 1]

    # 二阶导数（拉普拉斯）
    lap_u = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
    lap_v = torch.autograd.grad(v_x, xy, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0]

    # 连续性方程: u_x + v_y = 0
    continuity = u_x + v_y

    # x-动量: u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
    momentum_x = u * u_x + v * u_y + p_x - nu * lap_u

    # y-动量: u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0
    momentum_y = u * v_x + v * v_y + p_y - nu * lap_v

    return (continuity ** 2).mean() + (momentum_x ** 2).mean() + (momentum_y ** 2).mean()
```

### 拓扑优化集成

可微分 CFD 可用于流体拓扑优化：

```python
# 设计变量：材料分布 (0 = 流体, 1 = 固体)
gamma = torch.ones(n_cells, dtype=torch.float64, requires_grad=True)

# 阻力模型: 有效粘性 = nu * (1 + q * gamma / (1 - q * gamma))
q = 0.01
mu_eff = nu * (1 + q * gamma / (1 - q * gamma + 1e-10))

# 运行求解器，计算压力降
dp = forward_solve(mu_eff, gamma, ...)

# 反向传播
dp.backward()

# 更新设计变量（梯度下降）
with torch.no_grad():
    gamma -= 0.01 * gamma.grad
```

---

## GPU 加速深入

### PyTorch 后端架构

pyOpenFOAM 的所有场数据和矩阵运算基于 PyTorch 张量，自动受益于：

- **CUDA 内核融合**：连续的逐元素操作被融合为单个 GPU 内核
- **异步执行**：CPU-GPU 数据传输与计算重叠
- **内存池**：PyTorch 的缓存分配器减少 `cudaMalloc` 调用

### 设备管理

```python
from pyfoam.core.device import DeviceManager, device_context

dm = DeviceManager()
print(f"CUDA 可用: {dm.capabilities.cuda}")
print(f"MPS 可用:  {dm.capabilities.mps}")
print(f"当前设备:  {dm.device}")

# 强制使用特定 GPU
with device_context(device="cuda:1"):
    solver = SimpleFoam("large_case/")
    solver.run()
```

### 混合精度策略

虽然 CFD 通常需要双精度，但某些操作可用单精度加速：

```python
import torch

# 求解器核心用 float64
solver = IcoFoam("case/")
solver.run()

# 后处理/可视化可用 float32
U_plot = solver.U.float().cpu().numpy()  # 节省内存
```

| 操作 | 推荐精度 | 说明 |
|------|----------|------|
| 矩阵组装 | float64 | 数值稳定性 |
| 线性求解 | float64 | 条件数敏感 |
| 梯度计算 | float64 | 累积误差 |
| 后处理 | float32 | 足够精度 |
| 可视化 | float32 | 内存节省 |

### 大规模问题内存管理

```python
import torch
from pyfoam.core.device import get_device

device = get_device()

# 检查 GPU 内存
if device.type == "cuda":
    mem_total = torch.cuda.get_device_properties(device).total_mem
    mem_used = torch.cuda.memory_allocated(device)
    mem_free = mem_total - mem_used
    print(f"GPU 空闲内存: {mem_free / 1e9:.1f} GB")

    # 估算网格需求
    n_cells = 1_000_000
    mem_per_cell = 8 * 20  # float64, ~20 个标量场
    print(f"网格需求: {n_cells * mem_per_cell / 1e9:.1f} GB")
```

### 多 GPU 并行策略

对于超大规模问题，可采用区域分解 + 多 GPU：

```python
import torch
import torch.distributed as dist

# 初始化分布式环境
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# 每个 GPU 处理一个子区域
local_solver = SimpleFoam(f"processor{rank}/")

# 每步后交换边界数据
for step in range(n_steps):
    local_solver.run_step()

    # 交换重叠层数据
    for field in [local_solver.U, local_solver.p]:
        recv_buf = torch.zeros_like(field[halo_cells])
        dist.alltoall(recv_buf, field[halo_cells])
        field[halo_cells] = recv_buf
```

### 性能调优检查清单

| 项目 | 检查内容 | 预期效果 |
|------|----------|----------|
| 张量连续性 | 确保 `tensor.is_contiguous()` | 避免隐式拷贝 |
| 批量操作 | 用向量化代替逐单元循环 | 10-100x 加速 |
| 减少 CPU-GPU 同步 | 避免 `.item()` / `.cpu()` 在循环中 | 减少延迟 |
| 预分配输出 | 在循环外创建输出张量 | 减少内存分配 |
| 使用 `torch.no_grad()` | 后处理不需梯度的计算 | 节省内存和时间 |
