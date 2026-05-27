# pyOpenFOAM 实例教程

本指南提供 5 个完整的端到端实例，覆盖 pyOpenFOAM 的核心功能。
每个实例包括：问题描述、代码实现、运行方法和结果验证。

---

## 目录

1. [顶盖驱动方腔流（Lid-Driven Cavity）](#1-顶盖驱动方腔流)
2. [后台阶流动（Backward-Facing Step）](#2-后台阶流动)
3. [溃坝问题（Dam Break）](#3-溃坝问题)
4. [自然对流（Natural Convection）](#4-自然对流)
5. [湍流通道流动（Turbulent Channel）](#5-湍流通道流动)

---

## 1. 顶盖驱动方腔流

### 问题描述

经典的 CFD 验证案例。正方形腔体 [0,1]x[0,1] 中，
顶壁以 U=(1,0,0) 运动，其余三壁无滑移。
在 Re=100 下将中心线速度剖面与 Ghia et al. (1982) 的基准数据对比。

### 求解器

`icoFoam`（不可压缩层流 Navier-Stokes）

### 完整代码

```python
"""顶盖驱动方腔流 — Re=100, icoFoam."""
import torch
from pyfoam.applications.ico_foam import IcoFoam

# 创建求解器并加载网格（假设案例目录已准备好）
solver = IcoFoam("cases/cavity16")

# 运行模拟
converged = solver.run()
print(f"收敛状态: {converged}")

# 提取中心线 u-速度剖面
centres = solver.mesh.cell_centres.detach().cpu().numpy()
u = solver.U[:, 0].detach().cpu().numpy()

n_x = 16
mid_i = n_x // 2
centreline_y = [centres[j * n_x + mid_i, 1] for j in range(n_x)]
centreline_u = [u[j * n_x + mid_i] for j in range(n_x)]

# 与 Ghia et al. (1982) 数据对比（容差 25%，粗网格）
# y=0.5 处 u 应约为 -0.2058
print(f"y=0.5 处 u ≈ {centreline_u[8]:.4f} (参考值: -0.2058)")
```

### 关键结果

| 指标 | 预期值 | 说明 |
|------|--------|------|
| 峰值 \|U\| | 1.0 ~ 1.5 | 接近顶壁速度 |
| y=0.5 处 u | ~-0.21 | Ghia 基准值 -0.2058 |
| 最大残差 | < 1e-4 | PISO 收敛准则 |

### 运行方式

```bash
# 直接运行测试
CUDA_VISIBLE_DEVICES='' python -m pytest tests/validation/test_lid_driven_cavity.py -v
```

---

## 2. 后台阶流动

### 问题描述

突扩通道中的流动分离与再附着。上游通道高度 h，下游通道高度 2h。
在 Re_h 约 100 时，一级再附着长度 X_r/h 约 3~6。
验证参考：Armaly et al. (1983)。

### 求解器

`simpleFoam`（不可压缩 RANS）

### 完整代码

```python
"""后台阶流动 — simpleFoam."""
import torch
from pyfoam.applications.simple_foam import SimpleFoam

solver = SimpleFoam("cases/backwardStep")
converged = solver.run()
print(f"收敛状态: {converged}")

# 提取 x 方向速度场
u = solver.U[:, 0].detach().cpu().numpy()
centres = solver.mesh.cell_centres.detach().cpu().numpy()

# 寻找再附着点：在近底壁处 u 由负变正的位置
# （回流区末端）
import numpy as np
h = 0.5  # 台阶高度
y_near_bottom = 0.1  # 近底壁 y 坐标

# 底部附近单元
near_bottom = np.abs(centres[:, 1] - y_near_bottom) < 0.1
x_nb = centres[near_bottom, 0]
u_nb = u[near_bottom]

# 排序后查找 u=0 交叉点
sort_idx = np.argsort(x_nb)
x_sorted = x_nb[sort_idx]
u_sorted = u_nb[sort_idx]

# 再附着点：u 从负变正
sign_changes = np.where(np.diff(np.sign(u_sorted)))[0]
if len(sign_changes) > 0:
    x_reattach = x_sorted[sign_changes[0]]
    x_r_h = (x_reattach - 1.0) / h  # L_upstream = 1.0
    print(f"再附着长度 X_r/h ≈ {x_r_h:.1f} (参考值: 3~6)")
```

### 关键结果

| 指标 | 预期值 | 说明 |
|------|--------|------|
| X_r/h | 3 ~ 6 | Re_h ≈ 100 层流范围 |
| 最大残差 | < 1e-4 | SIMPLE 收敛准则 |
| 质量守恒 | < 1% | 入出口流量平衡 |

### 运行方式

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/validation/test_backward_facing_step.py -v
```

---

## 3. 溃坝问题

### 问题描述

经典 VoF 两相流验证案例。水柱在重力作用下坍塌，
比较水面锋面位置随时间的变化与 Martin & Moyce (1952) 的实验数据。

无量纲参数：
- t* = t * sqrt(2g/a)（无量纲时间）
- X/a（无量纲锋面位置，a 为初始水柱半宽）

### 求解器

`interFoam`（不可压缩两相 VoF）

### 完整代码

```python
"""溃坝问题 — interFoam VoF."""
import numpy as np
import torch
from pyfoam.applications.inter_foam import InterFoam

solver = InterFoam("cases/damBreak")

# 记录初始水体积
alpha_0 = solver.alpha.detach().cpu().numpy().copy()
volumes = solver.mesh.cell_volumes.detach().cpu().numpy()
V0 = np.sum(alpha_0 * volumes)
print(f"初始水体积: {V0:.6e} m³")

# 运行模拟
converged = solver.run()
print(f"收敛状态: {converged}")

# 检查质量守恒
alpha_f = solver.alpha.detach().cpu().numpy()
Vf = np.sum(alpha_f * volumes)
rel_error = abs(Vf - V0) / V0
print(f"体积守恒误差: {rel_error:.4%}")

# 计算锋面位置
centres = solver.mesh.cell_centres.detach().cpu().numpy()
water_mask = alpha_f > 0.5
if np.any(water_mask):
    x_front = centres[water_mask, 0].max()
    a = 0.146  # 初始水柱半宽
    print(f"锋面位置 X/a = {x_front / a:.2f} (t* ≈ 3 时参考值: ~1.86)")

# 验证 alpha 范围
print(f"alpha 范围: [{alpha_f.min():.4f}, {alpha_f.max():.4f}]")
assert alpha_f.min() >= -0.1 and alpha_f.max() <= 1.1
```

### Martin & Moyce (1952) 参考数据

| t* | X/a |
|----|-----|
| 0.0 | 1.00 |
| 2.0 | 1.40 |
| 4.0 | 2.36 |
| 6.0 | 3.36 |
| 8.0 | 4.27 |

### 运行方式

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/validation/test_multiphase_dam_break.py -v
```

---

## 4. 自然对流

### 问题描述

封闭方腔内的自然对流。左侧壁加热 T_h，右侧壁冷却 T_c，
其余壁面绝热。Rayleigh 数 Ra = 10^6 时，流动呈稳态对流涡结构。
验证参考：de Vahl Davis (1983) 基准数据。

### 求解器

`buoyantBoussinesqSimpleFoam`（Boussinesq 近似浮力驱动）

### 完整代码

```python
"""自然对流 — buoyantBoussinesqSimpleFoam."""
import numpy as np
import torch
from pyfoam.applications.solver_base import SolverBase

solver = SolverBase("cases/natConvection")

# 读取初始温度场
T = solver.get_field("T", 0).detach().cpu().numpy()
centres = solver.mesh.cell_centres.detach().cpu().numpy()

# 运行模拟
converged = solver.run()
print(f"收敛状态: {converged}")

# 提取水平中心线温度剖面
T_final = solver.get_field("T", 0).detach().cpu().numpy()
n_x = 20
mid_i = n_x // 2
n_y = 20

y_cl = []
T_cl = []
for j in range(n_y):
    idx = j * n_x + mid_i
    y_cl.append(centres[idx, 1])
    T_cl.append(T_final[idx])

# Nusselt 数计算
# Nu = -dT/dx|_{x=0} * L / (T_h - T_c)
# 粗略估计: Nu ≈ (T_h - T_1st_cell) / (dx/2) * L / deltaT
T_h, T_c = 1.0, 0.0
deltaT = T_h - T_c
L = 1.0

# 左壁第一列单元的平均温度梯度
left_cells = np.where(centres[:, 0] < 0.05)[0]
T_left = T_final[left_cells].mean()
dx = 1.0 / n_x
Nu_approx = (T_h - T_left) / (dx / 2) * L / deltaT
print(f"近似 Nusselt 数: {Nu_approx:.2f}")
# Ra=10^6 时参考值: Nu ≈ 8.80 (de Vahl Davis 1983)

# 验证温度范围
assert T_final.min() >= T_c - 0.1
assert T_final.max() <= T_h + 0.1
print(f"温度范围: [{T_final.min():.4f}, {T_final.max():.4f}]")
```

### de Vahl Davis (1983) 基准数据

| Ra | Nu (平均) |
|----|-----------|
| 10^3 | 1.118 |
| 10^4 | 2.243 |
| 10^5 | 4.519 |
| 10^6 | 8.800 |

### 运行方式

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/validation/test_natural_convection.py -v
```

---

## 5. 湍流通道流动

### 问题描述

两个无限大平行平板之间的充分发展湍流流动。
Re_tau = 395（基于摩擦速度和半通道高度）。
验证壁面法向速度剖面与 Moser, Kim & Mansour (1999) 的 DNS 数据对比。

### 求解器

`simpleFoam` + kOmegaSST 湍流模型

### 完整代码

```python
"""湍流通道流动 — simpleFoam + kOmegaSST."""
import numpy as np
import torch
from pyfoam.applications.simple_foam import SimpleFoam

solver = SimpleFoam("cases/turbChannel")

# 运行模拟
converged = solver.run()
print(f"收敛状态: {converged}")

# 提取中心线速度剖面
centres = solver.mesh.cell_centres.detach().cpu().numpy()
u = solver.U[:, 0].detach().cpu().numpy()

# Re_tau = 395: u_tau ≈ U_bulk / (1/0.41 * ln(Re_tau) + 5.0)
# 或直接从壁面剪切应力计算
n_y = 32  # 网格分辨率
n_x = 4
n_z = 4

# 法向速度剖面（沿 y 方向取平均）
y_profile = []
u_profile = []
for j in range(n_y):
    row_cells = [j * n_x + i for i in range(n_x)]
    y_profile.append(centres[row_cells[0], 1])
    u_profile.append(np.mean(u[row_cells]))

y_arr = np.array(y_profile)
u_arr = np.array(u_profile)

# 无量纲化: y+ = y * u_tau / nu, u+ = u / u_tau
# 粗略估计 u_tau from max velocity
u_tau_est = u_arr.max() / 20.0  # rough estimate
nu = 1e-4
y_plus = y_arr * u_tau_est / nu
u_plus = u_arr / u_tau_est

# 验证: 近壁处 y+ < 1 (粘性底层应被解析)
# Log-law 区域: u+ ≈ 1/0.41 * ln(y+) + 5.0
print(f"估计 u_tau: {u_tau_est:.4f}")
print(f"y+ 范围: [{y_plus.min():.1f}, {y_plus.max():.1f}]")

# 检查 log-law 区域 (y+ > 30)
log_law_mask = (y_plus > 30) & (y_plus < 200)
if np.any(log_law_mask):
    kappa = 0.41
    B = 5.0
    u_plus_log = 1.0 / kappa * np.log(y_plus[log_law_mask]) + B
    u_plus_sim = u_plus[log_law_mask]
    max_error = np.max(np.abs(u_plus_sim - u_plus_log))
    print(f"Log-law 区域最大误差: {max_error:.2f}")

# 速度应关于中心线对称
half = n_y // 2
for i in range(half):
    err = abs(u_arr[i] - u_arr[n_y - 1 - i])
    assert err < 0.1 * abs(u_arr.max()), f"速度不对称: y={y_arr[i]:.3f}"

print("湍流通道验证通过")
```

### Moser, Kim & Mansour (1999) DNS 参考数据

| y+ | u+ |
|----|-----|
| 0.5 | 0.5 |
| 5.0 | 5.0 |
| 11.0 | 8.0 |
| 50.0 | 17.0 |
| 100.0 | 21.0 |
| 200.0 | 24.5 |
| 395.0 | ~26.0 |

### 运行方式

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/validation/test_turbulent_channel.py -v
```

---

## 通用建议

### 性能优化

- **GPU 加速**：设置环境变量 `CUDA_VISIBLE_DEVICES=0` 使用 GPU
- **CPU 模式**：设置 `CUDA_VISIBLE_DEVICES=''` 强制使用 CPU
- **精度选择**：默认 float64，可通过 `pyfoam.core.dtype` 切换

### 调试技巧

1. **检查网格**：`from pyfoam.tools.check_mesh import check_mesh`
2. **查看场信息**：`from pyfoam.tools.foam_info import foam_info`
3. **导出 VTK**：`from pyfoam.tools.foam_to_vtk import foam_to_vtk` 用于 ParaView 可视化

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| NaN/Inf | 时间步过大 | 减小 deltaT 或启用 adjustTimeStep |
| 不收敛 | 初始条件不合理 | 使用 potentialFoam 初始化 |
| 质量不守恒 | VoF 界面扩散 | 减小时间步或使用 MULES 限制器 |
