# 验证指南

pyOpenFOAM 使用经典 CFD 基准案例来验证求解器的正确性。所有验证测试位于 `tests/validation/` 目录下。

---

## 验证案例总览

| 案例 | 求解器 | 基准文献 | 关键指标 | 容差 |
|------|--------|----------|----------|------|
| 盖驱动方腔 (Lid-driven cavity) | icoFoam | Ghia et al. 1982 | 中心线 u 速度剖面 | 20% (16x16 网格) |
| 泊肃叶流 (Poiseuille flow) | icoFoam | Hagen-Poiseuille 解析解 | 抛物线速度剖面 | 15% |
| 库埃特流 (Couette flow) | icoFoam | Couette 解析解 | 线性速度剖面 | 15% |
| 泰勒-格林涡 (Taylor-Green vortex) | icoFoam | Taylor & Green 1937 | 动能衰减率 | 25% |
| 后向台阶 (Backward-facing step) | simpleFoam | Armaly et al. 1983 | 再附着长度 Xr/h | 50% |
| 圆柱绕流 (Cylinder flow) | icoFoam | Williamson 1996 | Strouhal 数 | 50% |

---

## 如何运行验证测试

### 运行全部验证测试

```bash
CUDA_VISIBLE_DEVICES='' pytest tests/validation/ -v
```

### 运行单个验证案例

```bash
# 盖驱动方腔
CUDA_VISIBLE_DEVICES='' pytest tests/validation/test_lid_driven_cavity.py -v

# 泊肃叶流
CUDA_VISIBLE_DEVICES='' pytest tests/validation/test_poiseuille_flow.py -v

# 泰勒-格林涡
CUDA_VISIBLE_DEVICES='' pytest tests/validation/test_taylor_green_vortex.py -v
```

### 带详细输出运行

```bash
CUDA_VISIBLE_DEVICES='' pytest tests/validation/ -v -s
```

---

## 各案例详解

### 1. 盖驱动方腔 (Lid-driven Cavity)

**物理背景**: 封闭方腔内，顶壁以恒定速度运动，带动腔内流体形成涡旋结构。

**验证方法**: 将垂直中心线 (x=0.5) 上的 u 速度剖面与 Ghia et al. (1982) 的基准数据对比。

**基准数据**: Re=100 时，17 个 y 位置点的 u/U_lid 值。

**当前状态**:
- 使用 16x16 粗网格，容差 20%
- 精度目标: <5%（需网格加密和 SIMPLE/SIMPLEC 调优）

**参考文献**:
> Ghia, U., Ghia, K.N., Shin, C.T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *J. Comput. Phys.* 48, 387-411.

### 2. 泊肃叶流 (Poiseuille Flow)

**物理背景**: 压力驱动的平行板间流动，产生抛物线速度剖面。

**验证方法**: 对比数值速度剖面与解析解 u(y) = u_max * 4y(H-y) / H^2。

**当前状态**:
- 使用 8x16 网格
- PISO 求解器存在入口/出口 BC 传播问题，导致定量对比受限
- 验证求解器基本正确性（有限输出、形状合理性、非负性）

**参考**: Hagen-Poiseuille 解析解。

### 3. 库埃特流 (Couette Flow)

**物理背景**: 壁面驱动的平行板间流动，产生线性速度剖面。

**验证方法**: 对比数值速度剖面与解析解 u(y) = U_top * y / H。

**当前状态**:
- 使用解析解作为初始条件
- 验证求解器的形状保持和单调性

**参考**: Couette 解析解。

### 4. 泰勒-格林涡 (Taylor-Green Vortex)

**物理背景**: 二维周期性衰减涡旋，动能按 E(t)/E(0) = exp(-4*nut*t) 指数衰减。

**验证方法**: 监测动能衰减率并与解析衰减曲线对比。

**当前状态**:
- 壁面边界（非真周期 BC）引入边界效应
- 验证衰减率和流场结构，而非逐点精度

**参考文献**:
> Taylor, G.I. & Green, A.E. (1937). Mechanism of the production of small eddies from large ones. *Proc. R. Soc. Lond. A* 158, 499-521.

### 5. 后向台阶 (Backward-facing Step)

**物理背景**: 突然扩张通道，产生分离剪切层和回流区。

**验证方法**: 测量主回流区再附着长度 Xr/h，与 Armaly et al. (1983) 实验数据对比。

**几何参数**: 台阶高度 h=0.5，上游通道高度 h，下游通道高度 2h。

**当前状态**:
- Re_h ~ 100 时，Xr/h ~ 3-6（层流）
- 使用 simpleFoam（SIMPLE 算法）
- 较大容差（50%）反映粗网格和简化几何

**参考文献**:
> Armaly, B.F., Durst, F., Pereira, J.C.F., Schonung, B. (1983). Experimental and theoretical investigation of backward-facing step flow. *J. Fluid Mech.* 127, 473-496.

### 6. 圆柱绕流 (Cylinder Flow)

**物理背景**: 流体绕圆柱流动，在特定 Re 数下产生卡门涡街。

**验证方法**: 测量涡脱落频率的 Strouhal 数 St = f*D/U，与 Williamson (1996) 实验数据对比。

**基准数据**: Re=100 时，St = 0.164。

**当前状态**:
- 使用阶梯近似（staircase）网格表示圆柱
- 较大容差（50%）反映粗糙几何表示

**参考文献**:
> Williamson, C.H.K. (1996). Vortex dynamics in the cylinder wake. *Annu. Rev. Fluid Mech.* 28, 477-539.

---

## 解读结果

### 通过/失败标准

- **PASS**: 数值结果与基准数据的相对误差在指定容差内
- **FAIL**: 误差超出容差，需调查原因

### 常见失败原因

| 原因 | 解决方案 |
|------|----------|
| 网格太粗 | 加密网格（牺牲运行时间） |
| 未收敛 | 增加 SIMPLE/PISO 迭代次数或时间步数 |
| BC 设置错误 | 检查边界条件配置 |
| 数值格式精度不足 | 使用高阶格式（如 linearUpwind） |
| 时间步长过大 | 减小 deltaT |

### 提高精度的方法

1. **网格加密**: 更多单元 = 更小离散误差
2. **高阶格式**: `linearUpwind` 或 `cubic` 插值
3. **算法调优**: SIMPLEC 替代 SIMPLE 可加速收敛
4. **收敛准则**: 减小残差容差（如从 1e-4 降至 1e-6）

---

## 全单元测试套件

除验证测试外，还需运行完整单元测试确保代码正确性：

```bash
# 全部测试
CUDA_VISIBLE_DEVICES='' pytest tests/ -v

# 仅单元测试
CUDA_VISIBLE_DEVICES='' pytest tests/unit/ -v

# 仅验证测试
CUDA_VISIBLE_DEVICES='' pytest tests/validation/ -v
```
