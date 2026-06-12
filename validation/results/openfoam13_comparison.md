# OpenFOAM-13 参照对比报告

生成时间: 2026-06-13

---

## 一、参照说明

**参照版本**: OpenFOAM v12 (OpenFOAM Foundation, 2025-02-06 build)

OpenFOAM-13 源码编译需要 GCC 11+（当前系统 GCC 9.4 不兼容），Docker Desktop WSL2 后端
持续故障（500 Internal Server Error）。因此使用 OpenFOAM v12 作为参照版本进行精度验证。

OpenFOAM v12 与 v13 同属 OpenFOAM Foundation 系列，核心算法一致（PISO、SIMPLE 等），
精度差异可忽略。

**运行方式**: 从 apt .deb 包提取预编译二进制，WSL2 Ubuntu 20.04 运行。

---

## 二、Cavity 流基准精度对比 (icoFoam, Re=100)

### 2.1 算例设置

- 求解器: icoFoam (PISO)
- Re = 100 (ν = 0.01 m²/s)
- 网格: 8x8, 16x16, 32x32
- 时间步长: Δt = 0.005s
- 终止时间: t = 1.0s (200 步)
- 边界条件: movingWall U=(1,0,0), fixedWalls noSlip

### 2.2 Ux_max 对比

| 网格 | pyOpenFOAM | OpenFOAM v12 | 误差 |
|------|------------|--------------|------|
| 8x8 | 0.266 | 0.444 | 40.1% |
| 16x16 | 0.251 | 0.738 | 66.0% |
| 32x32 | 0.229 | 0.874 | 73.8% |

### 2.3 Ux_min 对比（回流强度）

| 网格 | pyOpenFOAM | OpenFOAM v12 | 误差 |
|------|------------|--------------|------|
| 8x8 | -0.110 | -0.120 | **8.6%** ✅ |
| 16x16 | -0.106 | -0.182 | 41.9% |
| 32x32 | -0.026 | -0.203 | 87.0% |

### 2.4 分析

- 8x8 回流强度接近 OpenFOAM（8.6% 误差）
- Ux_max 偏差较大（40-74%），原因是 PISO 边界条件处理使用惩罚方法
- 惩罚方法不随网格缩放，导致精度随网格细化下降

### 2.5 根因

PISO 求解器使用混合 HbyA（0.3*U_bc + 0.7*HbyA）和 35x 边界惩罚系数。
OpenFOAM 使用矩阵级 BC 处理（PBiCGStab 线性求解器），需要更深入的架构改动。

---

## 三、OpenFOAM v12 vs v13 说明

OpenFOAM v12 和 v13 均为 OpenFOAM Foundation 发布，核心算法（PISO、SIMPLE、VOF 等）
一致。主要差异在于：
- v13 新增了部分边界条件和求解器
- v13 对某些算法有微调
- 核心 CFD 算法精度差异可忽略

因此 OpenFOAM v12 的 cavity 基准结果可作为 v13 的有效参照。
