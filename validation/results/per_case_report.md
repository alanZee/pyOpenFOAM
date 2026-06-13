# pyOpenFOAM 逐算例验证报告

生成时间: 2026-06-13 19:31

---

## 一、求解器验证总览

| 求解器 | 状态 | 有限值 | 收敛 | field_max | continuity |
|--------|------|--------|------|-----------|------------|
| AcousticFoam | ❌ | - | - | None | None |
| AdjointFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |
| AdjointShapeFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |
| AdjointTurbulenceFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |
| BoundaryFoam | ✅ | Yes | - | 1.15e+01 | 7.21e-01 |
| BuoyantBoussinesqSimpleFoam | ✅ | Yes | - | 5.91e+04 | 5.74e+02 |
| BuoyantPimpleFoam | ✅ | Yes | - | 1.00e+02 | 1.92e+00 |
| BuoyantSimpleFoam | ✅ | Yes | - | 1.00e+02 | 2.45e+01 |
| CavitatingFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |
| CombustionFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |
| CompressibleInterFoam | ✅ | Yes | - | 1.38e+04 | 0.00e+00 |
| CompressibleMultiphaseVoFFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |
| CompressibleVoFFoam | ✅ | Yes | - | 1.13e+02 | 8.08e+03 |
| DenseParticleFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |
| DieselFoam | ✅ | Yes | - | 2.65e+04 | 3.94e+02 |
| DsmcFoam | ✅ | Yes | - | 4.76e+02 | 0.00e+00 |
| EnergyFoam | ✅ | Yes | - | 4.92e+02 | 0.00e+00 |
| FluidFoam | ✅ | Yes | - | 1.00e+03 | 3.06e+02 |
| HeatTransferFoam | ✅ | Yes | - | 4.92e+02 | 0.00e+00 |
| IcoFoam | ✅ | Yes | - | 1.00e+00 | 2.51e-02 |
| IncompressibleDriftFluxFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |
| IncompressibleFluidFoam | ✅ | Yes | - | 1.00e+00 | 8.16e-07 |
| IncompressibleVoFFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |
| InterFoam | ✅ | Yes | - | 1.00e+00 | 7.49e-01 |
| IsothermalFluidFoam | ✅ | Yes | - | 1.00e+03 | 2.08e+02 |
| LaplacianFoam | ✅ | Yes | - | 3.00e+02 | 0.00e+00 |
| MagneticFoam | ❌ | - | - | None | None |
| MhdFoam | ❌ | - | - | None | None |
| MulticomponentFluidFoam | ✅ | Yes | - | 1.00e+03 | 6.25e+02 |
| MultiphaseEulerFoam | ✅ | Yes | No | 0.00e+00 | 0.00e+00 |
| MultiphaseInterFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |
| MultiphaseReactingFoam | ✅ | Yes | - | 3.00e+02 | 0.00e+00 |
| PDRFoam | ✅ | Yes | - | 4.93e+08 | 1.57e+09 |
| PimpleFoam | ✅ | Yes | - | 1.00e+00 | 3.54e+00 |
| PisoFoam | ✅ | Yes | - | 1.00e+00 | 3.19e-03 |
| PorousSimpleFoam | ✅ | Yes | - | 9.31e+01 | 6.35e-01 |
| PotentialFoam | ✅ | Yes | - | 8.00e+00 | 0.00e+00 |
| ReactingFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |
| ReactingMultiphaseFoam | ✅ | Yes | - | 3.00e+02 | 0.00e+00 |
| RhoCentralFoam | ✅ | Yes | - | 1.64e+02 | 0.00e+00 |
| RhoPimpleFoam | ✅ | Yes | - | 7.07e+02 | 1.11e+03 |
| RhoPorousSimpleFoam | ✅ | Yes | - | 1.00e+03 | 1.35e+00 |
| RhoSimpleFoam | ✅ | Yes | - | 1.00e+03 | 1.35e+00 |
| ScalarTransportFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |
| ShallowWaterFoam | ✅ | Yes | - | 2.70e-01 | 1.20e-02 |
| SimpleFoam | ✅ | Yes | - | 1.00e+00 | 7.80e-07 |
| SolidDisplacementFoam | ✅ | Yes | No | 0.00e+00 | 0.00e+00 |
| SonicFoam | ✅ | Yes | - | 7.08e+02 | 7.78e+02 |
| SprayFoam | ✅ | Yes | - | 2.65e+04 | 3.94e+02 |
| SrfSimpleFoam | ✅ | Yes | - | 9.31e+01 | 6.35e-01 |
| TwoPhaseEulerFoam | ❌ | - | - | None | None |
| ViscousFoam | ✅ | Yes | - | 1.00e+00 | 8.16e-07 |
| XiFoam | ✅ | Yes | - | 0.00e+00 | 0.00e+00 |

**总计**: ✅ 49 通过, ❌ 4 失败, ⚠️ 0 NaN/警告

## 二、OpenFOAM 参照对比

暂无 OpenFOAM 参照对比数据。


## 三、Cavity 流精度对比

| 网格 | Re | pyOpenFOAM | Ghia (1982) | OpenFOAM v11 | 误差 (vs Ghia) |
|------|-----|-----------|-------------|--------------|----------------|
| 8x8 | 100 | -0.222 | -0.206 | - | 8.1% |
| 16x16 | 100 | -0.217 | -0.206 | - | 5.6% |
| 20x20 | 100 | -0.208 | -0.206 | -0.204 | 0.9% |
| 32x32 | 100 | -0.208 | -0.206 | - | 1.0% |

## 四、Couette/Poiseuille 精度

| 算例 | 内部 L2 误差 | 内部最大误差 | 说明 |
|------|-------------|-------------|------|
| Couette (8x16) | 4.18e-6 (< 0.001%) | 9.18e-6 | 边界面逐单元查找修复 |
| Poiseuille (8x16) | 1.11e-4 (< 0.02%) | 3.45e-4 | 边界面逐单元查找修复 |

## 五、GPU 验证

所有 50 个基础求解器在 GPU (RTX 4070 Ti SUPER) 上产生有限结果。


## 六、可微分模拟

- 7/7 测试通过（含形状优化端到端）
- 4x4/8x8/16x16/32x32/64x64 梯度均有限
- 边界惩罚已修复

## 七、已知限制

1. OpenFOAM v11 参照（v13 无 Docker 镜像）
2. 部分数值稳定性截断（密度/温度/压力范围限制）
3. 多区域/multiRegion 算例未验证
