# pyOpenFOAM 验证数据目录

本目录存储 pyOpenFOAM 的验证测试数据。

## 目录结构

```
validation/results/
├── unit_tests/          # 单元测试结果
├── tutorial_tests/      # Tutorial 测试结果
├── solver_tests/        # 求解器验证结果
└── reports/             # 验证报告
```

## 最新验证结果

- **单元测试**: 17,080 passed / 0 failed
- **Tutorial 测试**: 367 passed / 14 xfailed
- **验证测试**: 208 passed
- **总计**: 17,567+ 测试通过

## 验证覆盖

- 17 个 OpenFOAM 求解器类别
- 16 个湍流模型
- 17 个离散格式
- 11 个线性求解器
- 6 个 ODE 求解器
- 等等
