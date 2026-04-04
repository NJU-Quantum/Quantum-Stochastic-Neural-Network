# Chebyshev Comparison 总结报告

## 1. 目的

本报告总结目录 `experiments/chebyshev_comparision/` 下已有 benchmark 与可视化结果，回答三个问题：

1. `Stage-1` 的 Chebyshev 优化是否优于当前默认的 `exact_state`
2. `Stage-2` 的 Suzuki/Strang splitting（`split`）是否优于当前默认的结构化 `RK4`
3. 当把“文档启发的两项优化”联合使用时，相对基线策略在不同 `N`、不同 `T_u/T_d` 下是否能在**相同 loss 精度**下更快

相关结果文件：

- `experiments/chebyshev_comparision/benchmark_stage1_methods_results.json`
- `experiments/chebyshev_comparision/benchmark_stage2_methods_results.json`
- `experiments/chebyshev_comparision/benchmark_stage2_sweep_loss_target_results.json`
- `experiments/chebyshev_comparision/benchmark_joint_methods_sweep_loss_target_results.json`
- `experiments/chebyshev_comparision/benchmark_joint_methods_sweep_raw.png`
- `experiments/chebyshev_comparision/benchmark_joint_methods_sweep_summary.png`

---

## 2. 方法口径

### 2.1 Stage-1 对比

- `exact_state`
  - 先对纯态做精确幺正演化
  - `psi_u = exp(-iHT_u) psi`
  - 再重建 `rho_u = psi_u psi_u^\dagger`
- `chebyshev`
  - 用 Chebyshev 递推近似 `exp(-iHT_u) psi`
  - 再重建密度矩阵

### 2.2 Stage-2 对比

- `rk4`
  - 当前原始结构化耗散演化器
  - 直接对结构化 RHS 做 RK4
- `split`
  - 新增的对称分裂方法
  - 形式为“半步相干 + 一整步结构化耗散精确更新 + 半步相干”

### 2.3 联合策略对比

- `baseline`
  - `stage1 = exact_state`
  - `stage2 = rk4`
- `optimized`
  - `stage1 = chebyshev`
  - `stage2 = split`

联合对比采用“同 loss 精度”规则：

- 基线参考：`baseline` 且 `stage2_steps = 12`
- 目标 loss：`L_target = L_baseline,12`
- 容差：`eps = 0.002`
- 在各自方法的 `stage2_steps ∈ {8,12,16,20,24,32}` 中，找满足
  - `final_loss <= L_target + eps`
  的最小耗时配置

---

## 3. Stage-1 结论

来源：`benchmark_stage1_methods_results.json`

### 3.1 数值精度

`max_rho_diff` 始终在 `1e-6` 量级：

- `N=100`: `3.34e-06`
- `N=200`: `1.67e-06`
- `N=300`: `3.25e-06`

说明 `chebyshev` 在当前参数下数值上是可信的近似实现。

### 3.2 速度

`Stage-1 only` 结果显示，`exact_state` 始终快于 `chebyshev`：

- `N=100`
  - `exact_state = 0.002213s`
  - `chebyshev = 0.003180s`
  - `time_ratio_exact_to_chebyshev = 0.696`
- `N=200`
  - `exact_state = 0.002359s`
  - `chebyshev = 0.004423s`
  - `time_ratio_exact_to_chebyshev = 0.533`
- `N=300`
  - `exact_state = 0.002585s`
  - `chebyshev = 0.006538s`
  - `time_ratio_exact_to_chebyshev = 0.395`

这里 `time_ratio_exact_to_chebyshev < 1` 表示 `exact_state` 更快。

### 3.3 对全模型和训练的影响

- 全模型前向中，`exact_state` 仍略快于 `chebyshev`
  - `N=100`: `0.0956s` vs `0.0987s`
  - `N=200`: `0.4961s` vs `0.5031s`
- 训练 benchmark 中，`exact_state` 也略快
  - `9.218s` vs `9.826s`

### 3.4 判断

结论明确：

- `Stage-1` 的 Chebyshev 路径**没有带来速度优势**
- 当前主线把默认 `Stage-1` 保持为 `exact_state` 是正确的
- `chebyshev` 更适合作为研究性对照实现，而不是默认训练路径

根因可以概括为：

- 当前 `chebyshev` 采用 Python 层逐阶递推
- GPU 上 `torch.matrix_exp` 对当前尺度的 Hermitian 矩阵已经很高效
- 因此 `chebyshev` 没有在工程上赢过精确纯态指数

---

## 4. Stage-2 结论

来源：`benchmark_stage2_methods_results.json`

### 4.1 数值精度

`split` 与 `rk4` 的状态差异仍很小：

- `Stage-2 only`
  - `N=100`: `max_rho_diff = 2.33e-06`
  - `N=200`: `4.08e-06`
  - `N=300`: `3.02e-06`

说明新增的 splitting 方法在当前测试下与原始 `rk4` 保持了较接近的数值结果。

### 4.2 速度

`Stage-2 only` 中，`split` 有稳定而显著的加速：

- `N=100`: `time_ratio_rk4_to_split = 4.438`
- `N=200`: `3.202`
- `N=300`: `3.711`

全模型前向也显著加速：

- `N=100`: `3.395`
- `N=200`: `2.922`

训练 benchmark：

- `rk4 = 9.092s`
- `split = 3.853s`
- `time_ratio_rk4_to_split = 2.360`

### 4.3 训练效果

在固定 `stage2_steps = 12` 时：

- `rk4 final_loss = 0.063081`
- `split final_loss = 0.081333`

说明：

- `split` 明显更快
- 但在同样步数设置下，训练终点的 loss 略差于 `rk4`

### 4.4 判断

结论是：

- `Stage-2 split` 已经充分证明了**速度优势**
- 但它是否能直接无条件替代 `rk4`，还要看在“同精度”口径下是否仍更快

---

## 5. Stage-2 同精度结果

来源：`benchmark_stage2_sweep_loss_target_results.json`

### 5.1 规则

基线：

- `rk4`
- `stage2_steps = 12`

目标：

- `target_loss = 0.063081`
- `eps = 0.002`

### 5.2 结果

原始 sweep 中：

- `rk4, steps=12`
  - `train_seconds = 8.868s`
  - `final_loss = 0.063081`
- `split, steps=20`
  - `train_seconds = 5.217s`
  - `final_loss = 0.054602`

满足同精度容差条件后：

- `best_rk4 = steps 12, 8.868s`
- `best_split = steps 20, 5.217s`

### 5.3 判断

这说明即使按“同 loss 精度”比较：

- `split` 依然快于 `rk4`
- 且优势仍然明显，约快 `1.70x`

因此 `Stage-2 split` 不只是“低精度换速度”，而是在当前任务上已经表现出真实的精度-速度优势。

---

## 6. 联合策略结果

来源：`benchmark_joint_methods_sweep_loss_target_results.json`

策略定义：

- `baseline = exact_state + rk4`
- `optimized = chebyshev + split`

### 6.1 N=100

不同 `T_u/T_d` 下，`optimized` 都在“同 loss 精度”条件下更快。

#### `T_u/T_d = 0.5`

- `best_baseline`
  - `stage2_steps = 8`
  - `train_seconds = 7.392s`
  - `final_loss = 0.04819`
- `best_optimized`
  - `stage2_steps = 8`
  - `train_seconds = 4.157s`
  - `final_loss = 0.05462`

速度上 `optimized` 更快。

#### `T_u/T_d = 1.0`

- `best_baseline`
  - `stage2_steps = 12`
  - `train_seconds = 11.963s`
- `best_optimized`
  - `stage2_steps = 8`
  - `train_seconds = 3.677s`

加速非常明显。

#### `T_u/T_d = 2.0`

- `best_baseline`
  - `stage2_steps = 8`
  - `train_seconds = 6.480s`
- `best_optimized`
  - `stage2_steps = 12`
  - `train_seconds = 4.428s`

同样是 `optimized` 更快。

### 6.2 N=200

从结果文件可见，在 `N=200` 的各项 sweep 中：

- `baseline` 训练时间已明显升高，典型量级约 `22s` 到 `88s`
- `optimized` 对应量级明显更低，典型约 `8.5s` 到 `23s`

即使单个 `stage2_steps` 对应的 `final_loss` 有波动，整体趋势依然显示：

- 联合优化策略的主要时间收益来自 `stage2 split`
- `stage1 chebyshev` 本身并不快，但没有抵消掉 `stage2 split` 带来的大幅收益

### 6.3 如何理解联合策略结果

需要特别强调：

- 联合策略更快，**主要原因是 `stage2 split`**
- 不是因为 `stage1 chebyshev` 本身更快

也就是说：

- `optimized = chebyshev + split` 之所以整体表现好
- 本质上是 `split` 的收益大于 `chebyshev` 的损失

因此，如果未来追求“最佳工程默认组合”，更值得优先考虑的其实是：

- `exact_state + split`

而不是必须保留 `chebyshev + split`

---

## 7. CUDA 与并行实现观察

从当前代码和 benchmark 行为看，已有的 CUDA 友好优化包括：

- batched `psi` / `rho` 张量计算
- 广播式结构化 RHS
- 不显式构造 Liouvillian 大矩阵
- benchmark 中的 `torch.cuda.synchronize()` 保证计时可信

但也仍有局限：

- `chebyshev` 采用 Python 层递推循环
- 没有 `torch.compile`
- 没有 AMP / mixed precision
- benchmark 脚本本身没有特别激进的显存保护逻辑

这也是 `Stage-1 chebyshev` 难以在 GPU 上胜出的重要原因。

---

## 8. 最终结论

### 8.1 已经可以确定的结论

1. `Stage-1 Chebyshev`：
   - 数值上可靠
   - 但当前实现下**不如 `exact_state` 快**
   - 不适合作为默认 Stage-1 方法

2. `Stage-2 split`：
   - 对当前任务和当前 GPU 环境，**明显快于 `rk4`**
   - 在固定步数和同精度两种口径下都表现出显著速度优势
   - 是当前最成功、最值得保留的优化

3. 联合优化策略：
   - 在多个 `N` 与 `T_u/T_d` 条件下，整体训练时间通常优于基线
   - 主要收益来源依然是 `stage2 split`

### 8.2 当前最推荐的主线策略

如果以工程默认实现为目标，当前最推荐的方向是：

- `Stage-1 = exact_state`
- `Stage-2 = split`

原因是：

- `exact_state` 已经比 `chebyshev` 更快
- `split` 已经比 `rk4` 更快

这比当前联合 sweep 里的

- `chebyshev + split`

更符合现有 benchmark 的最优趋势。

### 8.3 后续建议

1. 用 `exact_state + split` 再做一轮正式同精度 sweep
2. 若继续研究 `chebyshev`，优先优化其实现而不是直接当默认方法
   - 更积极的截断
   - 缓存缩放参数和系数
   - 尝试 `torch.compile`
3. 对联合策略建议进一步增加多 seed 平均，减少单次训练波动对结论的影响

---

## 9. 一句话总结

本轮实验的真正有效优化是 **Stage-2 的 Suzuki/Strang splitting**；`Stage-1 Chebyshev` 在当前实现下更像一个理论可行但工程上暂未胜出的对照项。综合现有结果，最值得推进为主线默认方案的是 **`exact_state + split`**。
