# 单量子比特 Helstrom QSNN 实验

该任务直接把未知量子态的密度矩阵送入 QSNN，不使用经典坐标、像素或人工提取的密度矩阵元素作为分类特征。模型学习“相干旋转 + 结构化 Lindblad 吸收”测量，并与解析 Holevo–Helstrom 最优成功率比较。

## 模型

- 输入：两个非正交单量子比特状态 `rho0/rho1`；
- 相干阶段：三参数 Hermitian Hamiltonian；
- 耗散阶段：从两个输入基态向两个类别吸收节点的四条跳跃通道；
- 输出：两个类别节点的原始占据概率；
- 泄漏：仍留在输入子空间的概率，始终按判别失败处理，不做事后归一化；
- 审计：从训练后动力学重建有效 POVM，并检查两个类别效应和泄漏效应的正定性。

二态最小错误判别的理论上限为

$$
P_H=\frac12\left(1+\left\|\eta_0\rho_0-\eta_1\rho_1\right\|_1\right).
$$

## 快速运行

```powershell
python scripts/run_qubit_helstrom.py `
  --config configs/qubit_helstrom_smoke.json
```

若环境暂未安装 PyTorch，可先运行单量子比特 NumPy 参考后端。它在
`coherent_during_dissipation=false` 时与正式实现采用相同的“先相干、后耗散”
数学模型，并使用有限差分 Adam；该后端仅用于小系统交叉验证，不支持相干与耗散
交错推进：

```powershell
python scripts/run_qubit_helstrom_numpy.py `
  --config configs/qubit_helstrom_smoke.json
```

完整的 5 随机种子扫描：

```powershell
python scripts/run_qubit_helstrom.py `
  --config configs/qubit_helstrom_full.json
```

输出保存在 `outputs/quantum_tasks/qubit_helstrom/`，包括逐次运行 CSV、完整 JSON、汇总 JSON、模型 checkpoint 和训练轨迹。

## 晋级真机的建议条件

- 至少 5 个随机种子；
- 平均 Helstrom gap 不超过 1%；
- 最差 gap 不超过 2%；
- 加权输出泄漏不超过 0.5%；
- 有限 shots 评估不依赖一次偶然抽样；
- 有效 POVM 和泄漏效应均保持半正定；
- 与可训练幺正投影测量和固定 Pauli 测量同时比较。

满足条件后，优先把有效 POVM 编译为门模型设备上的 Naimark dilation；这属于功能等价部署。若要部署原始耗散动力学，还需要设备支持辅助比特、中途复位或等价的开放系统数字模拟。
