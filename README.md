# Quantum-Stochastic-Neural-Network

## 项目概览

该仓库实现了基于量子随机行走与 Lindblad 动力学的 QSNN（Quantum Stochastic Neural Network）模型，当前主线以 PyTorch 为核心，面向三类任务：

- 一维函数拟合：`QSNNFunction`
- 二维分类：`QSNN2D`
- 文本/句子识别：`QSNNText`

当前主线重点在于：

- 统一的量子演化后端 `qsw.py`
- 主模型定义 `models.py`
- 面向二维分类的结构化 Stage-2 优化
- 新增的 Stage-1 Chebyshev 对照路径与基准测试

---

## 当前目录结构

### 主线代码

- `qsw.py`
  - 量子演化数值后端
  - 包含 `evolve_expm`、`evolve_unitary`、`evolve_from_operators`
  - 包含 `QSNN2D` 的结构化 Stage-2 Lindblad RHS 与 RK4 演化器
  - 包含新增的 `evolve_state_chebyshev()`，用于 Stage-1 的纯态 Chebyshev 演化
- `models.py`
  - `QSNNFunction`：一维函数拟合
  - `QSNN2D`：二维分类
  - `QSNNText`：文本任务
- `data.py`
  - 二维玩具数据集，如 `make_circles`
- `Chebyshev.md`
  - Chebyshev 时间推进理论笔记

### 实验目录

- `experiments/tu_td_sweeps`
  - `plot_train_boundary.py`：不同总神经元规模下的二维分类训练与边界可视化
  - `sweep_tu_td_grid.py`：`T_u / T_d / stage2_steps` 网格实验
  - 若干 `.png/.csv/.md` 结果文件
- `experiments/chebyshev_comparision`
  - `benchmark_stage1_methods.py`：`Stage-1 exact` 与 `Stage-1 chebyshev` 对照基准

### 历史/归档区域

- `cpl_project_and_data`
- `prr_project_and_data`

这两部分主要用于历史实验和论文复现，不是当前主线优化的重点。

---

## 主线架构

### 1. 数据编码

输入样本先被编码成量子态。

- `QSNNFunction`：把标量 `x` 编码成幂次特征态
- `QSNN2D`：把二维输入 `(x, y)` 编码成纯态 `psi`
  - 前一半输入节点承载 `x` 的幂次
  - 后一半输入节点承载 `y` 的幂次
- 纯态再可重建为密度矩阵 `rho = psi psi^\dagger`

### 2. Stage-1：输入层相干演化

`QSNN2D` 的第一阶段只在输入层子块上施加 Hermitian Hamiltonian：

- 输入层神经元之间是可训练的相干耦合
- 输出节点在这一阶段不参与演化

当前支持两种 Stage-1 方法：

- `stage1_method="exact"`
  - 直接计算 `U = exp(-i H T_u)`
  - 再做 `rho_u = U rho_0 U^\dagger`
- `stage1_method="chebyshev"`
  - 先对纯态做 Chebyshev 演化
  - `psi_u = exp(-i H T_u) psi`
  - 再重建 `rho_u = psi_u psi_u^\dagger`

其中 `chebyshev` 路径是新增的对照实现，用于比较纯态递推近似与原始矩阵指数路径的效率。

### 3. Stage-2：结构化 Lindblad 演化

`QSNN2D` 第二阶段采用结构化的开放系统演化：

- 每个输入节点都单向连接到两个输出节点
- 跳跃算符固定为
  - `L_{o,j} = gamma[o,j] |o><j|`
- 当前实现不再显式构造整组 `L_k`
- 而是直接按固定拓扑计算结构化 RHS

时间推进采用：

- 结构化 RHS
- RK4 数值积分

这一段对应 `qsw.py` 中的：

- `_lindblad_rhs_qsnn2d_structured()`
- `evolve_qsnn2d_stage2_structured()`

### 4. 输出读出

Stage-2 演化结束后，读取两个输出节点的对角元：

- `p0 = rho[out0, out0]`
- `p1 = rho[out1, out1]`

然后归一化为二分类概率。

---

## 运行环境

### 依赖

见 `requirements.txt`：

- `torch`
- `numpy`
- `matplotlib`
- `scipy`
- `qutip`
- `nltk`
- `PyYAML`

## 常用命令

### 1. 运行二维分类边界实验

```bash
python experiments/tu_td_sweeps/plot_train_boundary.py
```

### 2. 运行 Stage-1 方法对照基准

脚本位置：

- `experiments/chebyshev_comparision/benchmark_stage1_methods.py`

默认运行全模型前向、Stage-1 单独耗时、以及训练对照：

```bash
python experiments/chebyshev_comparision/benchmark_stage1_methods.py
```

保存结果到 JSON：

```bash
python experiments/chebyshev_comparision/benchmark_stage1_methods.py \
  --out experiments/chebyshev_comparision/benchmark_stage1_methods_results.json
```

指定设备和规模：

```bash
python experiments/chebyshev_comparision/benchmark_stage1_methods.py \
  --device cpu \
  --forward-ns 100,200 \
  --stage1-ns 100,200,300 \
  --train-n 100 \
  --train-steps 100 \
  --batch-size 512
```

只跑训练对照：

```bash
python experiments/chebyshev_comparision/benchmark_stage1_methods.py \
  --skip-forward --skip-stage1
```

---

## 当前性能优化状态

相较于更早的通用实现，当前主线已经做了两类关键优化：

- `Stage-1`：当仅有 Hamiltonian 演化时，直接走幺正演化路径
- `Stage-2`：对 `QSNN2D` 使用结构化 Lindblad RHS，避免显式构造通用跳跃算符与大规模 Liouvillian

当前新增的 Chebyshev 路径主要用于继续评估：

- 对纯幺正 Stage-1，纯态 Chebyshev 递推是否比 `matrix_exp` 更高效
- 在不同 `N`、不同 batch、不同设备上，效率收益是否稳定

---

## 说明

`README` 现在主要描述当前仍在使用的主线实现和实验入口。历史实验区中的批量脚本、论文归档与旧路径没有逐一展开，如需追溯可再进入对应目录查看。

---

## QSNN-QGAN 增量实现

`qgan/` 在现有 `qsw.py` 后端上增加了概率幅度编码、PQC 生成器、QSNN 判别器薄封装、标准幺正 VQC 判别器、Trace-Z 目标、量子态指标和 checkpoint。两个组合后端的含义为：

- `cheby_suzuki`：Hamiltonian 相干子流使用 Chebyshev，耗散子流参与二阶 Suzuki/Strang 组合；
- `suzuki_global`：相干子流与耗散子流都按 Suzuki 组合推进。

结构化耗散子流利用当前固定跳跃拓扑的解析更新，不构造完整 Liouvillian。
QSNN 跳跃参数采用“每个输入节点的总物理速率 + real/fake 分支概率”，默认按
`target_output_mass: 0.8` 初始化；VQC 使用读出量子比特的两个完备子空间，因而
不会产生结构性输出泄漏。训练可用 `leakage_penalty` 抑制 QSNN 后续重新增大泄漏。

确认环境：

```powershell
conda run -n qsnn python -c "import torch, yaml; print(torch.__version__, torch.cuda.is_available(), yaml.__version__)"
```

运行全部 QGAN 测试（包含可用时的 CUDA 检查）：

```powershell
conda run -n qsnn python -m unittest discover -s tests -p "test_qgan_*.py" -v
```

运行不依赖下载数据的 64 维 smoke test：

```powershell
conda run -n qsnn python scripts/train_qgan.py --config configs/smoke.yaml
```

MNIST 0 的 QSNN 与标准幺正 VQC 训练：

```powershell
conda run -n qsnn python scripts/train_qgan.py --config configs/mnist0_64.yaml --download
conda run -n qsnn python scripts/train_qgan.py --config configs/mnist0_64_vqc.yaml --download
```

上面两份配置保留为“直接 8×8 像素生成”消融。当前主实验先训练共享的
28×28→64 概率瓶颈 Autoencoder，再冻结 Encoder/Decoder，只在可测的64维概率
潜空间中比较 QSNN full 与 VQC：

```powershell
conda run -n qsnn python scripts/train_autoencoder.py --config configs/autoencoder_mnist0_64.yaml --download
conda run -n qsnn python scripts/run_ae64_comparison.py --device cuda
```

两种判别器共用同一个冻结 Autoencoder、增强的4层噪声重上传生成器和每批次
`1D:3G` 更新预算。输出目录同时保存初始生成图、最终784维解码图和
`decoder_baselines.png`；后者展示随机/均匀潜变量经过Decoder的结果，用来防止把
Decoder自身携带的MNIST先验误判为QGAN学习效果。

增强生成器在内部仍可使用复数相位完成干涉，但送入判别器前会按计算基测量概率
规范化为非负实振幅。真实样本与生成样本因而采用同一种概率幅编码，判别器不能
利用与最终图像无关的相位差异取巧。

`H`-only 与 `L`-only 消融共用同一配置和训练预算：

```powershell
conda run -n qsnn python scripts/train_qgan.py --config configs/mnist0_64.yaml --ablation h_only --output-dir outputs/qgan/mnist0_64_h_only
conda run -n qsnn python scripts/train_qgan.py --config configs/mnist0_64.yaml --ablation l_only --output-dir outputs/qgan/mnist0_64_l_only
```

从 checkpoint 继续训练时，配置中的总 epoch 应大于 checkpoint 已完成的 epoch：

```powershell
conda run -n qsnn python scripts/train_qgan.py --config configs/smoke.yaml --epochs 2 --resume outputs/qgan/synthetic_64_qsnn_smoke/checkpoint_latest.pt
```

每次运行会在输出目录保存配置副本、环境与参数量信息、`metrics.csv`、`metrics.json` 和最新 checkpoint。指标同时包含 Trace-Z、直接判别成功率、输出质量、leakage、梯度范数、物理性漂移、量子态距离、耗时和 CUDA 峰值显存。

---

## 量子原生任务：单量子比特 Helstrom 判别

`tasks/quantum_state_discrimination/` 新增了直接接收量子密度矩阵的最小错误态判别实验。模型使用相干旋转与结构化 Lindblad 跳跃学习二输出 POVM，并与解析 Helstrom 上界、可训练幺正投影测量和固定 Pauli 测量比较。输出泄漏按失败处理，不通过归一化抬高成功率。

快速实验：

```powershell
python scripts/run_qubit_helstrom.py --config configs/qubit_helstrom_smoke.json
```

完整五随机种子扫描：

```powershell
python scripts/run_qubit_helstrom.py --config configs/qubit_helstrom_full.json
```

详细定义和评价口径见 `tasks/quantum_state_discrimination/README.md`。

当前 NumPy 参考后端已完成 125 次完整扫描，平均 Helstrom gap 为
`0.000505874`，全部预设数值门槛通过。结果仍需正式 PyTorch/autograd 后端
交叉验证；“硬件候选”不表示已授权提交硬件任务。简要记录见
`docs/reports/单量子比特_Helstrom_QSNN_实验报告.md`，包含模型推导、QSNN
作用与优势边界、代表性 POVM 和补充实验矩阵的正式报告见
`docs/reports/单量子比特_Helstrom_QSNN_正式实验报告.md`。
