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

- [qsw.py](/Users/hronrad/codes/py/quantum/Quantum-Stochastic-Neural-Network/qsw.py)
  - 量子演化数值后端
  - 包含 `evolve_expm`、`evolve_unitary`、`evolve_from_operators`
  - 包含 `QSNN2D` 的结构化 Stage-2 Lindblad RHS 与 RK4 演化器
  - 包含新增的 `evolve_state_chebyshev()`，用于 Stage-1 的纯态 Chebyshev 演化
- [models.py](/Users/hronrad/codes/py/quantum/Quantum-Stochastic-Neural-Network/models.py)
  - `QSNNFunction`：一维函数拟合
  - `QSNN2D`：二维分类
  - `QSNNText`：文本任务
- [data.py](/Users/hronrad/codes/py/quantum/Quantum-Stochastic-Neural-Network/data.py)
  - 二维玩具数据集，如 `make_circles`
- [Chebyshev.md](/Users/hronrad/codes/py/quantum/Quantum-Stochastic-Neural-Network/Chebyshev.md)
  - Chebyshev 时间推进理论笔记

### 实验目录

- [experiments/tu_td_sweeps](/Users/hronrad/codes/py/quantum/Quantum-Stochastic-Neural-Network/experiments/tu_td_sweeps)
  - `plot_train_boundary.py`：不同总神经元规模下的二维分类训练与边界可视化
  - `sweep_tu_td_grid.py`：`T_u / T_d / stage2_steps` 网格实验
  - 若干 `.png/.csv/.md` 结果文件
- [experiments/chebyshev_comparision](/Users/hronrad/codes/py/quantum/Quantum-Stochastic-Neural-Network/experiments/chebyshev_comparision)
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

见 [requirements.txt](/Users/hronrad/codes/py/quantum/Quantum-Stochastic-Neural-Network/requirements.txt)：

- `torch`
- `numpy`
- `matplotlib`
- `scipy`
- `qutip`
- `nltk`

### 推荐解释器

当前实验建议使用仓库上一级目录的虚拟环境：

```bash
../.venv311/bin/python
```

如果当前目录是仓库根目录：

```bash
cd /Users/hronrad/codes/py/quantum/Quantum-Stochastic-Neural-Network
../.venv311/bin/python -V
```

---

## 常用命令

### 1. 运行二维分类边界实验

```bash
cd /Users/hronrad/codes/py/quantum/Quantum-Stochastic-Neural-Network
../.venv311/bin/python experiments/tu_td_sweeps/plot_train_boundary.py
```

### 2. 运行 Stage-1 方法对照基准

脚本位置：

- [experiments/chebyshev_comparision/benchmark_stage1_methods.py](/Users/hronrad/codes/py/quantum/Quantum-Stochastic-Neural-Network/experiments/chebyshev_comparision/benchmark_stage1_methods.py)

默认运行全模型前向、Stage-1 单独耗时、以及训练对照：

```bash
cd /Users/hronrad/codes/py/quantum/Quantum-Stochastic-Neural-Network
../.venv311/bin/python experiments/chebyshev_comparision/benchmark_stage1_methods.py
```

保存结果到 JSON：

```bash
../.venv311/bin/python experiments/chebyshev_comparision/benchmark_stage1_methods.py \
  --out experiments/chebyshev_comparision/benchmark_stage1_methods_results.json
```

指定设备和规模：

```bash
../.venv311/bin/python experiments/chebyshev_comparision/benchmark_stage1_methods.py \
  --device cpu \
  --forward-ns 100,200 \
  --stage1-ns 100,200,300 \
  --train-n 100 \
  --train-steps 100 \
  --batch-size 512
```

只跑训练对照：

```bash
../.venv311/bin/python experiments/chebyshev_comparision/benchmark_stage1_methods.py \
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
