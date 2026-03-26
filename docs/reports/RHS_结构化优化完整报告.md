# QSNN2D 专用 RHS 结构化优化完整报告

## 1. RHS 全称

RHS 全称是 **Right-Hand Side**，即微分方程“右端项”。

在本项目中，`RHS` 指密度矩阵演化方程
$$
\dot{\rho}(t)=\mathcal{F}(\rho(t))
$$
里的右端算子 $\mathcal{F}(\rho)$。

对应实现核心位于：
- [qsw.py](qsw.py)
- 新增/使用的关键函数：`_lindblad_rhs_qsnn2d_structured()`、`evolve_qsnn2d_stage2_structured()`、`evolve_unitary()`

---

## 2. 背景：QSNN2D 两阶段动力学

`QSNN2D` 在 [models.py](models.py) 的 `forward()` 中分两段：

1. **Stage-1（相干）**：只含哈密顿项，得到 $\rho_u$
2. **Stage-2（耗散+相干）**：含 Lindblad 跳跃，得到最终 $\rho_{out}$

总方程（Stage-2）是标准 Lindblad 形式：
$$
\dot\rho=-i[H,\rho]+\sum_k\left(L_k\rho L_k^\dagger-\frac{1}{2}\{L_k^\dagger L_k,\rho\}\right)
$$
其中 $\{A,B\}=AB+BA$。

---

## 3. 修改前：计算结构（通用算符路径）

修改前，Stage-2 由 [models.py](models.py) 中 `QSNN2D.forward()` 动态构造 `Ls` 列表：
- 每个输入节点 $j$ 对两个输出节点各建一个算符
- 总数约 $2N_{in}$ 个 `L_k`

随后走通用演化路径（`evolve*`）：

- 方案 A：显式 Liouvillian + `matrix_exp`
- 方案 B：通用 `lindblad_rhs` + RK4
- 方案 C：Krylov 近似（大规模时 batch 路径会退化成逐样本）

### 3.1 结构代价

通用路径包含以下高开销：

1. 每轮前向重复构造 `L_k`
2. 大量矩阵乘法 `L_kρL_k^†` 和 `L_k^†L_k`
3. 若走显式 Liouvillian，维度从 $N$ 放大到 $N^2$，代价高

这对大规模（如 $N\approx100$）不友好。

---

## 4. 修改后：专用 RHS 结构化路径

核心思想：利用 `QSNN2D` Stage-2 的**固定拓扑**，把通用 Lindblad 项化简为结构表达。

### 4.1 固定跳跃结构

Stage-2 跳跃算符固定为
$$
L_{o,j}=\gamma_{o,j}|o\rangle\langle j|,
$$
其中：
- $o\in\{o_0,o_1\}$（两个输出节点）
- $j=0,1,\dots,N_{in}-1$（输入节点）

### 4.2 结构化分解

将 RHS 分解为三部分：

1. **相干项**
$$
-i[H,\rho]
$$

2. **反对易阻尼项（可对角化）**
$$
-\frac{1}{2}\{D,\rho\},\quad
D=\mathrm{diag}(d_0,\dots,d_{N-1}),
$$
其中输入节点阻尼系数
$$
d_j=\sum_{o\in\{o_0,o_1\}}|\gamma_{o,j}|^2
$$
输出节点对应阻尼为 0。

3. **跳跃增益项（只作用输出对角元）**
$$
\Delta\dot\rho_{o,o}=\sum_{j=0}^{N_{in}-1}|\gamma_{o,j}|^2\,\rho_{j,j}
$$
其余位置不加该增益。

### 4.3 代码实现对应

- 结构 RHS：`_lindblad_rhs_qsnn2d_structured()` in [qsw.py](qsw.py)
- 时间推进：`evolve_qsnn2d_stage2_structured()` in [qsw.py](qsw.py)（RK4）
- `QSNN2D` Stage-2 调用切换：`QSNN2D.forward()` in [models.py](models.py)

---

## 5. 修改前后计算结构对比
· 旧路径：先造“工具矩阵” L_k（(2N_{in}) 个）→ 再计算

· 新路径：直接算结果项（不造这些工具矩阵）

### 5.1 Stage-2 计算流程对比

**修改前（通用）**
1. 构造 `Ls = [L_k]`
2. 通用 Lindblad 路径计算 RHS
3. 通用演化器推进（expm / Krylov / RK4）

**修改后（结构化）**
1. 不再显式构造 `Ls`
2. 直接按固定拓扑计算结构 RHS
3. RK4 推进 `evolve_qsnn2d_stage2_structured()`

### 5.2 复杂度直观变化

- 删除了 `2N_{in}` 个算符对象构建与相关矩阵链乘
- 避免显式 Liouvillian 超大矩阵路径
- 对 batch 计算更友好（广播/向量化更充分）

---

## 6. 与 Stage-1 的协同优化

为配合大规模计算，`evolve_auto()` 在 `Ls` 为空时直接调用 `evolve_unitary()`：
$$
\rho(T)=U\rho(0)U^\dagger,\quad U=e^{-iHT}
$$

这一步是数学等价替换，不改变模型方程，仅去掉不必要的通用开销。

---

## 7. 数值精度影响评估

### 7.1 物理模型层面

未改变目标动力学形式：仍是同一 Lindblad 主方程，仅利用了 `QSNN2D` 的结构先验进行等价重写。

### 7.2 数值求解层面

Stage-2 采用 RK4 离散积分，存在步长误差：
- `stage2_steps` 越大，误差越小、耗时越高
- `stage2_steps` 越小，速度越快、误差略大

因此这是**可调的精度-速度折中**，不是模型定义改变。

---

## 8. 结果与意义（面向 N≈100）

本次改造目标是“大规模可运行”。结构化 RHS 路径解决了：

1. 大规模下通用路径的构造与内存压力
2. GPU 上某些路径（如批量 Krylov 退化）导致的超慢问题

结果上，`N=100` 的 `QSNN2D` 训练步已可稳定跑通（详见同目录实验图和脚本）。

---

## 9. 相关文件清单

- 核心动力学实现：[qsw.py](qsw.py)
- 模型调用入口：[models.py](models.py)
- N=100 可视化脚本：[experiments/tu_td_sweeps/plot_n100_train_boundary.py](experiments/tu_td_sweeps/plot_n100_train_boundary.py)
- N=100 图像结果：[experiments/tu_td_sweeps/train_2d_n100.png](experiments/tu_td_sweeps/train_2d_n100.png)

