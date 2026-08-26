# 开放系统轨迹 QSNN-GAN 对照实验详细计划书

> 文档性质：实验设计与服务器实施计划，不包含尚未运行的实验结果。
>
> 核心目标：比较 QSNN 判别器与参数量匹配的 VQC 判别器，能否在相同生成器、数据、优化预算和随机种子下，更有效地约束连续、不可逆并趋向稳态的开放量子系统轨迹。

---

## 1. 科学问题与预注册假设

本实验不只要求生成器在单个时间点生成正确的密度矩阵，而是要求一系列生成态

\[
\rho_G(t_0),\rho_G(t_1),\ldots,\rho_G(t_n)
\]

同时满足以下性质：

1. 每个时间点的状态接近真实状态；
2. 相邻状态具有正确的时间顺序和演化速度；
3. 相干振荡、能量衰减和相位退相干符合目标动力学；
4. 长时间演化趋向正确稳态；
5. 对未见时间、未见耗散参数和未见时间步长能够泛化。

主科学问题为：

> 当判别对象是一段开放系统动力学，而不是独立量子态快照时，具有相干—耗散结构的 QSNN 判别器，是否比纯幺正 VQC 判别器带来更好的生成质量、动力学一致性、数据效率或训练速度？

预注册主假设：QSNN-GAN 在“未见时间 + 未见耗散参数”的联合 OOD 测试集上取得更低的平均迹距离，并在动力学残差、稳态误差和时间外推上保持同方向优势。

---

## 2. 目标物理系统

### 2.1 两量子比特驱动—耗散模型

第一版正式任务使用两个相互耦合的量子比特。Hamiltonian 定义为

\[
H=
\frac{J}{2}(X_1X_2+Y_1Y_2)
+\frac{\Omega}{2}(X_1+X_2)
+\frac{\Delta}{2}(Z_1-Z_2).
\]

各项物理含义如下：

- \(J\)：两个量子比特之间的激发交换强度；
- \(\Omega\)：外场驱动强度；
- \(\Delta\)：两个量子比特的相对失谐；
- \(X_i,Y_i,Z_i\)：作用在第 \(i\) 个量子比特上的 Pauli 算符。

建议固定

\[
J=1,\qquad \Omega=0.7,\qquad \Delta=0.3,
\]

并采用无量纲时间

\[
\tau=Jt.
\]

### 2.2 能量衰减和纯退相干

两个量子比特使用相同的能量衰减率 \(\gamma\) 和纯退相干率 \(\gamma_\phi\)。跳跃算符定义为

\[
L_{1}=\sqrt{\gamma}\,\sigma_-^{(1)},\qquad
L_{2}=\sqrt{\gamma}\,\sigma_-^{(2)},
\]

\[
L_{3}=\sqrt{\frac{\gamma_\phi}{2}}\,Z_1,\qquad
L_{4}=\sqrt{\frac{\gamma_\phi}{2}}\,Z_2.
\]

必须使用上式中的 \(\sqrt{\gamma_\phi/2}\) 约定。这样单量子比特非对角元由纯退相干贡献的衰减率正好是 \(\gamma_\phi\)，并满足

\[
\frac{1}{T_2}=\frac{1}{2T_1}+\frac{1}{T_\phi}
=\frac{\gamma}{2}+\gamma_\phi.
\]

禁止在代码和报告中混用 \(\sqrt{\gamma_\phi}Z\) 与 \(\sqrt{\gamma_\phi/2}Z\) 两种约定。

### 2.3 Lindblad 主方程

真实轨迹满足

\[
\frac{d\rho}{dt}
=-i[H,\rho]
+\sum_{k=1}^{4}
\left(
L_k\rho L_k^\dagger
-\frac12\{L_k^\dagger L_k,\rho\}
\right).
\]

初始状态固定为

\[
\rho(0)=|10\rangle\langle10|.
\]

正式主实验先固定初态，以免同时改变时间、耗散参数和初态导致容量预检无法解释。多初态泛化列为后续扩展实验。

---

## 3. 真实轨迹的生成与数值验证

### 3.1 真值生成

对两量子比特密度矩阵进行向量化，构造 \(16\times16\) Liouvillian 超算符 \(\mathcal L\)，通过高精度矩阵指数生成真值：

\[
|\rho(t)\rangle\rangle
=e^{t\mathcal L}|\rho(0)\rangle\rangle.
\]

真值生成不得使用待比较的 QSNN 演化近似，否则会把数值后端偏差混入数据标签。

正式数据建议使用 `complex128` 生成并缓存。训练时允许将缓存转换为 `complex64`，但所有最终评价应至少用 `complex128` 复算一次。

### 3.2 真值单元测试

每个参数组合必须检查：

- \(\operatorname{Tr}\rho(t)=1\)；
- \(\rho(t)=\rho(t)^\dagger\)；
- 最小特征值不低于数值容差；
- \(t=0\) 时恢复指定初态；
- \(\gamma=\gamma_\phi=0\) 时与幺正演化一致；
- 纯退相干不直接改变计算基布居；
- 只有振幅衰减时激发数随时间总体下降；
- 矩阵指数结果与高精度 ODE/RK4 参考结果一致。

推荐容差：`complex128` 下最大绝对误差 \(<10^{-10}\)，`complex64` 下 \(<10^{-5}\)。

---

## 4. 轨迹对任务定义

### 4.1 为什么不能只判别单个快照

如果判别器只接收 \(\rho(t)\)，生成器可以把任务当作普通条件回归：

\[
t\mapsto\rho(t).
\]

这种模型不一定理解时间方向、半群关系或相邻状态的动力学联系，因而不能充分测试 QSNN 的开放系统归纳偏置。

### 4.2 时钟量子比特编码

对相邻时间点构造轨迹对：

\[
\left(\rho(t),\rho(t+\Delta t)\right).
\]

加入一个时钟量子比特，将轨迹对编码成合法的 \(8\times8\) 密度矩阵：

\[
\Xi(t,\Delta t)
=\frac12|0\rangle\langle0|\otimes\rho(t)
+\frac12|1\rangle\langle1|\otimes\rho(t+\Delta t).
\]

真实轨迹对为

\[
\Xi_R=\Xi\!\left(\rho_R(t),\rho_R(t+\Delta t)\right),
\]

生成轨迹对为

\[
\Xi_G=\Xi\!\left(\rho_G(t),\rho_G(t+\Delta t)\right).
\]

时钟态的 \(|0\rangle\) 与 \(|1\rangle\) 明确标记先后顺序，因此交换两个时间点会得到不同输入。

### 4.3 困难负样本

为了迫使判别器学习轨迹关系，而不是只检查单态质量，判别器训练中加入两类真实状态构成的困难负样本：

1. 时间反转：

   \[
   \Xi_{\mathrm{rev}}
   =\frac12|0\rangle\langle0|\otimes\rho(t+\Delta t)
   +\frac12|1\rangle\langle1|\otimes\rho(t).
   \]

2. 随机拼接：将不同 \(\gamma\)、\(\gamma_\phi\) 或非相邻时间的真实状态拼成一对。

真实正样本、生成负样本和困难负样本在每个 batch 中使用固定比例，QSNN 和 VQC 必须完全一致。

---

## 5. 条件变量与数据划分

### 5.1 条件向量

单态生成器接收

\[
c_G=(\tau,\gamma/J,\gamma_\phi/J).
\]

轨迹判别器接收

\[
c_D=(\tau,\Delta\tau,\gamma/J,\gamma_\phi/J).
\]

所有连续条件在进入模型前线性映射到 \([-1,1]\)。归一化上下界只由训练协议确定，不得根据正式测试结果重新选择。

### 5.2 训练集

训练时间：

\[
\tau\in\{0,0.3,0.6,\ldots,6.0\},
\qquad \Delta\tau=0.3.
\]

训练耗散参数：

\[
\gamma/J\in\{0.10,0.30,0.50\},
\]

\[
\gamma_\phi/J\in\{0,0.10,0.20\}.
\]

### 5.3 模型选择验证集

验证时间使用训练网格中点：

\[
\tau\in\{0.15,0.45,0.75,\ldots,5.85\}.
\]

验证耗散参数：

\[
\gamma/J\in\{0.20,0.40\},
\qquad
\gamma_\phi/J\in\{0.05,0.15\}.
\]

验证集只用于检查点选择和超参数冻结，不得用于正式显著性计算。

### 5.4 正式测试集

正式测试划分为四类：

1. 时间插值：训练参数下的中点时间；
2. 参数插值：

   \[
   \gamma/J\in\{0.15,0.25,0.35,0.45\},
   \]

   \[
   \gamma_\phi/J\in\{0.025,0.075,0.125,0.175\};
   \]

3. 联合 OOD：未见时间和未见耗散参数同时出现；
4. 长时间外推：

   \[
   \tau\in\{6.3,6.6,\ldots,8.0\}.
   \]

另外测试未见时间步长

\[
\Delta\tau\in\{0.15,0.45\}.
\]

主指标使用“未见时间 + 未见耗散参数”的联合 OOD 集。所有正式测试条件必须在运行正式种子前写入配置并冻结。

---

## 6. 公共生成器

### 6.1 结构

QSNN-GAN 和 VQC-GAN 使用同一个条件纯化 PQC 生成器：

- 系统量子比特：2；
- 环境量子比特：2；
- 初始深度：16 层；
- 每层包含单量子比特 `RY/RZ` 和交替方向的环形 CNOT；
- 条件 \((\tau,\gamma,\gamma_\phi)\) 在各层重上传；
- 输出对环境做偏迹，得到 \(4\times4\) 物理密度矩阵。

生成器输出为

\[
\rho_G(c_G)
=\operatorname{Tr}_{E}
|\Psi_G(c_G)\rangle\langle\Psi_G(c_G)|.
\]

### 6.2 禁止硬编码真实动力学

不得在公共生成器内部调用真实 \(e^{t\mathcal L}\)，不得把真实稳态或解析衰减公式直接写入输出层。否则任务会被结构先验提前解决，无法判断判别器是否带来优势。

允许使用通用的条件编码器、时间 Fourier 特征或更深的公共 PQC，但任何增强必须：

- 同时用于 QSNN-GAN 和 VQC-GAN；
- 在正式训练前通过容量预检确定；
- 在报告中记录参数量和选择理由；
- 不包含目标 Lindblad 模型的专用解析公式。

---

## 7. 判别器与参数匹配

### 7.1 QSNN 轨迹判别器

轨迹对输入维度为 8，建议使用 `8-8-2` QSNN：

- 输入层：8；
- 隐藏层：8；
- 输出层：2，分别表示 real/fake；
- 相干边：输入—隐藏全连接，加输入层完全图；
- 耗散边：输入到隐藏、隐藏到输出的单向跳跃。

基础可训练参数估算：

- 相干参数：\(8\times8+\binom82=92\)；
- 耗散率：\(8\times8+2\times8=80\)；
- 合计：172。

若四维条件对每个参数采用仿射调制，总参数约为

\[
172\times(1+4)=860.
\]

耗散率必须通过 `softplus` 等映射保持非负。必须记录输出层质量，防止输入质量长期滞留造成输出泄漏。

### 7.2 VQC 轨迹判别器

VQC 输入为三个量子比特的轨迹对状态，并增加一个读出辅助量子比特：

- 数据量子比特：3；
- 读出辅助量子比特：1；
- 每层使用 `RY/RZ` 和环形纠缠；
- 四维条件在各层仿射重上传；
- 初始建议深度：22 层。

估算参数量为

\[
2\times4\times22\times(1+4)=880,
\]

与 QSNN 的 860 个参数相差约 2.3%。最终实现必须以代码实际统计值为准，并要求差异不超过 5%。

### 7.3 QSNN 数值后端

主实验使用 `cheby_suzuki`：

- Hamiltonian 相干演化使用 Chebyshev 算法；
- 耗散演化使用 Suzuki 算法。

数值消融使用 `suzuki_global`：

- 相干演化和耗散演化都使用 Suzuki 算法。

两种后端必须先在小批量轨迹对上与高精度 `matrix_exp` 参考比较，检查状态误差和梯度误差。若近似误差达到主模型差异的 10% 以上，不能进入正式比较。

---

## 8. 对抗目标与训练方式

记判别器对输入轨迹对的 real 分数为 \(s_D(\Xi,c_D)\)。判别器损失建议为

\[
\mathcal L_D
=-\mathbb E[s_D(\Xi_R)]
+\frac12\mathbb E[s_D(\Xi_G)]
+\frac12\mathbb E[s_D(\Xi_{\mathrm{hard}})].
\]

生成器损失为

\[
\mathcal L_G=-\mathbb E[s_D(\Xi_G)].
\]

其中 \(\Xi_{\mathrm{hard}}\) 由时间反转和随机拼接样本等比例组成。

初始训练配置：

| 项目 | 初始值 |
|---|---:|
| epochs | 1000 |
| batch size | 16；显存允许时测试 32 |
| discriminator steps | 1 |
| generator steps | 5 |
| generator learning rate | 0.002 |
| discriminator learning rate | 0.002 |
| EMA decay | 0.995 |
| gradient clipping | 5.0 |
| learning-rate decay start | 300 |
| record interval | 25 epochs |

生成器的初始参数、条件 batch、真实轨迹对、困难负样本排列和随机数计划必须在两个模型之间配对一致。

---

## 9. 评价指标

### 9.1 主指标

联合 OOD 平均迹距离：

\[
\overline D_{\mathrm{joint}}
=\mathbb E_{(t,\gamma,\gamma_\phi)\in\mathrm{joint\ OOD}}
D\!\left(\rho_G(t),\rho_R(t)\right).
\]

其中

\[
D(\rho,\sigma)=\frac12\|\rho-\sigma\|_1.
\]

### 9.2 单态生成指标

- 平均与最差 Uhlmann 保真度；
- 平均与最大迹距离；
- 纯度 MAE；
- 激发态总布居 MAE；
- 非对角相干项 MAE；
- 两量子比特负性 MAE；
- 局域 Bloch 向量误差；
- Pauli 相关张量误差；
- 迹、Hermitian 性和最小特征值。

### 9.3 动力学一致性指标

使用真实 Liouvillian 只进行测试，不参与生成器前向过程。一步动力学误差定义为

\[
E_{\mathrm{dyn}}
=D\!\left(
\rho_G(t+\Delta t),
e^{\Delta t\mathcal L}\rho_G(t)
\right).
\]

有限差分 Lindblad 残差定义为

\[
E_{\mathrm{res}}
=\left\|
\frac{\rho_G(t+\Delta t)-\rho_G(t)}{\Delta t}
-\mathcal L[\rho_G(t)]
\right\|_F.
\]

半群一致性定义为

\[
E_{\mathrm{semi}}
=D\!\left(
\rho_G(t+s),
e^{s\mathcal L}\rho_G(t)
\right).
\]

稳态误差定义为

\[
E_{\mathrm{ss}}
=D\!\left(\rho_G(t_{\max}),\rho_{\mathrm{ss}}\right),
\]

其中 \(\mathcal L[\rho_{\mathrm{ss}}]=0\)。

### 9.4 训练效率指标

- 达到指定主指标阈值所需 epochs；
- 达到阈值所需墙钟时间；
- 单 epoch 时间；
- 峰值显存；
- 生成器和判别器梯度范数；
- QSNN 输出层质量；
- 训练失败或 NaN 比例。

---

## 10. 必要对照与消融

正式报告至少包含以下对照：

1. **VQC-GAN 主基线**：参数量和预算匹配；
2. **快照判别消融**：只判别 \(\rho(t)\)，不使用轨迹对；
3. **时间反转控制**：交换 \(t\) 和 \(t+\Delta t\)；
4. **时间标签打乱**：状态不变但随机置换 \(t\)；
5. **耗散标签打乱**：随机置换 \(\gamma,\gamma_\phi\)；
6. **随机拼接控制**：拼接来自不同轨迹的真实状态；
7. **闭系统控制**：令 \(\gamma=\gamma_\phi=0\)；
8. **生成器容量控制**：直接监督拟合真实轨迹；
9. **QSNN 输出质量控制**：排除输出泄漏；
10. **数值后端消融**：`cheby_suzuki` 对比 `suzuki_global`。

关键解释原则：如果 QSNN 只在轨迹对任务中领先，而在快照或闭系统控制中优势消失，才支持“优势来自开放系统动力学结构”。

---

## 11. 分阶段实验协议

### 阶段 A：代码与物理单元测试

必须通过：

- Liouvillian 构造测试；
- 退相干率归一化测试；
- 真值状态物理性测试；
- 轨迹对迹为 1、正定和 Hermitian 测试；
- 时间反转确实改变时钟编码测试；
- QSNN/VQC 前向、反向和参数计数测试；
- CPU 与 CUDA 最小一致性测试；
- 两种 QSNN 数值后端精度测试。

### 阶段 B：公共生成器容量预检

生成器直接最小化真实密度矩阵误差，不使用判别器。容量种子：

\[
0,1,2.
\]

通过门槛：

- 联合 OOD 平均保真度 \(>0.995\)；
- 联合 OOD 最差保真度 \(>0.98\)；
- 联合 OOD 平均迹距离 \(<0.02\)；
- 长时间外推平均迹距离 \(<0.05\)；
- 生成态物理性通过。

若失败，应先增强双方共享的生成器，不得继续正式 GAN 后把失败归因于判别器。

### 阶段 C：冒烟实验

使用种子 3：

- 2 个耗散参数组合；
- 4 个时间点；
- 20 epochs；
- QSNN 和 VQC 各运行一次。

检查 NaN、梯度、显存、输出质量、CSV 增量保存和检查点恢复。

### 阶段 D：服务器性能基准

分别测试 batch size 16/32、`complex64/complex128` 和 CPU/A100。记录 20 epochs 的墙钟时间，再决定正式批大小和数值类型。

不能假设四张 A100 会自动加速一个小矩阵进程；正式方案采用不同种子跨 GPU 并行。

### 阶段 E：超参数校准

校准种子：

\[
10,11,12.
\]

两个模型使用相同数量的候选。建议公共候选为：

- \(\mathrm{lr}_D\in\{0.001,0.002\}\)；
- generator steps \(\in\{5,10\}\)。

QSNN 输出目标质量固定为 0.90，不在正式校准网格中额外增加只属于 QSNN 的搜索预算。根据联合 OOD 验证迹距离选每类前两个候选。

### 阶段 F：冻结验证

验证种子：

\[
13,14,15,16,17.
\]

只运行校准阶段选出的每类前两个候选，然后各冻结一个最终配置。冻结后不得根据正式种子结果调整学习率、层数、损失权重或检查点规则。

### 阶段 G：正式盲测

正式种子：

\[
20,21,\ldots,39.
\]

每个种子对 QSNN/VQC 使用完全相同的生成器初态和数据顺序。检查点只根据独立验证集选择。

### 阶段 H：控制实验

控制种子：

\[
40,41,42,43,44.
\]

运行标签打乱、时间反转、随机拼接、快照判别和闭系统控制。控制实验不能与正式主结果混合统计。

---

## 12. 正式统计与优越性标准

主指标使用配对种子比较。报告至少给出：

- 均值、标准差和 95% 置信区间；
- 每个种子的配对散点；
- 配对置换或符号翻转检验；
- 配对效应量；
- QSNN 获胜种子比例。

只有同时满足以下条件，才声称 QSNN-GAN 在该任务上具有生成质量优势：

1. 联合 OOD 平均迹距离的配对检验 \(p<0.05\)；
2. 相对改善至少 10%；
3. QSNN 至少在 65% 的正式种子中获胜；
4. 最差迹距离、动力学残差和稳态误差方向一致；
5. 生成器容量预检通过；
6. 参数量差异不超过 5%；
7. 输出泄漏不能解释优势；
8. 时间反转或标签打乱后性能显著恶化；
9. 优势不能只出现在某一个数值后端。

如果只满足运行时间更短，应表述为“当前实现下的计算效率优势”，不能表述为生成质量优势。

---

## 13. 四张 A100 的服务器运行方案

### 13.1 并行原则

每个训练种子彼此独立，采用一张 GPU 运行一个进程。不要让四个进程同时写同一个 `final.csv` 或 `summary.json`。

每张卡必须使用独立分片目录，例如：

```text
outputs/qgan/open_system_trajectory/formal/shard0
outputs/qgan/open_system_trajectory/formal/shard1
outputs/qgan/open_system_trajectory/formal/shard2
outputs/qgan/open_system_trajectory/formal/shard3
```

正式种子建议分配为：

| GPU | seeds |
|---|---|
| GPU 0 | 20–24 |
| GPU 1 | 25–29 |
| GPU 2 | 30–34 |
| GPU 3 | 35–39 |

### 13.2 预期命令接口

实际脚本实现后，应支持类似命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_open_system_trajectory_qgan.py \
  --stage formal --seeds 20 21 22 23 24 \
  --output-dir outputs/qgan/open_system_trajectory/formal/shard0
```

其他 GPU 使用不同种子和分片目录。建议通过 `nohup`、`tmux` 或服务器调度系统运行，并分别保存 stdout/stderr。

全部分片结束后执行只读合并：

```bash
python scripts/run_open_system_trajectory_qgan.py \
  --stage merge-formal \
  --input-dirs outputs/qgan/open_system_trajectory/formal/shard0 \
               outputs/qgan/open_system_trajectory/formal/shard1 \
               outputs/qgan/open_system_trajectory/formal/shard2 \
               outputs/qgan/open_system_trajectory/formal/shard3
```

合并程序必须检查：

- 每个正式种子恰好出现一次；
- QSNN 和 VQC 种子完全配对；
- 配置哈希一致；
- 没有重复或缺失检查点；
- 所有指标为有限数；
- 所有分片均无错误日志。

### 13.3 服务器监控

至少记录：

- GPU 型号和 PyTorch/CUDA 版本；
- 每个进程的 GPU 编号；
- 峰值显存；
- 每个种子的开始/结束时间；
- 当前完成种子；
- stderr；
- 断点恢复状态。

脚本必须在每个种子完成后立即原子保存 CSV 行、检查点和阶段状态，不能等全部种子完成后一次写盘。

---

## 14. 预期代码与配置文件

建议新增：

```text
qgan/open_systems.py
qgan/trajectory_pairs.py
qgan/trajectory_metrics.py
configs/open_system_trajectory.yaml
scripts/run_open_system_trajectory_qgan.py
scripts/merge_open_system_trajectory_shards.py
tests/test_open_system_trajectory_qgan.py
docs/reports/OPEN_SYSTEM_TRAJECTORY_QGAN_REPORT.md
```

建议职责：

- `open_systems.py`：Hamiltonian、跳跃算符、Liouvillian、真值轨迹和稳态；
- `trajectory_pairs.py`：时钟编码、时间反转、随机拼接和 batch 构造；
- `trajectory_metrics.py`：动力学残差、半群误差、稳态误差和轨迹观测量；
- 主脚本：分阶段运行、断点恢复、分片输出、合并和报告生成。

不得覆盖现有 Werner、纠缠见证和 MNIST 实验脚本。

---

## 15. 报告与图表要求

最终报告至少包含两个四联图，共八个核心曲线面板。

训练四联图：

1. 联合 OOD 迹距离随 epoch；
2. 验证轨迹误差随 epoch；
3. 生成器梯度范数；
4. 判别器输出质量或读出归一化。

轨迹四联图：

5. 状态保真度随时间；
6. 状态迹距离随时间；
7. 激发态布居、纯度或相干项随时间；
8. 动力学残差与稳态误差随时间。

额外建议图：

- 20 个正式种子的配对散点图；
- 不同 \(\gamma,\gamma_\phi\) 下的误差热图；
- 正向轨迹与时间反转轨迹的判别分数；
- 时间插值与时间外推分区箱线图；
- `cheby_suzuki` 与 `suzuki_global` 数值误差图。

报告必须同时解释正结果和失败模式，尤其要指出生成轨迹是否存在纯度、纠缠或稳态偏差。

---

## 16. 预计耗时与停止条件

顺利情况下：

- 本地实现、测试和容量预检：8–14 小时；
- 四张 A100 校准和冻结验证：5–10 小时；
- 四张 A100 正式种子和控制实验：6–12 小时；
- 汇总、图表和报告：1–2 小时。

整体预计 18–30 小时；若容量预检失败并需要增强生成器，可能延长到 2–3 天。

必须停止正式实验并先修复的条件：

- 容量预检未通过；
- 真值轨迹物理性失败；
- QSNN/VQC 参数差超过 5%；
- 输出层质量长期低于预设门槛；
- 梯度出现 NaN/Inf；
- 两种数值后端误差不可忽略；
- 多 GPU 分片配置哈希不一致；
- 正式测试条件意外参与训练或模型选择。

---

## 17. 最终验收清单

- [ ] 真实 Lindblad 轨迹通过全部物理性测试；
- [ ] \(\gamma_\phi\) 采用统一的 \(\sqrt{\gamma_\phi/2}Z\) 约定；
- [ ] 轨迹对时钟编码和时间反转测试通过；
- [ ] 公共生成器容量预检通过；
- [ ] QSNN/VQC 参数量差不超过 5%；
- [ ] 两个模型使用配对初始生成器和数据顺序；
- [ ] 校准、验证和正式种子严格隔离；
- [ ] `cheby_suzuki` 与 `suzuki_global` 均完成数值验证；
- [ ] 20 个正式种子完整配对；
- [ ] 控制实验与正式实验分开统计；
- [ ] 四卡分片无重复、无缺失、配置哈希一致；
- [ ] 八个核心曲线面板和完整 Markdown 报告生成；
- [ ] 结论严格按照预注册判据表述。

---

## 18. 结果解释边界

本实验即使得到 QSNN 优势，也只能支持以下有限结论：

> 在当前两量子比特驱动—耗散系统、当前轨迹对编码、当前参数匹配架构和理想模拟条件下，QSNN 判别器为生成器提供了更有效的开放系统轨迹约束。

它不能直接证明 QSNN 在所有开放系统、所有噪声模型或真实量子硬件上普遍优于 VQC。若主指标不显著，但动力学残差、稳态误差或运行时间更好，应分别报告为动力学一致性趋势或计算效率优势，不能替代主生成质量结论。
