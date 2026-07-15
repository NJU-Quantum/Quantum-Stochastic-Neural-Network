# QSNN-QGAN 实际代码编写与训练测试任务书

> 用途：将本文件直接交给 Codex，要求其在现有 QSNN 项目中完成代码审查、增量实现、训练、baseline 对比、消融实验和结果归档。
>
> 核心原则：先复用并验证现有 QSNN 演化代码，再接入 QGAN；不要重写已经工作的高维 QSNN 后端，不要在高维情况下显式构造完整 Liouvillian。

## 0. Codex 开始工作前必须执行的检查

1. 阅读项目根目录及子目录中的 `AGENTS.md`、`README`、环境文件和测试说明。
2. 执行 `git status --short`，保留用户已有修改，不覆盖无关文件。
3. 使用 `rg` 搜索并确认现有接口，不要仅根据本任务书猜测文件位置：
   - `QSNNFunction`
   - `QSNN2D`
   - `cheby_suzuki`
   - `suzuki_global`
   - `krylov`
   - `expm`
   - `rk4`
   - `evolve_expm`
   - `evolve_vec_rk4`
   - `evolve_from_operators`
   - `evolve_unitary`
   - `_lindblad_rhs_qsnn2d_structured`
   - `evolve_qsnn2d_stage2_structured`
4. 已知历史代码中曾出现 `qsw.py`、`benchmarking/benchmark.py`、`max_dim_test.py` 等名称；只有在仓库中实际找到时才复用，不要为了匹配名称重复创建文件。
5. 先运行现有测试和最小 benchmark，记录修改前的结果；如果现有测试本身失败，先说明失败是否与本任务无关。
6. 确认当前 Python 环境和 GPU：

   ```text
   Conda environment: qsnn
   Python: 3.12
   PyTorch: 2.12.1+cu132
   Torchvision: 0.27.1+cu132
   GPU: NVIDIA GeForce RTX 5060, 8 GB
   CUDA available: True
   Compute capability: (12, 0)
   ```

7. 所有实现必须同时支持 CPU smoke test 和 CUDA 正式训练。默认数值类型建议为 `complex64`；小维度梯度校验允许使用 `complex128`。

---

## 1. 项目目标

实现一个以量子随机行走神经网络 QSNN（Quantum Stochastic Neural Network）作为判别器的量子生成对抗网络 QGAN。

- 生成器：参数化量子生成器 $G(\boldsymbol\theta_G,z)$，产生生成态 $\rho_G$。
- 判别器：开放系统量子通道 QSNN，参数包含 Hamiltonian 相干耦合与 Lindblad 耗散参数。
- 判别输出：`real` 与 `fake` 两个输出节点/输出态。
- 主损失：采用 Dallaire-Demers 与 Killoran 型线性 Trace-$Z$ 对抗目标。
- 主要科学问题：在保持生成器、数据、损失和训练设置一致时，QSNN 的耗散判别器是否比普通幺正量子判别器具有更好的收敛性、稳定性或噪声鲁棒性。
- 主要数据：MNIST，先完成无条件单数字生成，再完成数字 0/1 条件生成。
- 主要数据规模：依次完成 $8\times8=64$、$16\times16=256$，并把未经降维的 $28\times28=784$ 维原始像素实验列为正式必做阶段；Autoencoder 潜空间作为扩展实验，而不是替代完整像素实验。

本项目不是“用 QSNN 识别数字类别”。在 QGAN 中，QSNN 的标签是 `real/fake`；数字标签 $\lambda$ 只用于条件生成。

---

## 2. 数学定义（代码必须与此一致）

### 2.1 QSNN 判别器

QSNN 判别器是 GKLS/Lindblad 开放系统通道：

\[
\frac{d\rho}{dt}
=-i[H_D,\rho]
+\sum_k\left(
L_k\rho L_k^\dagger
-\frac12\{L_k^\dagger L_k,\rho\}
\right).
\]

记最终演化通道为

\[
\mathcal E_D(\rho;\boldsymbol\theta_D),
\qquad
\boldsymbol\theta_D=(\mathbf h,\boldsymbol\gamma).
\]

真实态和生成态经过判别器后分别为

\[
\rho_{DR}^{s}=\mathcal E_D(\rho_R^s;\boldsymbol\theta_D),
\]

\[
\rho_{DG}^{s}
=\mathcal E_D\!\left(
\rho_G^s(\boldsymbol\theta_G,z_s);
\boldsymbol\theta_D
\right).
\]

Hamiltonian 必须由代码显式保证 Hermitian。Lindblad 参数必须满足物理约束，建议对无约束参数使用 `softplus` 或平方映射得到非负量。

### 2.2 必须审计耗散参数的含义

现有 QSNN 文献/报告中写法可能为

\[
L_{ij}=\gamma_{ij}|i\rangle\langle j|.
\]

此时 dissipator 中的有效跃迁率与 $\gamma_{ij}^2$ 成正比。如果代码希望把可训练量直接解释为物理跃迁率 $\Gamma_{ij}$，应写为

\[
L_{ij}=\sqrt{\Gamma_{ij}}|i\rangle\langle j|.
\]

Codex 必须先确认现有仓库使用的是“跳跃振幅”还是“耗散率”，不得静默改变旧模型语义。应在配置、docstring 和实验元数据中明确记录该约定。

### 2.3 输出测量

定义

\[
\Pi_{\rm real}=|{\rm real}\rangle\langle{\rm real}|,
\qquad
\Pi_{\rm fake}=|{\rm fake}\rangle\langle{\rm fake}|,
\]

\[
Z=\Pi_{\rm real}-\Pi_{\rm fake}.
\]

输出概率直接计算为

\[
p_{\rm real}=\operatorname{Tr}(\Pi_{\rm real}\rho_{\rm out}),
\qquad
p_{\rm fake}=\operatorname{Tr}(\Pi_{\rm fake}\rho_{\rm out}).
\]

同时记录输出层总概率质量

\[
m_{\rm out}=p_{\rm real}+p_{\rm fake},
\qquad
p_{\rm leak}=1-m_{\rm out}.
\]

只有当 `real/fake` 构成完备二元输出（或 $p_{\rm leak}\approx0$）时，才能使用

\[
p_{\rm real}=\frac{1+\operatorname{Tr}(Z\rho)}{2},
\qquad
p_{\rm fake}=\frac{1-\operatorname{Tr}(Z\rho)}{2}.
\]

因此代码必须：

1. 把输出节点构造成吸收节点/概率汇聚终点；
2. hidden-to-output 只保留单向 Lindblad 跳跃；
3. 去掉或关闭 hidden-output Hamiltonian coupling；
4. 每个 batch 记录 `output_mass` 与 `leakage`；
5. 不允许在未检查完备性的情况下把上述二元概率公式写死。

建议同时实现两个可配置目标：

- `trace_z`：论文主目标；
- `direct_success`：直接用投影概率计算判别成功率，用作有限演化时间下的交叉验证。

### 2.4 主对抗目标

主目标函数：

\[
V(\boldsymbol\theta_D,\boldsymbol\theta_G)
=\frac12+
\frac{1}{4M}\sum_{s=1}^{M}
\operatorname{Tr}\left[
Z\left(\rho_{DR}^{s}-\rho_{DG}^{s}\right)
\right].
\]

优化形式：

\[
\min_{\boldsymbol\theta_G}
\max_{\boldsymbol\theta_D}V.
\]

在使用 PyTorch optimizer 时实现为：

- 判别器最小化 `loss_D = -V`；
- 生成器最小化

\[
\mathcal L_G
=-\frac1M\sum_{s=1}^{M}
\operatorname{Tr}(Z\rho_{DG}^{s}).
\]

最小化 $\mathcal L_G$ 会推动生成态被判为 `real`。训练代码必须清楚区分“数学上最大化 $V$”与“optimizer 最小化 `-V`”，避免符号写反。

### 2.5 直接成功率目标（验证用）

当输出存在有限泄漏时，直接判别成功率为

\[
V_{\rm success}
=\frac{1}{2M}\sum_s
\left[
\operatorname{Tr}(\Pi_{\rm real}\rho_{DR}^s)
+\operatorname{Tr}(\Pi_{\rm fake}\rho_{DG}^s)
\right].
\]

当 $m_{\rm out}=1$ 时，它与 Trace-$Z$ 目标等价。测试中必须验证二者差异随演化时间增大/泄漏减小而趋近于零。

---

## 3. 高维实现约束

现有 QSNN 已经通过结构化 Lindblad 演化和 `cheby_suzuki` 后端实现几百维训练：

- Chebyshev：处理相干 Hamiltonian 演化；
- Suzuki：处理结构化耗散演化；
- 不显式构造 $N^2\times N^2$ Liouvillian；
- 主要 GPU 设备为 CUDA。

必须遵守：

1. $N\ge64$ 时默认使用已有 `cheby_suzuki` 路径。
2. `expm` 仅用于很小维度参考计算和数值正确性对照。
3. `rk4`、`krylov`、`suzuki_global` 保留为可选 benchmark 后端，不作为大规模正式训练默认值。
4. 不得为方便而把高维 QSNN 改回显式 Liouvillian 指数。
5. 复用现有结构化 RHS、稀疏跳跃边、分层连接与自定义 autograd 逻辑。
6. 训练前后都要运行原有 backend benchmark，避免 QGAN 接入破坏 QSNN 性能。

历史代表性小规模 benchmark（仅作回归背景，不作为新硬件硬性阈值）：

| 模型 | 后端 | avg_total (s) | peak memory (MB) |
|---|---|---:|---:|
| QSNNFunction | cheby_suzuki | 0.008467 | 16.38 |
| QSNNFunction | expm | 0.010863 | 103.77 |
| QSNNFunction | rk4 | 0.123630 | 17.67 |
| QSNNFunction | krylov | 0.592875 | 26.82 |
| QSNNFunction | suzuki_global | 11.441674 | 57.59 |
| QSNN2D | cheby_suzuki | 0.043245 | 16.68 |

新 RTX 5060 上应重新 benchmark，不要直接把历史数值当作验收结果。

---

## 4. 数据方案 A：直接像素空间（主实验，包含完整 784 维）

### 4.1 数据集与任务阶段

使用 Torchvision MNIST。

按以下阶段实施：

1. `smoke`：合成小数据或 MNIST 极小子集；
2. `unconditional_0`：只使用数字 0，无条件生成；
3. `conditional_01`：数字 0/1 条件生成；
4. 可选扩展：更相似的数字对，例如 3/8；
5. 不要第一版直接训练完整十类条件模型。

数据切分必须固定随机种子。PCA、Autoencoder 或其他预处理只能在训练集拟合，不能泄漏验证/测试数据。

建议配置化样本数：

- smoke：每类 32–64；
- debug：每类训练 256、验证 64；
- main：每类至少训练 1000，并保留固定验证/测试集；
- final：根据实际训练时间扩大至完整筛选后的训练集。

### 4.2 分辨率设置

低维调试与中等规模主实验不使用 PCA，而是把 MNIST 从 $28\times28$ 使用面积下采样（area downsampling）变为

\[
16\times16=256.
\]

下采样保持二维局部结构，而不是把像素变成全局 PCA 主成分。预处理输出应缓存并带上版本、参数、随机种子和数据 split 信息。

必须提供可配置的三档尺度实验：

- $8\times8=64$：调试和数值校验；
- $16\times16=256$：中等规模主实验；
- $28\times28=784$：未经 PCA、Autoencoder、裁剪或缩放的完整像素主实验。

完整 784 维实验不是可选项。实施顺序仍然应为 $64\rightarrow256\rightarrow784/1024$，目的是先验证正确性再扩大规模，而不是取消高维实验。

### 4.3 概率幅度编码

对非负灰度图像展平为 $x_j\ge0$，定义

\[
p_j=\frac{x_j+\epsilon}{\sum_k(x_k+\epsilon)},
\]

\[
|\psi_x\rangle
=\sum_{j=1}^{N_{\rm in}}\sqrt{p_j}|j\rangle,
\qquad
\rho_x=|\psi_x\rangle\langle\psi_x|.
\]

要求：

- 检查 `sum(p)==1`；
- 检查态范数、密度矩阵 trace、Hermiticity 与 PSD；
- 提供从对角概率恢复图像的函数；
- 保存原图、处理后图像、量子编码后恢复图三联图，确认编码无误；对于 784 维实验，处理后图像必须与原始 $28\times28$ 图像逐像素一致。

### 4.4 条件标签

第一版无条件模型不需要标签态。

条件 0/1 模型可使用

\[
\rho_{\lambda,x}
=|\lambda\rangle\langle\lambda|\otimes\rho_x,
\qquad \lambda\in\{0,1\}.
\]

当图像维度为 256 时，总条件输入维度为 512。必须先在 256 维无条件模型上验证完整训练，再开启 512 维条件实验。

如果现有 QSNN 更适合 direct-sum 分层输入而不是 tensor-product 标签，Codex 应提出并实现明确的 block embedding，同时保证标准 QGAN 与 QSNN-QGAN 使用完全相同的信息，不能让某个模型额外看到标签。

### 4.5 未经降维的完整 $28\times28$ 实验

对每张原始 MNIST 图像直接展平：

\[
\mathbf x=(x_1,\ldots,x_{784}),
\qquad x_j\ge0.
\]

不进行 PCA、Autoencoder、空间缩放或特征选择，直接定义

\[
p_j=\frac{x_j+\epsilon}{\sum_{k=1}^{784}(x_k+\epsilon)},
\]

\[
|\psi_x^{784}\rangle
=\sum_{j=1}^{784}\sqrt{p_j}|j\rangle,
\qquad
\rho_x^{784}=|\psi_x^{784}\rangle\langle\psi_x^{784}|.
\]

必须实现两条完整分辨率路径：

#### 路径 A：QSNN 原生 784 维

如果现有 QSNN 和生成器支持任意 Hilbert 维数，直接使用 $N_{\rm in}=784$。该实验用于测量 QSNN 在非二次幂维度上的真实时间、显存和收敛能力。

#### 路径 B：无损补零到 1024 维

如果门模型生成器或标准 VQC baseline 必须使用 qubit Hilbert 空间，则使用 10 qubit：

\[
1024=2^{10}.
\]

把 784 个像素幅度嵌入前 784 个基态，其余 240 个基态严格补零：

\[
|\psi_x^{1024}\rangle
=\sum_{j=1}^{784}\sqrt{p_j}|j\rangle
+\sum_{j=785}^{1024}0\,|j\rangle.
\]

这不是降维，也没有丢失像素信息，只是把 $\mathbb C^{784}$ 无损嵌入 $\mathbb C^{1024}$。

定义有效像素子空间投影

\[
P_{\rm valid}=\sum_{j=1}^{784}|j\rangle\langle j|,
\]

并对生成态记录无效填充空间概率

\[
p_{\rm pad}
=1-\operatorname{Tr}(P_{\rm valid}\rho_G).
\]

要求：

1. 真实数据的 `padding_mass` 必须为零（数值容差内）；
2. 生成器的 `padding_mass` 必须逐 batch 记录，不得在重构图像时静默丢弃；
3. 可以实现可配置的 padding penalty，或设计保持有效子空间的生成器 ansatz；
4. 标准 QGAN 与 QSNN-QGAN 必须采用相同的 1024 维嵌入、mask 和 penalty；
5. 图像重构使用前 784 个概率，只有在报告 `padding_mass` 后才允许归一化并 reshape 为 $28\times28$；
6. 正式报告必须分别给出原生 784 维和 1024 维无损嵌入的资源结果。

未经降维的条件 0/1 实验会达到原生 1568 维或 11-qubit/2048 维，可作为完成无条件完整分辨率实验后的扩展，不得阻塞 784/1024 维无条件实验。

---

## 5. 数据方案 B：Autoencoder 潜空间（扩展实验）

Autoencoder 可能提高重构和生成图像质量，但会引入可训练经典前端，因此不能替代直接下采样主实验。

### 5.1 不能直接使用普通实数潜变量

普通潜变量 $z_i\in\mathbb R$ 幅度编码后，测量概率只保留 $z_i^2$，会丢失符号和整体模长。禁止在模拟器中直接读取不可测的 statevector amplitude 后送入 Decoder，并把它宣称为可测量方案。

### 5.2 概率单纯形潜空间

使用卷积 Autoencoder，Encoder 最后一层为 Softmax：

\[
p_i=\frac{e^{a_i}}{\sum_j e^{a_j}},
\qquad p_i\ge0,
\qquad \sum_i p_i=1.
\]

量子编码：

\[
|\psi_p\rangle=\sum_i\sqrt{p_i}|i\rangle.
\]

生成态读出：

\[
p_i^G=\operatorname{Tr}(|i\rangle\langle i|\rho_G)=(\rho_G)_{ii}.
\]

重构：

\[
\hat x=D_\omega(\mathbf p^G).
\]

推荐潜空间维度：

- 64：调试；
- 128：Autoencoder 主实验；
- 256：高维扩展。

### 5.3 训练顺序

1. 仅在训练集上训练 Autoencoder；
2. 保存 best checkpoint；
3. 记录 train/validation reconstruction loss 和重构图；
4. 冻结 Encoder 与 Decoder；
5. 所有 GAN/QGAN baseline 共用完全相同的冻结 Encoder/Decoder；
6. 对抗训练中不得更新 Autoencoder 参数；
7. 同时报告 latent-space 指标和 image-space 指标；
8. 报告 Autoencoder 自身重构误差，作为最终图像质量的上限/瓶颈背景。

---

## 6. 生成器实现

生成器必须对 $\boldsymbol\theta_G$ 可微并输出合法密度矩阵。

第一版建议使用参数化纯态量子生成器：

\[
|\psi_G(\boldsymbol\theta_G,z)\rangle
=U_G(\boldsymbol\theta_G,z)|0\cdots0\rangle,
\]

\[
\rho_G=|\psi_G\rangle\langle\psi_G|.
\]

对于 256 维数据使用 8 qubit；对于 64 维数据使用 6 qubit。基础 ansatz 可包含：

1. 噪声角度编码层；
2. 可训练单比特 `RY/RZ`；
3. ring entanglement；
4. 多层重复；
5. 输出态归一化检查。

但是 Codex 必须先检查仓库是否已有量子生成器/PQC 工具。如果已有，优先复用；如果没有，再以最少依赖实现。不要未经确认同时引入 PennyLane、Qiskit 和自定义模拟器三套栈。

生成器接口建议统一为：

```python
rho_g = generator(noise, labels=None)
```

输出形状、batch 维、dtype、device 必须有明确契约。生成器不能通过 `.detach()`、NumPy 转换或不可微测量切断到 $\mathcal L_G$ 的梯度。

如果条件标签作为量子子系统输入，生成器也必须输出与真实条件态相同维度和相同子系统顺序的状态。

---

## 7. 判别器与 baseline

### 7.1 提议模型：QSNN-QGAN

- 生成器：参数化量子生成器；
- 判别器：现有 QSNN；
- 默认后端：`cheby_suzuki`；
- 损失：Trace-$Z$；
- 输出：`real/fake`；
- 训练参数：Hamiltonian $\mathbf h$ 与 Lindblad $\boldsymbol\gamma$。

### 7.2 核心 baseline：标准幺正 QGAN

使用与提议模型完全相同的：

- 数据；
- 编码；
- 生成器结构与初始化策略；
- Trace-$Z$ 损失；
- optimizer、batch、epoch、随机种子；
- 输出测量定义。

只把判别器改为幺正 VQC：

\[
\mathcal E_D^{\rm VQC}(\rho)
=U_D(\boldsymbol\theta_D)\rho U_D^\dagger(\boldsymbol\theta_D).
\]

这是最重要的 baseline，因为它只改变判别器动力学。

尽量匹配两类判别器的可训练参数量；若无法完全匹配，至少同时报告参数量匹配版本和表达能力匹配版本。

### 7.3 经典 GAN（辅助 baseline）

- 数据：与量子模型相同的 256 维归一化像素向量，或相同冻结 Autoencoder 的潜变量；
- 生成器/判别器：小型 MLP；
- 损失：标准 BCE/non-saturating GAN；
- 参数量：报告并提供近似匹配设置。

经典 GAN 用于任务级性能参照，但不能单独证明 QSNN 判别器的贡献。

### 7.4 必做消融实验

1. `qsnn_full`：$H+L$；
2. `qsnn_h_only`：关闭 Lindblad 耗散；
3. `qsnn_l_only`：关闭 Hamiltonian 相干项；
4. 可选：不同演化时间 $T$；
5. 可选：不同 hidden 节点数；
6. 可选：不同耗散连接稀疏度。

### 7.5 可选补充 baseline

可实现 fidelity/SWAP-test QuGAN，但必须标记为“同时改变损失和判别机制的补充 baseline”，不能替代标准幺正 Trace-$Z$ QGAN。

---

## 8. 训练循环

### 8.1 交替训练

每个 iteration：

1. 固定生成器，更新判别器 `n_d_steps` 次；
2. 固定判别器，更新生成器 1 次；
3. 记录损失、梯度范数、输出质量和资源指标。

判别器阶段：

```python
loss_d = -value_v
loss_d.backward()
optimizer_d.step()
```

生成器阶段：

```python
loss_g = -mean(z_expectation_on_fake_outputs)
loss_g.backward()
optimizer_g.step()
```

冻结某一方时应设置 `requires_grad_(False)`，但不能破坏生成器梯度穿过冻结判别器通道回传到生成器输入态。

### 8.2 数值稳定性

必须提供可配置项：

- `lr_g`、`lr_d`；
- `n_d_steps`；
- batch size；
- gradient clipping；
- evolution time $T$；
- Chebyshev 阶数/容差；
- Suzuki steps/order；
- dtype；
- seed；
- backend；
- hidden/output topology。

记录并检测：

- NaN/Inf；
- density trace drift；
- Hermiticity drift；
- 最小特征值（小维度或抽样检查）；
- $\langle Z\rangle\notin[-1,1]$；
- 梯度为零或爆炸；
- `output_mass` 太低；
- 显存持续增长。

混合精度对 complex tensor 的支持有限，默认不要开启 AMP；只有经过单独数值验证后才能作为可选优化。

### 8.3 Checkpoint 与恢复

保存：

- 生成器状态；
- 判别器状态；
- 两个 optimizer；
- scheduler（如有）；
- epoch/step；
- RNG 状态；
- 完整配置；
- 数据预处理版本；
- Git commit；
- PyTorch/CUDA/GPU 信息。

训练必须支持从 checkpoint 恢复，并验证恢复前后下一步输出一致。

---

## 9. 指标与结果文件

### 9.1 对抗训练指标

每个 epoch 至少记录：

- `V_trace`；
- `V_direct_success`；
- `loss_D`；
- `loss_G`；
- 判别器 real/fake accuracy；
- $\langle Z\rangle_{\rm real}$；
- $\langle Z\rangle_{\rm fake}$；
- `output_mass_real/fake`；
- `leakage_real/fake`；
- `valid_subspace_mass` 与 `padding_mass`（1024 维完整分辨率实验）；
- generator/discriminator gradient norm。

当生成分布逼近真实分布时，理论上应观察到 $V\to1/2$、判别准确率趋近 50%，但不能仅凭这两个量判断生成成功，因为判别器过弱也会产生相同现象。

### 9.2 量子态指标

至少实现：

- 平均真实态与平均生成态的 fidelity；
- trace distance；
- 对角测量分布的 Hellinger distance；
- 对角测量分布的 total variation distance；
- purity（用于检查生成混合程度）；
- 条件模型按标签分别统计。

### 9.3 图像与分布指标

至少实现：

- 生成样本网格；
- 真实/生成平均图；
- 每像素均值和方差差异；
- MMD 或 Wasserstein 类分布距离；
- 最近邻检查，排查训练样本记忆；
- 多样性/模式坍塌指标；
- Autoencoder 实验中的重构误差和 latent/image 两层指标。

PCA-4 或极低分辨率实验中不要把 FID 当作唯一主指标。若后续使用 FID/KID，必须说明特征提取器、样本数和低分辨率处理方式。

### 9.4 性能指标

每个模型记录：

- 参数量；
- 每 epoch wall-clock time；
- forward/backward time；
- 峰值 GPU 显存；
- CPU 内存（可选）；
- 每个有效样本的训练成本；
- 不同维度 $N=64,256,512$ 的缩放趋势。

结果保存为机器可读的 CSV/JSON，同时生成 PNG/PDF 曲线。目录中必须包含本次运行的 config 副本。

---

## 10. 测试要求

### 10.1 单元测试

1. 图像下采样形状和范围正确；
2. 概率向量非负且和为 1；
3. 量子态范数为 1；
4. 密度矩阵 Hermitian、trace 1、PSD（容差内）；
5. generator 输出合法且对参数有非零梯度；
6. QSNN 输出 trace-preserving；
7. $Z=\Pi_{\rm real}-\Pi_{\rm fake}$ 构造正确；
8. $\operatorname{Tr}(Z\rho)\in[-1,1]$；
9. `trace_z` 与 `direct_success` 在完备输出时等价；
10. 有泄漏时二者差值符合推导；
11. `gamma` 参数映射后非负；
12. $H$ 始终 Hermitian；
13. 小维度 `cheby_suzuki` 与 `expm` 输出在设定容差内一致；
14. 小维度 autograd 与中心有限差分梯度一致；
15. batch 与单样本结果一致；
16. CPU 与 CUDA 小规模结果在容差内一致；
17. 784 维编码后逐像素重构与原图归一化结果一致；
18. 784 到 1024 的补零嵌入不改变前 784 个概率；
19. 真实 1024 维状态的 `padding_mass` 为零；
20. 生成态的 `padding_mass` 计算和梯度正确。

### 10.2 集成测试

1. 合成的完全可分 real/fake 状态：判别器能够快速过拟合；
2. tiny MNIST batch：完整 D/G 训练可运行至少若干 step；
3. 生成器更新后 `loss_G` 有下降趋势；
4. 判别器更新后 `V` 有上升趋势；
5. checkpoint 保存/恢复一致；
6. 64 维 smoke test 在 CPU 与 GPU 均通过；
7. 256 维一轮训练不 OOM；
8. 784 维原生 QSNN tiny subset 完成 forward/backward；
9. 1024 维无损嵌入 tiny subset 完成 VQC 与 QSNN smoke test；
10. 不显式分配 $N^2\times N^2$ 大矩阵；
11. 原有 QSNN tests/benchmarks 无回归。

### 10.3 物理与训练 sanity check

- 若真实态与生成态完全相同，理论上 $V=1/2$；
- 若判别器完美区分且输出完备，理论上 $V\to1$；
- `real` 输入的 $\langle Z\rangle$ 应趋向 $+1$；
- `fake` 输入在判别器训练时应趋向 $-1$；
- `fake` 输入在生成器训练时应被推动向 $+1$；
- 关闭耗散后，输出汇聚和泄漏行为应有可解释变化。

---

## 11. 推荐实验矩阵

### 阶段 A：正确性验证

| 数据 | 维度 | 模型 | 目的 |
|---|---:|---|---|
| 合成量子态 | 小维度 | QSNN 判别器 | 梯度、损失符号、输出投影 |
| MNIST 0 tiny subset | 64 | QSNN-QGAN | 端到端 smoke test |
| MNIST 0 tiny subset | 64 | 标准 QGAN | 核心 baseline smoke test |

### 阶段 B：无条件主实验

| 数据 | 维度 | 模型 |
|---|---:|---|
| MNIST 0 | 256 | 经典 GAN |
| MNIST 0 | 256 | 标准幺正 QGAN |
| MNIST 0 | 256 | QSNN-QGAN full |
| MNIST 0 | 256 | QSNN $H$-only |
| MNIST 0 | 256 | QSNN $L$-only |

每个配置至少运行 5 个随机种子，报告均值与标准差，而不是只展示最好的一次。

### 阶段 C：条件生成

| 数据 | 维度 | 模型 |
|---|---:|---|
| MNIST 0/1 + label | 512 | 标准幺正 QGAN |
| MNIST 0/1 + label | 512 | QSNN-QGAN full |
| MNIST 0/1 + label | 512 | 必要消融 |

### 阶段 D：Autoencoder 扩展

固定同一 Autoencoder，测试潜空间 $d=64,128,256$，重复经典 GAN、标准 QGAN、QSNN-QGAN 对比。

### 阶段 E：未经降维的完整分辨率实验（必做）

| 数据 | 表示维度 | 模型 | 目的 |
|---|---:|---|---|
| MNIST 0 原始 $28\times28$ | 784 | 经典 GAN | 完整像素经典参照 |
| MNIST 0 原始 $28\times28$ | 原生 784 | QSNN-QGAN full | 验证 QSNN 任意维高维训练 |
| MNIST 0 原始 $28\times28$ | 无损嵌入 1024 | 标准幺正 QGAN | 10-qubit 核心 baseline |
| MNIST 0 原始 $28\times28$ | 无损嵌入 1024 | QSNN-QGAN full | 与 10-qubit VQC 公平比较 |
| MNIST 0 原始 $28\times28$ | 784 或 1024 | 必要消融 | 检查耗散贡献 |

实施要求：

- 先用 tiny subset 完成完整分辨率 smoke test；
- 再用至少 1000 个数字 0 样本正式训练；
- 若完整训练受资源限制，必须保存可复现的最大 batch、最大维度、forward/backward 时间、峰值显存和具体失败位置；
- 不得用 256 维结果替代本阶段；
- 1024 维实验必须报告 `padding_mass`。

### 阶段 F：噪声与规模

- 在相同噪声模型/强度下比较 VQC 与 QSNN 判别器；
- 维度扩展 $64\rightarrow256\rightarrow512\rightarrow784/1024$；
- 记录时间、显存、收敛与最终质量；
- 在完成无条件 784/1024 维后，再评估原生 1568 或 2048 维的完整分辨率条件 0/1 模型。

---

## 12. 建议代码组织（仅在仓库没有既有规范时采用）

优先遵循现有项目结构。如果需要新增模块，可采用：

```text
src/
  qgan/
    data.py                 # MNIST 筛选、下采样、缓存
    encoding.py             # 概率幅度编码、标签 embedding
    generators.py           # 量子生成器
    qsnn_discriminator.py   # 对现有 QSNN 的薄封装，不复制核心演化代码
    vqc_discriminator.py    # 标准幺正 QGAN baseline
    classical_gan.py        # 经典 baseline
    objectives.py           # Trace-Z、direct success、generator loss
    metrics.py
    trainer.py
    checkpoint.py
  autoencoder/
    model.py
    train.py
configs/
  smoke.yaml
  mnist0_64.yaml
  mnist0_256_qsnn.yaml
  mnist0_256_vqc.yaml
  mnist01_512_qsnn.yaml
  mnist0_784_native_qsnn.yaml
  mnist0_1024_qsnn.yaml
  mnist0_1024_vqc.yaml
  autoencoder_128.yaml
scripts/
  train_qgan.py
  train_autoencoder.py
  evaluate_qgan.py
  run_experiment_matrix.py
tests/
  test_encoding.py
  test_objectives.py
  test_generator.py
  test_qsnn_wrapper.py
  test_backend_agreement.py
  test_training_smoke.py
```

QSNN wrapper 应当薄且可测试：只负责 batch、输入 embedding、输出 projector 和统一接口，不应复制 `qsw.py` 或现有 QSNN 后端实现。

---

## 13. 配置与命令行要求

训练代码不得依赖散落在源码中的硬编码超参数。配置至少包含：

```yaml
experiment:
  name: mnist0_256_qsnn
  seed: 0

data:
  dataset: mnist
  digits: [0]
  image_size: 16
  max_train_per_class: 1000
  encoding: probability_amplitude

model:
  generator: pqc
  discriminator: qsnn
  input_dim: 256
  backend: cheby_suzuki
  loss_mode: trace_z
  evolution_time: 1.0

training:
  epochs: 100
  batch_size: 16
  lr_g: 0.001
  lr_d: 0.001
  n_d_steps: 1
  grad_clip: null

runtime:
  device: cuda
  complex_dtype: complex64
```

数值只是初始默认值，必须允许命令行或配置覆盖。最终 README 必须给出：

1. 环境检查命令；
2. 数据准备命令；
3. smoke test 命令；
4. 单模型训练命令；
5. baseline 矩阵命令；
6. 原生 784 维与无损 1024 维完整分辨率命令；
7. 评估和绘图命令；
8. checkpoint 恢复命令。

---

## 14. Codex 的实施顺序

### Milestone 1：审计和设计确认

- 输出仓库结构与现有 QSNN API 摘要；
- 标出可直接复用的函数；
- 确认 `gamma` 语义；
- 确认梯度是否能从 QSNN 输出回传到输入态；
- 确认 batch 支持；
- 确认真实/生成态如何嵌入 QSNN 输入层；
- 不改代码前先提交简短实施计划。

### Milestone 2：数据与目标函数

- 实现 64/256 维 MNIST 预处理；
- 实现量子编码与重构；
- 实现 $\Pi_{\rm real},\Pi_{\rm fake},Z$；
- 实现 `V_trace`、`V_direct_success`、`loss_G`；
- 完成单元测试。

### Milestone 3：QSNN 判别器封装

- 接入现有 `cheby_suzuki`；
- 支持 batch、CUDA、梯度；
- 输出 leakage 指标；
- 小维度对照 `expm`；
- 不破坏旧 benchmark。

### Milestone 4：生成器与端到端 smoke test

- 实现/复用 PQC generator；
- tiny synthetic 和 MNIST 64 维跑通；
- 检查 D/G 更新方向；
- 检查显存泄漏和 checkpoint。

### Milestone 5：baseline 与消融

- 标准 VQC QGAN；
- 经典 GAN；
- $H$-only、$L$-only；
- 统一 config、seed、数据和指标。

### Milestone 6：256 维正式训练

- MNIST 0 无条件；
- 至少 5 个 seed；
- 保存完整结果、曲线、样本和资源统计；
- 根据结果调参，但所有调参记录必须可追踪。

### Milestone 7：512 维条件生成

- MNIST 0/1；
- 明确 label embedding；
- 按标签分别评估；
- 与标准 QGAN 对照。

### Milestone 8：未经降维的完整分辨率实验

- 实现原始 $28\times28=784$ 像素直接编码；
- 实现 QSNN 原生 784 维路径；
- 实现 10-qubit/1024 维无损补零路径；
- 实现 `valid_subspace_mass` 与 `padding_mass`；
- 在相同 1024 维嵌入下比较标准 VQC QGAN 与 QSNN-QGAN；
- 完成 tiny subset smoke test 后运行至少 1000 个数字 0 样本；
- 保存完整的时间、显存、收敛、生成图像与失败边界记录。

### Milestone 9：Autoencoder 扩展

- 概率单纯形 latent；
- 预训练并冻结；
- 共享 AE 的公平 baseline；
- latent/image 双层评估。

---

## 15. 最终交付物

Codex 完成任务时必须交付：

1. 代码修改清单；
2. 新增/修改文件的职责说明；
3. 可复现的 Conda/pip 依赖说明；
4. 单元测试与集成测试结果；
5. 原有 QSNN 回归测试结果；
6. CPU/CUDA smoke test；
7. baseline 与消融配置；
8. 每次正式运行的 config、日志、CSV/JSON；
9. loss/metric/time/memory 曲线；
10. 真实、重构、生成图像网格；
11. checkpoint 与恢复说明；
12. 已知限制和下一步建议；
13. 原生 784 维和无损 1024 维完整分辨率实验报告；
14. 一份简洁 README，确保另一台电脑能够复现。

---

## 16. 验收标准

最低可接受结果：

- [ ] 现有 QSNN tests 和 benchmark 无明显回归；
- [ ] 64 维端到端 QSNN-QGAN 在 CPU/CUDA 均能训练；
- [ ] 256 维 MNIST 0 训练不 OOM；
- [ ] 未经降维的 784 维 MNIST 编码、重构与 tiny subset 训练通过；
- [ ] 1024 维无损补零的标准 QGAN 与 QSNN-QGAN smoke test 通过；
- [ ] 1024 维实验完整记录 `padding_mass`；
- [ ] 至少启动并记录一组不少于 1000 个原始 $28\times28$ 数字 0 样本的正式训练；
- [ ] 判别器单独更新能提高 $V$；
- [ ] 生成器单独更新能降低 $\mathcal L_G$；
- [ ] 密度矩阵物理约束在容差内；
- [ ] 输出 leakage 被记录且可解释；
- [ ] 标准 VQC QGAN baseline 可运行；
- [ ] $H$-only、$L$-only 消融可运行；
- [ ] 固定 seed 可复现主要结果；
- [ ] 至少生成一套完整的对比表、训练曲线和图像样本。

论文级结果还应满足：

- [ ] 至少 5 个随机种子并报告均值/标准差；
- [ ] 参数量和训练预算公平；
- [ ] 量子态、图像质量和资源三类指标齐全；
- [ ] 对 QSNN 优势或无优势都给出诚实结论；
- [ ] 明确区分“QSNN 判别器贡献”与“Autoencoder/编码方式贡献”。
- [ ] 单独报告 64、256、784/1024 维的规模变化，不把完整分辨率结果与降维结果混为一谈。

---

## 17. 参考理论来源

1. Dallaire-Demers, P.-L. and Killoran, N., *Quantum generative adversarial networks*: <https://arxiv.org/abs/1804.08641>
2. Lloyd, S. and Weedbrook, C., *Quantum Generative Adversarial Learning*: <https://doi.org/10.1103/PhysRevLett.121.040502>
3. Stein, S. A. et al., *QuGAN: A Quantum State Fidelity based Generative Adversarial Network*: <https://arxiv.org/abs/2010.09036>
4. 用户报告：`QSNN量子判别器构想(1).pdf`。

---

## 18. 给 Codex 的最后指令

请把本项目视为“在现有高维 QSNN 代码上做增量集成和受控实验”，而不是从零重写一个演示脚本。

第一优先级是科学正确性和可复现性；第二优先级是复用高效后端和避免高维内存爆炸；第三优先级才是扩展数据规模和美化生成图像。

遇到接口、维度、参数语义或损失完备性不明确时，应先通过阅读现有代码和小规模数值实验给出证据，再决定实现；不要静默作出会改变物理模型的假设。
