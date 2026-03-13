# 量子神经网络

## 必看：一定要安装vscode扩展 Markdown Preview Enhanced 、Markdown All in One

**基于“量子随机行走（Quantum Stochastic Walk, QSW）”的量子神经网络模型 QSNN**，并用**梯度下降**去训练。

---

## 1） 模型是什么：把量子态基矢当作“神经元”，在图上做开系统演化

- 网络有 \(N\) 个“神经元”，对应希尔伯特空间的一组基 \(\{|i\rangle\}_{i=0}^{N-1}\)。网络状态用**密度矩阵** \(\rho\) 表示。
- 动力学不是经典前向传播，而是**GKLS/Lindblad 主方程**：
  \[
  \frac{d\rho}{dt}=-i[H,\rho]+\sum_k\left(L_k\rho L_k^\dagger-\frac12\{L_k^\dagger L_k,\rho\}\right)
  \]
  
- 两类可训练的参数：
  1. **相干连接**：由哈密顿量 \(H=\sum_{ij}h_{ij}|i\rangle\langle j|\) 决定，类似双向、可干涉的传播；
  2. **耗散/单向连接**：用 Lindblad 算符 \(L_{ij}=\gamma_{ij}|i\rangle\langle j|\) 表示从 \(j\to i\) 的“不可逆传输”（第2页 Eq.(2)）。

---

### 1.1） QSNN 的演化：

**\(\rho(t)=U\rho(0)U^\dagger\)** 只适用于**封闭系统的纯幺正演化**，对于Lindblad 主方程**：
\[
\frac{d\rho}{dt}=-i[H,\rho]+\sum_k\left(L_k\rho L_k^\dagger-\frac12\{L_k^\dagger L_k,\rho\}\right).
\]

只要 \(\sum_k\cdots\) 这部分不为 0，演化一般变成：
\[
\rho(t)=\mathcal{E}_t(\rho(0)),
\]
其中 \(\mathcal{E}_t\) 是一个**完全正、保迹（CPTP）的线性映射**（量子信道），通常不能写成单一的 \(U\rho U^\dagger\)。

> Lindblad 描述系统跟“环境/测量装置”交换信息，环境被迹掉以后，只剩系统的 \(\rho\)，它的演化就不再是幺正的。

即使开放系统，也总能写成 **Kraus 形式**
\[
\rho(t)=\sum_\alpha K_\alpha \rho(0) K_\alpha^\dagger,\quad \sum_\alpha K_\alpha^\dagger K_\alpha=I,
\]

---

### 1.2) 计算过程：把 \(\rho\) **向量化**

文章 Eq.(5) ：
\[
\frac{d}{dt}\lvert\rho\rangle = \mathcal{L}\lvert\rho\rangle,
\]
这不是说 \(\rho\) 变成纯态，而是**数值/代数技巧**：

把矩阵 \(\rho=\sum_{ij}\rho_{ij}\lvert i\rangle\langle j\rvert\)，变成一个更大空间里的向量
\[
\lvert\rho\rangle=\sum_{ij}\rho_{ij}\lvert i\rangle\lvert j\rangle \in \mathcal{H}\otimes\mathcal{H}.
\]

把密度矩阵的元素按某种顺序“摊平”成向量，方便把“超算符”写成普通矩阵乘法，即 **vectorization（向量化）**，也常和 Choi–Jamiolkowski/李维尔空间（Liouville space）表述联系在一起。

于是原来“对 \(\rho\) 的线性微分方程”就变成“对 \(\lvert\rho\rangle\) 的线性微分方程”，解写成：
\[
\lvert\rho(t)\rangle=e^{\mathcal{L}t}\lvert\rho(0)\rangle,
\]
即 Eq.(8)、(12)、(13) 那种形式。

---

### 1.3) 纯幺正情形

在纯幺正情形（无 Lindblad）：

\[
\rho(t)=U\rho(0)U^\dagger
\]

向量化后有一个常用恒等式（与 Eq.(6) 的结构一致）：
\[
\mathrm{vec}(A\rho B)= (B^{T}\otimes A)\,\mathrm{vec}(\rho).
\]

取 \(A=U,\;B=U^\dagger\)，得到
\[
\lvert\rho(t)\rangle = (U\otimes U^*)\lvert\rho(0)\rangle.
\]

而如果用生成元写：
\[
\frac{d}{dt}\lvert\rho\rangle = -i(H\otimes I - I\otimes H^T)\lvert\rho\rangle,
\]
解就是
\[
\lvert\rho(t)\rangle = e^{-it(H\otimes I - I\otimes H^T)}\lvert\rho(0)\rangle,
\]

---

### 1.4) 计算中最核心的一点：矩阵指数

> **矩阵 \(A\) 很稀疏，不代表 \(e^A\) 也稀疏。**

因为
\[
e^A = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \cdots
\]
而 \(A^2, A^3,\dots\) 往往会迅速产生很多“填充项”（fill-in）。  
所以即使原来的哈密顿量 \(H\) (dim=\(N^2\))是稀疏的，比如最近邻耦合的三对角矩阵，  
\[
U(t)=e^{-itH}
\]
通常也会变得接近稠密。

![变稠密](./sparsity.png)

---

### 先判断一件最重要的事

是要**完整的矩阵** \(e^A\)，还是**它作用在一个向量/态上**即
\[
e^A v
\]

这两种情况差别非常大。

---

### 如果只需要 \(e^A v\)：不要显式算 \(e^A\)

这通常是**最重要的优化**。

在量子演化里，很多时候真正要的是：

- 封闭系统：\(\psi(t)=e^{-itH}\psi(0)\)
- 开放系统：\(\mathrm{vec}(\rho(t))=e^{t\mathcal L}\mathrm{vec}(\rho(0))\)

这时最好的思路一般不是先构造 \(e^A\)，而是直接算它对态的作用。

### 优化方法

- **Krylov / Lanczos**（适合 Hermitian 或接近 Hermitian 的情况）
- 直接在密度矩阵上做 RK4 演化
- 很多库里对应的是 **`expm`** **`expm_multiply`** 之类的接口

它们只需要反复做**稀疏矩阵乘向量**，复杂度通常接近：
\[
O(m \cdot \text{nnz}(A))
\]
（nnz:非零元素个数）而不是去构造一个可能已经变稠密的 \(e^A\)。

---

## 2）经典数据输入编码到初态，再演化，最后测量输出层

总体套路：

1) 把输入编码到初态 \(\rho_\text{in}\)；  
2) 演化一段时间得到 \(\rho_\text{out}\)；  
3) 对输出神经元做投影测量，得到分类概率/回归值；  
4) 定义 loss，对 \(h_{ij},\gamma_{ij}\) 做梯度下降更新（第3页 Eq.(9)-(11)，第4页 Eq.(14)-(15)，Appendix A 给了 \(\partial e^{Lt}/\partial\theta\) 的积分形式）。

---

## 3）论文展示了三类任务（Fig.1 第2页）

### A. 函数逼近（回归）

- 输入是一维 \(x\) 时，用一种“带高阶项”的编码把 \(x^i\) 放进输入层幅度里（第3页 Eq.(7)），相当于人为引入非线性特征。
- 这部分只用相干演化（\(\gamma=0\)）。
- Fig.2（第4页）展示能拟合线性、二次、以及 \((1+\cos6x)/6\) 等函数：函数越复杂，输入层神经元（编码维度）要更多。

### B. 二维点分类

- 结构见 Fig.1(b)（第2页）：输入层多个神经元 + 2 个输出神经元（red/blue）。
- 演化分两段（第3页）：
  1) 先在输入层做一段**unitary**（由 \(H\) 产生的相干混合）；
  2) 再做一段**纯耗散**，把概率“泵”到输出层（用单向 Lindblad 连接）。
- Fig.3（第5页）显示训练后能把平面测试点分出合理的决策边界。

### C. 序列分类（重点：句子识别）

- 结构见 Fig.1(c)（第2页）：输入层只有 \(|0\rangle\)，隐藏层每个词/对象一个神经元，输出层表示类别。
- **序列怎么编码？** 关键是“按时间顺序打开不同的 Lindblad 输入通道”（第5-6页）：词 \(w_1,w_2,\dots\) 出现的顺序，对应依次开启从 \(|0\rangle\) 到相应隐藏神经元的单向耗散通道。
- 之后同样：相干混合（隐藏层 \(H\)）+ 耗散到输出层（隐藏→输出的 Lindblad）。

---

## 4）观察到的“优势”是什么（主要来自数值模拟）

作者不声称全面碾压经典 NN，而是在玩具任务上观察到：

### （1）训练步数更少：coherent QSNN 收敛最快

- 在一个最小玩具句子识别任务：只有两种序列 \((w_1,w_2)\) 与 \((w_2,w_1)\) 分别标 yes/no，使用 5 个神经元的网络（Fig.1(c)）。
- Fig.4（第7页）显示：平均 loss 随迭代下降速度
  **coherent QSNN（蓝）最快** > decoherent QSNN（红）> classical NN（灰）。

### （2）泛化到“新类型输入（诗句）”的准确率更好

- 他们构造了 11 神经元（隐藏层 8 词）训练“普通句子/非句子”，测试集里加入两条诗句（Table I 第13页；Fig.5 第8页）。
- Fig.5（第8页）显示 coherent QSNN 在测试集平均准确率最好，优势主要体现在诗句（verse1/verse2）上；对普通句子三者都接近 1，差距不大。

### （3）鲁棒性：相干版本对标签噪声、器件噪声更稳

- 标签噪声：先用错误标签训练一段，再纠正标签继续训练。Fig.6（第8页）显示纠正后 coherent QSNN loss 下降更快。
- 器件噪声：用输出成功率对 Lindblad 参数的敏感度（导数平方和）定义 robustness（第9页 Eq.(18)）。Fig.7（第9页）显示 coherent QSNN robustness 更高。

---

## 5）论文的意义与局限

**意义：**

- 给出一种“量子随机行走 + Lindblad 开系统动力学”的 QNN 方案，结构上很贴近“图上的传播/吸收”，并且天然适合序列的时间注入。
- 用同一套框架覆盖回归、点分类、序列分类，说明“表达形式”比较统一。

**局限：**

- 所有优势主要来自**经典计算机上的数值模拟**；真正量子硬件上如何高效算梯度/训练，作者在结论里明确说“高度非平凡，需要进一步探索”（第9页 Conclusion）。
- 展示任务规模都很小（toy model），优势是否能扩展到更大规模、真实 NLP/分类任务并不确定。

---
