# 1.1 Operators and Symmetry

## 1.1.1 

这一节的目标，是把无源、无耗散介质中的 Maxwell 方程改写成一种更对称的形式，从而把系统的守恒结构直接显示出来。

核心思想是：

- 先把 Maxwell 方程写成一阶时间演化方程；
- 再通过变量替换，把时间演化写成一个由反对称算符生成的系统；
- 最后推出时间演化算符是正交变换，因此电磁能量守恒。

这也是后续构造稳定数值算法的基础。

---

## 1.1.2 从 Maxwell 方程出发

在无源、无耗散情况下，Maxwell 方程写成：

式(2.3a)
$$
\frac{\partial H(t)}{\partial t}=
-\frac{1}{\mu}\nabla \times E(t)
$$

式(2.3b)
$$
\frac{\partial E(t)}{\partial t}=
\frac{1}{\epsilon}\nabla \times H(t)
$$

这里：

- $E(t)$ 是电场；
- $H(t)$ 是磁场；
- $\epsilon$ 是介电常数；
- $\mu$ 是磁导率。

这是一组一阶耦合方程。

---





## 1.1.3 引入新的场变量

为了把系统写成更对称的形式，文中定义：

式(2.8)
$$
X(t) = \sqrt{\mu}\, H(t)
$$

$$
Y(t) = \sqrt{\epsilon}\, E(t)
$$

这一步的作用是：

- 把电场和磁场重新加权；
- 使演化算符的反对称结构显式出现；
- 最终把能量守恒写成“范数守恒”。

---

## 1.1.4 推导关于 $X(t)$ 的方程

先看第一个定义：

$$
X(t) = \sqrt{\mu}\, H(t)
$$

对时间求导：

$$
\frac{\partial X(t)}{\partial t}=
\sqrt{\mu}\frac{\partial H(t)}{\partial t}
$$

再代入 Maxwell 方程式(2.3a)：

$$
\frac{\partial X(t)}{\partial t}=
\sqrt{\mu}\left( -\frac{1}{\mu}\nabla \times E(t) \right)
$$



再利用

$$
E(t) = \frac{Y(t)}{\sqrt{\epsilon}}
$$

得到：

$$
\frac{\partial X(t)}{\partial t}=
-\frac{1}{\sqrt{\mu}}\nabla \times \left( \frac{Y(t)}{\sqrt{\epsilon}} \right)
$$

也就是

$$
\frac{\partial X(t)}{\partial t}=
-\frac{1}{\sqrt{\mu}}\nabla \times \frac{1}{\sqrt{\epsilon}}\, Y(t)
$$

---




## 1.1.5 写成统一的一阶时间演化形式

于是，$X(t)$ 和 $Y(t)$ 满足如下耦合方程：

$$
\frac{\partial X(t)}{\partial t}=
-\frac{1}{\sqrt{\mu}}\nabla \times \frac{1}{\sqrt{\epsilon}}\, Y(t)
$$

$$
\frac{\partial Y(t)}{\partial t}=
\frac{1}{\sqrt{\epsilon}}\nabla \times \frac{1}{\sqrt{\mu}}\, X(t)
$$

现在定义一个总状态向量：

$$
\Psi(t) = (X(t), Y(t))^T
$$

则方程可以简写成：

式(2.11)
$$
\frac{\partial \Psi(t)}{\partial t}=
\mathcal{H}\Psi(t)
$$

其中时间演化算符 $\mathcal{H}$ 的结构可以理解为：

$$
\mathcal{H}=
\begin{pmatrix}
0 & -\frac{1}{\sqrt{\mu}}\nabla \times \frac{1}{\sqrt{\epsilon}} \\
\frac{1}{\sqrt{\epsilon}}\nabla \times \frac{1}{\sqrt{\mu}} & 0
\end{pmatrix}
$$

如果你的 Markdown 环境对矩阵支持不稳定，也可以不写成矩阵，而直接说：

- $\mathcal{H}$ 的左上角和右下角是 0；
- 右上角是 $-\frac{1}{\sqrt{\mu}}\nabla \times \frac{1}{\sqrt{\epsilon}}$；
- 左下角是 $\frac{1}{\sqrt{\epsilon}}\nabla \times \frac{1}{\sqrt{\mu}}$。



这一节最关键的结论是：

式(2.10)
$$
\mathcal{H}^T = -\mathcal{H}
$$

也就是说，$\mathcal{H}$ 是一个反对称算符。



---

## 1.1.6 时间演化算符的形式解

既然系统满足

$$
\frac{\partial \Psi(t)}{\partial t}=
\mathcal{H}\Psi(t)
$$

那么它的形式解就是：

式(2.12)
$$
\Psi(t) = e^{t\mathcal{H}}\Psi(0)
$$

定义

$$
U(t) = e^{t\mathcal{H}}
$$

则可以写成：

$$
\Psi(t) = U(t)\Psi(0)
$$

这里 $U(t)$ 就是时间演化算符。

---

## 1.1.7 反对称性能推出正交性

因为

$$
\mathcal{H}^T = -\mathcal{H}
$$

所以

$$
U(t)^T=
\left( e^{t\mathcal{H}} \right)^T=
e^{t\mathcal{H}^T}=
e^{-t\mathcal{H}}
$$

另一方面，

$$
U(t)^{-1} = e^{-t\mathcal{H}}
$$

因此得到：

式(2.15)
$$
U(t)^T = U(t)^{-1}
$$

这正是正交变换的定义。

所以，时间演化算符 $U(t)$ 是正交的。

---

## 1.1.8 正交性意味着范数守恒

既然 $U(t)$ 是正交变换，它就保持向量长度不变，因此：

$$
\| \Psi(t) \| = \| \Psi(0) \|
$$

或者写成平方范数形式：

$$
\langle \Psi(t), \Psi(t) \rangle=
\langle \Psi(0), \Psi(0) \rangle
$$

这说明随着时间演化，系统的“总长度”不发生变化。

---

## 2.1 Chebyshev time integration

### 2.1.1 这一节要解决什么问题？

在空间离散之后，Maxwell 方程已经被写成一阶线性常微分方程：

$$
\frac{\partial \Psi(t)}{\partial t} = H \Psi(t)
$$

它的形式解为：

$$
\Psi(t) = e^{tH}\Psi(0)
$$

因此，时间推进的核心问题就变成：

> 如何高效、准确地计算矩阵指数 $e^{tH}$ 对初态 $\Psi(0)$ 的作用？

Chebyshev 方法的基本思想是：

- 不直接逐步积分微分方程；
- 而是直接把时间演化算符 $e^{tH}$ 展开成 Chebyshev 多项式的级数；
- 然后利用递推关系高效计算。

这一方法的优点是精度极高，而且误差会随展开阶数快速衰减。

---

### 2.1.2  Chebyshev 多项式

Chebyshev 多项式 $T_n(x)$ 定义在区间 $[-1,1]$ 上，并满足良好的逼近性质。

一个标量函数 $f(x)$ 可以展开成：

$$
f(x) = \frac{1}{2}a_0 T_0(x) + \sum_{n=1}^{\infty} a_n T_n(x)
$$

其中展开系数为：

$$
a_n = \frac{2}{\pi}\int_0^\pi \cos(n\theta)\, f(\cos\theta)\, d\theta
$$

Chebyshev 多项式本身定义为：

$$
T_n(x) = \cos\left(n\cos^{-1}x\right)
$$

并满足递推关系：

$$
T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
$$

初值为：

$$
T_0(x) = 1
$$

$$
T_1(x) = x
$$

这意味着：如果一个算符的谱已经被归一化到区间 $[-1,1]$，那么就可以把关于这个算符的函数，例如指数函数，展开成 Chebyshev 多项式。

---



### 2.1.3 引入矩阵 $A=-iH$

由于 $H$ 是实反对称矩阵，所以它满足：

$$
H^T = -H
$$

反对称矩阵的本征值都是纯虚数。  
为了把问题转化成一个谱为实数的矩阵，定义：

$$
A = -iH
$$

因为 $H$ 的本征值是纯虚数，所以 $A$ 的本征值全为实数。

而且，$A$ 是 Hermitian 矩阵。  
这一步非常重要，因为只有这样，后面才能把谱压缩到实区间 $[-1,1]$ 上。

---

### 2.1.4 把矩阵谱归一化到 $[-1,1]$

设 $A$ 的谱半径为 $\rho(A)$，也就是最大本征值绝对值。

理想情况下，我们可以直接用 $\rho(A)$ 来归一化，但在大规模稀疏矩阵问题中，精确求谱半径通常代价很高。

因此文中采用一个上界：

$$
\rho(A) \leq \|A\|_\infty
$$

其中无穷范数定义为：

$$
\|A\|_\infty = \max_i \sum_j |A_{ij}|
$$

由于 $A$ 是 Hermitian 矩阵，也有：

$$
\|A\|_\infty = \|A\|_1
$$

于是定义归一化矩阵：

$$
B = \frac{A}{\|A\|_1}
$$

这样一来，$B$ 的全部本征值都落在区间 $[-1,1]$ 内。

这一步就是为了使 Chebyshev 多项式展开可以合法使用。

---

### 2.1.6 缩放时间

因为我们已经把矩阵除以 $\|A\|_1$，所以必须在时间变量中补回来这个缩放。

定义新的无量纲时间参数：

$$
z = t\|A\|_1
$$

由于

$$
A = \|A\|_1 B
$$

于是有：

$$
tA = t\|A\|_1 B = zB
$$

再利用 $A=-iH$，可得：

$$
H = iA
$$

所以：

$$
e^{tH} = e^{itA} = e^{izB}
$$

因此，原来的时间演化问题就变成了：

$$
\Psi(t) = e^{izB}\Psi(0)
$$

现在指数函数的自变量已经是谱位于 $[-1,1]$ 上的矩阵 $B$，于是可以进行 Chebyshev 展开。

---

### 2.1.7 对指数函数做 Chebyshev 展开

考虑标量函数：

$$
f(x) = e^{izx}
$$

根据 Chebyshev 展开公式：

$$
f(x) = \frac{1}{2}a_0(z)T_0(x) + \sum_{n=1}^{\infty} a_n(z)T_n(x)
$$

所以对矩阵 $B$ 也有：

$$
e^{izB}=
\frac{1}{2}a_0(z)I + \sum_{n=1}^{\infty} a_n(z)T_n(B)
$$

于是作用在初态上得到：

$$
\Psi(t)=
\left[
\frac{1}{2}a_0(z)I + \sum_{n=1}^{\infty} a_n(z)T_n(B)
\right]\Psi(0)
$$

这里 $I$ 是单位矩阵。

---

### 2.1.8 计算展开系数 $a_n(z)$

根据一般公式：

$$
a_n(z) = \frac{2}{\pi}\int_0^\pi \cos(n\theta)e^{iz\cos\theta}\, d\theta
$$

这个积分正好与 Bessel 函数的积分表示有关。  
计算后得到：

$$
a_n(z) = 2 i^n J_n(z)
$$

其中 $J_n(z)$ 是第一类 Bessel 函数。

因此，时间演化算符可以写成：

$$
e^{izB}=
J_0(z)I + 2\sum_{n=1}^{\infty} i^n J_n(z) T_n(B)
$$

于是状态向量满足：

$$
\Psi(t)=
\left[
J_0(z)I + 2\sum_{n=1}^{\infty} i^n J_n(z) T_n(B)
\right]\Psi(0)
$$

这就是文中 Chebyshev 时间积分的核心展开式。

---

### 2.1.9 重新定义 $ \widetilde{T}_n(B)$ 

上式里反复出现组合项：

$$
i^n T_n(B)
$$

为了简化记号，文中定义：

$$
\widetilde{T}_n(B) = i^n T_n(B)
$$

于是演化公式可写成更紧凑的形式：

$$
\Psi(t)=
\left[
J_0(z)I + 2\sum_{n=1}^{\infty} J_n(z)\widetilde{T}_n(B)
\right]\Psi(0)
$$

这样后面递推时公式会更简单。

---

### 2.1.10 递推关系

普通 Chebyshev 多项式满足：

$$
T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
$$

现在把它推广到矩阵情形，并乘上因子 $i^{n+1}$：

$$
i^{n+1}T_{n+1}(B)=
i^{n+1}\left( 2BT_n(B) - T_{n-1}(B) \right)
$$

把各项整理：

$$
i^{n+1}T_{n+1}(B)=
2iB \cdot i^n T_n(B) + i^{n-1}T_{n-1}(B)
$$

于是得到新的递推关系：

$$
\widetilde{T}_{n+1}(B)\Psi(0)=
2iB\,\widetilde{T}_n(B)\Psi(0)
+
\widetilde{T}_{n-1}(B)\Psi(0)
$$

初始两项为：

$$
\widetilde{T}_0(B)\Psi(0) = \Psi(0)
$$

$$
\widetilde{T}_1(B)\Psi(0) = iB\Psi(0)
$$

因此，所有高阶项都可以通过递推得到，而不需要真的去计算矩阵多项式。

---

### 2.1.11 为什么这个递推特别适合数值实现？

因为在整个递推过程中，我们只需要做两类操作：

- 稀疏矩阵 $B$ 与向量的乘法；
- 向量的线性组合。

也就是说，我们并不需要显式构造矩阵指数，也不需要存储高阶矩阵幂。

这对大规模稀疏系统非常重要，因为：

- 存储量小；
- 计算效率高；
- 易于实现。

---



### 2.1.12 实际计算时必须截断级数

理论上展开式是无穷级数：

$$
\Psi(t)=
\left[
J_0(z)I + 2\sum_{n=1}^{\infty} J_n(z)\widetilde{T}_n(B)
\right]\Psi(0)
$$

但数值实现时只能保留前 $m+1$ 项：

$$
\Psi(t)
\approx
\left[
J_0(z)I + 2\sum_{n=1}^{m} J_n(z)\widetilde{T}_n(B)
\right]\Psi(0)
$$

因此问题就变成：

> 如何选择截断阶数 $m$，使误差足够小？

---

### 2.1.13 估计截断误差

文中引入控制参数 $\kappa$，要求当 $n>m$ 时：

$$
|J_n(z)| < \kappa
$$

也就是说，只要后续 Bessel 系数已经足够小，就可以停止展开。

由于 Bessel 函数满足快速衰减性质：

$$
|J_n(z)| \leq \frac{|z|^n}{2^n n!}
$$

所以随着 $n$ 增大，系数会非常快地减小。

这意味着截断误差实际上是指数级衰减的。

文中指出，取

$$
\kappa = 10^{-13}
$$

通常就足以保证结果接近机器精度。

---

## 2.2 Lie-Trotter-Suzuki time integration



最直接的想法可能是用 Taylor 展开：

$$
e^{\tau H}=
I + \tau H + \frac{(\tau H)^2}{2!} + \frac{(\tau H)^3}{3!} + \cdots
$$

如果只保留前两项，就得到 Euler 格式：

$$
\widetilde{U}(\tau) = I + \tau H
$$

但是这个近似一般不是正交矩阵。  
因为：

$$
\widetilde{U}(\tau)^T \widetilde{U}(\tau)=
(I+\tau H)^T(I+\tau H)
$$

由于 $H^T=-H$，所以：

$$
(I+\tau H)^T = I-\tau H
$$

于是：

$$
\widetilde{U}(\tau)^T \widetilde{U}(\tau)=
(I-\tau H)(I+\tau H)=
I - \tau^2 H^2
$$

这通常不等于单位矩阵 $I$。

因此：

- Euler 格式不保持正交性；
- 范数不守恒；
- 能量会出现虚假增长或衰减；
- 算法不稳定。

这说明：  
对于 Maxwell 这样的守恒系统，简单截断指数级数并不是一个好办法。

---

### 2.2.1 Lie-Trotter-Suzuki 的基本思想

设矩阵 $H$ 可以分解为若干部分之和：

$$
H = H_1 + H_2 + \cdots + H_p
$$

并且要求每个 $H_i$ 都是实反对称矩阵，也就是说：

$$
H_i^T = -H_i
$$

如果每个 $H_i$ 的指数算符 $e^{\tau H_i}$ 都能高效计算，那么就可以尝试用这些小指数算符的乘积来近似整体指数算符：

$$
e^{\tau H}=
e^{\tau(H_1+\cdots+H_p)}
$$

Lie-Trotter 公式告诉我们：

$$
e^{t(H_1+\cdots+H_p)}=
\lim_{m\to\infty}
\left(
e^{tH_1/m} e^{tH_2/m}\cdots e^{tH_p/m}
\right)^m
$$

这就是这一方法的理论基础。

直观理解是：

- 如果一个大时间步不好算；
- 那么把它切成很多很小的步长；
- 在每个小步里，分别作用 $H_1,H_2,\cdots,H_p$；
- 当步长足够小时，这样的顺序作用就能逼近整体作用。

---

### 2.2.2 一阶 Lie-Trotter 近似

从上面的极限公式出发，如果只取一次乘积，就得到最简单的一阶近似：

$$
U_1(\tau) = e^{\tau H_1} e^{\tau H_2} \cdots e^{\tau H_p}
$$

这就是文中所谓的一阶 product formula。

为什么说它是一阶近似？

因为它与精确算符 $e^{\tau H}$ 在 $\tau$ 的一阶项上是一致的。

下面直接验证。



对每个指数分别作小步长展开：

$$
e^{\tau H_i} = I + \tau H_i + O(\tau^2)
$$

因此它们的乘积为：

$$
U_1(\tau)=
(I+\tau H_1 + O(\tau^2))
(I+\tau H_2 + O(\tau^2))
\cdots
(I+\tau H_p + O(\tau^2))
$$

把一阶项收集起来，得到：

$$
U_1(\tau)=
I + \tau(H_1+H_2+\cdots+H_p) + O(\tau^2)
$$

而由于

$$
H = H_1 + H_2 + \cdots + H_p
$$

所以：

$$
U_1(\tau) = I + \tau H + O(\tau^2)
$$

另一方面，精确算符也有展开：

$$
e^{\tau H} = I + \tau H + O(\tau^2)
$$

因此两者在一阶上完全一致，误差从二阶开始，于是称为一阶近似。

---

### 2.2.3 一阶近似是无条件稳定的

这正是 Lie-Trotter-Suzuki 方法最重要的优点。

由于每个 $H_i$ 都满足：

$$
H_i^T = -H_i
$$

所以每个小指数算符都是正交矩阵：

$$
\left( e^{\tau H_i} \right)^T = e^{-\tau H_i} = \left( e^{\tau H_i} \right)^{-1}
$$

因此：

$$
e^{\tau H_i}
$$

保持向量范数不变。

而正交矩阵的乘积仍然是正交矩阵，所以：

$$
U_1(\tau) = e^{\tau H_1} e^{\tau H_2} \cdots e^{\tau H_p}
$$

也是正交矩阵。

这意味着：

$$
U_1(\tau)^T U_1(\tau) = I
$$

于是无论步长 $\tau$ 取得多大，$U_1(\tau)$ 都不会放大解的范数。

所以这种算法是：

> 无条件稳定的。

这里“无条件稳定”指的是：  
稳定性不是靠把时间步长取得极小来换来的，而是由算法本身的结构保证的。

---

### 2.2.4 一阶近似的误差

虽然 $U_1(\tau)$ 保持正交性，但它并不等于精确算符 $e^{\tau H}$。

误差的根源在于：一般来说，不同的 $H_i$ 之间并不对易，也就是：

$$
[H_i, H_j] = H_iH_j - H_jH_i \neq 0
$$

如果所有 $H_i$ 都彼此对易，那么就有：

$$
e^{\tau(H_1+\cdots+H_p)}=
e^{\tau H_1} e^{\tau H_2}\cdots e^{\tau H_p}
$$

这时一阶公式就是精确的。

但通常并不对易，所以只能是近似。

文中给出误差上界：

$$
\| U(\tau) - U_1(\tau) \|
\le
\frac{\tau^2}{2}
\sum_{i<j}\| [H_i,H_j] \|
$$

这里：

- $U(\tau)=e^{\tau H}$ 是精确时间演化算符；
- $U_1(\tau)$ 是一阶近似；
- 误差量级是 $O(\tau^2)$；
- 误差大小由各个子算符之间的对易程度决定。

因此可以看出：

> 算符越接近对易，一阶 Lie-Trotter 近似越准确。

---

### 2.2.5 从一阶方法构造二阶方法

一阶近似的主要问题是精度有限。  
为了提高精度，一个自然的想法是做“对称化”。

文中给出的二阶近似为：

$$
U_2(\tau) = U_1(-\tau/2)^T U_1(\tau/2)
$$

下面解释为什么这个写法有意义。

先写出：

$$
U_1(\tau/2) = e^{\tau H_1/2} e^{\tau H_2/2} \cdots e^{\tau H_p/2}
$$

由于每个因子都是正交矩阵，所以：

$$
U_1(-\tau/2)^T=
e^{\tau H_p/2} e^{\tau H_{p-1}/2}\cdots e^{\tau H_1/2}
$$

因此二阶公式变成：

$$
U_2(\tau)=
e^{\tau H_p/2} \cdots e^{\tau H_2/2} e^{\tau H_1/2}
e^{\tau H_1/2} e^{\tau H_2/2}\cdots e^{\tau H_p/2}
$$

合并中间的两个 $e^{\tau H_1/2}$ 就得到一种首尾对称的分解。



这种对称结构非常重要，因为它会自动消去展开中的奇数阶误差项，从而使总误差提高到三阶，也就是说该方法成为二阶精度方法。

---



### 2.2.6 进一步构造四阶方法

如果还想提高精度，可以继续利用 Suzuki 的分形分解思想。

文中给出的四阶近似为：

$$
U_4(\tau)=
U_2(a\tau)
U_2(a\tau)
U_2((1-4a)\tau)
U_2(a\tau)
U_2(a\tau)
$$

其中常数 $a$ 取为：

$$
a = \frac{1}{4-4^{1/3}}
$$

这个公式看上去很特别，本质上是在用多个二阶块进行特殊组合，使得低阶误差进一步相消。

由于二阶方法的误差结构是已知的，Suzuki 通过精心选择系数 $a$，让三阶误差项相互抵消，于是总体误差提升到五阶，也就是说这个方法成为四阶近似。



高阶公式通过递推构造：

$$
S_{2k}(\tau)=
S_{2k-2}(p_k \tau)^2
S_{2k-2}((1 - 4p_k)\tau)
S_{2k-2}(p_k \tau)^2
$$

其中：

$$
p_k = \frac{1}{4 - 4^{1/(2k-1)}}
$$

---
## 3.1 Implementation

### 3.1.1 这一节的目标是什么？

在前面已经得到了一个很清楚的理论框架：

- 时间演化由指数算符决定；
- 可以用 Trotter-Suzuki 方法构造高精度、无条件稳定的时间推进算法。

但是，理论上的算符公式要变成真正可计算的程序，还需要完成一个关键步骤：

> 把连续空间中的薛定谔方程离散化，变成一个有限维矩阵方程。

这一节就是在做这件事。

它的核心任务有三步：

1. 对连续空间进行离散；
2. 把连续哈密顿量写成矩阵形式；
3. 把这个矩阵进一步分解成几个“容易指数化”的小块矩阵。

这样，抽象的指数算符问题就会变成一系列简单的向量更新和小矩阵旋转。

---

### 3.1.2 从连续薛定谔方程出发

文中首先考虑一个最简单的一维模型：

$$
H = -\frac{d^2}{dx^2} + V(x)
$$

这里：

- 第一项是动能项；
- 第二项是势能项。

系统定义在区间 $[0,X]$ 上，并取自由端边界条件。  
这意味着波函数在边界之外为零，也就是：

$$
\psi(x,t) = 0
\quad \text{for } x<0 \text{ or } x>X
$$

对应的时间依赖薛定谔方程为：

$$
\frac{\partial}{\partial t}\psi(x,t) = -i H \psi(x,t)
$$

也就是：

$$
\frac{\partial}{\partial t}\psi(x,t)=
-i\left(
-\frac{d^2}{dx^2} + V(x)
\right)\psi(x,t)
$$

这一方程还是连续形式，计算机不能直接处理，所以必须做空间离散化。

---

### 3.1.3 空间离散化

设空间网格长度为 $\delta$，把区间划分成一系列离散格点。

定义离散波函数：

$$
\psi_l(t) \approx \psi(x_l,t)
$$

其中：

$$
x_l = l\delta
$$

也就是说，我们用格点上的函数值来近似连续波函数。

接下来最关键的是：  
如何离散二阶导数？

---

### 3.1.4 二阶导数的有限差分近似

对连续函数的二阶导数，使用最简单的中心差分格式：

$$
\frac{d^2}{dx^2}\psi(x_l,t)
\approx
\frac{\psi_{l+1}(t) - 2\psi_l(t) + \psi_{l-1}(t)}{\delta^2}
$$

这一步非常标准，它把连续导数变成了相邻格点之间的差分关系。

现在把它代回哈密顿量：

$$
H\psi_l(t)=
-\frac{1}{\delta^2}
\left(
\psi_{l+1}(t) - 2\psi_l(t) + \psi_{l-1}(t)
\right)
+
V_l \psi_l(t)
$$

其中：

$$
V_l = V(x_l)
$$

展开以后可写成：

$$
H\psi_l(t)=
-\frac{1}{\delta^2}\psi_{l+1}(t)
-\frac{1}{\delta^2}\psi_{l-1}(t)
+
\left(
\frac{2}{\delta^2} + V_l
\right)\psi_l(t)
$$

为了简化记号，文中定义：

$$
v_l = V_l + \frac{2}{\delta^2}
$$

于是离散哈密顿量作用在波函数上的结果写成：

$$
H\psi_l(t)=
-\frac{1}{\delta^2}\psi_{l+1}(t)
-\frac{1}{\delta^2}\psi_{l-1}(t)
+
v_l \psi_l(t)
$$

---

### 3.1.5 离散后的时间依赖薛定谔方程

代入时间依赖薛定谔方程：

$$
\frac{\partial}{\partial t}\psi_l(t)=
-i H\psi_l(t)
$$

得到：

$$
\frac{\partial}{\partial t}\psi_l(t)=
-i\left[
-\frac{1}{\delta^2}\psi_{l+1}(t)
-\frac{1}{\delta^2}\psi_{l-1}(t)
+
v_l\psi_l(t)
\right]
$$

这就是离散格点上的演化方程。

如果把它整理成文中的形式，就是：

$$
\frac{\partial}{\partial t}\psi_l(t)=
-i
\left\{
-\delta^{-2}\left[\psi_{l+1}(t)+\psi_{l-1}(t)\right]
+
v_l \psi_l(t)
\right\}
$$

这一步非常重要，因为它说明：

> 连续偏微分方程已经被转换成一个有限维常微分方程组。

---

### 3.1.6 把离散方程写成矩阵形式

现在把所有格点上的波函数收集成一个列向量：

$$
\psi(t)=
\begin{pmatrix}
\psi_1(t) \\
\psi_2(t) \\
\vdots \\
\psi_{L+1}(t)
\end{pmatrix}
$$

那么整个系统可以写成矩阵方程：

$$
\frac{\partial}{\partial t}\psi(t) = -i H \psi(t)
$$

其中矩阵 $H$ 是一个三对角矩阵：

$$
H =
\begin{pmatrix}
v_1 & -\delta^{-2} & 0 & \cdots & 0 \\
-\delta^{-2} & v_2 & -\delta^{-2} & \cdots & 0 \\
0 & -\delta^{-2} & v_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & -\delta^{-2} \\
0 & 0 & 0 & -\delta^{-2} & v_{L+1}
\end{pmatrix}
$$

这个矩阵有非常明确的物理意义：

- 对角元 $v_l$ 来自势能和离散动能中的中心项；
- 非对角元 $-\delta^{-2}$ 表示粒子从一个格点跳到相邻格点。

因此，这个离散问题也可以理解为一个“单粒子在一维晶格上跳跃”的问题。

---



### 3.1.7 继续分解矩阵 $H$

虽然现在已经得到了矩阵形式，但我们最终要算的是：

$$
e^{-i\tau H}
$$

即时间步长 $\tau$ 上的演化算符。

问题是：

- 直接对整个大矩阵 $H$ 求指数，数值代价很高；
- 对大规模问题尤其不可行。

因此需要进一步把 $H$ 分解成若干个“容易指数化”的小块矩阵。

文中选取的分解方式是：

$$
H = H_0 + H_1 + H_2
$$

其中：

- $H_0$：对角矩阵；
- $H_1$：由若干互不相交的 $2\times 2$ 小块组成；
- $H_2$：由另外一组互不相交的 $2\times 2$ 小块组成。

这个分解方式是整个实现部分的关键。

---



### 3.1.8 对角部分 $H_0$

把原矩阵中的对角元全部取出来：

$$
H_0 =
\begin{pmatrix}
v_1 & 0 & 0 & \cdots & 0 \\
0 & v_2 & 0 & \cdots & 0 \\
0 & 0 & v_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & 0 \\
0 & 0 & 0 & 0 & v_{L+1}
\end{pmatrix}
$$

这一部分很容易指数化，因为：

$$
e^{-i\tau H_0}
$$

仍然是对角矩阵，只需对每个对角元分别取指数即可。

---

### 3.1.9 第一组跳跃块 $H_1$

把某些相邻格点之间的耦合集中到一起，例如：

- 第1和第2个格点；
- 第3和第4个格点；
- 第5和第6个格点；

等等。

这样得到的矩阵 $H_1$ 是一个块对角矩阵，每个非零块都是一个 $2\times 2$ 小矩阵：

$$
\begin{pmatrix}
0 & V \\
V & 0
\end{pmatrix}
$$

这些小块彼此互不重叠，因此可以独立处理。

---

### 3.1.10第二组跳跃块 $H_2$

剩余的相邻跳跃项放到 $H_2$ 里，例如：

- 第2和第3个格点；
- 第4和第5个格点；
- 第6和第7个格点；

等等。

于是 $H_2$ 也是一个由许多 $2\times 2$ 小块组成的块对角矩阵。

这两个矩阵交替地覆盖整条链上的所有邻接耦合。

---



### 3.1.11 计算 $2\times 2$ 小块的指数

设一个基本块为：

$$
M =
\begin{pmatrix}
0 & V \\
V & 0
\end{pmatrix}
$$

我们希望计算：

$$
e^{-i\tau M}
$$

先注意到：

$$
M^2 =
\begin{pmatrix}
V^2 & 0 \\
0 & V^2
\end{pmatrix}=
V^2 I
$$

这说明：

- 偶次幂给出单位矩阵；
- 奇次幂给出 $M$ 本身。

于是指数展开为：

$$
e^{-i\tau M}=
I - i\tau M + \frac{(-i\tau)^2}{2!}M^2 + \frac{(-i\tau)^3}{3!}M^3 + \cdots
$$

把偶次项和奇次项分别整理，可得：

$$
e^{-i\tau M}=
\cos(\tau |V|)\, I-
i \sin(\tau |V|)\, \frac{M}{|V|}
$$

如果把它写成矩阵形式，就是：

$$
e^{-i\tau M}=
\begin{pmatrix}
\cos(\tau |V|) & -i\sin(\tau |V|) \\
-i\sin(\tau |V|) & \cos(\tau |V|)
\end{pmatrix}
$$

这就是文中所谓的 plane rotation matrix。

它表明：  
每一个 $2\times 2$ 小块的时间推进，本质上就是对两个相邻分量做一次二维旋转。

---

### 3.1.12 一阶 Trotter 近似

既然已经分解出：

$$
H = H_0 + H_1 + H_2
$$

那么一阶 Trotter 公式就是：

$$
U_1(\tau)=
e^{-i\tau H_0}
e^{-i\tau H_1}
e^{-i\tau H_2}
$$

这意味着一个时间步可以分成三步：

1. 先作用对角势能部分 $e^{-i\tau H_0}$；
2. 再作用第一组相邻点旋转 $e^{-i\tau H_1}$；
3. 最后作用第二组相邻点旋转 $e^{-i\tau H_2}$。

每一步都很容易实现：

- 对角部分：逐点乘相位；
- 块对角部分：逐对做 $2\times 2$ 旋转。

于是整个时间推进就被转化为一套非常简单的局域更新规则。

---

### 3.1.13 为什么这种实现特别适合计算机？

因为整个算法最终只需要：

- 向量和标量相乘；
- 对相邻两个分量做 $2\times 2$ 旋转；
- 不需要存储完整矩阵指数；
- 不需要全矩阵对角化。

这意味着：

- 内存消耗低；
- 实现简单；
- 易于向量化；
- 易于并行化。

从程序实现角度看，这一点比抽象公式本身更重要。

---










