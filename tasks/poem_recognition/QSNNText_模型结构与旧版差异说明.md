# QSNNText 当前诗句识别模型：完整结构与原理（含旧版全量差异）

## 1. 文档目的

本文用于说明当前诗句识别实现（`QSNNText` + `retest_poem_recognition.py`）的完整结构、计算原理、训练/评估流程，并与旧版 `classical NN.py` 做逐项对照。

核心代码位置：
- 当前模型：`models.py` 中 `QSNNText`
- 当前测试脚本：`retest_poem_recognition.py`
- 旧版实现：`prr_project_and_data/prr_project_and_data/project_and_data/final_data/classicalNN-poem/classical NN.py`

---

## 2. 当前模型（QSNNText）完整结构

### 2.1 总体结构

当前模型是“文本预处理 + 量子句子编码 + 二分类读出头”的三段式结构：

1. 文本预处理（NLTK 或简化规则）
2. 量子编码（按词序做 Lindblad 演化，得到句向量）
3. 线性二分类头（2 logit + softmax）

### 2.2 输入与词表

- 默认词表：`[gold, sun, dawn, love, stay, day, go, noth]`
- 词数：`N_words = 8`
- 量子系统维度：`N = N_words + 3 = 11`

这与旧版的基础维度约定保持一致（11 维系统）。

### 2.3 文本预处理层

`QSNNText.preprocess_sentence()` 支持两条路径：

1. NLTK 路径（`use_nltk=True` 且语料可用）
- 小写与去标点
- `word_tokenize`
- `pos_tag`
- `WordNetLemmatizer`（结合 POS）
- 去停用词（`stopwords.words('english')`）
- `SnowballStemmer`
- 过滤到词表内 token

2. 简化路径（NLTK 不可用或显式关闭）
- 正则分词
- 内置 stem map（如 loves->love, goes->go, nothing->noth）
- 过滤到词表内 token

### 2.4 量子句子编码层

`QSNNText.sentence_feature()` 的流程如下：

1. 初态：
- $\rho_0 = |0\rangle\langle 0|$

2. 每个词对应一个 Lindblad 跳跃算符：
- 词 `i`（0-based）映射到 `line=i+1`
- $L_i = \gamma |line\rangle\langle 0|$

3. 时间设置：
- `delta_t = int(t_input / words_in_sentence)`
- `t_word = (delta_t - 1) / 2`（下界保护）

4. 按词序依次演化：
- 每一步调用 `qsw.evolve_auto(rho, H, [L_i], t_word)`
- 当前实现中 `H=0`（全零哈密顿量）

5. 读出：
- 取最终态对角线
- 使用槽位 `1..N_words` 作为 8 维句向量特征

### 2.5 二分类读出层

读出层是优化后的向量化二分类头：
- `nn.Linear(N_words, 2)`
- softmax 输出 `p(no), p(yes)`

它在判别形式上与旧版 “yes/no 两个线性分支 + softmax” 等价，但实现更紧凑、易训练、可批处理。

### 2.6 前向输出定义

`QSNNText.forward(sentences, labels=None)` 返回：
- `features`: `(B, N_words)`
- `logits`: `(B, 2)`
- `probs`: `(B, 2)`
- `legacy_cost`（可选）:
  - 定义为 $1 - \text{mean}(p_{correct})$
  - 与旧版损失口径保持一致

---

## 3. 当前训练与评估流程（retest 脚本）

`retest_poem_recognition.py` 的流程：

1. 构建训练集：6 条 yes + 6 条 no
2. 生成多样本轨迹（默认 15 个随机初始化样本）
3. 训练 200 次更新并记录每一步：
- `loss` 轨迹
- 4 个固定测试句（test1~test4）的 `p_yes` 轨迹
4. 导出旧格式结果文件：
- `*_loss.txt`
- `*_test1.txt` 到 `*_test4.txt`
5. 如提供旧结果目录，生成：
- `old_vs_new_comparison.csv`
6. 用 `plot_old_vs_new.py` 绘图

学习率调度也复刻了旧版分段策略：
- `lr = lr0 / (1 + u / 15)`（`u<=100`）
- `lr = lr0 / (1 + 100 / 15)`（`u>100`）

并新增 NLTK 强校验：
- `--use-nltk` 时若模型退回简化路径会直接报错
- 运行时打印 `NLTK active: True/False`

---

## 4. 与旧版的“所有关键差异”清单

下面按模块逐项列出差异：

### 4.1 文本预处理

相同点：
- 都采用小写、去标点、词法归一化思想
- 都把文本压缩到同一小词表空间

差异点：
1. 当前版支持双路径（NLTK + fallback），旧版只有 NLTK 路径
2. 当前版增加了运行时强校验与降级保护，旧版无显式防护
3. 当前版把预处理逻辑封装进模型类方法，可复用；旧版写在脚本全局流程中

### 4.2 量子动力学层

相同点：
- 都是按词序输入，逐词叠加到量子演化中
- 都从 $|0\rangle\langle0|$ 初态出发
- 都在 1..N_words 神经元上读出

差异点：
1. 旧版：
- 显式构造 `H_COMPONENTS` 与 `C_COMPONENTS`
- 使用 `valedian0910.ChannelModel / Controller / ParameterizedLindbladChannel`
- 参数列表与时间控制器结构更复杂
2. 当前版：
- 用 `qsw.evolve_auto` 统一演化接口
- 当前采用 `H=0` 与词级单算符 `L_i=gamma|i><0|`
- 以简化可复现实用为目标，结构更轻

### 4.3 分类头

相同点：
- 都是 yes/no 二分类，softmax 决策

差异点：
1. 旧版：
- 手写 `w_no, w_yes, b_no, b_yes`
- 手推梯度更新
2. 当前版：
- `nn.Linear(8,2)` 向量化读出
- 通过 autograd 反传，手动参数步进
- 代码更短、更稳定、便于扩展

### 4.4 损失与优化

相同点：
- 都支持旧口径损失：$1-\text{mean}(p_{correct})$
- 都使用分段学习率衰减

差异点：
1. 当前版显式暴露 `legacy_cost`，便于与旧结果直接对齐
2. 当前版训练流程更模块化（单样本训练函数、轨迹函数、评估函数拆分）

### 4.5 工程可复现性

旧版问题：
- 大量硬编码路径
- 强依赖历史目录结构和外部模块
- 迁移成本高

当前版改进：
1. 命令行参数化（输出目录、前缀、样本数、更新数、设备、NLTK 开关）
2. 旧格式导出兼容（可直接接旧画图/统计流程）
3. 自动生成新旧对比 CSV
4. 支持 CPU 直接复现

### 4.6 输出协议

旧版输出：
- `classicalNN_loss.txt`
- `classicalNN_test1..4.txt`

当前版输出：
- `<prefix>_loss.txt`
- `<prefix>_test1..4.txt`
- `old_vs_new_comparison.csv`
- `old_vs_new_comparison.png`（由绘图脚本生成）

即：当前版在兼容旧格式的同时，新增了结构化对比输出。

---

## 5. test1~test4 的语义场景（沿用旧实验口径）

`retest_poem_recognition.py` 固定四条测试句：

1. test1: `so dawn goes down to day`
- 诗化表达 + 一定分布偏移

2. test2: `nothing gold can stay`
- 经典短诗句（verse）

3. test3: `i love to stay here until the dawn`
- 普通连贯句（sentence）

4. test4: `i love to go out for love`
- 普通句，含词重复与轻微语义不自然

旧文献/旧脚本中一般把 test1/2 看作 verse 情景，test3/4 看作 sentence 情景。

---

## 6. 结论

当前 `QSNNText` 版本的定位是：
- 在保持旧任务口径（词表、标签、损失、轨迹输出格式）的前提下
- 用更清晰、可维护、可复现的工程实现重做同类诗句识别流程
- 并用优化后的向量化二分类头替换旧版手写线性读出

它不是“逐行原样复刻旧脚本”，而是“逻辑复刻 + 工程重构 + 兼容旧评估协议”的现代化实现。
