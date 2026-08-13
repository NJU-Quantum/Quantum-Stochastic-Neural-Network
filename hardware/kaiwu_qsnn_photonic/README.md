# Kaiwu 相干光平台 QSNN 二分类作品

本目录给出与 `hardware/quafu_qsnn_16q` 平级的相干光平台作品。它使用
[Kaiwu-PyTorch-Plugin](https://github.com/qboson/kaiwu-pytorch-plugin) 提供的
Boltzmann Machine 与 Kaiwu SDK 采样接口，可切换经典模拟退火和相干光 CIM。

## 1. 作品定位

Kaiwu 相干光量子计算机原生求解 Ising/Boltzmann 采样问题，不执行通用量子门线路，
也不直接积分 Lindblad 主方程。因此本作品不是把当前 `QSNN2D` 的密度矩阵演化逐项
搬到光机，而是构造一个 **QSNN 启发的相干光能量判别器**：

- 输入节点：将二维样本编码为二进制 Ising 节点；
- 隐藏节点：形成可训练的相干光能量耦合与吸引子；
- 输出节点：两个互斥节点分别表示类别 0 和类别 1；
- 耗散语义：CIM 的耗散收敛与 Boltzmann 采样承担“向低能吸引子汇聚”的角色；
- 读出：固定输入后采样输出节点，统计 `p0/p1`。

它保留 QSNN 的“相干耦合 + 耗散吸引 + 双输出判别”计算思想，但不宣称与
GKLS/Lindblad 动力学数学等价。

## 2. 模型结构

基线网络包含：

- 10 个输入节点：`x/y` thermometer 编码、交叉符号和半径特征；
- 2 个输出节点：`out0/out1`；
- 8 个隐藏节点；
- 总计 20 个二值节点，导出为 21×21 Kaiwu Ising 矩阵（最后一维是偏置辅助节点）。

当前大规模路线由 `large_scale_qsnn.py` 实现，使用 Qboson-1000 的完整容量：

- 128 个输入编码自旋，分别编码 $x$、$y$、半径和 $xy$ 交叉项；
- 3 层各 256 个稀疏相干储备池自旋，共 768 个；
- 类别 0/1 各 51 个铁磁吸引子自旋，共 102 个；
- 1 个类别选择自旋和 1 个偏置辅助自旋；
- 总计 $128+768+102+1+1=1000$ 个 Ising 自旋。

每个待分类样本产生独立条件矩阵。只有输入场被固定，储备池、类别吸引子和选择自旋
由 SA/CIM 自由寻找低能态；硬件系数量化为有符号 8 位整数。

能量函数为

$$
E(s)=-b^Ts-\frac12s^TJs.
$$

训练使用条件对比目标：正相固定“输入 + 正确输出”，负相只固定输入，让采样器自由
产生输出与隐藏状态。输出节点初始化了 one-hot 能量约束，减少 `00/11` 无效输出。

## 3. 文件清单

- `photonic_qsnn.py`：训练、推理、Kaiwu sampler 适配、Ising 导出；
- `kaiwu_cim_repro.py`：对已导出的 Ising 矩阵执行官方 SA/CIM 可复现求解；
- `large_scale_qsnn.py`：构建、预检并提交 1000 自旋结构化条件 QSNN；
- `test_large_scale_qsnn.py`：检查满容量布局、矩阵与全局翻转不变读出；
- `test_photonic_qsnn.py`：编码、训练、读出、存档和导出测试；
- `requirements-kaiwu.txt`：官方插件 GitHub 地址及版本环境；
- `outputs/`：训练后生成 checkpoint、Ising 矩阵和结果摘要。

## 4. 本地完整运行

本地模式只依赖仓库已有的 PyTorch/Numpy，可先验证作品闭环：

```bash
python3 hardware/kaiwu_qsnn_photonic/photonic_qsnn.py \
  --mode train \
  --sampler local \
  --epochs 40
```

训练后推理：

```bash
python3 hardware/kaiwu_qsnn_photonic/photonic_qsnn.py \
  --mode predict \
  --sampler local \
  --x 0.2 \
  --y -0.4
```

## 5. Kaiwu SDK / 插件安装

官方插件当前文档环境为 Python 3.10、Kaiwu SDK 1.3.1、PyTorch 2.7.0 和
Numpy 2.2.6。建议建立独立环境，不要覆盖仓库当前 PyTorch 环境：

两个官方 wheel 的 Python 依赖不兼容于同一个环境：Kaiwu wheel 是 CPython 3.10
专用，而 WuYue 1.0 依赖的 PennyLane 0.43 要求 Python 3.11。使用安装脚本分别部署：

```bash
hardware/install_official_sdks.sh \
  /Users/hronrad/Downloads/kaiwu-1.3.1-cp310-none-any.whl \
  /Users/hronrad/Downloads/wuyue-1.0-py3-none-any.whl
```

五岳云平台使用与通用量子线路相同的移动云 AK/SK，不需要在本项目中增加另一套
Kaiwu 用户凭据：

```bash
WUYUE_ACCESS_KEY_ID=your-access-key
WUYUE_ACCESS_KEY_SECRET=your-secret-key
WUYUE_PHOTONIC_DEVICE_ID=WuYue-QPU-Qboson-1000
```

必须使用移动云提供的 `kaiwu-1.3.1-cp310-none-any.whl`。该构建的
`kw.cim.CIMOptimizer` 接受 `access_key`、`secret_key` 和 `device_id`；不要用同版本号的
公开 PyPI 构建替代。代码和报告均不会输出 AK/SK。

## 6. Kaiwu 模拟退火

```bash
python3 hardware/kaiwu_qsnn_photonic/photonic_qsnn.py \
  --mode train \
  --sampler kaiwu-sa \
  --epochs 20
```

该模式使用官方 `SimulatedAnnealingOptimizer`，用于确认 Kaiwu 模型和 Ising 转换接口。

对已经训练并导出的矩阵执行最小可复现实验：

```bash
hardware/.venv-kaiwu/bin/python \
  hardware/kaiwu_qsnn_photonic/kaiwu_cim_repro.py \
  --mode sa
```

## 7. 相干光 CIM 真机

```bash
python3 hardware/kaiwu_qsnn_photonic/photonic_qsnn.py \
  --mode train \
  --sampler kaiwu-cim \
  --task-name QSNNPhotonicBinary \
  --epochs 5 \
  --batch-size 8
```

该模式使用官方：

```python
CIMOptimizer(...)
PrecisionReducer(...)
BoltzmannMachine.condition_sample(...)
```

每个条件样本都可能产生远程任务，真机训练应先用较小数据量和 epoch 验证额度与延迟。
更经济的实践是本地训练、导出 `photonic_qsnn_ising.npy`，再用 CIM 做负相重采样或推理。

安装五岳平台适配版 Kaiwu SDK 后，对应的单次真机复现入口为：

```bash
hardware/.venv-kaiwu/bin/python \
  hardware/kaiwu_qsnn_photonic/kaiwu_cim_repro.py \
  --mode cim --task-name QSNNPhotonicBinary
```

### 1000 自旋路线

先执行 8 样本 SA 门禁：

```bash
hardware/.venv-kaiwu/bin/python \
  hardware/kaiwu_qsnn_photonic/large_scale_qsnn.py \
  --mode sa --samples 400 --sa-eval-samples 8
```

门禁通过后提交 Qboson-1000：

```bash
hardware/.venv-kaiwu/bin/python \
  hardware/kaiwu_qsnn_photonic/large_scale_qsnn.py \
  --mode cim --samples 400 --sa-eval-samples 8 \
  --task-name LargeQSNN1000Q8
```

首次满规模任务 `2608130MSS0TJIC01AZ4XHUMYTD0QZZL` 已计算成功。CIM 返回 10 个
候选解，目标样本全部判为正确类别 0，最佳 Hamiltonian 为 `-39794`；同一样本 SA
最佳值为 `-37098`。完整报告位于 `outputs/large_scale/large_qsnn_report.json`。

这里的 QSNN 是输入注入、相干传播、耗散吸引和读出向 CIM Ising 网络的结构映射，
不是在光机上直接积分 GKLS/Lindblad 主方程。

## 8. 与超导作品的对应关系

| QSNN 语义 | Quafu 超导作品 | Kaiwu 相干光作品 |
|---|---|---|
| 输入编码 | 单激发站点编码 | 二值 Ising 特征编码 |
| 相干混合 | XY 门传播 | Ising/BM 耦合矩阵 |
| 耗散 | partial transfer + reset 近似 | CIM 耗散收敛与 Boltzmann 采样 |
| 输出 | 两个物理输出 qubit | 两个互斥输出节点 |
| 硬件接口 | OpenQASM + Quafu Task | Kaiwu BM + CIMOptimizer |

## 9. 与 WuYue 通用线路的边界

[WuYueSDK](https://gitee.com/OpenWuYue/WuYueSDK) 用于仓库中的独立通用门模型路线；
`hardware/wuyue_qsnn_native.py` 已完成状态制备、演化门、测量、云模拟器和百花真机提交。
相干光核心求解仍严格通过 Kaiwu SDK，两条路线只共享任务数据与对比指标。

## 10. 外部代码与许可

本目录没有复制 Kaiwu-PyTorch-Plugin 源码，只通过其公开 API 集成。上游仓库使用
Apache-2.0；本作品中的接口名称与调用方式参考了上游 README、`run_rbm.py`、
`full_boltzmann_machine.py`，依赖来源固定写在 `requirements-kaiwu.txt` 中。
