# WuYue SDK 通用量子计算机方案

本目录实现一个可训练的变分量子分类器（VQC），用于与 Kaiwu 相干光 QSNN-Ising 路线进行跨技术验证。线路完全由 WuYue SDK 的 `QuantumCircuit` 构建，使用 `RY`、`RZ`、`CX` 和测量门；本地精确结果由 WuYue `Backend` 全振幅模拟器复核，云端通过 `Runner` 提交。

## 算法结构

- 数据：与相干光模型一致的二维内外圆环二分类。
- 物理容器：9 qubit；有效数据比特为 `q[2]`、`q[3]`、`q[4]`。
- 拓扑：仅使用 Baihua 已接受的 `2-3`、`3-4` 双比特边。
- 编码：将 $x$、$y$、$r^2$ 和 $xy$ 写入参数化旋转角。
- 变分层：两组局域旋转与链式 CNOT 纠缠。
- 读出：测量 `q[4]`，其激发概率作为类别 1 概率。
- 训练：Nelder-Mead 最小化二元交叉熵；训练后用 WuYue 全振幅后端逐点核验。

该方案属于通用门模型量子算法，因为开发者显式构造了数据编码、参数化量子线路、纠缠门、测量、损失函数和经典优化闭环。它不是 Lindblad QSNN，也不声称具有可控耗散。

## 运行顺序

```bash
hardware/.venv-cloud/bin/python hardware/wuyue_vqc/wuyue_vqc.py \
  --mode train --maxiter 120

hardware/.venv-cloud/bin/python hardware/wuyue_vqc/wuyue_vqc.py \
  --mode cloud-simulator --shots 1024 --timeout 600

hardware/.venv-cloud/bin/python hardware/wuyue_vqc/wuyue_vqc.py \
  --mode baihua --shots 1024 --timeout 0
```

异步任务使用 `poll` 模式查询：

```bash
hardware/.venv-cloud/bin/python hardware/wuyue_vqc/wuyue_vqc.py \
  --mode poll --task-id TASK_ID --device-id WuYue-QPU-Baihua --timeout 600
```

凭据只从仓库根目录 `.env` 读取，不写入报告。
