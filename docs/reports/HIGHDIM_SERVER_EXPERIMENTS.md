# QSNN-QGAN 高维服务器实验说明

## 实验矩阵

本实验只比较相同 PQC 生成器下的两种判别器：QSNN full 与 VQC。

| 名义数据维度 | 数据表示 | 量子态维度 | 量子比特数 | 图像恢复方式 |
|---:|---|---:|---:|---|
| 128 | 冻结 Probability Autoencoder | 128 | 7 | AE Decoder |
| 256 | 冻结 Probability Autoencoder | 256 | 8 | AE Decoder |
| 784 | 原始 28×28 像素，尾部零填充 | 1024 | 10 | 截取前 784 维并重排 |

784 不是 2 的整数次幂，无法直接作为振幅编码量子态的 Hilbert 空间维度。这里将归一化后的 784 个像素无损嵌入 1024 维状态的前 784 个基态，后 240 个基态初始概率严格为零。训练指标会记录 `padding_mass_real_input` 与 `padding_mass_fake_input`，生成器损失还包含 `padding_mass_penalty`，防止生成器把概率藏在不可见的填充态中。因此报告中的“784维”表示保留全部原始像素信息，而不是把图像降维或缩放到 1024 个像素。

## 高维实现

正式高维配置启用 `statevector_training: true`。真实样本和 PQC 生成样本都是纯态；在当前 QSNN 跳跃拓扑中，输入子空间在耗散时保持为次归一纯态，real/fake 输出节点只累计两个标量概率。因此训练可以直接演化状态向量，而不必在每一步构造 `N×N` 密度矩阵。

这一路径保留 QSNN 的相干演化、结构化耗散、输出泄漏和 Trace-Z 目标。小维单元测试会把它与原密度矩阵实现逐项对照。评估阶段的经验混合态 fidelity、trace distance 与 purity 使用低秩公式计算，矩阵分解规模由评估 batch 决定，而不是由 1024 维 Hilbert 空间决定。

## 服务器准备

在仓库根目录执行：

```bash
conda activate qsnn
python -c "import torch, yaml; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -m unittest discover -s tests -p "test_qgan_*.py" -v
```

预期 CUDA 可用并显示 4 张 GPU。数据默认位于仓库的 `datasets/`，输出位于 `outputs/`，这两个目录均不会提交到 Git。

## 先做隔离 smoke test

以下命令使用独立的 Autoencoder 和 QGAN 输出目录，不会污染正式 checkpoint：

```bash
python scripts/run_highdim_comparison.py \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --dimensions 128 256 784 \
  --ae-epochs 1 \
  --qgan-epochs 1 \
  --samples-per-class 8 \
  --max-steps-per-epoch 1 \
  --autoencoder-root outputs/autoencoder/highdim_smoke \
  --output-root outputs/qgan/highdim_smoke \
  --download
```

成功标准：6 个 QGAN 子任务均生成 `metrics.csv`、`checkpoint_latest.pt` 和生成图，`runner_status.json` 最终为 `completed`，且所有数值有限。

## 正式运行

```bash
python scripts/run_highdim_comparison.py \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --dimensions 128 256 784 \
  --download
```

执行顺序如下：

1. 若缺少 AE128/AE256 checkpoint，先在空闲 GPU 上并行训练两个 Autoencoder。
2. Autoencoder 完成后，将 6 个 QGAN 任务放入 GPU 队列；每张 GPU 同时只运行一个任务。
3. 每个维度完成后生成 QSNN/VQC 汇总表、训练曲线和生成样本对照图。

关键输出：

```text
outputs/qgan/highdim_comparison/
├── runner_status.json
├── logs/
├── dim128/
│   ├── qsnn_full/
│   ├── vqc/
│   ├── final_summary.csv
│   └── training_curves.png
├── dim256/
└── dim784/
```

## 中断后继续

如果 QGAN 阶段因服务器重启中断，保留输出目录并执行：

```bash
python scripts/run_highdim_comparison.py \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --dimensions 128 256 784 \
  --resume
```

脚本只会从存在 `metrics.csv` 与 `checkpoint_latest.pt` 的任务继续；遇到不完整且无法确认的目录会停止，避免覆盖正式结果。

runner 对会话断开有防护：训练子进程运行在独立会话中（`start_new_session`），终端或 SSH 断开时 runner 捕获 SIGHUP 并把输出转入 `logs/runner.log` 继续运行，不会再出现整组任务被 SIGHUP 杀死、`runner_status.json` 停留在 `running` 的情况（2026-07-16 dim256 中断的原因，见 `Configs_结果核查与性能报告_20260727.md`）。收到 SIGTERM/Ctrl+C 时会终止子进程并把状态写为 `interrupted`。长时间运行仍建议配合 `nohup` 或 `tmux` 启动。Autoencoder 阶段目前以 `checkpoint_best.pt` 为完成标志，不会把一次短 smoke run误当作正式模型，因此 smoke test 必须使用单独的 `--autoencoder-root`。

## 单任务调试

也可以只运行某个模型：

```bash
python scripts/train_autoencoder.py --config configs/autoencoder_mnist0_128.yaml --device cuda:0 --download
python scripts/train_qgan.py --config configs/mnist0_ae128_qsnn.yaml --device cuda:0
python scripts/train_qgan.py --config configs/mnist0_full784_vqc.yaml --device cuda:1 --download
```

正式比较时，同一维度的 QSNN 与 VQC 必须使用同一个 Autoencoder checkpoint、样本数、随机种子和生成器配置。784 维实验不使用 Autoencoder。
