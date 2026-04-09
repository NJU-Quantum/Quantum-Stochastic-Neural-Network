# LLM测试 Tiny Shakespeare

## 主要文件

- 训练脚本：`tasks/poem_recognition/train_ts_modern_lm.py`
- tokenizer 模块：`tasks/poem_recognition/lm_tokenizer.py`
- 兼容入口：`tasks/poem_recognition/tokenizer.py`
- 训练语料：`tasks/poem_recognition/TS_modern_english.txt`，训练语料是改过的不是原版Tiny Shakespeare, 改动方式在modernize_early_modern_english.py

## 改动

1. `level=word` 不再按空格粗切，而是做了标点分离预分词。
2. 新增 `level=bpe`，采用 BPE 风格子词编码（可调 merges）。
3. tokenizer 完全拆到独立模块，训练脚本只负责训练逻辑，便于维护。

## tokenize的方式 

常用（词级 + 标点分离）：#可能vocab_size比较大

```bash
python tasks/poem_recognition/train_ts_modern_lm.py --level word
```

推荐（BPE 子词）：

```bash
python tasks/poem_recognition/train_ts_modern_lm.py \
  --level bpe \
  --bpe-merges 1500 \
  --bpe-min-pair-freq 2
```

字符级：

```bash
python tasks/poem_recognition/train_ts_modern_lm.py --level char
```

## 输出产物

默认目录：`outputs/poem_recognition/ts_modern_lm/`

- `model.pt`：模型与配置
- `vocab.json`：tokenizer配置（含词表，BPE模式下含 merges）
- `sample_generation.txt`：采样生成文本
- `loss_curve.png`：训练曲线（仅在安装 `matplotlib` 时生成）

## 依赖

- `torch`（必须）
- `matplotlib`（可选，仅用于画 loss 曲线）

