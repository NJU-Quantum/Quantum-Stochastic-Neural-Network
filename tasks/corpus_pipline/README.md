# 中文 Wikipedia 单分块清洗

这个目录提供一个面向单个 `Wikipedia XML bz2` 分块的清洗脚本，适合处理已经下载好的中文维基 dump 分块，例如：

- `zhwiki-pages-articles-part1.xml.bz2`
- `zhwiki-latest-pages-articles1.xml-p1p187712.bz2`

脚本按页面流式读取 `.bz2` 文件，不需要先完整解压，适合大文件单分块处理。

## 输出格式

输出格式选择为段落级 `JSONL`。

每一行是一条样本，结构如下：

```json
{"id":"12#1","title":"数学","text":"数学是研究数量、结构、变化以及空间等概念的一门学科。","source":"zhwiki-pages-articles-part1.xml.bz2"}
```

这样做的原因：

- 适合 LLM 预训练常见的数据读取方式
- 一行一条样本，方便后续切分、去重、合并多个分块
- 保留 `title` 和 `source` 元信息，后续追踪更方便
- 训练时通常直接读取 `text` 字段即可

## 清洗内容

脚本会对每个页面正文做保守清洗：

- 跳过非主名字空间页面和重定向页
- 去空行
- 去 URL
- 去常见 wiki/HTML 噪声，例如注释、`ref` 标签、部分模板和文件链接
- 规范化空白符
- 过滤中文占比过低的段落
- 繁体中文转简体中文

## 依赖

要求：

- Python 3.11

安装依赖：

```bash
pip install -r tasks/corpus_pipline/requirements.txt
```

## 用法

在仓库根目录执行：

```bash
python tasks/corpus_pipline/clean_zhwiki_chunk.py tasks/corpus_pipline/zhwiki-pages-articles-part1.xml.bz2
```

如果你已经在 `tasks/corpus_pipline` 目录下，也可以直接执行：

```bash
python clean_zhwiki_chunk.py zhwiki-pages-articles-part1.xml.bz2
```

默认会生成两个文件：

- `<输入文件名去掉 .xml.bz2>.cleaned.jsonl`
- `<输入文件名去掉 .xml.bz2>.stats.json`

例如：

- `zhwiki-pages-articles-part1.cleaned.jsonl`
- `zhwiki-pages-articles-part1.stats.json`

## 常用参数

```bash
python clean_zhwiki_chunk.py INPUT_XML_BZ2 \
  --output output/zhwiki_part1.cleaned.jsonl \
  --stats output/zhwiki_part1.stats.json \
  --min-chinese-ratio 0.6 \
  --min-chars 20 \
  --top-k 100
```

参数说明：

- `input`：单个 Wikipedia XML bz2 分块路径
- `--output`：输出 `JSONL` 路径
- `--stats`：输出统计文件路径
- `--min-chinese-ratio`：段落最低中文字符占比，默认 `0.60`
- `--min-chars`：段落最短长度，默认 `20`
- `--top-k`：`stats.json` 里保留多少个高频字符，默认 `100`
- `--max-pages`：只处理前 N 个页面，适合快速验证
- `--log-every`：每处理多少个页面打印一次进度，默认 `1000`

## stats.json 内容

`stats.json` 会包含：

- 页面数
- 保留下来的页面数
- 段落数
- 字符数
- 不同字符数（`vocab_size`）
- 平均段落长度
- 高频字符统计
- 过滤参数和丢弃计数

## 快速验证

先跑一个小样本确认流程：

```bash
python clean_zhwiki_chunk.py zhwiki-pages-articles-part1.xml.bz2 --max-pages 100
```

确认输出正常后，再去掉 `--max-pages` 跑完整个单分块。

## 适合后续预训练的组织方式

如果你后续会处理多个分块，建议：

- 每个分块单独输出一个 `.cleaned.jsonl`
- 每个分块单独保存一个 `.stats.json`
- 最后再做多分块合并、去重和 tokenizer 统计

这样更容易追踪问题，也便于断点续跑。

---

## 更小语料：中文诗词

如果你还想准备一个更小、更干净的中文语料，可以直接用本目录的：

- `prepare_chinese_poetry_corpus.py`

这个脚本适合把本地诗词数据集整理成预训练可用的段落级 `JSONL`，支持：

- 单个 `json`
- 单个 `jsonl`
- 单个 `txt`
- 一个目录下的多份 `json/jsonl/txt`

### 支持的常见诗词数据格式

脚本会尽量兼容常见字段：

- `title`
- `rhythmic`
- `author`
- `paragraphs`
- `content`
- `text`
- `body`

也兼容 `chinese-poetry/chinese-poetry` 这类古诗词仓库常见的对象结构，例如 `poet.tang.0.json`、`ci.song.0.json`。

例如下面这些都可以：

```json
[
  {
    "title": "静夜思",
    "author": "李白",
    "paragraphs": ["床前明月光", "疑是地上霜", "举头望明月", "低头思故乡"]
  }
]
```

或：

```json
{"title":"声声慢","author":"李清照","content":"寻寻觅觅，冷冷清清。\n凄凄惨惨戚戚。"}
```

### 诗词语料输出格式

默认输出仍然是段落级 `JSONL`，每行一条样本：

```json
{"id":"poems.json:1#1","title":"静夜思","author":"李白","text":"床前明月光","source":"poems.json","genre":"chinese_poetry"}
```

这对小语料预训练也比较合适：

- 可以按句训练
- 也可以用 `--join-paragraphs` 按整首诗训练
- 和 zhwiki 输出风格一致，后续合并方便

### 用法

单文件：

```bash
python prepare_chinese_poetry_corpus.py poems.json
```

如果你使用 `chinese-poetry/chinese-poetry` 仓库里的 `poet.tang.0.json`：

```bash
python prepare_chinese_poetry_corpus.py poet.tang.0.json \
  --output poet.tang.0.cleaned.jsonl \
  --stats poet.tang.0.stats.json
```

如果你希望一首诗作为一条训练样本，而不是一句一条：

```bash
python prepare_chinese_poetry_corpus.py poet.tang.0.json \
  --join-paragraphs \
  --output poet.tang.0.joined.cleaned.jsonl \
  --stats poet.tang.0.joined.stats.json
```

目录批量处理：

```bash
python prepare_chinese_poetry_corpus.py data/chinese_poetry
```

按整首诗合并成一个样本：

```bash
python prepare_chinese_poetry_corpus.py poems.json --join-paragraphs
```

自定义输出：

```bash
python prepare_chinese_poetry_corpus.py poems.json \
  --output output/poetry.cleaned.jsonl \
  --stats output/poetry.stats.json \
  --min-chinese-ratio 0.8 \
  --min-chars 4
```

### 什么时候适合用诗词小语料

如果你的目标是：

- 先快速验证 tokenizer / data pipeline
- 做一个很小的中文预训练玩具集
- 训练更偏中文韵律、短句、古典文本风格的模型

那诗词语料会比 zhwiki 更小、更容易收敛，也更便于做快速实验。

---

## 二次清洗：去低频字样本

如果第一次清洗后你觉得 `vocab_size` 还是偏大，可以再做一次字符频次过滤：

- 先统计整份 `JSONL` 语料里正文 `text` 的字符频率
- 取最低频的 `bottom-k` 个字符作为 `low_frequency_characters`
- 只要一条样本里包含这些低频字，就整条丢掉

这对诗词语料特别有用，因为它能进一步去掉包含生僻字、异体字、罕见用字的诗句或整首诗。

脚本：

- `filter_low_frequency_chars.py`

### 用法

按句版做二次清洗：

```bash
python filter_low_frequency_chars.py tang_selected.cleaned.jsonl \
  --bottom-k 1000 \
  --output tang_selected.lf1000.filtered.jsonl \
  --stats tang_selected.lf1000.stats.json
```

整首版做二次清洗：

```bash
python filter_low_frequency_chars.py tang_selected.joined.cleaned.jsonl \
  --bottom-k 1000 \
  --output tang_selected.joined.lf1000.filtered.jsonl \
  --stats tang_selected.joined.lf1000.stats.json
```

### 输出统计

二次清洗的 `stats.json` 会包含：

- `vocab_size_before`
- `vocab_size_after`
- `records_seen`
- `records_kept`
- `records_dropped`
- `low_frequency_characters`
- `high_frequency_characters_after`

默认只统计汉字，不把空白符算进去。
