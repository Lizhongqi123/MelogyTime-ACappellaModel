# 岁月弦歌・ACappella 语言模型

# MelogyTime · ACappella Language Model

> **MelogyTime 出品・约 5 亿参数・完全本地运行・隐私友好**
> **By MelogyTime · ~500M parameters · Fully local · Privacy-friendly**

一个从零实现的中文对话语言模型项目。模型名 **ACappella（阿卡贝拉）**，基于 MLA 注意力与细粒度 MoE 架构，配套真实 BPE 分词器、手写 KV Cache 推理与 Tkinter 对话界面。

A from-scratch Chinese conversational language model project. The model is named **ACappella**, built on MLA attention and fine-grained MoE architecture, with a real BPE tokenizer, hand-written KV-cache inference, and a Tkinter chat GUI.



***

## 目录 / Table of Contents



1. [项目简介 / Overview](#1-项目简介--overview)

2. [目录结构 / Project Structure](#2-目录结构--project-structure)

3. [模型架构 / Model Architecture](#3-模型架构--model-architecture)

4. [快速开始 / Quick Start](#4-快速开始--quick-start)

5. [数据格式（JSONL）/ Data Format (JSONL)](#5-数据格式jsonl--data-format-jsonl)

6. [配置说明 / Configuration](#6-配置说明--configuration)

7. [发布政策 / Publishing Policy](#7-发布政策--publishing-policy)

8. [隐私与安全 / Privacy & Security](#8-隐私与安全--privacy--security)

9. [常见问题 / FAQ](#9-常见问题--faq)

10. [许可与再分发 / License & Redistribution](#10-许可与再分发--license--redistribution)



***

## 1. 项目简介 / Overview

**中文**：岁月弦歌是一个完全从零实现的～5 亿参数中文对话语言模型（模型名 ACappella）。它复刻了主流高效 MoE 大模型的经典组件：MLA（多头潜在注意力，KV 压缩 + 解耦 RoPE）、细粒度专家混合（32 路由专家 + 1 共享专家 + Sigmoid 门控 + 无辅助损失偏置）、真实字节级 BPE 分词器，并配有一套手写的 KV Cache 增量推理实现与 Tkinter 对话界面。训练数据是中文日常对话 JSONL，损失只计算在 Assistant 回复上。整个项目（训练、推理、GUI）完全本地运行，唯一的联网行为是首次运行自动下载分词器。

**English**: MelogyTime is a \~500M-parameter Chinese conversational language model (named **ACappella**) implemented entirely from scratch. It reproduces the classic building blocks of modern efficient MoE LLMs: **MLA** (Multi-head Latent Attention with KV compression and decoupled RoPE), **fine-grained MoE** (32 routed experts + 1 shared expert, Sigmoid gating, no auxiliary-loss bias), a **real byte-level BPE tokenizer**, plus a hand-written KV-cache incremental inference engine and a Tkinter chat GUI. Training data is Chinese daily-conversation JSONL, and the loss is computed only on Assistant replies. The whole project — training, inference, and GUI — runs fully locally; the only network activity is an automatic tokenizer download on first run.



***

## 2. 目录结构 / Project Structure



```
岁月弦歌/                                  # 项目根目录 / project root

├── config.py               # 全局配置：模型结构 / 训练 / 推理参数（global config）

├── data\_gen.py             # 对话数据生成脚本：可生成 10,000 条 JSONL

│                           # (dataset generator: produces 10k JSONL samples)

├── tokenizer\_utils.py      # 分词器加载、对话模板、BOS/EOS 定义

│                           # (tokenizer loading, chat template, BOS/EOS)

├── dialogue.jsonl          # 对话数据集 v1（默认训练数据 / default training data）

├── dialogue\_v2.jsonl       # 对话数据集 v2（扩充版 / extended dataset）

├── requirment.txt          # 依赖清单：torch / transformers（dependencies）

├── README.md               # 本文档（this file）

├── ckpt.pt                 # ⚠ 训练检查点（模型+优化器+训练进度）——【不公开】

│                           # ⚠ training checkpoint (model+optimizer+progress) — \[NOT PUBLIC]

├── ckpt\_best.pt            # ⚠ 最优权重备份 ——【不公开】

│                           # ⚠ best-weight backup — \[NOT PUBLIC]

├── gui/

│   ├── \_\_init\_\_.py

│   └── app.py              # Tkinter 对话界面（GUI 入口 / GUI entry）

├── inference/

│   ├── \_\_init\_\_.py

│   └── generate.py         # 推理引擎：手写 KV Cache 增量生成

│                           # (inference: hand-written KV-cache generation)

├── model/

│   ├── \_\_init\_\_.py         # 模型包导出：from model import ACappella

│   ├── melogytime.py       # ACappella 模型主体（Transformer 主干 / main model）

│   ├── block.py            # 单个 Block：RMSNorm + MLA + MelogyTimeMoE

│   ├── mla.py              # MLA 注意力层（KV 压缩 + 解耦 RoPE）

│   └── moe.py              # MelogyTimeMoE 专家层（路由+共享+门控）

├── tokenizer/              # Qwen2.5 字节级 BPE 词表（首次运行自动下载）

│   │                       # (byte-level BPE vocab, auto-downloaded on first run)

│   ├── config.json

│   ├── tokenizer.json

│   ├── tokenizer\_config.json

│   ├── vocab.json

│   ├── merges.txt

│   └── .cache/             # HuggingFace 下载缓存（download cache）

└── train/

&#x20;   ├── \_\_init\_\_.py

&#x20;   ├── dataset.py          # 数据集：加载 JSONL、拼接模板、mask 只保留 Assistant loss

&#x20;   └── train.py            # 训练脚本：断点续训、best 备份、--fresh 重训
```

> **发布说明 / Publishing note**
>
> ：所有 
>
> `__pycache__`
>
> （
>
> `.pyc`
>
>  编译缓存）与所有 
>
> `.pt`
>
>  权重文件
>
> **不在公开范围内**
>
> ，详见
>
> [发布政策](#7-发布政策--publishing-policy)
>
> 。
> All 
>
> `__pycache__`
>
>  (
>
> `.pyc`
>
>  bytecode caches) and all 
>
> `.pt`
>
>  weight files are 
>
> **outside the public scope**
>
>  — see 
>
> [Publishing Policy](#7-发布政策--publishing-policy)
>
> .



***

## 3. 模型架构 / Model Architecture

### 3.1 总体参数 / Overall Parameters



| 项 / Item                | 值 / Value                                            |
| ----------------------- | ---------------------------------------------------- |
| 模型名 / Model             | ACappella                                            |
| 参数量 / Params            | 约 5 亿 / \~500M                                       |
| 词表大小 / Vocab size       | 151,936                                              |
| 隐藏维度 / Hidden dim       | 512                                                  |
| Transformer 层数 / Layers | 12                                                   |
| 注意力头数 / Heads           | 12                                                   |
| MLA 配置                  | qk\_nope = 64 · qk\_rope = 32 · kv\_lora\_rank = 128 |
| MoE 配置                  | 32 路由专家 + 1 共享专家・top\_k = 4・expert\_hidden = 1024    |

### 3.2 MLA 注意力 / Multi-head Latent Attention

**中文**：MLA 将每层的 K/V 先压缩成低秩潜在向量（`kv_lora_rank = 128`）缓存，再上投影还原为各头的 K/V，从而大幅降低 KV Cache 显存 / 内存占用。Query 被拆成「无 RoPE 部分（qk\_nope）」与「解耦 RoPE 部分（qk\_rope）」两路，分别做注意力后相加。缓存中保存的是压缩后的 latent 与**已旋转**的 k\_rope，保证增量解码时位置信息正确。

**English**: MLA compresses each layer's K/V into a low-rank latent vector (`kv_lora_rank = 128`) for caching, then up-projects it back into per-head K/V, greatly reducing KV-cache memory. The Query is split into a RoPE-free part (`qk_nope`) and a decoupled RoPE part (`qk_rope`); the two attention outputs are summed. The cache stores the compressed latent together with the **already-rotated** k\_rope, keeping positional correctness during incremental decoding.

### 3.3 细粒度 MoE / Fine-grained Mixture of Experts

**中文**：`MelogyTimeMoE` 层包含 32 个路由专家与 1 个共享专家。门控采用 Sigmoid 打分，叠加一个可学习的 `routing_bias`（无辅助损失偏置）；每个 token 激活 top\_k 个路由专家 + 全部共享专家，输出加权求和。训练时用 `bincount` 统计各专家负载，以 0.001 的步长微调 bias，起到负载均衡作用。

**English**: The `MelogyTimeMoE` layer has 32 routed experts and 1 shared expert. Gating uses Sigmoid scores plus a learnable `routing_bias` (no auxiliary-loss bias); each token activates the top\_k routed experts plus all shared experts, and outputs are weighted-summed. During training, per-expert load is counted with `bincount` and the bias is nudged at rate 0.001 for load balancing.

### 3.4 分词器 / Tokenizer

**中文**：使用 Qwen2.5 字节级 BPE 分词器（`AutoTokenizer`，`trust_remote_code=True`）。首次运行会从 HuggingFace 自动下载词表文件（非模型权重，约 10–20 MB），下载后完全离线使用。对话模板使用词表中真实存在的 `<|im_start|>` / `<|im_end|>` 控制符，避免特殊符被 BPE 拆碎导致生成停不下来。

**English**: Uses the Qwen2.5 byte-level BPE tokenizer (`AutoTokenizer`, `trust_remote_code=True`). On first run it automatically downloads the vocab files from HuggingFace (tokenizer only, \~10–20 MB, not model weights); afterwards it works fully offline. The chat template uses real `<|im_start|>` / `<|im_end|>` control tokens from the vocab, avoiding the special-token fragmentation that makes generation run away.

### 3.5 推理实现 / Inference

**中文**：手写 KV Cache 增量解码：整段输入一次前向，之后每步只对最后一个 token 前向。增量阶段使用**右下对齐的真实因果掩码**（`is_causal=True` 是左上对齐，在 `kv_len > q_len` 时会误屏蔽大部分缓存）。RoPE 旋转表预计算并覆盖 `max_seq_len + max_new_tokens` 的窗口，超出会被硬截断。

**English**: Hand-written KV-cache incremental decoding: the full prompt is processed in one forward pass, then each step runs a forward on the last token only. The incremental phase uses a **right-bottom-aligned true causal mask** (`is_causal=True` is top-left aligned and wrongly masks most of the cache when `kv_len > q_len`). The RoPE table is precomputed to cover `max_seq_len + max_new_tokens`; longer contexts are hard-truncated.



***

## 4. 快速开始 / Quick Start

### 4.1 环境要求 / Requirements



* Python 3.10+（项目在 3.13 上测试 /tested on 3.13）

* 依赖见 `requirment.txt`：`torch>=2.1`、`transformers>=4.40`

### 4.2 安装与运行 / Install & Run



```
\# 安装依赖 / install dependencies

pip install -r requirment.txt

\# 训练 / train（存在 ckpt.pt 时自动断点续训；resumes automatically if ckpt.pt exists）

python -m train.train

\# 忽略已有检查点，从零训练 / ignore existing checkpoint, train from scratch

python -m train.train --fresh

\# 启动对话界面 / launch the chat GUI

python -m gui.app
```

### 4.3 首次运行（唯一联网行为）/ First Run (the ONLY network activity)

**中文**：首次运行训练或 GUI 时，项目会自动从 HuggingFace 下载 Qwen2.5 字节级 BPE 分词器（约 10–20 MB，仅词表 / 分词器，**不是**模型权重）。这是整个项目**唯一**的联网行为；下载完成后，训练、推理、GUI 全程离线，不会建立任何连接或发送任何数据。

**English**: On the very first run of training or the GUI, the project automatically downloads the Qwen2.5 byte-level BPE tokenizer (\~10–20 MB, vocab/tokenizer only, **not** model weights) from HuggingFace. This is the **only** network activity of the entire project; after the download completes, training, inference, and the GUI all run offline and never open a connection or send any data.

> **注意 / Note**
>
> ：如果你的默认 
>
> `python`
>
>  提示 
>
> `ModuleNotFoundError: No module named 'torch'`
>
> ，请使用安装了 torch 的解释器运行，例如 
>
> `py -3.13`
>
>  或你自己的虚拟环境。
> If your default 
>
> `python`
>
>  reports 
>
> `ModuleNotFoundError: No module named 'torch'`
>
> , run with an interpreter that has torch, e.g. 
>
> `py -3.13`
>
>  or your own virtual environment.

### 4.4 硬件说明 / Hardware

**中文**：CPU 即可运行（较慢）；有 NVIDIA 显卡或服务器集群时训练速度大幅提升。训练规模约 5 亿参数，消费级 GPU 即可覆盖。

**English**: CPU works (slow); an NVIDIA GPU or cluster speeds up training substantially. At \~500M parameters, a consumer GPU is sufficient.



***

## 5. 数据格式（JSONL）/ Data Format (JSONL)

### 5.1 格式规范 / Format Spec

**中文**：训练数据为 JSONL（每行一个 JSON 对象），结构如下：

**English**: Training data is JSONL — one JSON object per line:



```
{"messages": \[{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}]}

{"messages": \[{"role": "user", "content": "介绍一下你自己"}, {"role": "assistant", "content": "我是 ACappella，一个约 5 亿参数的语言模型。"}]}

{"messages": \[{"role": "user", "content": "1+1 等于几？"}, {"role": "assistant", "content": "等于 2。"}]}
```

字段说明 / Field spec：



| 字段 / Field           | 说明 / Description             |
| -------------------- | ---------------------------- |
| `messages`           | 消息数组（必填，至少 2 条）              |
| `messages[].role`    | `"user"` 或 `"assistant"`     |
| `messages[].content` | 文本内容 /plain text             |
| 最后一条消息 /last message | **必须是** `assistant`（否则该行被跳过） |

### 5.2 训练规则 / Training Rules

**中文**：



* 完整对话按模板拼接为文本，模型做标准的「预测下一个 token」语言建模；

* 训练时**只对 Assistant 回复计算损失**：prompt 部分（User 内容与模板符）与尾部 padding 在 `labels` 中置为 `-100` 屏蔽；

* padding 使用真实长度截断，避免因 pad 与 eos 同 id 而误伤答案末尾的 `<|im_end|>`。

**English**:



* The full conversation is joined by the template and trained as standard next-token language modeling;

* **Loss is computed only on Assistant replies**: the prompt part (User content + template tokens) and trailing padding are masked with `-100` in `labels`;

* Padding is masked by real length, so the trailing `<|im_end|>` is not accidentally removed (pad and eos share the same id).

对话模板（`tokenizer_utils.build_prompt`）：

Chat template (`tokenizer_utils.build_prompt`):



```
<|im\_start|>User: 你好

Assistant: 你好！有什么可以帮你的吗？<|im\_end|>Assistant:
```

### 5.3 内置数据集 / Included Datasets

**中文**：



* `dialogue.jsonl` —— 默认训练数据（`config.data_path`），中文日常对话，部分数据为豆包 AI 生成；

* `dialogue_v2.jsonl` —— 扩充版数据集；

* `data_gen.py` —— 数据生成脚本：内置问候、情绪、日常、宠物、爱好、概念、建议、自我介绍、感谢等素材库，随机组合生成 **10,000** 条对话并写入 `dialogue.jsonl`（`random.seed(42)`，可复现）。

**English**:



* `dialogue.jsonl` — the default training set (`config.data_path`), Chinese daily conversations，Part of the data is generated by Doubao AI‌;

* `dialogue_v2.jsonl` — an extended dataset;

* `data_gen.py` — a generator script: built-in pools of greetings, feelings, daily life, pets, hobbies, concepts, advice, self-intro and thanks, randomly combined to produce **10,000** conversations written to `dialogue.jsonl` (`random.seed(42)` for reproducibility).

**中文**：数据加载时自动校验并跳过坏行：缺 `messages`、消息不足 2 条、最后一条不是 assistant、JSON 解析失败，均会打印提示并跳过。

**English**: Loading validates and skips bad lines: missing `messages`, fewer than 2 messages, last message not assistant, or JSON parse failure — each is reported and skipped.



***

## 6. 配置说明 / Configuration

所有配置集中在 `config.py` 的 `Config` 类。All settings live in the `Config` class of `config.py`.



| 分组 / Group    | 参数 / Parameter     | 默认值 / Default         | 说明 / Description       |
| ------------- | ------------------ | --------------------- | ---------------------- |
| 模型结构 /model   | `vocab_size`       | 151936                | 词表大小                   |
|               | `dim`              | 512                   | 隐藏维度                   |
|               | `n_layers`         | 12                    | Transformer 层数         |
|               | `n_heads`          | 12                    | 注意力头数                  |
| MLA           | `qk_nope_head_dim` | 64                    | Query 无 RoPE 维度        |
|               | `qk_rope_head_dim` | 32                    | Query 解耦 RoPE 维度       |
|               | `kv_lora_rank`     | 128                   | KV 压缩秩                 |
| MoE           | `n_routed_experts` | 32                    | 路由专家数                  |
|               | `n_shared_experts` | 1                     | 共享专家数                  |
|               | `top_k`            | 4                     | 每个 token 激活的路由专家数      |
|               | `expert_hidden`    | 1024                  | 专家 FFN 中间维度            |
| 推理 /inference | `max_seq_len`      | 128                   | 上下文窗口                  |
|               | `max_new_tokens`   | 256                   | 最大生成长度                 |
|               | `temperature`      | 0.5                   | 采样温度                   |
|               | `top_p`            | 0.7                   | 核采样阈值                  |
|               | `device`           | `"cpu"`               | 推理设备                   |
| 训练 /training  | `batch_size`       | 10                    | 批大小                    |
|               | `grad_accum`       | 4                     | 梯度累积步数                 |
|               | `lr`               | 1e-4                  | 学习率                    |
|               | `epochs`           | 5                     | 本次运行训练的 epoch 数（续训时累加） |
|               | `data_path`        | `"dialogue.jsonl"`    | 数据文件路径                 |
|               | `ckpt_path`        | `"ckpt.pt"`           | 检查点路径                  |
|               | `tokenizer_name`   | `"F:/岁月弦歌/tokenizer"` | 分词器路径                  |

> **建议 / Recommendation**
>
> ：
>
> `config.py`
>
>  是全局唯一权威配置，改动会影响模型结构与续训兼容性，
>
> **不建议随意修改**
>
> （尤其模型结构参数）。如需调整训练策略，优先修改 
>
> `train/train.py`
>
>  中的超参数。
> `config.py`
>
>  is the single source of truth; changing it affects architecture and resume compatibility — 
>
> **avoid casual edits**
>
>  (especially structure params). For training strategy, prefer tweaking hyperparameters in 
>
> `train/train.py`
>
> .



***

## 7. 发布政策 / Publishing Policy

### 公开（透明、随仓库分发）/ Public (open & transparent, shipped with the repo)



* 全部 Python 源码（`config.py`、`data_gen.py`、`tokenizer_utils.py`、`gui/`、`inference/`、`model/`、`train/`）

* 全局配置与文档（`config.py`、`README.md`）

* 分词器词表文件（`tokenizer/`，公开以保障可复现）

* 对话数据集（`dialogue.jsonl`、`dialogue_v2.jsonl`）

* 依赖清单（`requirment.txt`）

### 不公开（本地保留、不随仓库分发）/ NOT Public (kept local, not shipped)



| 类别 / Category | 内容 / Contents                      | 原因 / Reason                              |
| ------------- | ---------------------------------- | ---------------------------------------- |
| 编译缓存          | 所有 `__pycache__` 目录与 `.pyc` 文件     | 本地编译产物，无源码价值，运行时自动重建                     |
| 权重文件          | 所有 `.pt`（`ckpt.pt`、`ckpt_best.pt`） | 训练产物，体积大（GB 级），含完整模型与优化器状态；是否单独对外提供由作者决定 |

**中文**：对外分发仓库时，请**不要**包含任何 `.pyc` 与 `.pt` 文件；其余文件均可公开。`__pycache__` 可自行删除（运行时会自动重建）。

**English**: When distributing this repo, please **do not** include any `.pyc` or `.pt` files; everything else is public. `__pycache__` may be deleted freely (it is regenerated at runtime).



***

## 8. 隐私与安全 / Privacy & Security

### 8.1 网络访问 / Network Access



* **唯一联网行为**：首次运行自动下载 Qwen2.5 分词器（HuggingFace）。**The only network activity** is the one-time tokenizer download from HuggingFace.

* 下载完成后，训练、推理、GUI **全程零外联**：不建立任何连接、不发送任何数据、无遥测、无后台服务。After that, training/inference/GUI are **fully offline**: no connections, no data sent, no telemetry, no background services.

* 代码不引入 `socket` 等任何外联网络库（分词器下载由 `transformers` 内部完成）；依赖面仅 `torch` / `transformers` / `tkinter` 与标准库。No networking libraries such as `socket` are used (the download is handled internally by `transformers`); the dependency surface is only `torch` / `transformers` / `tkinter` plus the standard library.

### 8.2 数据 / Data



* 对话数据仅保存在本地 JSONL 文件中；**不**上传到任何服务器。Conversation data lives only in local JSONL files; nothing is uploaded.

* 用户输入与模型参数，**未经明确同意，绝不**用于任何外部训练或共享。User inputs and model parameters are **never** used for external training or sharing without explicit consent.

### 8.3 运行时 / Runtime



* 无任何 Agent、不调用底层系统命令、不执行危险操作。No agents, no low-level system commands, no dangerous operations.

* 所有操作均在本地完成，**透明、可审计、可追踪**。Everything runs locally — **transparent, auditable, traceable**.



***

## 9. 常见问题 / FAQ

**Q1：为什么不能移动&#x20;**`ckpt.pt`**&#x20;/&#x20;**`ckpt_best.pt`**？**

训练与推理通过 `config.ckpt_path` 路径加载，移动会导致找不到检查点；`ckpt_best.pt` 的命名也依赖主文件路径（`ckpt_path` 去掉扩展名 + `_best.pt`）。请保持它们位于项目根目录。

Why can't `ckpt.pt` / `ckpt_best.pt` be moved? They are loaded via `config.ckpt_path`; moving breaks loading, and the best-checkpoint name derives from the main path (stem + `_best.pt`). Keep them in the project root.

**Q2：**`.pyc`**&#x20;文件可以删吗？**

可以。`__pycache__` 只是本地编译缓存，运行时会自动重建；本仓库分发时也不包含它们。

Can `.pyc` files be deleted? Yes. `__pycache__` is just a local bytecode cache, regenerated at runtime; this repo does not distribute them either.

**Q3：默认&#x20;**`python`**&#x20;提示没有 torch？**

本机 PATH 上的 `python` 可能与项目运行环境不一致。请用带 torch 的解释器，例如 `py -3.13`（或你的虚拟环境）运行 `python -m train.train` / `python -m gui.app`。

Default `python` says no torch? The `python` on your PATH may differ from the project environment. Run with an interpreter that has torch, e.g. `py -3.13` (or your venv).

**Q4：断点续训是怎么工作的？**

`ckpt.pt` 是完整续训包（`model` / `optimizer` / `epoch` / `step` / `best_loss`），存在时自动从上次位置继续，并恢复 AdamW 动量；`--fresh` 可强制从零开始。旧格式（裸 `state_dict`、无元信息）或结构不匹配的检查点会被跳过，避免在错误对话模板下继续训练出乱码。

How does resume work? `ckpt.pt` is a full resume bundle (`model`/`optimizer`/`epoch`/`step`/`best_loss`) — it resumes automatically and restores AdamW momentum; `--fresh` forces a from-scratch run. Old-format (bare `state_dict`, no metadata) or incompatible checkpoints are skipped to avoid training on the wrong template into garbage output.

**Q5：模型开箱就能用吗？**

不能。仓库内是**未充分训练的骨架 / 演示权重**，需要先自行训练（`python -m train.train`）才能得到可用的对话效果。

Is the model usable out of the box? No. The repo ships an **untrained skeleton/demo weights**; you must train first (`python -m train.train`) to get usable chat quality.



***

## 10. 许可与再分发 / License & Redistribution

**中文**：



* 允许将本模型与源码**转发 / 再分发到 GitHub** 等平台；分发时请遵循[发布政策](#7-发布政策--publishing-policy)（不包含 `.pyc` 与 `.pt`）。

* 本仓库不含任何 Agent、遥测或隐藏网络行为；所有操作透明、可追踪。

* 本项目为学习 / 研究用途，请勿用于违反法律法规或平台规则的内容。

**English**:



* You are allowed to **forward/redistribute** this model and its source to GitHub or similar platforms; follow the [Publishing Policy](#7-发布政策--publishing-policy) (no `.pyc`, no `.pt`).

* This repo contains no agents, telemetry, or hidden network behavior; every operation is transparent and traceable.

* This is a learning/research project. Do not use it for anything violating laws, regulations, or platform rules.
