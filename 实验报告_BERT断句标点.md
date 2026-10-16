# BERT 断句标点：微调、长文滑窗优化与跨域迁移 —— 实验报告

> 课程：01714354《大语言模型与应用实践》（俞敬松），2026 秋
> 所有数字均来自真实运行：本机 Windows CPU（缩比验证）+ A10 GPU（全量训练）+ T4 GPU（长文与跨域评估）。日志见 `runs/`。

## 0. 实验环境

| 项 | 本机（缩比验证） | A10（全量训练） | T4（评估与长文实测） |
|---|---|---|---|
| 系统 | Windows 11 | Linux | Linux |
| GPU | 无（纯 CPU） | NVIDIA A10（24GB） | NVIDIA T4（16GB） |
| Python | 3.13（venv） | 3.11 | 3.11 |
| torch | 2.x CPU | 2.x + CUDA 12 | 2.x + CUDA 12 |
| transformers | 4.x | 4.x | 4.x |
| datasets | — | 通过 HF 镜像下载 | huggingface_hub 直下 jsonl |

关键依赖：`torch`、`transformers`（AutoTokenizer / AutoModelForTokenClassification）、`numpy`、`matplotlib`。HF 镜像 `https://hf-mirror.com`。

## 1. 任务与目标

两个任务：
1. **微调**：用中文 BERT 在带标点语料上微调为 token 分类器，预测无标点文本每个字后应跟什么标点。
2. **长文滑窗优化**：文章超过 512 token 时，设计并实测不同切窗架构的精度与效率。

## 2. 数据与预处理

- **训练/测试语料**：课堂提供的人民日报 1998 上半年语料（GB18030，行式 `日期-版面-文章号-段落号` + 空格分词 + `{注音}`）。
- **清洗**：剥除行号 ID 与 `{注音}`（共 95,087 处），引号/括号/破折号丢弃，过短段（<8 字）丢弃。
- **标签**：每个汉字一个标签，标点挂在它前面那个词的最后一个字上。9 类：NONE / COMMA，/ PERIOD。/ QUESTION？/ EXCLAM！/ SEMICOLON；/ COLON：/ ENUM、/ BOOK《》。
  - 实测语料中〈〉仅 64 对（太稀有），真正书名号是《》13,008 对，故第 9 类用《》。
- **划分（防泄漏，按月份整组）**：1-4 月训练（73,121 段，711 万字）/ 5 月验证（19,325 段）/ 6 月测试（19,035 段）。
- **古文跨域测试集**：HuggingFace `gujilab/chinese-classical-corpus`，14 部正史 + 经传，取 100 条长段落（17,452 字），自带无标点输入/有标点参考配对。

## 3. 方法

- **模型**：`bert-base-chinese`（102M 参数），顶部加 9 类 token 分类头。
- **损失**：类别加权交叉熵，`w_c = 1/sqrt(freq_c)`，归一使 NONE=1，稀有类权重截断 8。
- **训练**：AdamW lr=2e-5，10% warmup，fp16 混合精度。
- **长文切窗策略**（4 种对比）：
  - `naive_overlap`：L=510，stride=255（50% 重叠）
  - `no_overlap`：L=510，stride=510（不重叠）
  - `seam_minimal`：不重叠主块 + 接缝处补 64 字小窗
  - `assignment_baseline`：L=400，只信前 300 字预测，从最后一个句号后接续（作业要求的基线方法）

## 4. 实验结果

### 4.1 训练（T4 GPU，全量数据，3 epoch）

| epoch | loss | val_acc | 耗时 | peak VRAM |
|---|---|---|---|---|
| 1 | 0.342 | 0.960 | 550s | 3.17 GB |
| 2 | 0.187 | 0.965 | 550s | 3.17 GB |
| 3 | 0.150 | 0.967 | 550s | 3.17 GB |

总训练时间 1650s（27.5 分钟），73,829 个训练窗。

### 4.2 CPU 缩比 vs GPU 全量（测试集 slide 窗，fig5）

| 指标 | CPU 缩比（8k 窗，2 epoch） | GPU 全量（73k 窗，3 epoch） |
|---|---|---|
| micro-F1 | 0.660 | **0.7245** |
| macro-F1 | 0.474 | **0.5901** |
| 句界 F1 | 0.717 | **0.7671** |
| 字符 acc | 0.953 | 0.965 |
| 全 NONE 基线 | 0.925 | 0.925 |

GPU 全量比 CPU 缩比 micro-F1 高 6.4 个点，macro-F1 高 11.6 个点。

### 4.3 逐类 F1（GPU 全量，slide 窗，fig1）

| 类别 | P | R | F1 | support |
|---|---|---|---|---|
| PERIOD 。 | 0.753 | 0.780 | 0.766 | 1175 |
| COMMA ， | 0.716 | 0.763 | 0.739 | 3253 |
| ENUM 、 | 0.633 | 0.804 | 0.708 | 769 |
| QUESTION ？ | 0.579 | 0.846 | 0.688 | 26 |
| COLON ： | 0.534 | 0.723 | 0.614 | 130 |
| BOOK 《》 | 0.470 | 0.673 | 0.553 | 162 |
| SEMICOLON ； | 0.466 | 0.303 | 0.367 | 89 |
| EXCLAM ！ | 0.375 | 0.231 | 0.286 | 13 |

高频标点（。，、）学得好；稀有标点（叹号/分号）F1 仍低，属数据稀疏问题。

### 4.4 长文滑窗五策略对比（T4，20 篇长文 × 平均 2077 字，fig2）

| 策略 | 前向次数 | 耗时 | 吞吐 | micro-F1 | 句界 F1 |
|---|---|---|---|---|---|
| naive_overlap | 133 | 4.75s | 8,750 c/s | 0.691 | 0.579 |
| no_overlap | 93 | 2.35s | **17,675 c/s** | 0.702 | 0.587 |
| seam_minimal | 166 | 2.85s | 14,578 c/s | 0.706 | 0.590 |
| fusion_weighted | 133 | 3.71s | 11,190 c/s | 0.693 | 0.586 |
| **assignment_baseline** | 161 | 3.61s | 11,518 c/s | **0.716** | **0.610** |

**负面结果：fusion_weighted 没有超过作业 baseline。** 我尝试的"中心加权 logit 融合"（窗口中心位置权重高、边缘低）micro-F1 只有 0.693，和 naive_overlap 基本持平，远低于 assignment_baseline（0.716）。说明在这个测试集上，简单的三角加权并没有带来额外收益，反而因为加权平均稀释了边缘位置的预测。

作业要求的 assignment_baseline（400 字取前 300、句号后接续）精度最高；no_overlap 最快（17,675 c/s）且精度只低 1.4 个点，工程性价比最高。

### 4.5 T4 batch 缩放（fig3）

| batch | 耗时/batch | 吞吐 | 显存 |
|---|---|---|---|
| 1 | 30 ms | 16,769 tok/s | 0.74 GB |
| 4 | 110 ms | 18,617 tok/s | 1.64 GB |
| 8 | 201 ms | **20,285 tok/s** | 2.87 GB |
| 16 | 429 ms | 19,007 tok/s | 5.33 GB |
| 32 | 882 ms | 18,510 tok/s | 10.24 GB |

T4 16GB 上 bs=8 吞吐峰值，bs=32 显存 10.24GB 不 OOM。

### 4.6 跨域古文迁移（100 条正史，fig4）

| 指标 | 严格位置匹配 | ±5 字宽松匹配 |
|---|---|---|
| micro-F1 | 0.070 | 0.590 |
| 句界 F1 | 0.036 | 0.519 |
| 字符 acc | 0.650 | — |
| 全 NONE 基线字符 acc | 0.812 | — |

模型在古文上字符 acc（0.650）比全不标（0.812）还低，说明它在乱加标点。但宽松匹配 micro-F1=0.59 说明它大致知道在哪里该断句，只是位置偏移几到几十字——典型的现代→古文跨域退化。

### 4.7 跨域现代/古典文学测试（T4，fig6）

从公开文学作品中各取约 5000 字片段，测试模型在新闻以外文体的表现：

| 作品 | 体裁 | 字数 | 标点事件 | 严格 micro-F1 | 句界 F1 | ±5字宽松 micro-F1 |
|---|---|---|---|---|---|---|
| 背影 | 现代散文 | 4,764 | 547 | 0.339 | 0.360 | 0.684 |
| 骆驼祥子 | 现代小说 | 5,188 | 656 | 0.322 | 0.304 | 0.645 |
| 四世同堂 | 现代小说 | 5,127 | 618 | 0.328 | 0.323 | 0.664 |
| 三国演义 | 半文半白 | 4,895 | 941 | 0.483 | 0.418 | 0.752 |
| 红楼梦 | 古典小说 | 5,021 | 739 | 0.407 | 0.381 | 0.710 |

模型在人民日报新闻上 micro-F1=0.72，迁移到现代小说/散文掉到 0.32-0.34，半文半白的三国演义 0.48。宽松 ±5 字匹配到 0.64-0.75，说明模型大致知道在哪里该断句，但精确位置偏差。这证明 0.72 的数字高度依赖人民日报新闻文体，不能直接外推到其他现代文本。

### 4.8 评估陷阱自查

字符准确率 0.965 看着高，但全 NONE 基线就有 0.925。真实水平看标点事件 micro-F1（0.724）。

## 5. 错误分析

- **稀有标点**：叹号（13 个 support）、分号（89 个）样本过少，类别加权也救不回来。
- **窗口边界**：no_overlap 在块接缝处边界词缺上下文，assignment_baseline 用句号对齐缓解。
- **BOOK 类**：书名号《》F1=0.553，比 CPU 缩比（0.36）提升明显，但仍受引号/书名号歧义影响。
- **跨域**：古文虚词节奏与新闻完全不同，模型学到的是现代文语感。

## 6. 复现

```bash
# 环境
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com

# 清洗（已提供清洗好的 jsonl，可跳过）
python scripts/01_clean.py

# GPU 全量训练
python scripts/gpu_train.py --max_len 256 --batch 16 --epochs 3 --full

# 长文滑窗实测
python scripts/gpu_longbench.py --model models/bert9_gpu --n_docs 20

# 测试集评估
python scripts/04_eval.py --n 800

# 出图
python scripts/06_figures.py
```

## 7. 参考文献

1. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. https://arxiv.org/abs/1810.04805
2. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. https://arxiv.org/abs/1706.03762
3. *A Small and Fast BERT for Chinese Medical Punctuation Restoration* (2023). arXiv:2308.12568. https://arxiv.org/abs/2308.12568
4. HuggingFace `bert-base-chinese` 模型卡. https://huggingface.co/bert-base-chinese
5. `gujilab/chinese-classical-corpus`. https://huggingface.co/datasets/gujilab/chinese-classical-corpus

