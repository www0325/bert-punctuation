# bert-punctuation

中文 BERT 标点恢复（断句标点）微调 + 长文 >512 token 滑窗高效推理。

基于预训练 `bert-base-chinese`，在带标点新闻语料上微调为 token 分类器：给定无标点中文字序列，预测每个字后应跟什么标点。同时针对 BERT 512 token 上限，设计并实测了长文档滑窗切分的几种工程方案。

## 目录结构

```
├── scripts/                  # 全部可复现脚本（按编号顺序执行）
│   ├── 00_census.py          # 语料标点普查：统计各类标点出现次数，决定标签集
│   ├── 01_clean.py           # 清洗原始语料：剥行号/注音/特殊符号，输出 jsonl
│   ├── 02_dataset.py          # 把 jsonl 转成 BERT token 分类格式，按月划分 train/val/test
│   ├── 03_train.py            # CPU 缩比训练（本机无 GPU 时跑通流程）
│   ├── 04_eval.py             # 测试集评估：micro/macro-F1、句界 F1、逐类 F1
│   ├── 05_experiments.py      # CPU 侧三种短窗策略对比（fixed/slide/center）
│   ├── 06_figures.py          # 出图（读 runs/ 下 JSON 生成报告图）
│   ├── gpu_train.py           # GPU 全量微调（A10/T4），支持 --full / --batch / --epochs
│   ├── gpu_longbench.py       # GPU 长文滑窗五策略对比 + batch 缩放实测
│   ├── cross_domain_modern.py # 现代/古典文学跨域测试
│   └── ancient_eval.py        # 古文跨域评估（gujilab 语料，严格+宽松匹配）
├── data/
│   └── processed/             # 清洗后的 train/val/test.jsonl + stats.json（gitignore）
├── runs/                      # 全部实验日志与 JSON 结果
│   ├── gpu_results/           # A10 全量训练日志 + eval.json
│   ├── t4_results/            # T4 长文滑窗 + 古文跨域结果
│   └── fig*.png               # 报告用图
├── models/                    # 训练好的权重（gitignore）
├── 实验报告_BERT断句标点.md    # 完整实验报告
├── requirements.txt
└── .gitignore
```

## 快速开始

```bash
# 1. 安装依赖（建议 venv）
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像

# 2. 准备数据（原始语料由课堂提供，解压后放 data/raw/）
python scripts/00_census.py        # 先看标点分布
python scripts/01_clean.py        # 清洗
python scripts/02_dataset.py       # 切分

# 3. GPU 全量训练（约 27 分钟 on A10/T4）
python scripts/gpu_train.py --max_len 256 --batch 16 --epochs 3 --full

# 4. 测试集评估
python scripts/04_eval.py --n 800

# 5. 长文滑窗实测
python scripts/gpu_longbench.py --model models/bert9_gpu --n_docs 20

# 6. 古文跨域测试
python scripts/ancient_eval.py --model models/bert9_gpu --n 100

# 7. 出图
python scripts/06_figures.py
```

## 方法概览

- **标签**：9 类（NONE / 逗号，/ 句号。/ 问号？/ 叹号！/ 分号；/ 冒号：/ 顿号、/ 书名号《》）
- **划分**：按月份整组划分防泄漏（1-4月训练 / 5月验证 / 6月测试）
- **损失**：类别加权交叉熵 `w_c = 1/sqrt(freq_c)`
- **长文策略**：naive_overlap / no_overlap / seam_minimal / assignment_baseline（400字取前300、句号后接续）

## 主要结果

| 指标 | 数值 |
|---|---|
| GPU 全量 micro-F1 | 0.7245 |
| GPU 全量句界 F1 | 0.7671 |
| 长文 assignment_baseline F1 | 0.716（最准） |
| 长文 no_overlap 吞吐 | 16,752 字/s（最快） |
| T4 bs=32 显存 | 10.24 GB |
| 古文严格 micro-F1 | 0.07（±5字宽松 0.59） |

详见 `实验报告_BERT断句标点.md`。

## 参考

- BERT: https://arxiv.org/abs/1810.04805
- Transformer: https://arxiv.org/abs/1706.03762
- 中文医疗标点 BERT: https://arxiv.org/abs/2308.12568
- bert-base-chinese: https://huggingface.co/bert-base-chinese
- 古文语料: https://huggingface.co/datasets/gujilab/chinese-classical-corpus
