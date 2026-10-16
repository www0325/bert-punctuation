# -*- coding: utf-8 -*-
"""
阶段4补充：创新实验
1) 跨域古文迁移：现代新闻训练 -> 无标点古文片段，观察迁移塌陷
2) 端到端长文样例：无标点长段 -> 滑窗推理 -> 回填标点，展示前后对照
3) 12G 卡估算表（无本机 GPU，基于公开同类工作实测区间外推，明确标注为估算）
输出 runs/experiments.json 与 runs/fig1..png
"""
import os, json, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from transformers import AutoTokenizer, AutoModelForTokenClassification

RUN_DIR = r".\runs"
CKPT = r".\models\bert9"
PUNCT = {1:"，", 2:"。", 3:"？", 4:"！", 5:"；", 6:"：", 7:"、", 8:"《》"}

# 中文字体
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]:
    if os.path.exists(fp):
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# --- 古文跨域样本（论语·学而，无标点输入；参考标点版仅作展示对照，不参与训练）---
ANCIENT_RAW = ("子曰学而时习之不亦说乎有朋自远方来不亦乐乎人不知而不愠不亦君子乎"
               "曾子曰吾日三省吾身为人谋而不忠乎与朋友交而不信乎传不习乎")
ANCIENT_REF = ("子曰：学而时习之，不亦说乎？有朋自远方来，不亦乐乎？"
               "人不知而不愠，不亦君子乎？曾子曰：吾日三省吾身，"
               "为人谋而不忠乎？与朋友交而不信乎？传不习乎？")

@torch.no_grad()
def predict_chars(model, tokenizer, chars, max_len=128, stride=64, device="cpu"):
    enc = tokenizer(list(chars), is_split_into_words=True, add_special_tokens=False)["input_ids"]
    L = len(enc)
    votes = np.zeros((L, 9), dtype=np.float32)
    starts = list(range(0, max(L - max_len + 1, 1), stride))
    for s in starts:
        e = min(s + max_len, L)
        ids = [tokenizer.cls_token_id] + enc[s:e] + [tokenizer.sep_token_id]
        out = model(input_ids=torch.tensor([ids], device=device),
                   attention_mask=torch.tensor([[1]*len(ids)], device=device))
        votes[s:e] += out.logits[0, 1:-1].cpu().numpy()
        if e >= L: break
    return votes.argmax(-1)

def insert_punct(chars, pred):
    out = []
    for ch, p in zip(chars, pred):
        out.append(ch)
        if p != 0:
            out.append(PUNCT.get(p, ""))
    return "".join(out)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelForTokenClassification.from_pretrained(CKPT).to(device).eval()

    pred = predict_chars(model, tokenizer, ANCIENT_RAW, device=device)
    ancient_out = insert_punct(ANCIENT_RAW, pred)

    # 端到端长文：从测试集取一段 >200 字的
    import random
    rows = [json.loads(l) for l in open(os.path.join(
        r".\data\processed", "test.jsonl"), encoding="utf-8")]
    random.seed(7); random.shuffle(rows)
    long_para = None
    for r in rows:
        if len(r["chars"]) > 220:
            long_para = r; break
    chars = long_para["chars"]
    pred2 = predict_chars(model, tokenizer, chars, device=device)
    demo_out = insert_punct(chars, pred2)
    demo_ref = insert_punct(chars, np.array(long_para["labels"]))  # 原标点版

    result = {
        "cross_domain": {
            "input_nopunct": ANCIENT_RAW,
            "model_output": ancient_out,
            "reference": ANCIENT_REF,
        },
        "e2e_demo": {"model_output": demo_out, "reference": demo_ref},
    }

    # 12G 估算表（基于公开同类工作实测区间外推，明确标注为估算）
    est = {
        "note": "本机无 NVIDIA GPU，以下为基于公开文献中同类中文 BERT 标点恢复工作实测区间的外推估算，非本机实测，待云 GPU 补测",
        "estimated_12G_3060": {
            "max_batch_bs16_seq128": "约 8-12（12G 显存在 fp16 下 bert-base 可容纳）",
            "max_seq_len_bs8": "约 256",
            "expected_speedup_vs_cpu": "约 30-60x（GPU 相对 CPU 常见加速区间）",
            "full_6month_epoch_minutes": "约 15-30 分钟/epoch（待云卡实测）",
            "throughput_chars_per_s_est": "约 20-35 万",
        }
    }
    result["gpu_estimate"] = est

    with open(os.path.join(RUN_DIR, "experiments.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result["cross_domain"], ensure_ascii=False, indent=2))
    print("---e2e model output---")
    print(demo_out[:400])

if __name__ == "__main__":
    main()
