# -*- coding: utf-8 -*-
"""
阶段4：测试集评估。
指标：
  - 逐类 P/R/F1（标点类 1..8）
  - 标点事件 micro-F1 / macro-F1（不含 NONE）
  - 句界 F1（PERIOD/QUESTION/EXCLAM = 2/3/4）
  - 字符准确率 vs 全 NONE 基线（评估陷阱自查）
  - 窗策略对比：fixed / slide / center（同一测试段，重切窗聚合）
输出 runs/eval.json
"""
import os, json, random, argparse, collections
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

PROC_DIR = r".\data\processed"
RUN_DIR = r".\runs"
CKPT = r".\models\bert9"
LABEL_NAMES = ["NONE", "COMMA", "PERIOD", "QUESTION", "EXCLAM",
               "SEMICOLON", "COLON", "ENUM", "BOOK"]
BOUNDARY = {2, 3, 4}   # 句界类

def load_test_paras(n=2000):
    rows = []
    with open(os.path.join(PROC_DIR, "test.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    random.seed(42); random.shuffle(rows)
    return rows[:n]

@torch.no_grad()
def predict_paragraph(model, tokenizer, chars, labels, max_len, stride, device, center_only=False):
    enc = tokenizer(chars, is_split_into_words=True, add_special_tokens=False)["input_ids"]
    L = len(enc)
    # 聚合每个位置的预测（投票：中心加权=中心位置权重高）
    pred_votes = np.zeros((L, 9), dtype=np.float32)
    starts = list(range(0, max(L - max_len + 1, 1), stride))
    for s in starts:
        e = min(s + max_len, L)
        ids = [tokenizer.cls_token_id] + enc[s:e] + [tokenizer.sep_token_id]
        attn = [1] * len(ids)
        out = model(input_ids=torch.tensor([ids], device=device),
                    attention_mask=torch.tensor([attn], device=device))
        logits = out.logits[0, 1:-1].cpu().numpy()  # [chunk_len, 9]
        positions = np.arange(s, e)
        if center_only:
            # 只信中央 2/3：丢弃每头每尾 1/6
            cl = e - s
            lo, hi = cl // 6, cl - cl // 6
            mask = np.zeros(cl, dtype=bool); mask[lo:hi] = True
            logits = logits[mask]; positions = positions[mask]
        pred_votes[positions] += logits
        if e >= L: break
    pred = pred_votes.argmax(-1)
    return pred, np.array(labels[:L])

def prf1(pred, gold, n_classes=9):
    res = {}
    for c in range(1, n_classes):   # 只评标点类
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        res[LABEL_NAMES[c]] = {"P": round(p, 4), "R": round(r, 4),
                               "F1": round(f1, 4), "support": tp + fn}
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelForTokenClassification.from_pretrained(CKPT).to(device).eval()
    rows = load_test_paras(args.n)

    out = {}
    for strategy in ["fixed", "slide", "center"]:
        stride = args.max_len if strategy == "fixed" else args.stride
        center = (strategy == "center")
        all_pred, all_gold = [], []
        for r in rows:
            p, g = predict_paragraph(model, tokenizer, r["chars"], r["labels"],
                                     args.max_len, stride, device, center_only=center)
            all_pred.append(p); all_gold.append(g)
        pred = np.concatenate(all_pred); gold = np.concatenate(all_gold)
        per = prf1(pred, gold)
        # micro/macro over punct classes
        tp = sum(v["support"] and 0 for v in [])  # placeholder
        # micro: aggregate tp/fp/fn across classes 1..8
        mtp = mfp = mfn = 0
        f1s = []
        for c in range(1, 9):
            tp = int(((pred == c) & (gold == c)).sum())
            fp = int(((pred == c) & (gold != c)).sum())
            fn = int(((pred != c) & (gold == c)).sum())
            mtp += tp; mfp += fp; mfn += fn
            if tp + fp + fn:
                f1s.append(2*tp/(2*tp+fp+fn))
        micro_p = mtp/(mtp+mfp) if mtp+mfp else 0
        micro_r = mtp/(mtp+mfn) if mtp+mfn else 0
        micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if micro_p+micro_r else 0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        # 句界 F1
        bpred = np.isin(pred, list(BOUNDARY)); bgold = np.isin(gold, list(BOUNDARY))
        tp = int((bpred & bgold).sum()); fp = int((bpred & ~bgold).sum()); fn = int((~bpred & bgold).sum())
        bp = tp/(tp+fp) if tp+fp else 0; br = tp/(tp+fn) if tp+fn else 0
        bf1 = 2*bp*br/(bp+br) if bp+br else 0
        char_acc = float((pred == gold).mean())
        baseline_acc = float((gold == 0).mean())  # 全 NONE 基线
        out[strategy] = {
            "per_class": per, "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "boundary_f1": round(bf1, 4), "boundary_P": round(bp,4), "boundary_R": round(br,4),
            "char_acc": round(char_acc, 4),
            "all_none_baseline_char_acc": round(baseline_acc, 4)}
        print(f"[{strategy}] microF1={micro_f1:.4f} macroF1={macro_f1:.4f} "
              f"boundaryF1={bf1:.4f} charAcc={char_acc:.4f} baseline={baseline_acc:.4f}")

    with open(os.path.join(RUN_DIR, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved runs/eval.json")

if __name__ == "__main__":
    main()
