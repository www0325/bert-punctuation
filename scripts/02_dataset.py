# -*- coding: utf-8 -*-
"""
阶段2：把清洗后的 char 序列切成训练窗。
输出 data/windows/{split}.pt：
  input_ids [N, L], attention_mask [N, L], labels [N, L]
窗口策略（创新点3）：
  fixed : 不重叠切块
  slide : 滑窗，stride = L//2
  center: 与 slide 同切法，评估时只信中央 region（由 04_eval 实现）
用法：
  python 02_dataset.py --max_len 128 --max_train 8000 --max_val 2000 --max_test 2000
"""
import os, json, argparse, random
import torch
from transformers import AutoTokenizer

PROC_DIR = r".\data\processed"
OUT_DIR = r".\data\windows"
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

def load_split(split, max_paras):
    path = os.path.join(PROC_DIR, f"{split}.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    random.shuffle(rows)
    return rows[:max_paras]

def make_windows(rows, tokenizer, max_len, stride):
    """对每段 char 序列滑窗，[CLS] x L-2 [SEP]。label=-100 给特殊符号位。"""
    all_ids, all_mask, all_lab = [], [], []
    for r in rows:
        chars, labs = r["chars"], r["labels"]
        enc = tokenizer(chars, is_split_into_words=True,
                        add_special_tokens=False)["input_ids"]
        L = len(enc)
        if L == 0:
            continue
        # 短句直接成一个窗（pad 由 collate 做）
        starts = list(range(0, max(L - max_len + 1, 1), stride))
        for s in starts:
            e = min(s + max_len, L)
            chunk_ids = enc[s:e]
            chunk_lab = labs[s:e]
            ids = [tokenizer.cls_token_id] + chunk_ids + [tokenizer.sep_token_id]
            lab = [-100] + chunk_lab + [-100]
            mask = [1] * len(ids)
            all_ids.append(ids)
            all_mask.append(mask)
            all_lab.append(lab)
            if e >= L:
                break
    # pad 到等长
    def pad(seqs, pad_id):
        ml = max(len(x) for x in seqs)
        out = torch.full((len(seqs), ml), pad_id, dtype=torch.long)
        for i, x in enumerate(seqs):
            out[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        return out
    input_ids = pad(all_ids, tokenizer.pad_token_id)
    attention_mask = pad(all_mask, 0)
    labels = pad(all_lab, -100)
    return input_ids, attention_mask, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--max_train", type=int, default=8000)
    ap.add_argument("--max_val", type=int, default=2000)
    ap.add_argument("--max_test", type=int, default=2000)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    for split, n in [("train", args.max_train), ("val", args.max_val), ("test", args.max_test)]:
        rows = load_split(split, n)
        ids, mask, lab = make_windows(rows, tokenizer, args.max_len, args.stride)
        p = os.path.join(OUT_DIR, f"{split}.pt")
        torch.save({"input_ids": ids, "attention_mask": mask, "labels": lab,
                    "max_len": args.max_len, "stride": args.stride,
                    "n_paras": len(rows)}, p)
        print(f"{split}: paras={len(rows)} windows={ids.shape[0]} shape={tuple(ids.shape)}")

if __name__ == "__main__":
    main()
