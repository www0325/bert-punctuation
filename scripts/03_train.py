# -*- coding: utf-8 -*-
"""
阶段3：bert-base-chinese 9类断句标签微调（CPU 缩比版）。
- 类别加权交叉熵：w_c = 1/sqrt(freq_c)，归一使 NONE=1，稀有类权重截断到 8
- 日志写入 runs/train_log.txt
用法：
  python 03_train.py --epochs 2 --batch 16 --lr 2e-5
"""
import os, time, json, argparse, collections
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

WIN_DIR = r".\data\windows"
RUN_DIR = r".\runs"
CKPT_DIR = r".\models\bert9"
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

LABEL_NAMES = ["NONE", "COMMA", "PERIOD", "QUESTION", "EXCLAM",
               "SEMICOLON", "COLON", "ENUM", "BOOK"]

def load(split):
    d = torch.load(os.path.join(WIN_DIR, f"{split}.pt"), weights_only=False)
    return TensorDataset(d["input_ids"], d["attention_mask"], d["labels"])

def class_weights(labels_t, device):
    cnt = collections.Counter(labels_t[labels_t != -100].tolist())
    freqs = np.array([cnt.get(i, 1) for i in range(9)], dtype=float)
    w = 1.0 / np.sqrt(freqs)
    w = w / w[0]            # NONE -> 1
    w = np.minimum(w, 8.0)  # 截断
    print("class weights:", {LABEL_NAMES[i]: round(float(w[i]), 3) for i in range(9)})
    return torch.tensor(w, dtype=torch.float, device=device)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for ids, mask, lab in loader:
        ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        pred = logits.argmax(-1)
        m = lab != -100
        correct += (pred[m] == lab[m]).sum().item()
        total += m.sum().item()
    return correct / max(total, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_ds, val_ds = load("train"), load("val")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2)

    weights = class_weights(train_ds.tensors[2], device)
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-chinese", num_labels=9).to(device)
    # 统计可训练参数
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

    log = open(os.path.join(RUN_DIR, "train_log.txt"), "w", encoding="utf-8")
    def say(s):
        print(s); log.write(s + "\n"); log.flush()

    say(f"config: epochs={args.epochs} batch={args.batch} lr={args.lr} "
        f"train_windows={len(train_ds)} val_windows={len(val_ds)} device={device}")
    t0 = time.time()
    global_step = 0
    for ep in range(args.epochs):
        model.train()
        ep_t = time.time()
        run_loss, nb = 0.0, 0
        for ids, mask, lab in train_loader:
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            opt.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.view(-1, 9), lab.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run_loss += loss.item(); nb += 1; global_step += 1
            if global_step % 100 == 0:
                say(f"epoch{ep+1} step{global_step} loss={run_loss/nb:.4f}")
        val_acc = evaluate(model, val_loader, device)
        say(f"== epoch {ep+1} done: avg_loss={run_loss/nb:.4f} val_char_acc={val_acc:.4f} "
            f"epoch_time={time.time()-ep_t:.1f}s")
    say(f"total_train_time={time.time()-t0:.1f}s")
    model.save_pretrained(CKPT_DIR)
    print("saved to", CKPT_DIR)

if __name__ == "__main__":
    main()
