# -*- coding: utf-8 -*-
"""
GPU 全量微调脚本（T4 16G / A10 24G / 3060 12G 通用）。
- 自动检测 cuda；有卡则 fp16 混合精度，无卡退回 CPU。
- 记录：每步 loss、epoch 时间、峰值显存、吞吐（字符/s）。
用法示例（A10/T4）：
  python gpu_train.py --max_len 256 --batch 16 --epochs 3 --full
  python gpu_train.py --max_len 128 --batch 32 --epochs 2   # 缩比
"""
import os, time, json, argparse, collections
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

PROC_DIR = "data/processed"
RUN_DIR = "runs"
CKPT_DIR = "models/bert9_gpu"
os.makedirs(RUN_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
LABEL_NAMES = ["NONE","COMMA","PERIOD","QUESTION","EXCLAM","SEMICOLON","COLON","ENUM","BOOK"]

def load_rows(split, max_rows):
    rows = [json.loads(l) for l in open(os.path.join(PROC_DIR, f"{split}.jsonl"), encoding="utf-8")]
    if max_rows and len(rows) > max_rows:
        rows = rows[:max_rows]
    return rows

def windowize(rows, tok, max_len, stride):
    ids_all, mask_all, lab_all = [], [], []
    n_chars = 0
    for r in rows:
        enc = tok(r["chars"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        L = len(enc); n_chars += L
        if L == 0: continue
        starts = list(range(0, max(L - max_len + 1, 1), stride))
        for s in starts:
            e = min(s + max_len, L)
            ids = [tok.cls_token_id] + enc[s:e] + [tok.sep_token_id]
            lab = [-100] + r["labels"][s:e] + [-100]
            ids_all.append(ids); mask_all.append([1]*len(ids)); lab_all.append(lab)
            if e >= L: break
    def pad(seqs, p):
        ml = max(len(x) for x in seqs)
        out = torch.full((len(seqs), ml), p, dtype=torch.long)
        for i, x in enumerate(seqs): out[i,:len(x)] = torch.tensor(x)
        return out
    return (pad(ids_all, tok.pad_token_id), pad(mask_all, 0), pad(lab_all, -100)), n_chars

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--max_val", type=int, default=3000)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.full: args.max_train = 10**9; args.max_val = 10**9

    from transformers import AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| gpu:", torch.cuda.get_device_name(0) if device=="cuda" else "-")
    tok = AutoTokenizer.from_pretrained("bert-base-chinese")

    train_rows = load_rows("train", args.max_train)
    val_rows = load_rows("val", args.max_val)
    (tr_ids, tr_mask, tr_lab), tr_chars = windowize(train_rows, tok, args.max_len, args.stride)
    (va_ids, va_mask, va_lab), va_chars = windowize(val_rows, tok, args.max_len, args.stride)
    train_loader = DataLoader(TensorDataset(tr_ids, tr_mask, tr_lab), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(va_ids, va_mask, va_lab), batch_size=args.batch*2)

    cnt = collections.Counter(tr_lab[tr_lab!=-100].tolist())
    freq = np.array([cnt.get(i,1) for i in range(9)], float)
    w = 1/np.sqrt(freq); w = w/w[0]; w = np.minimum(w, 8.0)
    print("class weights:", dict(zip(LABEL_NAMES, w.round(3))))

    model = AutoModelForTokenClassification.from_pretrained("bert-base-chinese", num_labels=9).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1*len(train_loader)*args.epochs),
                                            len(train_loader)*args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float, device=device), ignore_index=-100)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    log = open(os.path.join(RUN_DIR, "gpu_train_log.txt"), "w", encoding="utf-8")
    def say(s): print(s); log.write(s+"\n"); log.flush()
    say(f"config: max_len={args.max_len} batch={args.batch} epochs={args.epochs} lr={args.lr} "
        f"train_windows={len(tr_ids)} train_chars={tr_chars} device={device}")
    t0 = time.time(); step = 0
    for ep in range(args.epochs):
        model.train(); ep_t = time.time(); run=0.; nb=0
        for ids, mask, lab in train_loader:
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda"), dtype=torch.float16):
                out = model(input_ids=ids, attention_mask=mask).logits
                loss = loss_fn(out.view(-1,9), lab.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); nb += 1; step += 1
            if step % 50 == 0:
                say(f"ep{ep+1} step{step} loss={run/nb:.4f}")
        # eval
        model.eval(); correct=tot=0; vt=time.time()
        with torch.no_grad():
            for ids,mask,lab in val_loader:
                ids,mask,lab = ids.to(device),mask.to(device),lab.to(device)
                p = model(input_ids=ids, attention_mask=mask).logits.argmax(-1)
                m = lab!=-100; correct += (p[m]==lab[m]).sum().item(); tot += m.sum().item()
        peak = torch.cuda.max_memory_allocated()/1e9 if device=="cuda" else 0
        say(f"== ep{ep+1}: loss={run/nb:.4f} val_acc={correct/max(tot,1):.4f} "
            f"epoch_time={time.time()-ep_t:.1f}s peak_vram={peak:.2f}GB")
    say(f"total_time={time.time()-t0:.1f}s")
    model.save_pretrained(CKPT_DIR)
    print("saved", CKPT_DIR)

if __name__ == "__main__":
    main()
