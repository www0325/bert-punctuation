# -*- coding: utf-8 -*-
"""
任务二：长文 >512 token 滑窗优化实测（GPU）。
本脚本在【长文档（数千字）】上对比四种切窗策略：
  A naive_overlap   : L=512, stride=256（50%重叠，基线，最准最慢）
  B no_overlap      : L=512, stride=512（不重叠，最快，边界词缺上下文）
  C seam_minimal    : 不重叠512主块 + 接缝处补64字小窗（优化方案）
  D assignment_baseline: L=400, 只信前300字预测, 从最后一个句号后接续（作业要求的基线方法）
另测 batch 1/4/8/16/32 吞吐加速比。

长文档构造：把 test.jsonl 中连续段落拼接成长文（每篇 ~2000-3000字）。
指标：峰值显存、推理耗时、字符吞吐、标点 micro-F1、句界 F1。
用法：python gpu_longbench.py --model models/bert9_gpu --n_docs 20
"""
import os, json, time, argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def build_long_docs(n_docs=20, min_chars=2000):
    """把连续段落拼成长文档，保留每个字的标签。每篇 >=2000 字。"""
    rows = [json.loads(l) for l in open("data/processed/test.jsonl", encoding="utf-8")]
    docs = []
    cur_chars, cur_labels = [], []
    for r in rows:
        cur_chars.extend(r["chars"])
        cur_labels.extend(r["labels"])
        if len(cur_chars) >= min_chars:
            docs.append({"chars": cur_chars, "labels": cur_labels})
            cur_chars, cur_labels = [], []
        if len(docs) >= n_docs:
            break
    return docs

@torch.no_grad()
def fwd(model, tok, enc_chunk, device):
    ids = [tok.cls_token_id] + enc_chunk + [tok.sep_token_id]
    out = model(input_ids=torch.tensor([ids], device=device),
               attention_mask=torch.tensor([[1]*len(ids)], device=device)).logits[0, 1:-1]
    return out.cpu().numpy()

def predict(model, tok, doc, L, device, mode):
    chars, labels = doc["chars"], doc["labels"]
    enc = tok(chars, is_split_into_words=True, add_special_tokens=False)["input_ids"]
    gold = np.array(labels[:len(enc)])
    votes = np.zeros((len(enc), 9), dtype=np.float32)
    n_fwd = 0

    if mode == "naive_overlap":
        stride = L // 2
        for s in range(0, max(len(enc)-L+1,1), stride):
            e = min(s+L, len(enc))
            votes[s:e] += fwd(model, tok, enc[s:e], device); n_fwd += 1
            if e >= len(enc): break

    elif mode == "no_overlap":
        for s in range(0, len(enc), L):
            e = min(s+L, len(enc))
            votes[s:e] += fwd(model, tok, enc[s:e], device); n_fwd += 1

    elif mode == "seam_minimal":
        seams = list(range(L, len(enc), L))
        for s in range(0, len(enc), L):
            e = min(s+L, len(enc))
            votes[s:e] += fwd(model, tok, enc[s:e], device); n_fwd += 1
        for sm in seams:
            s = max(0, sm-32); e = min(len(enc), sm+32)
            votes[s:e] += fwd(model, tok, enc[s:e], device); n_fwd += 1

    elif mode == "fusion_weighted":
        # 重叠窗 + 中心加权：窗口中心位置权重高，边缘位置权重低
        stride = L // 2
        weight_sum = np.zeros(len(enc), dtype=np.float32)
        for s in range(0, max(len(enc)-L+1,1), stride):
            e = min(s+L, len(enc))
            logits = fwd(model, tok, enc[s:e], device); n_fwd += 1
            # 三角权重：从窗口两端到中心线性递增
            w = np.concatenate([np.linspace(0.3, 1.0, (e-s)//2),
                                np.linspace(1.0, 0.3, (e-s)-(e-s)//2)])
            votes[s:e] += logits * w[:, None]
            weight_sum[s:e] += w
            if e >= len(enc): break
        votes = votes / np.maximum(weight_sum[:, None], 1e-8)

    elif mode == "assignment_baseline":
        # L=400, 只信前300字, 从最后一个句号(。=标签2)后接续
        L_win = 400; trust = 300
        s = 0
        while s < len(enc):
            e = min(s+L_win, len(enc))
            logits = fwd(model, tok, enc[s:e], device); n_fwd += 1
            te = min(s+trust, e)
            votes[s:te] += logits[:te-s]
            pred = logits[:te-s].argmax(-1)
            period_positions = np.where(pred == 2)[0]
            if len(period_positions) > 0 and e < len(enc):
                s = s + int(period_positions[-1]) + 1
            else:
                s = te
    return votes.argmax(-1), gold, n_fwd

def f1_scores(pred, gold):
    tp=fp=fn=0
    for c in range(1,9):
        tp += int(((pred==c)&(gold==c)).sum())
        fp += int(((pred==c)&(gold!=c)).sum())
        fn += int(((pred!=c)&(gold==c)).sum())
    p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
    micro = 2*p*r/(p+r) if p+r else 0
    bp = np.isin(pred,[2,3,4]); bg = np.isin(gold,[2,3,4])
    btp=int((bp&bg).sum()); bfp=int((bp&~bg).sum()); bfn=int((~bp&bg).sum())
    bf = 2*btp/(2*btp+bfp+bfn) if 2*btp+bfp+bfn else 0
    return round(micro,4), round(bf,4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bert9_gpu")
    ap.add_argument("--n_docs", type=int, default=20)
    ap.add_argument("--L", type=int, default=512)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()
    docs = build_long_docs(args.n_docs)
    total_chars = sum(len(d["chars"]) for d in docs)
    print(f"long docs: {len(docs)} | total chars: {total_chars} | avg len: {total_chars//len(docs)} | device: {device}")

    results = {}
    for mode in ["naive_overlap", "no_overlap", "seam_minimal", "fusion_weighted", "assignment_baseline"]:
        if device=="cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.time()
        all_pred, all_gold, n_fwd = [], [], 0
        for d in docs:
            p, g, nf = predict(model, tok, d, args.L, device, mode)
            all_pred.append(p); all_gold.append(g); n_fwd += nf
        if device=="cuda": torch.cuda.synchronize()
        dt = time.time()-t0
        pred = np.concatenate(all_pred); gold = np.concatenate(all_gold)
        mic, bf = f1_scores(pred, gold)
        vram = torch.cuda.max_memory_allocated()/1e9 if device=="cuda" else 0
        results[mode] = {"fwd_passes": n_fwd, "time_s": round(dt,2),
                         "throughput_chars_per_s": round(total_chars/dt,0),
                         "peak_vram_GB": round(vram,2),
                         "micro_f1": mic, "boundary_f1": bf}
        print(f"[{mode}] fwd={n_fwd} time={dt:.1f}s throughput={total_chars/dt:.0f}c/s "
              f"vram={vram:.2f}GB microF1={mic:.4f} boundaryF1={bf:.4f}")

    if device == "cuda":
        bench = docs[0]
        enc = tok(bench["chars"][:args.L], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        print("\nbatch scaling (L=%d):" % args.L)
        bs_res = {}
        for bs in [1,4,8,16,32]:
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0=time.time()
            for _ in range(20):
                ids = torch.tensor([[tok.cls_token_id]+enc+[tok.sep_token_id]]*bs, device=device)
                _ = model(input_ids=ids).logits
            torch.cuda.synchronize(); dt=(time.time()-t0)/20
            vram = torch.cuda.max_memory_allocated()/1e9
            bs_res[bs] = {"time_ms": round(dt*1000,1), "tok_per_s": round(bs*len(enc)/dt,0),
                         "vram_GB": round(vram,2)}
            print(f"  bs={bs}: {dt*1000:.0f}ms/batch | {bs*len(enc)/dt:.0f} tok/s | vram={vram:.2f}GB")
        results["batch_scaling"] = bs_res

    os.makedirs("runs", exist_ok=True)
    json.dump(results, open("runs/gpu_longbench.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved runs/gpu_longbench.json")

if __name__ == "__main__":
    main()
