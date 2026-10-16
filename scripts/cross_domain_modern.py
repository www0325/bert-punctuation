# -*- coding: utf-8 -*-
"""
现代跨域测试：在非人民日报的现代中文文本上评估标点模型。
用法：
  python cross_domain_modern.py --model models/bert9_gpu --text some_text.txt
如果不传 --text，自动下载一个公开中文散文样本。
"""
import os, json, re, argparse, urllib.request
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# 标点映射（与训练时一致）
PUNCT_MAP = {",":1, "。":2, "？":3, "！":4, "；":5, "：":6, "、":7}
LABELS = ["NONE","COMMA","PERIOD","QUESTION","EXCLAM","SEMICOLON","COLON","ENUM","BOOK"]

def load_text(path=None):
    if path and os.path.exists(path):
        text = open(path, encoding="utf-8").read()
    else:
        # 下载一个公开中文散文样本（朱自清《背影》，公有领域）
        url = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E6%9D%82%E6%96%87/beiying.json"
        try:
            text = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
            import json as _j
            data = _j.loads(text)
            text = data[0].get("paragraphs", [data[0].get("content","")])[0] if isinstance(data, list) else str(data)
        except Exception as e:
            print(f"下载失败: {e}，使用内置样本")
            text = "我们在田野散步。我，我的母亲，我的妻子和儿子。母亲本不愿出来的。她老了，身体不好，走远一点就觉得累。我说，正因为如此，才应该多走走。母亲信服地点点头，便去拿外套。她现在很听我的话，就像我小时候很听她的话一样。"
    # 只保留中文和常见标点
    text = re.sub(r"[^\u4e00-\u9fff，。？！；：、《》]", "", text)
    return text

def text_to_chars_labels(text):
    chars, labels = [], []
    for i, ch in enumerate(text):
        if ch in PUNCT_MAP:
            if chars:
                labels[-1] = PUNCT_MAP[ch]
            continue
        chars.append(ch)
        labels.append(0)
    return chars, labels

@torch.no_grad()
def predict_long(model, tok, chars, device, L=510):
    enc = tok(chars, is_split_into_words=True, add_special_tokens=False)["input_ids"]
    votes = np.zeros((len(enc), 9), dtype=np.float32)
    for s in range(0, len(enc), L):
        e = min(s+L, len(enc))
        ids = [tok.cls_token_id] + enc[s:e] + [tok.sep_token_id]
        logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, 1:-1]
        votes[s:e] += logits.cpu().numpy()
    return votes.argmax(-1)

def f1(pred, gold):
    tp=fp=fn=0
    for c in range(1,9):
        tp += int(((pred==c)&(gold==c)).sum())
        fp += int(((pred==c)&(gold!=c)).sum())
        fn += int(((pred!=c)&(gold==c)).sum())
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    micro = 2*p*r/(p+r) if p+r else 0
    bp = np.isin(pred,[2,3,4]); bg = np.isin(gold,[2,3,4])
    btp=int((bp&bg).sum()); bfp=int((bp&~bg).sum()); bfn=int((~bp&bg).sum())
    bf = 2*btp/(2*btp+bfp+bfn) if 2*btp+bfp+bfn else 0
    return round(micro,4), round(bf,4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bert9_gpu")
    ap.add_argument("--text", default=None)
    ap.add_argument("--out", default="runs/modern_cross_domain.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()

    text = load_text(args.text)
    chars, labels = text_to_chars_labels(text)
    gold = np.array(labels)
    print(f"text: {len(chars)} chars | {sum(1 for l in labels if l>0)} punctuation events")

    pred = predict_long(model, tok, chars, device)
    micro, bf = f1(pred, gold)

    # 宽松匹配 ±5 字
    loose_tp=loose_fp=loose_fn=0
    gold_positions = set((i,l) for i,l in enumerate(labels) if l>0)
    pred_positions = set((i,int(pred[i])) for i in range(len(pred)) if pred[i]>0)
    for gi, gl in gold_positions:
        matched = False
        for pi, pl in pred_positions:
            if abs(gi-pi)<=5 and gl==pl:
                matched = True; break
        if matched: loose_tp += 1
        else: loose_fn += 1
    loose_fp = len(pred_positions) - loose_tp
    lp = loose_tp/(loose_tp+loose_fp) if loose_tp+loose_fp else 0
    lr = loose_tp/(loose_tp+loose_fn) if loose_tp+loose_fn else 0
    loose_micro = 2*lp*lr/(lp+lr) if lp+lr else 0

    result = {
        "n_chars": len(chars),
        "n_punct_events": sum(1 for l in labels if l>0),
        "micro_f1": micro,
        "boundary_f1": bf,
        "loose_micro_f1": round(loose_micro,4),
        "source": args.text or "builtin_sample"
    }
    os.makedirs("runs", exist_ok=True)
    json.dump(result, open(args.out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"micro-F1={micro} boundary-F1={bf} loose-micro-F1={loose_micro:.4f}")
    print(f"saved {args.out}")

if __name__ == "__main__":
    main()
