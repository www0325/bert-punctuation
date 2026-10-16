# -*- coding: utf-8 -*-
"""
阶段1：清洗人民日报1998上半年语料 -> char 级断句标签。
输入：%TEMP%/corpus1998/1998-上半年-带音/1998-0X-2003版-带音.txt  (GB18030)
输出：
  data/processed/{train,val,test}.jsonl   每行 {"chars":[...], "labels":[...int], "src":"1998-01"}
  data/processed/stats.json                清洗与标签统计
标签体系（8核心 + 第9类书名号《》=创新）：
  0 NONE / 1 COMMA， / 2 PERIOD。 / 3 QUESTION？ / 4 EXCLAM！ /
  5 SEMICOLON； / 6 COLON： / 7 ENUM、 / 8 BOOK《》
"""
import os, re, json, collections

CORPUS_DIR = os.path.join(os.environ["TEMP"], "corpus1998", "1998-上半年-带音")
OUT_DIR = r".\data\processed"
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = ["NONE", "COMMA", "PERIOD", "QUESTION", "EXCLAM",
          "SEMICOLON", "COLON", "ENUM", "BOOK"]
PUNCT2ID = {"，": 1, "。": 2, "？": 3, "！": 4, "；": 5, "：": 6, "、": 7}
# 书名号《》统一归为第8类（创新点2）
BOOK_OPEN, BOOK_CLOSE = "《", "》"
# 丢弃的符号：引号/括号/破折号/省略号/噪声占位符/项目符号，不参与断句
DROP = set("“”‘’『』（）()【】[]〈〉〈〉〔〕———…·•●●△▲○★※＊*％%℃×±＝=／/－-—…")
# 注意：〈〉仅64对，并入丢弃（样本过少，单列不可学）；《》1.3万对保留为BOOK
NOTE_RE = re.compile(r"\{[^}]*\}")
LINE_RE = re.compile(r"^(\d{4})(\d{2})\d{2}-\d{2}-\d{3}-\d{3}\s+(.*)$")

MONTH_SPLIT = {"train": {"01", "02", "03", "04"},
               "val": {"05"},
               "test": {"06"}}

def clean_paragraph(body):
    """body=行号后正文。返回 (chars:list[str], labels:list[int]) 或 None。"""
    body = NOTE_RE.sub("", body)
    tokens = body.split()
    words = []          # list of (word_str, following_label_int)
    pending_label = 0   # 上一个word之后紧跟的标点
    for tok in tokens:
        if tok in PUNCT2ID:
            pending_label = PUNCT2ID[tok]
        elif tok == BOOK_OPEN or tok == BOOK_CLOSE:
            pending_label = 8          # 书名号统一记在它前面那个词上
        elif tok in DROP:
            continue                   # 引号括号等：既不产生字也不占用标签位
        elif tok == "〓" or set(tok) <= set("〓"):
            continue                   # 噪声占位符
        else:
            # 普通词
            chars = list(tok)
            if not chars:
                continue
            words.append((chars, pending_label))
            pending_label = 0
    # 展平成 char 序列
    chars, labels = [], []
    for chs, lab in words:
        for i, ch in enumerate(chs):
            chars.append(ch)
            labels.append(lab if i == len(chs) - 1 else 0)
    if len(chars) < 8:   # 过短段落丢弃
        return None
    return chars, labels

def main():
    stats = collections.Counter()
    label_counter = collections.Counter()
    out_files = {k: open(os.path.join(OUT_DIR, f"{k}.jsonl"), "w", encoding="utf-8")
                for k in MONTH_SPLIT}
    for fn in sorted(os.listdir(CORPUS_DIR)):
        m = re.match(r"1998-(\d{2})-", fn)
        if not m:
            continue
        month = m.group(1)
        split = next(k for k, v in MONTH_SPLIT.items() if month in v)
        with open(os.path.join(CORPUS_DIR, fn), encoding="gb18030", errors="replace") as f:
            for line in f:
                line = line.rstrip("\r\n")
                mm = LINE_RE.match(line)
                if not mm:
                    stats["bad_line"] += 1
                    continue
                body = mm.group(3)
                res = clean_paragraph(body)
                if res is None:
                    stats["drop_short"] += 1
                    continue
                chars, labels = res
                out_files[split].write(json.dumps(
                    {"chars": chars, "labels": labels, "src": f"1998-{month}"},
                    ensure_ascii=False) + "\n")
                stats[split + "_para"] += 1
                stats[split + "_chars"] += len(chars)
                for lb in labels:
                    label_counter[lb] += 1
    for f in out_files.values():
        f.close()
    stats["label_dist"] = {LABELS[i]: label_counter[i] for i in range(len(LABELS))}
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
