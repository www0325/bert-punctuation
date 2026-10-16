# -*- coding: utf-8 -*-
"""标点普查：只读 stdlib，扫描6个月语料，统计标点 token 分布与注音数量。"""
import os, re, sys, json, collections

CORPUS_DIR = os.path.join(os.environ["TEMP"], "corpus1998", "1998-上半年-带音")

def read_lines(path):
    with open(path, "r", encoding="gb18030", errors="replace") as f:
        for line in f:
            yield line.rstrip("\n").rstrip("\r")

punct_counter = collections.Counter()
note_counter = 0
line_total = 0
words_total = 0
char_total = 0
# 候选句读标点
TARGET = set("，。？！；：、")
bookmarks = set("《》〈〉〈〉〔〕【〗「」『』\"\"''""''")

for fn in sorted(os.listdir(CORPUS_DIR)):
    if not fn.endswith(".txt"):
        continue
    path = os.path.join(CORPUS_DIR, fn)
    for line in read_lines(path):
        line_total += 1
        # 去行号
        m = re.match(r"^\d{8}-\d{2}-\d{3}-\d{3}\s+(.*)$", line)
        if not m:
            continue
        body = m.group(1)
        # 注音
        notes = re.findall(r"\{[^}]*\}", body)
        note_counter += len(notes)
        body = re.sub(r"\{[^}]*\}", "", body)
        tokens = body.split()
        words_total += len(tokens)
        for tok in tokens:
            char_total += len(tok)
            if tok in TARGET or tok in bookmarks:
                punct_counter[tok] += 1
            else:
                # 多字符纯标点 token（如 ——  ……  （  ）等）
                stripped = tok
                if re.fullmatch(r"[\W_]+", stripped, flags=re.UNICODE):
                    punct_counter[tok] += 1

print("lines:", line_total)
print("notes:", note_counter)
print("tokens(words+punct):", words_total)
print("chars:", char_total)
print("--- punctuation-ish token distribution ---")
for tok, c in punct_counter.most_common(50):
    print(repr(tok), c)
