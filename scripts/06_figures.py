# -*- coding: utf-8 -*-
"""最终版出图：读 runs/ 下所有 JSON，生成报告用图。"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

RUN = r"C:\Users\test\Desktop\迁移桌面\研究生\上课\大语言模型与应用实践\作业2-BERT分词\runs"
OUT = RUN
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# fig1: GPU 全量模型逐类 F1（slide）
ev = json.load(open(os.path.join(RUN, "gpu_results", "runs", "eval.json"), encoding="utf-8"))
pc = ev["slide"]["per_class"]
names = list(pc.keys()); f1s = [pc[n]["F1"] for n in names]
fig, ax = plt.subplots(figsize=(7,4))
bars = ax.bar(names, f1s, color="#4C72B0")
ax.set_ylabel("F1"); ax.set_title("GPU 全量模型逐类标点 F1（测试集，slide 窗）")
for b,v in zip(bars,f1s): ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=8)
plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig1_per_class_f1.png"),dpi=130); plt.close()

# fig2: 长文四策略对比（F1 + 吞吐双轴）
lb = json.load(open(os.path.join(RUN,"t4_results","runs","gpu_longbench.json"),encoding="utf-8"))
strats = ["naive_overlap","no_overlap","seam_minimal","assignment_baseline"]
labels = ["naive 重叠","不重叠","seam 补缝","作业基线\n400/300句号接续"]
micro = [lb[s]["micro_f1"] for s in strats]
bound = [lb[s]["boundary_f1"] for s in strats]
thru = [lb[s]["throughput_chars_per_s"] for s in strats]
x = np.arange(4); w=0.25
fig, ax1 = plt.subplots(figsize=(8,4.5))
ax1.bar(x-w, micro, w, label="micro-F1", color="#55A868")
ax1.bar(x, bound, w, label="句界 F1", color="#C44E52")
ax1.set_ylabel("F1"); ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax2 = ax1.twinx()
ax2.bar(x+w, thru, w, label="吞吐 (字/s)", color="#8172B2", alpha=0.7)
ax2.set_ylabel("吞吐 (字/s)")
ax1.set_title("长文滑窗四策略对比（T4，20 篇长文 × 平均 2077 字）")
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig2_longdoc_strategies.png"),dpi=130); plt.close()

# fig3: batch 缩放（吞吐+显存）
bs = lb["batch_scaling"]
bss = [1,4,8,16,32]
tok = [bs[str(b)]["tok_per_s"] for b in bss]
vram = [bs[str(b)]["vram_GB"] for b in bss]
fig, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(bss, tok, "o-", color="#4C72B0", label="吞吐 (tok/s)")
ax1.set_xlabel("batch size"); ax1.set_ylabel("吞吐 (tok/s)", color="#4C72B0")
ax2 = ax1.twinx()
ax2.plot(bss, vram, "s--", color="#C44E52", label="显存 (GB)")
ax2.set_ylabel("显存 (GB)", color="#C44E52")
ax1.set_title("T4 batch 缩放：吞吐 vs 显存（seq=510）")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig3_batch_scaling.png"),dpi=130); plt.close()

# fig4: 跨域古文（严格 vs 宽松）
an = json.load(open(os.path.join(RUN,"t4_results","runs","ancient_eval.json"),encoding="utf-8"))
fig, ax = plt.subplots(figsize=(6,4))
metrics = ["micro-F1","句界 F1"]
strict = [an["micro_f1"], an["boundary_f1"]]
loose = [an["loose_micro_f1"], an["loose_boundary_f1"]]
x = np.arange(2); w=0.3
ax.bar(x-w/2, strict, w, label="严格位置匹配", color="#C44E52")
ax.bar(x+w/2, loose, w, label="±5字宽松匹配", color="#55A868")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_title("现代新闻模型 → 古文跨域退化（100 条正史）")
ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig4_cross_domain.png"),dpi=130); plt.close()

# fig5: CPU vs GPU 对比
fig, ax = plt.subplots(figsize=(6,4))
items = ["micro-F1","macro-F1","句界 F1"]
cpu = [0.660, 0.474, 0.717]
gpu = [ev["slide"]["micro_f1"], ev["slide"]["macro_f1"], ev["slide"]["boundary_f1"]]
x = np.arange(3); w=0.3
ax.bar(x-w/2, cpu, w, label="CPU 缩比(8k窗,2ep)", color="#8172B2")
ax.bar(x+w/2, gpu, w, label="GPU 全量(73k窗,3ep)", color="#4C72B0")
ax.set_xticks(x); ax.set_xticklabels(items)
ax.set_title("CPU 缩比 vs GPU 全量（测试集 slide 窗）")
ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig5_cpu_vs_gpu.png"),dpi=130); plt.close()

print("5 figures saved")

# fig6: 跨域文学作品 F1
import glob
v2 = os.path.join(RUN, "t4_results_v2", "runs")
works = []
for f in sorted(glob.glob(os.path.join(v2, "modern_*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    name = os.path.basename(f).replace("modern_","").replace(".json","")
    works.append((name, d.get("micro_f1",0), d.get("boundary_f1",0), d.get("loose_micro_f1",0)))
if works:
    fig, ax = plt.subplots(figsize=(8,4.5))
    x = np.arange(len(works)); w=0.25
    ax.bar(x-w, [w_[1] for w_ in works], w, label="严格 micro-F1", color="#C44E52")
    ax.bar(x, [w_[2] for w_ in works], w, label="句界 F1", color="#DD8452")
    ax.bar(x+w, [w_[3] for w_ in works], w, label="±5字宽松 micro-F1", color="#55A868")
    ax.axhline(y=0.7245, color="#4C72B0", linestyle="--", label="人民日报新闻基线 0.72")
    ax.set_xticks(x); ax.set_xticklabels([w_[0] for w_ in works], fontsize=9)
    ax.set_title("跨域文学作品标点 F1（严格 vs 宽松）")
    ax.legend(fontsize=8, ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"fig6_literature_cross_domain.png"),dpi=130); plt.close()
    print("fig6 saved")
