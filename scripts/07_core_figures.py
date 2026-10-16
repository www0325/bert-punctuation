# -*- coding: utf-8 -*-
"""核心两张图：训练 loss 曲线 + 长文滑窗精度对比。"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

RUN = r"C:\Users\test\Desktop\迁移桌面\研究生\上课\大语言模型与应用实践\作业2-BERT分词\runs"
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# 图1：训练 loss + val_acc 曲线
epochs = [1,2,3]
train_loss = [0.3424, 0.1866, 0.1501]
val_acc = [0.9598, 0.9649, 0.9674]

fig, ax1 = plt.subplots(figsize=(7,4.5))
ax1.plot(epochs, train_loss, "o-", color="#C44E52", linewidth=2, markersize=8, label="训练 loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("训练 loss", color="#C44E52")
ax1.set_xticks(epochs)
for x,y in zip(epochs, train_loss):
    ax1.text(x, y+0.02, f"{y:.3f}", ha="center", color="#C44E52", fontsize=10)

ax2 = ax1.twinx()
ax2.plot(epochs, val_acc, "s--", color="#4C72B0", linewidth=2, markersize=8, label="验证集 acc")
ax2.set_ylabel("验证集字符 acc", color="#4C72B0")
ax2.set_ylim(0.95, 0.975)
for x,y in zip(epochs, val_acc):
    ax2.text(x, y+0.002, f"{y:.4f}", ha="center", color="#4C72B0", fontsize=9)

plt.title("GPU 全量微调训练曲线（A10，3 epoch，73,829 训练窗）")
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, loc="center right")
plt.tight_layout()
plt.savefig(os.path.join(RUN,"核心图1_训练loss曲线.png"), dpi=140)
plt.close()
print("fig1 saved")

# 图2：长文滑窗精度对比（micro-F1 + 句界 F1）
lb = json.load(open(os.path.join(RUN,"t4_results_v2","runs","gpu_longbench.json"),encoding="utf-8"))
strats = ["naive_overlap","no_overlap","seam_minimal","fusion_weighted","assignment_baseline"]
labels = ["naive\n重叠50%", "no_overlap\n不重叠", "seam_minimal\n接缝补窗", "fusion_weighted\n中心加权", "assignment_baseline\n句号接续"]
micro = [lb[s]["micro_f1"] for s in strats]
bound = [lb[s]["boundary_f1"] for s in strats]

x = np.arange(5); w = 0.35
fig, ax = plt.subplots(figsize=(9,4.5))
b1 = ax.bar(x-w/2, micro, w, color="#4C72B0", label="micro-F1（全部标点）")
b2 = ax.bar(x+w/2, bound, w, color="#DD8452", label="句界 F1（句号/问号/叹号）")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("F1")
ax.set_ylim(0.5, 0.78)
for bars in [b1, b2]:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+0.005, f"{h:.3f}", ha="center", fontsize=8)
ax.set_title("长文滑窗五策略精度对比（T4，20 篇长文 × 平均 2077 字）")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(RUN,"核心图2_滑窗精度对比.png"), dpi=140)
plt.close()
print("fig2 saved")
