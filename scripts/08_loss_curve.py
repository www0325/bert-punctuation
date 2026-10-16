# -*- coding: utf-8 -*-
"""
重画核心图1：训练 loss 曲线（全 step 级，体现真实曲线形态）。

数据源：runs/gpu_train_log.txt（GPU 全量 3 epoch，73,829 训练窗，每 50 步记一次）。
原图问题：07_core_figures.py 只画了 3 个 epoch 末均值点，连线近乎直线，看不出训练过程。

本脚本处理：
  1) 日志里记录的是"每 epoch 内累计均值"（run/nb），单调平滑、显得太直；
  2) 从累计均值精确还原出"每 50 步区间平均 loss"（相邻累计和之差 / 50），
     可真实呈现：快速下降 → 抖动 → epoch 边界回弹 → 收敛变平 的典型曲线；
  3) 同时叠加 EMA 平滑趋势线和原累计均值作对照。

用法：python scripts/08_loss_curve.py
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.normpath(os.path.join(HERE, "..", "runs"))
LOG = os.path.join(RUN, "gpu_train_log.txt")
OUT = os.path.join(RUN, "核心图1_训练loss曲线.png")

for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]:
    if os.path.exists(fp):
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def parse_log(path):
    """返回 epochs: {ep: [(step, 累计均值), ...]}, ends: {ep: (step, loss, val_acc)}"""
    epochs, ends = {}, {}
    pat = re.compile(r"ep(\d+) step(\d+) loss=([\d.]+)")
    pat_end = re.compile(r"== ep(\d+): loss=([\d.]+) val_acc=([\d.]+)")
    for line in open(path, encoding="utf-8"):
        m = pat.search(line)
        if m:
            ep = int(m.group(1))
            epochs.setdefault(ep, []).append((int(m.group(2)), float(m.group(3))))
        m2 = pat_end.search(line)
        if m2:
            ep = int(m2.group(1))
            last_step = epochs[ep][-1][0]  # epoch 末行没有 step，取该 epoch 最后一条
            ends[ep] = (last_step, float(m2.group(2)), float(m2.group(3)))
    return epochs, ends


def recover_segments(pts):
    """从每 epoch 内累计均值还原每 50 步区间均值。
    累计和 S(k)=avg(k)*(k-start)；区间 [prev,k] 的均值=(S(k)-S(prev))/(k-prev)。"""
    start = pts[0][0] - 50
    sums = {st: avg * (st - start) for st, avg in pts}
    xs, ys = [], []
    for i, (st, avg) in enumerate(pts):
        if i == 0:
            seg = avg  # 前 50 步区间
        else:
            pst, _ = pts[i - 1]
            seg = (sums[st] - sums[pst]) / (st - pst)
        xs.append(st)
        ys.append(seg)
    return np.array(xs), np.array(ys)


def ema(y, alpha=0.25):
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def main():
    epochs, ends = parse_log(LOG)
    assert epochs, f"no data parsed from {LOG}"

    # 全局（按 step 升序拼接，时间上是连续的）
    all_cum = sorted((st, a) for pts in epochs.values() for st, a in pts)
    Xcum = np.array([s for s, _ in all_cum])
    Ycum = np.array([a for _, a in all_cum])

    Xseg, Yseg = [], []
    for ep in sorted(epochs):
        xs, ys = recover_segments(epochs[ep])
        Xseg.append(xs)
        Yseg.append(ys)

    x_flat = np.concatenate(Xseg)
    y_flat = np.concatenate(Yseg)
    y_smooth = ema(y_flat, alpha=0.2)

    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fig.patch.set_facecolor("white")

    # epoch 交替淡色底 + 分隔线
    for i, ep in enumerate(sorted(epochs)):
        x0 = epochs[ep][0][0] - 50
        x1 = epochs[ep][-1][0]
        if i % 2 == 0:
            ax.axvspan(x0, x1, color="#4C72B0", alpha=0.06)
        ax.axvline(x1, color="#8c8c8c", linestyle=":", linewidth=1, alpha=0.8)

    # 1) 原累计均值（很直，对照用）
    ax.plot(Xcum, Ycum, color="#C0C0C0", linewidth=1.2, alpha=0.9,
            label="累计均值（原画法，过平滑）")
    # 2) 还原的 50 步区间均值（原始抖动）
    for xs, ys in zip(Xseg, Yseg):
        ax.plot(xs, ys, color="#4C72B0", linewidth=0.9, alpha=0.35)
    # 3) EMA 平滑趋势
    ax.plot(x_flat, y_smooth, color="#C44E52", linewidth=2.2,
            label="50 步区间均值（还原，EMA 平滑）")

    # 标注每 epoch 末 loss 与 val_acc
    for ep in sorted(ends):
        st, loss, va = ends[ep]
        # epoch 末的区间 loss（最后一段）在还原序列里找最后一个 x==st 的点
        last_seg = None
        for xs, ys in zip(Xseg, Yseg):
            idx = np.where(xs == st)[0]
            if len(idx):
                last_seg = ys[idx[0]]
        ax.annotate(
            f"ep{ep} 末 loss={loss:.3f}\nval_acc={va:.4f}",
            xy=(st, last_seg if last_seg is not None else loss),
            xytext=(st, (last_seg if last_seg is not None else loss) + 0.28),
            ha="center", fontsize=9, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

    ax.set_xlabel("训练 step", fontsize=11)
    ax.set_ylabel("loss", fontsize=11)
    ax.set_title("GPU 全量微调训练损失曲线（3 epoch，73,829 训练窗，每 50 步采样）",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, which="major", axis="y", alpha=0.25, linestyle="--")
    ax.margins(x=0.01)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    plt.close()
    print(f"saved: {OUT}")
    print(f"points: cumulative={len(Xcum)}, segment={len(x_flat)}")


if __name__ == "__main__":
    main()
