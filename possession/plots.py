"""Figures for the write-up: our model vs the four baselines on the PUBLIC 5-minute window.

    python possession/plots.py      # -> docs/scores.png, docs/timeline.png

Reads evaluation/results/*.csv (copy submissions/submission.csv there as hmm_viterbi.csv first).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

RES = os.path.join(ROOT, "evaluation", "results")
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OUT = os.path.join(ROOT, "docs")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
OURS, BASE = "#2a78d6", "#b9b8b1"
HOME, AWAY, DEAD = "#2a78d6", "#eb6834", "#d6d5cf"

ENTRIES = [("hmm_viterbi", "Our model"), ("baseline_smooth125", "Baseline ±5 s"),
           ("baseline_smooth50", "Baseline ±2 s (official floor)"),
           ("baseline_smooth25", "Baseline ±1 s"), ("baseline_raw", "Baseline raw")]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def scores():
    gt = pm.label_codes(pm.read_label_csv(GT))
    rows = []
    for f, name in ENTRIES:
        pred = pm.label_codes(pm.read_label_csv(os.path.join(RES, f + ".csv")))
        r = pm.evaluate(gt, pred, bootstrap_n=1000)
        rows.append((name, r))
    return gt, rows


def plot_scores(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, facecolor=SURFACE)
    names = [n for n, _ in rows][::-1]
    y = np.arange(len(names))
    for ax, key, title in ((axes[0], "spell_f1_mean", "spellF1 (ranking metric)"),
                           (axes[1], "macro_f1", "macroF1 (tie-break)")):
        style(ax)
        vals = [r[key] for _, r in rows][::-1]
        cols = [OURS if n == "Our model" else BASE for n in names]
        ax.barh(y, vals, height=0.62, color=cols, edgecolor=SURFACE, linewidth=2)
        if key == "spell_f1_mean":
            lo = [r["spell_bootstrap_low"] for _, r in rows][::-1]
            hi = [r["spell_bootstrap_high"] for _, r in rows][::-1]
            ax.hlines(y, lo, hi, color=INK2, linewidth=1.2)
        for yi, v, n in zip(y, vals, names):
            ax.text(v + 0.015, yi + (0.22 if key == "spell_f1_mean" else 0), "{:.3f}".format(v),
                    va="center", fontsize=9, color=INK,
                    fontweight="bold" if n == "Our model" else "normal")
        ax.set_xlim(0, 1)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, color=INK)
    axes[0].text(0, -1.05, "Whiskers: 95% bootstrap interval. Public 5-minute window only (17 possessions).",
                 color=MUTED, fontsize=8, transform=axes[0].get_yaxis_transform())
    fig.suptitle("Our model vs the baselines, public 5-minute window", x=0.01, ha="left",
                 color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scores.png"), dpi=160, facecolor=SURFACE)


def runs(codes):
    out, start = [], 0
    for i in range(1, len(codes) + 1):
        if i == len(codes) or codes[i] != codes[start]:
            out.append((start, i, codes[start]))
            start = i
    return out


def plot_timeline(gt, rows):
    lo, hi = 11185, 18685
    lanes = [("Ground truth", gt)]
    for f, name in (("hmm_viterbi", "Our model"), ("baseline_smooth50", "Baseline ±2 s")):
        lanes.append((name, pm.label_codes(pm.read_label_csv(os.path.join(RES, f + ".csv")))))
    colour = {0: HOME, 1: AWAY, 2: DEAD}
    fig, ax = plt.subplots(figsize=(11, 2.9), facecolor=SURFACE)
    style(ax)
    ax.xaxis.grid(False)
    for row, (name, codes) in enumerate(lanes):
        seg = codes[lo:hi]
        n_poss = sum(1 for s, e, c in runs(seg) if c in (0, 1))
        for s, e, c in runs(seg):
            ax.barh(row, (e - s) / 25 / 60, left=s / 25 / 60, height=0.6, color=colour.get(int(c), "#fff"),
                    edgecolor=SURFACE, linewidth=1)
        ax.text(5.08, row, "{} possessions".format(n_poss), va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([n for n, _ in lanes], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 5)
    ax.set_xlabel("minutes from kick-off", color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (HOME, AWAY, DEAD)]
    ax.legend(handles, ["Barcelona (H)", "Athletic (A)", "Dead ball (D)"], ncol=3, frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "timeline.png"), dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    gt, rows = scores()
    plot_scores(rows)
    plot_timeline(gt, rows)
    print("wrote docs/scores.png, docs/timeline.png")
