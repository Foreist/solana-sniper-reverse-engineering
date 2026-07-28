"""Figures for the write-up's media gallery.

Each chart carries one finding that the text also states, so a judge can check the claim
against the picture without running anything.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

BG, FG, MUTED = "#0d1117", "#e6edf3", "#8b949e"
GREEN, RED, BLUE, ORANGE = "#3fb950", "#f85149", "#58a6ff", "#d29922"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#30363d", "grid.color": "#21262d",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_entry_latency(entries):
    """Part 1 — the bot is a zero-block entrant, and the mean hides it."""
    lat = entries.latency_s.dropna()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    b = ax[0]
    clipped = lat.clip(-1, 10)
    b.hist(clipped, bins=np.arange(-1, 11, 1), color=BLUE, edgecolor=BG)
    b.set_xlabel("seconds from deployment to first buy (clipped at 10)")
    b.set_ylabel("buys")
    b.set_title("Entry latency", loc="left", color=FG)
    shares = [(lat <= 0).mean(), (lat <= 1).mean(), (lat <= 2).mean(), (lat <= 10).mean()]
    c = ax[1]
    c.barh(["<=0s", "<=1s", "<=2s", "<=10s"], [s * 100 for s in shares], color=GREEN)
    for i, s in enumerate(shares):
        c.text(s * 100 + 1, i, f"{s*100:.1f}%", va="center", color=FG, fontsize=8)
    c.set_xlim(0, 108)
    c.set_xlabel("share of entries (%)")
    c.set_title(f"median {lat.median():.0f}s, mean {lat.mean():.0f}s "
                f"— the mean is a tail artefact", loc="left", color=FG, fontsize=9)
    save(fig, "01_entry_latency.png")


def fig_fee_impact(pnl):
    """Part 1 — the headline finding: fees are 37% of gross."""
    gross, fees, net = pnl.gross.sum(), pnl.fees.sum(), pnl.net.sum()
    hr_g, hr_n = (pnl.gross > 0).mean(), (pnl.net > 0).mean()
    roi_g, roi_n = pnl.roi_gross.median(), pnl.roi_net.median()

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    a = ax[0]
    a.bar(["gross", "fees", "net"], [gross, -fees, net], color=[BLUE, RED, GREEN])
    for i, v in enumerate([gross, -fees, net]):
        a.text(i, v + (30000 if v > 0 else -60000), f"${v:,.0f}", ha="center", color=FG, fontsize=8)
    a.axhline(0, color="#30363d", lw=0.8)
    a.set_ylabel("USD")
    a.set_title(f"Fees are {fees/gross*100:.0f}% of gross P&L", loc="left", color=FG)

    b = ax[1]
    b.bar(["gross", "net"], [hr_g * 100, hr_n * 100], color=[BLUE, GREEN])
    for i, v in enumerate([hr_g, hr_n]):
        b.text(i, v * 100 + 1.5, f"{v*100:.1f}%", ha="center", color=FG, fontsize=9)
    b.set_ylim(0, 95)
    b.set_ylabel("hit rate (%)")
    b.set_title(f"Hit rate drops {(hr_g-hr_n)*100:.1f} points", loc="left", color=FG)

    c = ax[2]
    c.bar(["gross", "net"], [roi_g * 100, roi_n * 100], color=[BLUE, GREEN])
    for i, v in enumerate([roi_g, roi_n]):
        c.text(i, v * 100 + 0.4, f"{v*100:.1f}%", ha="center", color=FG, fontsize=9)
    c.set_ylabel("median ROI per position (%)")
    c.set_title("Median ROI falls ~3x", loc="left", color=FG)
    save(fig, "02_fee_impact.png")


def fig_pnl_distribution(pnl):
    """Part 1 — per-trade P&L: many small wins, a thin fat tail."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    r = (pnl.roi_net * 100).clip(-100, 200)
    ax[0].hist(r, bins=80, color=BLUE, edgecolor=BG)
    ax[0].axvline(0, color=RED, lw=1, ls="--")
    ax[0].set_xlabel("net ROI per position (%)")
    ax[0].set_ylabel("positions")
    ax[0].set_title("Net ROI distribution", loc="left", color=FG)

    q = pnl.roi_net.quantile([.1, .25, .5, .75, .9]) * 100
    ax[1].barh([f"p{int(i*100)}" for i in q.index], q.values,
               color=[RED if v < 0 else GREEN for v in q.values])
    for i, v in enumerate(q.values):
        ax[1].text(v + (1 if v > 0 else -1), i, f"{v:.1f}%", va="center",
                   ha="left" if v > 0 else "right", color=FG, fontsize=8)
    ax[1].axvline(0, color="#30363d", lw=0.8)
    ax[1].set_xlabel("net ROI (%)")
    ax[1].set_title("Deciles — the left tail is real", loc="left", color=FG)
    save(fig, "03_pnl_distribution.png")


def fig_feature_importance(top):
    """Part 2 — what the bot's selection actually keys on."""
    t = top.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(t.feature, t.importance, color=BLUE)
    ax.set_xlabel("permutation importance (drop in ROC-AUC)")
    ax.set_title("Top 10 features — all deployer-history, none temporal",
                 loc="left", color=FG)
    save(fig, "04_feature_importance.png")


def fig_pr_curve(scores):
    """Part 2 — precision-recall against a 1:202 base rate."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y, s = scores.label.to_numpy(), scores.score_gbdt.to_numpy()
    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    base = y.mean()

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(rec, prec, color=GREEN, lw=1.6, label=f"GBDT  PR-AUC {ap:.4f}")
    ax.axhline(base, color=RED, ls="--", lw=1,
               label=f"prevalence {base:.5f} (1:{int(1/base)})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(prec[rec > 0.01].max() * 1.1, base * 5))
    ax.legend(frameon=False, labelcolor=FG, fontsize=8)
    ax.set_title(f"Held-out June: {ap/base:.1f}x the base rate", loc="left", color=FG)
    save(fig, "05_precision_recall.png")


def fig_slot_sensitivity(sens):
    """Part 3 — the whole strategy lives in the zero block."""
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    x = [f"{int(d)} slot\n(+{s:.1f}s)" for d, s in zip(sens.slot_delay, sens.slot_delay * 0.4)]
    cols = [GREEN if v > 0 else RED for v in sens.net_pnl]

    ax[0].bar(x, sens.median_exit_multiple, color=cols)
    ax[0].axhline(1.0, color=MUTED, ls="--", lw=0.8)
    for i, v in enumerate(sens.median_exit_multiple):
        ax[0].text(i, v + 0.01, f"{v:.3f}", ha="center", color=FG, fontsize=8)
    ax[0].set_ylim(0.9, max(sens.median_exit_multiple) * 1.08)
    ax[0].set_ylabel("median exit multiple")
    ax[0].set_title("Price decays immediately", loc="left", color=FG)

    ax[1].bar(x, sens.median_roi_net * 100, color=cols)
    ax[1].axhline(0, color="#30363d", lw=0.8)
    for i, v in enumerate(sens.median_roi_net * 100):
        ax[1].text(i, v + (1 if v > 0 else -2), f"{v:.1f}%", ha="center", color=FG, fontsize=8)
    ax[1].set_ylabel("median net ROI (%)")
    ax[1].set_title("0.4s of delay flips the sign", loc="left", color=FG)

    ax[2].bar(x, sens.net_pnl, color=cols)
    ax[2].axhline(0, color="#30363d", lw=0.8)
    for i, v in enumerate(sens.net_pnl):
        ax[2].text(i, v + (900 if v > 0 else -2200), f"${v:,.0f}", ha="center", color=FG, fontsize=8)
    ax[2].set_ylabel("net P&L (USD)")
    ax[2].set_title("Fees are unpayable once late", loc="left", color=FG)
    save(fig, "06_slot_sensitivity.png")


def fig_cover(pnl, sens):
    """Cover image — the one finding that reframes the whole competition."""
    fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.88, "The edge is 400 milliseconds wide",
            ha="center", fontsize=17, fontweight="bold", color=FG)
    ax.text(0.5, 0.755, "reverse-engineering a live pump.fun sniper — and pricing it honestly",
            ha="center", fontsize=7.6, color=MUTED, style="italic")
    vals = [(0, sens.median_roi_net.iloc[0] * 100, GREEN),
            (1, sens.median_roi_net.iloc[1] * 100, RED),
            (2, sens.median_roi_net.iloc[2] * 100, RED)]
    for i, v, c in vals:
        x = 0.13 + i * 0.27
        h = abs(v) / 100 * 0.9
        y0 = 0.34 if v > 0 else 0.34 - h
        ax.add_patch(plt.Rectangle((x, y0), 0.20, h, color=c))
        ax.text(x + 0.10, 0.34 + (h + 0.03 if v > 0 else -h - 0.07),
                f"{v:+.1f}%", ha="center", fontsize=12, fontweight="bold", color=c)
        ax.text(x + 0.10, 0.20, f"{i} slot delay", ha="center", fontsize=7, color=FG)
    ax.plot([0.08, 0.92], [0.34, 0.34], color="#30363d", lw=0.9)
    ax.text(0.5, 0.07, "median net ROI after fees  ·  fees are 37% of the incumbent's gross P&L",
            ha="center", fontsize=6.3, color="#6e7681")
    save(fig, "00_cover.png")


def main():
    entries = pd.read_csv(os.path.join(HERE, "part1_entries.csv"))
    pnl = pd.read_csv(os.path.join(HERE, "part1_pnl_per_token.csv"))
    top = pd.read_csv(os.path.join(HERE, "part2_top_features.csv"))
    sens = pd.read_csv(os.path.join(HERE, "part3_slot_sensitivity.csv"))
    scores = pd.read_parquet(os.path.join(HERE, "part2_test_scores.parquet"))

    fig_entry_latency(entries)
    fig_fee_impact(pnl)
    fig_pnl_distribution(pnl)
    fig_feature_importance(top)
    fig_pr_curve(scores)
    fig_slot_sensitivity(sens)
    fig_cover(pnl, sens)


if __name__ == "__main__":
    main()
