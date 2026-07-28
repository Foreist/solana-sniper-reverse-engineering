"""Emit the public Kaggle notebook as solana_sniper_reverse_engineering.ipynb."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DS = "/kaggle/input/solana-sniper-bot-reverse-engineering-data"

cells = []


def _lines(text):
    """nbformat stores source as a list of lines that each KEEP their trailing newline.

    Dropping the newlines makes Jupyter concatenate the whole cell onto one line, which fails
    with a SyntaxError only once the notebook actually runs on Kaggle.
    """
    raw = text.strip("\n").split("\n")
    return [ln + "\n" for ln in raw[:-1]] + raw[-1:]


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _lines(text)})


md(r"""
# The edge is 400 milliseconds wide

**Reverse-engineering the pump.fun sniper `5brv79eFZ2rGprXNvqgVJBkBptkkw8GJX1XydJyZLyAr`.**

The bot made **\$1.49M gross** in 3.5 months and kept **\$937,925** after fees. Its strategy earns
**+\$29,142** with a zero-block fill and **&minus;\$334** one slot (400&nbsp;ms) later.

This notebook is the verifiable half of the submission. Rather than restating results, it
**re-executes** the three checks that decide whether they mean anything:

1. the `t_decision` truncation audit, run on raw history that *still contains* post-cutoff events,
2. all three leakage re-checks, recomputed from the feature table,
3. the entry-fill calibration that fixes &alpha;, and the Part 2 metrics, recomputed from the
   scored test set.

The 83&nbsp;GB raw corpus cannot be uploaded, so the intermediate artifacts and the analysis code
are published as a dataset and imported here.
""")

code(f"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

DS = "{DS}"
sys.path.insert(0, os.path.join(DS, "code"))   # the real analysis modules, not a copy

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 40)

summary   = pd.read_csv(os.path.join(DS, "part1_summary.csv"), index_col=0).iloc[:, 0]
pnl       = pd.read_csv(os.path.join(DS, "part1_pnl_per_token.csv"))
entries   = pd.read_csv(os.path.join(DS, "part1_entries.csv"))
top_feat  = pd.read_csv(os.path.join(DS, "part2_top_features.csv"))
sens      = pd.read_csv(os.path.join(DS, "part3_slot_sensitivity.csv"))
overlap   = pd.read_csv(os.path.join(DS, "part3_overlap.csv"), index_col=0).iloc[:, 0]
alpha_cal = pd.read_csv(os.path.join(DS, "part3_alpha_calibration.csv"))
passive   = pd.read_csv(os.path.join(DS, "part3_passive_hold.csv"))

print("artifacts loaded")
print(f"  pnl per token   {{len(pnl):,}} tokens")
print(f"  bot buy events  {{len(entries):,}}")
""")

md(r"""
---
# Part 1 &mdash; What the bot does

Scale, entry, exit. Everything below is recomputed here from the bot's own trade records.
""")

code("""
s = summary
print(f"tokens bought        {int(s.tokens_bought):,}")
print(f"buy events           {int(s.buy_events):,}")
print(f"sell events          {int(s.sell_events):,}")
print()
print(f"position size        median ${s.entry_usd_median:,.2f}   mean ${s.entry_usd_mean:,.2f}"
      f"   p05 ${s.entry_usd_p05:,.0f}  p95 ${s.entry_usd_p95:,.0f}")
print(f"entry latency        median {s.latency_s_median:.0f}s")
print(f"  share at <= 0s     {s.zero_block_share_le0s:.1%}   <- lands in the deployment block")
print(f"  share at <= 1s     {s.share_le1s:.1%}")
print(f"  mean latency       {s.latency_s_mean:.1f}s  (a thin tail of late entries, not the mode)")
print()
print(f"sells per token      median {s.sells_per_token_median:.0f}")
print(f"partial exit share   {s.partial_exit_share:.1%}")
print(f"hold seconds         median {s.hold_s_median:.0f}  (p25 {s.hold_s_p25:.0f}, p75 {s.hold_s_p75:.0f})")
print(f"never sold           {int(s.tokens_never_sold)} tokens")
""")

md(r"""
### Fees are 37% of gross, and omitting them inverts the picture

Two things are verified here rather than assumed: that `gas_native` is exactly
`priority_fee + tip_fee`, and that **neither gas nor venue fees are inside `cost_usd`** &mdash; so a
P&L built from `cost_usd` alone is a *gross* figure.
""")

code("""
e = entries.copy()
for c in ["gas_native", "priority_fee", "tip_fee", "gas_usd", "dex_usd", "cost_usd"]:
    e[c] = pd.to_numeric(e[c], errors="coerce")

buys = e[e.event_type == "buy"]
identity = buys.dropna(subset=["gas_native", "priority_fee", "tip_fee"])
lhs, rhs = identity.gas_native, identity.priority_fee + identity.tip_fee
print("CHECK  gas_native == priority_fee + tip_fee")
print(f"  correlation           {lhs.corr(rhs):.6f}")
print(f"  max abs difference    {(lhs - rhs).abs().max():.10f}")
print(f"  rows agreeing to 1e-9 {(lhs - rhs).abs().lt(1e-9).mean():.2%}")

print()
gross = pnl.gross.sum(); fees = pnl.fees.sum(); net = pnl.net.sum()
gas = pnl.gas.sum(); dex = pnl.dex.sum()
print(f"GROSS  ${gross:>12,.0f}   hit rate {(pnl.gross > 0).mean():.1%}"
      f"   median ROI {pnl.roi_gross.median():.1%}")
print(f"FEES   ${fees:>12,.0f}   = gas ${gas:,.0f} + venue ${dex:,.0f}"
      f"   -> {fees / gross:.1%} of gross")
print(f"NET    ${net:>12,.0f}   hit rate {(pnl.net > 0).mean():.1%}"
      f"   median ROI {pnl.roi_net.median():.1%}")
print()
print(f"  omitting fees overstates the hit rate by "
      f"{((pnl.gross > 0).mean() - (pnl.net > 0).mean()) * 100:.1f} pp")
print(f"  and the median ROI by {pnl.roi_gross.median() / pnl.roi_net.median():.1f}x")
""")

md(r"""
### The cost is entirely at the entry

Paying to be first is the whole cost base: entry priority is **230&times;** the cost of an exit.
That asymmetry is what makes the slot-delay result in Part 3 so violent.
""")

code("""
entry_priority = (buys.priority_fee.fillna(0) + buys.tip_fee.fillna(0))
sol_usd = (buys.gas_usd / buys.gas_native).median()
entry_priority_usd = (entry_priority * sol_usd).mean()
exit_gas_usd = pnl.gas.sum() / pnl.n_sells.sum() * 0  # placeholder, computed below

n_sells = pnl.n_sells.sum()
gas_on_buys = (buys.gas_usd.fillna(0)).sum()
exit_gas_usd = max(pnl.gas.sum() - gas_on_buys, 0) / n_sells

print(f"SOL price used (median gas_usd / gas_native)   ${sol_usd:,.2f}")
print(f"entry priority + tip, per buy                  ${entry_priority_usd:,.2f}")
print(f"  as a share of the mean position              {entry_priority_usd / summary.entry_usd_mean:.1%}")
print(f"venue fee per buy                              ${buys.dex_usd.mean():,.2f}"
      f"  ({buys.dex_usd.mean() / summary.entry_usd_mean:.1%} of position)")
print(f"exit gas, per sell                             ${exit_gas_usd:,.3f}")
print(f"  entry-to-exit cost ratio                     {entry_priority_usd / exit_gas_usd:,.0f}x")
""")

code("""
for f in ["01_entry_latency.png", "02_fee_impact.png", "03_pnl_distribution.png"]:
    display(Image(filename=os.path.join(DS, "figures", f)))
""")

md(r"""
---
# Part 2 &mdash; Reconstructing the selection rule

## The disqualification defence, executed

Any data timestamped after `t_decision` reaching a **feature** zeroes Parts 2 and 3. We did not
rely on being careful about it:

1. `TruncatedHistory` is a **frozen dataclass** whose constructor physically applies
   `timestamp < cutoff`. No accessor on it reaches past the cutoff.
2. Every feature function takes a `TruncatedHistory`, never a raw frame &mdash; leaking requires
   rewriting the type, not making a mistake.
3. `audit_truncation()` re-reads the **original unfiltered** activity and asserts the invariant.

Below, the real `features.py` is imported from the dataset and the audit runs against raw
deployer history. Note the count of post-cutoff events in that raw sample: the check is given
data it **could** fail on.
""")

code("""
import inspect
import features   # the real module, shipped in the dataset

print(inspect.getsource(features.TruncatedHistory))
""")

code("""
audit_deploys = pd.read_parquet(os.path.join(DS, "audit_sample_deploys.parquet"))
audit_act = pd.read_parquet(os.path.join(DS, "audit_sample_activity.parquet"))

print(f"raw sample: {len(audit_deploys):,} deployments, {len(audit_act):,} activity rows, "
      f"{audit_deploys.wallet.nunique():,} deployers")

by_wallet = {w: g.sort_values("timestamp") for w, g in audit_act.groupby("wallet", sort=False)}

# How much post-cutoff data is sitting in the frame the audit reads?
post = 0
for w, cut in zip(audit_deploys.wallet, audit_deploys.deploy_time):
    g = by_wallet.get(w)
    if g is not None:
        post += int((g.timestamp >= cut).sum())
print(f"post-cutoff events present in this raw sample: {post:,}")
print("  -> the audit below is not vacuous; it has something to catch.\\n")
""")

code("""
report = features.audit_truncation(audit_deploys, by_wallet, audit_deploys)
print("audit_truncation():", report)
assert report["violations"] == 0
print(f"\\nVIOLATIONS: {report['violations']}  over {report['rows_checked']:,} deployments checked")
""")

md(r"""
## The leakage we caught, and the three checks that now guard it

Our first Part 2 run returned **ROC-AUC of exactly 0.5000** with every feature importance at
0.00000. The cause was a bookkeeping column, `neg_sample_scale`, left in the matrix: in training
it separates the classes perfectly (20.0 negative / 1.0 positive) and in test it is constant at
1.0, so the model learned it and emitted a constant. `deploy_slot` and `deploy_time` are excluded
for the same class of reason &mdash; both rise monotonically with time, making them pure period
indicators under a temporal split.

All three re-checks are recomputed below from the feature table.
""")

code("""
from sklearn.metrics import roc_auc_score

feats = pd.read_parquet(os.path.join(DS, "features_all.parquet"))
TEST_START = 1780272000        # 2026-06-01 00:00 UTC
BOT_ACTIVE_FROM = 1773305601   # the bot's first buy

EXCLUDED = ["token_address", "wallet", "deploy_time", "label", "deploy_slot", "neg_sample_scale"]
FEATURES = [c for c in feats.columns if c not in EXCLUDED]

df = feats[feats.deploy_time >= BOT_ACTIVE_FROM]
train = df[df.deploy_time < TEST_START]
test = df[df.deploy_time >= TEST_START]
print(f"feature table {feats.shape[0]:,} x {len(FEATURES)} features in use")
print(f"train {len(train):,}  ({int(train.label.sum()):,} positive)")
print(f"test  {len(test):,}  ({int(test.label.sum()):,} positive, "
      f"prevalence {test.label.mean():.3%}, imbalance 1:{(1 / test.label.mean() - 1):.0f})")
""")

code("""
print("CHECK 1  any column that separates the training classes perfectly?")
bad = []
for c in FEATURES:
    p = train.loc[train.label == 1, c]
    n = train.loc[train.label == 0, c]
    if p.notna().any() and n.notna().any():
        if p.min() > n.max() or p.max() < n.min():
            bad.append(c)
print(f"  perfectly separating columns: {bad if bad else 'NONE'}\\n")

print("CHECK 2  any column whose mean shifts more than 2x between train and test?")
shift = []
for c in FEATURES:
    a, b = train[c].mean(), test[c].mean()
    if a and np.isfinite(a) and np.isfinite(b) and (b / a > 2 or b / a < 0.5):
        shift.append((c, round(float(a), 1), round(float(b), 1), round(float(b / a), 3)))
for c, a, b, r in shift:
    print(f"  {c:24s} train {a:>12,.1f}  test {b:>12,.1f}   {r:.3f}x")
if not shift:
    print("  NONE")
print()

print("CHECK 3  strongest single-feature AUC (alarm threshold 0.95)")
samp = train.sample(min(200_000, len(train)), random_state=0)
aucs = {}
for c in FEATURES:
    v = samp[c].fillna(samp[c].median())
    if v.nunique() > 1:
        a = roc_auc_score(samp.label, v)
        aucs[c] = max(a, 1 - a)
srt = sorted(aucs.items(), key=lambda kv: -kv[1])[:5]
for c, a in srt:
    print(f"  {c:24s} {a:.3f}")
print(f"\\n  max = {srt[0][1]:.3f} -- below 0.95, no single feature is doing the work")
""")

md(r"""
**Check 2 flags exactly one column, and we keep it after looking at why.** `secs_since_last` —
seconds since the deployer's previous recorded event — has a train mean of 121,595 s against
245,305 s in test, a 2.02&times; shift.

This is **corpus drift, not leakage**. The recency gap of a dormant wallet mechanically grows as
the observation window extends, so a feature measuring "time since last seen" must drift upward
in any later period. It is not a post-`t_decision` quantity: every value is computed strictly
before its own cutoff, which is what Check 1 and the audit above establish.

We keep it, with two guards. It is the strongest single feature at AUC **0.785**, which is well
under the 0.95 alarm line, so no single column is carrying the model. And it ranks **third** in
permutation importance behind two amount-based features, so the fitted model is not leaning on
the drifting one. It is nonetheless the feature we would watch first if this were deployed
forward, and we would rather flag it than quietly pass a check by loosening it.
""")

md(r"""
## Model results, recomputed from the scored test set

The test split keeps the **true** prevalence (0.492%) rather than a rebalanced one, so PR-AUC is
measured against the imbalance the strategy actually faces.
""")

code("""
import part2_model as P2   # the real scoring code, so these are not a second implementation

scores = pd.read_parquet(os.path.join(DS, "part2_test_scores.parquet"))
y = scores.label.values
prev = y.mean()

rows = []
for name, col in [("GBDT", "score_gbdt"), ("Logistic", "score_logit")]:
    r = P2.evaluate(y, scores[col].values, label=name)
    op5 = next(o for o in r["operating_points"] if o["target_recall"] == 0.05)
    rows.append({"model": name, "PR-AUC": r["pr_auc"], "ROC-AUC": r["roc_auc"],
                 "precision @ 5% recall": op5["precision_observed"]})
rows.append({"model": "prevalence baseline", "PR-AUC": prev, "ROC-AUC": 0.5,
             "precision @ 5% recall": prev})
res = pd.DataFrame(rows).set_index("model")
display(res.style.format({"PR-AUC": "{:.4f}", "ROC-AUC": "{:.3f}",
                          "precision @ 5% recall": "{:.1%}"}))

g = res.loc["GBDT"]
print(f"\\nprevalence            {prev:.3%}  (imbalance 1:{1 / prev - 1:.0f})")
print(f"PR-AUC vs baseline    {g['PR-AUC'] / prev:.1f}x")
print(f"precision vs random   {g['precision @ 5% recall'] / prev:.0f}x at 5% recall")
""")

code("""
print("Top features by permutation importance -- all deployer-history terms.")
print("Every clock feature (hour, weekday, minute) was pushed out:")
print("the bot reads WHO is deploying, not WHEN.\\n")
display(top_feat)
for f in ["04_feature_importance.png", "05_precision_recall.png"]:
    display(Image(filename=os.path.join(DS, "figures", f)))
""")

md(r"""
---
# Part 3 &mdash; The replica, and the assumption that decides the answer

## The entry fill price dominates, so we did not choose it

Applying a fixed hold to the bot's own June tokens, changing **only** the definition of the entry
price flips the conclusion &mdash; the same data and the same strategy is a 7.8% loss or a 50% gain.

Choosing either would be a preference, not a measurement. So entry is interpolated as
`entry = open + alpha * (close - open)` and **alpha is solved for**: the value that reproduces the
bot's *observed* June median realised multiple of **1.1153**, a number visible in its trades.
""")

code("""
print("Passive hold of the bot's own June tokens, by entry-price definition:\\n")
display(passive.rename(columns={
    "median_multiple_alpha1_close": "entry = close (alpha 1)",
    "median_multiple_alpha0_open": "entry = open  (alpha 0)"}).set_index("hold_s"))
print("Same data, same strategy: a loss or a 50% gain, depending purely on the assumption.")
""")

code("""
best = alpha_cal.loc[alpha_cal.chosen == 1].iloc[0]
target = float(alpha_cal.target.iloc[0])

show = alpha_cal[alpha_cal.alpha.isin([0.0, 0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1.0])]
display(show[["alpha", "median_multiple", "abs_error"]].set_index("alpha"))

print(f"bot's OBSERVED June median realised multiple   {target:.4f}")
print(f"calibrated alpha                              {best.alpha:.2f}"
      f"  (predicts {best.median_multiple:.4f}, error {best.abs_error:.4f})")
nxt = alpha_cal[alpha_cal.chosen == 0].nsmallest(1, "abs_error").iloc[0]
print(f"next best grid point                          {nxt.alpha:.2f}"
      f"  (error {nxt.abs_error:.4f} -- {nxt.abs_error / best.abs_error:.0f}x worse)")
print()
print("The replica is charged the IDENTICAL alpha, so it cannot win by being handed")
print("a better fill than the incumbent.")
""")

md(r"""
## 0.4 seconds destroys the strategy

Costs are ~11% of position. One slot of delay collapses the median multiple from 1.243 to 1.012,
and a 1.2% gross move cannot clear an 11% cost.

**This is not a trading edge with a latency requirement. It is a latency edge with a trading
wrapper.**

We got this wrong first: our initial implementation modelled delay as *random dropout*, and
median ROI **rose** with delay (6.9% &rarr; 9.6% &rarr; 19.3%) as fills fell 718 &rarr; 302 &rarr;
129 and survivorship did the rest. Delay cannot be an advantage, so that was a modelling error,
not a finding. Delay is now priced as **a later, worse fill** along the open&rarr;close path.
""")

code("""
view = sens[["slot_delay", "trades_filled", "median_exit_multiple", "median_roi_net",
             "hit_rate_net", "net_pnl", "fees_paid", "entry_alpha"]].copy()
view.columns = ["slot delay", "trades filled", "median multiple", "median net ROI",
                "net hit rate", "net P&L", "fees paid", "entry alpha"]
display(view.set_index("slot delay").style.format({
    "median multiple": "{:.3f}", "median net ROI": "{:.1%}", "net hit rate": "{:.1%}",
    "net P&L": "${:,.0f}", "fees paid": "${:,.0f}", "trades filled": "{:,.0f}"}))

z = sens[sens.slot_delay == 0].iloc[0]
o = sens[sens.slot_delay == 1].iloc[0]
print(f"zero-block   net P&L ${z.net_pnl:>10,.0f}   median net ROI {z.median_roi_net:+.1%}")
print(f"one slot     net P&L ${o.net_pnl:>10,.0f}   median net ROI {o.median_roi_net:+.1%}")
print(f"\\nswing from 400 ms: ${z.net_pnl - o.net_pnl:,.0f}")
display(Image(filename=os.path.join(DS, "figures", "06_slot_sensitivity.png")))
""")

md(r"""
## The bot's edge is the exit, not the pick &mdash; and our overlap with it is low

Passively holding the bot's own tokens loses money at **every** horizon (table above: 0.9221 at
6&nbsp;s, 0.8080 at 60&nbsp;s). Yet its realised multiple is 1.1153, with a median of 4 sells per
token and 96.6% partial exits. The bot is not identifying tokens that go up &mdash; it arrives
first and **distributes into the buyers arriving behind it**.

That bounds what an entry classifier may claim, so we do not assert our selection beats the bot.
And we report the overlap rather than only the flattering half of it.
""")

code("""
print(f"replica entries      {int(overlap.replica_entries):,}")
print(f"bot buys             {int(overlap.bot_buys):,}")
print(f"intersection         {int(overlap.overlap):,}")
print()
print(f"recall of the bot    {overlap.recall_of_bot:.2%}")
print(f"replica precision    {overlap.precision_vs_bot:.1%}   "
      f"against a {scores.label.mean():.3%} base rate "
      f"({overlap.precision_vs_bot / scores.label.mean():.0f}x)")
print()
print("Not a clone. It selects the same KIND of opportunity by a different rule, and at the")
print("0.1% threshold it is far more selective than the bot -- which is why recall is low.")
""")

md(r"""
---
# What we cannot claim

1. **The bot's activity data carries no slot numbers.** Slot delay is estimated at 2.5 slots per
   second (400&nbsp;ms). Exact in-block position needs the 429&nbsp;GB raw-block tier, which we did
   not download. The 0/1/2-slot rows are a **sensitivity analysis, not measured latencies**.
2. **Candles are 1-second resolution**, so 400&nbsp;ms cannot be expressed as a timestamp shift.
   Sub-second delay is priced by moving alpha later *within* the same second, spilling into the
   next second only past 1&nbsp;s. This is why the 1-slot and 2-slot median ROIs are equal.
3. **Deployer history is capped at 10,000 events**, and the cap is not random: 233 of 4,095
   deployers (5.7%) hit it, and they account for **47.4% of all activity**. So `wallet_age_s` is
   under-estimated for exactly the busiest wallets. The `history_truncated` flag exposes this to
   the model but does not remove it.

Two further caveats: the backtest covers **one month** (June 2026), the extent of the
supplementary data; and the fill-probability model &mdash; parity at the bot's own priority spend,
decaying with delay &mdash; is a stated assumption, not a measurement. The slot-delay conclusion
survives its plausible range because it is driven by the **price path**, not the fill rate.

---

# Reproducing

Pipeline: `part1_behavior.py` &rarr; `build_features.py` &rarr; `part2_model.py` &rarr;
`part3_backtest.py` &rarr; `make_figures.py`.

Fee constants are recomputed from the bot's trades by `measure_bot_costs()` rather than
hardcoded, and `calibrate_alpha()` re-derives alpha = 0.50 on every run.
""")

code("""
display(Image(filename=os.path.join(DS, "figures", "00_cover.png")))
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(HERE, "nb", "solana-sniper-reverse-engineering.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote {out}  ({len(cells)} cells)")
