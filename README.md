# Reverse-engineering a live Solana pump.fun sniper

Analysis, feature reconstruction and a backtested replica of the sniper bot
`5brv79eFZ2rGprXNvqgVJBkBptkkw8GJX1XydJyZLyAr`, submitted to the Kaggle competition
[solana-sniper-bot-reverse-engineering](https://www.kaggle.com/competitions/solana-sniper-bot-reverse-engineering).

**Headline result:** the bot's strategy is worth `+$29,142` over June 2026 if you land in the
deploy block, and `-$334` if you land **one slot (400 ms) later**. Fees are 37% of gross P&L.
Both numbers are measured, not assumed.

---

## The disqualification rule, and how the code enforces it

Any data timestamped after `t_decision` (the token's deployment) that reaches a **feature** in
Part 2 or Part 3 zeroes both parts. Post-hoc data (`pumpfun_trades`, `mcap_candles`) is permitted
only for outcome labels and backtest P&L.

This is enforced architecturally in [`features.py`](features.py), not by convention:

1. `TruncatedHistory` is a **frozen dataclass** whose constructor physically filters
   `timestamp < cutoff`. There is no accessor that reaches past the cutoff.
2. Every feature function takes a `TruncatedHistory`, never a raw frame — a feature cannot leak
   even by mistake.
3. `audit_truncation()` re-reads the *original* unfiltered activity and asserts no feature row
   could have been derived from a post-cutoff event. The public notebook prints its output:
   **0 violations**.
4. The only permitted exception is the deployment transaction itself (dev-buy, fees, metadata),
   which is *at* `t_decision`, not after it.

Three columns are deliberately excluded from the feature set: `deploy_slot`, `deploy_time` and
`neg_sample_scale`. See "Leakage we caught" below.

---

## Reproducing

```
python part1_behavior.py     # -> part1_summary.csv, part1_entries.csv, part1_pnl_per_token.csv
python build_features.py     # -> features_all.parquet   (1,098,455 rows x 24 features)
python part2_model.py        # -> part2_top_features.csv, part2_test_scores.parquet
python part3_backtest.py     # -> part3_slot_sensitivity.csv, part3_overlap.csv
python make_figures.py       # -> figures/00_cover.png ... figures/06_slot_sensitivity.png
```

Requires `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `matplotlib`.

### Data

The raw corpus is ~83 GB and is not redistributed here. It is fetched from the competition's
data servers — core `65.21.203.147:48102`, supplementary `154.12.118.112:48110-48114` — and is
expected under `data/`:

| Path | Contents |
|---|---|
| `data/wallet/5brv79e_activity.parquet` | the target bot's own activity (12 MB) |
| `data/extracted/bought_deploy_txs_index.parquet` | deployments the bot bought (positives) |
| `data/extracted/bought_deployers_activity.parquet` | deployer history for positives |
| `data/extracted/not_bought_*` | 5.06M deployments it did not buy (negatives) |
| `data/supp/` | C3 `pumpfun_trades`, C4 `mcap_candles` — June 2026 only, outcomes/P&L only |

C1 (429 GB raw blocks) is deliberately skipped; it would only refine in-block position.

The intermediate artifacts every script consumes are published as a Kaggle dataset so the
notebook runs without the 83 GB corpus:
**`aleaiest/solana-sniper-bot-reverse-engineering-data`**.

---

## What each script does

| File | Role |
|---|---|
| `part1_behavior.py` | Scale, entry latency, exit pattern, and P&L **with full fee accounting** |
| `features.py` | `t_decision`-safe feature builder + `audit_truncation()` — the leakage defence |
| `features_fast.py` | Vectorised as-of index (167 → 5,801 rows/s), verified identical to `features.py` |
| `build_features.py` | Builds the full feature table for positives and sampled negatives |
| `part2_model.py` | GBDT + logistic baseline, time split, PR-AUC, permutation importance |
| `part3_backtest.py` | Replica + fees + fill probability + slot-delay sweep, `calibrate_alpha()` |
| `make_figures.py` | Seven figures plus the cover image |

---

## Method decisions that change the answer

**Fees are charged, not ignored.** `measure_bot_costs()` recomputes entry priority spend
($26.66/buy), venue fee (1.1% of position) and exit gas ($0.116/sell) from the bot's own trades
rather than hardcoding them. `gas_native == priority_fee + tip_fee` holds at correlation 1.000,
and none of it is inside `cost_usd`. Charging fees moves the hit rate 78.8% → 59.3%.

**The entry fill price is calibrated against the bot's own record, not chosen.** Holding the
bot's June tokens for its median 6 s returns 0.9221x if you enter at the deploy second's close
and 1.5043x at its open — the same data, the same strategy, a loss or a 50% gain. So entry is
interpolated as `entry = open + alpha * (close - open)` and `calibrate_alpha()` solves for the
alpha that reproduces the bot's **observed** June median realised multiple of 1.1153 →
**alpha = 0.50**. The replica is charged the identical alpha, so it cannot win by being handed a
better fill than the incumbent.

**Slot delay is modelled as a worse fill, not as a random dropout.** Our first implementation
dropped trades at random as delay grew, and median ROI *rose* (6.9% → 9.6% → 19.3%) because the
fill count fell 718 → 302 → 129 and survivorship did the rest. Delay cannot be an advantage;
that was a modelling error. It is now expressed as filling later along the open→close path.

**Negatives outside the bot's active window are removed.** The negative deployments span
2026-01-01 to 06-30 but the bot's first buy is 2026-03-12. Keeping January negatives teaches the
model the calendar, not the selection rule. `BOT_ACTIVE_FROM` drops them; imbalance 1:318 → 1:189.

**The test period is pinned to a date, not a quantile.** The supplementary candle/trade data
covers June 2026 only, so the split is train 2026-03-12 → 05-31, test = backtest 2026-06-01 →
06-30. The replica never trades on data it was trained on, and the backtest never needs data
that does not exist.

---

## Leakage we caught

The first Part 2 run returned ROC-AUC of exactly 0.5000 with every feature importance at
0.00000. The cause was a bookkeeping column, `neg_sample_scale`, left in the feature matrix: in
the training split it separates the classes perfectly (20.0 for negatives, 1.0 for positives)
and in the test split it is constant at 1.0, so the model learned it and emitted a constant.
`deploy_slot` and `deploy_time` are excluded for the same class of reason — both increase
monotonically with time, making them pure period indicators under a temporal split.

Three re-checks now run and pass: no column separates the training classes perfectly, no column's
mean shifts more than 2x between train and test, and the strongest single-feature AUC is 0.769
(`secs_since_last`) — well under a 0.95 alarm threshold.

---

## Known limits

1. **No slot numbers in the bot's activity data.** Slot delay is estimated at 2.5 slots/second
   (400 ms). Exact in-block position requires C1 (429 GB), which we did not download.
2. **Candles are 1-second resolution**, so 400 ms cannot be expressed as a timestamp. Sub-second
   delay is expressed as a worse fill within the same second instead.
3. **Deployer activity is capped at the 10,000 most recent events.** 233 of 4,095 deployers
   (5.7%) hit the cap, and those 233 account for 47.4% of all activity, so `wallet_age_s` is
   under-estimated for exactly the busiest wallets. The `history_truncated` flag is a feature, so
   the model can tell a measurement from a lower bound.

## Result summary

| | |
|---|---|
| Bot, gross | $1,488,270 · 78.8% hit rate · 13.3% median ROI |
| Fees | $550,345 (37.0% of gross) |
| Bot, net | $937,925 · 59.3% hit rate · 4.7% median ROI |
| Part 2 GBDT | PR-AUC 0.0761 (15.5x prevalence baseline), ROC-AUC 0.891, 17.6% precision at 5% recall |
| Replica, 0-slot delay | +$29,142 net, 56.0% net hit rate, +6.9% median net ROI |
| Replica, 1-slot delay | -$334 net, 28.5% net hit rate, -15.4% median net ROI |
| Overlap with the bot | 148 of 853 replica entries (17.4% precision, 0.92% recall of the bot) |

The replica is not a clone. It selects the same *kind* of opportunity differently, and we report
that rather than hiding it.

## License

MIT for the code. Data belongs to the competition organisers and is not redistributed.
