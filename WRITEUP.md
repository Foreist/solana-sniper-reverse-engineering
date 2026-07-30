# The edge is 400 milliseconds wide

**Reverse-engineering the pump.fun sniper `5brv79eFZ2rGprXNvqgVJBkBptkkw8GJX1XydJyZLyAr` — and pricing it honestly.**

The bot made **$1.49M gross** over 3.5 months and kept **$937,925** after fees. Its strategy
survives at zero-block execution and dies one slot later: our replica earns **+$29,142** with a
0-slot fill and **−$334** at 1 slot (400 ms). That single number is the finding — everything
else is the work of establishing that it is real rather than an artefact of a flattering
assumption.

---

## Part 1 — What the bot actually does

**Scale.** 16,163 tokens bought across 16,192 buy events and 70,812 sell events, 2026-03-12
08:54 → 2026-06-30 23:29 UTC. Only 2 burns. 15,927 buys match a deployment in the index.

**Entry.** Median position **$183** (mean $263, CV 0.89, p05 $100 / p95 $727). Latency from
deployment to first buy is a **median of 0 seconds, with 90.7% at ≤0 s** and 98.5% at ≤1 s. The
36.4 s mean is dragged by a thin tail of late entries; the mode of this strategy is landing in
the deployment block itself.

**Exit.** Median **4 sells per token**, **96.6% partial exits**, median hold **6 seconds**
(p25 4 s, p75 9 s). Three tokens were never sold. This is second-scale scalping with laddered
distribution, not position-taking.

### Fees are 37% of gross, and ignoring them inverts the picture

```
GROSS    $1,488,270   hit rate 78.8%   median ROI 13.3%
FEES     gas (priority + Jito tip) $439,839 + venue $110,506 = $550,345   (37.0% of gross)
NET      $937,925     hit rate 59.3%   median ROI  4.7%
```

Omitting fees overstates the hit rate by **19.5 pp** and the median ROI by roughly **3×**. The
fee structure is asymmetric in a way that matters for everything downstream: **entry priority
costs $26.66 per buy — 10.1% of the average position — while exit gas is $0.116 per sell, a
230× difference.** Paying to be first is essentially the entire cost base.

We verified the accounting rather than assuming it: `gas_native == priority_fee + tip_fee` at
correlation 1.000 to five decimals, `dex_native/dex_usd` is a separate venue fee averaging $2.90
per buy (≈1.1% of position), and **neither is included in `cost_usd`** — so a P&L computed from
`cost_usd` alone is a gross figure. Net ROI deciles: p10 −17.7%, p25 −5.9%, **p50 +4.7%**,
p75 +26.8%, p90 +56.4%. Average win $117.25 against average loss −$28.16.

---

## Part 2 — Reconstructing the selection rule

### The disqualification rule is enforced by architecture, not discipline

Any post-`t_decision` data reaching a feature zeroes Parts 2 and 3. We did not rely on being
careful:

1. `TruncatedHistory` is a **frozen dataclass** whose constructor physically applies
   `timestamp < cutoff`. No accessor on it can reach past the cutoff.
2. Every feature function accepts a `TruncatedHistory`, never a raw frame. Leaking requires
   rewriting the type, not making a mistake.
3. `audit_truncation()` re-reads the **original unfiltered** activity and asserts that no feature
   row could have been derived from a post-cutoff event. The public notebook runs it live on a
   shipped sample that still contains post-cutoff rows — so the check has something it could
   fail on — and prints **0 violations**.

The one permitted exception is the deployment transaction itself (dev-buy, fees, metadata),
which is *at* `t_decision`, not after it.

### Three data problems that would each have produced a good-looking, worthless model

**Negatives outside the bot's active window are a calendar artefact.** Negative deployments span
2026-01-01 → 06-30, but the bot's first buy is 2026-03-12. Of 5,060,494 negatives, 2.04M fall
before the bot existed. Keeping them teaches the model *"nobody buys in January"* — a fact about
the sampling frame, not a selection rule. Dropping them moves the imbalance from 1:318 to 1:189
and costs headline metrics, which is the point.

**The test period is pinned to a date, not a quantile.** The supplementary trade and candle data
covers **June 2026 only**. A quantile split would have put the backtest partly outside the data
that can price it. Fixing the boundary at 06-01 makes the Part 2 holdout and the Part 3 backtest
the *same* interval: the replica never trades on what it learned from, and the backtest never
needs data that does not exist.

```
train 2026-03-12 → 05-31        test = backtest 2026-06-01 → 06-30
```

**Deployer history is capped and the cap is not random.** The provider keeps only the 10,000 most
recent events per deployer. 233 of 4,095 deployers (5.7%) hit that cap — and those 233 account
for **47.4% of all activity**. Their earliest surviving record has median date 2026-03-12 against
2026-04-06 for the rest, so `wallet_age_s` is systematically under-estimated for exactly the
busiest wallets. Rather than leave a silent error, `history_truncated` is a feature, letting the
model distinguish a measurement from a lower bound.

We also fixed the join key empirically: `creator_address` is **0% populated** and unusable. The
deploy index's `tx_signer` is the deployer wallet — present in the activity table 100% of the
time, and cross-checked against `launch` events at 100% agreement on 13,818 matches.

### The leakage we caught

Our first run returned **ROC-AUC of exactly 0.5000** with every feature importance at 0.00000.
The cause was a bookkeeping column, `neg_sample_scale`, left in the matrix: in training it
separates the classes perfectly (20.0 negative, 1.0 positive) and in test it is constant at 1.0,
so the model learned it and emitted a constant. `deploy_slot` and `deploy_time` are excluded for
the same reason — both increase monotonically with time, making them pure period indicators
under a temporal split.

Three re-checks now run in the public notebook. **No column separates the training classes
perfectly**, and the strongest single-feature AUC is **0.785** (`secs_since_last`) — well below an
alarm threshold of 0.95, so no single column is carrying the model.

The mean-shift check flags exactly one column, and we report it rather than loosening the
threshold: `secs_since_last` has a train mean of 121,595 s against 245,305 s in test, a **2.02×**
shift. This is corpus drift, not leakage — the recency gap of a dormant wallet mechanically grows
as the observation window extends, so any "time since last seen" feature must drift upward in a
later period, and every value is still computed strictly before its own cutoff. We keep it
because it ranks only **third** in permutation importance, behind two amount-based features, so
the fitted model is not leaning on the drifting term. It is the first thing we would monitor if
this ran forward.

### Results

Feature table: **1,098,455 rows × 24 columns, of which 18 are used as features** (identifiers,
the label, `neg_sample_scale`, `deploy_slot` and `deploy_time` are dropped). Training 11,732 positives against 234,640
negatives subsampled 20:1; test keeps the true prevalence — 4,195 positives against 847,888
negatives, an imbalance of 1:202 (prevalence 0.492%).

| Model | PR-AUC | ROC-AUC | Precision @ 5% recall |
|---|---|---|---|
| **GBDT** | **0.0761** | **0.891** | **17.6%** |
| Logistic | 0.0157 | 0.790 | 2.6% |
| Prevalence baseline | 0.00492 | 0.500 | 0.49% |

PR-AUC is **15.5× the prevalence baseline**, and in the top-5%-recall band precision is **36×**
random. Precision, recall and F1 at the two thresholds that matter:

| GBDT operating point | Precision | Recall | F1 |
|---|---|---|---|
| Max-F1 (threshold 0.414) | 12.2% | 23.7% | **0.161** |
| Strategy point, top 0.1% (threshold 0.720) | **17.4%** | 3.5% | 0.059 |

The strategy point is deliberately off the F1 optimum. F1 rewards balancing precision against
recall, but a sniper does not need to catch most launches — it needs the ones it does take to be
right, because every entry costs 10.1% of position in priority fees before it can win anything.
So we trade recall away for precision and accept the low F1 as the correct choice for the task.
The top features are entirely deployer-history terms:

`mean_buy_usd` › `total_volume_usd` › `secs_since_last` › `launch_rate_per_day` › `hist_events` ›
`sell_to_buy_ratio` › `launches_last_24h` › `hist_launches` › `hist_sells` › `wallet_age_s`

Every clock-based feature (hour, weekday, minute) was pushed out. The bot is reading **who is
deploying**, not when.

---

## Part 3 — The replica, and the assumption that decides the answer

### The entry fill price dominates everything, and we did not pick the flattering one

Applying a fixed hold to the bot's own 4,195 June tokens, changing *only* the definition of the
entry price flips the conclusion:

| Hold | Entry = deploy-second **close** | Entry = deploy-second **open** |
|---|---|---|
| 3 s | 0.9502 | 1.5775 |
| **6 s** | **0.9221** | **1.5043** |
| 30 s | 0.8651 | 1.3219 |

**Same data, same strategy: a 7.8% loss or a 50% gain.** Choosing either is not a measurement.

So we interpolate `entry = open + α·(close − open)` and **solve for the α that reproduces the
bot's observed June median realised multiple of 1.1153** — a number we can see in its trades
rather than one we get to choose. A 21-point sweep in steps of 0.05 lands on **α = 0.50**,
predicting 1.1099 for an absolute error of 0.0054; its neighbours α = 0.45 and α = 0.55 predict
1.1357 and 1.0846, missing by 0.0204 and 0.0307 — **4× and 6× worse**, so the optimum is a real
minimum rather than a flat region we picked a point out of. It is also physically sensible: in a
sniping race you are neither the first nor the last fill.

**The replica is charged the identical α.** It is therefore structurally impossible for our
strategy to win by being handed a better fill than the incumbent.

### 0.4 seconds destroys the strategy

Entering on the top 0.1% of GBDT scores gives 853 candidates in June, 852 resolvable from candles.

| Slot delay | Median multiple | Median net ROI | Total net ROI | Net hit rate | Net P&L | Max drawdown |
|---|---|---|---|---|---|---|
| **0** | **1.243** | **+6.9%** | **+22.1%** | 56.0% | **+$29,142** | **$445** |
| 1 (0.4 s) | 1.012 | −15.4% | −0.6% | 28.5% | −$334 | $2,026 |
| 2 (0.8 s) | 1.000 | −15.4% | −3.7% | 27.1% | −$883 | $1,803 |

Drawdown moves the wrong way faster than P&L does. One slot late the strategy earns nothing, but
its worst peak-to-trough deepens **4.6×** to $2,026 — it is not merely unprofitable, it is
unprofitable *and* more violent, on a third as many trades.

The mechanism is the fee asymmetry from Part 1. Costs are ~11% of position; one slot of delay
collapses the median multiple from 1.243 to 1.012, and a 1.2% gross move cannot clear an 11%
cost. **This strategy is not a trading edge with a latency requirement — it is a latency edge
with a trading wrapper.**

We got this wrong first. Our initial implementation modelled delay as **random dropout**, and
median ROI *rose* with delay (6.9% → 9.6% → 19.3%) as fills fell 718 → 302 → 129 and
survivorship did the rest. Delay cannot be an advantage, so that was a modelling error, not a
finding. Delay is now priced as **a later, worse fill** along the open→close path.

### The edge is the fill position, and neither selection nor exit explains it

Holding the bot's own tokens for its median 6 s returns **1.5043** entering at the deployment
second's open and **0.9221** entering at its close. The token spikes inside that first second and
then decays — 0.9221 at 6 s, 0.8651 at 30 s, 0.8080 at 60 s. So the return is not a property of
*which* token was bought; it is a property of **where in the deployment second the fill landed**.

We are careful about what follows from this, because our own method limits it. The bot's realised
multiple is **1.1153**, and it exits with a median of **4 sells per token** at **96.6% partial
exits** — which invites the conclusion that laddered distribution is the edge. It does not
follow. Our α was *solved for* by requiring a passive 6 s hold to reproduce 1.1153, and at the
resulting α = 0.50 a passive hold returns **1.1099**. Having forced those two quantities to agree,
we cannot then use their agreement to credit the exit: **the calibration makes "good fill" and
"good exit" indistinguishable by construction.** The most we can say is that the laddered exit
does not appear to add much over simply leaving after 6 s at the same fill.

What survives is the fill. That is the same quantity Part 3 puts a price on: one slot of delay
moves the median multiple from 1.243 to 1.012. Selection, exit structure and hold length are all
second-order next to **being early inside the deployment block**.

This bounds what an entry classifier may claim, so **we do not assert that our selection model
beats the bot.**

### Head to head against the bot, same month, same fee model

The bot's own June trades, priced with the same fee accounting, against the replica at zero-slot
delay:

| June 2026 | Bot (realised) | Replica, 0-slot (simulated) |
|---|---|---|
| Trades | 4,194 | 718 |
| Median position | $160 | $183 |
| Capital deployed | $971,151 | $131,694 |
| Gross P&L | $319,637 | $52,310 |
| Fees | $131,553 (41.2% of gross) | $23,168 (44.3%) |
| **Net P&L** | **$188,084** | **$29,142** |
| Net hit rate | 57.7% | 56.0% |
| Median net ROI | 3.6% | **6.9%** |
| **Total net ROI on capital** | **19.4%** | **22.1%** |
| Max drawdown | $1,131 (0.6% of net) | $445 (1.5% of net) |
| Token overlap | — | 148 of 853 entries |

The replica is **more efficient per dollar and far smaller in absolute terms**: entering on the
top 0.1% of scores, it takes 17% as many trades and returns 15% as much money, but earns 22.1%
on deployed capital against the bot's 19.4%, at nearly the same hit rate. That is the expected
shape of a more selective threshold, not evidence of a better strategy. Absolute drawdown is
smaller ($445 against $1,131) purely because the book is smaller; **relative to net profit the
replica is the rougher ride** — 1.5% against 0.6% — which is what running a quarter of the trade
count buys you.

**Two reasons not to read this as beating the bot.** The bot's column is *realised* and carries
every real execution cost; the replica's is *simulated*, and its fills are granted by our
probability model at the bot's own priority spend. And the replica's whole margin is contingent
on the zero-slot assumption — at one slot it returns −$334, while the bot's $188,084 is money it
actually kept.

### Overlap with the bot is low, and we report it

```
replica entries 853   |   bot buys 16,163   |   intersection 148
recall of the bot 0.92%   |   replica precision 17.4%
```

Our replica is not a clone. It selects the same *kind* of opportunity by a different rule. At the
0.1% threshold it is deliberately far more selective than the bot, which explains the low recall,
but 17.4% precision against a 0.49% base rate is the honest characterisation — and we prefer
publishing it to quoting only the flattering half.

---

## What we cannot claim

1. **The bot's activity data carries no slot numbers.** Slot delay is estimated at 2.5 slots per
   second (400 ms). Exact in-block position needs the 429 GB raw-block tier, which we did not
   download. The 0/1/2-slot rows are a sensitivity analysis, not measured latencies.
2. **Candles are 1-second resolution**, so 400 ms cannot be expressed as a timestamp shift.
   Sub-second delay is priced by moving α later *within* the same second, spilling into the next
   second only past 1 s. This is the honest treatment at this resolution, and it is why the 1-slot
   and 2-slot median ROIs are equal.
3. **Deployer history is capped at 10,000 events**, biasing `wallet_age_s` downward for the 5.7%
   of deployers who generate 47.4% of activity. `history_truncated` exposes this to the model but
   does not remove it.

Two further caveats: the backtest covers **one month**, June 2026, because that is the extent of
the supplementary data; and the fill-probability model — parity at the bot's own priority spend,
decaying with delay — is a stated assumption, not a measurement, though the slot-delay conclusion
holds across its plausible range because it is driven by the price path rather than the fill rate.

---

## Reproducing this

All code is in the repository; the public notebook re-runs the truncation audit, all three
leakage re-checks and the α calibration live from a published dataset, so the claims above can be
executed rather than taken on trust.

- **Code:** https://github.com/Foreist/solana-sniper-reverse-engineering
- **Notebook:** https://www.kaggle.com/code/aleaiest/the-edge-is-400-milliseconds-wide
- **Data:** https://www.kaggle.com/datasets/aleaiest/solana-sniper-bot-reverse-engineering-data

The pipeline is deterministic end to end: re-running it from the 83 GB corpus reproduces every
intermediate artifact **byte for byte** (all seven files match by MD5, including the 51 MB scored
test set produced by refitting the GBDT). The numbers in this write-up are what the code emits,
not figures transcribed from a run.

Pipeline: `part1_behavior.py` → `build_features.py` → `part2_model.py` → `part3_backtest.py` →
`make_figures.py`. Fee constants are recomputed from the bot's trades by `measure_bot_costs()`
rather than hardcoded, and `calibrate_alpha()` re-derives α = 0.50 on every run.
