"""Vectorised t_decision-safe feature construction.

`features.py` builds one row at a time, which is clear but runs at ~170 deploys/second — eight
hours for the 3.0M negatives in scope. This module computes the same quantities with an as-of
aggregation: activity is sorted by (wallet, timestamp) once, per-wallet cumulative sums are
precomputed, and each deployment's history is then a pair of array lookups.

The truncation guarantee is preserved and is in fact easier to verify here: every aggregate is
`cumsum[start + k] - cumsum[start]` where `k = searchsorted(timestamps[start:end], cutoff,
side="left")`. Because `side="left"`, a row whose timestamp equals the cutoff is excluded, so
the window is strictly `timestamp < t_decision`. `verify_against_reference()` cross-checks a
random sample against the row-at-a-time implementation.
"""
import numpy as np
import pandas as pd

EVENT_TYPES = ["launch", "buy", "sell", "burn", "claim_fee"]
ACTIVITY_CAP = 9999          # provider keeps only the most recent 10,000 rows per deployer


class AsOfIndex:
    """Per-wallet cumulative aggregates enabling O(1) 'history strictly before t' queries."""

    def __init__(self, activity):
        a = activity.sort_values(["wallet", "timestamp"], kind="mergesort").reset_index(drop=True)
        self.wallets = a.wallet.to_numpy()
        self.ts = a.timestamp.to_numpy(dtype=np.int64)
        self.tokens = a.token_address.to_numpy()

        # Wallet block boundaries in the sorted array.
        uniq, starts = np.unique(self.wallets, return_index=True)
        order = np.argsort(starts)
        self.uniq = uniq[order]
        self.starts = starts[order]
        self.ends = np.append(self.starts[1:], len(a))
        self.wallet_pos = {w: i for i, w in enumerate(self.uniq)}

        # Cumulative sums with a leading zero so a window is cum[j] - cum[i].
        def cum(mask):
            return np.concatenate([[0], np.cumsum(mask.astype(np.int64))])

        et = a.event_type.to_numpy()
        self.cum_by_type = {t: cum(et == t) for t in EVENT_TYPES}
        self.cum_all = cum(np.ones(len(a), dtype=bool))
        cost = pd.to_numeric(a.get("cost_usd"), errors="coerce").fillna(0).to_numpy() \
            if "cost_usd" in a.columns else np.zeros(len(a))
        self.cum_cost = np.concatenate([[0.0], np.cumsum(cost)])
        self.cum_cost_buy = np.concatenate([[0.0], np.cumsum(np.where(et == "buy", cost, 0.0))])
        self._activity_len = len(a)

    def slice_for(self, wallet):
        i = self.wallet_pos.get(wallet)
        if i is None:
            return None
        return int(self.starts[i]), int(self.ends[i])

    def count_before(self, start, end, cutoff):
        """Number of rows in [start, end) with timestamp strictly below cutoff."""
        return int(np.searchsorted(self.ts[start:end], cutoff, side="left"))


def build(deploys, activity, label, reference_check=500, seed=0):
    """One row per deployment; features only from strictly before t_decision.

    deploys must carry token_address, deploy_time, tx_signer.
    """
    idx = AsOfIndex(activity)
    n = len(deploys)

    wallets = deploys.tx_signer.to_numpy()
    cutoffs = deploys.deploy_time.to_numpy(dtype=np.int64)

    # Resolve each deployment to its wallet's block and the count of prior rows.
    start = np.full(n, -1, dtype=np.int64)
    end = np.full(n, -1, dtype=np.int64)
    for i, w in enumerate(wallets):
        s = idx.slice_for(w)
        if s is not None:
            start[i], end[i] = s
    known = start >= 0
    k = np.zeros(n, dtype=np.int64)
    for i in np.flatnonzero(known):
        k[i] = idx.count_before(start[i], end[i], cutoffs[i])
    lo = np.where(known, start, 0)
    hi = lo + k                                   # exclusive end of the truncated window

    out = pd.DataFrame({
        "token_address": deploys.token_address.to_numpy(),
        "wallet": wallets,
        "deploy_time": cutoffs,
        "label": label,
        "hist_events": k,
    })

    for t in EVENT_TYPES:
        c = idx.cum_by_type[t]
        out[f"hist_{t}s" if t != "claim_fee" else "hist_claim_fees"] = c[hi] - c[lo]
    out = out.rename(columns={"hist_launchs": "hist_launches", "hist_burns": "hist_burns"})

    total_cost = idx.cum_cost[hi] - idx.cum_cost[lo]
    buy_cost = idx.cum_cost_buy[hi] - idx.cum_cost_buy[lo]
    out["total_volume_usd"] = total_cost
    out["mean_buy_usd"] = np.where(out.hist_buys > 0, buy_cost / np.maximum(out.hist_buys, 1), 0.0)

    # First/last timestamps inside the truncated window.
    first_ts = np.where(k > 0, idx.ts[np.minimum(lo, len(idx.ts) - 1)], cutoffs)
    last_ts = np.where(k > 0, idx.ts[np.clip(hi - 1, 0, len(idx.ts) - 1)], cutoffs)
    out["wallet_age_s"] = np.where(k > 0, cutoffs - first_ts, 0)
    out["secs_since_last"] = np.where(k > 0, cutoffs - last_ts, -1)

    days = np.maximum(out.wallet_age_s / 86400.0, 1e-9)
    out["launch_rate_per_day"] = out.hist_launches / days
    out["sell_to_buy_ratio"] = out.hist_sells / np.maximum(out.hist_buys, 1)

    # Recent-window launch counts need a second searchsorted per row.
    for label_, window in [("launches_last_24h", 86400), ("launches_last_7d", 7 * 86400)]:
        c = idx.cum_by_type["launch"]
        lo2 = np.zeros(n, dtype=np.int64)
        for i in np.flatnonzero(known):
            j = idx.count_before(start[i], end[i], cutoffs[i] - window)
            lo2[i] = start[i] + j
        out[label_] = c[hi] - c[np.minimum(lo2, hi)]

    out["history_truncated"] = (k >= ACTIVITY_CAP).astype(int)

    dt = pd.to_datetime(cutoffs, unit="s", utc=True)
    out["hour_utc"] = dt.hour
    out["dow"] = dt.dayofweek
    out["minute_of_hour"] = dt.minute
    if "blockSlot" in deploys.columns:
        out["deploy_slot"] = deploys.blockSlot.to_numpy()

    if reference_check:
        verify_truncation(out, idx, sample=reference_check, seed=seed)
    return out


def verify_truncation(features, idx: AsOfIndex, sample=500, seed=0):
    """Assert the window used for every sampled row contains nothing at/after its cutoff."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(features), size=min(sample, len(features)), replace=False)
    violations = []
    for i in rows:
        r = features.iloc[int(i)]
        s = idx.slice_for(r.wallet)
        if s is None or r.hist_events == 0:
            continue
        lo, _ = s
        window_ts = idx.ts[lo:lo + int(r.hist_events)]
        if len(window_ts) and int(window_ts.max()) >= int(r.deploy_time):
            violations.append((r.wallet, int(r.deploy_time), int(window_ts.max())))
    assert not violations, f"t_decision leakage in {len(violations)} sampled rows: {violations[:3]}"
    return {"sampled": len(rows), "violations": 0}


def verify_against_reference(deploys, activity, n=200, seed=0):
    """Cross-check the vectorised path against the row-at-a-time implementation."""
    import features as ref

    sub = deploys.sample(min(n, len(deploys)), random_state=seed)
    fast = build(sub, activity, label=1, reference_check=0).set_index("token_address")
    slow, _ = ref.build_feature_table(sub, activity, label=1, progress_every=0)
    slow = slow.set_index("token_address")

    shared = ["hist_events", "hist_launches", "hist_buys", "hist_sells",
              "wallet_age_s", "secs_since_last", "launches_last_24h", "launches_last_7d"]
    report = {}
    for c in shared:
        if c in fast.columns and c in slow.columns:
            a, b = fast[c].astype(float), slow.loc[fast.index, c].astype(float)
            report[c] = int((a != b).sum())
    return report
