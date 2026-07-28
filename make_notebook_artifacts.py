"""Produce the small artifacts the public notebook needs to re-run our checks live.

The raw corpus is 83 GB and cannot be uploaded, but two of the three checks a judge is asked to
see -- the truncation audit and the fill calibration -- are only convincing if they *execute*.
So we ship the inputs they need at a size that fits a Kaggle dataset:

  audit_sample_activity.parquet  raw, UNFILTERED activity for a sample of deployers, so
  audit_sample_deploys.parquet   `features.audit_truncation()` runs for real in the notebook
  part3_alpha_calibration.csv    the alpha sweep behind ENTRY_FILL_ALPHA = 0.50

Writes into ds_upload/ next to the other artifacts.
"""

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import part3_backtest as P3

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "ds_upload")

AUDIT_DEPLOYERS = 1500          # sampled deployers whose full history we ship
BOT_JUNE_START = 1780272000     # 2026-06-01 00:00 UTC
BOT_JUNE_END = 1782864000       # 2026-07-01 00:00 UTC


def build_audit_sample(seed=0):
    """Ship raw history for a sample of deployers so the audit is reproducible, not asserted."""
    deploys = pd.read_parquet(
        os.path.join(DATA, "extracted", "bought_deploy_txs_index.parquet"),
        columns=["token_address", "blockTime", "tx_signer"],
    ).rename(columns={"blockTime": "deploy_time", "tx_signer": "wallet"})
    deploys = deploys.dropna(subset=["wallet", "deploy_time"])
    deploys["deploy_time"] = pd.to_numeric(deploys.deploy_time, errors="coerce").astype("int64")

    rng = np.random.default_rng(seed)
    wallets = deploys.wallet.unique()
    keep = set(rng.choice(wallets, size=min(AUDIT_DEPLOYERS, len(wallets)), replace=False))
    d = deploys[deploys.wallet.isin(keep)].copy()

    act = pd.read_parquet(
        os.path.join(DATA, "extracted", "bought_deployers_activity.parquet"),
        columns=["wallet", "timestamp", "event_type", "token_address"],
    )
    act["timestamp"] = pd.to_numeric(act.timestamp, errors="coerce")
    act = act.dropna(subset=["timestamp"])
    act["timestamp"] = act.timestamp.astype("int64")
    a = act[act.wallet.isin(keep)].copy()

    d.to_parquet(os.path.join(OUT, "audit_sample_deploys.parquet"), index=False)
    a.to_parquet(os.path.join(OUT, "audit_sample_activity.parquet"), index=False)
    print(f"audit sample: {len(d):,} deploys, {len(a):,} activity rows, "
          f"{d.wallet.nunique():,} deployers")
    # The point of shipping the *unfiltered* frame: it still contains post-cutoff events, so the
    # audit has something it could have failed on.
    later = sum(int((a[a.wallet == w].timestamp >= c).sum())
                for w, c in zip(d.wallet.head(300), d.deploy_time.head(300)))
    print(f"post-cutoff events present in the shipped sample (first 300 deploys): {later:,}")


def build_alpha_calibration():
    """Re-derive ENTRY_FILL_ALPHA from the bot's own June trades."""
    wallet = pd.read_parquet(os.path.join(DATA, "wallet", "5brv79e_activity.parquet"))
    for c in ["cost_usd", "gas_usd", "dex_usd", "timestamp"]:
        if c in wallet.columns:
            wallet[c] = pd.to_numeric(wallet[c], errors="coerce")

    buys = wallet[wallet.event_type == "buy"].dropna(subset=["timestamp"])
    first = buys.groupby("token_address").timestamp.min()
    june = first[(first >= BOT_JUNE_START) & (first < BOT_JUNE_END)]
    entry_ts = {k: int(v) for k, v in june.items()}
    print(f"bot June tokens: {len(entry_ts):,}")

    wanted = set(entry_ts)
    pf = pq.ParquetFile(os.path.join(DATA, "supp", "mcap_candles.parquet"))
    parts = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=P3.CANDLE_COLS).to_pandas()
        t = t[t.token_address.isin(wanted)]
        if len(t):
            parts.append(t)
    candles = pd.concat(parts, ignore_index=True)
    print(f"candles kept: {len(candles):,} rows, {candles.token_address.nunique():,} tokens")

    best, table = P3.calibrate_alpha(candles, entry_ts, P3.BOT_JUNE_MEDIAN_MULTIPLE)
    table["target"] = P3.BOT_JUNE_MEDIAN_MULTIPLE
    table["abs_error"] = (table.median_multiple - P3.BOT_JUNE_MEDIAN_MULTIPLE).abs()
    table["chosen"] = (table.alpha == best).astype(int)
    table.to_csv(os.path.join(OUT, "part3_alpha_calibration.csv"), index=False)
    print(f"calibrated alpha = {best}")
    print(table.to_string(index=False))

    # The same sweep is the evidence for finding (4): passively holding the bot's own tokens
    # loses money at every horizon, so its edge is not token selection.
    rows = []
    for hold in [3, 6, 10, 30, 60]:
        m = P3.outcome_from_candles(candles, entry_ts, hold, alpha=1.0).median()
        o = P3.outcome_from_candles(candles, entry_ts, hold, alpha=0.0).median()
        rows.append({"hold_s": hold, "median_multiple_alpha1_close": float(m),
                     "median_multiple_alpha0_open": float(o)})
    hold_tbl = pd.DataFrame(rows)
    hold_tbl.to_csv(os.path.join(OUT, "part3_passive_hold.csv"), index=False)
    print(hold_tbl.to_string(index=False))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build_audit_sample()
    build_alpha_calibration()
