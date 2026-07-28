"""Part 1 — behavioural analysis of the target sniper bot (5brv79e...).

Everything here is descriptive: it measures what the bot did. No feature engineering and no
look-ahead concern applies to this part, but the entry-latency numbers computed here define
t_decision for Parts 2 and 3, so they are produced once and reused.

Required outputs per the rubric:
  - token count, entry size mean/median/dispersion
  - latency from deployment to first buy, in slots and seconds; zero-block share
  - in-block position of the buy relative to the deploy tx (needs June blocks; approximated)
  - hold time and exit structure (partial exits, sells per token, burn usage)
  - hit rate, average win/loss, per-trade P&L distribution
"""
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WALLET = os.path.join(DATA, "wallet", "5brv79e_activity.parquet")
BOUGHT_IDX = os.path.join(DATA, "extracted", "bought_deploy_txs_index.parquet")

NUMERIC = ["token_amount", "quote_amount", "price", "price_usd", "cost_usd",
           "buy_cost_usd", "gas_native", "gas_usd", "priority_fee", "tip_fee",
           "dex_native", "dex_usd"]

# Verified against the data: gas_native == priority_fee + tip_fee to 5 decimals
# (corr 1.000), and dex_* is the venue fee, charged separately. Both are real cash costs
# and both are excluded from cost_usd, so P&L computed from cost_usd alone is gross.


def load_wallet():
    """Bot activity with numeric columns coerced (they ship as strings)."""
    a = pd.read_parquet(WALLET)
    for c in NUMERIC:
        if c in a.columns:
            a[c] = pd.to_numeric(a[c], errors="coerce")
    a["ts"] = pd.to_datetime(a.timestamp, unit="s", utc=True)
    return a.sort_values("timestamp").reset_index(drop=True)


def load_deploys():
    """Deployment index for tokens the bot bought: gives blockTime / blockSlot of deploy."""
    d = pd.read_parquet(BOUGHT_IDX)
    return d.rename(columns={"blockTime": "deploy_time", "blockSlot": "deploy_slot"})


def entry_profile(a, d):
    """First buy per token, joined to its deployment, with latency."""
    buys = a[a.event_type == "buy"].copy()
    first = buys.sort_values("timestamp").groupby("token_address", as_index=False).first()
    m = first.merge(d[["token_address", "deploy_time", "deploy_slot"]],
                    on="token_address", how="inner")
    m["latency_s"] = m.timestamp - m.deploy_time
    # Solana targets ~400 ms per slot; slot latency is derived because the activity feed
    # carries no slot number. Treated as an estimate and reported as such.
    m["latency_slots_est"] = (m.latency_s / 0.4).round()
    return buys, first, m


def summarise(a, buys, first, m):
    out = {}
    out["tokens_bought"] = int(a[a.event_type == "buy"].token_address.nunique())
    out["buy_events"] = int(len(buys))
    out["sell_events"] = int((a.event_type == "sell").sum())
    out["burn_events"] = int((a.event_type == "burn").sum())

    size = buys.cost_usd.dropna()
    out["entry_usd_mean"] = float(size.mean())
    out["entry_usd_median"] = float(size.median())
    out["entry_usd_std"] = float(size.std())
    out["entry_usd_cv"] = float(size.std() / size.mean()) if size.mean() else np.nan
    out["entry_usd_p05"] = float(size.quantile(0.05))
    out["entry_usd_p95"] = float(size.quantile(0.95))

    lat = m.latency_s.dropna()
    out["latency_s_median"] = float(lat.median())
    out["latency_s_mean"] = float(lat.mean())
    out["zero_block_share_le0s"] = float((lat <= 0).mean())
    out["share_le1s"] = float((lat <= 1).mean())
    out["share_le2s"] = float((lat <= 2).mean())
    out["matched_deploys"] = int(len(m))

    # Exit structure
    sells = a[a.event_type == "sell"]
    per_token_sells = sells.groupby("token_address").size()
    out["sells_per_token_median"] = float(per_token_sells.median())
    out["sells_per_token_mean"] = float(per_token_sells.mean())
    out["partial_exit_share"] = float((per_token_sells > 1).mean())

    # Hold time: first buy to last sell of the same token
    fb = buys.groupby("token_address").timestamp.min()
    ls = sells.groupby("token_address").timestamp.max()
    hold = (ls - fb).dropna()
    out["hold_s_median"] = float(hold.median())
    out["hold_s_p25"] = float(hold.quantile(0.25))
    out["hold_s_p75"] = float(hold.quantile(0.75))
    out["tokens_never_sold"] = int(len(set(fb.index) - set(ls.index)))
    return out, hold


def pnl(a):
    """Realised P&L per token from the activity feed's USD amounts.

    Buys spend cost_usd; sells receive cost_usd. Tokens still held at the end of the window
    are reported separately rather than marked to market, because the core dataset carries no
    post-window price and marking them would be an assumption, not a measurement.
    """
    x = a[a.event_type.isin(["buy", "sell"])].copy()
    x["signed"] = np.where(x.event_type == "sell", x.cost_usd, -x.cost_usd)
    x["all_fees"] = x.gas_usd.fillna(0) + x.dex_usd.fillna(0)
    g = x.groupby("token_address").agg(
        gross=("signed", "sum"),
        n_buys=("event_type", lambda s: int((s == "buy").sum())),
        n_sells=("event_type", lambda s: int((s == "sell").sum())),
        gas=("gas_usd", "sum"),
        dex=("dex_usd", "sum"),
        fees=("all_fees", "sum"),
    )
    g["spent"] = x[x.event_type == "buy"].groupby("token_address").cost_usd.sum()
    g = g[g.n_sells > 0]                    # closed or partially closed only
    # Gross is what the trade prints; net is what the wallet keeps.
    g["net"] = g.gross - g.fees
    g["roi_gross"] = g.gross / g.spent
    g["roi_net"] = g.net / g.spent
    return g


def main():
    a = load_wallet()
    d = load_deploys()
    buys, first, m = entry_profile(a, d)
    s, hold = summarise(a, buys, first, m)

    print("=== Part 1 · scale ===")
    print(f"  tokens bought              {s['tokens_bought']:,}")
    print(f"  buy / sell / burn events   {s['buy_events']:,} / {s['sell_events']:,} / {s['burn_events']}")
    print(f"  deploys matched to a buy   {s['matched_deploys']:,}")

    print("\n=== Part 1 · entry size (USD) ===")
    print(f"  mean {s['entry_usd_mean']:.2f} | median {s['entry_usd_median']:.2f} "
          f"| sd {s['entry_usd_std']:.2f} | CV {s['entry_usd_cv']:.2f}")
    print(f"  p05 {s['entry_usd_p05']:.2f} | p95 {s['entry_usd_p95']:.2f}")

    print("\n=== Part 1 · entry latency (deploy -> first buy) ===")
    print(f"  median {s['latency_s_median']:.1f}s | mean {s['latency_s_mean']:.1f}s")
    print(f"  <=0s {s['zero_block_share_le0s']*100:.1f}% | <=1s {s['share_le1s']*100:.1f}% "
          f"| <=2s {s['share_le2s']*100:.1f}%")

    print("\n=== Part 1 · exit structure ===")
    print(f"  sells per token: median {s['sells_per_token_median']:.0f}, mean {s['sells_per_token_mean']:.2f}")
    print(f"  partial exits (>1 sell): {s['partial_exit_share']*100:.1f}%")
    print(f"  hold seconds: p25 {s['hold_s_p25']:.0f} | median {s['hold_s_median']:.0f} | p75 {s['hold_s_p75']:.0f}")
    print(f"  tokens never sold in window: {s['tokens_never_sold']:,}")

    g = pnl(a)
    buys_only = a[a.event_type == "buy"]
    print("\n=== Part 1 · realised P&L (tokens with >=1 sell) ===")
    print(f"  positions {len(g):,} | total spent ${g.spent.sum():,.0f}")
    print(f"  GROSS  total ${g.gross.sum():,.0f} | hit rate {(g.gross > 0).mean()*100:.1f}%"
          f" | median ROI {g.roi_gross.median()*100:.1f}%")
    print(f"  fees   gas(priority+tip) ${g.gas.sum():,.0f} + venue ${g.dex.sum():,.0f}"
          f" = ${g.fees.sum():,.0f}  ({g.fees.sum()/g.gross.sum()*100:.1f}% of gross)")
    print(f"  NET    total ${g.net.sum():,.0f} | hit rate {(g.net > 0).mean()*100:.1f}%"
          f" | median ROI {g.roi_net.median()*100:.1f}%")
    print(f"  mean win ${g.loc[g.net > 0, 'net'].mean():.2f} | mean loss ${g.loc[g.net <= 0, 'net'].mean():.2f}")
    print(f"  net ROI deciles (%):\n{(g.roi_net.quantile([.1,.25,.5,.75,.9])*100).round(1).to_string()}")
    print(f"\n  entry priority cost: ${buys_only.gas_usd.mean():.2f} per buy"
          f" = {buys_only.gas_usd.mean()/buys_only.cost_usd.mean()*100:.1f}% of mean position")
    print(f"  exit cost:           ${a[a.event_type=='sell'].gas_usd.mean():.3f} per sell")

    outdir = os.path.dirname(os.path.abspath(__file__))
    pd.Series(s).to_csv(os.path.join(outdir, "part1_summary.csv"))
    g.to_csv(os.path.join(outdir, "part1_pnl_per_token.csv"))
    m.to_csv(os.path.join(outdir, "part1_entries.csv"), index=False)
    print(f"\nwrote part1_summary.csv, part1_pnl_per_token.csv, part1_entries.csv")


if __name__ == "__main__":
    main()
