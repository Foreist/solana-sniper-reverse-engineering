"""Build the full feature table for Parts 2 and 3.

Memory is the binding constraint: the negative-class activity file is 25 GB / 177M rows across
1,006,625 deployers, and loading it whole as pandas objects would exceed available RAM. Two
measures keep it tractable without changing the analysis:

  1. **Negatives are sampled before activity is read.** Training only needs a known ratio of
     negatives (metrics are corrected back to true prevalence), while the June test period
     keeps *every* negative so the reported PR-AUC reflects the real 1:189 imbalance.
  2. **Activity is read filtered to the deployers actually needed**, using a pyarrow row filter
     rather than a post-hoc pandas mask.

Everything before 2026-03-12 08:53 UTC is dropped: the bot's first recorded buy. Labelling
earlier deployments "not bought" would encode a calendar fact rather than a selection rule.
"""
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import features_fast as FF

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "extracted")

BOT_ACTIVE_FROM = 1773305601   # 2026-03-12 08:53 UTC, the bot's first buy
TEST_START = 1780272000        # 2026-06-01 00:00 UTC
NEG_RATIO_TRAIN = 20           # negatives per positive in the training period
ACT_COLS = ["wallet", "timestamp", "event_type", "token_address", "cost_usd"]


def load_deploys(path, label):
    d = pd.read_parquet(path, columns=["token_address", "blockTime", "blockSlot", "tx_signer"])
    d = d.rename(columns={"blockTime": "deploy_time"})
    d["label"] = label
    return d[d.deploy_time >= BOT_ACTIVE_FROM].reset_index(drop=True)


def sample_negatives(neg, n_pos_train, ratio=NEG_RATIO_TRAIN, seed=0):
    """Subsample training-period negatives; keep the test period complete."""
    train = neg[neg.deploy_time < TEST_START]
    test = neg[neg.deploy_time >= TEST_START]
    take = min(len(train), n_pos_train * ratio)
    train_s = train.sample(take, random_state=seed)
    scale = len(train) / max(len(train_s), 1)
    return pd.concat([train_s, test], ignore_index=True), scale, len(train), len(test)


def read_activity_for(path, wallets, columns=ACT_COLS, batch_rows=4_000_000):
    """Stream the activity file, keeping only rows for the wallets we need."""
    pf = pq.ParquetFile(path)
    keep = []
    total = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        t = batch.to_pandas()
        total += len(t)
        t = t[t.wallet.isin(wallets)]
        if len(t):
            keep.append(t)
    print(f"    scanned {total:,} activity rows, kept {sum(len(k) for k in keep):,}")
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=columns)


def main():
    print("=== deployments ===")
    pos = load_deploys(os.path.join(DATA, "bought_deploy_txs_index.parquet"), 1)
    neg = load_deploys(os.path.join(DATA, "not_bought_deploy_txs_index.parquet"), 0)
    n_pos_train = int((pos.deploy_time < TEST_START).sum())
    print(f"  positives {len(pos):,} (train {n_pos_train:,}, test {len(pos)-n_pos_train:,})")

    neg_s, neg_scale, neg_train_all, neg_test_all = sample_negatives(neg, n_pos_train)
    print(f"  negatives in window {len(neg):,} -> using {len(neg_s):,}")
    print(f"    train {neg_train_all:,} subsampled to {len(neg_s)-neg_test_all:,} "
          f"(scale {neg_scale:.1f}x), test kept whole at {neg_test_all:,}")
    print(f"  test-period true prevalence 1:{neg_test_all/max(len(pos)-n_pos_train,1):.0f}")

    print("\n=== positive-class features ===")
    act_pos = FF.pd.read_parquet(os.path.join(DATA, "bought_deployers_activity.parquet"),
                                 columns=ACT_COLS)
    act_pos["cost_usd"] = pd.to_numeric(act_pos.cost_usd, errors="coerce")
    f_pos = FF.build(pos, act_pos, label=1, reference_check=1000)
    print(f"  {f_pos.shape}")
    del act_pos

    print("\n=== negative-class features ===")
    wallets = set(neg_s.tx_signer.unique())
    print(f"  deployers needed: {len(wallets):,}")
    act_neg = read_activity_for(os.path.join(DATA, "not_bought_deployers_activity.parquet"),
                                wallets)
    act_neg["cost_usd"] = pd.to_numeric(act_neg.cost_usd, errors="coerce")
    f_neg = FF.build(neg_s, act_neg, label=0, reference_check=1000)
    print(f"  {f_neg.shape}")
    del act_neg

    out = pd.concat([f_pos, f_neg], ignore_index=True)
    out["neg_sample_scale"] = np.where(
        (out.label == 0) & (out.deploy_time < TEST_START), neg_scale, 1.0)
    path = os.path.join(HERE, "features_all.parquet")
    out.to_parquet(path, index=False)
    print(f"\nwrote {path}  {out.shape}")
    print(out.groupby(["label", out.deploy_time >= TEST_START]).size().rename("rows").to_string())


if __name__ == "__main__":
    main()
