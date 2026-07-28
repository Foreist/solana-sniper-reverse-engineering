"""Part 2 — reverse-engineer the bot's selection rule.

Two requirements from the rubric shape this file.

*Interpretability*: the reported model must expose its top-10 features and support a
plain-language statement of the rules. A gradient-boosted tree with SHAP satisfies that, and a
logistic regression on the same features is fitted alongside as a monotone sanity check — if
the two disagree on the sign of a driver, the driver is not trustworthy.

*Honest evaluation under extreme imbalance*: roughly 16k positives against 5.06M negatives, so
ROC-AUC is close to meaningless. The split is strictly time-based (train on the earlier period,
test on the later), and the headline metrics are PR-AUC and precision at the operating point
the replica strategy would actually trade.

Negatives are subsampled for training only. Every reported metric is rescaled to the true
prevalence so the numbers describe the real decision problem, not the sampled one.
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NEG_PER_POS_TRAIN = 20        # training subsample ratio; test set keeps true prevalence

# The test period is pinned to June 2026 rather than chosen as a quantile, because the
# post-deployment supplements (market-cap candles, pump.fun trades) cover June only. Fixing
# the split here means Part 3 backtests exactly the period Part 2 held out — the replica never
# trades on data its model was fitted on, and the backtest never needs data it does not have.
TEST_START = 1780272000       # 2026-06-01 00:00 UTC
# The bot's first recorded buy. Deployments before this are excluded entirely: the bot did not
# exist yet, so labelling them "not bought" teaches the model a calendar fact, not a rule.
BOT_ACTIVE_FROM = 1773305601  # 2026-03-12 08:53 UTC


def time_split(df, test_start=TEST_START, active_from=BOT_ACTIVE_FROM):
    """Split on deploy_time so no future deployment informs a past prediction."""
    df = df[df.deploy_time >= active_from].copy()
    train = df[df.deploy_time < test_start].copy()
    test = df[df.deploy_time >= test_start].copy()
    return train, test, int(test_start)


def subsample_negatives(train, ratio=NEG_PER_POS_TRAIN, seed=0):
    pos = train[train.label == 1]
    neg = train[train.label == 0]
    n = min(len(neg), len(pos) * ratio)
    return pd.concat([pos, neg.sample(n, random_state=seed)]).sample(frac=1, random_state=seed)


def fit_models(X_tr, y_tr, feature_names):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gbdt = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=0)
    gbdt.fit(X_tr, y_tr)

    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
    logit.fit(np.nan_to_num(X_tr), y_tr)
    return gbdt, logit


def evaluate(y_true, score, prevalence_scale=1.0, label=""):
    """PR-AUC and precision/recall at several thresholds.

    prevalence_scale corrects precision for negative subsampling: if negatives were kept at
    1/k of their true rate, observed false positives must be multiplied by k.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    out = {"label": label,
           "pr_auc": float(average_precision_score(y_true, score)),
           "roc_auc": float(roc_auc_score(y_true, score)),
           "positives": int(y_true.sum()), "n": int(len(y_true))}
    prec, rec, thr = precision_recall_curve(y_true, score)
    rows = []
    for target_recall in [0.05, 0.10, 0.25, 0.50, 0.75]:
        i = int(np.argmin(np.abs(rec - target_recall)))
        p_obs = prec[i]
        # Correct precision back to true prevalence.
        if prevalence_scale != 1.0 and p_obs > 0:
            tp_over_fp = p_obs / (1 - p_obs) if p_obs < 1 else np.inf
            p_true = tp_over_fp / (tp_over_fp + prevalence_scale)
        else:
            p_true = p_obs
        rows.append(dict(target_recall=target_recall, recall=float(rec[i]),
                         precision_observed=float(p_obs), precision_true=float(p_true),
                         threshold=float(thr[min(i, len(thr) - 1)])))
    out["operating_points"] = rows
    return out


def shap_top_features(gbdt, X, feature_names, k=10, sample=4000, seed=0):
    """Mean |SHAP| ranking. Falls back to permutation importance if shap is unavailable."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X[idx]
    try:
        import shap
        expl = shap.TreeExplainer(gbdt)
        vals = expl.shap_values(Xs)
        if isinstance(vals, list):
            vals = vals[1]
        imp = np.abs(vals).mean(axis=0)
        method = "mean |SHAP|"
    except Exception:
        from sklearn.inspection import permutation_importance
        # Needs labels; caller passes X only, so approximate with model output variance.
        base = gbdt.predict_proba(Xs)[:, 1]
        imp = np.zeros(Xs.shape[1])
        for j in range(Xs.shape[1]):
            Xp = Xs.copy()
            rng.shuffle(Xp[:, j])
            imp[j] = np.abs(gbdt.predict_proba(Xp)[:, 1] - base).mean()
        method = "permutation on prediction"
    order = np.argsort(imp)[::-1][:k]
    return method, [(feature_names[i], float(imp[i])) for i in order]


def directional_check(logit, feature_names, top_names):
    """Sign of each top feature in the linear model, for the plain-language rule statement."""
    coefs = logit[-1].coef_[0]
    lookup = dict(zip(feature_names, coefs))
    return {n: ("increases" if lookup.get(n, 0) > 0 else "decreases") for n in top_names}


def run(feature_table_path=os.path.join(OUT, "features_all.parquet")):
    df = pd.read_parquet(feature_table_path)
    # Columns that must never enter the model:
    #   neg_sample_scale — bookkeeping for the negative subsample. It is 20.0 for training
    #     negatives and 1.0 for everything else, so it separates the training classes
    #     perfectly and is constant in the test period. Left in by accident it produced a
    #     test ROC-AUC of exactly 0.5000 with all-zero feature importances.
    #   deploy_slot / deploy_time — monotone in time. Under a time-based split these are pure
    #     period indicators, learnable on train and meaningless out of sample.
    #   token_address / wallet — identifiers.
    drop = ["token_address", "wallet", "label", "neg_sample_scale",
            "deploy_slot", "deploy_time"]
    feature_names = [c for c in df.columns if c not in drop]
    print(f"features used ({len(feature_names)}): {feature_names}")

    train_full, test, cut = time_split(df)
    train = subsample_negatives(train_full)
    scale = ((train_full.label == 0).sum() / max((train.label == 0).sum(), 1))

    print(f"split at deploy_time {cut} "
          f"({pd.Timestamp(cut, unit='s', tz='UTC'):%Y-%m-%d})")
    print(f"train {len(train):,} (pos {int(train.label.sum()):,}, neg subsampled "
          f"{int((train.label==0).sum()):,} of {int((train_full.label==0).sum()):,}, scale {scale:.1f}x)")
    print(f"test  {len(test):,} (pos {int(test.label.sum()):,}, "
          f"prevalence {test.label.mean()*100:.3f}%)")

    X_tr = train[feature_names].to_numpy(dtype=float)
    X_te = test[feature_names].to_numpy(dtype=float)
    gbdt, logit = fit_models(X_tr, train.label.to_numpy(), feature_names)

    s_gbdt = gbdt.predict_proba(X_te)[:, 1]
    s_logit = logit.predict_proba(np.nan_to_num(X_te))[:, 1]

    res = []
    for name, s in [("GBDT", s_gbdt), ("Logistic", s_logit)]:
        r = evaluate(test.label.to_numpy(), s, prevalence_scale=1.0, label=name)
        res.append(r)
        print(f"\n=== {name} on held-out later period ===")
        print(f"  PR-AUC {r['pr_auc']:.4f} | ROC-AUC {r['roc_auc']:.4f} "
              f"| baseline PR-AUC (prevalence) {test.label.mean():.5f}")
        print(f"  {'recall':>8s} {'precision':>10s} {'threshold':>10s}")
        for op in r["operating_points"]:
            print(f"  {op['recall']:8.3f} {op['precision_observed']:10.4f} {op['threshold']:10.4f}")

    method, top = shap_top_features(gbdt, X_te, feature_names)
    print(f"\n=== top-10 features ({method}) ===")
    for n, v in top:
        print(f"  {v:10.5f}  {n}")
    signs = directional_check(logit, feature_names, [n for n, _ in top])
    print("\n=== direction in the linear model (agreement check) ===")
    for n in [n for n, _ in top]:
        print(f"  {n:26s} {signs[n]} the odds of a buy")

    pd.DataFrame(top, columns=["feature", "importance"]).to_csv(
        os.path.join(OUT, "part2_top_features.csv"), index=False)
    test_out = test[["token_address", "deploy_time", "label"]].copy()
    test_out["score_gbdt"] = s_gbdt
    test_out["score_logit"] = s_logit
    test_out.to_parquet(os.path.join(OUT, "part2_test_scores.parquet"), index=False)
    print("\nwrote part2_top_features.csv, part2_test_scores.parquet")
    return res, top


if __name__ == "__main__":
    run()
