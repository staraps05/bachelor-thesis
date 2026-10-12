# %% 35_numbers_importance_table42.py
# Purpose: Reproduce the A0 (structured) and A2 (+FinBERT) single-lag fits and
#          save the full-precision total-gain importance shares behind Table 4.2:
#          per-feature shares, block sums and the sentiment-block members.
# Inputs:  Data/processed/model_frame_structured.parquet (via single_lag_frame)
#          Data/processed/features_lm.parquet
#          Data/processed/features_finbert.parquet
#          Data/processed/features_fingpt.parquet
# Outputs: Data/processed/numbers_check/table42_importance_perfeature.csv
#          Data/processed/numbers_check/table42_block_sums.csv
#          Data/processed/numbers_check/table42_sentiment_members.csv
# Notes:
# - Loads pipeline/27_covered_subsample.py via importlib for single_lag_frame
#   and the fit configuration, pipeline/30_per_year_primary.py for the
#   reference per-seed R2 of the gate, and pipeline/17_pre_model_integrity.py
#   for the CHARS_22 / MACRO_8 block membership.
# - Gate: the pooled R2 of every A0 and A2 seed (20 cells) must match the
#   reference from the pipeline/26 run log within GATE_TOL, else the run stops.
# - Importance = a feature's share of total gain in the booster truncated at
#   the selected iteration (bst[0:best_iteration+1]), as in Table 4.2; the
#   untruncated A0 share is stored alongside for comparison.
# - Shares are averaged over the 3 folds within seed, then mean and sd (ddof=1)
#   across the 10 seeds; block sums and their sd are formed per seed the same
#   way.
# - Per-feature rows are ordered by A0 share; the A2 column reports the same
#   30 structured inputs, the 5 sentiment inputs (4 FinBERT + had_news) go to
#   the members CSV.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "Data" / "processed" / "numbers_check"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_c27 = _load("cov27", "27_covered_subsample.py")
_c30 = _load("c30", "30_per_year_primary.py")
_c17 = _load("pre17", "17_pre_model_integrity.py")
single_lag_frame = _c27.single_lag_frame
FIXED_PARAMS, AMENDMENT = _c27.FIXED_PARAMS, _c27.AMENDMENT
DEPTH_GRID, NUM_ROUND_CAP = _c27.DEPTH_GRID, _c27.NUM_ROUND_CAP
EARLY_STOP, FOLDS, SEEDS = _c27.EARLY_STOP, _c27.FOLDS, _c27.SEEDS
TARGET_FWD, INPUTS_30 = _c27.TARGET_FWD, _c27.INPUTS_30
C26_SEED_R2, GATE_TOL = _c30.C26_SEED_R2, _c30.GATE_TOL
CHARS_22, MACRO_8 = _c17.CHARS_22, _c17.MACRO_8


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def share_vec(bst, inputs):
    sc = bst.get_score(importance_type="total_gain")
    tot = sum(sc.values())
    # Verification (in place): non-zero total gain (inside share_vec)
    if tot <= 0:
        stop("zero total gain")
    return np.array([sc.get(f, 0.0) / tot for f in inputs])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== 35: Table 4.2 importance (A0 + A2, truncated) xgb {xgb.__version__} ===")
    df = single_lag_frame()
    ARMS = {
        "A0_structured": list(INPUTS_30),
        "A2_finbert": list(INPUTS_30) + ["finbert_mean", "finbert_std",
                                         "finbert_range", "finbert_kurt",
                                         "had_news"],
    }
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {fid: (yr.isin(t), yr.isin(v), yr.isin(te))
                   for fid, t, v, te in FOLDS}
    dmats = {}
    for arm, inputs in ARMS.items():
        dmats[arm] = {}
        for fid, _, _, _ in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            dmats[arm][fid] = (
                xgb.DMatrix(df.loc[m_tr, inputs], label=df.loc[m_tr, TARGET_FWD]),
                xgb.DMatrix(df.loc[m_va, inputs], label=df.loc[m_va, TARGET_FWD]),
                xgb.DMatrix(df.loc[m_te, inputs]))
    te_meta = {fid: df.loc[split_masks[fid][2],
                           ["week_anchor", TARGET_FWD]].reset_index(drop=True)
               for fid, _, _, _ in FOLDS}

    a0_trunc, a0_full, a2_trunc = {}, {}, {}
    max_dev = 0.0
    for seed in SEEDS:
        for arm, inputs in ARMS.items():
            sse = ssy = 0.0
            for fid, _, _, _ in FOLDS:
                d_tr, d_va, d_te = dmats[arm][fid]
                best = None
                for depth in DEPTH_GRID:
                    params = dict(FIXED_PARAMS, **AMENDMENT, seed=seed,
                                  max_depth=depth)
                    bst = xgb.train(params, d_tr, num_boost_round=NUM_ROUND_CAP,
                                    evals=[(d_va, "val")],
                                    early_stopping_rounds=EARLY_STOP,
                                    verbose_eval=False)
                    val_mse = float(bst.best_score) ** 2
                    if best is None or val_mse < best["val_mse"]:
                        best = {"bst": bst, "val_mse": val_mse,
                                "n_est": int(bst.best_iteration) + 1}
                bst, n_est = best["bst"], best["n_est"]
                yhat = bst.predict(d_te, iteration_range=(0, n_est))
                y = te_meta[fid][TARGET_FWD].to_numpy()
                sse += float(((y - yhat) ** 2).sum())
                ssy += float((y ** 2).sum())
                vt = share_vec(bst[0:n_est], inputs)
                if arm == "A0_structured":
                    a0_trunc[(seed, fid)] = vt
                    a0_full[(seed, fid)] = share_vec(bst, inputs)
                else:
                    a2_trunc[(seed, fid)] = vt
            pool = (1.0 - sse / ssy) * 100
    # Verification (in place): per-seed-arm reproduction gate (inside loop)
            dev = abs(pool - C26_SEED_R2[seed][arm])
            max_dev = max(max_dev, dev)
            if dev > GATE_TOL:
                stop(f"GATE FAIL seed {seed} {arm}: dev {dev:.2e}")

    def seed_mean_sd(cells, idx=None):
        # cells: (seed, fid) -> share vector; fold-mean per seed, then mean/sd over seeds
        per_seed = np.array([np.mean([cells[(s, f)] for f, _, _, _ in FOLDS], axis=0)
                             for s in SEEDS])
        return per_seed.mean(axis=0), per_seed.std(axis=0, ddof=1)

    a0_m, a0_s = seed_mean_sd(a0_trunc)
    a0f_m, _ = seed_mean_sd(a0_full)
    a2_m, a2_s = seed_mean_sd(a2_trunc)
    A2_INPUTS = ARMS["A2_finbert"]

    # index shares by feature name
    a2_by_feat = {f: a2_m[i] for i, f in enumerate(A2_INPUTS)}
    a0_by_feat = {f: a0_m[i] for i, f in enumerate(INPUTS_30)}
    a0sd_by_feat = {f: a0_s[i] for i, f in enumerate(INPUTS_30)}
    a0full_by_feat = {f: a0f_m[i] for i, f in enumerate(INPUTS_30)}

    order = sorted(INPUTS_30, key=lambda f: -a0_by_feat[f])
    blk = {f: ("Macro" if f in MACRO_8 else "Characteristic") for f in INPUTS_30}
    rows = []
    for f in order:
        rows.append({"feature": f, "block": blk[f],
                     "a0_share": a0_by_feat[f], "a0_sd": a0sd_by_feat[f],
                     "a0_untrunc_share": a0full_by_feat[f],
                     "a2_share": a2_by_feat[f]})
    pd.DataFrame(rows).to_csv(OUT / "table42_importance_perfeature.csv", index=False)

    # block sums
    def blocksum(by_feat, feats):
        return float(sum(by_feat[f] for f in feats))
    sent_feats = A2_INPUTS[30:]  # 4 finbert + had_news
    a2_sent = float(sum(a2_by_feat[f] for f in sent_feats))
    # per-seed sd for block sums
    def block_sd(cells, inputs, feats):
        per_seed = []
        for s in SEEDS:
            fm = np.mean([cells[(s, f)] for f, _, _, _ in FOLDS], axis=0)
            fmap = {ff: fm[i] for i, ff in enumerate(inputs)}
            per_seed.append(sum(fmap[ff] for ff in feats))
        return float(np.std(per_seed, ddof=1))
    brows = [
        {"block": "macro_8", "a0_share": blocksum(a0_by_feat, MACRO_8),
         "a0_sd": block_sd(a0_trunc, INPUTS_30, MACRO_8),
         "a2_share": blocksum(a2_by_feat, MACRO_8),
         "a2_sd": block_sd(a2_trunc, A2_INPUTS, MACRO_8)},
        {"block": "characteristics_22", "a0_share": blocksum(a0_by_feat, CHARS_22),
         "a0_sd": block_sd(a0_trunc, INPUTS_30, CHARS_22),
         "a2_share": blocksum(a2_by_feat, CHARS_22),
         "a2_sd": block_sd(a2_trunc, A2_INPUTS, CHARS_22)},
        {"block": "sentiment_5", "a0_share": np.nan, "a0_sd": np.nan,
         "a2_share": a2_sent,
         "a2_sd": block_sd(a2_trunc, A2_INPUTS, sent_feats)},
    ]
    pd.DataFrame(brows).to_csv(OUT / "table42_block_sums.csv", index=False)

    # sentiment-block members
    srows = [{"feature": f, "a2_share": a2_by_feat[f]} for f in sent_feats]
    pd.DataFrame(srows).to_csv(OUT / "table42_sentiment_members.csv", index=False)

    # ---- Verification: consistency and reproduction checks ----
    print(f"  [PASS] reproduction gate (A0+A2, 20 pooled cells): max |dev| "
          f"{max_dev:.2e} pp")
    print("  top-10 (A0 order): " + ", ".join(
        f"{f} {a0_by_feat[f]:.4f}/{a2_by_feat[f]:.4f}" for f in order[:10]))
    print(f"  blocks A0: macro {brows[0]['a0_share']:.4f} char {brows[1]['a0_share']:.4f}")
    print(f"  blocks A2: macro {brows[0]['a2_share']:.4f} char {brows[1]['a2_share']:.4f} sent {a2_sent:.4f}")
    print(f"  dp untrunc {a0full_by_feat['dp']:.4f} trunc {a0_by_feat['dp']:.4f}")
    print(f"  max_dev_pp={max_dev:.3e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
