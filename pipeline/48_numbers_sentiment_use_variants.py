# %% 48_numbers_sentiment_use_variants.py
# Purpose: Compute the sentiment gain share (item 2 of the numbers check) under
#          two sentiment-use variants of the single-lag four-arm run, the
#          covered subsample and the winsorised target, plus the count of
#          fold-2/3 cells whose test predictions are identical to A0.
# Inputs:  Data/processed/model_frame_structured.parquet (via single_lag_frame)
#          Data/processed/features_lm.parquet
#          Data/processed/features_finbert.parquet
#          Data/processed/features_fingpt.parquet
# Outputs: Data/processed/numbers_check/item2_sentiment_share.csv
#          Data/processed/numbers_check/item2_identical_counts.csv
# Notes:
# - Loads pipeline/27_covered_subsample.py via importlib for single_lag_frame,
#   the fit configuration (params, depth grid, folds, seeds) and SENT_ARMS.
# - Variants: "covered60" keeps firms with had_news >= 60% of their 2021-2023
#   weeks, all years, as in pipeline/27; "winsorised" clips the train and
#   validation target at the training 0.5/99.5 percentiles per fold, as in
#   pipeline/37. Test rows are never clipped.
# - Arms: A0 = 30 structured inputs; A1/A2/A3 add the LM / FinBERT / FinGPT
#   mean, std, range, kurt and had_news. Fits: 10 seeds x 3 folds, depth
#   chosen on validation MSE with early stopping.
# - Sentiment share = total_gain of the five sentiment columns over all
#   columns in the truncated booster (first best_iteration+1 trees); reported
#   as the mean over seeds x folds, the SD over seeds of the per-seed fold
#   mean, and per fold.
# - Identical cells: of the 60 seed x sentiment-arm x fold-{2,3} cells, those
#   whose test predictions equal A0's exactly.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
NC = REPO_ROOT / "Data" / "processed" / "numbers_check"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c27 = _load("cov27", "27_covered_subsample.py")
single_lag_frame = c27.single_lag_frame
FIXED_PARAMS, AMENDMENT = c27.FIXED_PARAMS, c27.AMENDMENT
DEPTH_GRID, NUM_ROUND_CAP = c27.DEPTH_GRID, c27.NUM_ROUND_CAP
EARLY_STOP, FOLDS, SEEDS = c27.EARLY_STOP, c27.FOLDS, c27.SEEDS
TARGET_FWD, INPUTS_30 = c27.TARGET_FWD, c27.INPUTS_30
SENT_ARMS = c27.SENT_ARMS


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def arms():
    return {
        "A0_structured": list(INPUTS_30),
        "A1_lm": list(INPUTS_30) + ["lm_mean", "lm_std", "lm_range", "lm_kurt", "had_news"],
        "A2_finbert": list(INPUTS_30) + ["finbert_mean", "finbert_std", "finbert_range",
                                         "finbert_kurt", "had_news"],
        "A3_fingpt": list(INPUTS_30) + ["fingpt_mean", "fingpt_std", "fingpt_range",
                                        "fingpt_kurt", "had_news"],
    }


def sent_share(bst, inputs):
    """Total-gain share of the sentiment block (columns after the 30 structured inputs)."""
    sc = bst.get_score(importance_type="total_gain")
    tot = sum(sc.values())
    if tot <= 0:
        return 0.0
    return sum(sc.get(f, 0.0) for f in inputs[30:]) / tot


def run_variant(df, label, winsor):
    """Fit all four arms on one variant frame; returns (share, ident) dicts keyed
    by (seed, arm, fold)."""
    ARMS = arms()
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {fid: (yr.isin(t), yr.isin(v), yr.isin(te)) for fid, t, v, te in FOLDS}
    clips = {}
    if winsor:
        for fid, _, _, _ in FOLDS:
            ytr = df.loc[split_masks[fid][0], TARGET_FWD].to_numpy(float)
            lo, hi = np.percentile(ytr, [0.5, 99.5])
            clips[fid] = (float(lo), float(hi))
    dmats = {}
    for arm, inputs in ARMS.items():
        dmats[arm] = {}
        for fid, _, _, _ in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            if winsor:
                lo, hi = clips[fid]
                ytr = df.loc[m_tr, TARGET_FWD].clip(lo, hi)
                yva = df.loc[m_va, TARGET_FWD].clip(lo, hi)
            else:
                ytr = df.loc[m_tr, TARGET_FWD]
                yva = df.loc[m_va, TARGET_FWD]
            dmats[arm][fid] = (xgb.DMatrix(df.loc[m_tr, inputs], label=ytr),
                               xgb.DMatrix(df.loc[m_va, inputs], label=yva),
                               xgb.DMatrix(df.loc[m_te, inputs]))
    share = {}      # (seed, arm, fid) -> sentiment-block share
    ident = {}      # (seed, arm, fid) -> bool bit-identical to A0
    for seed in SEEDS:
        fold_yhat = {}
        for arm, inputs in ARMS.items():
            for fid, _, _, _ in FOLDS:
                d_tr, d_va, d_te = dmats[arm][fid]
                best = None
                for depth in DEPTH_GRID:
                    params = dict(FIXED_PARAMS, **AMENDMENT, seed=seed, max_depth=depth)
                    bst = xgb.train(params, d_tr, num_boost_round=NUM_ROUND_CAP,
                                    evals=[(d_va, "val")], early_stopping_rounds=EARLY_STOP,
                                    verbose_eval=False)
                    val_mse = float(bst.best_score) ** 2
                    if best is None or val_mse < best["val_mse"]:
                        best = {"bst": bst, "val_mse": val_mse,
                                "n_est": int(bst.best_iteration) + 1}
                bst, n_est = best["bst"], best["n_est"]
                yhat = bst.predict(d_te, iteration_range=(0, n_est))
                fold_yhat[(arm, fid)] = yhat
                if arm in SENT_ARMS:
                    share[(seed, arm, fid)] = sent_share(bst[0:n_est], inputs)
        for fid, _, _, _ in FOLDS:
            pass
        for arm in SENT_ARMS:
            for fid, _, _, _ in FOLDS:
                ident[(seed, arm, fid)] = bool(np.array_equal(
                    fold_yhat[(arm, fid)], fold_yhat[("A0_structured", fid)]))
    return share, ident


def main() -> int:
    NC.mkdir(parents=True, exist_ok=True)
    print(f"=== 48: sentiment use under covered-60 + winsorised (single lag) xgb {xgb.__version__} ===")
    df = single_lag_frame()
    yr = pd.to_datetime(df["week_anchor"]).dt.year

    variants = {}
    # covered subsample: firms with had_news >= 60% of their test-year weeks
    te = df[yr.isin({2021, 2022, 2023})]
    share60 = te.groupby("PERMNO")["had_news"].mean()
    qual = set(share60[share60 >= 0.60].index)
    df_cov = df[df["PERMNO"].isin(qual)].copy()
    variants["covered60"] = run_variant(df_cov, "covered60", winsor=False)
    # winsorised target on the full frame
    variants["winsorised"] = run_variant(df, "winsorised", winsor=True)

    share_rows, ident_rows = [], []
    for vname, (share, ident) in variants.items():
        for arm in SENT_ARMS:
            vals = [share[(s, arm, f)] for s in SEEDS for f, _, _, _ in FOLDS]
            byfold = {f: np.mean([share[(s, arm, f)] for s in SEEDS]) for f, _, _, _ in FOLDS}
            share_rows.append({"variant": vname, "arm": arm,
                               "mean_share": float(np.mean(vals)),
                               "sd_share_overseeds": float(np.std(
                                   [np.mean([share[(s, arm, f)] for f, _, _, _ in FOLDS])
                                    for s in SEEDS], ddof=1)),
                               "fold1": float(byfold[1]), "fold2": float(byfold[2]),
                               "fold3": float(byfold[3])})
        # cells in folds 2 and 3 whose predictions equal A0 (60 cells per variant)
        n_ident = sum(1 for s in SEEDS for arm in SENT_ARMS for f in (2, 3)
                      if ident[(s, arm, f)])
        ident_rows.append({"variant": vname, "cells_total": 60,
                           "cells_identical_to_A0_folds23": n_ident})
    pd.DataFrame(share_rows).to_csv(NC / "item2_sentiment_share.csv", index=False)
    pd.DataFrame(ident_rows).to_csv(NC / "item2_identical_counts.csv", index=False)

    print("  --- (a) sentiment-block share (mean | f1/f2/f3) ---")
    for r in share_rows:
        print(f"    {r['variant']:<11} {r['arm']:<12} mean {r['mean_share']:.4f} "
              f"(sd {r['sd_share_overseeds']:.4f}) | {r['fold1']:.4f}/{r['fold2']:.4f}/{r['fold3']:.4f}")
    print("  --- (b) folds 2+3 cells bit-identical to A0 (of 60) ---")
    for r in ident_rows:
        print(f"    {r['variant']:<11} {r['cells_identical_to_A0_folds23']}/60")
    # ---- Verification: consistency and reproduction checks ----
    print(f"  covered-60: {len(qual)} firms, {len(df_cov):,} rows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
