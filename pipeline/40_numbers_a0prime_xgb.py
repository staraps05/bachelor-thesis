# %% 40_numbers_a0prime_xgb.py
# Purpose: Original A0' check (A0 plus had_news): how many of the 30 fits split on had_news; reproduced by pipeline/47.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: Data/processed/numbers_check/ a0prime_xgb_cells.csv, a0prime_xgb_summary.csv
# Notes:
# - Loads pipeline/27 via importlib for single_lag_frame and the fit configuration.
# - Split counts from the booster truncated at the selected number of trees.
# - A0 is refitted alongside to test whether A0' predictions are bit-identical.

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


c27 = _load("cov27", "27_covered_subsample.py")
single_lag_frame = c27.single_lag_frame
FIXED_PARAMS, AMENDMENT = c27.FIXED_PARAMS, c27.AMENDMENT
DEPTH_GRID, NUM_ROUND_CAP = c27.DEPTH_GRID, c27.NUM_ROUND_CAP
EARLY_STOP, FOLDS, SEEDS = c27.EARLY_STOP, c27.FOLDS, c27.SEEDS
TARGET_FWD, INPUTS_30 = c27.TARGET_FWD, c27.INPUTS_30


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== 40: A0' XGBoost had_news split check xgb {xgb.__version__} ===")
    df = single_lag_frame()
    ARMS = {"A0_structured": list(INPUTS_30),
            "A0p_coverage": list(INPUTS_30) + ["had_news"]}
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {fid: (yr.isin(t), yr.isin(v), yr.isin(te)) for fid, t, v, te in FOLDS}
    dmats = {}
    for arm, inputs in ARMS.items():
        dmats[arm] = {}
        for fid, _, _, _ in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            dmats[arm][fid] = (
                xgb.DMatrix(df.loc[m_tr, inputs], label=df.loc[m_tr, TARGET_FWD]),
                xgb.DMatrix(df.loc[m_va, inputs], label=df.loc[m_va, TARGET_FWD]),
                xgb.DMatrix(df.loc[m_te, inputs]))
    te_rows = {fid: int(split_masks[fid][2].sum()) for fid, _, _, _ in FOLDS}

    rows = []
    fold_yhat = {}
    total_splits_on_hadnews = 0
    for seed in SEEDS:
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
                fold_yhat[(seed, arm, fid)] = yhat
                if arm == "A0p_coverage":
                    # "weight" = split count per feature, in the booster truncated at n_est
                    sc = bst[0:n_est].get_score(importance_type="weight")
                    hn = int(sc.get("had_news", 0))
                    total_splits_on_hadnews += hn
                    rows.append({"seed": seed, "fold": fid, "n_est": n_est,
                                 "had_news_split_count": hn,
                                 "had_news_in_model": hn > 0})
    # bit-identity A0' vs A0 per seed x fold
    for r in rows:
        s, f = r["seed"], r["fold"]
        r["identical_to_A0"] = bool(np.array_equal(
            fold_yhat[(s, "A0p_coverage", f)], fold_yhat[(s, "A0_structured", f)]))
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT / "a0prime_xgb_cells.csv", index=False)
    summary = {
        "n_cells": len(cells),
        "cells_with_had_news_split": int(cells["had_news_in_model"].sum()),
        "total_had_news_split_nodes": total_splits_on_hadnews,
        "cells_identical_to_A0": int(cells["identical_to_A0"].sum()),
        "any_tree_splits_on_had_news": bool(cells["had_news_in_model"].any()),
    }
    pd.DataFrame([summary]).to_csv(OUT / "a0prime_xgb_summary.csv", index=False)
    print(f"  cells {summary['n_cells']}; split on had_news in "
          f"{summary['cells_with_had_news_split']} cells "
          f"({summary['total_had_news_split_nodes']} split nodes total)")
    print(f"  A0' bit-identical to A0: {summary['cells_identical_to_A0']}/{summary['n_cells']}")
    print(f"  ANY tree splits on had_news: {summary['any_tree_splits_on_had_news']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
