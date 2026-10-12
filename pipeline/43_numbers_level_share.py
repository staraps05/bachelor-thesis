# %% 43_numbers_level_share.py
# Purpose: Table C.8: bias-vs-skill decomposition of the pooled R2 deficit per arm, single-lag run at seed 42.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: Data/processed/numbers_check/ tableC8_level_share.csv
# Notes:
# - Loads pipeline/27 via importlib for single_lag_frame, FOLDS and the XGBoost config.
# - Seed 42 only; reproduces verification/28 check 2.
# - Gate: each arm's seed-42 pooled R2 must match the pipeline/26 value within 2e-4 pp.

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
EARLY_STOP, FOLDS = c27.EARLY_STOP, c27.FOLDS
TARGET_FWD, INPUTS_30 = c27.TARGET_FWD, c27.INPUTS_30

SEED = 42
# seed-42 pooled R2 (%) per arm as recorded in pipeline/26
C26_SEED42 = {"A0_structured": -1.3365, "A1_lm": -1.2133,
              "A2_finbert": -1.2786, "A3_fingpt": -1.3710}


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== 43: level share (Table C.8), seed {SEED} xgb {xgb.__version__} ===")
    df = single_lag_frame()
    ARMS = {
        "A0_structured": list(INPUTS_30),
        "A1_lm": list(INPUTS_30) + ["lm_mean", "lm_std", "lm_range", "lm_kurt", "had_news"],
        "A2_finbert": list(INPUTS_30) + ["finbert_mean", "finbert_std", "finbert_range",
                                         "finbert_kurt", "had_news"],
        "A3_fingpt": list(INPUTS_30) + ["fingpt_mean", "fingpt_std", "fingpt_range",
                                        "fingpt_kurt", "had_news"],
    }
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {fid: (yr.isin(t), yr.isin(v), yr.isin(te)) for fid, t, v, te in FOLDS}
    rows = []
    for arm, inputs in ARMS.items():
        parts = []
        for fid, _, _, _ in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            d_tr = xgb.DMatrix(df.loc[m_tr, inputs], label=df.loc[m_tr, TARGET_FWD])
            d_va = xgb.DMatrix(df.loc[m_va, inputs], label=df.loc[m_va, TARGET_FWD])
            d_te = xgb.DMatrix(df.loc[m_te, inputs])
            best = None
            for depth in DEPTH_GRID:
                params = dict(FIXED_PARAMS, **AMENDMENT, seed=SEED, max_depth=depth)
                bst = xgb.train(params, d_tr, num_boost_round=NUM_ROUND_CAP,
                                evals=[(d_va, "val")], early_stopping_rounds=EARLY_STOP,
                                verbose_eval=False)
                val_mse = float(bst.best_score) ** 2
                if best is None or val_mse < best["val_mse"]:
                    best = {"bst": bst, "val_mse": val_mse, "n_est": int(bst.best_iteration) + 1}
            p = df.loc[m_te, [TARGET_FWD]].copy()
            p["yhat"] = best["bst"].predict(d_te, iteration_range=(0, best["n_est"]))
            parts.append(p)
        preds = pd.concat(parts, ignore_index=True)
        y = preds[TARGET_FWD].to_numpy(float)
        yh = preds["yhat"].to_numpy(float)
        pool = (1.0 - float(((y - yh) ** 2).sum()) / float((y ** 2).sum())) * 100
    # Verification (in place): seed-42 pooled R2 reproduces pipeline/26 (inside loop)
        if abs(pool - C26_SEED42[arm]) > 2e-4:
            stop(f"{arm}: seed-42 pooled R2 {pool:+.4f}% != pipeline/26 {C26_SEED42[arm]:+.4f}%")
        bias = float(yh.mean() - y.mean())
        r2_pts = bias ** 2 * len(y) / float((y ** 2).sum()) * 100  # R2 pp lost to the level error
        deficit = -pool                                             # positive (R2 is negative)
        level_share = r2_pts / deficit * 100                       # per cent of the deficit
        rows.append({"arm": arm, "pooled_r2_pct": pool, "mean_y": float(y.mean()),
                     "mean_yhat": float(yh.mean()), "bias": bias,
                     "level_r2_pts": r2_pts, "deficit_pts": deficit,
                     "level_share_pct": level_share})
        print(f"  {arm:<14} pool {pool:+.4f}% | level_pts {r2_pts:.4f} | "
              f"level_share {level_share:.1f}%")
    pd.DataFrame(rows).to_csv(OUT / "tableC8_level_share.csv", index=False)
    print("  wrote tableC8_level_share.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
