# %% 47_numbers_a0prime_r2delta.py
# Purpose: A0' (A0 plus had_news) under the single-lag pairing: fits that split on had_news and the A0' minus A0 R2 delta.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: Data/processed/numbers_check/ a0prime_r2delta_perseed.csv, a0prime_r2delta_splits.csv, a0prime_r2delta_summary.csv
# Notes:
# - Loads 3_models/27 (single_lag_frame, fit config) and 30 (gate table) via importlib.
# - Split counts when the truncated booster (first best_iteration+1 trees) uses had_news.
# - Gate: each seed's A0 pooled R2 must match the 3_models/26 value within GATE_TOL.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[2]
NC = REPO_ROOT / "Data" / "processed" / "numbers_check"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c27 = _load("cov27", "3_models/27_covered_subsample.py")
c30 = _load("c30", "3_models/30_per_year_primary.py")
single_lag_frame = c27.single_lag_frame
FIXED_PARAMS, AMENDMENT = c27.FIXED_PARAMS, c27.AMENDMENT
DEPTH_GRID, NUM_ROUND_CAP = c27.DEPTH_GRID, c27.NUM_ROUND_CAP
EARLY_STOP, FOLDS, SEEDS = c27.EARLY_STOP, c27.FOLDS, c27.SEEDS
TARGET_FWD, INPUTS_30 = c27.TARGET_FWD, c27.INPUTS_30
C26_SEED_R2, GATE_TOL = c30.C26_SEED_R2, c30.GATE_TOL
YEARS = c30.YEARS


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def main() -> int:
    NC.mkdir(parents=True, exist_ok=True)
    print(f"=== 47: A0' R2 deltas + had_news splits (single lag) xgb {xgb.__version__} ===")
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
    te_meta = {fid: df.loc[split_masks[fid][2], ["week_anchor", TARGET_FWD]].reset_index(drop=True)
               for fid, _, _, _ in FOLDS}
    te_year = {fid: sorted(te_y)[0] for fid, _, _, te_y in FOLDS}

    r2 = {}      # (seed, arm) -> {"pool":..,"2021":..}
    split_rows = []
    max_dev = 0.0
    for seed in SEEDS:
        for arm, inputs in ARMS.items():
            parts = []
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
                p = te_meta[fid].copy()
                p["yhat"] = yhat
                p["test_year"] = te_year[fid]
                parts.append(p)
                if arm == "A0p_coverage":
                    hn = int(bst[0:n_est].get_score(importance_type="weight").get("had_news", 0))
                    if hn > 0:
                        split_rows.append({"seed": seed, "fold": fid, "n_est": n_est,
                                           "had_news_splits": hn})
            preds = pd.concat(parts, ignore_index=True)
            e2 = (preds[TARGET_FWD] - preds["yhat"]) ** 2
            pool = 1.0 - float(e2.sum()) / float((preds[TARGET_FWD] ** 2).sum())
            m21 = preds["test_year"] == 2021
            r21 = 1.0 - float(e2[m21].sum()) / float((preds.loc[m21, TARGET_FWD] ** 2).sum())
            r2[(seed, arm)] = {"pool": pool, "2021": r21}
    # Verification (in place): per-seed A0 reproduction gate (inside loop)
        dev = abs(r2[(seed, "A0_structured")]["pool"] * 100 - C26_SEED_R2[seed]["A0_structured"])
        max_dev = max(max_dev, dev)
        if dev > GATE_TOL:
            stop(f"GATE FAIL seed {seed}: A0 pooled dev {dev:.2e}")

    # per-seed deltas
    perseed = []
    for seed in SEEDS:
        for scope in ("pool", "2021"):
            d = (r2[(seed, "A0p_coverage")][scope] - r2[(seed, "A0_structured")][scope]) * 100
            perseed.append({"seed": seed, "scope": scope, "delta_pp": d,
                            "a0_r2_pct": r2[(seed, "A0_structured")][scope] * 100,
                            "a0p_r2_pct": r2[(seed, "A0p_coverage")][scope] * 100})
    ps = pd.DataFrame(perseed)
    ps.to_csv(NC / "a0prime_r2delta_perseed.csv", index=False)
    pd.DataFrame(split_rows).to_csv(NC / "a0prime_r2delta_splits.csv", index=False)

    summary = []
    for scope in ("pool", "2021"):
        d = ps[ps.scope == scope]["delta_pp"].to_numpy()
        summary.append({"scope": scope, "mean_delta_pp": d.mean(), "sd_delta_pp": d.std(ddof=1),
                        "min_delta_pp": d.min(), "max_delta_pp": d.max()})
    pd.DataFrame(summary).to_csv(NC / "a0prime_r2delta_summary.csv", index=False)

    print("  PAIRING: single lag")
    print(f"  fits splitting on had_news ({len(split_rows)}): "
          + ", ".join(f"(s{r['seed']},f{r['fold']},{r['had_news_splits']})" for r in split_rows))
    for r in summary:
        print(f"  A0'-A0 {r['scope']:<5}: mean {r['mean_delta_pp']:+.5f}pp  sd {r['sd_delta_pp']:.5f}  "
              f"[{r['min_delta_pp']:+.4f}, {r['max_delta_pp']:+.4f}]")
    # ---- Verification: consistency and reproduction checks ----
    print(f"  [PASS] A0 reproduction gate vs pipeline/26 log (max |dev| {max_dev:.2e} pp)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
