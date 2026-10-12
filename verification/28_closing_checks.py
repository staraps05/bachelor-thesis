# %% 28_closing_checks.py
# Purpose: Two closing checks: (1) the 22 panel_final characteristics were rank-normalised cross-sectionally per week; (2) bias-vs-skill decomposition of the single-lag primary XGBoost predictions.
# Inputs:  Data/processed/panel_final.parquet, Data/processed/panel_structured.parquet, Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: prints only
# Notes:
# - Verification script: checks only; writes nothing to Data/
# - Loads pipeline/3_models/27 via importlib (single_lag_frame, fit configuration).
# - Check 2 stops unless regenerated R2 matches the pipeline/3_models/26 log (2e-4 pp); see 43.

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC = REPO_ROOT / "Data" / "processed"

_spec = importlib.util.spec_from_file_location(
    "cov27", REPO_ROOT / "pipeline" / "3_models" / "27_covered_subsample.py")
_c27 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c27)
single_lag_frame = _c27.single_lag_frame
FIXED_PARAMS, AMENDMENT = _c27.FIXED_PARAMS, _c27.AMENDMENT
DEPTH_GRID, NUM_ROUND_CAP = _c27.DEPTH_GRID, _c27.NUM_ROUND_CAP
EARLY_STOP, FOLDS = _c27.EARLY_STOP, _c27.FOLDS
TARGET_FWD, INPUTS_30 = _c27.TARGET_FWD, _c27.INPUTS_30

PANEL_FILE = PROC / "panel_final.parquet"
PANEL_SHA = "eb3ed4cdd461b9772db1a4e5240b0f46a5f21ca3c9948a094b5e654e5fed19f2"
STRUCT_FILE = PROC / "panel_structured.parquet"

CHARS_22 = list(INPUTS_30[:22])        # INPUTS_30 = CHARS_22 + MACRO_8
MACRO_8 = list(INPUTS_30[22:])
N_WEEK_SAMPLE = 20
SEED = 42
# seed-42 pooled R2 (%) from verification/audit_block9_10/c5_code26_output.log
C26_SEED42 = {"A0_structured": -1.3365, "A1_lm": -1.2133,
              "A2_finbert": -1.2786, "A3_fingpt": -1.3710}


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def main() -> int:
    if MACRO_8 != ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]:
        stop("INPUTS_30 ordering unexpected - CHARS_22 slice unsafe")

    # ---- check 1: per-week rank normalisation ----
    print("=== CHECK 1: rank-normalisation applied PER WEEK (leakage "
          "confirmation) ===")
    sha = hashlib.sha256(PANEL_FILE.read_bytes()).hexdigest()
    if sha != PANEL_SHA:
        stop(f"panel_final SHA mismatch: {sha}")
    pf = pd.read_parquet(PANEL_FILE)
    ps = pd.read_parquet(STRUCT_FILE)
    if ps.shape != (157_897, 34) or pf.shape != (157_897, 34):
        stop(f"shapes unexpected: final {pf.shape}, structured {ps.shape}")
    pf = pf.sort_values(["week_anchor", "PERMNO"]).reset_index(drop=True)
    ps = ps.sort_values(["week_anchor", "PERMNO"]).reset_index(drop=True)
    if not (pf["PERMNO"].equals(ps["PERMNO"])
            and pf["week_anchor"].equals(ps["week_anchor"])
            and pf["ret_excess"].equals(ps["ret_excess"])):
        stop("panel_structured / panel_final key or target mismatch - "
             "lineage broken")
    print(f"  [PASS] panel_final SHA == frozen; panel_structured aligned "
          f"1:1 on keys, ret_excess frame-identical (raw source for the "
          f"non-missing mask)")

    anchors = sorted(pf["week_anchor"].unique())
    idx = np.linspace(0, len(anchors) - 1, N_WEEK_SAMPLE).round().astype(int)
    sample_weeks = [anchors[i] for i in sorted(set(idx))]
    print(f"  sampling {len(sample_weeks)} weeks spread over "
          f"{pd.Timestamp(anchors[0]).date()}..{pd.Timestamp(anchors[-1]).date()}")

    fails, n_cells, skipped = [], 0, 0
    worst = {"min_of_max": 2.0, "max_of_min": -2.0, "max_abs_mean": 0.0,
             "min_spear": 2.0}
    for wk in sample_weeks:
        f_w = pf[pf["week_anchor"] == wk]
        s_w = ps[ps["week_anchor"] == wk]
        for c in CHARS_22:
            raw = s_w[c]
            tr = f_w[c]
            m = raw.notna()
            n = int(m.sum())
            if n < 3:
                skipped += 1
                continue
            n_cells += 1
            sub = tr[m.to_numpy()]
            mn, mx = float(sub.min()), float(sub.max())
            mean = float(sub.mean())
            rho = float(raw[m].corr(sub, method="spearman"))
            worst["min_of_max"] = min(worst["min_of_max"], mx)
            worst["max_of_min"] = max(worst["max_of_min"], mn)
            worst["max_abs_mean"] = max(worst["max_abs_mean"], abs(mean))
            worst["min_spear"] = min(worst["min_spear"], rho)
            ok = (mn <= -0.95 and mx >= 0.95 and abs(mean) <= 0.01
                  and rho >= 1.0 - 1e-9)
            if not ok:
                fails.append((c, pd.Timestamp(wk).date(), n, mn, mx, mean,
                              rho))
    print(f"  cells checked: {n_cells} (22 chars x {len(sample_weeks)} weeks"
          f"; {skipped} skipped with n < 3 non-missing)")
    print(f"  worst-case across all cells: per-week max as low as "
          f"{worst['min_of_max']:+.6f}, per-week min as high as "
          f"{worst['max_of_min']:+.6f}, max |within-week mean| "
          f"{worst['max_abs_mean']:.2e}, min within-week Spearman "
          f"{worst['min_spear']:.12f}")
    if fails:
        print(f"  [FAIL] {len(fails)} characteristic x week cells:")
        for c, wk, n, mn, mx, mean, rho in fails[:15]:
            print(f"    {c:<14} {wk} n={n} min {mn:+.4f} max {mx:+.4f} "
                  f"mean {mean:+.4f} spearman {rho:.6f}")
        stop("rank-normalisation check FAILED - diagnostics above, no fix "
             "attempted")
    print("  [PASS] every sampled week spans ~[-1, +1] with within-week "
          "mean ~ 0 and within-week Spearman(raw, transformed) = 1.0 -> the "
          "transform is CROSS-SECTIONAL PER WEEK; a pooled rank could not "
          "produce per-week endpoints at the extremes. No time-series "
          "information used -> no look-ahead from the transform.")

    # ---- check 2: bias vs skill ----
    print("\n=== CHECK 2: bias vs skill decomposition (single-lag PRIMARY, "
          "seed 42) ===")
    print("  [INFO] file: pipeline/26 wrote NO predictions parquet (in-memory "
          "by design). Regenerating seed-42 single-lag predictions with the "
          "identical procedure; verification below requires all four arms' "
          "pooled R2 to match pipeline/26's logged values before use.")
    df = single_lag_frame()
    ARMS = {
        "A0_structured": list(INPUTS_30),
        "A1_lm": list(INPUTS_30) + ["lm_mean", "lm_std", "lm_range",
                                    "lm_kurt", "had_news"],
        "A2_finbert": list(INPUTS_30) + ["finbert_mean", "finbert_std",
                                         "finbert_range", "finbert_kurt",
                                         "had_news"],
        "A3_fingpt": list(INPUTS_30) + ["fingpt_mean", "fingpt_std",
                                        "fingpt_range", "fingpt_kurt",
                                        "had_news"],
    }
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {fid: (yr.isin(t), yr.isin(v), yr.isin(te))
                   for fid, t, v, te in FOLDS}
    preds_by_arm = {}
    for arm, inputs in ARMS.items():
        parts = []
        for fid, _, _, te_y in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            d_tr = xgb.DMatrix(df.loc[m_tr, inputs],
                               label=df.loc[m_tr, TARGET_FWD])
            d_va = xgb.DMatrix(df.loc[m_va, inputs],
                               label=df.loc[m_va, TARGET_FWD])
            d_te = xgb.DMatrix(df.loc[m_te, inputs])
            best = None
            for depth in DEPTH_GRID:
                params = dict(FIXED_PARAMS, **AMENDMENT, seed=SEED,
                              max_depth=depth)
                bst = xgb.train(params, d_tr, num_boost_round=NUM_ROUND_CAP,
                                evals=[(d_va, "val")],
                                early_stopping_rounds=EARLY_STOP,
                                verbose_eval=False)
                val_mse = float(bst.best_score) ** 2
                if best is None or val_mse < best["val_mse"]:
                    best = {"bst": bst, "val_mse": val_mse,
                            "n_est": int(bst.best_iteration) + 1}
            p = df.loc[m_te, [TARGET_FWD]].copy()
            p["yhat"] = best["bst"].predict(
                d_te, iteration_range=(0, best["n_est"]))
            parts.append(p)
        preds = pd.concat(parts, ignore_index=True)
        pool = (1.0 - float(((preds[TARGET_FWD] - preds["yhat"]) ** 2).sum())
                / float((preds[TARGET_FWD] ** 2).sum())) * 100
        if abs(pool - C26_SEED42[arm]) > 2e-4:
            stop(f"{arm}: regenerated pooled R2 {pool:+.4f}% != pipeline/26's "
                 f"logged {C26_SEED42[arm]:+.4f}% - regeneration invalid")
        preds_by_arm[arm] = preds
        print(f"  [PASS] {arm:<14} regenerated pooled R2 {pool:+.4f}% == "
              f"pipeline/26 seed-42 value")

    print(f"\n  decomposition on pooled 2021-23 test rows "
          f"(n = {len(preds_by_arm['A0_structured']):,}):")
    print(f"  {'arm':<14} {'mean(y)':>10} {'mean(yhat)':>11} {'bias':>10} "
          f"{'bias^2':>11} {'MSE':>11} {'bias^2/MSE':>10} {'R2 pts':>8}")
    for arm, p in preds_by_arm.items():
        y, yh = p[TARGET_FWD].to_numpy(float), p["yhat"].to_numpy(float)
        mse = float(((y - yh) ** 2).mean())
        bias = float(yh.mean() - y.mean())
        b2 = bias ** 2
        share = b2 / mse
        r2_pts = b2 * len(y) / float((y ** 2).sum()) * 100
        print(f"  {arm:<14} {y.mean():>+10.6f} {yh.mean():>+11.6f} "
              f"{bias:>+10.6f} {b2:>11.3e} {mse:>11.6f} {share:>9.4%} "
              f"{r2_pts:>7.4f}pp")
    print("  reading: 'R2 pts' = percentage points of the negative pooled "
          "R2 attributable to the LEVEL error alone (bias^2 * n / sum(y^2)); "
          "the remainder is the variance/skill term (MSE - bias^2).")

    print("\n  STOP - nothing promoted, no further diagnostics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
