# %% 18_baseline_model.py
# Purpose: Fit the structured-only XGBoost baseline (arm A0) with expanding-window folds and save test predictions.
# Inputs:  Data/processed/model_frame_structured.parquet
# Outputs: Data/processed/baseline_predictions.parquet
# Notes:
# - Loads 3_models/17 via importlib (column lists); loaded by 3_models/19 for fit config.
# - FOLDS train|val|test: 18-19|20|21, 18-20|21|22, 18-21|22|23 (years); seed 42.
# - Stops on frame SHA256/shape mismatch or write-protected output; sanity gates only FLAG

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAME_FILE = REPO_ROOT / "Data" / "processed" / "model_frame_structured.parquet"
FRAME_SHA = "09b3f7e9929dfb099233706eaa3d792ffa2fa28ab7456c304d2438429609b883"
PRED_FILE = REPO_ROOT / "Data" / "processed" / "baseline_predictions.parquet"
N_ROWS, N_COLS = 157_277, 35

# column lists come from pipeline/17 (loaded via importlib, not re-declared)
_spec = importlib.util.spec_from_file_location(
    "pre17", REPO_ROOT / "pipeline" / "3_models" / "17_pre_model_integrity.py")
_pre17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pre17)
KEYS, TARGET_FWD = _pre17.KEYS, _pre17.TARGET_FWD
CHARS_22, MACRO_8, INPUTS_30 = _pre17.CHARS_22, _pre17.MACRO_8, _pre17.INPUTS_30

SEED = 42
FIXED_PARAMS = {
    "objective": "reg:squarederror",
    "eta": 0.05,                 # learning_rate
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "lambda": 1.0,               # reg_lambda
    "tree_method": "hist",
    "seed": SEED,
    "eval_metric": "rmse",       # early stopping on val RMSE; selection
                                 # compares RMSE^2 = MSE
}
DEPTH_GRID = [2, 3, 4]           # tuned per fold
NUM_ROUND_CAP = 2000
EARLY_STOP = 50

FOLDS = [  # (fold id, train years, val year(s), test year(s))
    (1, {2018, 2019}, {2020}, {2021}),
    (2, {2018, 2019, 2020}, {2021}, {2022}),
    (3, {2018, 2019, 2020, 2021}, {2022}, {2023}),
]

# sanity gates: a tripped gate is reported as FLAG in the final verdict
R2_POOLED_ALARM = 0.05   # |pooled OOS R^2| above this -> leakage alarm
R2_YEAR_ALARM = 0.10     # any single-year |R^2| above this -> leakage alarm
YHAT_ABS_BOUND = 0.50    # a 50% weekly excess-return prediction = blow-up
ACCOUNTING_RATIOS = {    # flagged if the top-importance feature is one of these
    "be_me", "ni_me", "sale_me", "chcsho_12m", "ope_be", "niq_be", "gp_at",
    "at_gr1", "sale_gr1", "oaccruals_at", "noa_at", "at_be", "niq_su",
}
CLUSTERS = {             # collinear clusters (pipeline/17 correlation pairs)
    "size/liquidity": ["market_equity", "dolvol_126d", "ami_126d"],
    "value": ["dp", "bm"],
    "risk(vol)": ["rvol_21d", "ivol_ff3_21d"],
    "momentum": ["ret_12_1", "ret_6_1", "ret_1_0"],
}

FLAGS: list[str] = []


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def r2_oos(y: pd.Series, yhat: pd.Series) -> float:
    """OOS R^2 against a zero forecast (GKX eq. 19); denominator not demeaned."""
    return 1.0 - float(((y - yhat) ** 2).sum()) / float((y ** 2).sum())


def main() -> int:
    # ---- load the frame ----
    print("=== STAGE 0: load the derived predictive frame ===")
    # Verification (in place): frame SHA256, shape, no NaN
    sha = hashlib.sha256(FRAME_FILE.read_bytes()).hexdigest()
    if sha != FRAME_SHA:
        stop(f"model_frame_structured.parquet SHA mismatch:\n  found  {sha}"
             f"\n  expected {FRAME_SHA}\n  (regenerate via pipeline/17)")
    df = pd.read_parquet(FRAME_FILE)
    if df.shape != (N_ROWS, N_COLS):
        stop(f"shape {df.shape}, expected ({N_ROWS:,}, {N_COLS})")
    if df[INPUTS_30 + [TARGET_FWD]].isna().any().any():
        stop("NaN in inputs/target - frame should be clean")
    print("  [INFO] OpenMP runtime: sklearn-vendored libomp.dylib copied into "
          "xgboost/lib (@loader_path rpath, ad-hoc re-signed) - no Homebrew "
          "on this machine; recorded for reproducibility")

    # ---- expanding-window splits ----
    print("\n=== STAGE 1: expanding-window splits by FEATURE-week year ===")
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {}
    # Verification (in place): fold years and feature weeks disjoint
    for fid, tr_y, va_y, te_y in FOLDS:
        if tr_y & va_y or tr_y & te_y or va_y & te_y:
            stop(f"fold {fid}: year sets overlap")
        m_tr, m_va, m_te = yr.isin(tr_y), yr.isin(va_y), yr.isin(te_y)
        w_tr = set(df.loc[m_tr, "week_anchor"])
        w_va = set(df.loc[m_va, "week_anchor"])
        w_te = set(df.loc[m_te, "week_anchor"])
        if w_tr & w_va or w_tr & w_te or w_va & w_te:
            stop(f"fold {fid}: feature-week sets not disjoint")
        split_masks[fid] = (m_tr, m_va, m_te)
        print(f"  fold {fid}: train {sorted(tr_y)} = {int(m_tr.sum()):,} rows | "
              f"val {sorted(va_y)} = {int(m_va.sum()):,} rows | "
              f"test {sorted(te_y)} = {int(m_te.sum()):,} rows   "
              f"[feature-week sets disjoint: PASS]")

    # ---- fit per fold ----
    print("\n=== STAGE 2: fit per fold (grid max_depth, early stop on val) ===")
    all_preds = []
    fold_importance = {}
    for fid, tr_y, va_y, te_y in FOLDS:
        m_tr, m_va, m_te = split_masks[fid]
        d_tr = xgb.DMatrix(df.loc[m_tr, INPUTS_30],
                           label=df.loc[m_tr, TARGET_FWD])
        d_va = xgb.DMatrix(df.loc[m_va, INPUTS_30],
                           label=df.loc[m_va, TARGET_FWD])
        d_te = xgb.DMatrix(df.loc[m_te, INPUTS_30])
        best = None
        for depth in DEPTH_GRID:  # ascending -> ties resolve to lower depth
            params = dict(FIXED_PARAMS, max_depth=depth)
            bst = xgb.train(params, d_tr, num_boost_round=NUM_ROUND_CAP,
                            evals=[(d_va, "val")],
                            early_stopping_rounds=EARLY_STOP,
                            verbose_eval=False)
            val_mse = float(bst.best_score) ** 2
            n_est = int(bst.best_iteration) + 1
            print(f"  fold {fid} depth {depth}: best n_estimators {n_est:>4} | "
                  f"val MSE {val_mse:.8f} (val RMSE {float(bst.best_score):.6f})")
            if best is None or val_mse < best["val_mse"]:
                best = {"depth": depth, "bst": bst, "val_mse": val_mse,
                        "n_est": n_est}
        print(f"  fold {fid} SELECTED: max_depth={best['depth']}, "
              f"n_estimators={best['n_est']}, val MSE {best['val_mse']:.8f}")
        yhat = best["bst"].predict(d_te, iteration_range=(0, best["n_est"]))
        p = df.loc[m_te, KEYS + [TARGET_FWD]].copy()
        p["yhat"] = yhat
        p["fold"] = fid
        p["test_year"] = sorted(te_y)[0]
        all_preds.append(p)
        sc = best["bst"].get_score(importance_type="total_gain")
        tot = sum(sc.values())
        fold_importance[fid] = {f: sc.get(f, 0.0) / tot for f in INPUTS_30}

    preds = pd.concat(all_preds, ignore_index=True)
    # Verification (in place): refuse to overwrite a read-only output
    if PRED_FILE.exists() and not PRED_FILE.stat().st_mode & 0o200:
        stop(f"{PRED_FILE.name} exists read-only - new version required")
    preds.to_parquet(PRED_FILE, index=False)
    pred_sha = hashlib.sha256(PRED_FILE.read_bytes()).hexdigest()
    print(f"  saved {PRED_FILE.name}: {len(preds):,} rows "
          f"(keys + y + yhat + fold + test_year)")
    print(f"  SHA256 {pred_sha}")

    # ---- pooled and per-year R^2 ----
    print("\n=== STAGE 3: pooled OOS R^2 vs zero (GKX eq. 19, un-demeaned) ===")
    r2_pool = r2_oos(preds[TARGET_FWD], preds["yhat"])
    print(f"  pooled OOS 2021-23: R2_oos = {r2_pool:+.6f} "
          f"({r2_pool * 100:+.4f}%)   n = {len(preds):,}")
    r2_years = {}
    for y in (2021, 2022, 2023):
        sub = preds[preds["test_year"] == y]
        r2_years[y] = r2_oos(sub[TARGET_FWD], sub["yhat"])
        print(f"  per-year {y}:      R2_oos = {r2_years[y]:+.6f} "
              f"({r2_years[y] * 100:+.4f}%)   n = {len(sub):,}")

    # ---- feature importance and prediction sanity ----
    print("\n=== STAGE 4: feature importance (total-gain share) + sanity ===")
    imp = pd.DataFrame(fold_importance)          # rows: features, cols: folds
    imp.columns = [f"fold{c}" for c in imp.columns]
    imp["avg"] = imp.mean(axis=1)
    imp = imp.sort_values("avg", ascending=False)
    print(f"  {'feature':<14} {'avg':>7}  {'fold1':>7} {'fold2':>7} {'fold3':>7}")
    for f, r in imp.iterrows():
        print(f"  {f:<14} {r['avg']:>7.4f}  {r['fold1']:>7.4f} "
              f"{r['fold2']:>7.4f} {r['fold3']:>7.4f}")
    print("  cluster-level view (avg share summed; collinear groups from "
          "pipeline/17 Stage 3):")
    clustered = set()
    for name, feats in CLUSTERS.items():
        clustered.update(feats)
        print(f"    {name:<15} = {imp.loc[feats, 'avg'].sum():.4f}  "
              f"({' + '.join(feats)})")
    singles = imp.loc[[f for f in imp.index if f not in clustered], "avg"]
    print(f"    top unclustered: "
          + ", ".join(f"{f} {v:.4f}" for f, v in singles.head(5).items()))

    # ---- Verification: consistency and reproduction checks ----
    print(f"  [PASS] SHA256 == {FRAME_SHA[:8]}... | shape {N_ROWS:,} x {N_COLS} "
          f"| 0 NaN in inputs+target")
    print(f"  [PASS] input lists imported from pipeline/17 (single source): "
          f"{len(CHARS_22)} chars + {len(MACRO_8)} macro = {len(INPUTS_30)} "
          f"inputs; target = {TARGET_FWD}")
    print(f"  [INFO] xgboost {xgb.__version__}, pandas {pd.__version__}, "
          f"seed {SEED}, tree_method hist")
    print("  [INFO] splits are by FEATURE week_anchor year; a December feature "
          "week's TARGET return may lie one week into the next calendar year "
          "by construction (t->t+1). Documented, harmless: no (feature, "
          "target) pair appears in two splits, and no future feature is used.")
    y, yh = preds[TARGET_FWD], preds["yhat"]
    n_bad = int((~np.isfinite(yh)).sum())
    if n_bad:
        stop(f"{n_bad} NaN/inf predictions")
    wk_std = preds.groupby("week_anchor")["yhat"].std()
    print(f"  yhat: mean {yh.mean():+.6f} | std {yh.std():.6f} | "
          f"min {yh.min():+.6f} | max {yh.max():+.6f} | NaN/inf 0 [PASS]")
    print(f"  cross-sectional spread: mean weekly std(yhat) = {wk_std.mean():.6f} "
          f"(realised y: std {y.std():.4f})")
    # ---- sanity gates ----
    print("\n=== STAGE 5: SANITY GATE + HARD STOP (audit verdict) ===")
    if abs(r2_pool) > R2_POOLED_ALARM:
        FLAGS.append(f"LEAKAGE ALARM: |pooled R2_oos| = {abs(r2_pool):.4f} "
                     f"> {R2_POOLED_ALARM}")
    for yy, r2 in r2_years.items():
        if abs(r2) > R2_YEAR_ALARM:
            FLAGS.append(f"LEAKAGE ALARM: |R2_oos {yy}| = {abs(r2):.4f} "
                         f"> {R2_YEAR_ALARM}")
    top_feat = imp.index[0]
    if top_feat in ACCOUNTING_RATIOS:
        FLAGS.append(f"importance review: accounting ratio '{top_feat}' "
                     f"dominates (avg share {imp.loc[top_feat, 'avg']:.4f})")
    if float(yh.abs().max()) > YHAT_ABS_BOUND:
        FLAGS.append(f"prediction blow-up: max |yhat| = {yh.abs().max():.4f} "
                     f"> {YHAT_ABS_BOUND}")
    print(f"  gates: |pooled R2| <= {R2_POOLED_ALARM}: "
          f"{abs(r2_pool):.4f} {'OK' if abs(r2_pool) <= R2_POOLED_ALARM else 'TRIPPED'}"
          f" | per-year |R2| <= {R2_YEAR_ALARM}: max "
          f"{max(abs(v) for v in r2_years.values()):.4f} "
          f"{'OK' if all(abs(v) <= R2_YEAR_ALARM for v in r2_years.values()) else 'TRIPPED'}"
          f" | top importance: {top_feat} "
          f"({'accounting - TRIPPED' if top_feat in ACCOUNTING_RATIOS else 'not an accounting ratio - OK'})"
          f" | max |yhat| {yh.abs().max():.4f} "
          f"{'OK' if yh.abs().max() <= YHAT_ABS_BOUND else 'TRIPPED'}")

    top5 = preds.loc[yh.abs().sort_values(ascending=False).head(5).index]
    print("  5 largest |yhat|:")
    for _, r in top5.iterrows():
        print(f"    PERMNO {int(r['PERMNO'])} week {r['week_anchor']} "
              f"yhat {r['yhat']:+.4f} (y {r[TARGET_FWD]:+.4f}, "
              f"fold {int(r['fold'])})")

    # Verification (in place): FLAG verdict
    if FLAGS:
        print(f"\n  Verdict: FLAG ({len(FLAGS)}) - STOP, await review:")
        for f in FLAGS:
            print(f"    - {f}")
    else:
        print("\n  Verdict: PASS - audit clean, baseline plausible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
