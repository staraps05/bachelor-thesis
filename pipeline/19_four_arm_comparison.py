# %% 19_four_arm_comparison.py
# Purpose: Merge the three sentiment feature sets onto the model frame and run
#          the four-arm comparison (A0 structured, A1 +LM, A2 +FinBERT,
#          A3 +FinGPT) with the fit procedure of pipeline/18; reports pooled and
#          per-year OOS R^2, Diebold-Mariano tests and sentiment-block importance.
# Inputs:  Data/processed/model_frame_structured.parquet
#          Data/processed/features_lm.parquet
#          Data/processed/features_finbert.parquet
#          Data/processed/features_fingpt.parquet
# Outputs: Data/processed/predictions_A0_structured.parquet
#          Data/processed/predictions_A1_lm.parquet
#          Data/processed/predictions_A2_finbert.parquet
#          Data/processed/predictions_A3_fingpt.parquet
# Notes:
# - The fit configuration (FIXED_PARAMS, DEPTH_GRID, NUM_ROUND_CAP, EARLY_STOP,
#   FOLDS, SEED), the input lists and the frame path/SHA256 are loaded from
#   pipeline/18_baseline_model.py via importlib; only the input list differs
#   between arms. This script is in turn loaded by pipeline/20_seed_stability.py
#   and verification/25_event_spot_check.py.
# - The feature files are keyed on target_week, which is the panel week_anchor
#   grid: sentiment at feature week t (headlines before anchor t) predicts
#   ret_excess_fwd, the week-(t+1) excess return. Features are merged left m:1
#   and not rank-normalised; had_news is identical across the three sets and
#   enters the sentiment arms only.
# - A0 must reproduce the pipeline/18 baseline (pooled R^2 and prediction-file
#   SHA256 hard-coded below) or the script stops.
# - DM test: GKX cross-sectional-average form on weekly mean squared errors,
#   Newey-West lag 4, two-sided normal p-value. Primary: DM > 0 means the
#   sentiment arm beats A0; pairwise: DM > 0 means the second arm is better.
# - All four input SHA256s and shapes are checked; the script stops on mismatch
#   or if a prediction output exists write-protected.

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC = REPO_ROOT / "Data" / "processed"

# fit configuration from pipeline/18 (which loads the input lists from
# pipeline/17); nothing is re-declared here
_spec = importlib.util.spec_from_file_location(
    "base18", REPO_ROOT / "pipeline" / "18_baseline_model.py")
_b18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b18)
FIXED_PARAMS, DEPTH_GRID = _b18.FIXED_PARAMS, _b18.DEPTH_GRID
NUM_ROUND_CAP, EARLY_STOP = _b18.NUM_ROUND_CAP, _b18.EARLY_STOP
FOLDS, SEED = _b18.FOLDS, _b18.SEED
KEYS, TARGET_FWD, INPUTS_30 = _b18.KEYS, _b18.TARGET_FWD, _b18.INPUTS_30
FRAME_FILE, FRAME_SHA = _b18.FRAME_FILE, _b18.FRAME_SHA
N_ROWS, N_COLS = _b18.N_ROWS, _b18.N_COLS
r2_oos = _b18.r2_oos

# SHA256 values as recorded in docs/decisions_log.md
SENT_FILES = {
    "lm": (PROC / "features_lm.parquet",
           "b07a8f1e236d34d6d68dabd1038d1dfc632f6a2b220f4ccf9e3af88e0a7d50a3"),
    "finbert": (PROC / "features_finbert.parquet",
                "24eb3541e0d5388ff2aafee48bff81ed76d3e54a9737399499dd71e035e86289"),
    "fingpt": (PROC / "features_fingpt.parquet",
               "8cc910caf14633244601a56c19fffaaad47707705e942e6c15f4ce6beb034710"),
}

# values from the pipeline/18 run log (docs/decisions_log.md)
BASE_R2 = -0.015412
BASE_PRED_SHA = "87418301a27263ce8f147d961556bf296bedc601269c74c939b4d753a768ec26"

DM_LAG = 4          # Newey-West lag for the adapted Diebold-Mariano
ALPHA = 0.05

FLAGS: list[str] = []


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    # ---- load the frame and merge the sentiment features ----
    print("=== STAGE 0: load + merge the three frozen sentiment sets ===")
    # Verification (in place): frame SHA256 and shape
    sha = sha256_of(FRAME_FILE)
    if sha != FRAME_SHA:
        stop(f"model_frame SHA mismatch: {sha}")
    df = pd.read_parquet(FRAME_FILE)
    if df.shape != (N_ROWS, N_COLS):
        stop(f"model_frame shape {df.shape}")
    print(f"  [PASS] model_frame_structured: SHA {FRAME_SHA[:8]}..., "
          f"{N_ROWS:,} x {N_COLS}, target {TARGET_FWD}")

    hn_cols = []
    for arm, (path, want_sha) in SENT_FILES.items():
        got = sha256_of(path)
        if got != want_sha:
            stop(f"{path.name} SHA mismatch:\n  found  {got}\n  frozen {want_sha}")
        f = pd.read_parquet(path)
        want_cols = ["PERMNO", "target_week", f"{arm}_mean", f"{arm}_std",
                     f"{arm}_range", f"{arm}_kurt", "had_news"]
        if list(f.columns) != want_cols or f.shape != (157_897, 7):
            stop(f"{path.name} schema unexpected: {f.shape}, {list(f.columns)}")
        # target_week is the panel week_anchor grid
        f = f.rename(columns={"target_week": "week_anchor",
                              "had_news": f"had_news_{arm}"})
        if f["week_anchor"].dtype != df["week_anchor"].dtype:
            print(f"  [INFO] {arm}: week key dtype {f['week_anchor'].dtype} -> "
                  f"coerced to frame dtype (S1 dtype-trap guard)")
            f["week_anchor"] = pd.to_datetime(f["week_anchor"]).dt.date
        n0 = len(df)
        df = df.merge(f, on=["PERMNO", "week_anchor"], how="left",
                      validate="m:1")
        if len(df) != n0:
            stop(f"merge of {arm} changed row count {n0:,} -> {len(df):,}")
        hn_cols.append(f"had_news_{arm}")
        print(f"  [PASS] {path.name}: SHA == frozen ({want_sha[:8]}...), "
              f"LEFT-merged m:1, rows {len(df):,}")

    sent_cols = [f"{a}_{s}" for a in SENT_FILES for s in
                 ("mean", "std", "range", "kurt")] + hn_cols
    # Verification (in place): no NaN in merged sentiment columns
    n_nan = int(df[sent_cols].isna().sum().sum())
    if n_nan:
        bad = df.loc[df[sent_cols].isna().any(axis=1), KEYS].head(20)
        stop(f"{n_nan} NaN introduced in sentiment columns - coverage "
             f"mismatch. First offending keys:\n{bad}")
    print(f"  [PASS] 0 NaN across all {len(sent_cols)} merged sentiment "
          f"columns (full-grid coverage; no-news rows are 0 by construction)")

    # Verification (in place): had_news identical across arms
    if not (df[hn_cols[0]].equals(df[hn_cols[1]])
            and df[hn_cols[0]].equals(df[hn_cols[2]])):
        stop("had_news differs between arms - must be byte-identical")
    df["had_news"] = df[hn_cols[0]]
    df = df.drop(columns=hn_cols[1:]).drop(columns=hn_cols[:1])
    print(f"  [PASS] had_news identical across all three arms -> collapsed to "
          f"ONE column (share {df['had_news'].mean():.4f})")

    print("  [PASS] ALIGNMENT: sentiment_t (strict pre-anchor headline lag, "
          "S1) joined on feature week t; target ret_excess_fwd = week t+1 "
          "(D-EVAL-0, 0 look-ahead violations in 17) -> sentiment_t predicts "
          "return_{t+1}, look-ahead-free")

    # ---- the four input sets ----
    print("\n=== STAGE 1: the four experiment sets ===")
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
    # Verification (in place): arm input counts
    for arm, cols in ARMS.items():
        want = 30 if arm == "A0_structured" else 35
        if len(cols) != want or len(set(cols)) != want:
            stop(f"{arm}: {len(cols)} inputs, expected {want}")
        print(f"  {arm:<14} {len(cols)} inputs"
              + ("" if arm == "A0_structured" else
                 "  (= 30 structured + 4 sentiment + had_news)"))
    print("  [INFO] sentiment features "
          "NOT rank-normalised - XGBoost is scale-invariant; *_mean bounded "
          "[-1,1]; no-news rows form a legitimate spike at 0 (had_news=0) "
          "that rank-normalisation would smear. had_news enters ONLY the "
          "sentiment arms, never A0.")

    # ---- fit per arm and fold ----
    print("\n=== STAGE 2: identical fit procedure, all four arms ===")
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {}
    # Verification (in place): fold feature weeks disjoint
    for fid, tr_y, va_y, te_y in FOLDS:
        m_tr, m_va, m_te = yr.isin(tr_y), yr.isin(va_y), yr.isin(te_y)
        w = [set(df.loc[m, "week_anchor"]) for m in (m_tr, m_va, m_te)]
        if w[0] & w[1] or w[0] & w[2] or w[1] & w[2]:
            stop(f"fold {fid}: feature-week sets not disjoint")
        split_masks[fid] = (m_tr, m_va, m_te)

    preds_by_arm, imp_by_arm = {}, {}
    for arm, inputs in ARMS.items():
        all_preds, fold_imp = [], {}
        for fid, tr_y, va_y, te_y in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            d_tr = xgb.DMatrix(df.loc[m_tr, inputs],
                               label=df.loc[m_tr, TARGET_FWD])
            d_va = xgb.DMatrix(df.loc[m_va, inputs],
                               label=df.loc[m_va, TARGET_FWD])
            d_te = xgb.DMatrix(df.loc[m_te, inputs])
            best = None
            for depth in DEPTH_GRID:   # ascending -> ties resolve to lower depth
                params = dict(FIXED_PARAMS, max_depth=depth)
                bst = xgb.train(params, d_tr, num_boost_round=NUM_ROUND_CAP,
                                evals=[(d_va, "val")],
                                early_stopping_rounds=EARLY_STOP,
                                verbose_eval=False)
                val_mse = float(bst.best_score) ** 2
                n_est = int(bst.best_iteration) + 1
                if best is None or val_mse < best["val_mse"]:
                    best = {"depth": depth, "bst": bst, "val_mse": val_mse,
                            "n_est": n_est}
            print(f"  {arm:<14} fold {fid}: SELECTED max_depth={best['depth']}"
                  f", n_estimators={best['n_est']:>4}, "
                  f"val MSE {best['val_mse']:.8f}")
            yhat = best["bst"].predict(d_te,
                                       iteration_range=(0, best["n_est"]))
            p = df.loc[m_te, KEYS + [TARGET_FWD]].copy()
            p["yhat"] = yhat
            p["fold"] = fid
            p["test_year"] = sorted(te_y)[0]
            all_preds.append(p)
            sc = best["bst"].get_score(importance_type="total_gain")
            tot = sum(sc.values())
            fold_imp[fid] = {f: sc.get(f, 0.0) / tot for f in inputs}
        preds = pd.concat(all_preds, ignore_index=True)
        out = PROC / f"predictions_{arm}.parquet"
    # Verification (in place): refuse to overwrite a read-only output
        if out.exists() and not out.stat().st_mode & 0o200:
            stop(f"{out.name} exists read-only - new version required")
        preds.to_parquet(out, index=False)
        preds_by_arm[arm] = preds
        imp_by_arm[arm] = fold_imp
        print(f"  {arm:<14} saved {out.name} ({len(preds):,} rows, "
              f"SHA {sha256_of(out)[:8]}...)")

    # Verification (in place): A0 reproduces the pipeline/18 baseline
    # A0 must reproduce the pipeline/18 baseline exactly
    a0_sha = sha256_of(PROC / "predictions_A0_structured.parquet")
    a0_r2 = r2_oos(preds_by_arm["A0_structured"][TARGET_FWD],
                   preds_by_arm["A0_structured"]["yhat"])
    if a0_sha != BASE_PRED_SHA or abs(a0_r2 - BASE_R2) > 1e-6:
        stop(f"A0 does NOT reproduce the audited baseline - pipeline changed."
             f"\n  A0 pred SHA {a0_sha}\n  audited     {BASE_PRED_SHA}"
             f"\n  A0 R2 {a0_r2:.10f} vs audited {BASE_R2}")
    print(f"  [PASS] CONSISTENCY: A0 pooled R2 = {a0_r2:+.6f} == audited "
          f"-0.015412 AND predictions SHA == 87418301... (bit-identical "
          f"reproduction of pipeline/18)")

    # ---- comparison table ----
    print("\n=== STAGE 3: comparison table (pooled OOS R2, GKX eq. 19, "
          "common denominator) ===")
    r2 = {}
    for arm, p in preds_by_arm.items():
        r2[arm] = {"pool": r2_oos(p[TARGET_FWD], p["yhat"])}
        for y in (2021, 2022, 2023):
            s = p[p["test_year"] == y]
            r2[arm][y] = r2_oos(s[TARGET_FWD], s["yhat"])
    print(f"  {'arm':<14} {'pooled R2':>10} {'dR2 vs A0':>10} "
          f"{'2021':>9} {'2022':>9} {'2023':>9}")
    for arm in ARMS:
        d = (r2[arm]["pool"] - r2["A0_structured"]["pool"]) * 100
        print(f"  {arm:<14} {r2[arm]['pool'] * 100:>+9.4f}% "
              f"{d:>+9.4f}pp {r2[arm][2021] * 100:>+8.4f}% "
              f"{r2[arm][2022] * 100:>+8.4f}% {r2[arm][2023] * 100:>+8.4f}%")

    # ---- Diebold-Mariano tests ----
    print("\n=== STAGE 4: adapted Diebold-Mariano (GKX cross-sectional-"
          "average form, NW lag 4) ===")

    def weekly_e2(p):
        """Cross-sectional mean squared error per test week."""
        q = p.copy()
        q["e2"] = (q[TARGET_FWD] - q["yhat"]) ** 2
        return q.groupby("week_anchor")["e2"].mean().sort_index()

    wk_e2 = {arm: weekly_e2(p) for arm, p in preds_by_arm.items()}
    base_idx = wk_e2["A0_structured"].index
    # Verification (in place): test-week sets identical across arms
    for arm in ARMS:
        if not wk_e2[arm].index.equals(base_idx):
            stop(f"{arm}: test-week set differs from A0")

    def dm_test(d):
        """Mean of d over its Newey-West (lag DM_LAG) SE; two-sided normal p."""
        n = len(d)
        u = d - d.mean()
        s = float((u * u).sum()) / n
        for lag in range(1, DM_LAG + 1):
            w = 1.0 - lag / (DM_LAG + 1.0)
            s += 2.0 * w * float((u[lag:] * u[:-lag]).sum()) / n
        se = math.sqrt(s / n)
        stat = float(d.mean()) / se
        p = math.erfc(abs(stat) / math.sqrt(2.0))
        return stat, p, n

    dm_res = {}
    print("  PRIMARY (H1): d_t = mean_i e2(A0) - mean_i e2(arm); "
          "DM > 0 = arm beats A0")
    for arm in ("A1_lm", "A2_finbert", "A3_fingpt"):
        d = (wk_e2["A0_structured"] - wk_e2[arm]).to_numpy()
        stat, p, n = dm_test(d)
        dm_res[arm] = (stat, p)
        sig = "SIGNIFICANT" if (p < ALPHA and stat > 0) else "not significant"
        print(f"    {arm:<12} vs A0: DM = {stat:+.4f}, two-sided p = {p:.4f} "
              f"(n = {n} weeks) -> {sig} at {ALPHA}")
    print("  SECONDARY (extraction method, pairwise): d_t = e2(first) - "
          "e2(second); DM > 0 = SECOND arm better")
    for a, b in (("A1_lm", "A2_finbert"), ("A1_lm", "A3_fingpt"),
                 ("A2_finbert", "A3_fingpt")):
        d = (wk_e2[a] - wk_e2[b]).to_numpy()
        stat, p, n = dm_test(d)
        print(f"    {a} vs {b}: DM = {stat:+.4f}, p = {p:.4f}")

    # ---- Verification: consistency and reproduction checks ----
    print(f"  [INFO] config imported from pipeline/18: fixed {FIXED_PARAMS}, "
          f"grid max_depth {DEPTH_GRID}, cap {NUM_ROUND_CAP}, "
          f"early stop {EARLY_STOP}, seed {SEED}; xgboost {xgb.__version__}")

    # ---- per-year deltas, sentiment-block importance, verdict ----
    print("\n=== STAGE 5: read-support diagnostics + HARD STOP ===")
    print("  per-year dR2 vs A0 (pp) + pooled excl-2021 (svar/COVID caveat):")
    for arm in ("A1_lm", "A2_finbert", "A3_fingpt"):
        dy = {y: (r2[arm][y] - r2["A0_structured"][y]) * 100
              for y in (2021, 2022, 2023)}
        p_arm = preds_by_arm[arm]
        p_a0 = preds_by_arm["A0_structured"]
        m_arm = p_arm["test_year"] != 2021
        m_a0 = p_a0["test_year"] != 2021
        d_ex = (r2_oos(p_arm.loc[m_arm, TARGET_FWD], p_arm.loc[m_arm, "yhat"])
                - r2_oos(p_a0.loc[m_a0, TARGET_FWD],
                         p_a0.loc[m_a0, "yhat"])) * 100
        d_pool = (r2[arm]["pool"] - r2["A0_structured"]["pool"]) * 100
        print(f"    {arm:<12} 2021 {dy[2021]:+.4f} | 2022 {dy[2022]:+.4f} | "
              f"2023 {dy[2023]:+.4f} | pooled {d_pool:+.4f} | "
              f"excl-2021 {d_ex:+.4f}")
        if d_pool > 0 and d_ex <= 0:
            FLAGS.append(f"{arm}: pooled advantage ({d_pool:+.4f}pp) is "
                         f"2021-concentrated (excl-2021 {d_ex:+.4f}pp <= 0) "
                         f"- svar/COVID caveat")

    print("  sentiment-block importance (avg total-gain share across folds):")
    for arm in ("A1_lm", "A2_finbert", "A3_fingpt"):
        block = ARMS[arm][30:]          # 4 sentiment features + had_news
        avg = {f: np.mean([imp_by_arm[arm][fid][f] for fid in (1, 2, 3)])
               for f in block}
        tot = sum(avg.values())
        parts = ", ".join(f"{f} {v:.4f}" for f, v in
                          sorted(avg.items(), key=lambda kv: -kv[1]))
        print(f"    {arm:<12} block total {tot:.4f}  [{parts}]")
        if avg["had_news"] > sum(v for f, v in avg.items()
                                 if f != "had_news"):
            print(f"      [INFO] {arm} leans more on had_news ('news exists') "
                  f"than on the sentiment values combined")

    return 0


if __name__ == "__main__":
    sys.exit(main())
