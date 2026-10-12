# %% 22_dispersion_vs_count.py
# Purpose: Check whether the dispersion features (std, range, kurt) proxy the headline count, and
#          whether the dispersion-return relation survives residualising on the count.
# Inputs:  Data/processed/scoring_input.parquet, Data/processed/scoring_input_coverage.csv, Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: prints only
# Notes:
# - Loads pipeline/20 via importlib (merge_frame SHA-checks inputs, TARGET_FWD, YEARS).
# - Stored count is recomputed from scoring_input.parquet and must match, else it stops.

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC = REPO_ROOT / "Data" / "processed"

_spec = importlib.util.spec_from_file_location(
    "seed20", REPO_ROOT / "pipeline" / "20_seed_stability.py")
_c20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c20)
merge_frame, TARGET_FWD, YEARS = _c20.merge_frame, _c20.TARGET_FWD, _c20.YEARS

# SHA as checked in pipeline/15; test sample size from the pipeline/21 log
SCORING_FILE = PROC / "scoring_input.parquet"
SCORING_SHA = "eb42e0e3aa05c7c973e407c722125468bb506f1cc51e629042ab4d0a027ccc39"
COVERAGE_FILE = PROC / "scoring_input_coverage.csv"
N_TEST_HADNEWS = 44_001

SCORERS = ["lm", "finbert", "fingpt"]
DISP_FEATS = ["std", "range", "kurt"]


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def clustered_ols(y: np.ndarray, x: np.ndarray, weeks: np.ndarray):
    """Univariate OLS with intercept; week-clustered t and normal p on the slope."""
    Z = np.column_stack([np.ones(len(x)), x])
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    beta = ZtZ_inv @ (Z.T @ y)
    e = y - Z @ beta
    meat = np.zeros((2, 2))
    for w in pd.unique(weeks):
        idx = weeks == w
        s = Z[idx].T @ e[idx]
        meat += np.outer(s, s)
    V = ZtZ_inv @ meat @ ZtZ_inv
    t = beta[1] / math.sqrt(V[1, 1])
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return beta[1], t, p


def main() -> int:
    # ---- recover and verify the headline count ----
    print("=== STAGE 1: recover the per-firm-week headline count ===")
    # Verification (in place): scoring-input SHA256
    sha = hashlib.sha256(SCORING_FILE.read_bytes()).hexdigest()
    if sha != SCORING_SHA:
        stop(f"scoring_input.parquet SHA mismatch: {sha}")
    cov = pd.read_csv(COVERAGE_FILE)
    # Verification (in place): coverage columns and unique keys
    if list(cov.columns) != ["PERMNO", "target_week", "n_headlines"]:
        stop(f"coverage columns unexpected: {list(cov.columns)}")
    if cov.duplicated(["PERMNO", "target_week"]).any():
        stop("coverage keys not unique")
    print(f"  [PASS] STORED count found: {COVERAGE_FILE.name} "
          f"({len(cov):,} firm-weeks) - the S1 aggregation-side artefact")
    # Verification (in place): stored count equals recount
    si = pd.read_parquet(SCORING_FILE, columns=["PERMNO", "target_week"])
    rec = (si.groupby(["PERMNO", "target_week"]).size()
           .rename("n_rec").reset_index())
    cov["target_week"] = pd.to_datetime(cov["target_week"]).dt.date
    if not isinstance(rec["target_week"].iloc[0], type(cov["target_week"]
                                                       .iloc[0])):
        rec["target_week"] = pd.to_datetime(rec["target_week"]).dt.date
    chk = cov.merge(rec, on=["PERMNO", "target_week"], how="outer",
                    validate="1:1")
    if len(chk) != len(cov) or chk.isna().any().any() \
            or (chk["n_headlines"] != chk["n_rec"]).any():
        stop("stored count != recompute from frozen scoring_input.parquet")
    print(f"  [PASS] stored count == recompute from the frozen scored "
          f"headline table (SHA {SCORING_SHA[:8]}...), all {len(cov):,} "
          f"groups - these are exactly the groups module v2 aggregated "
          f"(count is arm-independent: same headline set feeds all three)")

    df = merge_frame()
    n0 = len(df)
    df = df.merge(cov.rename(columns={"target_week": "week_anchor"}),
                  on=["PERMNO", "week_anchor"], how="left", validate="m:1")
    # Verification (in place): count merge preserves row count
    if len(df) != n0:
        stop("count merge changed row count")
    hn1, hn0 = df["had_news"] == 1, df["had_news"] == 0
    # Verification (in place): count aligned with had_news
    if df.loc[hn1, "n_headlines"].isna().any() \
            or (df.loc[hn1, "n_headlines"] < 1).any():
        stop("had_news==1 row without a valid count - alignment broken")
    if df.loc[hn0, "n_headlines"].notna().any():
        stop("had_news==0 row carries a count - alignment broken")
    print(f"  [PASS] 1:1 alignment on model-frame keys: every had_news==1 "
          f"row has n>=1 ({int(hn1.sum()):,} rows); every had_news==0 row "
          f"has none. ({len(cov):,} covered grid keys - {int(hn1.sum()):,} "
          f"in the model frame = {len(cov) - int(hn1.sum())} on the 620 "
          f"D-EVAL-0-dropped rows)")
    sub = df[hn1].copy()
    one = sub["n_headlines"] == 1
    zero_cols = [f"{s}_{f}" for s in SCORERS for f in DISP_FEATS]
    # Verification (in place): n==1 edge rule
    if not (sub.loc[one, zero_cols] == 0).all().all():
        stop("edge-rule cross-check failed: n==1 rows with nonzero "
             "std/range/kurt - count does not match the feature groups")
    print(f"  [PASS] edge-rule cross-check: all {int(one.sum()):,} n==1 rows "
          f"have std=range=kurt=0 in ALL three arms (module v2 edge rule) - "
          f"count provably aligned with the feature groups")
    nn = sub["n_headlines"]
    print(f"  count distribution (had_news==1): median {nn.median():.0f}, "
          f"mean {nn.mean():.1f}, p90 {nn.quantile(0.9):.0f}, "
          f"max {int(nn.max()):,}")

    # ---- count-feature correlations ----
    print("\n=== STAGE 2: corr(count, feature) - all had_news==1 rows "
          f"(n = {len(sub):,}) ===")
    print(f"  {'feature':<15} {'spearman':>9} {'pearson':>9}")
    sp = {}
    for s in SCORERS:
        for f in DISP_FEATS + ["mean"]:
            col = f"{s}_{f}"
            r_s = sub["n_headlines"].corr(sub[col], method="spearman")
            r_p = sub["n_headlines"].corr(sub[col])
            sp[col] = r_s
            tag = "  <- contrast (level, not dispersion)" if f == "mean" else ""
            print(f"  {col:<15} {r_s:>+9.4f} {r_p:>+9.4f}{tag}")

    # ---- return slopes before and after residualising on count ----
    print("\n=== STAGE 3: does dispersion survive controlling for count? "
          "(test rows 2021-23, had_news==1) ===")
    yr = pd.to_datetime(sub["week_anchor"]).dt.year
    te = sub[yr.isin(YEARS)].copy()
    # Verification (in place): test had_news sample size
    if len(te) != N_TEST_HADNEWS:
        stop(f"test had_news sample {len(te):,} != pipeline/21's "
             f"{N_TEST_HADNEWS:,}")
    weeks = te["week_anchor"].to_numpy()
    y = te[TARGET_FWD].to_numpy(float)
    n_arr = te["n_headlines"].to_numpy(float)
    print(f"  sample: n = {len(te):,}, {len(pd.unique(weeks))} weeks "
          f"(== pipeline/21 Stage-2 sample)")
    print(f"  {'feature':<15} {'R2(disp~n)':>10} | "
          f"{'raw b':>9} {'raw t':>7} | {'resid b':>9} {'resid t':>7}")
    surv = {}
    for s in SCORERS:
        for f in ("std", "range"):        # kurt: tiny, unstable ridge coefs
            col = f"{s}_{f}"
            x = te[col].to_numpy(float)
            Z = np.column_stack([np.ones(len(x)), n_arr])
            g = np.linalg.lstsq(Z, x, rcond=None)[0]
            resid = x - Z @ g
            r2 = 1.0 - float((resid ** 2).sum()) / float(
                ((x - x.mean()) ** 2).sum())
            b_raw, t_raw, _ = clustered_ols(y, x, weeks)
            b_res, t_res, _ = clustered_ols(y, resid, weeks)
            surv[col] = (r2, t_raw, t_res)
            print(f"  {col:<15} {r2:>10.4f} | {b_raw * 100:>+8.4f}% "
                  f"{t_raw:>+7.2f} | {b_res * 100:>+8.4f}% {t_res:>+7.2f}")
    b_n, t_n, p_n = clustered_ols(y, n_arr, weeks)
    b_l, t_l, _ = clustered_ols(y, np.log1p(n_arr), weeks)
    print(f"  count alone:    ret_excess_fwd ~ n_headlines: "
          f"b = {b_n * 100:+.5f}%/headline, t = {t_n:+.2f}, p = {p_n:.4f} "
          f"(context: ~log(1+n): t = {t_l:+.2f})")

    # ---- verdict ----
    print("\n=== STAGE 4: verdict (pre-stated thresholds) + STOP ===")
    labels = {}
    for s in SCORERS:
        r_std, r_rng = abs(sp[f"{s}_std"]), abs(sp[f"{s}_range"])
        worst = max(r_std, r_rng)
        _, t_raw_s, t_res_s = surv[f"{s}_std"]
        _, t_raw_r, t_res_r = surv[f"{s}_range"]
        died = (abs(t_raw_s) >= 1.96 and abs(t_res_s) < 1.0) or \
               (abs(t_raw_r) >= 1.96 and abs(t_res_r) < 1.0)
        if worst > 0.6 or died:
            lab = "NEWS VOLUME, not disagreement"
        elif worst >= 0.3:
            lab = "disagreement, count-attenuated"
        else:
            lab = "disagreement (largely NOT count)"
        labels[s] = lab
        print(f"  {s:<8} max |Spearman(n, std/range)| = {worst:.4f} | "
              f"raw t (std/range) {t_raw_s:+.2f}/{t_raw_r:+.2f} -> resid t "
              f"{t_res_s:+.2f}/{t_res_r:+.2f} | label: {lab}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
