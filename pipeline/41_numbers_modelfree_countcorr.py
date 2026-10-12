# %% 41_numbers_modelfree_countcorr.py
# Purpose: Reproduce Table C.7 (Spearman correlation between each sentiment
#          feature and the weekly headline count on covered firm-weeks) and the
#          round-1 model-free values of Table C.5 (univariate OLS slope with
#          week-clustered t and weekly quintile Q5-Q1 spread on covered test
#          firm-weeks), on the double-lag frame.
# Inputs:  Data/processed/model_frame_structured.parquet (via merge_frame)
#          Data/processed/features_lm.parquet
#          Data/processed/features_finbert.parquet
#          Data/processed/features_fingpt.parquet
#          Data/processed/scoring_input_coverage.csv (weekly headline counts)
#          Data/processed/scoring_input.parquet (hashed only; the digest is unused)
# Outputs: Data/processed/numbers_check/tableC5_modelfree.csv
#          Data/processed/numbers_check/tableC7_count_corr.csv
# Notes:
# - Loads pipeline/20_seed_stability.py via importlib for merge_frame, dm_test,
#   TARGET_FWD and YEARS. This script is itself loaded by pipeline/46, which
#   re-runs Table C.5 single-lag (clustered_ols, SCORERS, COVERAGE_FILE) and
#   reads the two CSVs above for comparison.
# - Pairing is double lag (sentiment keyed on the feature week t, target is
#   week t+1), as in pipeline/21-22; pipeline/46 is the single-lag counterpart.
# - Table C.5 uses test-year (2021-2023) rows with had_news == 1. The slope t
#   uses week-clustered SEs; quintiles are formed within each week by rank
#   (ties broken by row order); the spread's t is Newey-West via dm_test (lag
#   DM_LAG from pipeline/19).
# - Table C.7 uses all had_news == 1 rows of every year; headline counts are
#   merged from the coverage file and must be present exactly on the
#   had_news == 1 rows or the run stops.
# - Deterministic: no XGBoost, no seeds.

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC = REPO_ROOT / "Data" / "processed"
OUT = PROC / "numbers_check"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_c20 = _load("seed20", "20_seed_stability.py")
merge_frame, dm_test = _c20.merge_frame, _c20.dm_test
TARGET_FWD, YEARS = _c20.TARGET_FWD, _c20.YEARS

COVERAGE_FILE = PROC / "scoring_input_coverage.csv"
SCORERS = ["lm", "finbert", "fingpt"]


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def clustered_ols(y, x, weeks):
    """Univariate OLS of y on x with week-clustered SEs; returns (slope, t)."""
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
    return beta[1], t


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== 41: model-free (C.5) + count-corr (C.7), double-lag frame ===")
    df = merge_frame()
    yr = pd.to_datetime(df["week_anchor"]).dt.year

    # ---------- Table C.5: model-free on test rows 2021-23, had_news==1 ----------
    scorer_mean = {"lm": "lm_mean", "finbert": "finbert_mean", "fingpt": "fingpt_mean"}
    sub = df.loc[yr.isin(YEARS) & (df["had_news"] == 1),
                 ["week_anchor", TARGET_FWD] + list(scorer_mean.values())].copy()
    n_test = len(sub)
    weeks = sub["week_anchor"].to_numpy()
    y = sub[TARGET_FWD].to_numpy(float)
    c5 = []
    for sc in SCORERS:
        col = scorer_mean[sc]
        b, t = clustered_ols(y, sub[col].to_numpy(float), weeks)
        # weekly quintile sort
        rows = []
        for w, g in sub.groupby("week_anchor"):
            r = g[col].rank(method="first")
            q = pd.qcut(r, 5, labels=False)
            rows.append((g.loc[q == 4, TARGET_FWD].mean(), g.loc[q == 0, TARGET_FWD].mean()))
        spread = np.array([a for a, _ in rows]) - np.array([b2 for _, b2 in rows])
        ts, _ = dm_test(spread)
        c5.append({"scorer": sc, "slope_pct": b * 100, "slope_t": t,
                   "q5mq1_spread_pct": spread.mean() * 100, "spread_nw_t": ts,
                   "n": n_test, "n_weeks": len(spread)})
    pd.DataFrame(c5).to_csv(OUT / "tableC5_modelfree.csv", index=False)
    print(f"  C.5 model-free (n={n_test:,}):")
    for r in c5:
        print(f"    {r['scorer']:<8} slope {r['slope_pct']:+.4f} (t {r['slope_t']:+.2f}) | "
              f"spread {r['q5mq1_spread_pct']:+.4f} (NW t {r['spread_nw_t']:+.2f})")

    # ---------- Table C.7: Spearman(count, feature) on all had_news==1 rows ----------
    sha = hashlib.sha256((PROC / "scoring_input.parquet").read_bytes()).hexdigest()
    cov = pd.read_csv(COVERAGE_FILE)
    cov["target_week"] = pd.to_datetime(cov["target_week"]).dt.date
    n0 = len(df)
    dfc = df.merge(cov.rename(columns={"target_week": "week_anchor"}),
                   on=["PERMNO", "week_anchor"], how="left", validate="m:1")
    # Verification (in place): count merge preserves row count
    if len(dfc) != n0:
        stop("count merge changed row count")
    hn1 = dfc["had_news"] == 1
    # Verification (in place): count aligned with had_news
    if dfc.loc[hn1, "n_headlines"].isna().any() or dfc.loc[~hn1, "n_headlines"].notna().any():
        stop("count alignment broken")
    cov_rows = dfc[hn1].copy()
    n_cov = len(cov_rows)
    c7 = []
    for sc in SCORERS:
        rec = {"scorer": sc, "n": n_cov}
        for feat in ["mean", "std", "range", "kurt"]:
            col = f"{sc}_{feat}"
            rec[f"spearman_{feat}"] = float(cov_rows["n_headlines"].corr(
                cov_rows[col], method="spearman"))
        c7.append(rec)
    pd.DataFrame(c7).to_csv(OUT / "tableC7_count_corr.csv", index=False)
    print(f"  C.7 count-corr (n={n_cov:,}):")
    for r in c7:
        print(f"    {r['scorer']:<8} mean {r['spearman_mean']:+.4f} std {r['spearman_std']:+.4f} "
              f"range {r['spearman_range']:+.4f} kurt {r['spearman_kurt']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
