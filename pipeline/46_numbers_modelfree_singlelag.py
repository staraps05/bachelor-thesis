# %% 46_numbers_modelfree_singlelag.py
# Purpose: Reproduce Table C.5 (model-free univariate slope and weekly Q5-Q1
#          quintile spread per scorer) under the single-lag pairing of
#          pipeline/26, next to the double-lag values of pipeline/41's round-1
#          outputs.
# Inputs:  Data/processed/model_frame_structured.parquet (via merge_frame, single_lag_frame)
#          Data/processed/features_lm.parquet
#          Data/processed/features_finbert.parquet
#          Data/processed/features_fingpt.parquet
#          Data/processed/scoring_input_coverage.csv
#          Data/processed/numbers_check/tableC5_modelfree.csv
#          Data/processed/numbers_check/tableC7_count_corr.csv
# Outputs: Data/processed/numbers_check/tableC5_modelfree_singlelag.csv
# Notes:
# - Loads pipeline/41_numbers_modelfree_countcorr.py (clustered_ols, dm_test,
#   merge_frame, constants) and pipeline/27_covered_subsample.py
#   (single_lag_frame) via importlib; the test logic is pipeline/41's
#   unchanged, only the frame differs.
# - Double lag (pipeline/41): sentiment keyed on feature week t, two-week gap
#   to the target. Single lag (pipeline/26): sentiment and had_news
#   re-attached from the week-(t+1) key.
# - Sample: test-year (2021-2023) firm-weeks with had_news == 1. The slope t
#   is clustered by week; the spread is the weekly mean target of quintile 5
#   minus quintile 1 (rank ties broken by row order), tested with the NW DM
#   test of pipeline/41; weeks with fewer than 5 covered firms are skipped.
# - Two gates stop the run: the double-lag C.5 and C.7 values must match the
#   recorded CSVs within 1e-9, and the single-lag frame must have the
#   pipeline/26 row count, test-row count and had_news share.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC = REPO_ROOT / "Data" / "processed"
NC = PROC / "numbers_check"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_c41 = _load("mf41", "41_numbers_modelfree_countcorr.py")
_c27 = _load("cov27", "27_covered_subsample.py")
clustered_ols = _c41.clustered_ols
merge_frame, dm_test = _c41.merge_frame, _c41.dm_test
single_lag_frame = _c27.single_lag_frame
TARGET_FWD, YEARS = _c41.TARGET_FWD, _c41.YEARS
SCORERS = _c41.SCORERS
COVERAGE_FILE = _c41.COVERAGE_FILE

SCORER_MEAN = {"lm": "lm_mean", "finbert": "finbert_mean", "fingpt": "fingpt_mean"}


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def modelfree(df):
    """Table C.5 per scorer: univariate OLS slope (t clustered by week) and mean
    weekly Q5-Q1 spread (NW t) on test firm-weeks with had_news == 1."""
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    sub = df.loc[yr.isin(YEARS) & (df["had_news"] == 1),
                 ["week_anchor", TARGET_FWD] + list(SCORER_MEAN.values())].copy()
    weeks = sub["week_anchor"].to_numpy()
    y = sub[TARGET_FWD].to_numpy(float)
    out = {}
    for sc in SCORERS:
        col = SCORER_MEAN[sc]
        b, t = clustered_ols(y, sub[col].to_numpy(float), weeks)
        rows = []
        for w, g in sub.groupby("week_anchor"):
            if len(g) < 5:            # a quintile sort needs >= 5 firms
                continue
            r = g[col].rank(method="first")
            q = pd.qcut(r, 5, labels=False)
            rows.append((g.loc[q == 4, TARGET_FWD].mean(), g.loc[q == 0, TARGET_FWD].mean()))
        spread = np.array([a for a, _ in rows]) - np.array([b2 for _, b2 in rows])
        ts, _ = dm_test(spread)
        out[sc] = (b * 100, t, spread.mean() * 100, ts, len(sub), len(spread))
    return out


def countcorr(df):
    """Table C.7: Spearman correlation of the headline count with each sentiment
    feature on covered firm-weeks; used for the double-lag gate only."""
    cov = pd.read_csv(COVERAGE_FILE)
    cov["target_week"] = pd.to_datetime(cov["target_week"]).dt.date
    dfc = df.merge(cov.rename(columns={"target_week": "week_anchor"}),
                   on=["PERMNO", "week_anchor"], how="left", validate="m:1")
    rows = dfc[dfc["had_news"] == 1].copy()
    out = {}
    for sc in SCORERS:
        out[sc] = {f: float(rows["n_headlines"].corr(rows[f"{sc}_{f}"], method="spearman"))
                   for f in ("mean", "std", "range", "kurt")}
    out["_n"] = len(rows)
    return out


def main() -> int:
    NC.mkdir(parents=True, exist_ok=True)
    print("=== 46: model-free single-lag re-run ===")

    # gate 1: double-lag C.5/C.7 must match the CSVs written by pipeline/41
    df_dl = merge_frame()
    mf_dl = modelfree(df_dl)
    # Verification (in place): gate 1: double-lag C.5 and C.7 reproduce pipeline/41's CSVs
    cc_dl = countcorr(df_dl)
    c5r = pd.read_csv(NC / "tableC5_modelfree.csv").set_index("scorer")
    c7r = pd.read_csv(NC / "tableC7_count_corr.csv").set_index("scorer")
    maxdev = 0.0
    for sc in SCORERS:
        b, t, sp, ts, n, nwk = mf_dl[sc]
        maxdev = max(maxdev, abs(b - c5r.loc[sc, "slope_pct"]), abs(t - c5r.loc[sc, "slope_t"]),
                     abs(sp - c5r.loc[sc, "q5mq1_spread_pct"]), abs(ts - c5r.loc[sc, "spread_nw_t"]))
        for f in ("mean", "std", "range", "kurt"):
            maxdev = max(maxdev, abs(cc_dl[sc][f] - c7r.loc[sc, f"spearman_{f}"]))
    if maxdev > 1e-9:
        stop(f"GATE1 FAIL: double-lag C.5/C.7 reproduction max dev {maxdev:.3e}")

    # gate 2: single-lag frame must match pipeline/26 (rows, test rows, had_news share)
    df = single_lag_frame()
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    n_test = int(yr.isin(list(YEARS)).sum())
    hn = df["had_news"].mean()
    # Verification (in place): gate 2: single-lag frame rows, test rows, had_news share
    if len(df) != 157_277 or n_test != 78_056 or abs(hn - 0.6620) > 5e-4:
        stop(f"GATE2 FAIL: rows {len(df)}, test {n_test}, had_news {hn:.4f}")

    # single-lag C.5
    mf_sl = modelfree(df)
    rows = []
    for sc in SCORERS:
        b_dl, t_dl, sp_dl, ts_dl, n_dl, nw_dl = mf_dl[sc]
        b_sl, t_sl, sp_sl, ts_sl, n_sl, nw_sl = mf_sl[sc]
        rows.append({"scorer": sc,
                     "slope_dl_pct": b_dl, "slope_dl_t": t_dl, "slope_dl_sig5": abs(t_dl) > 1.96,
                     "slope_sl_pct": b_sl, "slope_sl_t": t_sl, "slope_sl_sig5": abs(t_sl) > 1.96,
                     "spread_dl_pct": sp_dl, "spread_dl_t": ts_dl, "spread_dl_sig5": abs(ts_dl) > 1.96,
                     "spread_sl_pct": sp_sl, "spread_sl_t": ts_sl, "spread_sl_sig5": abs(ts_sl) > 1.96,
                     "n_firmweeks_dl": n_dl, "n_weeks_dl": nw_dl,
                     "n_firmweeks_sl": n_sl, "n_weeks_sl": nw_sl})
    pd.DataFrame(rows).to_csv(NC / "tableC5_modelfree_singlelag.csv", index=False)
    print(f"  single-lag sample: n_firmweeks={mf_sl['lm'][4]:,}, n_weeks={mf_sl['lm'][5]}")
    print("  --- C.5 (dl -> sl) slope[t] | spread[t] ---")
    for r in rows:
        print(f"    {r['scorer']:<8} slope {r['slope_dl_pct']:+.4f}[{r['slope_dl_t']:+.2f}] -> "
              f"{r['slope_sl_pct']:+.4f}[{r['slope_sl_t']:+.2f}] | spread {r['spread_dl_pct']:+.4f}"
              f"[{r['spread_dl_t']:+.2f}] -> {r['spread_sl_pct']:+.4f}[{r['spread_sl_t']:+.2f}]")
    print("  wrote tableC5_modelfree_singlelag.csv")
    # ---- Verification: consistency and reproduction checks ----
    print(f"  [PASS] GATE 1: double-lag C.5 & C.7 reproduce exactly (max dev {maxdev:.2e})")
    print(f"  [PASS] GATE 2: single-lag frame {len(df):,} rows, {n_test:,} test rows, had_news {hn:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
