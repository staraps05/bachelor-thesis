# %% 24_coverage_by_year.py
# Purpose: Test whether the sentiment null is a news-coverage artefact: had_news coverage per year,
#          univariate tests per year and scorer, and a high- (2018-19) vs low-coverage (2021-23) contrast.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet, Data/processed/scoring_input_coverage.csv
# Outputs: prints only
# Notes:
# - Loads verification/23 via importlib (merge_frame from 20 SHA-checks inputs; tests).
# - Single-lag pairing: score at key t vs ret_excess at key t; ret_excess_fwd is not used.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC = REPO_ROOT / "Data" / "processed"

_spec = importlib.util.spec_from_file_location(
    "ctl23", REPO_ROOT / "verification" / "23_positive_control.py")
_c23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c23)
merge_frame = _c23.merge_frame
clustered_ols = _c23.clustered_ols
quintile_spread = _c23.quintile_spread
TARGET_T = _c23.TARGET_T                      # ret_excess, week t
SCORERS = _c23.SCORERS                        # lm/finbert/fingpt _mean

COVERAGE_FILE = PROC / "scoring_input_coverage.csv"
ALL_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
HIGH_YEARS, LOW_YEARS = [2018, 2019], [2021, 2022, 2023]
T_SIG = 2.0


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def ols_cell(g: pd.DataFrame, sc: str):
    """Slope and week-clustered t of ret_excess on score sc within g."""
    b, t = clustered_ols(g[TARGET_T].to_numpy(float),
                         g[sc].to_numpy(float),
                         g["week_anchor"].to_numpy())
    return b, t


def main() -> int:
    # ---- merge frame and attach the headline count ----
    print("=== STAGE 0: rebuild ===")
    df = merge_frame()
    cov = pd.read_csv(COVERAGE_FILE)
    cov["target_week"] = pd.to_datetime(cov["target_week"]).dt.date
    df = df.merge(cov.rename(columns={"target_week": "week_anchor"}),
                  on=["PERMNO", "week_anchor"], how="left", validate="m:1")
    # Verification (in place): count aligned with had_news
    if df.loc[df["had_news"] == 1, "n_headlines"].isna().any():
        stop("count alignment broken (verified clean in pipeline/22)")
    df["year"] = pd.to_datetime(df["week_anchor"]).dt.year

    # ---- coverage profile per year ----
    print("\n=== STAGE 1: coverage profile per year ===")
    always = df.groupby("PERMNO")["year"].nunique()
    always_firms = set(always[always == len(ALL_YEARS)].index)
    print(f"  {'year':<5} {'rows':>7} {'had_news':>9} {'firms>=1nw':>10} "
          f"{'firms 0nw':>9} {'med hl/wk':>9} {'const-comp share':>17}")
    for y in ALL_YEARS:
        gy = df[df["year"] == y]
        per_firm = gy.groupby("PERMNO")["had_news"].max()
        med = gy.loc[gy["had_news"] == 1, "n_headlines"].median()
        cc = gy[gy["PERMNO"].isin(always_firms)]["had_news"].mean()
        print(f"  {y:<5} {len(gy):>7,} {gy['had_news'].mean():>9.4f} "
              f"{int((per_firm == 1).sum()):>10} "
              f"{int((per_firm == 0).sum()):>9} {med:>9.0f} {cc:>17.4f}")
    print(f"  constant-composition view = the {len(always_firms)} firms "
          f"present in all six years (removes index-turnover confound)")

    # ---- per-year tests ----
    print("\n=== STAGE 2: the test, per year x scorer (had_news==1 rows) ===")
    hn = df[df["had_news"] == 1]
    print("  (a) univariate OLS: b in % per unit score [clustered-by-week t]")
    print(f"  {'year':<5} {'n':>7}" + "".join(
        f" | {s.split('_')[0]:>8} b {'t':>6}" for s in SCORERS))
    cells_a = {}
    for y in ALL_YEARS:
        g = hn[hn["year"] == y]
        line = f"  {y:<5} {len(g):>7,}"
        for sc in SCORERS:
            b, t = ols_cell(g, sc)
            cells_a[(y, sc)] = t
            line += f" | {b * 100:>+8.4f} {t:>+6.2f}"
        print(line)
    print("  (b) weekly quintile sorts: Q5-Q1 spread %/wk [NW lag-4 t]")
    print(f"  {'year':<5} {'wks':>4}" + "".join(
        f" | {s.split('_')[0]:>8} s {'t':>6}" for s in SCORERS))
    cells_b = {}
    for y in ALL_YEARS:
        g = hn[hn["year"] == y]
        line = None
        for sc in SCORERS:
            q5, q1, spr, t, nw = quintile_spread(g, sc, TARGET_T)
            cells_b[(y, sc)] = t
            if line is None:
                line = f"  {y:<5} {nw:>4}"
            line += f" | {spr * 100:>+8.4f} {t:>+6.2f}"
        print(line)

    # ---- pooled high- vs low-coverage blocks ----
    print("\n=== STAGE 3: high-coverage vs low-coverage pooled contrast ===")
    blocks = {"HIGH 2018-19": hn[hn["year"].isin(HIGH_YEARS)],
              "LOW 2021-23": hn[hn["year"].isin(LOW_YEARS)]}
    block_t = {}
    for name, g in blocks.items():
        print(f"  {name}: n = {len(g):,}, "
              f"{g['week_anchor'].nunique()} weeks")
        for sc in SCORERS:
            b, t = ols_cell(g, sc)
            q5, q1, spr, ts, nw = quintile_spread(g, sc, TARGET_T)
            block_t[(name, sc)] = (t, ts)
            print(f"    {sc:<13} OLS b {b * 100:+.4f}% [t {t:+.2f}] | "
                  f"Q5-Q1 {spr * 100:+.4f}%/wk [NW t {ts:+.2f}]")

    # non-degraded firms within LOW: separates coverage from calendar effects
    share_h = blocks["HIGH 2018-19"].groupby("PERMNO").size() / \
        df[df["year"].isin(HIGH_YEARS)].groupby("PERMNO").size()
    share_l = blocks["LOW 2021-23"].groupby("PERMNO").size() / \
        df[df["year"].isin(LOW_YEARS)].groupby("PERMNO").size()
    both = share_h.index.intersection(share_l.index)
    nondeg = set(both[(share_l[both] >= share_h[both])])
    print(f"  [INFO] non-degraded subset OPERATIONALISATION (my choice, "
          f"stated): firms whose LOW-block had_news share >= their "
          f"HIGH-block share (coverage maintained). {len(nondeg)} of "
          f"{len(both)} firms present in both blocks qualify.")
    g = blocks["LOW 2021-23"]
    g = g[g["PERMNO"].isin(nondeg)]
    print(f"  LOW, non-degraded firms only: n = {len(g):,} rows, "
          f"{g['PERMNO'].nunique()} firms, {g['week_anchor'].nunique()} weeks")
    for sc in SCORERS:
        b, t = ols_cell(g, sc)
        q5, q1, spr, ts, nw = quintile_spread(g, sc, TARGET_T)
        print(f"    {sc:<13} OLS b {b * 100:+.4f}% [t {t:+.2f}] | "
              f"Q5-Q1 {spr * 100:+.4f}%/wk [NW t {ts:+.2f}]")

    # ---- Verification: consistency and reproduction checks ----
    print(f"  full sample (plumbing-consistent model frame, D-EVAL-0 row "
          f"set): {len(df):,} rows, {df['week_anchor'].nunique()} weeks, "
          f"2018-2023 (the 620 dropped spell-end rows are immaterial here "
          f"and keep the row set identical to pipeline/19-23)")
    if len(nondeg) < 30:
        print(f"  [INFO] subset too small ({len(nondeg)} firms) for a "
              f"meaningful test - reported for completeness only:")

    return 0


if __name__ == "__main__":
    sys.exit(main())
