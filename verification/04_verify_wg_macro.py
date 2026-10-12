# %% 04_verify_wg_macro.py
# Purpose: Verify the Welch-Goyal monthly macro predictor file (extended
#          Goyal-Welch-Zafirov version): monthly frequency over the full sample,
#          gap-free coverage of 2015-01..2023-12, in-window missingness per column,
#          duplicate months, and an eyeball printout of rate columns.
#          Read-only: writes nothing.
# Inputs:  Data/Macro_data_aggregated.csv
# Outputs: prints only
# Notes:
# - Strips the write bits of Data/Macro_data_aggregated.csv before reading, so the raw file stays immutable.
# - Column names are only whitespace-stripped, not lower-cased: WG names such as
#   "Rfree", "AAA" and "b/m" are case-sensitive conventions.
# - Columns with more than 20% NaN inside the window are flagged (several extended
#   WG series are discontinued in recent years); flags do not fail the run.
# - Rates (tbl, Rfree, AAA, ...) are expected as decimals (0.04 = 4%); the eyeball
#   rows exist to confirm this by inspection, there is no automatic check.
# - Deterministic; pandas + stdlib only. Exit status 1 if any check fails.
# Verification script: checks only; writes nothing to Data/

import stat
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = REPO_ROOT / "Data" / "Macro_data_aggregated.csv"

DATE_COL = "yyyymm"
WINDOW = ("2015-01", "2023-12")           # must be gap-free
NAN_FLAG_SHARE = 0.20                     # flag columns above this NaN share in-window
EYEBALL_COLS = ["price", "tbl", "Rfree", "infl", "AAA", "BAA", "tms", "dfy"]


def check(label: str, passed: bool, observed: str) -> bool:
    """Print one PASS/FAIL line with the observed value; return passed."""
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {observed}")
    return passed


def main() -> int:
    if not RAW_FILE.exists():
        sys.exit(f"Required file not found: {RAW_FILE}")

    # strip all write bits; the raw file is read-only from here on
    RAW_FILE.chmod(RAW_FILE.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    # ---- load ----
    df = pd.read_csv(RAW_FILE)
    df.columns = df.columns.str.strip()
    df["month"] = pd.to_datetime(df[DATE_COL].astype(str), format="%Y%m")

    print("=== Welch-Goyal macro file overview ===")
    print(f"  file    : {RAW_FILE}")
    print(f"  columns ({df.shape[1] - 1}): {[c for c in df.columns if c != 'month']}")
    print(f"  rows    : {len(df):,}")
    print(f"  range   : {df['month'].min():%Y-%m} .. {df['month'].max():%Y-%m}")

    all_pass = True
    # ---- check: monthly frequency over the full sample ----
    # compare integer month ordinals; diffing Periods gives DateOffsets, not ints
    ords = df["month"].dt.to_period("M").astype("int64").sort_values()
    n_bad_steps = int((ords.diff().dropna() != 1).sum())
    all_pass &= check("frequency is monthly (all steps = 1 month)", n_bad_steps == 0,
                      f"{n_bad_steps} non-monthly step(s)")

    # ---- check: window coverage ----
    print(f"\n=== Window coverage {WINDOW[0]} .. {WINDOW[1]} ===")
    expected = pd.period_range(WINDOW[0], WINDOW[1], freq="M")
    present = set(df["month"].dt.to_period("M"))
    missing = [str(p) for p in expected if p not in present]
    all_pass &= check(f"all {len(expected)} months present", not missing,
                      f"{len(missing)} missing" + (f" -> {missing}" if missing else ""))

    # ---- in-window missingness (flags only) ----
    win = df[(df["month"].dt.to_period("M") >= expected[0])
             & (df["month"].dt.to_period("M") <= expected[-1])]
    pred_cols = [c for c in df.columns if c not in (DATE_COL, "month")]
    nan_share = win[pred_cols].isna().mean().sort_values(ascending=False)
    flagged = nan_share[nan_share > NAN_FLAG_SHARE]
    print(f"\n=== In-window missingness ({len(win)} rows) ===")
    print(f"  columns > {NAN_FLAG_SHARE:.0%} NaN in-window: {len(flagged)}")
    for col, share in flagged.items():
        print(f"    FLAG {col:<12s} {share * 100:6.2f}% NaN")
    clean = nan_share[nan_share == 0]
    partial = nan_share[(nan_share > 0) & (nan_share <= NAN_FLAG_SHARE)]
    print(f"  columns fully populated in-window : {len(clean)}")
    print(f"    {sorted(clean.index.tolist())}")
    if len(partial):
        print(f"  columns partially missing (<= {NAN_FLAG_SHARE:.0%}):")
        for col, share in partial.items():
            print(f"    {col:<12s} {share * 100:6.2f}% NaN")

    # ---- check: duplicate months ----
    n_dups = int(df[DATE_COL].duplicated().sum())
    print("\n=== Duplicates ===")
    all_pass &= check("no duplicate months", n_dups == 0, f"{n_dups} duplicate(s)")

    # ---- eyeball rows: rates should read as decimals (0.04 = 4%) ----
    print("\n=== Eyeball: first 3 / last 3 rows ===")
    show = df[[DATE_COL] + [c for c in EYEBALL_COLS if c in df.columns]]
    print(pd.concat([show.head(3), show.tail(3)]).to_string(index=False))

    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED - inspect above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
