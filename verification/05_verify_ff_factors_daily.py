# %% 05_verify_ff_factors_daily.py
# Purpose: Verify the Fama-French 3-factor daily file (source of daily RF and Mkt-RF): columns, 2016-01-04..2023-12-29 coverage, business-day frequency, percent scaling, missing values, duplicate dates.
# Inputs:  Data/F-F_Research_Data_Factors_daily.csv
# Outputs: prints only
# Notes:
# - Verification script: checks only; writes nothing to Data/
# - Strips the write bits of the raw file before reading, so it stays immutable.
# - Values are in percent (consumers divide by 100). Exit status 1 if any check fails.

import io
import re
import stat
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = REPO_ROOT / "Data" / "F-F_Research_Data_Factors_daily.csv"

REQUIRED_WINDOW = (date(2016, 1, 4), date(2023, 12, 29))    # first/last trading days
DAILY_ROW = re.compile(r"^\s*\d{8}\s*,")                    # row starting with YYYYMMDD
MISSING_CODES = (-99.99, -999.0)                            # French missing-value codes


def check(label: str, passed: bool, observed: str) -> bool:
    """Print one PASS/FAIL line with the observed value; return passed."""
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {observed}")
    return passed


def main() -> int:
    if not RAW_FILE.exists():
        sys.exit(f"Required file not found: {RAW_FILE}")

    # strip all write bits; the raw file is read-only from here on
    RAW_FILE.chmod(RAW_FILE.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    # ---- locate the daily block: contiguous run of YYYYMMDD rows ----
    lines = RAW_FILE.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if DAILY_ROW.match(ln))
    end = start
    while end < len(lines) and DAILY_ROW.match(lines[end]):
        end += 1
    daily_lines = lines[start:end]

    # header is the line above the first data row; its empty first field is the date
    header = [f.strip() or "Date" for f in lines[start - 1].split(",")]

    # numeric rows after the daily block belong to an appended annual/monthly block
    trailing = lines[end:]
    appendix = [ln for ln in trailing if re.match(r"^\s*\d{4,6}\s*,", ln)]

    print("=== File structure ===")
    print(f"  file                    : {RAW_FILE}")
    print(f"  preamble rows skipped   : {start - 1} text line(s) + 1 header line")
    print(f"  daily block             : lines {start + 1}..{end} ({len(daily_lines):,} rows)")
    print(f"  annual/appendix block   : "
          + (f"DETECTED and excluded ({len(appendix)} data row(s))" if appendix
             else f"none detected ({len(trailing)} trailing text/blank line(s) ignored)"))

    # ---- load the daily block only ----
    df = pd.read_csv(io.StringIO("\n".join(daily_lines)), names=header)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

    print("\n=== Daily factor table ===")
    print(f"  columns : {header}")
    print(f"  rows    : {len(df):,}")

    all_pass = True
    all_pass &= check("expected columns present",
                      header == ["Date", "Mkt-RF", "SMB", "HML", "RF"], f"{header}")

    # ---- check: window coverage and business-day frequency ----
    dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
    print(f"  range   : {dmin} .. {dmax}")
    all_pass &= check("covers 2016-01 .. 2023-12",
                      dmin <= REQUIRED_WINDOW[0] and dmax >= REQUIRED_WINDOW[1],
                      f"{dmin} .. {dmax}")
    in_win = df[(df["Date"].dt.date >= REQUIRED_WINDOW[0])
                & (df["Date"].dt.date <= REQUIRED_WINDOW[1])]
    print(f"  trading days in 2016-01-04..2023-12-29: {len(in_win):,} (~252/yr * 8y = ~2016)")

    # full-sample weekend rows are context only; the frequency check is in-window
    n_weekend_full = int((df["Date"].dt.dayofweek >= 5).sum())
    print(f"  full-sample quirks: {n_weekend_full} Saturday session(s) (NYSE pre-1952), "
          f"gaps >5d at 1933 bank holiday and 9/11 - expected, not errors")
    n_weekend = int((in_win["Date"].dt.dayofweek >= 5).sum())
    max_gap = int(in_win["Date"].sort_values().diff().dt.days.max())
    all_pass &= check("business-day frequency in thesis window",
                      n_weekend == 0 and max_gap <= 5,
                      f"{n_weekend} weekend row(s), max calendar gap {max_gap} day(s)")

    # ---- check: scale ----
    # median daily |Mkt-RF| is ~0.5 in percent, ~0.005 in decimals
    print("\n=== Scale check (values stored in PERCENT?) ===")
    for col in ("Mkt-RF", "RF"):
        s = df[col]
        print(f"  {col:<6s}: min {s.min():+8.4f}, max {s.max():+8.4f}, mean {s.mean():+8.5f}")
    med_abs_mkt = df["Mkt-RF"].abs().median()
    is_percent = med_abs_mkt > 0.05
    all_pass &= check("values are in PERCENT (divide by 100 for decimals)", is_percent,
                      f"median |Mkt-RF| = {med_abs_mkt:.4f} "
                      f"({'~0.5 => percent' if is_percent else '~0.005 => decimals!'})")

    # ---- check: missingness, including French missing codes ----
    print("\n=== Missingness ===")
    for col in header[1:]:
        n_nan = int(df[col].isna().sum())
        n_code = int(df[col].isin(MISSING_CODES).sum())
        print(f"  {col:<6s}: {n_nan} NaN ({n_nan / len(df) * 100:.3f}%), "
              f"{n_code} French missing-code value(s)")
        all_pass &= (n_nan == 0 and n_code == 0) or check(
            f"{col} has missing data", False, f"{n_nan} NaN / {n_code} coded")

    # ---- check: duplicates ----
    n_dups = int(df["Date"].duplicated().sum())
    print("\n=== Duplicates ===")
    all_pass &= check("Date unique", n_dups == 0, f"{n_dups} duplicate date(s)")

    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED - inspect above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
