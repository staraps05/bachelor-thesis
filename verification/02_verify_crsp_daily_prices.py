# %% 02_verify_crsp_daily_prices.py
# Purpose: Verify the WRDS CRSP daily stock file for the S&P 500 universe: PERMNO coverage, AAPL 4:1 split, return range, missingness, negative-price convention, (permno, dlycaldt) uniqueness.
# Inputs:  Data/crsp_daily_prices.csv, Data/processed/sp500_permnos.txt
# Outputs: prints only
# Notes:
# - Verification script: checks only; writes nothing to Data/
# - Strips the write bits of the raw file before reading, so it stays immutable.
# - Exit status 1 if any check fails; universe PERMNOs absent from the file only reported.

import stat
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = REPO_ROOT / "Data" / "crsp_daily_prices.csv"
PERMNO_FILE = REPO_ROOT / "Data" / "processed" / "sp500_permnos.txt"

# only the columns the checks need
USECOLS = ["permno", "ticker", "dlycaldt", "dlyprc", "dlyret"]

# spot-check expectations: AAPL 4:1 split, first post-split trading day 2020-08-31
AAPL_PERMNO = 14593
SPLIT_LAST_PRESPLIT = date(2020, 8, 28)
SPLIT_FIRST_POSTSPLIT = date(2020, 8, 31)
SPLIT_WINDOW = (date(2020, 8, 26), date(2020, 9, 2))  # printed for eyeballing
SPLIT_RATIO_BAND = (3.0, 5.0)             # accepted pre/post price ratio (~4 expected)


def check(label: str, passed: bool, observed: str) -> bool:
    """Print one PASS/FAIL line with the observed value; return passed."""
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {observed}")
    return passed


def main() -> int:
    for f in (RAW_FILE, PERMNO_FILE):
        if not f.exists():
            sys.exit(f"Required file not found: {f}")

    # strip all write bits; the raw file is read-only from here on
    RAW_FILE.chmod(RAW_FILE.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    # ---- load ----
    all_cols = [c.lower() for c in pd.read_csv(RAW_FILE, nrows=0).columns]
    df = pd.read_csv(
        RAW_FILE,
        usecols=lambda c: c.lower() in USECOLS,
        dtype={"PERMNO": "int64", "Ticker": "string"},
        parse_dates=["DlyCalDt"],
    )
    df.columns = df.columns.str.lower()

    print("=== Price file overview ===")
    print(f"  file           : {RAW_FILE}")
    print(f"  columns ({len(all_cols)})   : {all_cols}")
    print(f"  rows           : {len(df):,}")
    print(f"  dlycaldt range : {df['dlycaldt'].min().date()} .. {df['dlycaldt'].max().date()}")
    price_permnos = set(int(p) for p in df["permno"].unique())
    print(f"  unique permno  : {len(price_permnos)}")

    # ---- check: coverage vs membership universe (reported, not failed) ----
    universe = [int(line) for line in PERMNO_FILE.read_text().split()]
    covered = [p for p in universe if p in price_permnos]
    missing = [p for p in universe if p not in price_permnos]
    extra = sorted(price_permnos - set(universe))
    print("\n=== Coverage vs membership universe ===")
    print(f"  universe size            : {len(universe)}")
    print(f"  covered by price file    : {len(covered)}")
    print(f"  missing from price file  : {len(missing)}"
          + (f" -> {missing}" if missing else ""))
    print(f"  in price file but not in universe: {len(extra)}"
          + (f" -> {extra}" if extra else ""))

    # ---- spot-checks ----
    print("\n=== Spot-checks ===")
    all_pass = True

    # AAPL 4:1 split must show as a pre/post price ratio inside SPLIT_RATIO_BAND
    aapl = df.loc[df["permno"] == AAPL_PERMNO].sort_values("dlycaldt")
    if aapl.empty:
        all_pass &= check("AAPL present (permno 14593)", False, "no rows found")
    else:
        print(f"  AAPL: {aapl['dlycaldt'].min().date()} .. {aapl['dlycaldt'].max().date()}, "
              f"{len(aapl):,} rows, mean dlyprc {aapl['dlyprc'].mean():.2f}")
        window = aapl.loc[
            (aapl["dlycaldt"].dt.date >= SPLIT_WINDOW[0])
            & (aapl["dlycaldt"].dt.date <= SPLIT_WINDOW[1]),
            ["dlycaldt", "dlyprc", "dlyret"],
        ]
        print("  AAPL dlyprc around the 2020-08-31 4:1 split:")
        for _, r in window.iterrows():
            print(f"    {r['dlycaldt'].date()}  dlyprc {r['dlyprc']:>8.2f}  dlyret {r['dlyret']:+.4f}")
        pre = aapl.loc[aapl["dlycaldt"].dt.date == SPLIT_LAST_PRESPLIT, "dlyprc"]
        post = aapl.loc[aapl["dlycaldt"].dt.date == SPLIT_FIRST_POSTSPLIT, "dlyprc"]
        if pre.empty or post.empty:
            all_pass &= check("AAPL 4:1 split visible", False,
                              "pre- or post-split day missing from file")
        else:
            ratio = abs(float(pre.iloc[0])) / abs(float(post.iloc[0]))
            ok = SPLIT_RATIO_BAND[0] <= ratio <= SPLIT_RATIO_BAND[1]
            all_pass &= check("AAPL 4:1 split visible (~4x price drop)", ok,
                              f"pre {float(pre.iloc[0]):.2f} / post {float(post.iloc[0]):.2f}"
                              f" = ratio {ratio:.2f}")

    # |dlyret| > 1.0 is printed for review but does not fail the run
    ret = df["dlyret"]
    print(f"\n  dlyret: min {ret.min():+.4f}, max {ret.max():+.4f}, mean {ret.mean():+.6f}")
    extreme = df.loc[ret.abs() > 1.0, ["permno", "ticker", "dlycaldt", "dlyprc", "dlyret"]]
    if extreme.empty:
        all_pass &= check("no |dlyret| > 1.0", True, "0 rows")
    else:
        check("no |dlyret| > 1.0", False, f"{len(extreme)} row(s) - REVIEW, not fatal")
        print(extreme.to_string(index=False))

    # missingness (informational)
    print(f"\n  null dlyret : {ret.isna().sum():,} rows ({ret.isna().mean() * 100:.3f}%)")
    print(f"  null dlyprc : {df['dlyprc'].isna().sum():,} rows "
          f"({df['dlyprc'].isna().mean() * 100:.3f}%)")

    # CRSP stores bid/ask midpoints as negative prices; downstream code must take abs(dlyprc)
    n_neg = int((df["dlyprc"] < 0).sum())
    print(f"  negative dlyprc (bid/ask midpoint convention): {n_neg:,} rows "
          f"({n_neg / len(df) * 100:.4f}%)")
    n_zero = int((df["dlyprc"] == 0).sum())
    print(f"  dlyprc == 0 (price missing/placeholder)       : {n_zero:,} rows "
          f"({n_zero / len(df) * 100:.4f}%)")

    # ---- check: duplicates ----
    # identical duplicate rows (WRDS export artifact) only warn; conflicting rows for
    # the same (permno, dlycaldt) fail
    dup_pairs = int(df.duplicated(subset=["permno", "dlycaldt"]).sum())
    dup_identical = int(df.duplicated().sum())          # identical on all loaded columns
    dup_conflicting = dup_pairs - dup_identical
    if dup_pairs == 0:
        all_pass &= check("(permno, dlycaldt) unique", True, "0 duplicate pair(s)")
    else:
        all_pass &= check("no CONFLICTING (permno, dlycaldt) duplicates",
                          dup_conflicting == 0,
                          f"{dup_pairs} duplicate pair(s): {dup_identical} identical "
                          f"(WRDS export artifact, drop_duplicates() downstream), "
                          f"{dup_conflicting} conflicting")

    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED - inspect above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
