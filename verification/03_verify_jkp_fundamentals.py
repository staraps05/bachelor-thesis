# %% 03_verify_jkp_fundamentals.py
# Purpose: Verify the JKP monthly characteristics file for the S&P 500 universe: ID columns, end-of-month dates, PERMNO coverage, US-only rows, panel shape, missingness, (permno, eom) uniqueness.
# Inputs:  Data/Fundamentals.csv, Data/processed/sp500_permnos.txt
# Outputs: prints only
# Notes:
# - Verification script: checks only; writes nothing to Data/
# - Strips the write bits of the raw file before reading, so it stays immutable.
# - Exit status 1 if any check fails; universe PERMNOs absent from the file only reported.

import stat
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = REPO_ROOT / "Data" / "Fundamentals.csv"
PERMNO_FILE = REPO_ROOT / "Data" / "processed" / "sp500_permnos.txt"

REQUIRED_ID_COLS = ["id", "permno", "gvkey", "iid", "excntry", "eom"]

# identifier / price / return / classification columns; everything else is
# treated as a characteristic in the missingness scan
META_COLS = {
    "obs_main", "exch_main", "common", "primary_sec", "permno", "date",
    "permco", "gvkey", "iid", "id", "excntry", "eom", "size_grp",
    "me", "me_company", "ret_exc_lead1m", "ret_exc", "ret", "ret_local",
    "ret_lag_dif", "prc_local", "prc_high", "prc_low", "bidask", "curcd",
    "fx", "gics", "naics", "sic", "ff49", "dolvol", "shares", "tvol",
    "adjfct", "comp_tpci", "crsp_shrcd", "comp_exchg", "crsp_exchcd",
    "source_crsp", "prc",
}

N_SPARSEST = 15   # how many highest-NaN-share characteristics to list


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
    df = pd.read_csv(RAW_FILE, low_memory=False)
    df.columns = df.columns.str.lower()
    df["eom"] = pd.to_datetime(df["eom"])

    print("=== JKP fundamentals overview ===")
    print(f"  file      : {RAW_FILE}")
    print(f"  columns   : {df.shape[1]}")
    print(f"  rows      : {len(df):,}")

    all_pass = True
    missing_ids = [c for c in REQUIRED_ID_COLS if c not in df.columns]
    all_pass &= check("required ID columns present", not missing_ids,
                      "all of " + ", ".join(REQUIRED_ID_COLS) if not missing_ids
                      else f"missing: {missing_ids}")

    print(f"  eom range : {df['eom'].min().date()} .. {df['eom'].max().date()}")
    n_not_eom = int((~df["eom"].dt.is_month_end).sum())
    all_pass &= check("eom is end-of-month only", n_not_eom == 0,
                      f"{n_not_eom} intra-month date(s)")

    # ---- check: coverage vs membership universe (reported, not failed) ----
    universe = [int(line) for line in PERMNO_FILE.read_text().split()]
    jkp_permnos = set(int(p) for p in df["permno"].dropna().unique())
    covered = [p for p in universe if p in jkp_permnos]
    missing = [p for p in universe if p not in jkp_permnos]
    print("\n=== Coverage vs membership universe ===")
    print(f"  universe size          : {len(universe)}")
    print(f"  covered by JKP file    : {len(covered)}")
    print(f"  missing from JKP file  : {len(missing)}"
          + (f" -> {missing}" if missing else ""))

    # ---- check: country ----
    print("\n=== Country ===")
    vc = df["excntry"].value_counts(dropna=False)
    for country, n in vc.items():
        print(f"  {country}: {n:,}")
    all_pass &= check("excntry == USA for all rows", vc.index.tolist() == ["USA"],
                      f"{len(vc)} distinct value(s)")

    # ---- panel shape ----
    print("\n=== Panel shape ===")
    print(f"  unique permno : {len(jkp_permnos)}")
    print(f"  unique eom    : {df['eom'].nunique()}")
    print("  rows per year :")
    for year, n in df.groupby(df["eom"].dt.year).size().items():
        print(f"    {year}: {n:,}")

    # ---- characteristic missingness ----
    char_cols = [c for c in df.columns if c not in META_COLS]
    char = df[char_cols]
    overall_nan = char.isna().to_numpy().mean() * 100
    print("\n=== Characteristic missingness ===")
    print(f"  characteristic columns (non-meta): {len(char_cols)}")
    print(f"  overall NaN share                : {overall_nan:.2f}%")
    print(f"  {N_SPARSEST} sparsest characteristics:")
    for col, share in char.isna().mean().sort_values(ascending=False).head(N_SPARSEST).items():
        print(f"    {col:<28s} {share * 100:6.2f}% NaN")

    # ---- check: duplicates ----
    print("\n=== Duplicates ===")
    n_dups = int(df.duplicated(subset=["permno", "eom"]).sum())
    all_pass &= check("(permno, eom) unique", n_dups == 0, f"{n_dups} duplicate pair(s)")

    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED - inspect above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
