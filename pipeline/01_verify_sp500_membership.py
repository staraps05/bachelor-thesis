# %% 01_verify_sp500_membership.py
# Purpose: Verify the CRSP S&P 500 daily membership export (TSLA inclusion, FB->META
#          switch, universe size) and write the sorted PERMNO universe for the price pull.
# Inputs:  Data/membership_file.csv
# Outputs: Data/processed/sp500_permnos.txt
# Notes:
# - Strips write bits of Data/membership_file.csv before reading (raw stays immutable).
# - PERMNO file is written even when a spot-check fails; exit status is then 1.
import stat
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# paths
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = REPO_ROOT / "Data" / "membership_file.csv"
PROCESSED_DIR = REPO_ROOT / "Data" / "processed"
OUT_FILE = PROCESSED_DIR / "sp500_permnos.txt"

# only the columns needed for verification; no price fields
USECOLS = ["permno", "permco", "hdrcusip", "ticker", "dlycaldt"]

# spot-check expectations
TSLA_JOIN = date(2020, 12, 21)   # Tesla's S&P 500 inclusion date
TSLA_TOL_DAYS = 5                # tolerance for weekend / effective-date quirks
FB_META_SWITCH = (2022, 5, 7)    # FB should end / META begin in 2022-05..2022-07
PERMNO_BAND = (700, 1000)        # expected band; outside PERMNO_HARD is a failure
PERMNO_HARD = (500, 1100)


def check(label: str, passed: bool, observed: str) -> bool:
    """Print one PASS/FAIL line with the observed value; return passed."""
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {observed}")
    return passed


def main() -> int:
    # Verification (in place): raw membership file exists before chmod and read
    if not RAW_FILE.exists():
        sys.exit(f"Raw membership file not found: {RAW_FILE}")

    # strip all write bits: the raw file is read-only from here on
    RAW_FILE.chmod(RAW_FILE.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    # load: header first for the full column list, then only the needed columns
    all_cols = [c.lower() for c in pd.read_csv(RAW_FILE, nrows=0).columns]
    df = pd.read_csv(
        RAW_FILE,
        usecols=lambda c: c.lower() in USECOLS,
        dtype={"PERMNO": "int64", "PERMCO": "int64", "HdrCUSIP": "string", "Ticker": "string"},
        parse_dates=["DlyCalDt"],
    )
    df.columns = df.columns.str.lower()

    all_pass = True

    # write the PERMNO universe
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    permnos = sorted(df["permno"].unique().tolist())
    OUT_FILE.write_text("\n".join(str(p) for p in permnos) + "\n")
    print("\n=== Output ===")
    print(f"  wrote {len(permnos)} PERMNOs -> {OUT_FILE}")

    # ---- Verification: consistency and reproduction checks ----
    print("=== Membership file overview ===")
    print(f"  file           : {RAW_FILE}")
    print(f"  columns ({len(all_cols)})   : {all_cols}")
    print(f"  rows           : {len(df):,}")
    print(f"  dlycaldt range : {df['dlycaldt'].min().date()} .. {df['dlycaldt'].max().date()}")
    n_permno = df["permno"].nunique()
    n_ticker = df["ticker"].nunique()
    print(f"  unique permno  : {n_permno}")
    print(f"  unique ticker  : {n_ticker}")

    # spot-checks
    print("\n=== Membership spot-checks ===")

    # (a) TSLA first member-day ~2020-12-21
    tsla = df.loc[df["ticker"] == "TSLA", "dlycaldt"]
    if tsla.empty:
        all_pass &= check("TSLA first appearance", False, "TSLA not found in file")
    else:
        first = tsla.min().date()
        ok = abs((first - TSLA_JOIN).days) <= TSLA_TOL_DAYS
        all_pass &= check("TSLA first appearance ~2020-12-21", ok, f"first member-day {first}")

    # (b) FB -> META switch ~2022-06 with an unchanged PERMNO
    fb = df.loc[df["ticker"] == "FB"]
    meta = df.loc[df["ticker"] == "META"]
    if fb.empty or meta.empty:
        all_pass &= check("FB->META switch", False,
                          f"FB rows={len(fb)}, META rows={len(meta)} (one is missing)")
    else:
        fb_end = fb["dlycaldt"].max().date()
        meta_start = meta["dlycaldt"].min().date()
        window = (date(2022, 5, 1), date(2022, 7, 31))          # ~2022-06 +/- one month
        ok_dates = (window[0] <= fb_end <= window[1]) and (window[0] <= meta_start <= window[1])
        all_pass &= check("FB ends / META begins ~2022-06", ok_dates,
                          f"FB last day {fb_end}, META first day {meta_start}")

        fb_permnos = set(int(p) for p in fb["permno"].unique())
        meta_permnos = set(int(p) for p in meta["permno"].unique())
        ok_permno = fb_permnos == meta_permnos and len(fb_permnos) == 1
        all_pass &= check("FB/META share one identical PERMNO", ok_permno,
                          f"FB permno(s) {sorted(fb_permnos)}, META permno(s) {sorted(meta_permnos)}")

    # (c) universe size band
    in_band = PERMNO_BAND[0] <= n_permno <= PERMNO_BAND[1]
    in_hard = PERMNO_HARD[0] <= n_permno <= PERMNO_HARD[1]
    band_msg = f"{n_permno} unique permnos (expected {PERMNO_BAND[0]}-{PERMNO_BAND[1]})"
    if in_band:
        all_pass &= check("Unique PERMNO count in expected band", True, band_msg)
    elif in_hard:
        # outside the soft band but inside hard limits: warn, don't fail the run
        check("Unique PERMNO count in expected band", False, band_msg + " - REVIEW, not fatal")
    else:
        all_pass &= check("Unique PERMNO count within hard limits (500-1100)", False, band_msg)

    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED - inspect above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
