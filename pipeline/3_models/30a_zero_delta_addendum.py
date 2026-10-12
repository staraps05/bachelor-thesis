# %% 30a_zero_delta_addendum.py
# Purpose: Count how many of the 60 fold-2/3 cells (10 seeds x 3 sentiment arms x 2022/2023) have a
#          delta vs A0 that is exactly 0.0 in float64, and list the non-zero ones in full precision.
# Inputs:  Data/processed/per_year_primary_r2.csv
# Outputs: prints only
# Notes:
# - Reads the derived CSV written by pipeline/3_models/30 (160 rows expected).
# - A zero delta means the year's R2 is bit-equal, not that predictions are identical.

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV = REPO_ROOT / "Data" / "processed" / "per_year_primary_r2.csv"

SENT_ARMS = ("A1_lm", "A2_finbert", "A3_fingpt")
YEARS_23 = ("2022", "2023")   # folds 2-3


def main() -> int:
    df = pd.read_csv(CSV)
    # Verification (in place): 160 rows
    if len(df) != 160:
        sys.exit(f"STOP: {CSV.name} has {len(df)} rows, expected 160")
    sub = df[df["scope"].isin(YEARS_23) & df["arm"].isin(SENT_ARMS)]
    # Verification (in place): 60 fold-2/3 sentiment cells
    if len(sub) != 60:
        sys.exit(f"STOP: fold-2/3 sentiment cells = {len(sub)}, expected 60")

    print("=== ADDENDUM: folds 2-3 (2022/2023) zero-delta census, "
          "FULL float precision ===")
    print(f"  source: {CSV.relative_to(REPO_ROOT)} (DERIVED, pipeline/30); "
          f"deltas are float64 as computed in pipeline/30, no rounding")
    nz_all = []
    for arm in SENT_ARMS:
        for y in YEARS_23:
            cells = sub[(sub["arm"] == arm) & (sub["scope"] == y)]
            z = cells[cells["delta_vs_a0_pp"] == 0.0]
            nz = cells[cells["delta_vs_a0_pp"] != 0.0]
            print(f"  {arm:<12} {y}: exact-zero {len(z):2d}/10"
                  + ("" if nz.empty else "  non-zero: " + ", ".join(
                      f"seed {int(r.seed)} {r.delta_vs_a0_pp:+.6e}pp"
                      for r in nz.itertuples())))
            nz_all.append((arm, y, nz))
    n_zero = 60 - sum(len(nz) for _, _, nz in nz_all)
    print(f"\n  TOTAL: {n_zero}/60 fold-2/3 seed x arm cells have delta "
          f"EXACTLY 0.0 in float64")
    print("  non-zero cells in full:")
    any_nz = False
    for arm, y, nz in nz_all:
        for r in nz.itertuples():
            any_nz = True
            print(f"    {arm} {y} seed {int(r.seed)}: "
                  f"delta {r.delta_vs_a0_pp:+.17e} pp "
                  f"(R2 {r.r2_pct:+.10f}% vs A0 "
                  f"{r.r2_pct - r.delta_vs_a0_pp:+.10f}%)")
    if not any_nz:
        print("    (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
