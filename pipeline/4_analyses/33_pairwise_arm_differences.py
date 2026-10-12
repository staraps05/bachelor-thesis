# %% 33_pairwise_arm_differences.py
# Purpose: Original pairwise differences among the sentiment arms behind Table C.1 (now reproduced by pipeline/5_tables/34).
# Inputs:  Data/processed/per_year_primary_r2.csv
# Outputs: prints only
# Notes:
# - No refit: within-seed R2 differences, mean and sd (ddof=1) over the 10 seeds.
# - Stops unless the CSV holds the full 10 seeds x 4 arms x 4 scopes grid (160 rows).
# - Across-seed dispersion of R2 itself, not the pairwise DM tests of a single run.

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV = REPO_ROOT / "Data" / "processed" / "per_year_primary_r2.csv"

SEEDS = list(range(42, 52))
ARMS = ["A0_structured", "A1_lm", "A2_finbert", "A3_fingpt"]
SCOPES = ["pooled", "2021", "2022", "2023"]
PAIRS = [("A1_lm", "A2_finbert"), ("A1_lm", "A3_fingpt"),
         ("A2_finbert", "A3_fingpt")]


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def main() -> int:
    print("=== STAGE 0: load frozen per-seed R2 (pipeline/30 artifact) ===")
    df = pd.read_csv(CSV, dtype={"scope": str})
    # Verification (in place): 160 rows
    if len(df) != 160:
        stop(f"{CSV.name}: {len(df)} rows, expected 160")
    # Verification (in place): seed set
    if sorted(df["seed"].unique()) != SEEDS:
        stop("seed set != 42..51")
    # Verification (in place): arm set
    if sorted(df["arm"].unique()) != sorted(ARMS):
        stop("arm set unexpected")
    # Verification (in place): scope set
    if sorted(df["scope"].unique()) != sorted(SCOPES):
        stop("scope set unexpected")
    piv = df.pivot(index=["seed", "scope"], columns="arm", values="r2_pct")
    # Verification (in place): pivot complete
    if piv.isna().any().any():
        stop("pivot has NaN - seed x scope x arm not complete")

    print("\n=== STAGE 1: pairwise differences, mean (sd ddof=1) across "
          "seeds, pp ===")
    print("  sign: positive = first-listed arm higher R2")
    print(f"  {'pair':<26}" + "".join(f"{s:>20}" for s in SCOPES))
    max_ratio, max_cell = 0.0, None
    for a, b in PAIRS:
        d = piv[a] - piv[b]
        cells = []
        for s in SCOPES:
            v = d.xs(s, level="scope")
    # Verification (in place): 10 seeds per cell (inside loop)
            if len(v) != 10:
                stop(f"{a}-{b} {s}: {len(v)} seeds, expected 10")
            m, sd = float(v.mean()), float(v.std(ddof=1))
            cells.append(f"{m:+.4f} ({sd:.4f})")
            if sd > 0 and abs(m) / sd > max_ratio:
                max_ratio, max_cell = abs(m) / sd, (a, b, s)
    # Verification (in place): |mean| vs sd flag (inside loop)
            if abs(m) > sd:
                print(f"  [FLAG] {a}-{b} {s}: |mean| {m:+.4f} EXCEEDS "
                      f"sd {sd:.4f}")
        print(f"  {a + ' - ' + b:<26}" + "".join(f"{c:>20}" for c in cells))

    # ---- Verification: consistency and reproduction checks ----
    print(f"  [PASS] 160 rows = 10 seeds x 4 arms x 4 scopes, complete grid")
    print("\n=== STAGE 2: verdict ===")
    print(f"  no cell has |mean| > across-seed sd; max |mean|/sd ratio "
          f"{max_ratio:.2f} at {max_cell[0]} - {max_cell[1]}, "
          f"{max_cell[2]}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
