# %% 16a_export_for_colab.py
# Purpose: Export the scoring input as a CSV for the Colab FinBERT notebook,
#          adding a positional row id `hid` (0..796,037) that
#          pipeline/16c_finbert_aggregate_freeze.py uses to join the scores back.
# Inputs:  Data/processed/scoring_input.parquet
# Outputs: Data/colab/finbert_input.csv
# Notes:
# - hid is the row position in the parquet (sorted by PERMNO, target_week,
#   headline_date); 16c rebuilds it the same way, so the parquet must not change.
# - The scoring-input SHA256 and row count are checked against hard-coded
#   constants before export.
# - Stops if any headline contains a newline (unsafe for CSV transport).
# - Next: upload the CSV to Drive, run Colab/16b_score_finbert.ipynb, download
#   finbert_scores.csv into Data/colab/, then run 16c.

import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORING_FILE = REPO_ROOT / "Data" / "processed" / "scoring_input.parquet"
OUT_DIR = REPO_ROOT / "Data" / "colab"
OUT_FILE = OUT_DIR / "finbert_input.csv"

SCORING_ROWS = 796_038
SCORING_SHA = "eb42e0e3aa05c7c973e407c722125468bb506f1cc51e629042ab4d0a027ccc39"


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def main() -> int:
    sha_in = hashlib.sha256(SCORING_FILE.read_bytes()).hexdigest()
    # Verification (in place): scoring-input SHA256
    if sha_in != SCORING_SHA:
        stop(f"scoring_input.parquet SHA mismatch: {sha_in}")
    df = pd.read_parquet(SCORING_FILE,
                         columns=["PERMNO", "target_week", "headline"])
    # Verification (in place): row count
    if len(df) != SCORING_ROWS:
        stop(f"{len(df):,} rows, expected {SCORING_ROWS:,}")

    out = df.reset_index(drop=True).reset_index(names="hid")
    # Verification (in place): hid is the complete unique range
    if not (out["hid"].is_unique and out["hid"].min() == 0
            and out["hid"].max() == SCORING_ROWS - 1):
        stop("hid is not the complete unique range 0..796,037")
    # Verification (in place): no embedded newline
    if out["headline"].str.contains("\n", regex=False).any():
        stop("embedded newline in a headline - CSV transport unsafe")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out[["hid", "PERMNO", "target_week", "headline"]].to_csv(OUT_FILE,
                                                             index=False)
    sha_out = hashlib.sha256(OUT_FILE.read_bytes()).hexdigest()
    print(f"wrote {OUT_FILE}")
    print(f"rows: {len(out):,} (hid 0..{len(out) - 1}, unique, complete)")
    print(f"CSV SHA256: {sha_out}")

    # ---- Verification: consistency and reproduction checks ----
    print(f"scoring input SHA verified: {SCORING_SHA[:8]}... OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
