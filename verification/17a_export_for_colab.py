# %% 17a_export_for_colab.py
# Purpose: Confirm the Colab export finbert_input.csv is unchanged (SHA256 and row count) so the FinGPT notebook can reuse the FinBERT upload, and print the Drive instructions for Colab/17b.
# Inputs:  Data/colab/finbert_input.csv
# Outputs: prints only
# Notes:
# - Verification script: checks only; writes nothing to Data/
# - EXPORT_SHA and EXPORT_ROWS (796,038) recorded when 2_sentiment/16a wrote the file.
# - Exits with STOP on a missing file, SHA or row-count mismatch; then run colab/17b.

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_FILE = REPO_ROOT / "Data" / "colab" / "finbert_input.csv"
EXPORT_SHA = "e87ba39d4f43081736e03d9fa4a9e89df54658e3d08dea3e48d51ed01ebef346"
EXPORT_ROWS = 796_038


def main() -> int:
    if not EXPORT_FILE.exists():
        sys.exit(f"STOP: {EXPORT_FILE} missing - re-run pipeline/2_sentiment/16a_export_for_colab.py")
    sha = hashlib.sha256(EXPORT_FILE.read_bytes()).hexdigest()
    if sha != EXPORT_SHA:
        sys.exit(f"STOP: export SHA mismatch ({sha}) - re-run pipeline/2_sentiment/16a and "
                 f"re-verify the chain")
    n = sum(1 for _ in open(EXPORT_FILE)) - 1
    if n != EXPORT_ROWS:
        sys.exit(f"STOP: {n:,} data rows, expected {EXPORT_ROWS:,}")
    print(f"export verified: {EXPORT_FILE.name}")
    print(f"  SHA256 {sha} == frozen chain value")
    print(f"  {n:,} rows (hid 0..{EXPORT_ROWS - 1})")
    print("REUSE the existing Drive copy at MyDrive/thesis_finbert/"
          "finbert_input.csv - no re-upload.")
    print("Run Colab/17b_score_fingpt.ipynb (outputs go to MyDrive/"
          "thesis_fingpt/), download fingpt_scores.csv into Data/colab/, "
          "then run pipeline/2_sentiment/17c.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
