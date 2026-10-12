# %% 44_numbers_nonews_share.py
# Purpose: Compute the share of firm-weeks without news (had_news == 0), the
#          44 per cent quoted in the text, per year and pooled over 2021-23,
#          on the full panel and on the single-lag test rows.
# Inputs:  Data/processed/features_lm.parquet (full panel, had_news column)
#          Data/processed/model_frame_structured.parquet (via merge_frame)
#          Data/processed/features_finbert.parquet
#          Data/processed/features_fingpt.parquet
# Outputs: Data/processed/numbers_check/nonews_share.csv
# Notes:
# - Loads pipeline/27_covered_subsample.py via importlib for single_lag_frame
#   and SENT_FILES.
# - Frame (i): the full 157,897-row panel from features_lm.parquet, had_news
#   keyed at target_week; shares per year 2018-2023, pooled 2021-2023 and
#   over all years.
# - Frame (ii): the single-lag frame's test rows (week_anchor years 2021-2023,
#   78,056 rows), i.e. the rows the four arms are evaluated on; had_news is
#   re-attached from the week-(t+1) key there.
# - Both row counts are asserted; the run stops on a mismatch.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC = REPO_ROOT / "Data" / "processed"
OUT = PROC / "numbers_check"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c27 = _load("cov27", "27_covered_subsample.py")
single_lag_frame = c27.single_lag_frame
SENT_FILES = c27.SENT_FILES


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== 44: no-news share ===")
    rows = []

    # (i) full panel, had_news keyed at target_week
    full = pd.read_parquet(SENT_FILES["lm"][0], columns=["PERMNO", "target_week", "had_news"])
    # Verification (in place): full panel row count
    assert len(full) == 157_897, f"full panel {len(full)} != 157,897"
    full["year"] = pd.to_datetime(full["target_week"]).dt.year
    full["no_news"] = (full["had_news"] == 0).astype(int)
    for y in range(2018, 2024):
        sub = full[full["year"] == y]
        rows.append({"frame": "full_panel_157897", "scope": str(y), "n_rows": len(sub),
                     "n_no_news": int(sub["no_news"].sum()),
                     "no_news_share": float(sub["no_news"].mean())})
    pooled = full[full["year"].isin([2021, 2022, 2023])]
    rows.append({"frame": "full_panel_157897", "scope": "2021-2023", "n_rows": len(pooled),
                 "n_no_news": int(pooled["no_news"].sum()),
                 "no_news_share": float(pooled["no_news"].mean())})
    rows.append({"frame": "full_panel_157897", "scope": "all_2018-2023", "n_rows": len(full),
                 "n_no_news": int(full["no_news"].sum()),
                 "no_news_share": float(full["no_news"].mean())})

    # (ii) single-lag test rows, 2021-2023
    df = single_lag_frame()
    df = df.copy()
    df["year"] = pd.to_datetime(df["week_anchor"]).dt.year
    df["no_news"] = (df["had_news"] == 0).astype(int)
    test = df[df["year"].isin([2021, 2022, 2023])]
    # Verification (in place): single-lag test row count
    assert len(test) == 78_056, f"test rows {len(test)} != 78,056"
    for y in (2021, 2022, 2023):
        sub = test[test["year"] == y]
        rows.append({"frame": "singlelag_test_rows", "scope": str(y), "n_rows": len(sub),
                     "n_no_news": int(sub["no_news"].sum()),
                     "no_news_share": float(sub["no_news"].mean())})
    rows.append({"frame": "singlelag_test_rows", "scope": "2021-2023", "n_rows": len(test),
                 "n_no_news": int(test["no_news"].sum()),
                 "no_news_share": float(test["no_news"].mean())})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "nonews_share.csv", index=False)
    print(out.to_string(index=False))
    print(f"  wrote {OUT / 'nonews_share.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
