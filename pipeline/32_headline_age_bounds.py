# %% 32_headline_age_bounds.py
# Purpose: Bound the age of a headline at the time its target return is
#          realised under the single-lag construction, on actual calendar
#          dates, to check the chapter wording "one to twelve days".
# Inputs:  Data/processed/scoring_input.parquet
#          Data/processed/features_lm.parquet
#          Data/F-F_Research_Data_Factors_daily.csv
# Outputs: prints only
# Notes:
# - Headline window of key t+1 = Mon-Sun of the calendar week of anchor t;
#   the target return runs over the trading days after anchor t through
#   anchor t+1.
# - Headline age is in calendar days. gap_a = latest headline (Sunday) to the
#   first trading day after anchor t (minimum age); gap_b = earliest headline
#   (Monday) to anchor t+1 (maximum age); both over all 312 consecutive
#   anchor pairs.
# - The scoring-input keying (every headline inside its key's Mon-Sun window)
#   is re-verified first; the trading calendar is read from the FF daily file.
# - The script stops on any violation.
import bisect
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def main() -> int:
    si = pd.read_parquet(
        REPO_ROOT / "Data" / "processed" / "scoring_input.parquet",
        columns=["target_week", "headline_date"])
    si["target_week"] = pd.to_datetime(si["target_week"])
    si["headline_date"] = pd.to_datetime(si["headline_date"])
    lm = pd.read_parquet(
        REPO_ROOT / "Data" / "processed" / "features_lm.parquet",
        columns=["target_week"])
    anchors = sorted(pd.to_datetime(lm["target_week"]).unique())
    # Verification (in place): 313 anchors
    if len(anchors) != 313:
        stop(f"{len(anchors)} anchors, expected 313")
    wd = pd.Series([a.dayofweek for a in anchors]).value_counts()
    print(f"anchors: 313 ({wd.get(4, 0)} Fridays, {wd.get(3, 0)} Thursdays)")

    # headline window per key: Mon-Sun of the calendar week before the anchor's week
    win = {}
    for a in anchors:
        mon = a - pd.Timedelta(days=a.dayofweek)
        win[a] = (mon - pd.Timedelta(days=7), mon - pd.Timedelta(days=1))

    g = si.groupby("target_week")["headline_date"].agg(["min", "max"])
    # Verification (in place): headlines inside the Mon-Sun window
    bad = sum(1 for a, r in g.iterrows()
              if r["min"] < win[a][0] or r["max"] > win[a][1])
    if bad:
        stop(f"{bad} keys with headlines outside the Mon-Sun window")
    print(f"[PASS] all {len(si):,} headlines inside the Mon-Sun window "
          f"preceding their key's week ({len(g)} keys, 0 violations)")

    days = []
    ff = REPO_ROOT / "Data" / "F-F_Research_Data_Factors_daily.csv"
    for line in ff.read_text().splitlines():
        m = re.match(r"^\s*(\d{8})\s*,", line)
        if m:
            days.append(pd.Timestamp(m.group(1)))
    tset = sorted(set(days))
    print(f"trading calendar: {len(tset):,} days, {tset[0].date()} .. "
          f"{tset[-1].date()}")
    # Verification (in place): anchors are trading days
    if not all(a in set(tset) for a in anchors):
        stop("anchor not a trading day")

    rows = []
    for i in range(312):
        t, t1 = anchors[i], anchors[i + 1]
        w_lo, w_hi = win[t1]              # headline Mon..Sun (week of anchor t)
        tstart = tset[bisect.bisect_right(tset, t)]   # first day AFTER anchor t
        rows.append({"t1": t1, "win_mon": w_lo, "win_sun": w_hi,
                     "tgt_start": tstart,
                     "gap_a": (tstart - w_hi).days,
                     "gap_b": (t1 - w_lo).days})
    df = pd.DataFrame(rows)
    print("\ngap_a (Sun headline -> first trading day of target week):")
    print(df["gap_a"].value_counts().sort_index().to_string())
    print("\ngap_b (Mon headline -> anchor t+1 = target week end):")
    print(df["gap_b"].value_counts().sort_index().to_string())
    print(f"\nBOUNDS: min gap_a = {df['gap_a'].min()} day(s); "
          f"max gap_b = {df['gap_b'].max()} day(s) -> the correct chapter "
          f"wording is 'one to eleven days' ('one to twelve' is wrong under "
          f"any consistent counting: inclusive counting gives 2-12)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
