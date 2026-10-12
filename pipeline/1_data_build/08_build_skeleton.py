# %% 08_build_skeleton.py
# Purpose: Skeleton: one (PERMNO, week_anchor) row per S&P 500 member-week, 2018-2023.
# Inputs:  Data/membership_file.csv, Data/crsp_daily_prices.csv, Data/processed/sp500_permnos.txt
# Outputs: Data/processed/skeleton.parquet
# Notes:
# - week_anchor_map() is the single week definition; 09, 10 and 13 load it via importlib.
# - MbrEndDt is treated as inclusive; ambiguous evidence in the daily member rows aborts.
# - Failed hard checks abort with STOP; failed soft checks print FAIL and exit 1.
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MEMBERSHIP_FILE = REPO_ROOT / "Data" / "membership_file.csv"
PRICES_FILE = REPO_ROOT / "Data" / "crsp_daily_prices.csv"
UNIVERSE_FILE = REPO_ROOT / "Data" / "processed" / "sp500_permnos.txt"
OUT_FILE = REPO_ROOT / "Data" / "processed" / "skeleton.parquet"

WINDOW_START = pd.Timestamp("2018-01-01")   # Monday of the first panel week
WINDOW_END = pd.Timestamp("2023-12-31")     # Sunday of the last panel week

SP500_INDNO = 1000500                        # CRSP index id for the S&P 500

# spot-check PERMNOs
TSLA_PERMNO = 93436   # S&P 500 entry 2020-12-21
AAPL_PERMNO = 14593   # member for the whole window


def check(label: str, passed: bool, observed: str) -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {observed}")
    return passed


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}\nNothing frozen. Fix/clarify and re-run.")


def week_anchor_map(trading_days: pd.DatetimeIndex) -> pd.Series:
    """Map each Mon-Sun week in the panel window to its anchor: the last trading day
    <= that week's Friday that lies within the week. Returns a Series (index = week
    Monday, value = anchor day); weeks containing no trading day are absent."""
    mondays = pd.date_range(WINDOW_START, WINDOW_END, freq="W-MON")
    if len(mondays) != 313 or mondays[0] != WINDOW_START:
        raise ValueError(f"expected 313 Mon-Sun weeks starting {WINDOW_START.date()}; "
                         f"got {len(mondays)} starting {mondays[0].date()}")
    out = {}
    for mon in mondays:
        fri = mon + pd.Timedelta(days=4)
        idx = trading_days.searchsorted(fri, side="right") - 1
        if idx >= 0 and trading_days[idx] >= mon:
            out[mon] = trading_days[idx]
    return pd.Series(out, name="week_anchor")


def main() -> int:
    all_pass = True

    # step 1: inspect the membership file
    print("=== STEP 1: membership file inspection ===")
    # Verification (in place): membership file exists before read
    if not MEMBERSHIP_FILE.exists():
        stop(f"missing input {MEMBERSHIP_FILE}")

    with open(MEMBERSHIP_FILE, "r") as fh:
        raw_lines = [next(fh).rstrip("\n") for _ in range(4)]

    # Verification (in place): required columns present in the raw header before pd.read_csv
    header = raw_lines[0].split(",")
    needed = {"INDNO", "PERMNO", "MbrStartDt", "MbrEndDt", "DlyCalDt"}
    if not needed.issubset(header):
        stop(f"expected columns {sorted(needed)} not all present in header: {header}")

    mem = pd.read_csv(
        MEMBERSHIP_FILE,
        usecols=["INDNO", "PERMNO", "MbrStartDt", "MbrEndDt", "DlyCalDt"],
        parse_dates=["MbrStartDt", "MbrEndDt", "DlyCalDt"],
    )

    # Verification (in place): single S&P 500 INDNO before building spells
    indnos = mem["INDNO"].unique()
    if not (len(indnos) == 1 and indnos[0] == SP500_INDNO):
        stop(f"expected a single INDNO {SP500_INDNO} (S&P 500); found {indnos!r}")
    print(f"  single INDNO {SP500_INDNO} (S&P 500) confirmed; {len(mem):,} daily member rows")

    # Verification (in place): no NaT MbrStartDt before building spells
    if mem["MbrStartDt"].isna().any():
        stop("NaT values in MbrStartDt - spell starts must be known")

    # the file is daily; the spell table is the unique (PERMNO, MbrStartDt, MbrEndDt)
    spells = (
        mem[["PERMNO", "MbrStartDt", "MbrEndDt"]]
        .drop_duplicates()
        .sort_values(["PERMNO", "MbrStartDt"])
        .reset_index(drop=True)
    )

    # Verification (in place): no overlapping spells within a PERMNO before step 4
    prev_end = spells.groupby("PERMNO")["MbrEndDt"].shift()
    overlaps = (spells["MbrStartDt"] <= prev_end).sum()
    if overlaps:
        stop(f"{overlaps} overlapping membership spells within a PERMNO - S1 ambiguous")
    print("  no overlapping spells within any PERMNO")

    # Verification (in place): MbrEndDt inclusivity evidence; gate fixing the membership rule used in step 4
    # End-date inclusivity: if MbrEndDt is the last member day, daily rows exist
    # on it (for spells ending in-window) and never after it. Securities that
    # stop trading shortly before index removal explain a minority without a row
    # on the end date.
    file_last_day = mem["DlyCalDt"].max()
    last_daily = mem.groupby(["PERMNO", "MbrStartDt", "MbrEndDt"], dropna=False)["DlyCalDt"].max()
    n_after = int((last_daily > last_daily.index.get_level_values("MbrEndDt")).sum())

    ended = spells[spells["MbrEndDt"] < file_last_day]
    daily_keys = pd.MultiIndex.from_frame(mem[["PERMNO", "DlyCalDt"]])
    on_end = pd.MultiIndex.from_frame(ended[["PERMNO", "MbrEndDt"]]).isin(daily_keys)
    frac_on_end = on_end.mean() if len(ended) else float("nan")

    print(f"  inclusivity evidence: {len(ended)} spells end before the last data day "
          f"({file_last_day.date()});")
    print(f"    - member daily row exists ON MbrEndDt for {on_end.sum()}/{len(ended)} "
          f"({frac_on_end:.0%}) of them")
    print(f"    - spells with member daily rows AFTER MbrEndDt: {n_after}")
    if n_after > 0 or not (len(ended) > 0 and frac_on_end >= 0.5):
        stop("end-date inclusivity is ambiguous on this evidence - do not guess")
    print("  => MbrEndDt is INCLUSIVE (last member day). Membership test: "
          "MbrStartDt <= anchor <= MbrEndDt")
    print("  => open-ended spells (NaT or end date beyond the data window, e.g. 2025-12-31)")
    print("     count as 'still a member' for every in-window anchor")

    # step 2: trading-day calendar from the CRSP daily price file
    print("\n=== STEP 2: trading-day calendar ===")
    # Verification (in place): price file exists before the calendar read
    if not PRICES_FILE.exists():
        stop(f"missing input {PRICES_FILE}")
    caldt = pd.read_csv(PRICES_FILE, usecols=["DlyCalDt"], parse_dates=["DlyCalDt"])
    trading_days = pd.DatetimeIndex(caldt["DlyCalDt"].unique()).sort_values()
    n_days = len(trading_days)
    # Verification (in place): trading-day count sanity band before week_anchor_map
    if not 2240 <= n_days <= 2290:
        stop(f"trading-day count {n_days} outside sanity band 2240-2290 (~2,265 expected)")

    # step 3: weekly anchors
    print("\n=== STEP 3: weekly anchors (C1) ===")
    anchor_map = week_anchor_map(trading_days)
    mondays = pd.date_range(WINDOW_START, WINDOW_END, freq="W-MON")
    skipped = mondays.difference(anchor_map.index)

    anchors = pd.DatetimeIndex(anchor_map.values)
    # Verification (in place): anchors unique and increasing before step 4
    if not (anchors.is_unique and anchors.is_monotonic_increasing):
        stop("anchors are not unique/increasing - week logic is broken")

    # step 4: skeleton rows, one block of weeks per membership spell
    print("\n=== STEP 4: skeleton rows (S1) ===")
    end_eff = spells["MbrEndDt"].fillna(pd.Timestamp.max)   # NaT end -> still a member
    parts = []
    for permno, start, end in zip(spells["PERMNO"], spells["MbrStartDt"], end_eff):
        lo = anchors.searchsorted(start, side="left")       # anchor >= start
        hi = anchors.searchsorted(end, side="right")        # anchor <= end (inclusive)
        if hi > lo:
            parts.append(pd.DataFrame({"PERMNO": permno, "week_anchor": anchors[lo:hi]}))
    skel = pd.concat(parts, ignore_index=True).sort_values(
        ["PERMNO", "week_anchor"]).reset_index(drop=True)
    skel["PERMNO"] = skel["PERMNO"].astype("int64")
    print(f"  built {len(skel):,} rows from {len(spells)} spells "
          f"({len(spells) - len(parts)} spells contain no in-window anchor)")

    # Verification (in place): no duplicated (PERMNO, week_anchor) pairs before the parquet write
    if skel.duplicated(["PERMNO", "week_anchor"]).any():
        stop("duplicated (PERMNO, week_anchor) pairs - membership spells inconsistent")

    # step 5: save
    print("\n=== STEP 5: save ===")
    out = skel.copy()
    out["week_anchor"] = out["week_anchor"].dt.date          # parquet date32, not timestamp
    out = out[["PERMNO", "week_anchor"]]
    out.to_parquet(OUT_FILE, engine="pyarrow", index=False)

    # ---- Verification: consistency and reproduction checks ----
    print("  header + first 3 rows (raw text):")
    for ln in raw_lines:
        print(f"    {ln}")
    n_nat_end = int(spells["MbrEndDt"].isna().sum())
    print(f"  membership spells: {len(spells)} across {spells['PERMNO'].nunique()} PERMNOs "
          f"({n_nat_end} open-ended NaT end dates)")
    n_years = (trading_days.max() - trading_days.min()).days / 365.25
    print(f"  first: {trading_days.min().date()}   last: {trading_days.max().date()}   "
          f"count: {n_days:,} (~{n_days / n_years:.0f}/year)")
    all_pass &= check("trading-day count ~2,265", True, f"{n_days:,} days")
    if len(skipped):
        print(f"  weeks with NO trading day (skipped): {[d.date() for d in skipped]}")
    else:
        print("  weeks with no trading day: none")

    print(f"  anchor weeks: {len(anchors)}   first: {anchors[0].date()}   "
          f"last: {anchors[-1].date()}")
    all_pass &= check("~313 anchor weeks", 310 <= len(anchors) <= 313, f"{len(anchors)}")

    non_fri = anchors[anchors.dayofweek != 4]
    print(f"  non-Friday anchors ({len(non_fri)}; expected: a handful of holiday Fridays):")
    for a in non_fri:
        print(f"    {a.date()} ({a.day_name()}) - week's Friday "
              f"{(a + pd.Timedelta(days=4 - a.dayofweek)).date()} was not a trading day")

    reread = pd.read_parquet(OUT_FILE)
    ok_file = list(reread.columns) == ["PERMNO", "week_anchor"] and len(reread) == len(skel)
    if not ok_file:
        stop(f"re-read mismatch: columns {list(reread.columns)}, rows {len(reread):,}")
    print(f"  wrote {OUT_FILE}")
    print(f"  re-read OK: {len(reread):,} rows, columns {list(reread.columns)}, "
          f"dtypes [{reread['PERMNO'].dtype}, date]")

    # step 6: validation report
    print("\n=== STEP 6: VALIDATION REPORT ===")

    n_rows = len(skel)
    print(f"\n  TOTAL ROWS (FROZEN INVARIANT): {n_rows:,}")

    # distinct PERMNOs vs the step-01 universe
    n_permnos = skel["PERMNO"].nunique()
    universe = sorted(int(x) for x in UNIVERSE_FILE.read_text().split())
    missing = sorted(set(universe) - set(skel["PERMNO"].unique()))
    print(f"\n  distinct PERMNOs: {n_permnos} (universe file: {len(universe)}; "
          f"missing from skeleton: {len(missing)})")
    all_pass &= check("distinct PERMNOs <= 719", n_permnos <= 719, f"{n_permnos}")
    if missing:
        print(f"  first {min(10, len(missing))} missing PERMNOs (of {len(missing)}):")
        for p in missing[:10]:
            sp = spells[spells["PERMNO"] == p]
            rng = "; ".join(f"{s.date()}..{('open' if pd.isna(e) else e.date())}"
                            for s, e in zip(sp["MbrStartDt"], sp["MbrEndDt"]))
            print(f"    {p}: spell(s) {rng}")
        # a missing PERMNO must have no anchor inside any of its spells
        bad = []
        for p in missing:
            sp = spells[spells["PERMNO"] == p]
            for s, e in zip(sp["MbrStartDt"], sp["MbrEndDt"].fillna(pd.Timestamp.max)):
                if anchors.searchsorted(e, "right") > anchors.searchsorted(s, "left"):
                    bad.append(p)
        if bad:
            stop(f"PERMNOs missing from skeleton DESPITE a spell covering an anchor: {bad[:10]}")
        print("  (verified: every missing PERMNO has no membership spell covering any "
              "2018-2023 anchor - i.e. exited before the window or dipped in between anchors)")

    # distinct weeks
    n_weeks = skel["week_anchor"].nunique()
    all_pass &= check("distinct weeks == step-3 anchor count",
                      n_weeks == len(anchors), f"{n_weeks} vs {len(anchors)}")
    if n_weeks != len(anchors):
        stop("some anchor weeks have zero members - membership data inconsistent")

    # members per week
    per_week = skel.groupby("week_anchor").size()
    print(f"\n  members per week: min {per_week.min()}, median {per_week.median():.0f}, "
          f"max {per_week.max()}")
    outside = per_week[(per_week < 490) | (per_week > 515)]
    if len(outside):
        print("  weeks OUTSIDE 490-515:")
        for wk, n in outside.items():
            print(f"    {wk.date()}: {n}")
    all_pass &= check("all weeks within 490-515 members", len(outside) == 0,
                      f"{len(outside)} weeks outside")

    # duplicates
    all_pass &= check("no duplicated (PERMNO, week_anchor) pairs", True,
                      "asserted at build + on the saved file")
    if reread.duplicated(["PERMNO", "week_anchor"]).any():
        stop("duplicates in the saved parquet file")

    # spot check 1: Tesla (entered mid-window)
    tsla = skel.loc[skel["PERMNO"] == TSLA_PERMNO, "week_anchor"].reset_index(drop=True)
    print(f"\n  spot check 1 - TSLA {TSLA_PERMNO} ({len(tsla)} weeks):")
    if tsla.empty:
        all_pass &= check("TSLA present", False, "no rows")
    else:
        f2 = [d.date() for d in tsla.head(2)]
        l2 = [d.date() for d in tsla.tail(2)]
        print(f"    first 2 weeks: {f2}    last 2 weeks: {l2}")
        ok_first = date(2020, 12, 21) <= f2[0] <= date(2020, 12, 31)
        all_pass &= check("TSLA first week in late Dec 2020 (entry 2020-12-21)",
                          ok_first, f"{f2[0]}")
        all_pass &= check("TSLA last week == final anchor of 2023",
                          l2[-1] == anchors[-1].date(), f"{l2[-1]}")

    # spot check 2: Apple (present the whole window)
    n_aapl = int((skel["PERMNO"] == AAPL_PERMNO).sum())
    print(f"\n  spot check 2 - AAPL {AAPL_PERMNO}:")
    all_pass &= check("AAPL row count == total number of weeks",
                      n_aapl == len(anchors), f"{n_aapl} vs {len(anchors)}")

    # spot check 3: a PERMNO with more than one membership spell
    multi = spells.groupby("PERMNO").size()
    multi = multi[multi > 1]
    print(f"\n  spot check 3 - re-entries (S3): {len(multi)} PERMNOs with >1 spell "
          f"{sorted(multi.index.tolist())}")
    if multi.empty:
        print("    none exist in the file - nothing to check")
    else:
        p = int(multi.index.min())
        print(f"    example PERMNO {p}:")
        n_blocks = 0
        for _, r in spells[spells["PERMNO"] == p].iterrows():
            e = pd.Timestamp.max if pd.isna(r["MbrEndDt"]) else r["MbrEndDt"]
            lo = anchors.searchsorted(r["MbrStartDt"], "left")
            hi = anchors.searchsorted(e, "right")
            wk = (f"{anchors[lo].date()} .. {anchors[hi - 1].date()} ({hi - lo} weeks)"
                  if hi > lo else "no in-window skeleton weeks")
            n_blocks += hi > lo
            end_str = "open" if pd.isna(r["MbrEndDt"]) else r["MbrEndDt"].date()
            print(f"      spell {r['MbrStartDt'].date()} .. {end_str}  ->  skeleton {wk}")
        # positions of this firm's weeks in the anchor sequence; a jump > 1 is a
        # between-spell gap
        pos = anchors.get_indexer(pd.DatetimeIndex(skel.loc[skel["PERMNO"] == p,
                                                            "week_anchor"]))
        breaks = int((pd.Series(pos).diff() > 1).sum())
        all_pass &= check(f"PERMNO {p} skeleton forms {n_blocks} contiguous blocks "
                          f"(one per in-window spell)",
                          breaks == n_blocks - 1,
                          f"{breaks} gap(s) between blocks, {n_blocks} spells with weeks")

    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED - inspect above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
