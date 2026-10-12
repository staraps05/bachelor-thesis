# %% 09_build_target.py
# Purpose: Weekly excess-return target ret_excess and n_days diagnostic per skeleton row.
# Inputs:  Data/processed/skeleton.parquet, Data/crsp_daily_prices.csv, Data/F-F_Research_Data_Factors_daily.csv, Data/membership_file.csv
# Outputs: Data/processed/panel_target.parquet
# Notes:
# - Loads pipeline/08_build_skeleton.py via importlib to reuse week_anchor_map().
# - Expects 157,901 skeleton rows; 146 byte-identical duplicate CRSP rows are dropped.
# - Rows with no valid return day are dropped; VW-vs-FF-market corr > 0.99 is pass/fail.
import importlib.util
import io
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SKELETON_FILE = REPO_ROOT / "Data" / "processed" / "skeleton.parquet"
PRICES_FILE = REPO_ROOT / "Data" / "crsp_daily_prices.csv"
FF_FILE = REPO_ROOT / "Data" / "F-F_Research_Data_Factors_daily.csv"
MEMBERSHIP_FILE = REPO_ROOT / "Data" / "membership_file.csv"
OUT_FILE = REPO_ROOT / "Data" / "processed" / "panel_target.parquet"

FROZEN_ROWS = 157_901          # expected skeleton row count
EXPECTED_DUP_ROWS = 146        # byte-identical CRSP rows
FF_DAILY_ROW = re.compile(r"^\s*\d{8}\s*,")

AAPL_PERMNO = 14593
MANUAL_WEEK = pd.Timestamp("2023-06-02")      # manual compounding spot check
RF_WEEK_2020 = pd.Timestamp("2020-06-05")     # rf_w sanity week (5 trading days)
RF_WEEK_2023 = pd.Timestamp("2023-06-09")     # rf_w sanity week (5 trading days)
COVID_WEEKS = [pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-20"),
               pd.Timestamp("2020-03-27")]

# reuse week_anchor_map from pipeline/08_build_skeleton.py
_spec = importlib.util.spec_from_file_location(
    "build_skeleton_08", Path(__file__).resolve().with_name("08_build_skeleton.py"))
_skel_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_skel_mod)                 # defs only; main() is guarded
week_anchor_map = _skel_mod.week_anchor_map


def check(label: str, passed: bool, observed: str) -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {observed}")
    return passed


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}\nNothing written should be trusted. Fix/clarify and re-run.")


def load_ff_daily(path: Path) -> pd.DataFrame:
    """Parse the Ken French daily file: text preamble, header row with an unnamed
    date column, a contiguous block of YYYYMMDD rows, text footer."""
    lines = path.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if FF_DAILY_ROW.match(ln))
    end = start
    while end < len(lines) and FF_DAILY_ROW.match(lines[end]):
        end += 1
    header = [f.strip() or "Date" for f in lines[start - 1].split(",")]
    ff = pd.read_csv(io.StringIO("\n".join(lines[start:end])), names=header)
    ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m%d")
    return ff


def main() -> int:
    all_pass = True

    # step 1: load inputs
    print("=== STEP 1: load inputs ===")
    skeleton = pd.read_parquet(SKELETON_FILE)
    skeleton["week_anchor"] = pd.to_datetime(skeleton["week_anchor"])
    # Verification (in place): skeleton row count equals the frozen invariant
    if len(skeleton) != FROZEN_ROWS:
        stop(f"skeleton has {len(skeleton):,} rows, frozen invariant is {FROZEN_ROWS:,}")
    print(f"  skeleton: {len(skeleton):,} rows (frozen invariant confirmed)")

    crsp = pd.read_csv(PRICES_FILE,
                       usecols=["PERMNO", "DlyCalDt", "DlyRet", "DlyCap"],
                       parse_dates=["DlyCalDt"])   # DlyCap: VW cross-check only
    print(f"  CRSP daily: {len(crsp):,} rows "
          f"({crsp['DlyCalDt'].min().date()} .. {crsp['DlyCalDt'].max().date()})")

    ff = load_ff_daily(FF_FILE)
    proof_date = pd.Timestamp("2023-06-01")
    rf_raw = float(ff.loc[ff["Date"] == proof_date, "RF"].iloc[0])
    ff["RF"] = ff["RF"] / 100.0            # percent -> decimal
    ff["Mkt-RF"] = ff["Mkt-RF"] / 100.0    # for the aggregate cross-check
    rf_scaled = float(ff.loc[ff["Date"] == proof_date, "RF"].iloc[0])

    # step 2: clean
    print("\n=== STEP 2: clean ===")
    n0 = len(crsp)
    crsp = crsp.drop_duplicates()          # identical rows only - never picks a value
    n_dropped_dup = n0 - len(crsp)

    # Verification (in place): no conflicting (PERMNO, DlyCalDt) duplicates remain
    conflict = crsp[crsp.duplicated(["PERMNO", "DlyCalDt"], keep=False)]
    if len(conflict):
        print(conflict.sort_values(["PERMNO", "DlyCalDt"]).head(20).to_string())
        stop(f"{len(conflict)} rows share (PERMNO, DlyCalDt) with CONFLICTING values")
    print("  no remaining duplicated (PERMNO, DlyCalDt) pairs "
          "(any conflict would have STOPped)")

    # step 3: week assignment (week_anchor_map from pipeline/08)
    print("\n=== STEP 3: week assignment (week_anchor_map imported from 08) ===")
    trading_days = pd.DatetimeIndex(crsp["DlyCalDt"].unique()).sort_values()
    amap = week_anchor_map(trading_days)                 # week Monday -> anchor
    anchors = pd.DatetimeIndex(amap.values)

    # Verification (in place): anchor set equals the skeleton's anchor set
    skel_anchors = pd.DatetimeIndex(skeleton["week_anchor"].unique()).sort_values()
    if not anchors.equals(skel_anchors):
        stop("anchor set from week_anchor_map does not equal the skeleton's anchor set")
    print(f"  {len(anchors)} anchors, identical to the skeleton's anchor set "
          f"({anchors[0].date()} .. {anchors[-1].date()})")

    day_mondays = trading_days - pd.to_timedelta(trading_days.dayofweek, unit="D")
    day_anchor = pd.Series(amap.reindex(day_mondays).to_numpy(), index=trading_days)
    crsp["week_anchor"] = crsp["DlyCalDt"].map(day_anchor)

    n_buffer = int(crsp["week_anchor"].isna().sum())
    crsp_w = crsp.dropna(subset=["week_anchor"]).copy()
    # Verification (in place): daily rows mapped vs excluded buffer rows
    print(f"  daily rows mapped to an anchor: {len(crsp_w):,}; excluded 2015-2017 "
          f"buffer rows (no anchor): {n_buffer:,}")
    # Verification (in place): every day maps inside its own Mon-Sun week
    if not ((crsp_w["DlyCalDt"] <= crsp_w["week_anchor"])
            & (crsp_w["DlyCalDt"] > crsp_w["week_anchor"] - pd.Timedelta(days=7))).all():
        stop("week assignment broken: a day maps outside its own Mon-Sun week")
    print(f"  mapped day range: {crsp_w['DlyCalDt'].min().date()} .. "
          f"{crsp_w['DlyCalDt'].max().date()}")

    # step 4: compute ret_w, rf_w, ret_excess, n_days
    print("\n=== STEP 4: weekly compounding ===")
    crsp_w = crsp_w.sort_values(["PERMNO", "DlyCalDt"])  # deterministic first()
    crsp_w["gross"] = 1.0 + crsp_w["DlyRet"]
    g = crsp_w.groupby(["PERMNO", "week_anchor"], sort=True)
    firm_week = pd.DataFrame({
        "ret_w": g["gross"].prod(min_count=1) - 1.0,     # all-NaN week -> NaN, not 0
        "n_days": g["DlyRet"].count(),                   # non-missing return days
        "wcap": g["DlyCap"].first(),   # first non-NaN cap of the week (VW cross-check
    }).reset_index()                   # only; never saved)
    # Verification (in place): firm-week count and all-NaN weeks
    print(f"  firm-weeks computed: {len(firm_week):,} "
          f"({firm_week['ret_w'].isna().sum()} with zero non-missing return days)")

    # weekly risk-free + market (for the cross-check) on the CRSP trading days
    cal = pd.DataFrame({"DlyCalDt": trading_days[trading_days >= anchors[0].normalize()
                                                 - pd.Timedelta(days=6)]})
    cal["week_anchor"] = cal["DlyCalDt"].map(day_anchor)
    cal = cal.dropna(subset=["week_anchor"])
    cal = cal.merge(ff[["Date", "RF", "Mkt-RF"]], left_on="DlyCalDt",
                    right_on="Date", how="left", validate="1:1")
    # Verification (in place): every CRSP trading day has an FF RF row
    if cal["RF"].isna().any():
        stop(f"{cal['RF'].isna().sum()} CRSP trading days have no FF RF row - "
             "calendar mismatch")
    # Verification (in place): FF has no in-window trading days CRSP lacks
    ff_extra = ff[(ff["Date"] >= cal["DlyCalDt"].min())
                  & (ff["Date"] <= cal["DlyCalDt"].max())
                  & ~ff["Date"].isin(set(cal["DlyCalDt"]))]
    if len(ff_extra):
        stop(f"FF has {len(ff_extra)} trading days CRSP lacks in-window: "
             f"{[d.date() for d in ff_extra['Date'].head(5)]}")
    print(f"  FF and CRSP trading calendars identical over the window "
          f"({len(cal):,} days)")

    gw = cal.groupby("week_anchor")
    week_ff = pd.DataFrame({
        "rf_w": gw["RF"].apply(lambda s: (1.0 + s).prod() - 1.0),
        "mkt_w": gw.apply(lambda d: (1.0 + d["Mkt-RF"] + d["RF"]).prod() - 1.0,
                          include_groups=False),
    }).reset_index()

    firm_week = firm_week.merge(week_ff[["week_anchor", "rf_w"]],
                                on="week_anchor", how="left", validate="m:1")
    # Verification (in place): every firm-week has an rf_w
    if firm_week["rf_w"].isna().any():
        stop("some firm-weeks have no rf_w - anchor mismatch")
    firm_week["ret_excess"] = firm_week["ret_w"] - firm_week["rf_w"]

    # step 5: merge onto the skeleton (left, 1:1), drop NaN targets, save
    print("\n=== STEP 5: merge onto skeleton + S4 + save ===")
    merged = skeleton.merge(firm_week, on=["PERMNO", "week_anchor"],
                            how="left", validate="1:1")
    # Verification (in place): merged row count equals the frozen invariant
    if len(merged) != FROZEN_ROWS:
        stop(f"merged frame has {len(merged):,} rows, expected {FROZEN_ROWS:,}")

    dropped = merged[merged["ret_excess"].isna()].copy()
    kept = merged.dropna(subset=["ret_excess"]).copy()
    kept["n_days"] = kept["n_days"].astype("int64")

    out = kept[["PERMNO", "week_anchor", "ret_excess", "n_days"]].copy()
    out["week_anchor"] = out["week_anchor"].dt.date
    out.to_parquet(OUT_FILE, engine="pyarrow", index=False)
    print(f"  wrote {OUT_FILE}")

    # ---- Verification: consistency and reproduction checks ----
    print(f"  FF daily: {len(ff):,} rows; ÷100 proof: RF({proof_date.date()}) "
          f"raw = {rf_raw} (percent) -> {rf_scaled:.6f} (decimal) "
          f"~= 5%/252 = {0.05 / 252:.6f}")
    all_pass &= check("scaled 2023 daily RF ~ 0.0002", 0.0001 <= rf_scaled <= 0.0003,
                      f"{rf_scaled:.6f}")

    all_pass &= check("duplicate CRSP rows dropped == 146 (byte-identical)",
                      n_dropped_dup == EXPECTED_DUP_ROWS, f"{n_dropped_dup} dropped")

    n_matched = int(merged["n_days"].notna().sum())
    print(f"  left-merged 1:1 onto skeleton: {len(merged):,} rows "
          f"(firm-weeks computed {len(firm_week):,}; "
          f"{len(firm_week) - n_matched:,} were non-member weeks, discarded by the join)")

    reread = pd.read_parquet(OUT_FILE)
    if list(reread.columns) != ["PERMNO", "week_anchor", "ret_excess", "n_days"] \
            or len(reread) != len(kept):
        stop(f"re-read mismatch: {list(reread.columns)}, {len(reread):,} rows")
    print(f"  re-read OK: {len(reread):,} rows, columns {list(reread.columns)}")

    # step 6: validation report
    print("\n=== STEP 6: VALIDATION REPORT ===")

    # (a) row accounting
    print(f"\n  rows before S4 drop : {len(merged):,} (frozen invariant)")
    print(f"  rows dropped by S4  : {len(dropped):,} "
          f"({len(dropped) / len(merged):.4%})")
    print(f"  rows after S4       : {len(kept):,}  <- NEW WORKING ROW COUNT")
    all_pass &= check("S4 drop is small (< 0.5%)",
                      len(dropped) / len(merged) < 0.005,
                      f"{len(dropped) / len(merged):.4%}")

    # (b) every dropped row: zero return days + why
    # independent recount from the cleaned daily frame, not the groupby table
    nonmiss = crsp_w[crsp_w["DlyRet"].notna()].groupby(["PERMNO", "week_anchor"]).size()
    drop_counts = nonmiss.reindex(
        pd.MultiIndex.from_frame(dropped[["PERMNO", "week_anchor"]]), fill_value=0)
    if (drop_counts > 0).any():
        bad = drop_counts[drop_counts > 0]
        stop(f"BUG: {len(bad)} dropped rows DO have non-missing return data:\n"
             f"{bad.head(10).to_string()}")
    all_pass &= check("all dropped rows verified to have zero non-missing return days",
                      True, f"{len(dropped)} rows recounted from daily data")

    span = crsp.groupby("PERMNO")["DlyCalDt"].agg(["min", "max"])
    print(f"  dropped rows ({'all ' if len(dropped) <= 20 else 'first 20 of '}"
          f"{len(dropped)}):")
    for _, r in dropped.head(20).iterrows():
        p, a = int(r["PERMNO"]), r["week_anchor"]
        week_mon = a - pd.Timedelta(days=int(a.dayofweek))
        n_rows_wk = int(((crsp_w["PERMNO"] == p) & (crsp_w["week_anchor"] == a)).sum())
        first_d, last_d = span.loc[p, "min"], span.loc[p, "max"]
        if first_d > a:
            why = f"member before CRSP data starts ({first_d.date()}) - entry/spin-off week"
        elif first_d >= week_mon:
            why = f"firm's first trading week (CRSP starts {first_d.date()}; " \
                  f"day-1 return undefined)"
        elif last_d < week_mon:
            why = f"CRSP data ends {last_d.date()} - membership outlasts price data"
        else:
            why = "mid-history week with zero return days - INSPECT"
        print(f"    PERMNO {p:>6}  week {a.date()}  crsp_rows={n_rows_wk} "
              f"nonmissing=0  -> {why}")

    # (c) n_days distribution
    print("\n  n_days value counts (kept rows):")
    for v, c in kept["n_days"].value_counts().sort_index(ascending=False).items():
        print(f"    {v} days: {c:>7,} rows ({c / len(kept):.2%})")
    low = kept[kept["n_days"] <= 2].sort_values(["week_anchor", "PERMNO"])
    print(f"  rows with n_days <= 2: {len(low)}")
    for _, r in low.head(40).iterrows():
        print(f"    PERMNO {int(r['PERMNO']):>6}  week {r['week_anchor'].date()}  "
              f"n_days={int(r['n_days'])}  ret_excess={r['ret_excess']:+.4f}")
    if len(low) > 40:
        print(f"    ... and {len(low) - 40} more")

    # (d) ret_excess distribution
    d = kept["ret_excess"]
    q = d.quantile([0.001, 0.999])
    print(f"\n  ret_excess: mean {d.mean():+.5f}  std {d.std():.5f}  "
          f"median {d.median():+.5f}")
    print(f"              min {d.min():+.4f}  max {d.max():+.4f}  "
          f"q0.1% {q[0.001]:+.4f}  q99.9% {q[0.999]:+.4f}")
    all_pass &= check("weekly std ~ 4-6%", 0.03 <= d.std() <= 0.07, f"{d.std():.4%}")

    tick = (pd.read_csv(MEMBERSHIP_FILE, usecols=["PERMNO", "Ticker", "DlyCalDt"],
                        parse_dates=["DlyCalDt"])
            .sort_values("DlyCalDt").groupby("PERMNO")["Ticker"].last())
    ext = kept.reindex(d.abs().sort_values(ascending=False).index[:5])
    print("  5 most extreme ret_excess values (ticker = last observed):")
    for _, r in ext.iterrows():
        print(f"    {tick.get(int(r['PERMNO']), '?'):>5} (PERMNO {int(r['PERMNO'])})  "
              f"week {r['week_anchor'].date()}  ret_excess {r['ret_excess']:+.4f}  "
              f"n_days={int(r['n_days'])}")

    # (e) rf_w sanity
    rf2020 = float(week_ff.loc[week_ff["week_anchor"] == RF_WEEK_2020, "rf_w"].iloc[0])
    rf2023 = float(week_ff.loc[week_ff["week_anchor"] == RF_WEEK_2023, "rf_w"].iloc[0])
    nd2020 = int((cal["week_anchor"] == RF_WEEK_2020).sum())
    nd2023 = int((cal["week_anchor"] == RF_WEEK_2023).sum())
    print(f"\n  rf_w({RF_WEEK_2020.date()}) = {rf2020:.6f} ({nd2020} trading days)   "
          f"rf_w({RF_WEEK_2023.date()}) = {rf2023:.6f} ({nd2023} trading days)")
    all_pass &= check("2020 weekly RF ~ 0.000-0.0002", 0.0 <= rf2020 <= 0.0002,
                      f"{rf2020:.6f}")
    all_pass &= check("2023 weekly RF ~ 0.0009-0.0011 (~5%/52)",
                      0.0009 <= rf2023 <= 0.0011, f"{rf2023:.6f}")

    # (f) aggregate cross-check vs FF market
    ew = merged.groupby("week_anchor")["ret_w"].mean()
    ffm = week_ff.set_index("week_anchor")["mkt_w"]
    both = pd.concat([ew.rename("ew_panel"), ffm.rename("ff_mkt")], axis=1).dropna()
    corr = both["ew_panel"].corr(both["ff_mkt"])
    print(f"\n  aggregate cross-check over {len(both)} weeks: corr(EW panel mean ret_w, "
          f"FF market compounded) = {corr:.4f}")
    # the EW correlation is informational only; the VW check below is pass/fail
    gaps = (both["ew_panel"] - both["ff_mkt"]).abs().sort_values(ascending=False)
    print("  3 weeks with the largest EW-vs-FF gap (equal- vs cap-weighting):")
    for wk in gaps.index[:3]:
        print(f"    {wk.date()}: EW {both.loc[wk, 'ew_panel']:+.4f} "
              f"vs FF {both.loc[wk, 'ff_mkt']:+.4f} "
              f"(gap {gaps[wk]:.4f})")

    # per-year EW correlation, then the cap-weighted panel mean, which is the
    # like-for-like comparison with FF's cap-weighted market
    print("  correlation by year (EW panel vs FF market):")
    for y in sorted(both.index.year.unique()):
        sub = both[both.index.year == y]
        print(f"    {y}: {sub['ew_panel'].corr(sub['ff_mkt']):+.4f}  ({len(sub)} weeks)")
    m_vw = merged[merged["ret_w"].notna() & merged["wcap"].notna()
                  & (merged["wcap"] > 0)]
    vw = (m_vw["ret_w"] * m_vw["wcap"]).groupby(m_vw["week_anchor"]).sum() \
        / m_vw.groupby("week_anchor")["wcap"].sum()
    both_vw = pd.concat([vw.rename("vw_panel"), ffm.rename("ff_mkt")], axis=1).dropna()
    corr_vw = both_vw["vw_panel"].corr(both_vw["ff_mkt"])
    all_pass &= check("VW panel mean tracks FF market (> 0.99, like-for-like)",
                      corr_vw > 0.99, f"{corr_vw:.4f} over {len(both_vw)} weeks")

    # (g) COVID spot check
    print("\n  COVID spot check (cross-sectional median ret_excess):")
    signs = [-1, -1, +1]
    for wk, sgn in zip(COVID_WEEKS, signs):
        med = kept.loc[kept["week_anchor"] == wk, "ret_excess"].median()
        lbl = "strongly negative" if sgn < 0 else "strongly positive"
        all_pass &= check(f"{wk.date()} median {lbl}",
                          (med < -0.03) if sgn < 0 else (med > 0.03), f"{med:+.4f}")

    # (h) manual compounding spot check: AAPL, week anchored 2023-06-02
    print(f"\n  manual spot check - AAPL {AAPL_PERMNO}, week anchored "
          f"{MANUAL_WEEK.date()}:")
    wk_rows = crsp_w[(crsp_w["PERMNO"] == AAPL_PERMNO)
                     & (crsp_w["week_anchor"] == MANUAL_WEEK)].sort_values("DlyCalDt")
    running = 1.0
    for _, r in wk_rows.iterrows():
        running *= 1.0 + r["DlyRet"]
        print(f"    {r['DlyCalDt'].date()} ({r['DlyCalDt'].day_name()[:3]})  "
              f"DlyRet {r['DlyRet']:+.6f}  cumprod {running:.6f}")
    ret_w_manual = running - 1.0
    rf_w_val = float(week_ff.loc[week_ff["week_anchor"] == MANUAL_WEEK, "rf_w"].iloc[0])
    print(f"    ret_w = {running:.6f} - 1 = {ret_w_manual:+.6f}   rf_w = {rf_w_val:.6f}"
          f"   ret_excess = {ret_w_manual - rf_w_val:+.6f}")
    stored = float(kept.loc[(kept["PERMNO"] == AAPL_PERMNO)
                            & (kept["week_anchor"] == MANUAL_WEEK), "ret_excess"].iloc[0])
    all_pass &= check("manual compounding == stored ret_excess",
                      abs(stored - (ret_w_manual - rf_w_val)) < 1e-12,
                      f"stored {stored:+.6f}")

    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED - inspect above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
