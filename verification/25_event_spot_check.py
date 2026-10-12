# %% 25_event_spot_check.py
# Purpose: Event-level spot check of actual panel rows against real-world events:
#          target moments and units, named-event returns, and headline-to-
#          firm-week attachment read by hand. Read-only: writes nothing.
# Inputs:  Data/processed/panel_final.parquet
#          Data/processed/scoring_input.parquet
#          Data/processed/features_lm.parquet
#          Data/processed/features_finbert.parquet
#          Data/processed/features_fingpt.parquet
#          Data/membership_file.csv
#          Data/F-F_Research_Data_Factors_daily.csv
# Outputs: prints only
# Notes:
# - Loads pipeline/08_build_skeleton.py via importlib for week_anchor_map (the
#   week definition) and pipeline/19_four_arm_comparison.py for SENT_FILES
#   (feature paths + SHA256s). PANEL_SHA and SCORING_SHA are the values
#   recorded by pipeline/17 and pipeline/13.
# - Part A: weekly ret_excess moments (annualised mean 5-15%, vol 20-35%),
#   weekly RF recompounded from the FF daily file (must be decimal), yearly
#   signs (2018 <= 0, 2019 > 0, 2022 < 0, 2023 > 0; 2020 most volatile).
# - Part B event windows: META week of 2022-02-03 (-30..-15%), SIVB week of
#   2023-03-08 (<= -40%), NFLX week of 2022-04-20 (-45..-20%); cross-sectional
#   mean < -8% in the week of 2020-03-16 and > +5% in the week of 2020-04-06.
# - Part C: every scored headline must lie in the Mon-Sun window preceding its
#   target_week, so an event's coverage week is the week after it. META (absent
#   from FNSPID), SIVB (coverage ends 2020-06-09) and NFLX (scrape horizon
#   ~June 2020) are expected zero-filled with had_news=0, not flagged; the
#   COVID-crash coverage week must score negative on FinBERT and FinGPT.
# - Random attachment sample: 2 firm-weeks per year with >= 6 headlines,
#   distinct firms, seed 42.
# Verification script: checks only; writes nothing to Data/

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC = REPO_ROOT / "Data" / "processed"

PANEL_FILE = PROC / "panel_final.parquet"
PANEL_SHA = "eb3ed4cdd461b9772db1a4e5240b0f46a5f21ca3c9948a094b5e654e5fed19f2"
SCORING_FILE = PROC / "scoring_input.parquet"
SCORING_SHA = "eb42e0e3aa05c7c973e407c722125468bb506f1cc51e629042ab4d0a027ccc39"
FF_FILE = REPO_ROOT / "Data" / "F-F_Research_Data_Factors_daily.csv"
MEMBER_FILE = REPO_ROOT / "Data" / "membership_file.csv"

_s08 = importlib.util.spec_from_file_location(
    "skel08", REPO_ROOT / "pipeline" / "08_build_skeleton.py")
_m08 = importlib.util.module_from_spec(_s08)
_s08.loader.exec_module(_m08)
week_anchor_map = _m08.week_anchor_map          # the week definition

_s19 = importlib.util.spec_from_file_location(
    "cmp19", REPO_ROOT / "pipeline" / "19_four_arm_comparison.py")
_m19 = importlib.util.module_from_spec(_s19)
_s19.loader.exec_module(_m19)
SENT_FILES = _m19.SENT_FILES                    # feature paths + SHA256s

SEED = 42
FLAGS: list[str] = []


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def sha_ok(path: Path, want: str, name: str) -> None:
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        stop(f"{name} SHA mismatch: {got}")
    print(f"  [PASS] {name}: SHA == frozen ({want[:8]}...)")


def flag(msg: str) -> None:
    FLAGS.append(msg)
    print(f"  [FLAG] {msg}")


def main() -> int:
    print("=== SETUP: load frozen artefacts (read-only) ===")
    sha_ok(PANEL_FILE, PANEL_SHA, "panel_final.parquet")
    sha_ok(SCORING_FILE, SCORING_SHA, "scoring_input.parquet")
    panel = pd.read_parquet(
        PANEL_FILE, columns=["PERMNO", "week_anchor", "ret_excess"])
    si = pd.read_parquet(SCORING_FILE)
    feats = {}
    for arm, (path, want) in SENT_FILES.items():
        sha_ok(path, want, path.name)
        feats[arm] = pd.read_parquet(path)
    mem = pd.read_csv(MEMBER_FILE, usecols=["PERMNO", "Ticker"])
    tick_of = mem.drop_duplicates("PERMNO", keep="last") \
        .set_index("PERMNO")["Ticker"].to_dict()

    wa = pd.to_datetime(panel["week_anchor"])
    panel = panel.assign(_wa=wa, _yr=wa.dt.year)
    anchors = pd.DatetimeIndex(sorted(wa.unique()))
    mon_of = {a: a - pd.Timedelta(days=a.weekday()) for a in anchors}
    anchor_by_mon = {m: a for a, m in mon_of.items()}

    def anchor_containing(date_str: str) -> pd.Timestamp:
        d = pd.Timestamp(date_str)
        return anchor_by_mon[d - pd.Timedelta(days=d.weekday())]

    def permno_of_ticker(tk: str) -> int:
        cands = set(mem.loc[mem["Ticker"] == tk, "PERMNO"].unique()) \
            & set(panel["PERMNO"].unique())
        if len(cands) != 1:
            stop(f"ticker {tk} resolves to {sorted(cands)} - not unique")
        return int(cands.pop())

    # ---- part A: target sanity ----
    print("\n=== PART A: TARGET SANITY (ret_excess, full 157,897-row panel) ===")
    r = panel["ret_excess"]
    ann_m, ann_v = r.mean() * 52, r.std() * np.sqrt(52)
    print(f"  A1 weekly: mean {r.mean():+.6f} sd {r.std():.4f} "
          f"min {r.min():+.4f} max {r.max():+.4f} "
          f"p1 {r.quantile(0.01):+.4f} p99 {r.quantile(0.99):+.4f}")
    print(f"     annualised: mean {ann_m * 100:+.2f}% (band 5-15%), "
          f"vol {ann_v * 100:.2f}% (band 20-35%)")
    if not (0.05 <= ann_m <= 0.15):
        flag(f"A1 annualised mean {ann_m * 100:+.2f}% outside 5-15% band")
    if not (0.20 <= ann_v <= 0.35):
        flag(f"A1 annualised vol {ann_v * 100:.2f}% outside 20-35% band")

    ff = pd.read_csv(FF_FILE, skiprows=4)
    ff = ff.rename(columns={ff.columns[0]: "Date"})
    ff["Date"] = pd.to_numeric(ff["Date"], errors="coerce")
    ff = ff.dropna(subset=["Date"])
    ff["Date"] = pd.to_datetime(ff["Date"].astype(int).astype(str))
    ff = ff[(ff["Date"] >= "2018-01-01") & (ff["Date"] <= "2023-12-31")]
    ff["RF"] = pd.to_numeric(ff["RF"]) / 100.0        # percent -> decimal
    wmap = week_anchor_map(pd.DatetimeIndex(ff["Date"]))
    day_mon = ff["Date"] - pd.to_timedelta(ff["Date"].dt.weekday, unit="D")
    ff["anchor"] = day_mon.map(wmap)
    rf_w = ff.groupby("anchor")["RF"].apply(lambda s: float((1 + s).prod() - 1))
    print(f"  A2 weekly RF actually subtracted (FF daily RF/100, compounded "
          f"per week; {len(rf_w)} weeks):")
    for y in range(2018, 2024):
        m = rf_w[rf_w.index.year == y].mean()
        print(f"     {y}: mean {m:+.6f}/wk (~{m * 52 * 100:.2f}%/yr)")
    overall = rf_w.mean()
    print(f"     overall mean {overall:+.6f}/wk -> "
          f"{'DECIMAL units confirmed' if 0 < overall < 0.002 else 'UNIT PROBLEM'}")
    if not (0 < overall < 0.002):
        flag(f"A2 weekly RF mean {overall:.6f} not in decimal range "
             f"(scaling trap?)")

    print("  A3 mean ret_excess by year (sd in parens):")
    sd_by_year = {}
    for y in range(2018, 2024):
        g = panel.loc[panel["_yr"] == y, "ret_excess"]
        sd_by_year[y] = g.std()
        print(f"     {y}: {g.mean():+.6f}/wk (~{g.mean() * 52 * 100:+.1f}%/yr)"
              f"  (sd {g.std():.4f})")
    checks = [(2018, "<=", 0), (2019, ">", 0), (2022, "<", 0), (2023, ">", 0)]
    for y, op, val in checks:
        m = panel.loc[panel["_yr"] == y, "ret_excess"].mean()
        ok = (m <= val) if op == "<=" else (m < val) if op == "<" \
            else (m > val)
        if not ok:
            flag(f"A3 {y} mean {m:+.6f} conflicts with market history "
                 f"(expected {op} {val})")
    if max(sd_by_year, key=sd_by_year.get) != 2020:
        flag("A3 2020 is not the most volatile year - unexpected")
    else:
        print("     2020 has the largest sd (volatile year) - as expected")

    print("  A4 extreme weekly ret_excess rows (eyeball vs real events):")
    ext = pd.concat([panel.nlargest(5, "ret_excess"),
                     panel.nsmallest(5, "ret_excess")])
    for _, row in ext.iterrows():
        tk = tick_of.get(int(row["PERMNO"]), "?")
        print(f"     PERMNO {int(row['PERMNO']):>6} {tk:<6} "
              f"{row['_wa'].date()}  ret_excess {row['ret_excess']:+.4f}")

    # ---- part B: named-event returns ----
    print("\n=== PART B: NAMED-EVENT RETURN CHECK (raw rows) ===")

    def firm_week(permno: int, anchor: pd.Timestamp):
        g = panel[(panel["PERMNO"] == permno) & (panel["_wa"] == anchor)]
        return None if g.empty else float(g["ret_excess"].iloc[0])

    events = [
        ("META", 13407, anchor_containing("2022-02-03"),
         (-0.30, -0.15), "Meta earnings crash 2022-02-03, expect ~-25%"),
        ("SIVB", permno_of_ticker("SIVB"), anchor_containing("2023-03-08"),
         (-1.00, -0.40), "SVB collapse week, expect <= -40%"),
        ("NFLX", permno_of_ticker("NFLX"), anchor_containing("2022-04-20"),
         (-0.45, -0.20), "Netflix sub-loss crash 2022-04-20, expect ~-35%"),
    ]
    for tk, pn, a, (lo, hi), desc in events:
        v = firm_week(pn, a)
        if v is None:
            flag(f"B {tk} row missing for week {a.date()} ({desc})")
            continue
        ok = lo <= v <= hi
        print(f"  {tk} (PERMNO {pn}) week {a.date()}: ret_excess {v:+.4f}  "
              f"[{desc}] -> {'OK' if ok else 'OUT OF BAND'}")
        if not ok:
            flag(f"B {tk} {a.date()} ret_excess {v:+.4f} outside "
                 f"[{lo}, {hi}]")
    for a_str, cmp_, thr, desc in [
            ("2020-03-16", "<", -0.08, "COVID crash week, expect large "
             "negative cross-sectional mean"),
            ("2020-04-06", ">", 0.05, "rebound week, expect large positive "
             "mean")]:
        a = anchor_containing(a_str)
        g = panel.loc[panel["_wa"] == a, "ret_excess"]
        m = g.mean()
        ok = (m < thr) if cmp_ == "<" else (m > thr)
        print(f"  week {a.date()}: cross-sectional mean {m:+.4f} "
              f"(n={len(g)}, min {g.min():+.4f}, max {g.max():+.4f}) "
              f"[{desc}] -> {'OK' if ok else 'OUT OF BAND'}")
        if not ok:
            flag(f"B week {a.date()} mean {m:+.4f} fails {cmp_} {thr}")

    # ---- part C: headline attachment ----
    print("\n=== PART C: HEADLINE ATTACHMENT SPOT CHECK ===")
    tw = pd.to_datetime(si["target_week"])
    hd = pd.to_datetime(si["headline_date"])
    mon_tw = tw - pd.to_timedelta(tw.dt.weekday, unit="D")
    lo_d, hi_d = mon_tw - pd.Timedelta(days=7), mon_tw - pd.Timedelta(days=1)
    viol = (hd < lo_d) | (hd > hi_d)
    print(f"  FULL-TABLE window audit (all {len(si):,} headlines): "
          f"{int(viol.sum())} outside the expected preceding Mon-Sun window")
    if viol.any():
        flag(f"C {int(viol.sum())} headlines outside their S1 window")
        print(si.loc[viol].head(5).to_string())
    else:
        print("  [PASS] every scored headline falls in the Mon-Sun window "
              "immediately PRECEDING its target_week's week (S1 keying "
              "verified row-level, table-wide)")

    fkey = {}
    for arm in SENT_FILES:
        f = feats[arm]
        f = f.assign(_tw=pd.to_datetime(f["target_week"]))
        fkey[arm] = f.set_index(["PERMNO", "_tw"])

    def block(pn: int, a: pd.Timestamp, label: str, expect_gap: bool = False,
              check_sign: bool = False):
        rows = si[(si["PERMNO"] == pn)
                  & (pd.to_datetime(si["target_week"]) == a)]
        scores = {}
        for arm in SENT_FILES:
            try:
                srow = fkey[arm].loc[(pn, a)]
                scores[arm] = float(srow[f"{arm}_mean"])
                hn = int(srow["had_news"])
            except KeyError:
                scores[arm], hn = np.nan, -1
        tk = tick_of.get(pn, "?")
        print(f"\n  -- {label}: PERMNO {pn} ({tk}), target_week "
              f"{a.date()}, n_headlines {len(rows)}, had_news {hn}")
        print(f"     scores: lm {scores['lm']:+.4f} | finbert "
              f"{scores['finbert']:+.4f} | fingpt {scores['fingpt']:+.4f}")
        if len(rows) == 0:
            if expect_gap:
                print("     0 headlines - EXPECTED (documented FNSPID gap); "
                      "features zero-filled, had_news 0 "
                      + ("[PASS]" if hn == 0 and scores["lm"] == 0
                         else "[FLAG]"))
                if not (hn == 0 and scores["lm"] == 0):
                    flag(f"C {label}: gap week not zero-filled")
            else:
                flag(f"C {label}: no headlines but block expected some")
            return
        exp_lo = a - pd.Timedelta(days=int(a.weekday()) + 7)
        exp_hi = a - pd.Timedelta(days=int(a.weekday()) + 1)
        n_out = 0
        for _, hrow in rows.head(5).iterrows():
            d = pd.Timestamp(hrow["headline_date"])
            in_w = exp_lo <= d <= exp_hi
            n_out += (not in_w)
            print(f"     {d.date()} [{'ok' if in_w else 'OUT'}] "
                  f"({hrow['ticker']}) {str(hrow['headline'])[:95]}")
        all_d = pd.to_datetime(rows["headline_date"])
        out_all = int(((all_d < exp_lo) | (all_d > exp_hi)).sum())
        if out_all:
            flag(f"C {label}: {out_all}/{len(rows)} headlines outside "
                 f"window [{exp_lo.date()}..{exp_hi.date()}]")
        if check_sign and not (scores["finbert"] < 0 and scores["fingpt"] < 0):
            flag(f"C {label}: crash-coverage scores not negative "
                 f"(fb {scores['finbert']:+.3f}, fg {scores['fingpt']:+.3f})")

    # the week after an event carries its headlines; documented-gap firms
    # expect 0
    block(13407, anchor_containing("2022-02-10"),
          "META event-coverage week (gap expected: META absent from FNSPID)",
          expect_gap=True)
    block(permno_of_ticker("SIVB"), anchor_containing("2023-03-15"),
          "SIVB event-coverage week (gap expected: coverage ends 2020-06-09)",
          expect_gap=True)
    block(permno_of_ticker("NFLX"), anchor_containing("2022-04-27"),
          "NFLX event-coverage week (gap likely: ~June-2020 scrape horizon)",
          expect_gap=True)
    # highest-count firm-week whose target_week carries the 2020-03-16
    # crash week's headlines
    crash_cov = anchor_containing("2020-03-23")
    cnt = si[pd.to_datetime(si["target_week"]) == crash_cov] \
        .groupby("PERMNO").size().sort_values(ascending=False)
    block(int(cnt.index[0]), crash_cov,
          "COVID-crash coverage week, highest-count firm", check_sign=True)

    cov = si.assign(_tw=tw).groupby(["PERMNO", "_tw"]).size() \
        .rename("n").reset_index()
    cov["_yr"] = cov["_tw"].dt.year
    rng = np.random.default_rng(SEED)
    used = {13407, permno_of_ticker("SIVB"), permno_of_ticker("NFLX"),
            int(cnt.index[0])}
    print(f"\n  random high-count sample (seed {SEED}, 2 per year, "
          f"n_headlines >= 6, distinct firms):")
    for y in range(2018, 2024):
        pool = cov[(cov["_yr"] == y) & (cov["n"] >= 6)
                   & ~cov["PERMNO"].isin(used)].sort_values(
            ["PERMNO", "_tw"]).reset_index(drop=True)
        for i in rng.choice(len(pool), size=2, replace=False):
            row = pool.iloc[int(i)]
            used.add(int(row["PERMNO"]))
            block(int(row["PERMNO"]), row["_tw"],
                  f"random {y}")

    # ---- part D: verdict ----
    print("\n=== PART D: VERDICT + HARD STOP ===")
    if FLAGS:
        print(f"  Verdict: FLAG ({len(FLAGS)}):")
        for f in FLAGS:
            print(f"    - {f}")
    else:
        print("  Verdict: PASS - target moments plausible, RF in decimal "
              "units, yearly signs match market history, named events show "
              "expected magnitudes, every scored headline sits in its "
              "correct S1 window (table-wide), attachment blocks read "
              "correctly, crash-coverage scores negative.")
    print("  Nothing promoted (exploratory, read-only). STOP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
