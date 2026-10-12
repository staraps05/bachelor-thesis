# %% 13_build_scoring_input.py
# Purpose: Map FNSPID headlines to PERMNO and target week: input for LM, FinBERT, FinGPT.
# Inputs:  Data/processed/fnspid_headlines_2015_2023.csv, Data/crsp_daily_prices.csv, Data/processed/panel_final.parquet
# Outputs: Data/processed/scoring_input.parquet, Data/processed/scoring_input_coverage.csv
# Notes:
# - Loads week_anchor_map from pipeline/08_build_skeleton.py via importlib.
# - Ambiguous (ticker, day) pairs are dropped, not guessed; stops if > 0.5% are ambiguous.
# - Output is write-protected only when all checks pass; stops if a protected one exists.
import hashlib
import html
import importlib.util
import stat
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FNSPID_FILE = REPO_ROOT / "Data" / "processed" / "fnspid_headlines_2015_2023.csv"
PRICES_FILE = REPO_ROOT / "Data" / "crsp_daily_prices.csv"
PANEL_FILE = REPO_ROOT / "Data" / "processed" / "panel_final.parquet"
OUT_FILE = REPO_ROOT / "Data" / "processed" / "scoring_input.parquet"
COV_FILE = REPO_ROOT / "Data" / "processed" / "scoring_input_coverage.csv"

EXPECTED_ROWS = 5_971_447
PANEL_ROWS = 157_897
MAP_TOLERANCE = pd.Timedelta("5D")     # calendar-day lookback for the ticker map
AMBIG_STOP_SHARE = 0.005               # >0.5% ambiguous (ticker, day) pairs -> STOP
MIN_LEN, MAX_LEN = 3, 500
RNG_SEED = 20260716                    # deterministic eyeball sample

# week_anchor_map from pipeline/08_build_skeleton.py (the panel's week definition)
_spec = importlib.util.spec_from_file_location(
    "build_skeleton_08", Path(__file__).resolve().with_name("08_build_skeleton.py"))
_skel_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_skel_mod)
week_anchor_map = _skel_mod.week_anchor_map


def check(label: str, passed: bool, observed: str) -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {observed}")
    return passed


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}\nFreeze NOT granted. Fix/clarify and re-run.")


def main() -> int:
    all_pass = True
    wf = {}                                        # the row-accounting waterfall

    # ---- STEP 1 - Parse and inspect ----
    print("=== STEP 1: parse + inspect ===")
    header = list(pd.read_csv(FNSPID_FILE, nrows=0).columns)
    # Verification (in place): FNSPID header
    if header != ["Date", "Stock_symbol", "Article_title"]:
        stop(f"unexpected header: {header}")
    hl = pd.read_csv(FNSPID_FILE, dtype=str, low_memory=False)
    n_input = len(hl)
    wf["input"] = n_input

    # exact-format parse with fallbacks; calendar date taken as given (no TZ shift)
    ts = pd.to_datetime(hl["Date"], format="%Y-%m-%d %H:%M:%S UTC", errors="coerce")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        m = ts.isna()
        if not m.any():
            break
        ts.loc[m] = pd.to_datetime(hl.loc[m, "Date"], format=fmt, errors="coerce")

    hl["d"] = ts.dt.normalize()
    hl["ticker"] = hl["Stock_symbol"].str.strip().str.upper()
    hl["title"] = hl["Article_title"]
    n_bad_date = int(hl["d"].isna().sum())
    n_bad_sym = int((hl["ticker"].isna() | (hl["ticker"] == "")).sum())
    n_bad_ttl = int((hl["title"].isna() | (hl["title"].str.strip() == "")).sum())
    ok_mask = (hl["d"].notna() & hl["ticker"].notna() & (hl["ticker"] != "")
               & hl["title"].notna() & (hl["title"].str.strip() != ""))
    wf["missing-field"] = int((~ok_mask).sum())
    # Verification (in place): date range, tickers, missing fields before the filter
    print(f"  rows: {n_input:,}   date range: {hl['d'].min().date()} .. "
          f"{hl['d'].max().date()}   distinct tickers: {hl['ticker'].nunique():,}")
    print(f"  missing/empty fields (dropped): Date {n_bad_date}, "
          f"Stock_symbol {n_bad_sym}, Article_title {n_bad_ttl} "
          f"-> {wf['missing-field']} rows")
    hl = hl.loc[ok_mask, ["d", "ticker", "title"]]

    # ---- STEP 2 - Point-in-time ticker->PERMNO map ----
    print("\n=== STEP 2: point-in-time ticker->PERMNO map ===")
    crsp = pd.read_csv(PRICES_FILE, usecols=["PERMNO", "Ticker", "DlyCalDt"],
                       parse_dates=["DlyCalDt"]).drop_duplicates()
    cmap = crsp.dropna(subset=["Ticker"]).copy()
    cmap["Ticker"] = cmap["Ticker"].str.strip().str.upper()

    # Verification (in place): ticker->PERMNO ambiguity audit
    # ambiguity audit: (Ticker, day) pairs carrying 2+ PERMNOs
    nuniq = cmap.groupby(["Ticker", "DlyCalDt"])["PERMNO"].nunique()
    ambig = nuniq[nuniq > 1]
    share = len(ambig) / len(nuniq)
    print(f"  ambiguity audit: {len(ambig):,} of {len(nuniq):,} (Ticker, day) "
          f"pairs map to 2+ PERMNOs ({share:.4%})")
    if len(ambig):
        ex = ambig.reset_index().groupby("Ticker").agg(
            days=("DlyCalDt", "size"), first=("DlyCalDt", "min"),
            last=("DlyCalDt", "max"))
        for t, r in ex.head(20).iterrows():
            permnos = sorted(cmap.loc[cmap["Ticker"] == t, "PERMNO"].unique())
            print(f"    {t}: {r['days']} ambiguous day(s) "
                  f"{r['first'].date()}..{r['last'].date()}, PERMNOs {permnos}")
    if share > AMBIG_STOP_SHARE:
        stop(f"ambiguous share {share:.3%} exceeds 0.5% - map not trustworthy")

    # Resolve: one row per (Ticker, day); ambiguous days keep PERMNO=NaN + flag.
    map_tbl = (cmap.groupby(["Ticker", "DlyCalDt"], as_index=False)
               .agg(PERMNO=("PERMNO", "first"), n=("PERMNO", "nunique")))
    map_tbl["ambig"] = map_tbl["n"] > 1
    map_tbl.loc[map_tbl["ambig"], "PERMNO"] = np.nan
    map_tbl = map_tbl.drop(columns="n")

    # tickers reused by different PERMNOs over time (handled by the point-in-time map)
    recyc = cmap.groupby("Ticker")["PERMNO"].nunique()
    recyc = recyc[recyc > 1]
    print(f"  tickers carried by 2+ PERMNOs over time (recycled/changed - "
          f"handled point-in-time): {len(recyc)}")
    if len(recyc):
        t0 = recyc.index[0]
        spells = (cmap[cmap["Ticker"] == t0].groupby("PERMNO")["DlyCalDt"]
                  .agg(["min", "max"]).sort_values("min"))
        print(f"    example '{t0}': " + "; ".join(
            f"PERMNO {p} {r['min'].date()}..{r['max'].date()}"
            for p, r in spells.iterrows()))

    print("  mapping rule: pd.merge_asof(headlines sorted by date, map sorted by "
          "day, by=ticker,")
    print(f"  direction='backward', tolerance={MAP_TOLERANCE.days} calendar days) "
          f"- last trading day <= headline date")
    hl = hl.sort_values("d", kind="mergesort")     # stable: file order kept in ties
    map_tbl = map_tbl.sort_values("DlyCalDt", kind="mergesort")
    hl = pd.merge_asof(hl, map_tbl, left_on="d", right_on="DlyCalDt",
                       left_by="ticker", right_by="Ticker",
                       direction="backward", tolerance=MAP_TOLERANCE)

    crsp_tickers = set(map_tbl["Ticker"].unique())
    in_crsp = hl["ticker"].isin(crsp_tickers)
    matched = hl["PERMNO"].notna()
    is_ambig = hl["ambig"].eq(True)          # NaN (no match) -> False, no downcast
    wf["unmapped-ticker"] = int((~in_crsp).sum())
    wf["ambiguous"] = int((in_crsp & is_ambig).sum())
    wf["tolerance-failed"] = int((in_crsp & ~is_ambig & ~matched).sum())
    print(f"  mapping waterfall: {len(hl):,} headlines -> matched "
          f"{int(matched.sum()):,} | unmapped ticker (never in CRSP - expected "
          f"LARGE, FNSPID ~4,775 firms vs our 719): {wf['unmapped-ticker']:,} | "
          f"ambiguous-dropped: {wf['ambiguous']:,} | tolerance-failed: "
          f"{wf['tolerance-failed']:,}")
    hl = hl.loc[matched, ["d", "ticker", "title", "PERMNO"]]
    hl["PERMNO"] = hl["PERMNO"].astype("int64")
    hl_matched = hl.copy()                         # kept for the event checks

    # ---- STEP 3 - Week assignment + t->t+1 lag ----
    print("\n=== STEP 3: week assignment + t->t+1 lag ===")
    trading_days = pd.DatetimeIndex(crsp["DlyCalDt"].unique()).sort_values()
    amap = week_anchor_map(trading_days)
    print("  anchors from week_anchor_map (pipeline/08, imported - not re-implemented);")
    print("  domain = Mondays 2018-01-01..2023-12-25 = exactly the valid TARGET")
    print("  weeks, so the call needs no extension: target = amap[Monday(d) + 7d];")
    print("  outside-domain rows are the no-target-week bucket.")

    monday = hl["d"] - pd.to_timedelta(hl["d"].dt.dayofweek, unit="D")
    target = pd.Series(amap.reindex(monday + pd.Timedelta(days=7)).to_numpy(),
                       index=hl.index)
    no_tgt = target.isna()
    pre18 = no_tgt & (hl["d"] <= pd.Timestamp("2017-12-24"))
    final_wk = no_tgt & (hl["d"] >= pd.Timestamp("2023-12-25"))
    # Verification (in place): no-target rows accounted for
    if int(no_tgt.sum()) != int(pre18.sum() + final_wk.sum()):
        stop("no-target rows beyond the two expected classes")
    wf["no-target-week"] = int(no_tgt.sum())
    print(f"  no-target-week: {wf['no-target-week']:,} "
          f"(dated <= 2017-12-24: {int(pre18.sum()):,}; final anchor's own week "
          f"2023-12-25..31: {int(final_wk.sum()):,})")

    hl, target = hl.loc[~no_tgt], target.loc[~no_tgt]
    hl["target_week"] = target
    # Verification (in place): feeder week of the first target
    feed_first = hl.loc[hl["target_week"] == pd.Timestamp("2018-01-05"), "d"]
    all_pass &= check("week 2017-12-25..31 feeds target 2018-01-05",
                      feed_first.min() >= pd.Timestamp("2017-12-25")
                      and feed_first.max() <= pd.Timestamp("2017-12-31"),
                      f"feeder dates {feed_first.min().date()} .. "
                      f"{feed_first.max().date()} ({len(feed_first):,} headlines)")
    # .isin coerces dtypes; a set() of Timestamps would not hash-match np.datetime64
    # Verification (in place): every target week is a panel anchor
    if not hl["target_week"].isin(amap.to_numpy()).all():
        stop("a target week is not one of the 313 panel anchors")
    # Verification (in place): strict lag target_week > headline_date
    n_viol = int((hl["target_week"] <= hl["d"]).sum())
    if n_viol:
        stop(f"{n_viol} rows violate the strict lag target_week > headline_date")
    all_pass &= check("strict lag: target_week_anchor > headline_date, zero "
                      "violations", True, f"0 of {len(hl):,}")

    # ---- STEP 4 - Filter to the panel key set ----
    print("\n=== STEP 4: filter to the frozen panel key set ===")
    panel = pd.read_parquet(PANEL_FILE, columns=["PERMNO", "week_anchor"])
    # Verification (in place): panel row count
    if len(panel) != PANEL_ROWS:
        stop(f"panel_final has {len(panel):,} rows, expected {PANEL_ROWS:,}")
    panel_keys = pd.MultiIndex.from_arrays(
        [panel["PERMNO"].to_numpy(), pd.to_datetime(panel["week_anchor"])])
    hl_keys = pd.MultiIndex.from_arrays([hl["PERMNO"].to_numpy(),
                                         hl["target_week"].to_numpy()])
    in_panel = hl_keys.isin(panel_keys)
    wf["not-in-panel"] = int((~in_panel).sum())
    print(f"  dropped (firm not a member in the target week / S4-dropped row): "
          f"{wf['not-in-panel']:,}")
    hl = hl.loc[in_panel]

    # ---- STEP 5 - Clean + dedup ----
    print("\n=== STEP 5: clean + dedup ===")
    t = hl["title"]
    has_ent = t.str.contains("&", regex=False)
    decoded = t.loc[has_ent].map(html.unescape)
    n_ent = int((decoded != t.loc[has_ent]).sum())
    t = t.copy()
    t.loc[has_ent] = decoded
    print(f"  HTML entities found and decoded (&amp; etc.): {n_ent:,} rows")
    t = t.str.replace(r"\s+", " ", regex=True).str.strip()
    hl["headline"] = t

    L = t.str.len()
    cls = {"empty after cleaning": L == 0,
           f"shorter than {MIN_LEN}": (L > 0) & (L < MIN_LEN),
           f"longer than {MAX_LEN}": L > MAX_LEN}
    drop_clean = pd.Series(False, index=hl.index)
    for name, m in cls.items():
        print(f"  dropped - {name}: {int(m.sum()):,}")
        for s in hl.loc[m, "headline"].head(5):
            print(f"      example: {s[:120]!r}")
        drop_clean |= m
    wf["cleaned-away"] = int(drop_clean.sum())
    hl = hl.loc[~drop_clean]

    # exact dedup within (PERMNO, headline_date), keep first in file order (all
    # sorts were stable); the same title on another firm or date is kept
    dup = hl.duplicated(subset=["PERMNO", "d", "headline"], keep="first")
    wf["deduped"] = int(dup.sum())
    print(f"  exact duplicates within (PERMNO, headline_date) dropped "
          f"(keep first): {wf['deduped']:,}")
    hl = hl.loc[~dup]
    wf["kept"] = len(hl)

    # ---- STEP 6 - Save ----
    print("\n=== STEP 6: save + freeze ===")
    out = (hl.rename(columns={"d": "headline_date"})
           [["PERMNO", "target_week", "headline_date", "ticker", "headline"]]
           .sort_values(["PERMNO", "target_week", "headline_date"],
                        kind="mergesort").reset_index(drop=True))
    out["target_week"] = out["target_week"].dt.date
    out["headline_date"] = out["headline_date"].dt.date
    # Verification (in place): refuse to overwrite a frozen output
    if OUT_FILE.exists() and not (OUT_FILE.stat().st_mode & stat.S_IWUSR):
        stop(f"{OUT_FILE.name} exists and is frozen - new version required")
    out.to_parquet(OUT_FILE, engine="pyarrow", index=False)

    cov = (out.groupby(["PERMNO", "target_week"]).size()
           .rename("n_headlines").reset_index())
    cov.to_csv(COV_FILE, index=False)
    sha = hashlib.sha256(OUT_FILE.read_bytes()).hexdigest()
    print(f"  wrote {OUT_FILE} ({len(out):,} rows)")
    print(f"  wrote {COV_FILE} ({len(cov):,} firm-weeks with >=1 headline)")
    print(f"  SHA256: {sha}")
    print("  (write-protection is applied at the END, only if ALL checks pass)")

    # ---- Verification: consistency and reproduction checks ----
    all_pass &= check("row count == expected 5,971,447",
                      n_input == EXPECTED_ROWS,
                      f"{n_input:,} ({n_input - EXPECTED_ROWS:+,} deviation)")
    date_only = float((ts.dt.hour.eq(0) & ts.dt.minute.eq(0)
                       & ts.dt.second.eq(0)).mean())
    print(f"  convention: calendar date of the timestamp AS GIVEN (no TZ shift);")
    all_pass &= check("date-only (00:00:00) share ~79%", 0.73 <= date_only <= 0.85,
                      f"{date_only:.1%}")
    n_no_tick = int(crsp["Ticker"].isna().sum())
    print(f"  CRSP rows with missing Ticker (excluded from the map): {n_no_tick}")

    # ---- STEP 7 - VALIDATION REPORT ----
    print("\n=== STEP 7: VALIDATION REPORT ===")

    print("\n  row-accounting waterfall (every input row in exactly one bucket):")
    for k, v in wf.items():
        if k != "input":
            print(f"    {k:<18} {v:>11,}")
    total = sum(v for k, v in wf.items() if k != "input")
    if total != n_input:
        stop(f"waterfall sums to {total:,}, input was {n_input:,}")
    all_pass &= check("buckets sum exactly to the input count", True,
                      f"{total:,} == {n_input:,}")

    aa = hl_matched[hl_matched["ticker"] == "AA"]
    aa_pre = aa.loc[aa["d"] <= pd.Timestamp("2016-10-31"), "PERMNO"].unique()
    aa_post = aa.loc[aa["d"] >= pd.Timestamp("2016-11-01"), "PERMNO"].unique()
    all_pass &= check("substitute point-in-time proof: AA headlines switch "
                      "PERMNO exactly at the 2016-11-01 Alcoa/Arconic split",
                      sorted(aa_pre.tolist()) == [24643]
                      and sorted(aa_post.tolist()) == [16347],
                      f"AA<=2016-10-31 -> {sorted(aa_pre.tolist())} "
                      f"({(aa['d'] <= pd.Timestamp('2016-10-31')).sum()} rows); "
                      f"AA>=2016-11-01 -> {sorted(aa_post.tolist())} "
                      f"({(aa['d'] >= pd.Timestamp('2016-11-01')).sum()} rows)")

    # SIVB event check (FNSPID's SIVB coverage ends in 2020, so the else branch runs)
    ev_days = [pd.Timestamp("2023-03-09"), pd.Timestamp("2023-03-10")]
    sivb = hl_matched[(hl_matched["ticker"] == "SIVB") & hl_matched["d"].isin(ev_days)]
    svb_title = hl_matched[hl_matched["d"].isin(ev_days)
                           & hl_matched["title"].str.contains(
                               "Silicon Valley Bank", case=False, regex=False)]
    sivb_all = hl_matched.loc[hl_matched["ticker"] == "SIVB"]
    if len(sivb):
        smon = sivb["d"] - pd.to_timedelta(sivb["d"].dt.dayofweek, unit="D")
        stgt = pd.Series(amap.reindex(smon + pd.Timedelta(days=7)).to_numpy()).unique()
        all_pass &= check("SIVB 2023-03-09/10 headlines map to 11786, target "
                          "2023-03-17",
                          set(sivb["PERMNO"]) == {11786}
                          and list(stgt) == [pd.Timestamp("2023-03-17")],
                          f"{len(sivb)} headlines -> "
                          f"{sorted(set(sivb['PERMNO']))}, target {stgt[0]}")
    else:
        print(f"  [INFO] SIVB 2023-03-09/10 event check as pre-registered: 0 SIVB "
              f"rows on those dates (coverage: {len(sivb_all)} rows "
              f"{sivb_all['d'].min().date()}..{sivb_all['d'].max().date()}; title "
              f"search 'Silicon Valley Bank': {len(svb_title)} rows)")
        all_pass &= check("substitute: the SIVB rows that DO exist map to 11786",
                          sorted(sivb_all["PERMNO"].unique().tolist()) == [11786],
                          f"{sorted(sivb_all['PERMNO'].unique().tolist())}")

    # TSLA pre-membership check
    tsla = hl_matched[hl_matched["PERMNO"] == 93436]
    tsla_nov = tsla[tsla["d"].dt.strftime("%Y-%m") == "2020-11"]
    entry = pd.Timestamp("2020-12-24")              # TSLA's first panel anchor
    if len(tsla_nov):
        tsla_kept = out[(out["PERMNO"] == 93436)
                        & (pd.to_datetime(out["headline_date"])
                           .dt.strftime("%Y-%m") == "2020-11")]
        all_pass &= check("TSLA 2020-11 headlines mapped but EXCLUDED "
                          "(panel entry 2020-12-24)",
                          len(tsla_kept) == 0,
                          f"{len(tsla_nov):,} mapped -> {len(tsla_kept)} kept; "
                          f"e.g. {tsla_nov['title'].iloc[0][:60]!r}")
    else:
        m2020 = (tsla.loc[tsla["d"].dt.year == 2020, "d"]
                 .dt.strftime("%Y-%m").value_counts().sort_index().to_dict())
        print(f"  [INFO] TSLA 2020-11 check as pre-registered: 0 TSLA rows dated "
              f"2020-11 (FNSPID TSLA months in 2020: {m2020})")
    # every TSLA headline whose target week precedes the entry anchor must have
    # been excluded by the panel filter
    tmon = tsla["d"] - pd.to_timedelta(tsla["d"].dt.dayofweek, unit="D")
    ttgt = pd.Series(amap.reindex(tmon + pd.Timedelta(days=7)).to_numpy(),
                     index=tsla.index)
    n_pre = int((ttgt.notna() & (ttgt < entry)).sum())
    kept_pre = out[(out["PERMNO"] == 93436)
                   & (pd.to_datetime(out["target_week"]) < entry)]
    all_pass &= check("substitute: ALL pre-membership TSLA headlines excluded "
                      "(target week < 2020-12-24)",
                      n_pre > 0 and len(kept_pre) == 0,
                      f"{n_pre:,} mapped pre-membership headlines -> "
                      f"{len(kept_pre)} kept")

    all_pass &= check("lag assertion on the final file: 0 violations",
                      (pd.to_datetime(out["target_week"])
                       > pd.to_datetime(out["headline_date"])).all(),
                      f"target_week > headline_date on all {len(out):,} rows")

    # coverage statistics
    print(f"\n  FINAL HEADLINE COUNT: {len(out):,}")
    q = cov["n_headlines"].quantile([0.25, 0.5, 0.75, 0.95])
    mx = cov.loc[cov["n_headlines"].idxmax()]
    mx_tick = out.loc[out["PERMNO"] == mx["PERMNO"], "ticker"].iloc[0]
    print(f"  headlines per (firm, week) with >=1: p25 {q[0.25]:.0f}  "
          f"median {q[0.5]:.0f}  p75 {q[0.75]:.0f}  p95 {q[0.95]:.0f}  "
          f"max {mx['n_headlines']:,} ({mx_tick}, PERMNO {int(mx['PERMNO'])}, "
          f"week {mx['target_week']})")

    pk = panel.copy()
    pk["week_anchor"] = pd.to_datetime(pk["week_anchor"])
    pk = pk.merge(cov.rename(columns={"target_week": "week_anchor"})
                  .assign(week_anchor=lambda x: pd.to_datetime(x["week_anchor"])),
                  on=["PERMNO", "week_anchor"], how="left")
    pk["n_headlines"] = pk["n_headlines"].fillna(0)
    no_news = pk["n_headlines"].eq(0)
    print(f"  no-news share of the {PANEL_ROWS:,} panel firm-weeks: "
          f"{no_news.mean():.2%}")
    print("    per year: " + "  ".join(
        f"{y}: {no_news[pk['week_anchor'].dt.year == y].mean():.1%}"
        for y in range(2018, 2024)))
    print("  kept headlines per year (by headline_date): " + "  ".join(
        f"{y}: {n:,}" for y, n in pd.to_datetime(out['headline_date'])
        .dt.year.value_counts().sort_index().items()))


    # Verification (in place): freeze only if all checks passed
    if not all_pass:
        print("\nOverall: SOME CHECKS FAILED - artifacts written but NOT write-protected")
        return 1
    OUT_FILE.chmod(OUT_FILE.stat().st_mode
                   & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print("\n" + "=" * 72)
    print("  SENTIMENT S1 - SCORING INPUT FROZEN")
    print(f"  {OUT_FILE}")
    print(f"  {len(out):,} headlines | SHA256: {sha}")
    print("=" * 72)
    print("\nOverall: ALL CHECKS PASSED — FREEZE GRANTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
