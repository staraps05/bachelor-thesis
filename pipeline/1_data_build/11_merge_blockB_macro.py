# %% 11_merge_blockB_macro.py
# Purpose: Merge Block B (14 JKP monthly) and 8 Welch-Goyal macro columns onto Block A.
# Inputs:  Data/processed/panel_blockA.parquet, Data/Fundamentals.csv, Data/Macro_data_aggregated.csv, Data/membership_file.csv
# Outputs: Data/processed/panel_structured.parquet
# Notes:
# - Block B staleness is capped at 45 calendar days; beyond the cap the row gets NaN.
# - Panel firms absent from JKP get NaN in all 14 Block-B columns and are kept.
# - Lookups only, NaNs left for 12; 34 output columns; membership file for tickers only.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_FILE = REPO_ROOT / "Data" / "processed" / "panel_blockA.parquet"
FUND_FILE = REPO_ROOT / "Data" / "Fundamentals.csv"
MACRO_FILE = REPO_ROOT / "Data" / "Macro_data_aggregated.csv"
MEMBERSHIP_FILE = REPO_ROOT / "Data" / "membership_file.csv"
OUT_FILE = REPO_ROOT / "Data" / "processed" / "panel_structured.parquet"

WORKING_ROWS = 157_897
TOLERANCE = pd.Timedelta("45D")
JKP_BUFFER_START = pd.Timestamp("2017-06-01")   # so the first 2018 anchors inherit a value

BLOCKA = ["ret_12_1", "ret_6_1", "ret_1_0", "rvol_21d", "ivol_ff3_21d",
          "market_equity", "dolvol_126d", "ami_126d"]
BLOCKB = ["be_me", "ni_me", "sale_me", "chcsho_12m", "ope_be", "niq_be",
          "gp_at", "at_gr1", "sale_gr1", "oaccruals_at", "noa_at", "at_be",
          "niq_su", "beta_60m"]
MACRO_MAP = {"d/p": "dp", "e/p": "ep", "b/m": "bm", "ntis": "ntis",
             "tbl": "tbl", "tms": "tms", "dfy": "dfy", "svar": "svar"}
MACRO8 = list(MACRO_MAP.values())

AAPL_PERMNO = 14593
FFILL_WEEKS = [pd.Timestamp(d) for d in
               ["2023-05-19", "2023-05-26", "2023-06-02", "2023-06-09",
                "2023-06-16"]]                   # around eom 2023-05-31
TBL_WEEK_2020 = pd.Timestamp("2020-06-05")
TBL_WEEK_2023 = pd.Timestamp("2023-06-02")
SAMPLE_MONTHS = ["2019-03", "2021-07", "2023-10"]   # macro month-boundary check


def check(label: str, passed: bool, observed: str) -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {observed}")
    return passed


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}\nNothing from this run should be trusted. Fix/clarify and re-run.")


def main() -> int:
    all_pass = True

    # ---- step 1: Block B, JKP monthly forward-filled via merge_asof ----
    print("=== STEP 1: Block B (14 JKP monthly characteristics) ===")
    panel = pd.read_parquet(PANEL_FILE)
    panel["week_anchor"] = pd.to_datetime(panel["week_anchor"])
    # Verification (in place): panel_blockA row count
    if len(panel) != WORKING_ROWS:
        stop(f"panel_blockA has {len(panel):,} rows, expected {WORKING_ROWS:,}")
    # Verification (in place): panel keys unique
    if panel.duplicated(["PERMNO", "week_anchor"]).any():
        stop("panel keys not unique")
    print(f"  panel_blockA: {len(panel):,} rows, {len(panel.columns)} cols")

    # Verification (in place): Block-B columns present in Fundamentals.csv
    header = pd.read_csv(FUND_FILE, nrows=0).columns
    missing = [c for c in BLOCKB if c not in header]
    if missing:
        stop(f"Block-B columns missing from Fundamentals.csv: {missing}")

    jkp = pd.read_csv(FUND_FILE, usecols=["permno", "eom"] + BLOCKB)
    jkp["eom"] = pd.to_datetime(jkp["eom"])
    # Verification (in place): (permno, eom) unique
    if jkp.duplicated(["permno", "eom"]).any():
        stop("duplicated (permno, eom) pairs in Fundamentals.csv")
    print(f"  JKP loaded: {len(jkp):,} firm-months; (permno, eom) unique")

    firms = set(panel["PERMNO"].unique())
    jkp_firms_ever = set(jkp["permno"].unique())
    no_cov = sorted(firms - jkp_firms_ever)

    jkp = jkp[(jkp["eom"] >= JKP_BUFFER_START) & jkp["permno"].isin(firms)]
    print(f"  restricted to eom >= {JKP_BUFFER_START.date()} and panel firms: "
          f"{len(jkp):,} firm-months")

    # Verification (in place): zero-JKP-coverage firms listed
    tick = (pd.read_csv(MEMBERSHIP_FILE, usecols=["PERMNO", "Ticker", "DlyCalDt"],
                        parse_dates=["DlyCalDt"])
            .sort_values("DlyCalDt").groupby("PERMNO")["Ticker"].last())
    print(f"  zero-JKP-coverage panel firms (REIT/share-code screen): {len(no_cov)} "
          f"of {len(firms)} -> all 14 columns stay NaN (NOT dropped)")
    for p in no_cov[:10]:
        print(f"    {tick.get(p, '?'):>5} (PERMNO {p})")
    if len(no_cov) > 10:
        print(f"    ... and {len(no_cov) - 10} more")

    panel_s = panel.sort_values("week_anchor", kind="mergesort")
    jkp_s = jkp.sort_values("eom", kind="mergesort")
    merged = pd.merge_asof(panel_s, jkp_s, left_on="week_anchor", right_on="eom",
                           left_by="PERMNO", right_by="permno",
                           direction="backward", tolerance=TOLERANCE)
    merged = merged.drop(columns=["permno"]).rename(columns={"eom": "eom_matched"})
    # Verification (in place): Block-B merge preserves row count
    if len(merged) != WORKING_ROWS:
        stop(f"Block-B merge changed the row count: {len(merged):,}")

    # Verification (in place): rows NaN'd by the 45-day cap
    # rows that would have matched without the 45-day cap
    nt = pd.merge_asof(panel_s[["PERMNO", "week_anchor"]], jkp_s[["permno", "eom"]],
                       left_on="week_anchor", right_on="eom",
                       left_by="PERMNO", right_by="permno", direction="backward")
    hit_tol = merged["eom_matched"].isna().to_numpy() & nt["eom"].notna().to_numpy()
    n_hit = int(hit_tol.sum())
    print(f"  firm-weeks where the 45-day cap produced NaN instead of a stale "
          f"value: {n_hit}")
    if n_hit:
        hits = nt.loc[hit_tol].copy()
        hits["stale_days"] = (hits["week_anchor"] - hits["eom"]).dt.days
        for p, g in list(hits.groupby("PERMNO"))[:10]:
            print(f"    {tick.get(p, '?'):>5} (PERMNO {p}): {len(g)} weeks, "
                  f"blocked staleness {g['stale_days'].min()}-{g['stale_days'].max()}d "
                  f"({g['week_anchor'].min().date()} .. {g['week_anchor'].max().date()})")
        print("  [INFO] 45-day cap kept - refusing stale "
              "carries is the cap working as designed (CBOE: JKP coverage ends "
              "2018-08-31; its weeks fall to stage-5 imputation)")

    # ---- step 2: macro, Welch-Goyal monthly, month m-1 rule ----
    print("\n=== STEP 2: macro (GKX 8, Welch-Goyal monthly) ===")
    macro_raw = pd.read_csv(MACRO_FILE)
    found = [c for c in MACRO_MAP if c in macro_raw.columns]
    # Verification (in place): macro columns present
    if len(found) != 8:
        stop(f"macro columns missing: {sorted(set(MACRO_MAP) - set(found))}; "
             f"file has: {list(macro_raw.columns)}")
    print(f"  actual column names found: {found}")
    macro = macro_raw[["yyyymm"] + list(MACRO_MAP)].rename(columns=MACRO_MAP)
    macro["month"] = pd.PeriodIndex(
        pd.to_datetime(macro["yyyymm"], format="%Y%m"), freq="M")
    # Verification (in place): macro months unique
    if macro["month"].duplicated().any():
        stop("duplicated months in the macro file")

    merged["assign_month"] = merged["week_anchor"].dt.to_period("M") - 1
    need = merged["assign_month"].unique()
    have = set(macro.loc[macro[MACRO8].notna().all(axis=1), "month"])
    # Verification (in place): macro months complete for every assignment month
    if not set(need) <= have:
        stop(f"macro months missing/incomplete for: {sorted(set(need) - have)[:5]}")

    n_before = len(merged)
    merged = merged.merge(macro[["month"] + MACRO8], left_on="assign_month",
                          right_on="month", how="left", validate="m:1")
    # Verification (in place): macro merge preserves row count and creates no NaN
    if len(merged) != n_before or len(merged) != WORKING_ROWS:
        stop(f"macro merge changed the row count: {len(merged):,}")
    if merged[MACRO8].isna().any().any():
        stop("macro merge created NaNs - macro must be fully populated in-window")

    # ---- step 3: assemble + save ----
    print("\n=== STEP 3: assemble + save ===")
    final_cols = (["PERMNO", "week_anchor", "ret_excess", "n_days"]
                  + BLOCKA + BLOCKB + MACRO8)
    audit = merged[final_cols + ["eom_matched", "assign_month"]].sort_values(
        ["PERMNO", "week_anchor"]).reset_index(drop=True)
    out = audit[final_cols].copy()
    # Verification (in place): 34 enumerated columns
    if len(out.columns) != 34:
        stop(f"expected the 34 enumerated columns, got {len(out.columns)}")

    out_save = out.copy()
    out_save["week_anchor"] = out_save["week_anchor"].dt.date
    out_save.to_parquet(OUT_FILE, engine="pyarrow", index=False)

    # ---- Verification: consistency and reproduction checks ----
    all_pass &= check("all 14 Block-B codes exist verbatim (re-assert of verification/07)",
                      True, "be_me .. beta_60m")
    all_pass &= check("row count preserved through Block-B merge", True,
                      f"{WORKING_ROWS:,}")
    all_pass &= check("row count preserved through macro merge", True,
                      f"{WORKING_ROWS:,}")
    all_pass &= check("zero NaNs created by the macro merge", True,
                      "all 8 columns fully populated")
    all_pass &= check("column set == the 34 enumerated columns", True,
                      "4 id/target + 8 A + 14 B + 8 macro")
    reread = pd.read_parquet(OUT_FILE)
    if len(reread) != WORKING_ROWS or list(reread.columns) != final_cols:
        stop("re-read mismatch on panel_structured.parquet")
    print(f"  wrote {OUT_FILE} ({len(reread):,} rows x {len(reread.columns)} cols)")

    # ---- step 4: validation report ----
    print("\n=== STEP 4: VALIDATION REPORT ===")

    # Block B NaN shares, overall and per year
    yr = audit["week_anchor"].dt.year
    print("\n  Block B NaN share (overall | 2018..2023):")
    flagged = []
    for c in BLOCKB:
        ov = audit[c].isna().mean()
        per = [audit.loc[yr == y, c].isna().mean() for y in range(2018, 2024)]
        flag = "  <-- FLAG >15%" if ov > 0.15 else ""
        if ov > 0.15:
            flagged.append(c)
        print(f"    {c:<13} {ov:7.2%} | " +
              " ".join(f"{p:7.2%}" for p in per) + flag)

    # staleness of the matched JKP row
    stale = (audit["week_anchor"] - audit["eom_matched"]).dt.days
    sv = stale.dropna()
    print(f"\n  staleness (week_anchor - matched eom), matched rows "
          f"({len(sv):,}):")
    print(f"    min {sv.min():.0f}  p25 {sv.quantile(.25):.0f}  "
          f"median {sv.median():.0f}  p75 {sv.quantile(.75):.0f}  "
          f"max {sv.max():.0f} days")
    buckets = pd.cut(sv, [-1, 7, 14, 21, 31, 45],
                     labels=["0-7", "8-14", "15-21", "22-31", "32-45"])
    for b, n in buckets.value_counts().sort_index().items():
        print(f"    {b:>6} days: {n:>7,} ({n / len(sv):6.2%})")
    all_pass &= check("staleness mass sits at 3-31 days",
                      (sv <= 31).mean() > 0.95, f"{(sv <= 31).mean():.2%} <= 31d")
    print(f"    firm-weeks NaN'd by the 45-day cap (from step 1): {n_hit}")

    # forward-fill correctness: AAPL be_me around eom 2023-05-31
    print(f"\n  forward-fill spot check - AAPL {AAPL_PERMNO}, be_me around "
          f"eom 2023-05-31:")
    aapl = audit[(audit["PERMNO"] == AAPL_PERMNO)
                 & audit["week_anchor"].isin(FFILL_WEEKS)].sort_values("week_anchor")
    for _, r in aapl.iterrows():
        print(f"    week {r['week_anchor'].date()}  be_me = {r['be_me']:.6f}  "
              f"matched eom = {r['eom_matched'].date()}")
    n_vals = aapl["be_me"].nunique()
    n_eoms = aapl["eom_matched"].nunique()
    const_within = (aapl.groupby("eom_matched")["be_me"].nunique() == 1).all()
    all_pass &= check("be_me changes at most once, exactly at the eom boundary",
                      n_vals <= 2 and n_eoms == 2 and const_within,
                      f"{n_vals} distinct values across {n_eoms} matched eoms, "
                      f"constant within each")

    # no-look-ahead assertions
    print("\n  no-look-ahead assertions:")
    lookahead_b = (audit["eom_matched"] > audit["week_anchor"]).sum()
    if lookahead_b:
        stop(f"{lookahead_b} rows have matched eom AFTER the week anchor")
    all_pass &= check("Block B: matched eom <= week_anchor for every matched row",
                      True, f"0 violations in {len(sv):,} matched rows")
    anchor_month = audit["week_anchor"].dt.to_period("M")
    la_macro = (~(audit["assign_month"] < anchor_month)).sum()
    if la_macro:
        stop(f"{la_macro} rows where macro month does not strictly precede "
             f"the anchor month")
    all_pass &= check("macro: assigned month strictly precedes the anchor month",
                      True, f"0 violations in {len(audit):,} rows")

    # macro spot checks
    tbl20 = audit.loc[audit["week_anchor"] == TBL_WEEK_2020, "tbl"].iloc[0]
    tbl23 = audit.loc[audit["week_anchor"] == TBL_WEEK_2023, "tbl"].iloc[0]
    print(f"\n  macro spot checks: tbl({TBL_WEEK_2020.date()}) = {tbl20:.4f}, "
          f"tbl({TBL_WEEK_2023.date()}) = {tbl23:.4f}")
    all_pass &= check("tbl 2020 ~ 0.001-0.02 (near-zero rates)",
                      0.001 <= tbl20 <= 0.02, f"{tbl20:.4f}")
    all_pass &= check("tbl 2023 ~ 0.05", 0.04 <= tbl23 <= 0.06, f"{tbl23:.4f}")

    wk_nuniq = audit.groupby("week_anchor")[MACRO8].nunique()
    if (wk_nuniq > 1).any().any():
        stop("a macro column varies across firms within a week")
    all_pass &= check("all 8 macro columns constant within each week across firms",
                      True, f"asserted over {audit['week_anchor'].nunique()} weeks")

    ok_months = True
    for m in SAMPLE_MONTHS:
        sub = audit[audit["assign_month"] == pd.Period(m, freq="M")]
        ok_months &= (sub[MACRO8].nunique() == 1).all()
    all_pass &= check("macro values change only at month boundaries "
                      f"(sampled {', '.join(SAMPLE_MONTHS)})", bool(ok_months),
                      "constant within each assignment month")

    # cross-source consistency: JKP market_equity vs Block-A market_equity
    jme = pd.read_csv(FUND_FILE, usecols=["permno", "eom", "market_equity"])
    jme["eom"] = pd.to_datetime(jme["eom"])
    jme = jme.rename(columns={"permno": "PERMNO", "eom": "eom_matched",
                              "market_equity": "me_jkp"})
    xs = audit.merge(jme, on=["PERMNO", "eom_matched"], how="left",
                     validate="m:1")
    both = xs[["market_equity", "me_jkp"]].dropna()
    r = both["market_equity"].corr(both["me_jkp"])
    med_ratio = (both["market_equity"] / both["me_jkp"]).median()
    print(f"\n  cross-source key-alignment check ({len(both):,} rows): "
          f"corr(Block-A me, JKP me at matched eom) = {r:.4f}, "
          f"median ratio = {med_ratio:.1f}")
    all_pass &= check("cross-source correlation ~ 0.999 (>= 0.995)",
                      r >= 0.995, f"{r:.4f}")
    all_pass &= check("known x1000 unit factor (DlyCap $k vs JKP $m)",
                      990 <= med_ratio <= 1010, f"{med_ratio:.1f}")

    # coverage bottom line
    nn = audit[BLOCKB].notna().sum(axis=1)
    n_full = int((nn == 14).sum())
    n_12 = int((nn >= 12).sum())
    n_zero_firm = int(audit["PERMNO"].isin(no_cov).sum())
    print(f"\n  coverage bottom line ({WORKING_ROWS:,} rows):")
    print(f"    complete Block B (14/14): {n_full:,} ({n_full / len(audit):.2%})")
    print(f"    >= 12 of 14             : {n_12:,} ({n_12 / len(audit):.2%})")
    print(f"    rows of zero-JKP-coverage firms: {n_zero_firm:,} "
          f"({n_zero_firm / len(audit):.2%}, {len(no_cov)} firms)")



    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED - inspect above'}")
    print(f"panel_structured.parquet: {WORKING_ROWS:,} rows x {len(final_cols)} "
          f"columns (target + 8 A + 14 B + 8 macro)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
