# %% 49_numbers_data_figures.py
# Purpose: Recompute the Chapter 3 data figures (lost exit weeks, spin-offs, SVB week, anchor-day prices, JKP staleness, absent firms, rank bounds, midnight share).
# Inputs:  Data/crsp_daily_prices.csv, Data/membership_file.csv, Data/F-F_Research_Data_Factors_daily.csv, Data/Fundamentals.csv, Data/fnspid_nasdaq_news_raw.csv, Data/processed/skeleton.parquet, Data/processed/panel_structured.parquet, Data/processed/panel_target.parquet, Data/processed/panel_final.parquet, Data/processed/scoring_input.parquet
# Outputs: Data/processed/numbers_check/ item3_data_figures.csv, item3_a_lost_weeks.csv, item3_h_absent_firms.csv
# Notes:
# - Loads pipeline/1_data_build/08 via importlib for week_anchor_map.
# - Reads the 22 GB raw FNSPID file for the midnight share; this step dominates runtime.
# - scoring_input.parquet is checked against the SHA256 recorded in 2_sentiment/15.

import hashlib
import html
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "Data"
PROC = DATA / "processed"
NC = PROC / "numbers_check"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c08 = _load("skel08", "1_data_build/08_build_skeleton.py")
week_anchor_map = c08.week_anchor_map


def main() -> int:
    NC.mkdir(parents=True, exist_ok=True)
    out = {}
    print("=== 49: data-foundation figures ===")

    # ---- week anchors ----
    caldt = pd.read_csv(DATA / "crsp_daily_prices.csv", usecols=["DlyCalDt"],
                        parse_dates=["DlyCalDt"])
    trading_days = pd.DatetimeIndex(caldt["DlyCalDt"].unique()).sort_values()
    amap = week_anchor_map(trading_days)            # index=Monday, value=anchor
    anchors = pd.DatetimeIndex(sorted(amap.values))
    first_a, last_a = anchors[0], anchors[-1]

    def anchor_of(dts):
        mon = dts - pd.to_timedelta(pd.DatetimeIndex(dts).dayofweek, unit="D")
        return pd.DatetimeIndex(mon).normalize().map(amap)

    # ---- membership file (daily member rows) ----
    mem = pd.read_csv(DATA / "membership_file.csv",
                      usecols=["PERMNO", "MbrStartDt", "MbrEndDt", "DlyCalDt",
                               "DlyPrc", "DlyRet", "DlyCap", "DlyVol", "SICCD",
                               "USIncFlg", "Ticker"],
                      parse_dates=["MbrStartDt", "MbrEndDt", "DlyCalDt"])
    mem["DlyRet"] = pd.to_numeric(mem["DlyRet"], errors="coerce")
    mem["DlyPrc"] = pd.to_numeric(mem["DlyPrc"], errors="coerce")
    mem["DlyCap"] = pd.to_numeric(mem["DlyCap"], errors="coerce")
    mem["DlyVol"] = pd.to_numeric(mem["DlyVol"], errors="coerce")
    # member-days restricted to within each row's own spell (inclusive end)
    mem = mem[(mem["DlyCalDt"] >= mem["MbrStartDt"])
              & (mem["DlyCalDt"] <= mem["MbrEndDt"].fillna(pd.Timestamp.max))].copy()
    file_last = mem["DlyCalDt"].max()
    spells = (mem[["PERMNO", "MbrStartDt", "MbrEndDt"]].drop_duplicates()
              .reset_index(drop=True))

    skel = pd.read_parquet(PROC / "skeleton.parquet")
    skel["week_anchor"] = pd.to_datetime(skel["week_anchor"])
    skel_set = set(zip(skel["PERMNO"], skel["week_anchor"]))
    panel_permnos = set(pd.read_parquet(PROC / "panel_structured.parquet",
                                        columns=["PERMNO"])["PERMNO"].unique())

    # ---- (a) lost exit weeks ----
    # spell ends inside the window before its week's anchor and the firm has no row
    lost = []
    for p, s, e in zip(spells["PERMNO"], spells["MbrStartDt"], spells["MbrEndDt"]):
        if pd.isna(e) or e >= file_last:
            continue                                  # still a member / open
        w = anchor_of(pd.DatetimeIndex([e]))[0]
        if pd.isna(w) or w < first_a or w > last_a:
            continue
        if e < w and (p, w) not in skel_set:          # left before the anchor
            lost.append((int(p), w))
    lost = sorted(set(lost))
    n_lost = len(lost)
    pct_lost = n_lost / 157_897 * 100      # panel row count as in pipeline/10
    out["a_lost_exit_weeks"] = n_lost
    out["a_pct_of_157897"] = pct_lost
    print(f"  (a) lost exit weeks: {n_lost}  ({pct_lost:.3f}% of 157,897)")

    # ---- (b) first post-exit weekly excess return of each lost week ----
    ff = pd.read_csv(DATA / "F-F_Research_Data_Factors_daily.csv", skiprows=4)
    ff = ff.rename(columns={ff.columns[0]: "Date"})
    ff["Date"] = pd.to_numeric(ff["Date"], errors="coerce")
    ff = ff.dropna(subset=["Date"])
    ff["Date"] = pd.to_datetime(ff["Date"].astype(int).astype(str))
    ff["RF"] = pd.to_numeric(ff["RF"]) / 100.0
    rf = ff.set_index("Date")["RF"]
    lost_df = pd.DataFrame(lost, columns=["PERMNO", "w"])
    lost_df["mon"] = lost_df["w"] - pd.to_timedelta(pd.DatetimeIndex(lost_df["w"]).dayofweek, unit="D")
    excs, raws = [], []
    n_noret = 0
    for p, w, mon in zip(lost_df["PERMNO"], lost_df["w"], lost_df["mon"]):
        sun = mon + pd.Timedelta(days=6)
        g = mem[(mem["PERMNO"] == p) & (mem["DlyCalDt"] >= mon)
                & (mem["DlyCalDt"] <= sun)].dropna(subset=["DlyRet"])
        if len(g) == 0:
            n_noret += 1
            continue
        days = pd.DatetimeIndex(g["DlyCalDt"])
        ret_w = float(np.prod(1.0 + g["DlyRet"].to_numpy()) - 1.0)
        rf_w = float(np.prod(1.0 + rf.reindex(days).fillna(0.0).to_numpy()) - 1.0)
        raws.append(ret_w)
        excs.append(ret_w - rf_w)
    excs = np.array(excs)
    raws = np.array(raws)
    excs66 = np.concatenate([excs, np.zeros(n_noret)])   # no-return weeks as 0
    # market-adjusted: minus the equal-weighted panel ret_excess that week
    tgt = pd.read_parquet(PROC / "panel_target.parquet",
                          columns=["week_anchor", "ret_excess"])
    tgt["week_anchor"] = pd.to_datetime(tgt["week_anchor"])
    ew = tgt.groupby("week_anchor")["ret_excess"].mean()
    adj = []
    for p, w, mon in zip(lost_df["PERMNO"], lost_df["w"], lost_df["mon"]):
        sun = mon + pd.Timedelta(days=6)
        g = mem[(mem["PERMNO"] == p) & (mem["DlyCalDt"] >= mon)
                & (mem["DlyCalDt"] <= sun)].dropna(subset=["DlyRet"])
        if len(g) == 0:
            continue
        days = pd.DatetimeIndex(g["DlyCalDt"])
        ret_w = float(np.prod(1.0 + g["DlyRet"].to_numpy()) - 1.0)
        rf_w = float(np.prod(1.0 + rf.reindex(days).fillna(0.0).to_numpy()) - 1.0)
        adj.append((ret_w - rf_w) - float(ew.get(w, 0.0)))
    adj = np.array(adj)
    out["b_n_with_returns"] = len(excs)
    out["b_n_no_return"] = n_noret
    out["b_mean_excess_ret_pct_of50"] = float(excs.mean() * 100)
    out["b_mean_raw_ret_pct_of50"] = float(raws.mean() * 100)
    out["b_mean_excess_ret_pct_of66"] = float(excs66.mean() * 100)
    out["b_median_excess_ret_pct_of50"] = float(np.median(excs) * 100)
    out["b_mean_market_adj_pct_of50"] = float(adj.mean() * 100)
    print(f"  (b) lost weeks with returns: {len(excs)} (no-return {n_noret}); "
          f"excess/50 {excs.mean() * 100:+.3f}% | raw/50 {raws.mean() * 100:+.3f}% | "
          f"excess/66 {excs66.mean() * 100:+.3f}% | mkt-adj/50 {adj.mean() * 100:+.3f}% | "
          f"median/50 {np.median(excs) * 100:+.3f}%")

    # ---- (c) between-anchor spin-off entities ----
    end_eff = spells["MbrEndDt"].fillna(pd.Timestamp.max)
    spinoffs = []
    for p, s, e in zip(spells["PERMNO"], spells["MbrStartDt"], end_eff):
        if e < first_a or s > last_a:                 # no window overlap
            continue
        lo = anchors.searchsorted(s, "left")
        hi = anchors.searchsorted(e, "right")
        if hi == lo:                                   # spell covers no anchor
            if p in panel_permnos:
                continue                               # has rows elsewhere
            tk = mem.loc[mem["PERMNO"] == p, "Ticker"].iloc[0]
            cal = mem.loc[mem["PERMNO"] == p, "DlyCalDt"]
            ndays = int(((cal >= s) & (cal <= (e if e != pd.Timestamp.max else cal.max()))).sum())
            spinoffs.append((int(p), str(tk), ndays))
    out["c_spinoff_entities"] = len(spinoffs)
    out["c_spinoff_list"] = ";".join(f"{t}({n}d)" for _, t, n in spinoffs)
    print(f"  (c) between-anchor spin-off entities: {len(spinoffs)} -> {out['c_spinoff_list']}")

    # ---- (d) SVB exit week ----
    svb = mem[mem["PERMNO"] == 11786].copy().sort_values("DlyCalDt")
    w = pd.Timestamp("2023-03-10")
    mon = w - pd.Timedelta(days=w.dayofweek)
    prior_anchor = pd.Timestamp("2023-03-03")
    pc = abs(float(svb.loc[svb["DlyCalDt"] == prior_anchor, "DlyPrc"].iloc[0]))
    inwk = svb[(svb["DlyCalDt"] >= mon) & (svb["DlyCalDt"] <= w)].copy()
    traded = inwk[(inwk["DlyVol"].fillna(0) > 0) & inwk["DlyPrc"].notna()]
    last_traded_px = abs(float(traded["DlyPrc"].iloc[-1]))
    last_traded_dt = traded["DlyCalDt"].iloc[-1]
    last_any_px = abs(float(inwk.dropna(subset=["DlyPrc"])["DlyPrc"].iloc[-1]))
    last_any_dt = inwk.dropna(subset=["DlyPrc"])["DlyCalDt"].iloc[-1]
    out["d_svb_prior_close_0303"] = pc
    out["d_svb_last_traded_px"] = last_traded_px
    out["d_svb_last_traded_date"] = str(last_traded_dt.date())
    out["d_svb_loss_last_traded_pct"] = (last_traded_px / pc - 1.0) * 100
    out["d_svb_last_any_px"] = last_any_px
    out["d_svb_loss_last_any_pct"] = (last_any_px / pc - 1.0) * 100
    print(f"  (d) SVB prior close {pc:.2f}; last TRADED {last_traded_px:.2f} ({last_traded_dt.date()}) "
          f"-> {out['d_svb_loss_last_traded_pct']:+.1f}% | last any-price {last_any_px:.2f} "
          f"({last_any_dt.date()}) -> {out['d_svb_loss_last_any_pct']:+.1f}%")

    # ---- (e) spells ending on an anchor with no traded price that day ----
    # the panel row then takes market_equity from the last in-week capitalisation
    anchor_set = set(pd.DatetimeIndex(anchors))
    prc_set = set(zip(mem.loc[mem["DlyPrc"].notna(), "PERMNO"],
                      mem.loc[mem["DlyPrc"].notna(), "DlyCalDt"]))
    cap_set = set(zip(mem.loc[mem["DlyCap"].notna(), "PERMNO"],
                      mem.loc[mem["DlyCap"].notna(), "DlyCalDt"]))
    end_on_anchor = 0
    friday_leavers = []
    for p, e in zip(spells["PERMNO"], spells["MbrEndDt"]):
        if pd.isna(e) or e not in anchor_set:
            continue
        end_on_anchor += 1
        if (p, e) not in prc_set:                      # no traded price on the anchor
            friday_leavers.append((int(p), e.date()))
    # context: all skeleton rows with no DlyPrc / no DlyCap on their anchor day
    skel_no_prc = sum(1 for (p, a) in skel_set if (p, a) not in prc_set)
    skel_no_cap = [(p, a) for (p, a) in skel_set if (p, a) not in cap_set]
    cap_fallback = 0
    for p, a in skel_no_cap:
        mon = a - pd.Timedelta(days=a.dayofweek)
        if any((p, d) in cap_set for d in pd.date_range(mon, a - pd.Timedelta(days=1))):
            cap_fallback += 1
    out["e_spells_ending_on_anchor"] = end_on_anchor
    out["e_friday_leavers_no_price"] = len(friday_leavers)
    out["e_friday_leaver_list"] = ";".join(f"{p}@{d}" for p, d in friday_leavers)
    out["e_all_skel_no_prc_on_anchor"] = skel_no_prc
    out["e_cap_fallback_rows"] = cap_fallback
    print(f"  (e) spells ending on an anchor: {end_on_anchor}; of those, no traded "
          f"price on the anchor day (Friday-leavers): {len(friday_leavers)} "
          f"-> {out['e_friday_leaver_list']}")

    # ---- (g) JKP merge staleness ----
    fund = pd.read_csv(DATA / "Fundamentals.csv", usecols=["permno", "eom"])
    fund["eom"] = pd.to_datetime(fund["eom"], errors="coerce")
    fund = fund.dropna(subset=["eom"]).rename(columns={"permno": "PERMNO"})
    fund["PERMNO"] = fund["PERMNO"].astype("int64")
    fund = fund[["PERMNO", "eom"]].drop_duplicates().sort_values(["PERMNO", "eom"])
    sk = skel.rename(columns={"week_anchor": "anchor"}).copy()
    sk["PERMNO"] = sk["PERMNO"].astype("int64")
    sk = sk.sort_values(["PERMNO", "anchor"])
    merged = pd.merge_asof(sk.sort_values("anchor"), fund.sort_values("eom"),
                           left_on="anchor", right_on="eom", by="PERMNO",
                           direction="backward", tolerance=pd.Timedelta("45D"))
    m = merged.dropna(subset=["eom"])
    stale = (m["anchor"] - m["eom"]).dt.days
    out["g_matched_rows"] = int(len(m))
    out["g_stale_min"] = int(stale.min())
    out["g_stale_max"] = int(stale.max())
    out["g_share_le_31d_pct"] = float((stale <= 31).mean() * 100)
    print(f"  (g) JKP staleness: matched {len(m):,}; min {stale.min()} max {stale.max()} days; "
          f"<=31d {(stale <= 31).mean() * 100:.4f}%")

    # ---- (h) panel firms absent from JKP ----
    absent = sorted(panel_permnos - set(fund["PERMNO"].unique()))
    meta = mem.drop_duplicates("PERMNO").set_index("PERMNO")
    rows = []
    for p in absent:
        sic = meta.loc[p, "SICCD"] if p in meta.index else None
        usinc = meta.loc[p, "USIncFlg"] if p in meta.index else None
        tk = meta.loc[p, "Ticker"] if p in meta.index else None
        rows.append({"PERMNO": p, "Ticker": tk, "SICCD": sic, "USIncFlg": usinc})
    hdf = pd.DataFrame(rows)
    # IssuerType from the same CRSP membership file (first row per PERMNO)
    issuer = (pd.read_csv(DATA / "membership_file.csv", usecols=["PERMNO", "IssuerType"])
              .drop_duplicates("PERMNO").set_index("PERMNO")["IssuerType"])
    hdf["IssuerType"] = hdf["PERMNO"].map(issuer)
    hdf = hdf[["PERMNO", "Ticker", "IssuerType", "USIncFlg", "SICCD"]]
    # flag rules of the stored table: foreign = USIncFlg N; REIT by CRSP IssuerType
    # (IRM is labelled CORP by CRSP and so is False here); REIT by SIC 6798
    hdf["is_foreign_USIncN"] = hdf["USIncFlg"].astype(str).str.upper().eq("N")
    hdf["reit_issuertype"] = hdf["IssuerType"].astype(str).str.upper().eq("REIT")
    hdf["reit_sic6798"] = pd.to_numeric(hdf["SICCD"], errors="coerce").eq(6798)
    out["h_absent_total"] = len(hdf)
    out["h_reits"] = int(hdf["reit_sic6798"].sum())
    out["h_foreign"] = int(hdf["is_foreign_USIncN"].sum())
    out["h_foreign_tickers"] = ";".join(hdf.loc[hdf["is_foreign_USIncN"], "Ticker"].astype(str))
    out["h_non_reit_tickers"] = ";".join(hdf.loc[~hdf["reit_sic6798"], "Ticker"].astype(str))
    hdf.to_csv(NC / "item3_h_absent_firms.csv", index=False)
    n_it_reit = int(hdf["reit_issuertype"].sum())
    corp = hdf.loc[~hdf["reit_issuertype"], "Ticker"].astype(str).tolist()
    print(f"  (h) 36-absent: total {len(hdf)}; IssuerType REIT {n_it_reit}, "
          f"CORP {len(corp)} [{';'.join(corp)}]; foreign(USIncFlg=N) {out['h_foreign']} "
          f"[{out['h_foreign_tickers']}]; REITs(SIC 6798) {out['h_reits']}; "
          f"non-REIT by SIC: {out['h_non_reit_tickers']}")

    # ---- (j) rank-transform bounds ----
    pf = pd.read_parquet(PROC / "panel_final.parquet")
    ps = pd.read_parquet(PROC / "panel_structured.parquet")
    pf = pf.sort_values(["week_anchor", "PERMNO"]).reset_index(drop=True)
    ps = ps.sort_values(["week_anchor", "PERMNO"]).reset_index(drop=True)
    CHARS_22 = ['ret_12_1', 'ret_6_1', 'ret_1_0', 'rvol_21d', 'ivol_ff3_21d',
                'market_equity', 'dolvol_126d', 'ami_126d', 'be_me', 'ni_me',
                'sale_me', 'chcsho_12m', 'ope_be', 'niq_be', 'gp_at', 'at_gr1',
                'sale_gr1', 'oaccruals_at', 'noa_at', 'at_be', 'niq_su', 'beta_60m']
    worst = {"min_of_max": 2.0, "max_of_min": -2.0, "max_abs_mean": 0.0}
    n_cells = n_skip = 0
    for wk, idx in pf.groupby("week_anchor").groups.items():
        fw = pf.loc[idx]
        sw = ps.loc[idx]
        for c in CHARS_22:
            mask = sw[c].notna().to_numpy()
            n = int(mask.sum())
            if n < 3:
                n_skip += 1
                continue
            n_cells += 1
            sub = fw[c].to_numpy()[mask]
            worst["min_of_max"] = min(worst["min_of_max"], float(sub.max()))
            worst["max_of_min"] = max(worst["max_of_min"], float(sub.min()))
            worst["max_abs_mean"] = max(worst["max_abs_mean"], abs(float(sub.mean())))
    out["j_cells_checked"] = n_cells
    out["j_cells_skipped_n_lt_3"] = n_skip
    out["j_min_of_per_week_max"] = worst["min_of_max"]
    out["j_max_of_per_week_min"] = worst["max_of_min"]
    out["j_max_abs_within_week_mean"] = worst["max_abs_mean"]
    print(f"  (j) rank transform ({n_cells} cells, {n_skip} skipped n<3): "
          f"per-week max >= {worst['min_of_max']:.6f}, per-week min <= {worst['max_of_min']:.6f}, "
          f"max|mean| {worst['max_abs_mean']:.2e}")

    # ---- (l) midnight-timestamp share ----
    # SHA as recorded in pipeline/15
    SCORING_SHA = "eb42e0e3aa05c7c973e407c722125468bb506f1cc51e629042ab4d0a027ccc39"
    si = pd.read_parquet(PROC / "scoring_input.parquet")
    # Verification (in place): scoring-input SHA256
    assert hashlib.sha256((PROC / "scoring_input.parquet").read_bytes()).hexdigest() == SCORING_SHA
    raw = pd.read_csv(DATA / "fnspid_nasdaq_news_raw.csv",
                      usecols=["Date", "Stock_symbol", "Article_title"])
    raw = raw.dropna(subset=["Date"])
    raw_dt = pd.to_datetime(raw["Date"].astype(str).str.replace(" UTC", "", regex=False),
                            errors="coerce")
    raw = raw.assign(_dt=raw_dt).dropna(subset=["_dt"])
    raw["_midnight"] = (raw["_dt"].dt.hour == 0) & (raw["_dt"].dt.minute == 0) & (raw["_dt"].dt.second == 0)
    out["l_raw_all_midnight_share_pct"] = float(raw["_midnight"].mean() * 100)
    # kept subset: join back on (ticker, date, cleaned title)
    TOK = re.compile(r"\s+")
    def clean(s):
        s = html.unescape(str(s))
        return TOK.sub(" ", s).strip()
    raw["ticker"] = raw["Stock_symbol"].astype(str).str.strip().str.upper()
    raw["d"] = raw["_dt"].dt.date
    raw["htxt"] = raw["Article_title"].map(clean)
    raw_key = raw.drop_duplicates(["ticker", "d", "htxt"])[["ticker", "d", "htxt", "_midnight"]]
    si2 = si.copy()
    si2["ticker"] = si2["ticker"].astype(str).str.strip().str.upper()
    si2["d"] = pd.to_datetime(si2["headline_date"]).dt.date
    si2["htxt"] = si2["headline"].astype(str)
    j = si2.merge(raw_key, on=["ticker", "d", "htxt"], how="left")
    matched = j["_midnight"].notna()
    out["l_kept_n"] = int(len(si2))
    out["l_kept_matched"] = int(matched.sum())
    out["l_kept_midnight_share_pct"] = float(j.loc[matched, "_midnight"].mean() * 100)
    print(f"  (l) raw-all midnight {out['l_raw_all_midnight_share_pct']:.2f}%; "
          f"kept matched {matched.sum():,}/{len(si2):,}, midnight "
          f"{out['l_kept_midnight_share_pct']:.2f}%")

    pd.DataFrame([out]).T.rename(columns={0: "value"}).to_csv(NC / "item3_data_figures.csv")
    pd.DataFrame(lost, columns=["PERMNO", "week_anchor"]).to_csv(
        NC / "item3_a_lost_weeks.csv", index=False)
    print("  wrote item3_data_figures.csv (+ a_lost_weeks, h_absent_firms)")
    # ---- Verification: consistency and reproduction checks ----
    print(f"  anchors: {len(anchors)} ({first_a.date()}..{last_a.date()})")
    print("  (d) SVB daily rows in week 2023-03-06..10:")
    for _, r in inwk.iterrows():
        print(f"      {r['DlyCalDt'].date()} prc {r['DlyPrc']} vol {r['DlyVol']} ret {r['DlyRet']}")
    print(f"      [context] all panel rows with no price on anchor: {skel_no_prc}; "
          f"DlyCap in-week fallback rows: {cap_fallback}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
