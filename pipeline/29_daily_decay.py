# %% 29_daily_decay.py
# Purpose: Diagnostic on how fast per-headline sentiment predicts excess returns
#          at daily resolution, for horizons h = 0..5 trading days after the
#          headline date, with LM, FinBERT and FinGPT scores. Univariate only.
# Inputs:  Data/processed/scoring_input.parquet
#          Data/colab/finbert_scores.csv
#          Data/colab/fingpt_scores.csv
#          Data/Loughran-McDonald_MasterDictionary_*.csv (last match by name)
#          Data/processed/features_lm.parquet
#          Data/crsp_daily_prices.csv
#          Data/F-F_Research_Data_Factors_daily.csv
# Outputs: prints only
# Notes:
# - Loads pipeline/15_score_lm.py (TOKEN_RE, DICT_GLOB) and
#   pipeline/sentiment_aggregation.py (aggregate_scores) via importlib.
# - LM scores are regenerated per headline with pipeline/15's recipe; the run
#   stops unless their weekly re-aggregation reproduces features_lm.parquet
#   bit-exactly. FinBERT/FinGPT scores are p_pos - p_neg from the Colab CSVs.
# - Scores are averaged per firm-day. h = 0 pairs a headline day with its own
#   session (non-trading dates dropped); h >= 1 pairs it with the h-th session
#   strictly after the headline date, so weekend headlines roll forward.
# - Two statistics per (scorer, h): pooled OLS slope with date-clustered t, and
#   the daily Q5-Q1 quintile spread (days with >= 20 firms) with a NW(5) t.
# - The test is univariate: it says nothing about value added on top of the
#   structured baseline.

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC = REPO_ROOT / "Data" / "processed"

_s15 = importlib.util.spec_from_file_location(
    "lm15", REPO_ROOT / "pipeline" / "15_score_lm.py")
_c15 = importlib.util.module_from_spec(_s15)
_s15.loader.exec_module(_c15)
TOKEN_RE, DICT_GLOB = _c15.TOKEN_RE, _c15.DICT_GLOB

_sagg = importlib.util.spec_from_file_location(
    "sagg", REPO_ROOT / "pipeline" / "sentiment_aggregation.py")
_agg = importlib.util.module_from_spec(_sagg)
_sagg.loader.exec_module(_agg)
aggregate_scores = _agg.aggregate_scores

# SHA256 constants: scoring input as published by pipeline/15, features_lm as
# by pipeline/19; the two Colab CSV hashes are recorded only here.
SCORING_FILE = PROC / "scoring_input.parquet"
SCORING_SHA = "eb42e0e3aa05c7c973e407c722125468bb506f1cc51e629042ab4d0a027ccc39"
LM_FILE = PROC / "features_lm.parquet"
LM_SHA = "b07a8f1e236d34d6d68dabd1038d1dfc632f6a2b220f4ccf9e3af88e0a7d50a3"
FB_CSV = REPO_ROOT / "Data" / "colab" / "finbert_scores.csv"
FB_CSV_SHA = "e39307f52d262426e4579b315d8c4cf9fee03cd6888d3bdb87e48844987f8731"
FG_CSV = REPO_ROOT / "Data" / "colab" / "fingpt_scores.csv"
FG_CSV_SHA = "0daee9398a6714229a83ee53f12ac157469de572725a5b0abaa6ebc632883207"
CRSP_FILE = REPO_ROOT / "Data" / "crsp_daily_prices.csv"
FF_FILE = REPO_ROOT / "Data" / "F-F_Research_Data_Factors_daily.csv"

N_ROWS = 796_038
HORIZONS = [0, 1, 2, 3, 4, 5]
SCORERS = ["s_lm", "s_fb", "s_fg"]
LABELS = {"s_lm": "LM", "s_fb": "FinBERT", "s_fg": "FinGPT"}
MIN_FIRMS_DAY = 20
NW_LAG = 5


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def sha_ok(path: Path, want: str, name: str) -> None:
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        stop(f"{name} SHA mismatch: {got}")


def norm_p(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2.0))


def clustered_ols_fast(y: np.ndarray, x: np.ndarray, cl: np.ndarray):
    """Univariate OLS with intercept; cluster-robust t from per-cluster score
    sums (same estimator as verification/23's clustered_ols)."""
    Z = np.column_stack([np.ones(len(x)), x])
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    beta = ZtZ_inv @ (Z.T @ y)
    e = y - Z @ beta
    codes, _ = pd.factorize(cl)
    se = np.bincount(codes, weights=e)
    sxe = np.bincount(codes, weights=x * e)
    meat = np.array([[np.sum(se * se), np.sum(se * sxe)],
                     [np.sum(se * sxe), np.sum(sxe * sxe)]])
    V = ZtZ_inv @ meat @ ZtZ_inv
    t = beta[1] / math.sqrt(V[1, 1])
    return float(beta[1]), float(t)


def nw_t(d: np.ndarray, lag: int = NW_LAG) -> float:
    n = len(d)
    u = d - d.mean()
    s = float((u * u).sum()) / n
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * float((u[l:] * u[:-l]).sum()) / n
    return float(d.mean()) / math.sqrt(s / n)


def main() -> int:
    print("=== STEP 1: rebuild per-headline scores (hard join asserts) ===")
    # Verification (in place): scoring-input SHA256
    sha_ok(SCORING_FILE, SCORING_SHA, "scoring_input.parquet")
    si = pd.read_parquet(SCORING_FILE)
    # Verification (in place): scoring-input row count
    if len(si) != N_ROWS:
        stop(f"{len(si):,} rows, expected {N_ROWS:,}")
    si = si.reset_index(drop=True)
    si["hid"] = np.arange(len(si), dtype="int64")
    # Verification (in place): scoring-input summary
    print(f"  [PASS] scoring_input: {len(si):,} rows, SHA == frozen; "
          f"hid = positional 0..{N_ROWS - 1}")

    # Verification (in place): FinBERT/FinGPT score files: SHA, hid, NaN, row sums (inside loop)
    for path, want, col in ((FB_CSV, FB_CSV_SHA, "s_fb"),
                            (FG_CSV, FG_CSV_SHA, "s_fg")):
        sha_ok(path, want, path.name)
        sc = pd.read_csv(path)
        if len(sc) != N_ROWS or sc["hid"].nunique() != N_ROWS:
            stop(f"{path.name}: rows/hid not 1:1 complete")
        if sc[["p_pos", "p_neu", "p_neg"]].isna().any().any():
            stop(f"{path.name}: NaN probabilities")
        dev = (sc[["p_pos", "p_neu", "p_neg"]].sum(axis=1) - 1.0).abs().max()
        if dev > 1e-4:
            stop(f"{path.name}: row sums deviate up to {dev:.2e}")
        n0 = len(si)
        si = si.merge(sc.assign(**{col: sc["p_pos"] - sc["p_neg"]})
                      [["hid", col]], on="hid", validate="1:1")
        if len(si) != n0:
            stop(f"{path.name}: join changed row count")
        print(f"  [PASS] {path.name}: SHA == frozen, 1:1 on hid, 0 unmatched,"
              f" 0 NaN, row sums within {dev:.1e}; s = p_pos - p_neg")

    # ---- LM per-headline scores, pipeline/15 recipe ----
    cands = sorted((REPO_ROOT / "Data").glob(DICT_GLOB))
    # Verification (in place): LM dictionary present
    if not cands:
        stop(f"LM dictionary missing ({DICT_GLOB})")
    lmdict = pd.read_csv(cands[-1])
    lmdict["Word"] = lmdict["Word"].astype(str).str.upper()
    NEG = set(lmdict.loc[lmdict["Negative"] > 0, "Word"])
    POS = set(lmdict.loc[lmdict["Positive"] > 0, "Word"])
    toks = si["headline"].str.upper().str.findall(TOKEN_RE)
    ex = toks.explode()
    ex = ex[ex.notna()]
    n_pos = (ex.isin(POS).groupby(level=0).sum()
             .reindex(si.index, fill_value=0).astype("int64"))
    n_neg = (ex.isin(NEG).groupby(level=0).sum()
             .reindex(si.index, fill_value=0).astype("int64"))
    tot = n_pos + n_neg
    si["s_lm"] = np.where(tot > 0, (n_pos - n_neg) / tot.where(tot > 0, 1),
                          0.0)
    print(f"  LM regenerated from {cands[-1].name} (pos {len(POS)}, neg "
          f"{len(NEG)}); zero-share {(si['s_lm'] == 0).mean():.2%}")

    # ---- weekly re-aggregation must reproduce features_lm ----
    # Verification (in place): features_lm SHA256
    sha_ok(LM_FILE, LM_SHA, "features_lm.parquet")
    feats = pd.read_parquet(LM_FILE)
    agg_in = si[["PERMNO", "target_week", "s_lm"]].rename(
        columns={"s_lm": "s"}).copy()
    agg_in["target_week"] = pd.to_datetime(agg_in["target_week"])
    agg = aggregate_scores(agg_in)
    feats["_tw"] = pd.to_datetime(feats["target_week"])
    m = feats.merge(agg.rename(columns={"target_week": "_tw"}),
                    on=["PERMNO", "_tw"], how="left")
    cov = m["mean"].notna()
    ok_cov = all(np.array_equal(m.loc[cov, f"lm_{c}"].to_numpy(),
                                m.loc[cov, c].to_numpy())
                 for c in ("mean", "std", "range", "kurt"))
    ok_unc = bool((m.loc[~cov, ["lm_mean", "lm_std", "lm_range", "lm_kurt"]]
                   .to_numpy() == 0).all())
    ok_hn = bool((m["had_news"].to_numpy() == cov.astype(int).to_numpy())
                 .all())
    # Verification (in place): weekly re-aggregation reproduces features_lm
    if not (ok_cov and ok_unc and ok_hn and len(m) == 157_897):
        stop(f"LM regeneration does NOT reproduce features_lm exactly "
             f"(covered {ok_cov}, uncovered-zeros {ok_unc}, had_news {ok_hn})"
             f" - regeneration unusable, no fix attempted")
    print(f"  [PASS] weekly re-aggregation of the regenerated LM scores "
          f"reproduces features_lm.parquet EXACTLY (bitwise on all four "
          f"moments, {int(cov.sum()):,} covered keys; zeros + had_news on "
          f"{int((~cov).sum()):,} uncovered) - regeneration is faithful")

    print("\n  spot-check (top 5 negative / positive per scorer):")
    for col in SCORERS:
        srt = si.sort_values(col)
        print(f"  -- {LABELS[col]}:")
        for _, r in srt.head(5).iterrows():
            print(f"    {r[col]:+.3f} ({r['ticker']}) "
                  f"{str(r['headline'])[:85]}")
        for _, r in srt.tail(5).iterrows():
            print(f"    {r[col]:+.3f} ({r['ticker']}) "
                  f"{str(r['headline'])[:85]}")

    print("\n=== STEP 2: daily aggregation + trading-day alignment ===")
    si["hdate"] = pd.to_datetime(si["headline_date"])
    daily = (si.groupby(["PERMNO", "hdate"])
             .agg(s_lm=("s_lm", "mean"), s_fb=("s_fb", "mean"),
                  s_fg=("s_fg", "mean"), n_headlines=("hid", "size"))
             .reset_index())
    print(f"  {len(daily):,} firm-days with >= 1 headline "
          f"(from {N_ROWS:,} headlines); "
          f"{(pd.to_datetime(daily['hdate']).dt.weekday >= 5).mean():.1%} "
          f"fall on weekends")

    crsp = pd.read_csv(CRSP_FILE, usecols=["PERMNO", "DlyCalDt", "DlyRet"])
    crsp = crsp.drop_duplicates()          # byte-identical duplicate rows
    crsp["DlyCalDt"] = pd.to_datetime(crsp["DlyCalDt"])
    crsp["DlyRet"] = pd.to_numeric(crsp["DlyRet"], errors="coerce")
    ff = pd.read_csv(FF_FILE, skiprows=4)
    ff = ff.rename(columns={ff.columns[0]: "Date"})
    ff["Date"] = pd.to_numeric(ff["Date"], errors="coerce")
    ff = ff.dropna(subset=["Date"])
    ff["Date"] = pd.to_datetime(ff["Date"].astype(int).astype(str))
    ff["RF"] = pd.to_numeric(ff["RF"]) / 100.0     # percent -> decimal
    rf_2023 = float(ff.loc[ff["Date"] == "2023-06-09", "RF"].iloc[0])
    # Verification (in place): RF units
    print(f"  RF units check: daily RF(2023-06-09) = {rf_2023:.6f} "
          f"(DECIMAL ~2e-4, not percent ~2e-2) -> "
          f"{'OK' if 1e-4 < rf_2023 < 3e-4 else 'UNIT PROBLEM'}")
    if not 1e-4 < rf_2023 < 3e-4:
        stop("RF scaling trap")
    crsp = crsp.merge(ff[["Date", "RF"]], left_on="DlyCalDt",
                      right_on="Date", how="left")
    # Verification (in place): CRSP dates present in FF
    if crsp["RF"].isna().any():
        stop("CRSP dates missing from FF RF")
    crsp["xret"] = crsp["DlyRet"] - crsp["RF"]
    cal = np.sort(crsp["DlyCalDt"].unique())
    cal_set = set(cal)
    print(f"  trading calendar from CRSP: {len(cal):,} sessions "
          f"{pd.Timestamp(cal[0]).date()}..{pd.Timestamp(cal[-1]).date()}")

    ret = crsp[["PERMNO", "DlyCalDt", "xret"]].rename(
        columns={"DlyCalDt": "rdate"})
    hd = daily["hdate"].to_numpy()
    pairs = {}
    print(f"  {'h':>2} {'mapped':>8} {'drop:nontrad':>12} {'drop:cal-end':>12} "
          f"{'drop:no-ret':>11} {'used n':>8}")
    for h in HORIZONS:
        d = daily.copy()
        if h == 0:
            in_cal = np.array([x in cal_set for x in hd])
            d["rdate"] = np.where(in_cal, hd, np.datetime64("NaT"))
            n_nontrad = int((~in_cal).sum())
            n_end = 0
        else:
            idx = np.searchsorted(cal, hd, side="right") + h - 1
            valid = idx < len(cal)
            d["rdate"] = np.where(valid, cal[np.minimum(idx, len(cal) - 1)],
                                  np.datetime64("NaT"))
            n_nontrad = 0
            n_end = int((~valid).sum())
        d = d.dropna(subset=["rdate"])
    # Verification (in place): return date strictly after headline date (inside loop)
        if h >= 1 and not (d["rdate"] > d["hdate"]).all():
            stop(f"h={h}: a return date is not strictly after its headline "
                 f"date")
        n0 = len(d)
        d = d.merge(ret, on=["PERMNO", "rdate"], how="inner")
        n_noret = n0 - len(d)
        d = d.dropna(subset=["xret"])
        n_noret += n0 - n_noret - len(d)
        pairs[h] = d
        print(f"  {h:>2} {len(d):>8,} {n_nontrad:>12,} {n_end:>12,} "
              f"{n_noret:>11,} {len(d):>8,}")

    print("\n=== STEP 3: the decay curve ===")
    res = {}
    for col in SCORERS:
        for h in HORIZONS:
            d = pairs[h]
            b, t = clustered_ols_fast(d["xret"].to_numpy(float),
                                      d[col].to_numpy(float),
                                      d["rdate"].to_numpy())
            spreads = []
            for _, g in d.groupby("rdate"):
                if len(g) < MIN_FIRMS_DAY:
                    continue
                r = g[col].rank(method="first")
                q = pd.qcut(r, 5, labels=False)
                spreads.append(g.loc[q == 4, "xret"].mean()
                               - g.loc[q == 0, "xret"].mean())
            sp = np.array(spreads)
            ts = nw_t(sp)
            res[(col, h)] = (b, t, len(d), sp.mean(), ts, len(sp))
    print(f"  (a) pooled OLS b (% per unit score) [date-clustered t] | "
          f"(b) Q5-Q1 spread %/day [NW({NW_LAG}) t]")
    for col in SCORERS:
        print(f"  -- {LABELS[col]}:")
        print(f"  {'h':>3} {'b %':>9} {'t':>7} {'n':>9} | {'spread %':>9} "
              f"{'NW t':>7} {'days':>6}")
        for h in HORIZONS:
            b, t, n, spm, ts, nd = res[(col, h)]
            star = " ***" if abs(t) >= 2 and abs(ts) >= 2 else \
                ("  * " if (abs(t) >= 2) != (abs(ts) >= 2) else "")
            print(f"  {h:>3} {b * 100:>+9.4f} {t:>+7.2f} {n:>9,} | "
                  f"{spm * 100:>+9.4f} {ts:>+7.2f} {nd:>6}{star}")

    # ---- Verification: consistency and reproduction checks ----
    print("  [INFO] FULL-DAY LAG BY DESIGN: ~79% of headlines are stamped "
          "00:00:00 (intraday position unknown) -> h=1 means ALL of day t's "
          "headlines vs the NEXT trading day's return; look-ahead-free "
          "without intraday timestamps.")
    print("  [PASS] h>=1 mapping strictly forward (weekend/holiday headlines"
          " roll to the next session); drops = non-trading h=0 dates, "
          "calendar end, or no CRSP return for that firm-day")

    print("\n=== STEP 4: verdict + STOP ===")
    for col in SCORERS:
        b0, t0, _, sp0, ts0, _ = res[(col, 0)]
        sig_h = [h for h in HORIZONS[1:]
                 if abs(res[(col, h)][1]) >= 2 or abs(res[(col, h)][4]) >= 2]
        first_dead = next((h for h in HORIZONS[1:]
                           if abs(res[(col, h)][1]) < 2
                           and abs(res[(col, h)][4]) < 2), None)
        print(f"  {LABELS[col]:<8} h=0 (same-day): b {b0 * 100:+.4f}% "
              f"(t {t0:+.1f}), spread {sp0 * 100:+.4f}%/day (NW t {ts0:+.1f})"
              f" | h>=1 significant at: "
              f"{sig_h if sig_h else 'NONE'} | first fully-insignificant "
              f"horizon: h={first_dead}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
