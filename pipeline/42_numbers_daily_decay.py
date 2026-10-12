# %% 42_numbers_daily_decay.py
# Purpose: Table C.6: persistence of the sentiment-return relation at daily horizons h = 0..5; reproduces pipeline/29.
# Inputs:  Data/processed/scoring_input.parquet, Data/colab/finbert_scores.csv, Data/colab/fingpt_scores.csv, Data/Loughran-McDonald_MasterDictionary_*.csv, Data/processed/features_lm.parquet, Data/crsp_daily_prices.csv, Data/F-F_Research_Data_Factors_daily.csv
# Outputs: Data/processed/numbers_check/ tableC6_persistence.csv
# Notes:
# - Loads pipeline/15 (TOKEN_RE, DICT_GLOB) and sentiment_aggregation.py via importlib.
# - Input SHA256s checked (pipeline/29); stops unless LM regeneration matches features_lm.
# - Univariate daily test: no weekly lag structure and no structured baseline.

import hashlib
import importlib.util
import math
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


_c15 = _load("lm15", "15_score_lm.py")
TOKEN_RE, DICT_GLOB = _c15.TOKEN_RE, _c15.DICT_GLOB
_agg = _load("sagg", "sentiment_aggregation.py")
aggregate_scores = _agg.aggregate_scores

# SHA values as recorded in pipeline/29
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


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def sha_ok(path, want, name):
    if hashlib.sha256(path.read_bytes()).hexdigest() != want:
        stop(f"{name} SHA mismatch")


def clustered_ols_fast(y, x, cl):
    """OLS slope of y on x and its t-statistic with errors clustered on cl."""
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
    return float(beta[1]), float(beta[1] / math.sqrt(V[1, 1]))


def nw_t(d, lag=NW_LAG):
    """t-statistic of mean(d) with a Newey-West (Bartlett) variance at `lag`."""
    n = len(d)
    u = d - d.mean()
    s = float((u * u).sum()) / n
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * float((u[l:] * u[:-l]).sum()) / n
    return float(d.mean()) / math.sqrt(s / n)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== 42: daily decay (Table C.6) ===")
    # Verification (in place): scoring-input SHA256
    sha_ok(SCORING_FILE, SCORING_SHA, "scoring_input")
    si = pd.read_parquet(SCORING_FILE).reset_index(drop=True)
    # Verification (in place): scoring-input row count
    if len(si) != N_ROWS:
        stop(f"{len(si)} rows")
    si["hid"] = np.arange(len(si), dtype="int64")
    for path, want, col in ((FB_CSV, FB_CSV_SHA, "s_fb"), (FG_CSV, FG_CSV_SHA, "s_fg")):
    # Verification (in place): Colab score file SHA256 (inside loop)
        sha_ok(path, want, path.name)
        sc = pd.read_csv(path)
    # Verification (in place): Colab score rows and hid (inside loop)
        if len(sc) != N_ROWS or sc["hid"].nunique() != N_ROWS:
            stop(f"{path.name} rows/hid")
        si = si.merge(sc.assign(**{col: sc["p_pos"] - sc["p_neg"]})[["hid", col]],
                      on="hid", validate="1:1")
    # LM score per headline, pipeline/15 recipe
    cands = sorted((REPO_ROOT / "Data").glob(DICT_GLOB))
    lmdict = pd.read_csv(cands[-1])
    lmdict["Word"] = lmdict["Word"].astype(str).str.upper()
    NEG = set(lmdict.loc[lmdict["Negative"] > 0, "Word"])
    POS = set(lmdict.loc[lmdict["Positive"] > 0, "Word"])
    toks = si["headline"].str.upper().str.findall(TOKEN_RE)
    ex = toks.explode()
    ex = ex[ex.notna()]
    n_pos = ex.isin(POS).groupby(level=0).sum().reindex(si.index, fill_value=0).astype("int64")
    n_neg = ex.isin(NEG).groupby(level=0).sum().reindex(si.index, fill_value=0).astype("int64")
    tot = n_pos + n_neg
    si["s_lm"] = np.where(tot > 0, (n_pos - n_neg) / tot.where(tot > 0, 1), 0.0)
    # Verification (in place): LM regeneration reproduces features_lm (gates the pairing)
    # gate: weekly re-aggregation must reproduce features_lm.parquet
    sha_ok(LM_FILE, LM_SHA, "features_lm")
    feats = pd.read_parquet(LM_FILE)
    agg_in = si[["PERMNO", "target_week", "s_lm"]].rename(columns={"s_lm": "s"}).copy()
    agg_in["target_week"] = pd.to_datetime(agg_in["target_week"])
    agg = aggregate_scores(agg_in)
    feats["_tw"] = pd.to_datetime(feats["target_week"])
    m = feats.merge(agg.rename(columns={"target_week": "_tw"}), on=["PERMNO", "_tw"], how="left")
    cov = m["mean"].notna()
    ok = (all(np.array_equal(m.loc[cov, f"lm_{c}"].to_numpy(), m.loc[cov, c].to_numpy())
              for c in ("mean", "std", "range", "kurt"))
          and bool((m.loc[~cov, ["lm_mean", "lm_std", "lm_range", "lm_kurt"]].to_numpy() == 0).all())
          and len(m) == 157_897)
    if not ok:
        stop("LM regeneration does not reproduce features_lm bitwise")
    print("  [PASS] LM regen reproduces features_lm.parquet bitwise")

    # firm-day means, then h-day-ahead excess returns
    si["hdate"] = pd.to_datetime(si["headline_date"])
    daily = (si.groupby(["PERMNO", "hdate"])
             .agg(s_lm=("s_lm", "mean"), s_fb=("s_fb", "mean"), s_fg=("s_fg", "mean"))
             .reset_index())
    crsp = pd.read_csv(CRSP_FILE, usecols=["PERMNO", "DlyCalDt", "DlyRet"]).drop_duplicates()
    crsp["DlyCalDt"] = pd.to_datetime(crsp["DlyCalDt"])
    crsp["DlyRet"] = pd.to_numeric(crsp["DlyRet"], errors="coerce")
    ff = pd.read_csv(FF_FILE, skiprows=4)
    ff = ff.rename(columns={ff.columns[0]: "Date"})
    ff["Date"] = pd.to_numeric(ff["Date"], errors="coerce")
    ff = ff.dropna(subset=["Date"])
    ff["Date"] = pd.to_datetime(ff["Date"].astype(int).astype(str))
    ff["RF"] = pd.to_numeric(ff["RF"]) / 100.0
    crsp = crsp.merge(ff[["Date", "RF"]], left_on="DlyCalDt", right_on="Date", how="left")
    # Verification (in place): CRSP dates present in FF
    if crsp["RF"].isna().any():
        stop("CRSP dates missing from FF RF")
    crsp["xret"] = crsp["DlyRet"] - crsp["RF"]
    cal = np.sort(crsp["DlyCalDt"].unique())
    cal_set = set(cal)
    ret = crsp[["PERMNO", "DlyCalDt", "xret"]].rename(columns={"DlyCalDt": "rdate"})
    hd = daily["hdate"].to_numpy()
    pairs = {}
    for h in HORIZONS:
        d = daily.copy()
        if h == 0:
            in_cal = np.array([x in cal_set for x in hd])
            d["rdate"] = np.where(in_cal, hd, np.datetime64("NaT"))
        else:
            idx = np.searchsorted(cal, hd, side="right") + h - 1
            valid = idx < len(cal)
            d["rdate"] = np.where(valid, cal[np.minimum(idx, len(cal) - 1)], np.datetime64("NaT"))
        d = d.dropna(subset=["rdate"])
    # Verification (in place): return date strictly after headline date (inside loop)
        if h >= 1 and not (d["rdate"] > d["hdate"]).all():
            stop(f"h={h}: return date not strictly after headline")
        d = d.merge(ret, on=["PERMNO", "rdate"], how="inner").dropna(subset=["xret"])
        pairs[h] = d

    rows = []
    for col in SCORERS:
        for h in HORIZONS:
            d = pairs[h]
            b, t = clustered_ols_fast(d["xret"].to_numpy(float), d[col].to_numpy(float),
                                      d["rdate"].to_numpy())
            spreads = []
            for _, g in d.groupby("rdate"):
                if len(g) < MIN_FIRMS_DAY:
                    continue
                r = g[col].rank(method="first")
                q = pd.qcut(r, 5, labels=False)
                spreads.append(g.loc[q == 4, "xret"].mean() - g.loc[q == 0, "xret"].mean())
            sp = np.array(spreads)
            rows.append({"scorer": LABELS[col], "h": h, "slope_pct": b * 100,
                         "slope_t": t, "spread_pct": sp.mean() * 100,
                         "spread_nw_t": nw_t(sp), "n_firmdays": len(d), "n_days": len(sp)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tableC6_persistence.csv", index=False)
    for col in SCORERS:
        for h in (0, 1):
            r = next(x for x in rows if x["scorer"] == LABELS[col] and x["h"] == h)
            print(f"    {LABELS[col]:<8} h={h} slope {r['slope_pct']:+.3f} (t {r['slope_t']:+.2f}) "
                  f"spread {r['spread_pct']:+.3f} (NW t {r['spread_nw_t']:+.2f})")

    # ---- Verification: consistency and reproduction checks ----
    print("  wrote tableC6_persistence.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
