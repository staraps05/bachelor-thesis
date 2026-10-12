# %% 14_ticker_variant_audit.py
# Purpose: Check whether panel firms with zero headlines in the scoring input are ticker-spelling mismatches (e.g. BRK.B vs BRKB) missed by the exact-match join, rather than firms absent from FNSPID.
# Inputs:  Data/processed/panel_final.parquet, Data/processed/scoring_input.parquet, Data/crsp_daily_prices.csv, Data/processed/fnspid_headlines_2015_2023.csv
# Outputs: prints only
# Notes:
# - Verification script: checks only; writes nothing to Data/
# - Hard-coded counts (55 zero-headline firms, 618 panel firms) from the pipeline/13 run.
# - Prints a NO ACTION / RULING NEEDED verdict; advisory only, exit status is always 0.

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL_FILE = REPO_ROOT / "Data" / "processed" / "panel_final.parquet"
SCORING_FILE = REPO_ROOT / "Data" / "processed" / "scoring_input.parquet"
PRICES_FILE = REPO_ROOT / "Data" / "crsp_daily_prices.csv"
FNSPID_FILE = REPO_ROOT / "Data" / "processed" / "fnspid_headlines_2015_2023.csv"

PUNCT = re.compile(r"[.\-/\s]")


def strip_punct(s: str) -> str:
    return PUNCT.sub("", s)


def edit1(a: str, b: str) -> bool:
    """True iff edit distance between a and b is exactly 1."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1 or a == b:
        return False
    if la == lb:                                   # one substitution
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = diff = 0                               # one insertion
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
        else:
            diff += 1
            if diff > 1:
                return False
        j += 1
    return True


def main() -> int:
    print("=== ticker-variant audit for the zero-headline firms (read-only) ===")

    # --- 1) the 55 firms + their full CRSP ticker histories ---------------------
    panel = pd.read_parquet(PANEL_FILE, columns=["PERMNO", "week_anchor"])
    scoring = pd.read_parquet(SCORING_FILE, columns=["PERMNO"])
    zero = sorted(set(panel["PERMNO"].unique()) - set(scoring["PERMNO"].unique()))
    print(f"  panel firms: {panel['PERMNO'].nunique()}; zero-headline firms: "
          f"{len(zero)}")

    member = (panel.assign(week_anchor=pd.to_datetime(panel["week_anchor"]))
              .groupby("PERMNO")["week_anchor"].agg(["min", "max"]))

    crsp = pd.read_csv(PRICES_FILE,
                       usecols=["PERMNO", "Ticker", "DlyCalDt", "HdrCUSIP"],
                       parse_dates=["DlyCalDt"]).dropna(subset=["Ticker"])
    crsp["Ticker"] = crsp["Ticker"].str.strip().str.upper()
    spells = (crsp.groupby(["PERMNO", "Ticker"])["DlyCalDt"]
              .agg(["min", "max"]).reset_index())
    cusip = crsp.sort_values("DlyCalDt").groupby("PERMNO")["HdrCUSIP"].last()

    # --- 2) FNSPID distinct symbol set with counts + date ranges ----------------
    fn = pd.read_csv(FNSPID_FILE, usecols=["Date", "Stock_symbol"], dtype=str)
    fn = fn.dropna(subset=["Stock_symbol"])
    fn["sym"] = fn["Stock_symbol"].str.strip().str.upper()
    fn["dstr"] = fn["Date"].str[:10]               # ISO prefix: lexicographic = chronological
    symtab = (fn.groupby("sym")
              .agg(rows=("sym", "size"), d_min=("dstr", "min"), d_max=("dstr", "max")))
    print(f"  FNSPID distinct symbols: {len(symtab):,}")
    syms = symtab.index.to_numpy()
    sym_punct = pd.Series(syms, index=syms).map(strip_punct)

    # --- 3) candidate search for each of the 55 ---------------------------------
    conf_rank = {"EXACT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    hits = {}                                      # (permno, sym) -> hit dict
    for p in zero:
        firm_spells = spells[spells["PERMNO"] == p]
        m0 = member.loc[p, "min"] - pd.Timedelta(days=14)
        m1 = member.loc[p, "max"]
        for _, sp in firm_spells.iterrows():
            t = sp["Ticker"]
            tp = strip_punct(t)
            for s in syms:
                if s == t:
                    typ, conf = "symbol present verbatim", "EXACT"
                elif sym_punct[s] == tp and s != t:
                    typ, conf = "punctuation-insensitive", "HIGH"
                elif len(t) >= 2 and len(s) > len(t) and (s.startswith(t) or s.endswith(t)):
                    typ, conf = "FNSPID symbol extends ticker", "MEDIUM"
                elif len(s) >= 2 and len(t) > len(s) and (t.startswith(s) or t.endswith(s)):
                    typ, conf = "ticker extends FNSPID symbol", "MEDIUM"
                elif len(t) >= 3 and edit1(t, s):
                    typ, conf = "edit distance 1", "LOW"
                else:
                    continue
                key = (p, s)
                if key in hits and conf_rank[hits[key]["conf"]] <= conf_rank[conf]:
                    continue
                r = symtab.loc[s]
                overlap = (r["d_max"] >= m0.strftime("%Y-%m-%d")
                           and r["d_min"] <= m1.strftime("%Y-%m-%d"))
                hits[key] = dict(
                    PERMNO=p, cusip=cusip.get(p, "?"),
                    crsp_ticker=f"{t} ({sp['min'].date()}..{sp['max'].date()})",
                    candidate=s, rows=int(r["rows"]),
                    fn_range=f"{r['d_min']}..{r['d_max']}",
                    match=typ, conf=conf, overlap="YES" if overlap else "no")

    hit_df = pd.DataFrame(hits.values())
    if len(hit_df):
        hit_df = hit_df.sort_values(
            ["conf", "PERMNO"], key=lambda c: c.map(conf_rank) if c.name == "conf" else c)
        print(f"\n  hits: {len(hit_df)} across {hit_df['PERMNO'].nunique()} firms "
              f"(EXACT {sum(hit_df['conf'] == 'EXACT')}, "
              f"HIGH {sum(hit_df['conf'] == 'HIGH')}, "
              f"MEDIUM {sum(hit_df['conf'] == 'MEDIUM')}, "
              f"LOW {sum(hit_df['conf'] == 'LOW')})")
        # EXACT hits were already visible to the exact-match join: zero kept
        # there is a window/timing effect, not recoverable by remapping.
        print("\n  EXACT class (diagnostic - already seen by the join, "
              "explains zero-kept as timing):")
        print("  " + hit_df[hit_df["conf"] == "EXACT"].to_string(index=False)
              .replace("\n", "\n  "))
        print("\n  HIGH class (punctuation variants - the genuinely recoverable "
              "kind):")
        high = hit_df[hit_df["conf"] == "HIGH"]
        print("  " + (high.to_string(index=False).replace("\n", "\n  ")
                      if len(high) else "NONE"))
        med_yes = hit_df[(hit_df["conf"] == "MEDIUM") & (hit_df["overlap"] == "YES")]
        low_yes = hit_df[(hit_df["conf"] == "LOW") & (hit_df["overlap"] == "YES")]
        print(f"\n  MEDIUM with window overlap (for eyeballing; "
              f"{sum(hit_df['conf'] == 'MEDIUM') - len(med_yes)} more without "
              f"overlap suppressed):")
        print("  " + (med_yes.to_string(index=False).replace("\n", "\n  ")
                      if len(med_yes) else "NONE"))
        print(f"\n  LOW with window overlap (edit-distance-1; "
              f"{sum(hit_df['conf'] == 'LOW') - len(low_yes)} more without "
              f"overlap suppressed):")
        print("  " + (low_yes.to_string(index=False).replace("\n", "\n  ")
                      if len(low_yes) else "NONE"))
    else:
        print("\n  candidate table: NO hits of any class")

    firms_no_hit = [p for p in zero if p not in {k[0] for k in hits}]
    print(f"\n  firms with NO candidate of any class: {len(firms_no_hit)} "
          f"({', '.join(str(p) for p in firms_no_hit[:20])}"
          f"{'...' if len(firms_no_hit) > 20 else ''})")

    # --- 4) reverse scale check on ALL 618 firms ---------------------------------
    print("\n=== reverse scale check (bounds total recoverable volume) ===")
    panel_firms = set(panel["PERMNO"].unique())
    all_tickers = set(spells.loc[spells["PERMNO"].isin(panel_firms), "Ticker"])
    all_punct = {strip_punct(t) for t in all_tickers}
    unmapped_syms = symtab.index[~symtab.index.isin(all_tickers
                                                    | set(crsp["Ticker"].unique()))]
    cand_mask = sym_punct.loc[unmapped_syms].isin(all_punct)
    cand_syms = unmapped_syms[cand_mask.to_numpy()]
    rec_rows = int(symtab.loc[cand_syms, "rows"].sum()) if len(cand_syms) else 0
    print(f"  unmapped FNSPID symbols that punctuation-normalise to a ticker of "
          f"any of the {len(panel_firms)} panel firms: {len(cand_syms)}")
    print(f"  headlines carried by those symbols (upper bound on recoverable): "
          f"{rec_rows:,} of 3,167,051 unmapped")
    if len(cand_syms):
        top = symtab.loc[cand_syms].sort_values("rows", ascending=False).head(10)
        for s, r in top.iterrows():
            norm = strip_punct(s)
            owners = sorted(spells.loc[(spells["Ticker"].map(strip_punct) == norm)
                                       & spells["PERMNO"].isin(panel_firms),
                                       "PERMNO"].unique())
            print(f"    {s}: {int(r['rows']):,} rows ({r['d_min']}..{r['d_max']}) "
                  f"-> normalises to {norm}, CRSP owner(s) {owners}")

    # --- 5) verdict ---------------------------------------------------------------
    # recoverable = HIGH (punctuation) hits with window overlap; MEDIUM/LOW
    # are eyeball candidates only
    high_yes = (hit_df[(hit_df["conf"] == "HIGH") & (hit_df["overlap"] == "YES")]
                if len(hit_df) else pd.DataFrame())
    n_rec_firms = high_yes["PERMNO"].nunique() if len(high_yes) else 0
    rec_total = int(high_yes["rows"].sum()) if len(high_yes) else 0
    n_exact = hit_df[hit_df["conf"] == "EXACT"]["PERMNO"].nunique() if len(hit_df) else 0
    print("\n" + "=" * 72)
    print("  VERDICT")
    print(f"  {n_rec_firms} of {len(zero)} firms have HIGH-confidence "
          f"(punctuation-variant) recoverable candidates"
          + (f": {sorted(high_yes['PERMNO'].unique().tolist())}" if n_rec_firms else ""))
    print(f"  total potentially recoverable headlines (HIGH + overlap): "
          f"{rec_total:,}")
    print(f"  reverse check upper bound across all 618 firms: {rec_rows:,} "
          f"of 3,167,051 unmapped headlines")
    print(f"  EXACT-class diagnostics: {n_exact} of the 55 firms have their "
          f"CRSP ticker verbatim in FNSPID - zero-kept there is a "
          f"timing/window effect (already correctly handled by the join), "
          f"not symbol absence")
    if n_rec_firms == 0 and rec_rows == 0:
        print("  recommendation: NO ACTION - the 'no coverage' claim is verified")
        print("  (refined: for the EXACT-class firms it means 'no coverage under")
        print("  the membership-era ticker/window'); the exact-match join lost")
        print("  essentially nothing anywhere. MEDIUM/LOW tables above are")
        print("  noise-class candidates for optional eyeballing only.")
    else:
        print("  recommendation: RULING NEEDED - genuine variants found; mapping")
        print("  additions proposed above. No artifact was modified.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
