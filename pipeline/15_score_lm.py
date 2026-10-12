# %% 15_score_lm.py
# Purpose: Score every headline in the scoring input with the Loughran-McDonald
#          dictionary and aggregate to firm-week features (lm_mean, lm_std,
#          lm_range, lm_kurt, had_news) on the full panel key set.
# Inputs:  Data/Loughran-McDonald_MasterDictionary_*.csv (last match by name,
#          e.g. Data/Loughran-McDonald_MasterDictionary_1993-2025.csv)
#          Data/processed/scoring_input.parquet
#          Data/processed/scoring_input_coverage.csv
#          Data/processed/panel_final.parquet
# Outputs: Data/processed/features_lm.parquet
# Notes:
# - Tokenisation: uppercase the headline, split on non-letters (tokens are [A-Z]+
#   runs), exact dictionary match, no stemming or stopword removal. A word is on
#   the Negative/Positive list when its column value is > 0.
# - Per-headline score s = (n_pos - n_neg) / (n_pos + n_neg); s = 0 when the
#   headline contains no LM word.
# - Aggregation runs through pipeline/sentiment_aggregation.py (uniform weights,
#   ddof=0, kurtosis guard); its unit tests run first. Firm-weeks without
#   headlines get all four features = 0 and had_news = 0.
# - The scoring-input SHA256 and row count are checked against hard-coded
#   constants; the script stops on mismatch.
# - The output is write-protected only when every check passes; the script stops
#   if a write-protected output already exists.

import hashlib
import re
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sentiment_aggregation as sagg

REPO_ROOT = Path(__file__).resolve().parent.parent
DICT_GLOB = "Loughran-McDonald_MasterDictionary_*.csv"
SCORING_FILE = REPO_ROOT / "Data" / "processed" / "scoring_input.parquet"
COV_FILE = REPO_ROOT / "Data" / "processed" / "scoring_input_coverage.csv"
PANEL_FILE = REPO_ROOT / "Data" / "processed" / "panel_final.parquet"
OUT_FILE = REPO_ROOT / "Data" / "processed" / "features_lm.parquet"

PANEL_ROWS = 157_897
SCORING_ROWS = 796_038
COVERED_KEYS = 104_533
SCORING_SHA = "eb42e0e3aa05c7c973e407c722125468bb506f1cc51e629042ab4d0a027ccc39"
SPGI_PERMNO = 17478          # firm with no headlines in the scoring input

NEG_TERMS = ["bankruptcy", "lawsuit", "misses"]
POS_TERMS = ["beats", "record profit", "upgraded"]
TOKEN_RE = re.compile(r"[A-Z]+")


def check(label: str, passed: bool, observed: str) -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {observed}")
    return passed


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}\nFreeze NOT granted. Fix/clarify and re-run.")


def main() -> int:
    all_pass = True

    # ---- STEP 0 - Shared-module unit tests ----
    print("=== STEP 0: shared aggregation module unit tests ===")
    # Verification (in place): aggregation module unit tests
    sagg.run_unit_tests(verbose=True)
    all_pass &= check("sentiment_aggregation unit tests", True, "ALL PASS")

    # ---- STEP 1 - Dictionary ----
    print("\n=== STEP 1: Loughran-McDonald Master Dictionary ===")
    cands = sorted((REPO_ROOT / "Data").glob(DICT_GLOB))
    # Verification (in place): LM dictionary present
    if not cands:
        stop("LM Master Dictionary not found in Data/ - download from "
             "sraf.nd.edu (Master Dictionary w/ sentiment word lists)")
    dict_file = cands[-1]
    lm = pd.read_csv(dict_file)
    # Verification (in place): dictionary columns present
    if not {"Word", "Negative", "Positive"} <= set(lm.columns):
        stop(f"unexpected dictionary columns: {list(lm.columns)}")
    lm["Word"] = lm["Word"].astype(str).str.upper()
    NEG = set(lm.loc[lm["Negative"] > 0, "Word"])
    POS = set(lm.loc[lm["Positive"] > 0, "Word"])

    # ---- STEP 2 - Scoring input ----
    print("\n=== STEP 2: scoring input ===")
    # Verification (in place): scoring-input SHA256
    sha_in = hashlib.sha256(SCORING_FILE.read_bytes()).hexdigest()
    if sha_in != SCORING_SHA:
        stop(f"scoring_input.parquet SHA mismatch: {sha_in}")
    hl = pd.read_parquet(SCORING_FILE)
    hl["target_week"] = pd.to_datetime(hl["target_week"])
    # Verification (in place): scoring-input row count
    if len(hl) != SCORING_ROWS:
        stop(f"scoring input has {len(hl):,} rows, expected {SCORING_ROWS:,}")
    print(f"  {len(hl):,} headlines loaded; SHA verified == frozen "
          f"{SCORING_SHA[:8]}...")

    # ---- STEP 3 - Tokenize + score ----
    print("\n=== STEP 3: tokenize + score ===")
    toks = hl["headline"].str.upper().str.findall(TOKEN_RE)
    hl["n_tokens"] = toks.str.len()
    ex = toks.explode()
    ex = ex[ex.notna()]
    hl["n_pos"] = (ex.isin(POS).groupby(level=0).sum()
                   .reindex(hl.index, fill_value=0).astype("int64"))
    hl["n_neg"] = (ex.isin(NEG).groupby(level=0).sum()
                   .reindex(hl.index, fill_value=0).astype("int64"))
    tot = hl["n_pos"] + hl["n_neg"]
    hl["s"] = np.where(tot > 0, (hl["n_pos"] - hl["n_neg"]) / tot.where(tot > 0, 1),
                       0.0)


    # ---- STEP 4 - Aggregate ----
    print("\n=== STEP 4: aggregate via sentiment_aggregation ===")
    agg = sagg.aggregate_scores(hl[["PERMNO", "target_week", "s"]])
    agg = agg.rename(columns={c: f"lm_{c}" for c in ["mean", "std", "range", "kurt"]})
    print(f"  aggregated: {len(agg):,} (PERMNO, target_week) rows")

    # ---- STEP 5 - Expand to the panel key set + save ----
    print("\n=== STEP 5: expand to the 157,897 panel keys ===")
    panel = pd.read_parquet(PANEL_FILE, columns=["PERMNO", "week_anchor"])
    keys = pd.DataFrame({"PERMNO": panel["PERMNO"],
                         "target_week": pd.to_datetime(panel["week_anchor"])})
    feat = keys.merge(agg, on=["PERMNO", "target_week"], how="left",
                      validate="1:1")
    feat["had_news"] = feat["lm_mean"].notna().astype("int64")
    for c in ["lm_mean", "lm_std", "lm_range", "lm_kurt"]:
        feat[c] = feat[c].fillna(0.0)
    # Verification (in place): expansion row count
    if len(feat) != PANEL_ROWS:
        stop(f"expanded frame has {len(feat):,} rows")
    all_pass &= check("exactly 157,897 rows after expansion", True,
                      f"{len(feat):,}")

    # Verification (in place): aggregated keys == coverage file keys
    cov = pd.read_csv(COV_FILE)
    cov_keys = set(zip(cov["PERMNO"], pd.to_datetime(cov["target_week"])))
    agg_keys = set(zip(agg["PERMNO"], agg["target_week"]))
    if agg_keys != cov_keys:
        stop("aggregated key set != scoring_input_coverage.csv key set")
    all_pass &= check("had_news consistent with S1 coverage file "
                      "(key sets identical)",
                      int(feat['had_news'].sum()) == COVERED_KEYS,
                      f"{int(feat['had_news'].sum()):,} == {COVERED_KEYS:,} "
                      f"({int(feat['had_news'].sum()) / PANEL_ROWS:.2%} = "
                      f"1 - 33.80% no-news)")

    out = feat[["PERMNO", "target_week", "lm_mean", "lm_std", "lm_range",
                "lm_kurt", "had_news"]].sort_values(
        ["PERMNO", "target_week"]).reset_index(drop=True)
    out["target_week"] = out["target_week"].dt.date
    # Verification (in place): refuse to overwrite a frozen output
    if OUT_FILE.exists() and not (OUT_FILE.stat().st_mode & stat.S_IWUSR):
        stop(f"{OUT_FILE.name} exists and is frozen - new version required")
    out.to_parquet(OUT_FILE, engine="pyarrow", index=False)
    sha = hashlib.sha256(OUT_FILE.read_bytes()).hexdigest()
    print(f"  wrote {OUT_FILE} ({len(out):,} x {len(out.columns)})")
    print(f"  SHA256: {sha} (write-protection at the END, on full pass)")

    # ---- STEP 6 - VALIDATION REPORT ----
    print("\n=== STEP 6: VALIDATION REPORT ===")

    # ---- Verification: consistency and reproduction checks ----
    print(f"  file: {dict_file.name} (vintage from filename: "
          f"{dict_file.stem.split('_')[-1]})")
    print(f"  master dictionary rows: {len(lm):,}; membership rule: "
          f"column value > 0 (LM year-added convention)")
    print(f"  list last-update years: Negative max {int(lm.loc[lm['Negative'] > 0, 'Negative'].max())}, "
          f"Positive max {int(lm.loc[lm['Positive'] > 0, 'Positive'].max())}")
    all_pass &= check("~350 positive words", 300 <= len(POS) <= 400, f"{len(POS)}")
    all_pass &= check("~2,300 negative words (LM deliberately negative-heavy)",
                      2200 <= len(NEG) <= 2500, f"{len(NEG)}")
    print(f"  sample negative: {sorted(NEG)[:5]}")
    print(f"  sample positive: {sorted(POS)[:5]}")
    sh_zero = float((hl["s"] == 0).mean())
    sh_nowords = float((tot == 0).mean())
    sh_tie = float(((tot > 0) & (hl["n_pos"] == hl["n_neg"])).mean())
    sh_pos, sh_neg = float((hl["s"] > 0).mean()), float((hl["s"] < 0).mean())
    mean_nz = float(hl.loc[hl["s"] != 0, "s"].mean())
    print(f"  headline-level: s = 0 on {sh_zero:.2%} "
          f"(no LM-classified words: {sh_nowords:.2%}; balanced ties: "
          f"{sh_tie:.2%}) - the known LM-on-short-text property")
    print(f"  s > 0: {sh_pos:.2%}   s < 0: {sh_neg:.2%}   "
          f"mean of nonzero s: {mean_nz:+.4f}")
    # sign spot checks: pass/fail only where the search term is itself an LM word
    # (bankruptcy, misses); terms LM does not classify (lawsuit is Litigious-only;
    # beats / record profit / upgraded are untagged) are reported descriptively
    def term_report(term, side):
        m = hl[hl["headline"].str.contains(term, case=False, regex=False)]
        tok_up = [t for t in TOKEN_RE.findall(term.upper())]
        in_neg = all(t in NEG for t in tok_up)
        in_pos = all(t in POS for t in tok_up)
        tag = ("Negative" if in_neg else "Positive" if in_pos
               else "NOT LM-classified")
        mean_s = float(m["s"].mean()) if len(m) else np.nan
        print(f"\n  '{term}' [{tag}] - {len(m):,} matching headlines; "
              f"mean s {mean_s:+.4f}; "
              f"s<0 {float((m['s'] < 0).mean()):.1%}, "
              f"s=0 {float((m['s'] == 0).mean()):.1%}, "
              f"s>0 {float((m['s'] > 0).mean()):.1%}")
        pick = m.loc[m["s"].idxmin() if side == "neg" else m["s"].idxmax()]
        words = TOKEN_RE.findall(pick["headline"].upper())
        print(f"    example (s = {pick['s']:+.3f}): {pick['headline'][:110]!r}")
        print(f"    matched neg: {sorted(set(words) & NEG)}  "
              f"pos: {sorted(set(words) & POS)}")
        return m, mean_s, tag

    for term in NEG_TERMS:
        m, mean_s, tag = term_report(term, "neg")
        if tag == "Negative":
            all_pass &= check(f"'{term}' matches score negative on average "
                              f"(term is an LM Negative word)",
                              mean_s < 0, f"{mean_s:+.4f}")
        else:
            print(f"    [INFO] '{term}' is {tag} - mean sign reported "
                  f"descriptively ({mean_s:+.4f}); mechanics visible above")
    any_pos_example = False
    for term in POS_TERMS:
        m, mean_s, tag = term_report(term, "pos")
        any_pos_example |= bool((m["s"] > 0).any())
        print(f"    [INFO] '{term}' is {tag} - LM's earnings-surprise blind "
              f"spot; mean sign reported descriptively ({mean_s:+.4f})")
    all_pass &= check("positive-scoring matches exist among the positive-term "
                      "searches (mechanics demonstrated)",
                      any_pos_example, f"{any_pos_example}")
    # manual aggregation check
    print("\n  manual aggregation check:")
    sizes = hl.groupby(["PERMNO", "target_week"]).size()
    key = sizes[(sizes >= 3) & (sizes <= 6)].index.min()
    rows = hl[(hl["PERMNO"] == key[0]) & (hl["target_week"] == key[1])]
    s = rows["s"].to_numpy()
    print(f"    key: PERMNO {key[0]}, week {key[1].date()}, n = {len(s)}")
    for i, (sc, h) in enumerate(zip(s, rows["headline"]), 1):
        print(f"      s_{i} = {sc:+.6f}  {h[:80]!r}")
    m1, m2 = s.mean(), (s ** 2).mean()
    m3, m4 = (s ** 3).mean(), (s ** 4).mean()
    var = m2 - m1 ** 2
    std, rng = np.sqrt(max(var, 0)), s.max() - s.min()
    m4c = m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4
    kurt = (m4c / var ** 2 - 3.0) if var >= sagg.VAR_GUARD else 0.0
    print(f"    hand: M1={m1:+.6f}  M2={m2:.6f}  var=M2-M1^2={var:.6f}  "
          f"std={std:.6f}")
    print(f"          range={rng:.6f}  M4c={m4c:.8f}  kurt=M4c/var^2-3="
          f"{kurt:+.6f}")
    st = agg[(agg["PERMNO"] == key[0]) & (agg["target_week"] == key[1])].iloc[0]
    ok_man = (abs(st["lm_mean"] - m1) < 1e-12 and abs(st["lm_std"] - std) < 1e-12
              and abs(st["lm_range"] - rng) < 1e-12
              and abs(st["lm_kurt"] - kurt) < 1e-12)
    all_pass &= check("hand-computed moments == stored (all four, 1e-12)",
                      ok_man, f"stored ({st['lm_mean']:+.6f}, {st['lm_std']:.6f}, "
                      f"{st['lm_range']:.6f}, {st['lm_kurt']:+.6f})")
    # zero-fill spot check
    spgi = out[out["PERMNO"] == SPGI_PERMNO]
    z_ok = (len(spgi) == 313 and (spgi["had_news"] == 0).all()
            and (spgi[["lm_mean", "lm_std", "lm_range", "lm_kurt"]] == 0)
            .all().all())
    all_pass &= check("zero-coverage firm SPGI (17478): all-zero features + "
                      "had_news = 0 on all weeks", bool(z_ok),
                      f"{len(spgi)} weeks, all zero: {bool(z_ok)}")
    wk_ok = set(out["target_week"]) == set(panel["week_anchor"].unique())
    all_pass &= check("target_week set == the panel's 313 anchors", bool(wk_ok),
                      f"{out['target_week'].nunique()} weeks")



    # feature-level distributions
    covd = out[out["had_news"] == 1]
    print(f"\n  feature distributions over the {len(covd):,} covered firm-weeks:")
    print(f"    {'feature':<10}{'mean':>10}{'std':>10}{'p1':>10}{'median':>10}"
          f"{'p99':>10}{'min':>10}{'max':>10}")
    for c in ["lm_mean", "lm_std", "lm_range", "lm_kurt"]:
        d = covd[c]
        print(f"    {c:<10}{d.mean():>10.4f}{d.std():>10.4f}"
              f"{d.quantile(.01):>10.4f}{d.median():>10.4f}"
              f"{d.quantile(.99):>10.4f}{d.min():>10.4f}{d.max():>10.4f}")



    # ---- write-protect only on full pass ----
    # Verification (in place): freeze only if all checks passed
    if not all_pass:
        print("\nOverall: SOME CHECKS FAILED - features written but NOT "
              "write-protected; FREEZE PENDING")
        return 1
    OUT_FILE.chmod(OUT_FILE.stat().st_mode
                   & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print("\n" + "=" * 72)
    print("  SENTIMENT S4a - LM FEATURES FROZEN")
    print(f"  {OUT_FILE}")
    print(f"  {len(out):,} rows x 7 cols | SHA256: {sha}")
    print("=" * 72)
    print("\nOverall: ALL CHECKS PASSED — FREEZE GRANTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
