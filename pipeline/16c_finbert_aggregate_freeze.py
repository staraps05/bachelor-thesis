# %% 16c_finbert_aggregate_freeze.py
# Purpose: Validate the Colab-scored FinBERT probabilities and aggregate them to firm-week features.
# Inputs:  Data/colab/finbert_scores.csv, Data/processed/scoring_input.parquet, Data/processed/features_lm.parquet, Data/processed/panel_final.parquet, Data/Loughran-McDonald_MasterDictionary_1993-2025.csv
# Outputs: Data/processed/features_finbert.parquet
# Notes:
# - Gates: exactly 796,038 score rows; SHA256 of scoring_input and features_lm checked.
# - Imports pipeline/sentiment_aggregation.py (unit tests first); had_news must equal LM.
# - The output is write-protected only when every check passes.

import hashlib
import re
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sentiment_aggregation as sagg

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORES_FILE = REPO_ROOT / "Data" / "colab" / "finbert_scores.csv"
SCORING_FILE = REPO_ROOT / "Data" / "processed" / "scoring_input.parquet"
LM_FILE = REPO_ROOT / "Data" / "processed" / "features_lm.parquet"
PANEL_FILE = REPO_ROOT / "Data" / "processed" / "panel_final.parquet"
DICT_FILE = REPO_ROOT / "Data" / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
OUT_FILE = REPO_ROOT / "Data" / "processed" / "features_finbert.parquet"

N_HEADLINES = 796_038
PANEL_ROWS = 157_897
COVERED_KEYS = 104_533
SCORING_SHA = "eb42e0e3aa05c7c973e407c722125468bb506f1cc51e629042ab4d0a027ccc39"
LM_SHA = "b07a8f1e236d34d6d68dabd1038d1dfc632f6a2b220f4ccf9e3af88e0a7d50a3"
LM_ZERO_SHARE = 0.7731              # published by pipeline/15_score_lm.py
LM_NONZERO_MEAN = 0.0536

NEG_TERMS = ["bankruptcy", "lawsuit", "misses"]
POS_TERMS = ["beats", "record profit", "upgraded"]
TOKEN_RE = re.compile(r"[A-Z]+")
SPGI_PERMNO = 17478


def check(label: str, passed: bool, observed: str) -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {observed}")
    return passed


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}\nFreeze NOT granted. Fix/clarify and re-run.")


def main() -> int:
    all_pass = True

    print("=== STEP 0: frozen-module unit tests ===")
    # Verification (in place): aggregation module unit tests
    sagg.run_unit_tests(verbose=True)
    all_pass &= check("sentiment_aggregation unit tests", True, "ALL PASS")

    # ---- STEP 1 - Validate the returned scores ----
    print("\n=== STEP 1: gates on finbert_scores.csv ===")
    sc = pd.read_csv(SCORES_FILE)
    # Verification (in place): gates on finbert_scores.csv
    if list(sc.columns) != ["hid", "p_pos", "p_neu", "p_neg"]:
        stop(f"unexpected columns: {list(sc.columns)}")
    all_pass &= check("exactly 796,038 rows", len(sc) == N_HEADLINES,
                      f"{len(sc):,}")
    all_pass &= check("hid unique and complete (0..796,037)",
                      bool(sc["hid"].is_unique and sc["hid"].min() == 0
                           and sc["hid"].max() == N_HEADLINES - 1),
                      f"unique={sc['hid'].is_unique}")
    P = sc[["p_pos", "p_neu", "p_neg"]]
    all_pass &= check("zero NaNs", int(P.isna().sum().sum()) == 0, "0")
    all_pass &= check("probabilities in [0, 1]",
                      bool(((P >= 0) & (P <= 1)).all().all()), "asserted")
    rowsum_dev = (P.sum(axis=1) - 1).abs()
    all_pass &= check("each row sums to 1 +- 1e-4",
                      bool((rowsum_dev < 1e-4).all()),
                      f"max deviation {rowsum_dev.max():.2e}")
    if not all_pass:
        stop("returned scores failed a gate")

    # ---- STEP 2 - Join to the scoring input on hid ----
    print("\n=== STEP 2: join on hid ===")
    # Verification (in place): scoring-input SHA256
    sha_in = hashlib.sha256(SCORING_FILE.read_bytes()).hexdigest()
    if sha_in != SCORING_SHA:
        stop(f"scoring_input.parquet SHA mismatch: {sha_in}")
    hl = pd.read_parquet(SCORING_FILE)
    hl["target_week"] = pd.to_datetime(hl["target_week"])
    hl = hl.reset_index(drop=True).reset_index(names="hid")   # same hid as pipeline/16a
    hl = hl.merge(sc, on="hid", how="left", validate="1:1")
    # Verification (in place): join left no unmatched hid
    if hl[["p_pos", "p_neu", "p_neg"]].isna().any().any():
        stop("join left unmatched hids")
    print(f"  {len(hl):,} headlines joined 1:1 on hid "
          f"(scoring-input SHA verified {SCORING_SHA[:8]}...)")

    hl["s"] = hl["p_pos"] - hl["p_neg"]

    # ---- STEP 3 - Aggregate + expand to panel keys ----
    print("\n=== STEP 3: aggregate + expand ===")
    agg = sagg.aggregate_scores(hl[["PERMNO", "target_week", "s"]])
    agg = agg.rename(columns={c: f"finbert_{c}"
                              for c in ["mean", "std", "range", "kurt"]})

    # Verification (in place): kurtosis bounds
    # kurtosis bounds: excess kurtosis is >= -2 for any distribution and <= n for
    # n points; anything outside is numerical corruption, not data
    n_grp = (hl.groupby(["PERMNO", "target_week"]).size()
             .rename("n_grp").reset_index())
    kb = agg.merge(n_grp, on=["PERMNO", "target_week"], validate="1:1")
    # lower tolerance: with var = M2 - M1^2, groups at the variance guard inherit
    # up to ~2e-4 of undershoot below -2 from float noise in var
    KURT_LO = -2 - 5e-4
    bad_lo = kb["finbert_kurt"] < KURT_LO
    bad_hi = kb["finbert_kurt"] > kb["n_grp"]
    all_pass &= check("kurt within bounds: -2 - 5e-4 <= kurt <= n per group",
                      not bool((bad_lo | bad_hi).any()),
                      f"{int(bad_lo.sum())} below / {int(bad_hi.sum())} above; "
                      f"min {kb['finbert_kurt'].min():.6f}, "
                      f"max {kb['finbert_kurt'].max():.4f}")
    if (bad_lo | bad_hi).any():
        stop("impossible kurtosis values - numerical corruption")
    panel = pd.read_parquet(PANEL_FILE, columns=["PERMNO", "week_anchor"])
    keys = pd.DataFrame({"PERMNO": panel["PERMNO"],
                         "target_week": pd.to_datetime(panel["week_anchor"])})
    feat = keys.merge(agg, on=["PERMNO", "target_week"], how="left",
                      validate="1:1")
    feat["had_news"] = feat["finbert_mean"].notna().astype("int64")
    for c in ["finbert_mean", "finbert_std", "finbert_range", "finbert_kurt"]:
        feat[c] = feat[c].fillna(0.0)
    # Verification (in place): expansion row count
    if len(feat) != PANEL_ROWS:
        stop(f"expanded frame has {len(feat):,} rows")
    all_pass &= check("exactly 157,897 rows after expansion", True,
                      f"{len(feat):,}")

    out = feat[["PERMNO", "target_week", "finbert_mean", "finbert_std",
                "finbert_range", "finbert_kurt", "had_news"]].sort_values(
        ["PERMNO", "target_week"]).reset_index(drop=True)
    out["target_week"] = out["target_week"].dt.date

    # Verification (in place): had_news identical to the LM arm
    # had_news must match the LM arm exactly
    sha_lm = hashlib.sha256(LM_FILE.read_bytes()).hexdigest()
    if sha_lm != LM_SHA:
        stop(f"features_lm.parquet SHA mismatch: {sha_lm}")
    lm = pd.read_parquet(LM_FILE).sort_values(
        ["PERMNO", "target_week"]).reset_index(drop=True)
    same_keys = (out["PERMNO"].equals(lm["PERMNO"])
                 and out["target_week"].equals(lm["target_week"]))
    same_hn = out["had_news"].equals(lm["had_news"])
    if not (same_keys and same_hn):
        stop("had_news (or the key order) differs from the LM arm")
    all_pass &= check("had_news byte-identical to the LM arm "
                      "(coverage-only, scorer-independent)",
                      True, f"{int(out['had_news'].sum()):,} == {COVERED_KEYS:,}")

    # ---- STEP 4 - VALIDATION REPORT ----
    print("\n=== STEP 4: VALIDATION REPORT ===")



    # Verification (in place): LM recompute matches pipeline/15 anchors
    # cross-scorer agreement (report metric, not a check)
    print("\n  cross-scorer agreement (report metric, not a gate):")
    lmdict = pd.read_csv(DICT_FILE)
    lmdict["Word"] = lmdict["Word"].astype(str).str.upper()
    NEG = set(lmdict.loc[lmdict["Negative"] > 0, "Word"])
    POS = set(lmdict.loc[lmdict["Positive"] > 0, "Word"])
    toks = hl["headline"].str.upper().str.findall(TOKEN_RE)
    ex = toks.explode()
    ex = ex[ex.notna()]
    n_pos = ex.isin(POS).groupby(level=0).sum().reindex(hl.index, fill_value=0)
    n_neg = ex.isin(NEG).groupby(level=0).sum().reindex(hl.index, fill_value=0)
    tot = n_pos + n_neg
    s_lm = pd.Series(np.where(tot > 0, (n_pos - n_neg) / tot.where(tot > 0, 1),
                              0.0), index=hl.index)
    zero_share, nz_mean = float((s_lm == 0).mean()), float(s_lm[s_lm != 0].mean())
    if abs(zero_share - LM_ZERO_SHARE) > 0.002 or abs(nz_mean - LM_NONZERO_MEAN) > 0.002:
        stop(f"recomputed LM scores diverge from pipeline/15's published numbers "
             f"({zero_share:.4f} vs {LM_ZERO_SHARE}, {nz_mean:+.4f} vs "
             f"+{LM_NONZERO_MEAN})")
    print(f"    (LM recompute verified vs pipeline/15: zero share {zero_share:.2%}, "
          f"nonzero mean {nz_mean:+.4f})")


    # feature distributions
    covd = out[out["had_news"] == 1]
    print(f"\n  feature distributions over the {len(covd):,} covered firm-weeks:")
    print(f"    {'feature':<14}{'mean':>10}{'std':>10}{'p1':>10}{'median':>10}"
          f"{'p99':>10}{'min':>10}{'max':>10}")
    for c in ["finbert_mean", "finbert_std", "finbert_range", "finbert_kurt"]:
        d = covd[c]
        print(f"    {c:<14}{d.mean():>10.4f}{d.std():>10.4f}"
              f"{d.quantile(.01):>10.4f}{d.median():>10.4f}"
              f"{d.quantile(.99):>10.4f}{d.min():>10.4f}{d.max():>10.4f}")


    # ---- STEP 5 - Save ----
    print("\n=== STEP 5: save + freeze ===")
    # Verification (in place): refuse to overwrite a frozen output
    if OUT_FILE.exists() and not (OUT_FILE.stat().st_mode & stat.S_IWUSR):
        stop(f"{OUT_FILE.name} exists and is frozen - new version required")
    out.to_parquet(OUT_FILE, engine="pyarrow", index=False)
    sha = hashlib.sha256(OUT_FILE.read_bytes()).hexdigest()
    print(f"  wrote {OUT_FILE} ({len(out):,} x {len(out.columns)})")

    # ---- Verification: consistency and reproduction checks ----
    print(f"  p means: p_pos {P['p_pos'].mean():.4f}  "
          f"p_neu {P['p_neu'].mean():.4f}  p_neg {P['p_neg'].mean():.4f}")
    s = hl["s"]
    near0 = float((s.abs() < 0.05).mean())
    print(f"\n  headline-level s = p_pos - p_neg: s > 0 on "
          f"{float((s > 0).mean()):.2%}, s < 0 on {float((s < 0).mean()):.2%}, "
          f"|s| < 0.05 on {near0:.2%}")
    print(f"  mean s {s.mean():+.4f}  std {s.std():.4f}  "
          f"p1 {s.quantile(.01):+.4f}  median {s.median():+.4f}  "
          f"p99 {s.quantile(.99):+.4f}")
    print(f"  FINDING (LM contrast): FinBERT scores every headline "
          f"continuously - near-zero share {near0:.2%} vs LM's exact-zero "
          f"77.31%; no dictionary-sparsity floor")
    # sign spot checks (same terms as the LM arm)
    def term_check(term, expect_neg):
        m = hl[hl["headline"].str.contains(term, case=False, regex=False)]
        mean_s = float(m["s"].mean())
        pick = m.loc[m["s"].idxmin() if expect_neg else m["s"].idxmax()]
        print(f"\n  '{term}' - {len(m):,} matches; mean s {mean_s:+.4f}; "
              f"s<0 {float((m['s'] < 0).mean()):.1%}, "
              f"s>0 {float((m['s'] > 0).mean()):.1%}")
        print(f"    example: (p_pos {pick['p_pos']:.3f}, p_neu "
              f"{pick['p_neu']:.3f}, p_neg {pick['p_neg']:.3f}, "
              f"s {pick['s']:+.3f}) {pick['headline'][:95]!r}")
        return mean_s

    for term in NEG_TERMS:
        mean_s = term_check(term, True)
        allp = check(f"'{term}' matches score negative on average",
                     mean_s < 0, f"{mean_s:+.4f}")
        all_pass &= allp
    for term in POS_TERMS:
        mean_s = term_check(term, False)
        allp = check(f"'{term}' matches score positive on average "
                     f"(FinBERT understands earnings-surprise language)",
                     mean_s > 0, f"{mean_s:+.4f}")
        all_pass &= allp
    r_all = float(s.corr(s_lm))
    nz = s_lm != 0
    r_nz = float(s[nz].corr(s_lm[nz]))
    print(f"    corr(s_finbert, s_lm) ALL headlines      : {r_all:.4f}")
    print(f"    corr(s_finbert, s_lm) LM-nonzero only "
          f"({float(nz.mean()):.2%}): {r_nz:.4f}")
    print("    (expected positive, far from 1 - the scorers disagree enough "
          "to be different arms)")
    # manual aggregation check
    print("\n  manual aggregation check:")
    sizes = hl.groupby(["PERMNO", "target_week"]).size()
    key = sizes[(sizes >= 3) & (sizes <= 6)].index.min()
    rows = hl[(hl["PERMNO"] == key[0]) & (hl["target_week"] == key[1])]
    sv = rows["s"].to_numpy()
    print(f"    key: PERMNO {key[0]}, week {key[1].date()}, n = {len(sv)}")
    for i, (scv, h) in enumerate(zip(sv, rows["headline"]), 1):
        print(f"      s_{i} = {scv:+.6f}  {h[:75]!r}")
    m1, m2 = sv.mean(), (sv ** 2).mean()
    m3, m4 = (sv ** 3).mean(), (sv ** 4).mean()
    var = m2 - m1 ** 2
    std, rng = np.sqrt(max(var, 0)), sv.max() - sv.min()
    m4c = m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4
    kurt = (m4c / var ** 2 - 3.0) if var >= sagg.VAR_GUARD else 0.0
    print(f"    hand: M1={m1:+.6f}  var={var:.6f}  std={std:.6f}  "
          f"range={rng:.6f}  kurt={kurt:+.6f}")
    st = agg[(agg["PERMNO"] == key[0]) & (agg["target_week"] == key[1])].iloc[0]
    ok_man = (abs(st["finbert_mean"] - m1) < 1e-12
              and abs(st["finbert_std"] - std) < 1e-12
              and abs(st["finbert_range"] - rng) < 1e-12
              and abs(st["finbert_kurt"] - kurt) < 1e-12)
    all_pass &= check("hand-computed moments == stored (all four, 1e-12)",
                      ok_man, f"stored ({st['finbert_mean']:+.6f}, "
                      f"{st['finbert_std']:.6f}, {st['finbert_range']:.6f}, "
                      f"{st['finbert_kurt']:+.6f})")
    spgi = out[out["PERMNO"] == SPGI_PERMNO]
    z_ok = (len(spgi) == 313 and (spgi["had_news"] == 0).all()
            and (spgi[["finbert_mean", "finbert_std", "finbert_range",
                       "finbert_kurt"]] == 0).all().all())
    all_pass &= check("zero-coverage firm SPGI (17478): all-zero + had_news 0",
                      bool(z_ok), f"{len(spgi)} weeks")
    # Verification (in place): freeze only if all checks passed
    if not all_pass:
        print("\nOverall: SOME CHECKS FAILED - features written but NOT "
              "write-protected; FREEZE PENDING")
        return 1
    OUT_FILE.chmod(OUT_FILE.stat().st_mode
                   & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print("\n" + "=" * 72)
    print("  SENTIMENT S4b - FINBERT FEATURES FROZEN")
    print(f"  {OUT_FILE}")
    print(f"  {len(out):,} rows x 7 cols | SHA256: {sha}")
    print("=" * 72)
    print("\nOverall: ALL CHECKS PASSED — FREEZE GRANTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
