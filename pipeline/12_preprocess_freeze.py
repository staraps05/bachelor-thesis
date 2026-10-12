# %% 12_preprocess_freeze.py
# Purpose: Rank-normalise the 22 characteristics (8 Block A + 14 Block B) week
#   by week to [-1, +1], impute missing values with the cross-sectional median
#   (0), validate, and write the final model panel plus its manifest.
# Inputs:  Data/processed/panel_structured.parquet
#          Data/membership_file.csv
# Outputs: Data/processed/panel_final.parquet
#          Data/processed/panel_final_manifest.md
# Notes:
# - Transform per (week_anchor, characteristic): rank the non-missing values
#   across firms (ties: average), map rank r of n to 2*(r-1)/(n-1) - 1
#   (n == 1 -> 0), then missing -> 0. Gu, Kelly & Xiu (2020) fn. 29-30, with
#   the week as the period.
# - Not transformed: ret_excess (stays a raw excess return), the 8 macro
#   columns (constant across firms within a week; enter raw), n_days
#   (diagnostic, not a feature) and the two keys. No winsorising, no row drops.
# - panel_final.parquet is written before validation; the manifest only after
#   every check passes. Both files then have their write bits stripped, and the
#   script refuses to overwrite a read-only panel_final.parquet.
# - TARGET_REF hard-codes the ret_excess mean/std/min/max the input must
#   reproduce (taken from the pipeline/11 output).
# - membership_file.csv is read only for tickers in the printed report.

import hashlib
import stat
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
IN_FILE = REPO_ROOT / "Data" / "processed" / "panel_structured.parquet"
MEMBERSHIP_FILE = REPO_ROOT / "Data" / "membership_file.csv"
OUT_FILE = REPO_ROOT / "Data" / "processed" / "panel_final.parquet"
MANIFEST_FILE = REPO_ROOT / "Data" / "processed" / "panel_final_manifest.md"

WORKING_ROWS = 157_897
SKELETON_ROWS = 157_901

BLOCKA = ["ret_12_1", "ret_6_1", "ret_1_0", "rvol_21d", "ivol_ff3_21d",
          "market_equity", "dolvol_126d", "ami_126d"]
BLOCKB = ["be_me", "ni_me", "sale_me", "chcsho_12m", "ope_be", "niq_be",
          "gp_at", "at_gr1", "sale_gr1", "oaccruals_at", "noa_at", "at_be",
          "niq_su", "beta_60m"]
MACRO8 = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]
CHARS = BLOCKA + BLOCKB                      # the 22 transformed columns
UNTOUCHED = ["PERMNO", "week_anchor", "ret_excess", "n_days"] + MACRO8

SPOT_WEEKS = [date(2018, 6, 15), date(2020, 3, 27), date(2023, 9, 29)]
SPOT_CHARS = ["be_me", "ret_12_1", "rvol_21d"]
ORDER_WEEK = date(2020, 3, 27)
PSA_PERMNO, PSA_WEEK = 64653, date(2021, 6, 4)      # REIT: zero JKP coverage
CBOE_PERMNO, CBOE_WEEK = 93429, date(2020, 6, 5)    # JKP coverage ends 2018-08

# ret_excess statistics the input must reproduce (target untouched)
TARGET_REF = {"mean": 0.00202, "std": 0.04993, "min": -0.8626, "max": 0.7705}


def check(label: str, passed: bool, observed: str) -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {observed}")
    return passed


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}\nFreeze NOT granted. Fix/clarify and re-run.")


def main() -> int:
    all_pass = True

    # ---- step 1: transform the 22 characteristics ----
    print("=== STEP 1: rank-normalise + impute (GKX fn. 29-30, weekly) ===")
    raw = pd.read_parquet(IN_FILE)
    # Verification (in place): input shape
    if len(raw) != WORKING_ROWS or len(raw.columns) != 34:
        stop(f"input is {len(raw):,} x {len(raw.columns)}, expected "
             f"{WORKING_ROWS:,} x 34")
    print(f"  input: {len(raw):,} rows x {len(raw.columns)} cols (stage 4)")

    g = raw.groupby("week_anchor")[CHARS]
    R = g.rank(method="average")             # NaN stays NaN
    C = g.transform("count")                 # n non-missing per (week, char)
    T = 2.0 * (R - 1.0) / (C - 1.0) - 1.0    # rank r -> [-1, +1]
    n1 = int(((C == 1) & R.notna()).sum().sum())
    T = T.mask((C == 1) & R.notna(), 0.0)    # n == 1 -> 0
    T = T.fillna(0.0)                        # missing -> cross-sectional median
    print(f"  n==1 cells (single non-missing value in a week): {n1} "
          f"{'(mapped to 0)' if n1 else '(never occurred)'}")

    final = raw.copy()
    final[CHARS] = T
    print(f"  transformed exactly {len(CHARS)} columns "
          f"(8 Block A + 14 Block B); target/macro/keys/n_days untouched")

    # ---- step 2: save the panel (manifest follows only after validation) ----
    print("\n=== STEP 2: save panel_final.parquet ===")
    # Verification (in place): refuse to overwrite a read-only panel_final.parquet
    if OUT_FILE.exists() and not (OUT_FILE.stat().st_mode & stat.S_IWUSR):
        stop(f"{OUT_FILE.name} exists and is read-only (frozen) - freezing "
             f"again requires a NEW version, never an in-place edit")
    final.to_parquet(OUT_FILE, engine="pyarrow", index=False)
    # Verification (in place): re-read equals the frame written
    reread = pd.read_parquet(OUT_FILE)
    if not reread.equals(final):
        stop("re-read mismatch on panel_final.parquet")
    print(f"  wrote {OUT_FILE} ({len(reread):,} rows x {len(reread.columns)} cols)")

    # ---- Verification: consistency and reproduction checks ----
    # ---- step 3: validation report ----
    print("\n=== STEP 3: VALIDATION REPORT ===")

    # row/column assertions; untransformed columns must be identical to the input
    # Verification (in place): 157,897 x 34 and untouched columns identical
    all_pass &= check("exactly 157,897 x 34",
                      len(final) == WORKING_ROWS and len(final.columns) == 34,
                      f"{len(final):,} x {len(final.columns)}")
    if not final[UNTOUCHED].equals(raw[UNTOUCHED]):
        stop("an untransformed column changed - transform leaked beyond the 22")
    all_pass &= check("keys, target, macro, n_days byte-identical to stage 4 "
                      "(12 columns)", True, "frame-equality asserted")

    # range + NaN assertions
    # Verification (in place): range and NaN
    in_range = ((final[CHARS] >= -1.0) & (final[CHARS] <= 1.0)).all().all()
    no_nan = final[CHARS].notna().all().all()
    if not (in_range and no_nan):
        stop("transformed values outside [-1, 1] or NaNs remain")
    all_pass &= check("all 22 transformed columns in [-1, +1], zero NaNs",
                      True, "asserted")

    # pre-transform NaN share vs post-transform exact-zero share
    # Verification (in place): pre-NaN vs post-zero share
    print("\n  pre-transform NaN share vs post-transform exact-zero share")
    print("  (post >= pre; small excess = genuine mid-rank hits: a true median "
          "maps to exactly 0 only for odd n or a centre tie):")
    print(f"    {'char':<14}{'mean':>9}{'pre-NaN':>10}{'post-0':>10}{'diff':>8}")
    max_diff = 0.0
    for c in CHARS:
        pre = raw[c].isna().mean()
        post = (final[c] == 0.0).mean()
        d = post - pre
        max_diff = max(max_diff, abs(d))
        print(f"    {c:<14}{final[c].mean():>+9.4f}{pre:>10.3%}{post:>10.3%}"
              f"{d:>+8.3%}")
    all_pass &= check("post-zero share matches pre-NaN share (diff <= 0.30pp "
                      "with the odd-n caveat)", max_diff <= 0.0030,
                      f"max |diff| = {max_diff:.3%}")

    # per-week spot checks: min/max hit -1/+1 and the top raw firm is the +1 firm
    # Verification (in place): per-week integrity spot checks
    tick = (pd.read_csv(MEMBERSHIP_FILE, usecols=["PERMNO", "Ticker", "DlyCalDt"],
                        parse_dates=["DlyCalDt"])
            .sort_values("DlyCalDt").groupby("PERMNO")["Ticker"].last())
    print("\n  per-week integrity spot checks:")
    for wk in SPOT_WEEKS:
        for c in SPOT_CHARS:
            m = final["week_anchor"] == wk
            rv, tv = raw.loc[m, c], final.loc[m, c]
            nn = int(rv.notna().sum())
            tnn = tv[rv.notna()]
            top_raw = int(raw.loc[m].loc[rv.idxmax(), "PERMNO"])
            plus1 = final.loc[m & (final[c] == 1.0), "PERMNO"].tolist()
            ok = (abs(tnn.min() + 1) < 1e-12 and abs(tnn.max() - 1) < 1e-12
                  and plus1 == [top_raw])
            all_pass &= check(
                f"{wk} {c}: min/max = -1/+1, top raw firm == +1 firm", ok,
                f"n={nn}, range [{tnn.min():+.3f}, {tnn.max():+.3f}], "
                f"top {tick.get(top_raw, '?')} ({top_raw}) -> +1 "
                f"{[f'{tick.get(p, chr(63))} ({p})' for p in plus1]}")

    # ordering preservation
    # Verification (in place): ordering preservation
    print(f"\n  ordering preservation (week {ORDER_WEEK}):")
    m = final["week_anchor"] == ORDER_WEEK
    for c in SPOT_CHARS:
        mask = m & raw[c].notna()
        sp = raw.loc[mask, c].corr(final.loc[mask, c], method="spearman")
        all_pass &= check(f"Spearman(raw, transformed) == 1.0 for {c}",
                          sp > 1 - 1e-12, f"{sp:.15f}")

    # imputation spot checks on two firms with no JKP row at that week
    # Verification (in place): imputation spot checks
    print("\n  imputation spot checks:")
    for label, p, wk in [("PSA (REIT, zero JKP coverage)", PSA_PERMNO, PSA_WEEK),
                         ("CBOE (JKP coverage ended 2018-08)", CBOE_PERMNO,
                          CBOE_WEEK)]:
        rowm = (final["PERMNO"] == p) & (final["week_anchor"] == wk)
        pre_b = raw.loc[rowm, BLOCKB].isna().all(axis=1).iloc[0]
        post_b = (final.loc[rowm, BLOCKB] == 0.0).all(axis=1).iloc[0]
        a_pop = raw.loc[rowm, BLOCKA].notna().all(axis=1).iloc[0]
        mac_pop = final.loc[rowm, MACRO8].notna().all(axis=1).iloc[0]
        all_pass &= check(
            f"{label} @ {wk}: 14 Block-B NaN pre -> exactly 0 post; "
            f"Block A + macro populated",
            bool(pre_b and post_b and a_pop and mac_pop),
            f"BlockB all-0: {post_b}, BlockA raw non-NaN: {a_pop}, "
            f"macro non-NaN: {mac_pop}")

    # target untouched
    # Verification (in place): target moments unchanged
    t = final["ret_excess"]
    print(f"\n  target: mean {t.mean():+.5f}  std {t.std():.5f}  "
          f"min {t.min():+.4f}  max {t.max():+.4f}")
    ok_t = (abs(t.mean() - TARGET_REF["mean"]) < 5e-6
            and abs(t.std() - TARGET_REF["std"]) < 5e-5
            and abs(t.min() - TARGET_REF["min"]) < 5e-5
            and abs(t.max() - TARGET_REF["max"]) < 5e-5)
    all_pass &= check("ret_excess statistics identical to the stage-4 figures",
                      ok_t, "mean/std/min/max match the stage-2/4 report")

    # final summary
    # Verification (in place): firm and week counts, window
    n_firms = final["PERMNO"].nunique()
    n_weeks = final["week_anchor"].nunique()
    wmin, wmax = min(final["week_anchor"]), max(final["week_anchor"])
    all_pass &= check("618 firms x 313 weeks, window 2018-01-05 .. 2023-12-29",
                      n_firms == 618 and n_weeks == 313
                      and wmin == date(2018, 1, 5) and wmax == date(2023, 12, 29),
                      f"{n_firms} firms, {n_weeks} weeks, {wmin} .. {wmax}")
    summary = (f"rows {len(final):,} | firms {n_firms} | weeks {n_weeks} "
               f"({wmin} .. {wmax}) | model inputs 30 (22 rank-normalised "
               f"characteristics + 8 raw macro) | target ret_excess (raw weekly "
               f"excess return) | n_days diagnostic, non-feature")
    print(f"\n  FINAL PANEL SUMMARY: {summary}")

    # Verification (in place): all checks passed before the manifest write
    if not all_pass:
        stop("validation failed - freeze NOT granted")

    # ---- step 4: write the manifest, then make both outputs read-only ----
    # Verification (in place): SHA256 of the written parquet (manifest input)
    sha = hashlib.sha256(OUT_FILE.read_bytes()).hexdigest()

    roles = {"PERMNO": "key", "week_anchor": "key", "ret_excess": "target",
             "n_days": "diagnostic (non-feature)"}
    roles.update({c: "feature-characteristic (rank-normalised, Block A)"
                  for c in BLOCKA})
    roles.update({c: "feature-characteristic (rank-normalised, Block B)"
                  for c in BLOCKB})
    roles.update({c: "feature-macro (raw, identical across firms within a week)"
                  for c in MACRO8})
    col_table = "\n".join(f"| `{c}` | {roles[c]} |" for c in final.columns)

    MANIFEST_FILE.write_text(f"""# panel_final.parquet — Gate 2 Freeze Manifest

- **Frozen:** {date.today().isoformat()}
- **File:** `Data/processed/panel_final.parquet`
- **SHA256:** `{sha}`
- **Rows x columns:** {len(final):,} x {len(final.columns)}
- **Summary:** {summary}
- **Frozen invariants:** skeleton {SKELETON_ROWS:,} rows (S1–S5) → working panel
  {WORKING_ROWS:,} (4 verified S4 target-drops). Every merge in the chain was
  row-count-asserted against these.

## Build chain (scripts, in order)
1. `pipeline/08_build_skeleton.py` — weekly grid + membership rows (S1–S5; C1 Friday
   anchors; exports `week_anchor_map`, the single week definition, D1)
2. `pipeline/09_build_target.py` — weekly excess-return target (S4 drops; VW
   aggregate cross-check 0.9958)
3. `pipeline/10_build_blockA.py` — 8 self-computed characteristics
   (`blockA_build_spec.md` v1.2 verbatim; F6 vs JKP: all 8 PASS)
4. `pipeline/11_merge_blockB_macro.py` — 14 JKP characteristics (merge_asof,
   backward, 45D staleness cap) + 8 Welch–Goyal macro (month m−1 rule)
5. `pipeline/12_preprocess_freeze.py` — rank-normalise to [−1, +1], median(0)
   imputation, validation, freeze (this file)

## Preprocessing method (locked)
Characteristics cross-sectionally ranked period-by-period (ties: average),
mapped linearly to [−1, +1]; missing values set to the cross-sectional median
(= 0 in rank space). Method: Gu, Kelly & Xiu (2020, RFS), footnotes 29–30; the
identical recipe in Drobetz & Otto (2021). Adaptation: the period is the WEEK
(panel frequency) instead of the month. Applied to the 22 characteristics only
— target, macro, keys and `n_days` untouched. No winsorisation, no row drops,
no other transforms.

## Columns ({len(final.columns)})
| column | role |
|---|---|
{col_table}

## Change policy
Read-only from freeze (write bits stripped on both artifacts). Any future
change requires a NEW versioned file with a new manifest
— never an in-place edit.
""")

    # Verification (in place): strip write bits on both outputs
    OUT_FILE.chmod(OUT_FILE.stat().st_mode
                   & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    MANIFEST_FILE.chmod(MANIFEST_FILE.stat().st_mode
                        & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    print(f"\n  manifest written: {MANIFEST_FILE}")
    print("  write bits stripped on both artifacts (read-only by convention)")
    print("\nOverall: ALL CHECKS PASSED — FREEZE GRANTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
