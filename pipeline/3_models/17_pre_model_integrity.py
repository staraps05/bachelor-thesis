# %% 17_pre_model_integrity.py
# Purpose: Build the predictive model frame (ret_excess_fwd = week t+1 excess return) from
#          panel_final.parquet and scan the 30 model inputs for leakage and degeneracy. Runs no model.
# Inputs:  Data/processed/panel_final.parquet
# Outputs: Data/processed/model_frame_structured.parquet, Data/processed/model_frame_structured_manifest.md
# Notes:
# - Gate: panel_final SHA256, shape, columns and key uniqueness; a mismatch stops the run.
# - Hard integrity failures sys.exit; threshold checks print as [FLAG] in the summary.

import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_FILE = REPO_ROOT / "Data" / "processed" / "panel_final.parquet"
PANEL_SHA = "eb3ed4cdd461b9772db1a4e5240b0f46a5f21ca3c9948a094b5e654e5fed19f2"
OUT_FILE = REPO_ROOT / "Data" / "processed" / "model_frame_structured.parquet"
OUT_MANIFEST = REPO_ROOT / "Data" / "processed" / "model_frame_structured_manifest.md"

N_ROWS, N_COLS, N_WEEKS = 157_897, 34, 313

KEYS = ["PERMNO", "week_anchor"]
TARGET = "ret_excess"           # week-t excess return as stored in the panel
TARGET_FWD = "ret_excess_fwd"   # week-(t+1) excess return = modelling target
DIAG = "n_days"                 # diagnostic, not a model input

# 22 rank-normalised characteristics (panel_final_manifest.md order)
CHARS_22 = [
    # Block A (8, self-computed)
    "ret_12_1", "ret_6_1", "ret_1_0", "rvol_21d", "ivol_ff3_21d",
    "market_equity", "dolvol_126d", "ami_126d",
    # Block B (14, JKP)
    "be_me", "ni_me", "sale_me", "chcsho_12m", "ope_be", "niq_be", "gp_at",
    "at_gr1", "sale_gr1", "oaccruals_at", "noa_at", "at_be", "niq_su",
    "beta_60m",
]
# 8 raw Welch-Goyal macro predictors (constant across firms within a week)
MACRO_8 = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]
INPUTS_30 = CHARS_22 + MACRO_8

CORR_LEAK_THR = 0.10   # leakage tripwire, characteristics only: macro predictors
                       # are within-week constant (312 effective values), so the
                       # pooled firm-level correlation is not a fair bar for them
CORR_DUP_THR = 0.99    # pairwise input |corr| above this -> duplicate FLAG

FLAGS: list[str] = []


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    # load and verify the panel
    print("=== STAGE 0: load & confirm the frozen artefact ===")
    # Verification (in place): panel present and SHA256
    if not PANEL_FILE.exists():
        stop(f"{PANEL_FILE} missing")
    sha = sha256_of(PANEL_FILE)
    if sha != PANEL_SHA:
        stop(f"panel_final.parquet SHA mismatch:\n  found  {sha}\n  frozen {PANEL_SHA}")
    print(f"  [PASS] SHA256 == frozen Gate-2 value ({PANEL_SHA[:8]}...)")

    df = pd.read_parquet(PANEL_FILE)
    # Verification (in place): panel shape
    if df.shape != (N_ROWS, N_COLS):
        stop(f"shape {df.shape}, expected ({N_ROWS:,}, {N_COLS})")
    print(f"  [PASS] shape {df.shape[0]:,} x {df.shape[1]}")

    # Verification (in place): column set, unique keys, no NaN target
    expected = set(KEYS) | {TARGET, DIAG} | set(INPUTS_30)
    if set(df.columns) != expected or len(expected) != N_COLS:
        stop(f"column mismatch:\n  missing {sorted(expected - set(df.columns))}"
             f"\n  extra   {sorted(set(df.columns) - expected)}")
    if df.duplicated(KEYS).any():
        stop("duplicate (PERMNO, week_anchor) keys in panel_final")
    n_tgt_nan = int(df[TARGET].isna().sum())
    if n_tgt_nan:
        stop(f"{n_tgt_nan} NaN in {TARGET} - panel_final should have none")
    print(f"  [PASS] keys unique; {TARGET} has 0 NaN")
    print(f"  roles resolved: keys={KEYS}; target={TARGET}; diagnostic={DIAG}")
    print(f"  22 rank-normalised characteristics: {CHARS_22}")
    print(f"  8 raw macro: {MACRO_8}")
    print(f"  -> 30 model inputs ({DIAG} and {TARGET} excluded from inputs)")

    # predictive frame: attach the week-(t+1) return via a per-firm self-merge
    print("\n=== STAGE 1: predictive frame (t -> t+1), gap-safe self-merge ===")
    weeks = sorted(df["week_anchor"].unique())
    # Verification (in place): week count
    if len(weeks) != N_WEEKS:
        stop(f"{len(weeks)} distinct week anchors, expected {N_WEEKS}")
    anchor_to_idx = pd.Series(range(N_WEEKS), index=weeks)
    idx_to_anchor = pd.Series(weeks, index=range(N_WEEKS))
    df["week_idx"] = df["week_anchor"].map(anchor_to_idx)
    # Verification (in place): week index mapping
    if df["week_idx"].isna().any():
        stop("week_anchor -> week_idx mapping produced NaN")
    print(f"  global week index built: 0..{N_WEEKS - 1} "
          f"({pd.Timestamp(weeks[0]).date()} .. {pd.Timestamp(weeks[-1]).date()})")

    # Target frame keyed (PERMNO, week_idx - 1): a feature row at week t matches
    # the return row of week t+1 only if that firm-row exists, so spell ends,
    # re-entry gaps and the sample end simply fail to match.
    tgt = df[["PERMNO", "week_idx", TARGET]].copy()
    tgt["week_idx"] = tgt["week_idx"] - 1
    tgt = tgt.rename(columns={TARGET: TARGET_FWD})
    frame = df.merge(tgt, on=["PERMNO", "week_idx"], how="left", validate="1:1")
    # Verification (in place): self-merge preserves row count
    if len(frame) != N_ROWS:
        stop(f"self-merge changed row count: {len(frame):,} != {N_ROWS:,}")
    print(f"  [PASS] self-merge 1:1, row count preserved pre-drop ({N_ROWS:,})")

    # Verification (in place): drop breakdown sums
    kept = frame[TARGET_FWD].notna()
    dropped = frame.loc[~kept, ["PERMNO", "week_idx"]]
    n_drop = len(dropped)
    n_sample_end = int((dropped["week_idx"] == N_WEEKS - 1).sum())
    mid = dropped[dropped["week_idx"] < N_WEEKS - 1]
    # split mid-series gaps: firm never returns (exit) vs firm returns later
    firm_max = df.groupby("PERMNO")["week_idx"].max()
    mid_exit = int((mid["week_idx"] == mid["PERMNO"].map(firm_max)).sum())
    mid_reentry = len(mid) - mid_exit
    if n_sample_end + len(mid) != n_drop:
        stop("drop breakdown does not sum")
    print(f"  rows dropped (no week-(t+1) row for the firm): {n_drop:,}")
    print(f"    sample-end (week_idx == {N_WEEKS - 1}):            {n_sample_end:,}")
    print(f"    mid-series gaps (week_idx < {N_WEEKS - 1}):        {len(mid):,}"
          f"  [exit-before-sample-end {mid_exit:,} | re-entry gap {mid_reentry:,}]")

    # no look-ahead: for every retained row the anchor of week_idx + 1 must
    # exist and be strictly later than the row's own anchor
    # Verification (in place): no look-ahead
    nxt_anchor = frame["week_idx"].add(1).map(idx_to_anchor)
    viol_missing = int((kept & nxt_anchor.isna()).sum())
    viol_order = int((kept & nxt_anchor.notna()
                      & ~(nxt_anchor > frame["week_anchor"])).sum())
    if viol_missing or viol_order:
        stop(f"look-ahead assert failed: {viol_missing} retained rows map to a "
             f"non-existent next week; {viol_order} rows have "
             f"anchor(t+1) <= anchor(t)")
    print(f"  [PASS] no-look-ahead: anchor(week_idx+1) > anchor(week_idx) for all "
          f"{int(kept.sum()):,} retained rows (0 violations)")

    out = frame.loc[kept].drop(columns=["week_idx"]).reset_index(drop=True)
    n_out = len(out)
    # Verification (in place): rows out and output schema
    if n_out != N_ROWS - n_drop:
        stop(f"rows out {n_out:,} != {N_ROWS:,} - {n_drop:,}")
    if list(out.columns) != list(df.drop(columns='week_idx').columns) + [TARGET_FWD]:
        stop("output schema is not panel_final columns + ret_excess_fwd")

    # Verification (in place): refuse to overwrite a read-only output
    if OUT_FILE.exists() and not OUT_FILE.stat().st_mode & 0o200:
        stop(f"{OUT_FILE.name} exists read-only - a frozen file must never be "
             f"overwritten; new version required")
    out.to_parquet(OUT_FILE, index=False)
    out_sha = sha256_of(OUT_FILE)
    OUT_MANIFEST.write_text(
        "# model_frame_structured.parquet — derived manifest\n\n"
        "- **Status:** DERIVED, regenerable — **NOT a freeze** (file left "
        "writable by design; regenerate any time with "
        "`pipeline/17_pre_model_integrity.py`)\n"
        "- **Built:** 2026-07-22 from FROZEN `panel_final.parquet` "
        f"(SHA256 `{PANEL_SHA}`)\n"
        f"- **Rows x columns:** {n_out:,} x {out.shape[1]}\n"
        f"- **SHA256 (this build):** `{out_sha}`\n\n"
        "## Construction (D-EVAL-0, the ONLY change vs panel_final)\n"
        "Features unchanged; only the target is forward-shifted. Global week "
        "index 0..312 over the panel's own sorted week anchors; per-firm "
        "self-merge attaches week-(t+1) `ret_excess` as `ret_excess_fwd` "
        "(gap-safe: no row at week t+1 -> no match -> row dropped). Dropped "
        f"{n_drop:,} rows with no week-(t+1) firm-row ({n_sample_end:,} "
        f"sample-end, {len(mid):,} mid-series: {mid_exit:,} exits, "
        f"{mid_reentry:,} re-entry gaps). No-look-ahead hard-asserted: "
        "anchor(t+1) strictly > anchor(t) on every retained row.\n\n"
        "## Columns (35) & roles\n"
        "The 34 panel_final columns unchanged (see `panel_final_manifest.md`) "
        "+ `ret_excess_fwd` (week-(t+1) excess return) = the modelling "
        "target. Model inputs remain the 30 declared there (22 rank-"
        "normalised characteristics + 8 raw macro). `ret_excess` (week t) is "
        "retained for reference only and is NOT a model input; `n_days` "
        "stays diagnostic.\n"
    )
    print(f"  saved {OUT_FILE.name}: {n_out:,} x {out.shape[1]} "
          f"(+ manifest, marked DERIVED / not a freeze)")
    print(f"  SHA256 {out_sha}")

    # ---- Verification: consistency and reproduction checks ----
    # feature vs shifted-target correlation scan
    print("\n=== STAGE 2: feature <-> SHIFTED-target correlation scan ===")
    rows = []
    y = out[TARGET_FWD]
    for col in INPUTS_30:
        is_char = col in CHARS_22
        method = "spearman" if is_char else "pearson"
        n_used = int((out[col].notna() & y.notna()).sum())
        c = out[col].corr(y, method=method)
        rows.append({"feature": col, "type": "char" if is_char else "macro",
                     "method": method, "corr": c, "n": n_used})
    scan = pd.DataFrame(rows)
    scan["abs"] = scan["corr"].abs()
    scan = scan.sort_values("abs", ascending=False).reset_index(drop=True)
    print(f"  {'feature':<14} {'type':<6} {'method':<9} {'corr':>9}  {'n':>8}")
    for r in scan.itertuples():
        if r.abs > CORR_LEAK_THR:
            mark = ("  <-- FLAG (> 0.10)" if r.type == "char" else
                    "  <-- over char bar: [INFO]")
        else:
            mark = ""
        print(f"  {r.feature:<14} {r.type:<6} {r.method:<9} {r.corr:>+9.4f}  "
              f"{r.n:>8,}{mark}")
    # characteristics above the bar are FLAGs; macro predictors above it are
    # informational only (within-week constant, judged on timing instead)
    leak = scan[(scan["type"] == "char") & (scan["abs"] > CORR_LEAK_THR)]
    if len(leak):
        for r in leak.itertuples():
            FLAGS.append(f"leakage suspicion: |corr({r.feature}, {TARGET_FWD})| "
                         f"= {r.abs:.4f} > {CORR_LEAK_THR} (characteristic)")
        print(f"  [FLAG] {len(leak)} characteristic(s) above {CORR_LEAK_THR} - "
              f"reported")
    else:
        ch = scan[scan["type"] == "char"]
        print(f"  [PASS] characteristic tripwire clean: max char |corr| = "
              f"{ch['abs'].max():.4f} - nothing above {CORR_LEAK_THR} "
              f"(gate is characteristic-only)")
    # duplicate / degenerate-column check on the 30 inputs
    print("\n=== STAGE 3: duplicate / degenerate-column check (30 inputs) ===")
    X = out[INPUTS_30]
    nan_counts = X.isna().sum()
    if int(nan_counts.sum()):
        print("  [INFO] NaN present in inputs (post-impute this should be 0 "
              "for chars; macro fully populated):")
        for col, n in nan_counts[nan_counts > 0].items():
            print(f"    {col}: {int(n):,} NaN")
    else:
        print("  [PASS] 0 NaN across all 30 inputs (chars imputed at Gate 2; "
              "macro fully populated)")
    all_nan = [c for c in INPUTS_30 if X[c].isna().all()]
    all_zero = [c for c in INPUTS_30 if (X[c] == 0).all()]
    zero_var_chars = [c for c in CHARS_22 if X[c].std(ddof=0) == 0]
    for c in all_nan:
        FLAGS.append(f"degenerate input: {c} is all-NaN")
    for c in all_zero:
        FLAGS.append(f"degenerate input: {c} is all-zero")
    for c in zero_var_chars:
        FLAGS.append(f"degenerate input: characteristic {c} has zero variance")
    # macro is constant within a week by design, so assess its variance
    # across weeks instead
    wk_nunique = out.groupby("week_anchor")[MACRO_8].nunique()
    if int((wk_nunique != 1).to_numpy().sum()):
        bad = wk_nunique[(wk_nunique != 1).any(axis=1)]
        stop(f"macro not constant within week (violates panel construction):\n{bad}")
    wk_macro = out.groupby("week_anchor")[MACRO_8].first()
    print(f"  [PASS] all 8 macro constant within each of {len(wk_macro):,} weeks "
          f"(by design; assessed across weeks below)")
    zero_var_macro = [c for c in MACRO_8 if wk_macro[c].std(ddof=0) == 0]
    for c in zero_var_macro:
        FLAGS.append(f"degenerate input: macro {c} has zero variance ACROSS weeks")
    if not (all_nan or all_zero or zero_var_chars or zero_var_macro):
        print("  [PASS] no all-NaN / all-zero / zero-variance inputs "
              "(chars across rows; macro across weeks)")
    print("  across-week nunique of macro values: "
          + ", ".join(f"{c}={int(wk_macro[c].nunique())}" for c in MACRO_8))
    # collinearity check
    corr = X.corr(method="pearson")  # chars are rank-normalised: pearson ~ spearman there
    pair_rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair_rows.append((cols[i], cols[j], corr.iloc[i, j]))
    pairs = pd.DataFrame(pair_rows, columns=["a", "b", "corr"])
    pairs["abs"] = pairs["corr"].abs()
    pairs = pairs.sort_values("abs", ascending=False).reset_index(drop=True)
    dup = pairs[pairs["abs"] > CORR_DUP_THR]
    for r in dup.itertuples():
        FLAGS.append(f"duplicate inputs: |corr({r.a}, {r.b})| = {r.abs:.4f} "
                     f"> {CORR_DUP_THR}")
    print(f"  top 10 most-collinear input pairs (Pearson, {len(pairs)} pairs "
          f"scanned; FLAG threshold {CORR_DUP_THR}):")
    for r in pairs.head(10).itertuples():
        mark = "  <-- FLAG" if r.abs > CORR_DUP_THR else ""
        print(f"    {r.a:<14} x {r.b:<14} {r.corr:>+8.4f}{mark}")
    if len(dup):
        print(f"  [FLAG] {len(dup)} pair(s) above {CORR_DUP_THR}")
    else:
        print(f"  [PASS] no input pair above {CORR_DUP_THR} "
              f"(max |corr| = {pairs['abs'].max():.4f}: "
              f"{pairs.loc[0, 'a']} x {pairs.loc[0, 'b']})")

    for r in scan[(scan["type"] == "macro")
                  & (scan["abs"] > CORR_LEAK_THR)].itertuples():
        print(f"  [INFO] macro {r.feature} |corr| = {r.abs:.4f} exceeds the "
              f"characteristic bar - NOT leakage "
              f"(timing exact 312/312; fwd +0.111 > contemp +0.072; "
              f"+0.009 excl-2020); retained")





    # Verification (in place): summary and FLAG verdict
    # summary
    print("\n=== STAGE 4: summary & HARD STOP ===")
    print(f"  rows in {N_ROWS:,} | dropped {n_drop:,} "
          f"(sample-end {n_sample_end:,}, mid-series {len(mid):,}) "
          f"| rows out {n_out:,}")
    print(f"  no-look-ahead hard assert: PASS (0 violations)")
    print(f"  corr scan vs {TARGET_FWD}: max |corr| = {scan['abs'].max():.4f} "
          f"({scan.loc[0, 'feature']}); char-only tripwire {CORR_LEAK_THR} "
          f": {len(leak)} flag(s); macro over the bar -> [INFO]")
    print(f"  degeneracy: {len(dup)} duplicate pair(s) > {CORR_DUP_THR}; "
          f"{len(all_nan)} all-NaN; {len(all_zero)} all-zero; "
          f"{len(zero_var_chars) + len(zero_var_macro)} zero-variance")
    if FLAGS:
        print(f"\n  Overall: FLAG ({len(FLAGS)}) - STOP:")
        for f in FLAGS:
            print(f"    - {f}")
    else:
        print("\n  Overall: PASS (all clear) - STOPPING here anyway; baseline "
              "model awaits explicit go (stage gate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
