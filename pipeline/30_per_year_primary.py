# %% 30_per_year_primary.py
# Purpose: Regenerate the primary single-lag four-arm XGBoost run and compute per-year (2021-2023)
#          out-of-sample R2 next to the pooled values, per seed and arm.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: Data/processed/per_year_primary_r2.csv (160 rows, derived and regenerable)
# Notes:
# - Loads pipeline/20 via importlib; loaded by pipeline/31, 31a, 31b, 34, 35 and 47.
# - Gate: all 40 seed x arm pooled R2 match the hard-coded reference within 2e-4 pp.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "seed20", REPO_ROOT / "pipeline" / "20_seed_stability.py")
_c20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c20)
merge_frame = _c20.merge_frame
FIXED_PARAMS, AMENDMENT = _c20.FIXED_PARAMS, _c20.AMENDMENT
DEPTH_GRID, NUM_ROUND_CAP = _c20.DEPTH_GRID, _c20.NUM_ROUND_CAP
EARLY_STOP, FOLDS, SEEDS = _c20.EARLY_STOP, _c20.FOLDS, _c20.SEEDS
TARGET_FWD, INPUTS_30 = _c20.TARGET_FWD, _c20.INPUTS_30
SENT_FILES, YEARS, SENT_ARMS = _c20.SENT_FILES, _c20.YEARS, _c20.SENT_ARMS

OUT_CSV = REPO_ROOT / "Data" / "processed" / "per_year_primary_r2.csv"

ARM_ORDER = ("A0_structured", "A1_lm", "A2_finbert", "A3_fingpt")

# Per-seed pooled R2 (%) of the pipeline/26 run, 40 cells, transcribed from
# verification/audit_block9_10/c5_code26_output.log.
C26_SEED_R2 = {
    42: {"A0_structured": -1.3365, "A1_lm": -1.2133,
         "A2_finbert": -1.2786, "A3_fingpt": -1.3710},
    43: {"A0_structured": -1.3482, "A1_lm": -1.3056,
         "A2_finbert": -1.3397, "A3_fingpt": -1.3473},
    44: {"A0_structured": -1.4479, "A1_lm": -1.4598,
         "A2_finbert": -1.3789, "A3_fingpt": -1.3483},
    45: {"A0_structured": -1.3562, "A1_lm": -1.3681,
         "A2_finbert": -1.3202, "A3_fingpt": -1.1642},
    46: {"A0_structured": -1.2329, "A1_lm": -1.2388,
         "A2_finbert": -1.3147, "A3_fingpt": -1.3215},
    47: {"A0_structured": -1.0269, "A1_lm": -1.0537,
         "A2_finbert": -1.1404, "A3_fingpt": -1.0325},
    48: {"A0_structured": -1.4714, "A1_lm": -1.2960,
         "A2_finbert": -1.5335, "A3_fingpt": -1.6465},
    49: {"A0_structured": -0.9965, "A1_lm": -0.9660,
         "A2_finbert": -1.0387, "A3_fingpt": -1.1912},
    50: {"A0_structured": -1.3526, "A1_lm": -1.3934,
         "A2_finbert": -1.2898, "A3_fingpt": -1.3962},
    51: {"A0_structured": -1.4232, "A1_lm": -1.5371,
         "A2_finbert": -1.5331, "A3_fingpt": -1.4760},
}
# Pooled mean/sd/min/max per arm from the same log; transcription guard only.
C26_POOL_AGG = {"A0_structured": (-1.2992, 0.1657, -1.4714, -0.9965),
                "A1_lm": (-1.2832, 0.1751, -1.5371, -0.9660),
                "A2_finbert": (-1.3167, 0.1523, -1.5335, -1.0387),
                "A3_fingpt": (-1.3295, 0.1711, -1.6465, -1.0325)}

GATE_TOL = 2e-4     # pp


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def main() -> int:
    # Verification (in place): transcribed 40-cell table consistent with the recorded aggregates
    print("=== STAGE 0: reference-table transcription guard ===")
    for arm in ARM_ORDER:
        v = np.array([C26_SEED_R2[sd][arm] for sd in SEEDS])
        got = (v.mean(), v.std(ddof=1), v.min(), v.max())
        ref = C26_POOL_AGG[arm]
        if max(abs(g - r) for g, r in zip(got, ref)) > 1e-3:
            stop(f"{arm}: transcribed 40-cell table inconsistent with "
                 f"the recorded aggregates: got {got}, log {ref}")
    print("  [PASS] 40-cell reference table reproduces the recorded "
          "pooled mean/sd/min/max for all four arms (tol 1e-3)")

    # ---- single-lag frame: re-attach sentiment from key t+1 ----
    print("\n=== STAGE 1: rebuild single-lag frame (verbatim pipeline/26) ===")
    df = merge_frame()
    sent_cols = [f"{a}_{s}" for a in SENT_FILES
                 for s in ("mean", "std", "range", "kurt")]
    df = df.drop(columns=sent_cols + ["had_news"])

    lm = pd.read_parquet(SENT_FILES["lm"][0],
                         columns=["PERMNO", "target_week"])
    grid_anchors = sorted(lm["target_week"].unique())          # 313 anchors
    frame_anchors = sorted(df["week_anchor"].unique())         # 312 anchors
    # Verification (in place): anchor grids 313/312
    if len(grid_anchors) != 313 or frame_anchors != grid_anchors[:312]:
        stop("anchor grids unexpected (frame must be grid minus final week)")
    g_idx = {a: i for i, a in enumerate(grid_anchors)}
    df["week_next"] = df["week_anchor"].map(
        lambda a: grid_anchors[g_idx[a] + 1])
    # Verification (in place): week_next mapping
    if df["week_next"].isna().any():
        stop("week_next mapping produced NaN")

    hn_cols = []
    for arm, (path, _) in SENT_FILES.items():   # SHAs checked in merge_frame
        f = pd.read_parquet(path)
        f = f.rename(columns={"target_week": "week_next",
                              "had_news": f"had_news_{arm}"})
        n0 = len(df)
        df = df.merge(f, on=["PERMNO", "week_next"], how="left",
                      validate="m:1")
    # Verification (in place): re-merge preserves row count (inside loop)
        if len(df) != n0:
            stop(f"re-merge of {arm} changed row count")
        hn_cols.append(f"had_news_{arm}")
    n_nan = int(df[sent_cols + hn_cols].isna().sum().sum())
    n_drop = int(df[hn_cols[0]].isna().sum())
    # Verification (in place): no rows dropped, no NaN after re-merge
    if n_drop or n_nan:
        stop(f"{n_drop} rows without week-(t+1) key / {n_nan} NaN after "
             f"re-merge - pipeline/26 had 0/0")
    # Verification (in place): had_news identical across arms
    if not (df[hn_cols[0]].equals(df[hn_cols[1]])
            and df[hn_cols[0]].equals(df[hn_cols[2]])):
        stop("had_news differs between arms after re-alignment")
    df["had_news"] = df[hn_cols[0]].astype(int)
    df = df.drop(columns=hn_cols + ["week_next"])
    hn_share = df["had_news"].mean()
    # Verification (in place): had_news share
    if abs(hn_share - 0.6620) > 5e-4:
        stop(f"had_news share {hn_share:.4f} != pipeline/26's 0.6620")

    # ---- fit per fold, then gate on pooled R2 ----
    print("\n=== STAGE 2: refit 4 arms x 3 folds x 10 seeds "
          "(identical protocol) + HARD reproduction gate ===")
    ARMS = {
        "A0_structured": list(INPUTS_30),
        "A1_lm": list(INPUTS_30) + ["lm_mean", "lm_std", "lm_range",
                                    "lm_kurt", "had_news"],
        "A2_finbert": list(INPUTS_30) + ["finbert_mean", "finbert_std",
                                         "finbert_range", "finbert_kurt",
                                         "had_news"],
        "A3_fingpt": list(INPUTS_30) + ["fingpt_mean", "fingpt_std",
                                        "fingpt_range", "fingpt_kurt",
                                        "had_news"],
    }
    for arm, cols in ARMS.items():
        want = 30 if arm == "A0_structured" else 35
    # Verification (in place): arm input counts
        if len(cols) != want or len(set(cols)) != want:
            stop(f"{arm}: {len(cols)} inputs, expected {want}")
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {}
    for fid, tr_y, va_y, te_y in FOLDS:
        m_tr, m_va, m_te = yr.isin(tr_y), yr.isin(va_y), yr.isin(te_y)
        w = [set(df.loc[m, "week_anchor"]) for m in (m_tr, m_va, m_te)]
    # Verification (in place): fold feature weeks disjoint (inside loop)
        if w[0] & w[1] or w[0] & w[2] or w[1] & w[2]:
            stop(f"fold {fid}: feature-week sets not disjoint")
        split_masks[fid] = (m_tr, m_va, m_te)
    dmats = {}
    for arm, inputs in ARMS.items():
        dmats[arm] = {}
        for fid, _, _, _ in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            dmats[arm][fid] = (
                xgb.DMatrix(df.loc[m_tr, inputs],
                            label=df.loc[m_tr, TARGET_FWD]),
                xgb.DMatrix(df.loc[m_va, inputs],
                            label=df.loc[m_va, TARGET_FWD]),
                xgb.DMatrix(df.loc[m_te, inputs]),
            )
    te_meta = {fid: df.loc[split_masks[fid][2],
                           ["week_anchor", TARGET_FWD]].reset_index(drop=True)
               for fid, _, _, _ in FOLDS}
    te_year = {fid: sorted(te_y)[0] for fid, _, _, te_y in FOLDS}
    # Verification (in place): test year equals fold
    if sorted(te_year.values()) != list(YEARS):
        stop(f"test years {sorted(te_year.values())} != {list(YEARS)} - "
             f"year==fold mapping broken")

    res = {}          # (seed, arm) -> {"pool": r2, "year": {y: r2}}
    max_dev = 0.0
    for seed in SEEDS:
        for arm, inputs in ARMS.items():
            parts = []
            for fid, _, _, _ in FOLDS:
                d_tr, d_va, d_te = dmats[arm][fid]
                best = None
                for depth in DEPTH_GRID:   # ascending -> ties to lower depth
                    params = dict(FIXED_PARAMS, **AMENDMENT,
                                  seed=seed, max_depth=depth)
                    bst = xgb.train(params, d_tr,
                                    num_boost_round=NUM_ROUND_CAP,
                                    evals=[(d_va, "val")],
                                    early_stopping_rounds=EARLY_STOP,
                                    verbose_eval=False)
                    val_mse = float(bst.best_score) ** 2
                    if best is None or val_mse < best["val_mse"]:
                        best = {"bst": bst, "val_mse": val_mse,
                                "n_est": int(bst.best_iteration) + 1}
                yhat = best["bst"].predict(
                    d_te, iteration_range=(0, best["n_est"]))
    # Verification (in place): finite predictions (inside loop)
                if not np.isfinite(yhat).all():
                    stop(f"seed {seed} {arm} fold {fid}: NaN/inf predictions")
                p = te_meta[fid].copy()
                p["yhat"] = yhat
                p["test_year"] = te_year[fid]
                parts.append(p)
            preds = pd.concat(parts, ignore_index=True)
            e2 = (preds[TARGET_FWD] - preds["yhat"]) ** 2
            # GKX eq. 19, un-demeaned: per year uses its own sum(y^2);
            # pooled is the ratio of the summed pieces.
            year_r2, sse_y, ssy_y = {}, {}, {}
            for y in YEARS:
                m = preds["test_year"] == y
                sse_y[y] = float(e2[m].sum())
                ssy_y[y] = float((preds.loc[m, TARGET_FWD] ** 2).sum())
                year_r2[y] = 1.0 - sse_y[y] / ssy_y[y]
            pool = 1.0 - sum(sse_y.values()) / sum(ssy_y.values())
            res[(seed, arm)] = {"pool": pool, "year": year_r2}
    # Verification (in place): per-seed reproduction gate against the transcribed table (inside loop)
        # reproduction gate: 4 pooled cells of this seed vs reference table
        line, fails = [], []
        for arm in ARM_ORDER:
            got = res[(seed, arm)]["pool"] * 100
            ref = C26_SEED_R2[seed][arm]
            dev = abs(got - ref)
            max_dev = max(max_dev, dev)
            line.append(f"{arm.split('_')[0]} {got:+.4f}% "
                        f"(log {ref:+.4f}, dev {dev:.2e})")
            if dev > GATE_TOL:
                fails.append(f"{arm}: got {got:+.4f}%, log {ref:+.4f}%, "
                             f"|dev| {dev:.2e} > {GATE_TOL:.0e}")
        print(f"  seed {seed}: " + " | ".join(line))
        if fails:
            stop(f"REPRODUCTION GATE FAILED at seed {seed} - per-year "
                 f"numbers NOT reported.\n  " + "\n  ".join(fails))

    print("\n=== STAGE 3: per-year OOS R2 (the Table 4.1 numbers) ===")

    def stats(v):
        a = np.array(v)
        return a.mean(), a.std(ddof=1), a.min(), a.max()

    for y in YEARS:
        print(f"\n  {y}: R2 (%) across 10 seeds, then delta vs A0 within "
              f"seed (pp):")
        print(f"  {'arm':<14} {'meanR2':>9} {'sdR2':>7} "
              f"{'meanD':>9} {'sdD':>7} {'minD':>9} {'maxD':>9} {'beats':>6}")
        for arm in ARM_ORDER:
            m, s, _, _ = stats([res[(sd, arm)]["year"][y] * 100
                                for sd in SEEDS])
            if arm == "A0_structured":
                print(f"  {arm:<14} {m:>+9.4f} {s:>7.4f}")
                continue
            d = [(res[(sd, arm)]["year"][y]
                  - res[(sd, "A0_structured")]["year"][y]) * 100
                 for sd in SEEDS]
            md, sdd, lo, hi = stats(d)
            nb = sum(1 for x in d if x > 0)
            print(f"  {arm:<14} {m:>+9.4f} {s:>7.4f} {md:>+9.4f} {sdd:>7.4f} "
                  f"{lo:>+9.4f} {hi:>+9.4f} {nb:>4}/10")

    print("\n=== STAGE 4: write DERIVED results CSV ===")
    rows = []
    for seed in SEEDS:
        for arm in ARM_ORDER:
            scopes = [("pooled", res[(seed, arm)]["pool"])] + \
                     [(str(y), res[(seed, arm)]["year"][y]) for y in YEARS]
            for scope, r2 in scopes:
                a0 = (res[(seed, "A0_structured")]["pool"] if scope == "pooled"
                      else res[(seed, "A0_structured")]["year"][int(scope)])
                rows.append({
                    "seed": seed, "arm": arm, "scope": scope,
                    "r2_pct": r2 * 100,
                    "delta_vs_a0_pp": (np.nan if arm == "A0_structured"
                                       else (r2 - a0) * 100),
                    "logged_c26_pooled_pct": (C26_SEED_R2[seed][arm]
                                              if scope == "pooled"
                                              else np.nan),
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"  [PASS] wrote {OUT_CSV.relative_to(REPO_ROOT)} "
          f"({len(out)} rows = 10 seeds x 4 arms x 4 scopes) - DERIVED, "
          f"regenerable by this script; NOT a freeze")

    # ---- Verification: consistency and reproduction checks ----
    print(f"  [PASS] single-lag frame rebuilt: {len(df):,} rows, 0 dropped, "
          f"had_news share {hn_share:.4f} == pipeline/26")
    print(f"  [PASS] reproduction gate: all 40 seed x arm pooled cells match "
          f"the pipeline/26 log within {GATE_TOL:.0e} pp "
          f"(max |dev| {max_dev:.2e} pp)")
    print("\n  [INFO] no per-year DM tests: DM is pooled-only per Sec. 3.9; "
          "the pooled DM record stands in the pipeline/26 log entry.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
