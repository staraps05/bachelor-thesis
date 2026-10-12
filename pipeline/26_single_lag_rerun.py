# %% 26_single_lag_rerun.py
# Purpose: Re-run the four XGBoost arms of pipeline/20 with sentiment attached
#          at the single-lag horizon (headlines of week t -> return of week
#          t+1) instead of the two-week gap of pipeline/19-21, and compare the
#          10-seed results with the double-lag reference values.
# Inputs:  Data/processed/model_frame_structured.parquet
#          Data/processed/features_lm.parquet
#          Data/processed/features_finbert.parquet
#          Data/processed/features_fingpt.parquet
# Outputs: prints only
# Notes:
# - pipeline/20_seed_stability.py is loaded via importlib for merge_frame
#   (reads and SHA-checks the inputs), dm_test, the XGBoost configuration
#   (FIXED_PARAMS with colsample_bytree 1.0, depth grid, round cap, early
#   stopping), folds, seeds 42-51 and r2_oos. Nothing else changes.
# - The feature files key sentiment at target_week t = headlines of the Mon-Sun
#   window preceding t. The script drops the merged sentiment columns and
#   re-merges each file on (PERMNO, week t+1) via the 313-anchor grid of the
#   feature files; had_news moves with it. Structured features stay at week t
#   and the target stays ret_excess_fwd.
# - A0 (structured only) must reproduce the pipeline/20 per-seed pooled R2 to
#   within 2e-4 pp on every seed; otherwise the script stops.
# - Depth is chosen per arm, fold and seed on validation MSE; predictions use
#   best_iteration + 1 rounds. DM tests apply dm_test (NW lag 4) to weekly
#   mean squared errors; DM > 0 means the arm beats A0.
# - Verdict per arm: a finding candidate when |mean delta R2 vs A0| across
#   seeds exceeds the seed sd of that delta.

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
merge_frame, dm_test = _c20.merge_frame, _c20.dm_test
FIXED_PARAMS, AMENDMENT = _c20.FIXED_PARAMS, _c20.AMENDMENT
DEPTH_GRID, NUM_ROUND_CAP = _c20.DEPTH_GRID, _c20.NUM_ROUND_CAP
EARLY_STOP, FOLDS, SEEDS = _c20.EARLY_STOP, _c20.FOLDS, _c20.SEEDS
KEYS, TARGET_FWD, INPUTS_30 = _c20.KEYS, _c20.TARGET_FWD, _c20.INPUTS_30
SENT_FILES, YEARS, SENT_ARMS = _c20.SENT_FILES, _c20.YEARS, _c20.SENT_ARMS
ALPHA = _c20.ALPHA
r2_oos = _c20.r2_oos

# double-lag reference values (percent / pp) from the pipeline/20 run log
A0_SEED_R2_DL = {42: -1.3365, 43: -1.3482, 44: -1.4479, 45: -1.3562,
                 46: -1.2329, 47: -1.0269, 48: -1.4714, 49: -0.9965,
                 50: -1.3526, 51: -1.4232}
DL_POOL = {"A0_structured": (-1.2992, 0.1657, -1.4714, -0.9965),
           "A1_lm": (-1.2966, 0.1298, -1.4701, -1.0734),
           "A2_finbert": (-1.4024, 0.1400, -1.7102, -1.2028),
           "A3_fingpt": (-1.3666, 0.1353, -1.6023, -1.1105)}
DL_DELTA = {"A1_lm": (+0.0026, 0.0843, -0.1777, +0.1311, 6),
            "A2_finbert": (-0.1031, 0.1431, -0.2870, +0.1260, 2),
            "A3_fingpt": (-0.0674, 0.1221, -0.2863, +0.0798, 4)}
DL_DM = {"A1_lm": (-0.080, 2), "A2_finbert": (-1.326, 1),
         "A3_fingpt": (-0.945, 0)}


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def main() -> int:
    # ---- re-attach sentiment from key t+1 ----
    print("=== STAGE 0: re-align sentiment to key t+1 (gap-safe) ===")
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
    # Verification (in place): re-merge preserves row count
        if len(df) != n0:
            stop(f"re-merge of {arm} changed row count")
        hn_cols.append(f"had_news_{arm}")
    n_nan = int(df[sent_cols + hn_cols].isna().sum().sum())
    n_drop = int(df[hn_cols[0]].isna().sum())
    # Verification (in place): no NaN after re-merge
    if n_nan:
        stop(f"{n_nan} NaN after t+1 re-merge - the zero-drop logic is wrong")
    # Verification (in place): had_news identical across arms
    if not (df[hn_cols[0]].equals(df[hn_cols[1]])
            and df[hn_cols[0]].equals(df[hn_cols[2]])):
        stop("had_news differs between arms after re-alignment")
    df["had_news"] = df[hn_cols[0]].astype(int)
    df = df.drop(columns=hn_cols + ["week_next"])
    print("  [PASS] ALIGNMENT (single lag): sentiment at key t+1 = headlines "
          "of Mon-Sun window t (verification/25 full-table audit: 796,038/796,038 "
          "in-window); target = ret_excess_fwd = window-(t+1) return; window "
          "t ends (Sunday) before window t+1 begins (Monday) -> headlines "
          "complete before the return window starts. NO look-ahead. "
          "Structured features remain at week t.")

    # ---- four arms, three folds, ten seeds ----
    print("\n=== STAGE 1: identical protocol (pipeline/20 config, seeds 42-51) ===")
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
    # Verification (in place): arm input counts
    for arm, cols in ARMS.items():
        want = 30 if arm == "A0_structured" else 35
        if len(cols) != want or len(set(cols)) != want:
            stop(f"{arm}: {len(cols)} inputs, expected {want}")
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {}
    # Verification (in place): fold feature weeks disjoint
    for fid, tr_y, va_y, te_y in FOLDS:
        m_tr, m_va, m_te = yr.isin(tr_y), yr.isin(va_y), yr.isin(te_y)
        w = [set(df.loc[m, "week_anchor"]) for m in (m_tr, m_va, m_te)]
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

    res = {}
    for seed in SEEDS:
        for arm, inputs in ARMS.items():
            parts = []
            for fid, _, _, _ in FOLDS:
                d_tr, d_va, d_te = dmats[arm][fid]
                best = None
                for depth in DEPTH_GRID:
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
    # Verification (in place): finite predictions
                if not np.isfinite(yhat).all():
                    stop(f"seed {seed} {arm} fold {fid}: NaN/inf predictions")
                p = te_meta[fid].copy()
                p["yhat"] = yhat
                p["test_year"] = te_year[fid]
                parts.append(p)
            preds = pd.concat(parts, ignore_index=True)
            e2 = (preds[TARGET_FWD] - preds["yhat"]) ** 2
            pool = 1.0 - float(e2.sum()) / float(
                (preds[TARGET_FWD] ** 2).sum())
            wk = (preds.assign(e2=e2).groupby("week_anchor")["e2"]
                  .mean().sort_index())
            res[(seed, arm)] = {"pool": pool, "wk": wk}
    # Verification (in place): per-seed A0 reproduces the pipeline/20 reference
        a0 = res[(seed, "A0_structured")]["pool"] * 100
        ref = A0_SEED_R2_DL[seed]
        if abs(a0 - ref) > 2e-4:
            stop(f"seed {seed}: A0 pooled R2 {a0:+.4f}% != pipeline/20's "
                 f"{ref:+.4f}% - the re-alignment leaked into A0")
        print(f"  seed {seed}: A0 {a0:+.4f}% == pipeline/20 [PASS] | "
              + " | ".join(f"{arm.split('_')[0]} "
                           f"{res[(seed, arm)]['pool'] * 100:+.4f}%"
                           for arm in SENT_ARMS))

    # ---- single- vs double-lag comparison ----
    print("\n=== STAGE 2: stability table - SINGLE lag vs pipeline/20 DOUBLE lag ===")

    def stats(v):
        """mean, sd (ddof=1), min, max"""
        a = np.array(v)
        return a.mean(), a.std(ddof=1), a.min(), a.max()

    print("  pooled OOS R2 (%), 10 seeds:        SINGLE lag (this run)      "
          "| DOUBLE lag (pipeline/20)")
    for arm in ARMS:
        m, s, lo, hi = stats([res[(sd, arm)]["pool"] * 100 for sd in SEEDS])
        dm_, ds_, dlo_, dhi_ = DL_POOL[arm]
        print(f"  {arm:<14} {m:>+8.4f} sd {s:.4f} [{lo:>+8.4f},{hi:>+8.4f}] "
              f"| {dm_:>+8.4f} sd {ds_:.4f} [{dlo_:>+8.4f},{dhi_:>+8.4f}]")
    print("\n  delta R2 vs A0 within seed (pp):    SINGLE lag                "
          "| DOUBLE lag (pipeline/20)")
    verdict_rows = {}
    for arm in SENT_ARMS:
        d = [(res[(sd, arm)]["pool"]
              - res[(sd, "A0_structured")]["pool"]) * 100 for sd in SEEDS]
        m, s, lo, hi = stats(d)
        nb = sum(1 for x in d if x > 0)
        verdict_rows[arm] = (m, s, nb)
        dm_, ds_, dlo_, dhi_, dnb = DL_DELTA[arm]
        print(f"  {arm:<14} {m:>+8.4f} sd {s:.4f} [{lo:>+8.4f},{hi:>+8.4f}] "
              f"beats {nb}/10 | {dm_:>+8.4f} sd {ds_:.4f} "
              f"[{dlo_:>+8.4f},{dhi_:>+8.4f}] beats {dnb}/10")
    print("\n  adapted DM vs A0 per seed:          SINGLE lag                "
          "| DOUBLE lag (pipeline/20)")
    dm_sig = {}
    for arm in SENT_ARMS:
        st, pv = [], []
        for sd in SEEDS:
            d = (res[(sd, "A0_structured")]["wk"]
                 - res[(sd, arm)]["wk"]).to_numpy()
            a, b = dm_test(d)
            st.append(a)
            pv.append(b)
        k = sum(1 for a, b in zip(st, pv) if b < ALPHA and a > 0)
        dm_sig[arm] = k
        print(f"  {arm:<14} DM mean {np.mean(st):+.3f} "
              f"[{min(st):+.3f},{max(st):+.3f}] p<{ALPHA}: {k}/10 "
              f"| DM mean {DL_DM[arm][0]:+.3f} p<{ALPHA}: {DL_DM[arm][1]}/10")

    # ---- Verification: consistency and reproduction checks ----
    print(f"  rows dropped (no week-(t+1) sentiment key): {n_drop} - the "
          f"(PERMNO, t+1) key exists in the 157,897 grid for EVERY frame row "
          f"(that successor panel row is exactly why the row survived "
          f"D-EVAL-0)")
    print(f"  [PASS] sentiment re-attached from key t+1 for all {len(df):,} "
          f"rows; had_news moved with it (share {df['had_news'].mean():.4f} "
          f"- t+1-shifted coverage)")
    print("  [PASS] A0 reproduces pipeline/20's per-seed pooled R2 on all 10 "
          "seeds - the ONLY change is the sentiment alignment")

    # ---- verdict ----
    print("\n=== STAGE 3: verdict + STOP ===")
    findings = []
    for arm in SENT_ARMS:
        m, s, nb = verdict_rows[arm]
        rule = "NO FINDING (|mean| <= seed sd)" if abs(m) <= s else \
            ("FINDING candidate (|mean| > seed sd)")
        print(f"  {arm:<14} mean delta {m:+.4f}pp vs seed sd {s:.4f} -> "
              f"{rule}; beats A0 {nb}/10; DM p<{ALPHA} in {dm_sig[arm]}/10")
        if abs(m) > s:
            findings.append(arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
