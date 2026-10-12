# %% 31b_a2_block_sums.py
# Purpose: Refit the A2 (FinBERT) arm of the primary single-lag specification
#          (3 folds x 10 seeds) and print block sums of truncated total-gain
#          shares: macro (8), characteristics (22), sentiment (5).
# Inputs:  Data/processed/model_frame_structured.parquet
#          Data/processed/features_lm.parquet
#          Data/processed/features_finbert.parquet
#          Data/processed/features_fingpt.parquet
# Outputs: prints only
# Notes:
# - pipeline/30_per_year_primary.py and pipeline/17_pre_model_integrity.py are
#   loaded via importlib for merge_frame, the fit configuration, the per-seed
#   reference R2 table and the characteristic/macro block lists.
# - Same A2 fit loop as pipeline/31a; the refit must reproduce A2's ten
#   per-seed pooled R2 values within GATE_TOL or the script stops.
# - Importance = total-gain share from the booster truncated at
#   best_iteration+1; block sums are taken per fold x seed cell, averaged over
#   the 3 folds per seed, then mean and sd (ddof=1) across the 10 seeds.
# - The three block masks must partition the 35 inputs exactly.
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_c30 = _load("c30", "30_per_year_primary.py")
merge_frame = _c30.merge_frame
FIXED_PARAMS, AMENDMENT = _c30.FIXED_PARAMS, _c30.AMENDMENT
DEPTH_GRID, NUM_ROUND_CAP = _c30.DEPTH_GRID, _c30.NUM_ROUND_CAP
EARLY_STOP, FOLDS, SEEDS = _c30.EARLY_STOP, _c30.FOLDS, _c30.SEEDS
TARGET_FWD, INPUTS_30 = _c30.TARGET_FWD, _c30.INPUTS_30
SENT_FILES = _c30.SENT_FILES
C26_SEED_R2, GATE_TOL = _c30.C26_SEED_R2, _c30.GATE_TOL
_c17 = _load("pre17", "17_pre_model_integrity.py")
CHARS_22, MACRO_8 = _c17.CHARS_22, _c17.MACRO_8

ARM = "A2_finbert"
SENT_5 = ["finbert_mean", "finbert_std", "finbert_range", "finbert_kurt",
          "had_news"]


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def main() -> int:
    # ---- rebuild the single-lag frame: sentiment re-keyed to week t+1 ----
    print("=== STAGE 0: rebuild single-lag frame (verbatim pipeline/26) ===")
    # Verification (in place): INPUTS_30 single source
    if INPUTS_30 != CHARS_22 + MACRO_8:
        stop("INPUTS_30 != CHARS_22 + MACRO_8 - single-source violated")
    df = merge_frame()
    sent_cols = [f"{a}_{s}" for a in SENT_FILES
                 for s in ("mean", "std", "range", "kurt")]
    df = df.drop(columns=sent_cols + ["had_news"])
    lm = pd.read_parquet(SENT_FILES["lm"][0],
                         columns=["PERMNO", "target_week"])
    grid_anchors = sorted(lm["target_week"].unique())
    frame_anchors = sorted(df["week_anchor"].unique())
    # Verification (in place): anchor grids 313/312
    if len(grid_anchors) != 313 or frame_anchors != grid_anchors[:312]:
        stop("anchor grids unexpected")
    g_idx = {a: i for i, a in enumerate(grid_anchors)}
    df["week_next"] = df["week_anchor"].map(
        lambda a: grid_anchors[g_idx[a] + 1])
    hn_cols = []
    for arm, (path, _) in SENT_FILES.items():
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
    # Verification (in place): no NaN after re-merge
    if int(df[sent_cols + hn_cols].isna().sum().sum()):
        stop("NaN after t+1 re-merge - pipeline/26 had 0")
    # Verification (in place): had_news identical across arms
    if not (df[hn_cols[0]].equals(df[hn_cols[1]])
            and df[hn_cols[0]].equals(df[hn_cols[2]])):
        stop("had_news differs between arms")
    df["had_news"] = df[hn_cols[0]].astype(int)
    df = df.drop(columns=hn_cols + ["week_next"])
    # Verification (in place): had_news share
    if abs(df["had_news"].mean() - 0.6620) > 5e-4:
        stop(f"had_news share {df['had_news'].mean():.4f} != 0.6620")

    # ---- A2 refit, reproduction gate, truncated gain shares ----
    print("\n=== STAGE 1: A2 refit, 3 folds x 10 seeds + HARD gate ===")
    inputs = list(INPUTS_30) + SENT_5
    # Verification (in place): input count
    if len(inputs) != 35 or len(set(inputs)) != 35:
        stop(f"A2: {len(inputs)} inputs, expected 35")
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    dmats, y_te = {}, {}
    for fid, tr_y, va_y, te_y in FOLDS:
        m_tr, m_va, m_te = yr.isin(tr_y), yr.isin(va_y), yr.isin(te_y)
        w = [set(df.loc[m, "week_anchor"]) for m in (m_tr, m_va, m_te)]
    # Verification (in place): fold feature weeks disjoint (inside loop)
        if w[0] & w[1] or w[0] & w[2] or w[1] & w[2]:
            stop(f"fold {fid}: feature-week sets not disjoint")
        dmats[fid] = (
            xgb.DMatrix(df.loc[m_tr, inputs], label=df.loc[m_tr, TARGET_FWD]),
            xgb.DMatrix(df.loc[m_va, inputs], label=df.loc[m_va, TARGET_FWD]),
            xgb.DMatrix(df.loc[m_te, inputs]),
        )
        y_te[fid] = df.loc[m_te, TARGET_FWD].to_numpy()

    shares = {}                     # (seed, fid) -> share vector (35,)
    max_dev = 0.0
    for seed in SEEDS:
        sse = ssy = 0.0
        for fid, _, _, _ in FOLDS:
            d_tr, d_va, d_te = dmats[fid]
            best = None
            for depth in DEPTH_GRID:   # ascending -> ties to lower depth
                params = dict(FIXED_PARAMS, **AMENDMENT,
                              seed=seed, max_depth=depth)
                bst = xgb.train(params, d_tr, num_boost_round=NUM_ROUND_CAP,
                                evals=[(d_va, "val")],
                                early_stopping_rounds=EARLY_STOP,
                                verbose_eval=False)
                val_mse = float(bst.best_score) ** 2
                if best is None or val_mse < best["val_mse"]:
                    best = {"bst": bst, "val_mse": val_mse,
                            "n_est": int(bst.best_iteration) + 1}
            bst, n_est = best["bst"], best["n_est"]
            yhat = bst.predict(d_te, iteration_range=(0, n_est))
    # Verification (in place): finite predictions (inside loop)
            if not np.isfinite(yhat).all():
                stop(f"seed {seed} fold {fid}: NaN/inf predictions")
            sse += float(((y_te[fid] - yhat) ** 2).sum())
            ssy += float((y_te[fid] ** 2).sum())
            sc = bst[0:n_est].get_score(importance_type="total_gain")
            tot = sum(sc.values())
    # Verification (in place): non-zero total gain (inside loop)
            if tot <= 0:
                stop(f"seed {seed} fold {fid}: zero total gain")
            shares[(seed, fid)] = np.array([sc.get(f, 0.0) / tot
                                            for f in inputs])
        pool = (1.0 - sse / ssy) * 100
        ref = C26_SEED_R2[seed][ARM]
        dev = abs(pool - ref)
        max_dev = max(max_dev, dev)
    # Verification (in place): per-seed reproduction gate (inside loop)
        if dev > GATE_TOL:
            stop(f"REPRODUCTION GATE FAILED: seed {seed} A2 pooled "
                 f"{pool:+.4f}% vs pipeline/26 log {ref:+.4f}% (|dev| {dev:.2e})")

    # ---- block sums ----
    print("\n=== STAGE 2: A2 block sums - truncated total-gain shares "
          "(mean over 3 folds x 10 seeds; sd across seeds) ===")
    blocks = {"macro (8)": np.array([f in MACRO_8 for f in inputs]),
              "chars (22)": np.array([f in CHARS_22 for f in inputs]),
              "sentiment (5)": np.array([f in SENT_5 for f in inputs])}
    # Verification (in place): block masks partition the inputs
    if sum(int(m.sum()) for m in blocks.values()) != 35:
        stop("block masks do not partition the 35 inputs")
    print(f"  {'block':<15}{'mean':>9}{'sd(seeds)':>11}"
          f"{'fold1':>9}{'fold2':>9}{'fold3':>9}")
    for name, mask in blocks.items():
        per_seed = np.array([np.mean([float(shares[(sd, fid)][mask].sum())
                                      for fid, _, _, _ in FOLDS])
                             for sd in SEEDS])
        by_fold = [float(np.mean([shares[(sd, fid)][mask].sum()
                                  for sd in SEEDS]))
                   for fid, _, _, _ in FOLDS]
        print(f"  {name:<15}{per_seed.mean():>9.4f}"
              f"{per_seed.std(ddof=1):>11.4f}"
              + "".join(f"{v:>9.4f}" for v in by_fold))

    # ---- Verification: consistency and reproduction checks ----
    print(f"  [PASS] single-lag frame rebuilt: {len(df):,} rows, "
          f"had_news share {df['had_news'].mean():.4f} == pipeline/26")
    print(f"  [PASS] gate: all 10 A2 per-seed pooled cells match the pipeline/26 "
          f"log within {GATE_TOL:.0e} pp (max |dev| {max_dev:.2e} pp)")
    print("  [INFO] reference points: A0 (pipeline/31) macro 0.7741 / chars "
          "0.2259; A2 sentiment block record 0.0269 (pipeline/31 (b), pipeline/31a)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
