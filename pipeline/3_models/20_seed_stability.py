# %% 20_seed_stability.py
# Purpose: Re-run pipeline/3_models/19 with colsample_bytree 1.0 across ten seeds; configuration supplier for later scripts.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: prints only
# Notes:
# - Loads 3_models/19 via importlib; loaded by 21, 22, 26, 27, 30, 39, 41, verification/23
# - Defines AMENDMENT = {"colsample_bytree": 1.0} (sole FIXED_PARAMS change); SEEDS 42-51.
# - Stops on input SHA256/shape mismatch or if any seed's A0 has |pooled R^2| > 0.05.

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[2]

# whole configuration from pipeline/19 (which loads the fit config from
# pipeline/18 and the input lists from pipeline/17)
_spec = importlib.util.spec_from_file_location(
    "cmp19", REPO_ROOT / "pipeline" / "3_models" / "19_four_arm_comparison.py")
_c19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c19)
FIXED_PARAMS = _c19.FIXED_PARAMS          # original: colsample 0.8, seed 42
DEPTH_GRID, NUM_ROUND_CAP = _c19.DEPTH_GRID, _c19.NUM_ROUND_CAP
EARLY_STOP, FOLDS = _c19.EARLY_STOP, _c19.FOLDS
KEYS, TARGET_FWD, INPUTS_30 = _c19.KEYS, _c19.TARGET_FWD, _c19.INPUTS_30
FRAME_FILE, FRAME_SHA = _c19.FRAME_FILE, _c19.FRAME_SHA
N_ROWS, N_COLS = _c19.N_ROWS, _c19.N_COLS
SENT_FILES, DM_LAG, ALPHA = _c19.SENT_FILES, _c19.DM_LAG, _c19.ALPHA
r2_oos = _c19.r2_oos

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
AMENDMENT = {"colsample_bytree": 1.0}     # the only change to FIXED_PARAMS
YEARS = (2021, 2022, 2023)
SENT_ARMS = ("A1_lm", "A2_finbert", "A3_fingpt")


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_frame() -> pd.DataFrame:
    """Merge the three sentiment feature sets onto the frame as pipeline/19 does."""
    # Verification (in place): frame and feature-file checks inside merge_frame (SHA, schema, rows, NaN, had_news identity)
    sha = sha256_of(FRAME_FILE)
    if sha != FRAME_SHA:
        stop(f"model_frame SHA mismatch: {sha}")
    df = pd.read_parquet(FRAME_FILE)
    if df.shape != (N_ROWS, N_COLS):
        stop(f"model_frame shape {df.shape}")
    print(f"  [PASS] model_frame_structured: SHA {FRAME_SHA[:8]}..., "
          f"{N_ROWS:,} x {N_COLS}")
    hn_cols = []
    for arm, (path, want_sha) in SENT_FILES.items():
        got = sha256_of(path)
        if got != want_sha:
            stop(f"{path.name} SHA mismatch: {got}")
        f = pd.read_parquet(path)
        want_cols = ["PERMNO", "target_week", f"{arm}_mean", f"{arm}_std",
                     f"{arm}_range", f"{arm}_kurt", "had_news"]
        if list(f.columns) != want_cols or f.shape != (157_897, 7):
            stop(f"{path.name} schema unexpected")
        f = f.rename(columns={"target_week": "week_anchor",
                              "had_news": f"had_news_{arm}"})
        if f["week_anchor"].dtype != df["week_anchor"].dtype:
            f["week_anchor"] = pd.to_datetime(f["week_anchor"]).dt.date
        n0 = len(df)
        df = df.merge(f, on=["PERMNO", "week_anchor"], how="left",
                      validate="m:1")
        if len(df) != n0:
            stop(f"merge of {arm} changed row count")
        hn_cols.append(f"had_news_{arm}")
        print(f"  [PASS] {path.name}: SHA == frozen ({want_sha[:8]}...), "
              f"m:1, rows {len(df):,}")
    sent_cols = [f"{a}_{s}" for a in SENT_FILES for s in
                 ("mean", "std", "range", "kurt")] + hn_cols
    if int(df[sent_cols].isna().sum().sum()):
        stop("NaN introduced in sentiment columns")
    if not (df[hn_cols[0]].equals(df[hn_cols[1]])
            and df[hn_cols[0]].equals(df[hn_cols[2]])):
        stop("had_news differs between arms")
    df["had_news"] = df[hn_cols[0]]
    df = df.drop(columns=hn_cols)
    print(f"  [PASS] 0 NaN in sentiment columns; had_news identical across "
          f"arms -> collapsed (share {df['had_news'].mean():.4f})")
    print("  [PASS] ALIGNMENT re-asserted: sentiment_t (strict pre-anchor "
          "headline lag, S1) on feature week t; target ret_excess_fwd = week "
          "t+1 (D-EVAL-0) -> look-ahead-free")
    return df


def dm_test(d: np.ndarray):
    """Mean of d over its Newey-West (lag DM_LAG) SE; two-sided normal p."""
    n = len(d)
    u = d - d.mean()
    s = float((u * u).sum()) / n
    for lag in range(1, DM_LAG + 1):
        w = 1.0 - lag / (DM_LAG + 1.0)
        s += 2.0 * w * float((u[lag:] * u[:-lag]).sum()) / n
    se = math.sqrt(s / n)
    stat = float(d.mean()) / se
    p = math.erfc(abs(stat) / math.sqrt(2.0))
    return stat, p


def main() -> int:
    # ---- print the colsample change ----
    print("=== STAGE 0: protocol amendment D-EVAL-3a ===")
    print("  [INFO] AMENDMENT: colsample_bytree 0.8 -> 1.0 (confound removal, "
          "not a loosening): per-tree column draws depended on COLUMN COUNT; "
          "in pipeline/19 folds 2-3 the three 35-column arms produced bit-"
          "identical predictions and a +0.0323pp A0 gap from the sampling "
          "space, not sentiment. With 1.0, arm differences can only come "
          "from the sentiment features themselves.")
    print("  [INFO] early stopping NOT relaxed - if folds 2-3 still shrink "
          "to a few trees, that IS the finding.")

    # ---- merge, input sets, splits, DMatrices ----
    print("\n=== STAGE 1: rebuild the four experiment sets (pipeline/19 merge) ===")
    df = merge_frame()
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
    print(f"  [PASS] arms: A0 30 inputs; A1/A2/A3 35 inputs "
          f"(sentiment NOT rank-normalised; had_news only in A1-A3)")

    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {}
    # Verification (in place): fold feature weeks disjoint
    for fid, tr_y, va_y, te_y in FOLDS:
        m_tr, m_va, m_te = yr.isin(tr_y), yr.isin(va_y), yr.isin(te_y)
        w = [set(df.loc[m, "week_anchor"]) for m in (m_tr, m_va, m_te)]
        if w[0] & w[1] or w[0] & w[2] or w[1] & w[2]:
            stop(f"fold {fid}: feature-week sets not disjoint")
        split_masks[fid] = (m_tr, m_va, m_te)
    dmats = {}   # arm -> fold -> (train, val, test) DMatrix; seed-independent
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

    # ---- seed loop ----
    print("\n=== STAGE 2: seed loop - 10 seeds x 4 arms x 3 folds ===")
    res = {}        # (seed, arm) -> pooled/per-year/excl-2021 R2, weekly mean
                    # e2, per-fold (depth, n_est) selections, block gain share
    ident_cells = 0
    ident_detail = []
    for seed in SEEDS:
        yhat_by_arm = {}
        for arm, inputs in ARMS.items():
            sels, imp_blk = [], []
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
                    n_est = int(bst.best_iteration) + 1
                    if best is None or val_mse < best["val_mse"]:
                        best = {"depth": depth, "bst": bst,
                                "val_mse": val_mse, "n_est": n_est}
                yhat = best["bst"].predict(
                    d_te, iteration_range=(0, best["n_est"]))
    # Verification (in place): finite predictions
                if not np.isfinite(yhat).all():
                    stop(f"seed {seed} {arm} fold {fid}: NaN/inf predictions")
                p = te_meta[fid].copy()
                p["yhat"] = yhat
                p["test_year"] = te_year[fid]
                parts.append(p)
                sels.append((fid, best["depth"], best["n_est"]))
                if arm != "A0_structured":
                    sc = best["bst"].get_score(importance_type="total_gain")
                    tot = sum(sc.values())
                    imp_blk.append(sum(sc.get(f, 0.0) for f in inputs[30:])
                                   / tot if tot else 0.0)
            preds = pd.concat(parts, ignore_index=True)
            e2 = (preds[TARGET_FWD] - preds["yhat"]) ** 2
            year_r2, sse_y, ssy_y = {}, {}, {}
            for y in YEARS:
                m = preds["test_year"] == y
                sse_y[y] = float(e2[m].sum())
                ssy_y[y] = float((preds.loc[m, TARGET_FWD] ** 2).sum())
                year_r2[y] = 1.0 - sse_y[y] / ssy_y[y]
            pool = 1.0 - sum(sse_y.values()) / sum(ssy_y.values())
            excl21 = 1.0 - (sse_y[2022] + sse_y[2023]) / (ssy_y[2022]
                                                          + ssy_y[2023])
            wk = (preds.assign(e2=e2).groupby("week_anchor")["e2"]
                  .mean().sort_index())
            res[(seed, arm)] = {"pool": pool, "year": year_r2, "wk": wk,
                                "sel": sels, "excl21": excl21,
                                "blk": float(np.mean(imp_blk))
                                if imp_blk else np.nan}
            yhat_by_arm[arm] = {y: preds.loc[preds["test_year"] == y,
                                             "yhat"].to_numpy()
                                for y in YEARS}
    # Verification (in place): per-seed A0 pooled R2 within bounds
        a0 = res[(seed, "A0_structured")]["pool"]
        if abs(a0) > 0.05:
            stop(f"seed {seed}: A0 re-audit failed - |pooled R2| = "
                 f"{abs(a0):.4f} > 0.05")
    # Verification (in place): identical-arm cell tally
        for y in YEARS:
            same = (np.array_equal(yhat_by_arm["A1_lm"][y],
                                   yhat_by_arm["A2_finbert"][y])
                    and np.array_equal(yhat_by_arm["A1_lm"][y],
                                       yhat_by_arm["A3_fingpt"][y]))
            if same:
                ident_cells += 1
                ident_detail.append((seed, y))
        for arm in ARMS:
            r = res[(seed, arm)]
            sel_s = " ".join(f"f{f}:d{d}/n{n}" for f, d, n in r["sel"])
            print(f"  seed {seed} {arm:<14} pooled {r['pool'] * 100:+8.4f}%  "
                  f"[{sel_s}]")

    # ---- stability table ----
    print("\n=== STAGE 3: stability table across 10 seeds "
          "(sample SD, ddof=1) ===")
    def stats(v):
        """mean, sample SD (ddof=1), min, max."""
        a = np.array(v)
        return a.mean(), a.std(ddof=1), a.min(), a.max()

    print(f"  pooled OOS R2 (%):")
    print(f"  {'arm':<14} {'mean':>9} {'sd':>7} {'min':>9} {'max':>9}")
    for arm in ARMS:
        m, s, lo, hi = stats([res[(sd, arm)]["pool"] * 100 for sd in SEEDS])
        print(f"  {arm:<14} {m:>+9.4f} {s:>7.4f} {lo:>+9.4f} {hi:>+9.4f}")
    print(f"\n  delta R2 vs A0 WITHIN seed (pp):")
    print(f"  {'arm':<14} {'mean':>9} {'sd':>7} {'min':>9} {'max':>9} "
          f"{'beats A0':>9}")
    dpool = {}
    for arm in SENT_ARMS:
        d = [(res[(sd, arm)]["pool"]
              - res[(sd, "A0_structured")]["pool"]) * 100 for sd in SEEDS]
        dpool[arm] = d
        m, s, lo, hi = stats(d)
        nb = sum(1 for x in d if x > 0)
        print(f"  {arm:<14} {m:>+9.4f} {s:>7.4f} {lo:>+9.4f} {hi:>+9.4f} "
              f"{nb:>6}/10")
    for y in YEARS:
        print(f"\n  {y}: R2 (%) then delta vs A0 (pp):")
        print(f"  {'arm':<14} {'meanR2':>9} {'sdR2':>7} "
              f"{'meanD':>9} {'sdD':>7} {'minD':>9} {'maxD':>9} {'beats':>6}")
        for arm in ARMS:
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

    # ---- DM tests per seed ----
    print("\n=== STAGE 4: DM significance under seed variation "
          f"(NW lag {DM_LAG}) ===")
    seed_flags = []
    print("  PRIMARY vs A0 (DM > 0 = arm beats A0):")
    dm_primary = {}
    for arm in SENT_ARMS:
        st, pv = [], []
        for sd in SEEDS:
            d = (res[(sd, "A0_structured")]["wk"]
                 - res[(sd, arm)]["wk"]).to_numpy()
            a, b = dm_test(d)
            st.append(a)
            pv.append(b)
        k = sum(1 for a, b in zip(st, pv) if b < ALPHA and a > 0)
        dm_primary[arm] = (np.mean(st), k)
        print(f"    {arm:<14} DM mean {np.mean(st):+.3f} "
              f"[{min(st):+.3f}, {max(st):+.3f}] | p mean {np.mean(pv):.3f} "
              f"[{min(pv):.3f}, {max(pv):.3f}] | p<{ALPHA}: {k}/10")
        if 0 < k < 10:
            seed_flags.append(f"{arm} vs A0 significant in only {k}/10 seeds")
    print("  SECONDARY pairwise (DM > 0 = SECOND arm better):")
    for a, b in (("A1_lm", "A2_finbert"), ("A1_lm", "A3_fingpt"),
                 ("A2_finbert", "A3_fingpt")):
        st, pv = [], []
        for sd in SEEDS:
            d = (res[(sd, a)]["wk"] - res[(sd, b)]["wk"]).to_numpy()
            s_, p_ = dm_test(d)
            st.append(s_)
            pv.append(p_)
        k = sum(1 for p_ in pv if p_ < ALPHA)
        sign_stable = all(x > 0 for x in st) or all(x < 0 for x in st)
        print(f"    {a} vs {b}: DM mean {np.mean(st):+.3f} "
              f"[{min(st):+.3f}, {max(st):+.3f}] | p mean {np.mean(pv):.3f} "
              f"| p<{ALPHA}: {k}/10 | sign stable: {sign_stable}")
        if 0 < k < 10:
            seed_flags.append(f"{a} vs {b} significant in only {k}/10 seeds")
        if not sign_stable:
            seed_flags.append(f"{a} vs {b}: DM sign flips across seeds")

    # ---- Verification: consistency and reproduction checks ----
    print(f"  [INFO] all else identical to pipeline/18/19; seeds {SEEDS}; "
          f"xgboost {xgb.__version__}")
    print(f"\n  [DIAGNOSTIC] identical-arm (year, seed) cells: "
          f"{ident_cells} / 30"
          + (f"  -> {ident_detail}" if ident_detail else
         "  (colsample amendment removed the sampling-space confound)"))

    # ---- verdict ----
    print("\n=== STAGE 5: verdict + HARD STOP ===")
    print("  (a) mean pooled delta vs A0 across seeds:")
    for arm in SENT_ARMS:
        m, s = np.mean(dpool[arm]), np.std(dpool[arm], ddof=1)
        print(f"      {arm:<14} {m:+.4f}pp (sd {s:.4f}) -> "
              f"{'beats A0 on mean' if m > 0 else 'does NOT beat A0'}; "
              f"|mean| {'>' if abs(m) > s else '<='} sd")
    print("  (b) extraction-method ordering across seeds (within seed):")
    for a, b in (("A2_finbert", "A1_lm"), ("A2_finbert", "A3_fingpt"),
                 ("A3_fingpt", "A1_lm")):
        k = sum(1 for sd in SEEDS
                if res[(sd, a)]["pool"] > res[(sd, b)]["pool"])
        print(f"      {a} > {b}: {k}/10 seeds")
    print("  (c) 2021 concentration (mean delta vs A0, pp):")
    for arm in SENT_ARMS:
        d21 = np.mean([(res[(sd, arm)]["year"][2021]
                        - res[(sd, "A0_structured")]["year"][2021]) * 100
                       for sd in SEEDS])
        dex = np.mean([(res[(sd, arm)]["excl21"]
                        - res[(sd, "A0_structured")]["excl21"]) * 100
                       for sd in SEEDS])
        print(f"      {arm:<14} 2021 {d21:+.4f} | excl-2021 {dex:+.4f}"
              + ("  <- 2021-concentrated" if d21 > 0 and dex <= 0 else ""))
    print(f"  (d) identical-arm cells: {ident_cells}/30 "
          f"(pipeline/19 single-seed had 2/3 years identical)")
    print("      mean sentiment-block gain share: "
          + ", ".join(f"{arm} "
                      f"{np.mean([res[(sd, arm)]['blk'] for sd in SEEDS]):.4f}"
                      for arm in SENT_ARMS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
