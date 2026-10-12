# %% 27_covered_subsample.py
# Purpose: Re-run the four-arm single-lag XGBoost comparison on firms with high news coverage in the test years (had_news >= 60% of 2021-2023 weeks; 50%/70% robustness).
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: prints only
# Notes:
# - Loads pipeline/20 via importlib; loaded by pipeline/34-38, 40, 43-48, verification/28.
# - Gate: per-seed |A0 pooled R2| <= 0.05 or stop (A0 need not match the full panel).
# - Full-panel reference values from verification/audit_block9_10/c5_code26_output.log.

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
TARGET_FWD, INPUTS_30 = _c20.TARGET_FWD, _c20.INPUTS_30
SENT_FILES, SENT_ARMS, ALPHA = _c20.SENT_FILES, _c20.SENT_ARMS, _c20.ALPHA

THRESH_PRIMARY = 0.60
THRESH_ROBUST = [0.50, 0.70]
TEST_YEARS = {2021, 2022, 2023}

# Full-panel single-lag reference: verification/audit_block9_10/c5_code26_output.log
FULL26_POOL = {"A0_structured": (-1.2992, 0.1657), "A1_lm": (-1.2832, 0.1751),
               "A2_finbert": (-1.3167, 0.1523), "A3_fingpt": (-1.3295, 0.1711)}
FULL26_DELTA = {"A1_lm": (+0.0161, 0.0828, 4), "A2_finbert": (-0.0175, 0.0728, 5),
                "A3_fingpt": (-0.0302, 0.1155, 3)}
FULL26_DM = {"A1_lm": (+0.180, 1), "A2_finbert": (-0.534, 1),
             "A3_fingpt": (-0.702, 1)}

ARMS_DEF = None  # filled in main


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def single_lag_frame() -> pd.DataFrame:
    """Single-lag frame: sentiment and had_news re-attached from the
    week-(t+1) key (pipeline/26 alignment)."""
    df = merge_frame()
    sent_cols = [f"{a}_{s}" for a in SENT_FILES
                 for s in ("mean", "std", "range", "kurt")]
    df = df.drop(columns=sent_cols + ["had_news"])
    lm = pd.read_parquet(SENT_FILES["lm"][0],
                         columns=["PERMNO", "target_week"])
    grid_anchors = sorted(lm["target_week"].unique())
    frame_anchors = sorted(df["week_anchor"].unique())
    # Verification (in place): anchor grids 313/312 (inside single_lag_frame)
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
        if len(df) != n0:
            stop(f"re-merge of {arm} changed row count")
        hn_cols.append(f"had_news_{arm}")
    if int(df[sent_cols + hn_cols].isna().sum().sum()):
        stop("NaN after t+1 re-merge")
    if not (df[hn_cols[0]].equals(df[hn_cols[1]])
            and df[hn_cols[0]].equals(df[hn_cols[2]])):
        stop("had_news differs between arms")
    df["had_news"] = df[hn_cols[0]].astype(int)
    df = df.drop(columns=hn_cols + ["week_next"])
    print(f"  [PASS] single-lag frame rebuilt (pipeline/26 alignment): "
          f"{len(df):,} rows, 0 dropped, had_news share "
          f"{df['had_news'].mean():.4f}")
    return df


def fit_all(df_sub: pd.DataFrame, collect_dm: bool, label: str) -> dict:
    """Fit all arms x folds x seeds on a row subset (pipeline/26 protocol);
    returns pooled R2 per (seed, arm) and weekly squared errors if asked."""
    yr = pd.to_datetime(df_sub["week_anchor"]).dt.year
    split_masks = {}
    for fid, tr_y, va_y, te_y in FOLDS:
        m_tr, m_va, m_te = yr.isin(tr_y), yr.isin(va_y), yr.isin(te_y)
        w = [set(df_sub.loc[m, "week_anchor"]) for m in (m_tr, m_va, m_te)]
    # Verification (in place): fold feature weeks disjoint (inside fit_all)
        if w[0] & w[1] or w[0] & w[2] or w[1] & w[2]:
            stop(f"{label} fold {fid}: feature-week sets not disjoint")
        split_masks[fid] = (m_tr, m_va, m_te)
    dmats = {}
    for arm, inputs in ARMS_DEF.items():
        dmats[arm] = {}
        for fid, _, _, _ in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            dmats[arm][fid] = (
                xgb.DMatrix(df_sub.loc[m_tr, inputs],
                            label=df_sub.loc[m_tr, TARGET_FWD]),
                xgb.DMatrix(df_sub.loc[m_va, inputs],
                            label=df_sub.loc[m_va, TARGET_FWD]),
                xgb.DMatrix(df_sub.loc[m_te, inputs]),
            )
    te_meta = {fid: df_sub.loc[split_masks[fid][2],
                               ["week_anchor", TARGET_FWD]]
               .reset_index(drop=True) for fid, _, _, _ in FOLDS}
    res = {}
    for seed in SEEDS:
        for arm, inputs in ARMS_DEF.items():
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
    # Verification (in place): finite predictions (inside fit_all)
                if not np.isfinite(yhat).all():
                    stop(f"{label} seed {seed} {arm} fold {fid}: NaN/inf")
                p = te_meta[fid].copy()
                p["yhat"] = yhat
                parts.append(p)
            preds = pd.concat(parts, ignore_index=True)
            e2 = (preds[TARGET_FWD] - preds["yhat"]) ** 2
            pool = 1.0 - float(e2.sum()) / float(
                (preds[TARGET_FWD] ** 2).sum())
            entry = {"pool": pool}
            if collect_dm:
                entry["wk"] = (preds.assign(e2=e2)
                               .groupby("week_anchor")["e2"]
                               .mean().sort_index())
            res[(seed, arm)] = entry
    # Verification (in place): per-seed A0 pooled R2 within bounds (inside fit_all)
        a0 = res[(seed, "A0_structured")]["pool"]
        if abs(a0) > 0.05:
            stop(f"{label} seed {seed}: |A0 pooled R2| = {abs(a0):.4f} > 0.05")
    return res


def delta_stats(res: dict):
    out = {}
    for arm in SENT_ARMS:
        d = [(res[(sd, arm)]["pool"]
              - res[(sd, "A0_structured")]["pool"]) * 100 for sd in SEEDS]
        a = np.array(d)
        out[arm] = (a.mean(), a.std(ddof=1), int((a > 0).sum()))
    return out


def main() -> int:
    global ARMS_DEF
    print("=== STAGE 0: build the coverage filter (single-lag frame) ===")
    df = single_lag_frame()
    ARMS_DEF = {
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
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    te = df[yr.isin(TEST_YEARS)]
    share = te.groupby("PERMNO")["had_news"].mean()
    n_firms = df["PERMNO"].nunique()
    n_no_test = n_firms - len(share)
    subs = {}
    for thr in [THRESH_PRIMARY] + THRESH_ROBUST:
        qual = set(share[share >= thr].index)
        subs[thr] = df[df["PERMNO"].isin(qual)].copy()
        tag = " (PRIMARY)" if thr == THRESH_PRIMARY else ""
        print(f"  threshold {thr:.2f}{tag}: {len(qual)} / {n_firms} firms "
              f"qualify -> {len(subs[thr]):,} / {len(df):,} rows retained "
              f"(ALL years of qualifying firms; firm-level filter)")
    sub = subs[THRESH_PRIMARY]
    yr_s = pd.to_datetime(sub["week_anchor"]).dt.year
    print("  retained had_news share by year (threshold 0.60): "
          + ", ".join(f"{y} {sub.loc[yr_s == y, 'had_news'].mean():.3f}"
                      for y in range(2018, 2024)))

    print("\n=== STAGE 1: balance check (caveat, not a gate) ===")
    qual = set(sub["PERMNO"].unique())
    q, nq = sub, df[~df["PERMNO"].isin(qual)]
    last_anchor = max(df["week_anchor"].unique())
    ex_q = (q.groupby("PERMNO")["week_anchor"].max() < last_anchor)
    ex_n = (nq.groupby("PERMNO")["week_anchor"].max() < last_anchor)
    print(f"  {'':<16} {'qualifying':>12} {'non-qualifying':>15}")
    print(f"  {'firms':<16} {len(qual):>12} {nq['PERMNO'].nunique():>15}")
    print(f"  {'ret_excess mean':<16} {q['ret_excess'].mean():>12.6f} "
          f"{nq['ret_excess'].mean():>15.6f}")
    print(f"  {'ret_excess sd':<16} {q['ret_excess'].std():>12.4f} "
          f"{nq['ret_excess'].std():>15.4f}")
    print(f"  {'me rank mean':<16} {q['market_equity'].mean():>12.4f} "
          f"{nq['market_equity'].mean():>15.4f}   (rank-normalised [-1,1])")
    print(f"  {'exit before end':<16} {int(ex_q.sum()):>12} "
          f"{int(ex_n.sum()):>15}   ({ex_q.mean():.1%} vs {ex_n.mean():.1%})")
    tilt = []
    if q["market_equity"].mean() > nq["market_equity"].mean() + 0.05:
        tilt.append("larger-cap")
    if ex_q.mean() < ex_n.mean() - 0.05:
        tilt.append("survivor")
    print("  NOTE: subsample tilts "
          + (" and ".join(tilt) if tilt else "roughly balanced")
          + " - reported as a caveat alongside the result.")

    print("\n=== STAGE 2: run, identical protocol (threshold 0.60, "
          "seeds 42-51) ===")
    res = fit_all(sub, collect_dm=True, label="thr0.60")
    for seed in SEEDS:
        print(f"  seed {seed}: " + " | ".join(
            f"{arm.split('_')[0]} {res[(seed, arm)]['pool'] * 100:+.4f}%"
            for arm in ARMS_DEF))

    print("\n=== STAGE 3: table (0.60) + threshold robustness ===")

    def pool_stats(arm):
        a = np.array([res[(sd, arm)]["pool"] * 100 for sd in SEEDS])
        return a.mean(), a.std(ddof=1)

    print("  pooled OOS R2 (%), 10 seeds:      COVERED >=0.60          "
          "| FULL PANEL (pipeline/26)")
    for arm in ARMS_DEF:
        m, s = pool_stats(arm)
        fm, fs = FULL26_POOL[arm]
        print(f"  {arm:<14} {m:>+9.4f} sd {s:.4f}   | {fm:>+9.4f} sd {fs:.4f}")
    d60 = delta_stats(res)
    print("\n  delta vs A0 within seed (pp):     COVERED >=0.60          "
          "| FULL PANEL (pipeline/26)")
    for arm in SENT_ARMS:
        m, s, nb = d60[arm]
        fm, fs, fnb = FULL26_DELTA[arm]
        print(f"  {arm:<14} {m:>+9.4f} sd {s:.4f} beats {nb}/10 | "
              f"{fm:>+9.4f} sd {fs:.4f} beats {fnb}/10")
    print("\n  adapted DM vs A0 per seed:        COVERED >=0.60          "
          "| FULL PANEL (pipeline/26)")
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
        fdm, fk = FULL26_DM[arm]
        print(f"  {arm:<14} DM mean {np.mean(st):+.3f} "
              f"[{min(st):+.3f},{max(st):+.3f}] p<{ALPHA}: {k}/10 | "
              f"DM mean {fdm:+.3f} p<{ALPHA}: {fk}/10")

    d_rob = {}
    for thr in THRESH_ROBUST:
        print(f"\n  threshold {thr:.2f} robustness (deltas only, "
              f"same protocol):")
        r = fit_all(subs[thr], collect_dm=False, label=f"thr{thr}")
        d_rob[thr] = delta_stats(r)
        for arm in SENT_ARMS:
            m, s, nb = d_rob[thr][arm]
            print(f"    {arm:<14} delta {m:+.4f}pp sd {s:.4f} beats {nb}/10")

    # ---- Verification: consistency and reproduction checks ----
    print(f"  [INFO] {n_no_test} firms have no 2021-23 rows (exited earlier) "
          f"-> cannot qualify; selection uses COVERAGE ONLY, never outcomes")

    print("\n=== STAGE 4: verdict + STOP ===")
    no_finding_all = True
    for arm in SENT_ARMS:
        m, s, nb = d60[arm]
        rule = "NO FINDING" if abs(m) <= s else "FINDING candidate"
        no_finding_all &= abs(m) <= s
        signs = [np.sign(d60[arm][0])] + \
            [np.sign(d_rob[t][arm][0]) for t in THRESH_ROBUST]
        stable = "sign stable" if len(set(signs)) == 1 else "sign varies"
        print(f"  {arm:<14} 0.60: {m:+.4f}pp (sd {s:.4f}) -> {rule}; "
              f"DM p<{ALPHA} in {dm_sig[arm]}/10 | thresholds 0.50/0.60/0.70: "
              f"{d_rob[0.50][arm][0]:+.4f}/{m:+.4f}/"
              f"{d_rob[0.70][arm][0]:+.4f}pp ({stable})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
