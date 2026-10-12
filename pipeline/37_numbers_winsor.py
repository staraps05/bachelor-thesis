# %% 37_numbers_winsor.py
# Purpose: Four-arm single-lag run with a winsorised target: Table C.2 winsorised rows and DM counts.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: Data/processed/numbers_check/ tableC2_winsor.csv, tableC2_winsor_perseed.csv, tableC2_winsor_dm.csv, tableC2_winsor_clips.csv
# Notes:
# - Loads pipeline/27 via importlib for single_lag_frame, dm_test and the fit config.
# - Target clipped per fold at training 0.5/99.5 percentiles; bounds saved to clips CSV.
# - R2 evaluated on the winsorised (winsor_eval, Table C.2) and raw test target; no gate.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "Data" / "processed" / "numbers_check"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c27 = _load("cov27", "27_covered_subsample.py")
single_lag_frame, dm_test = c27.single_lag_frame, c27.dm_test
FIXED_PARAMS, AMENDMENT = c27.FIXED_PARAMS, c27.AMENDMENT
DEPTH_GRID, NUM_ROUND_CAP = c27.DEPTH_GRID, c27.NUM_ROUND_CAP
EARLY_STOP, FOLDS, SEEDS = c27.EARLY_STOP, c27.FOLDS, c27.SEEDS
TARGET_FWD, INPUTS_30 = c27.TARGET_FWD, c27.INPUTS_30
SENT_ARMS, ALPHA = c27.SENT_ARMS, c27.ALPHA


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== 37: winsorised target xgb {xgb.__version__} ===")
    df = single_lag_frame()
    ARMS = {
        "A0_structured": list(INPUTS_30),
        "A1_lm": list(INPUTS_30) + ["lm_mean", "lm_std", "lm_range", "lm_kurt",
                                    "had_news"],
        "A2_finbert": list(INPUTS_30) + ["finbert_mean", "finbert_std",
                                         "finbert_range", "finbert_kurt",
                                         "had_news"],
        "A3_fingpt": list(INPUTS_30) + ["fingpt_mean", "fingpt_std",
                                        "fingpt_range", "fingpt_kurt",
                                        "had_news"],
    }
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {fid: (yr.isin(t), yr.isin(v), yr.isin(te))
                   for fid, t, v, te in FOLDS}
    clips = {}
    for fid, _, _, _ in FOLDS:
        m_tr = split_masks[fid][0]
        ytr = df.loc[m_tr, TARGET_FWD].to_numpy(float)
        lo, hi = np.percentile(ytr, [0.5, 99.5])
        clips[fid] = (float(lo), float(hi))
    dmats = {}
    for arm, inputs in ARMS.items():
        dmats[arm] = {}
        for fid, _, _, _ in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            lo, hi = clips[fid]
            dmats[arm][fid] = (
                xgb.DMatrix(df.loc[m_tr, inputs], label=df.loc[m_tr, TARGET_FWD].clip(lo, hi)),
                xgb.DMatrix(df.loc[m_va, inputs], label=df.loc[m_va, TARGET_FWD].clip(lo, hi)),
                xgb.DMatrix(df.loc[m_te, inputs]))
    te_meta = {}
    for fid, _, _, _ in FOLDS:
        m_te = split_masks[fid][2]
        lo, hi = clips[fid]
        p = df.loc[m_te, ["week_anchor", TARGET_FWD]].reset_index(drop=True)
        p["y_w"] = p[TARGET_FWD].clip(lo, hi)
        te_meta[fid] = p

    pool_w, pool_raw, wk_w, wk_raw = {}, {}, {}, {}
    for seed in SEEDS:
        for arm, inputs in ARMS.items():
            parts = []
            for fid, _, _, _ in FOLDS:
                d_tr, d_va, d_te = dmats[arm][fid]
                best = None
                for depth in DEPTH_GRID:
                    params = dict(FIXED_PARAMS, **AMENDMENT, seed=seed, max_depth=depth)
                    bst = xgb.train(params, d_tr, num_boost_round=NUM_ROUND_CAP,
                                    evals=[(d_va, "val")],
                                    early_stopping_rounds=EARLY_STOP, verbose_eval=False)
                    val_mse = float(bst.best_score) ** 2
                    if best is None or val_mse < best["val_mse"]:
                        best = {"bst": bst, "val_mse": val_mse,
                                "n_est": int(bst.best_iteration) + 1}
                yhat = best["bst"].predict(d_te, iteration_range=(0, best["n_est"]))
                p = te_meta[fid].copy()
                p["yhat"] = yhat
                parts.append(p)
            preds = pd.concat(parts, ignore_index=True)
            e2w = (preds["y_w"] - preds["yhat"]) ** 2
            e2r = (preds[TARGET_FWD] - preds["yhat"]) ** 2
            pool_w[(seed, arm)] = 1.0 - float(e2w.sum()) / float((preds["y_w"] ** 2).sum())
            pool_raw[(seed, arm)] = 1.0 - float(e2r.sum()) / float((preds[TARGET_FWD] ** 2).sum())
            wk_w[(seed, arm)] = preds.assign(e2=e2w).groupby("week_anchor")["e2"].mean().sort_index()
            wk_raw[(seed, arm)] = preds.assign(e2=e2r).groupby("week_anchor")["e2"].mean().sort_index()

    rows, perseed = [], []
    for tag, pool in (("winsor_eval", pool_w), ("raw_eval", pool_raw)):
        for arm in SENT_ARMS:
            d = [(pool[(s, arm)] - pool[(s, "A0_structured")]) * 100 for s in SEEDS]
            a = np.array(d)
            rows.append({"eval": tag, "arm": arm, "mean_delta_pp": a.mean(),
                         "sd_delta_pp": a.std(ddof=1), "beats_a0_of_10": int((a > 0).sum())})
            for s, v in zip(SEEDS, d):
                perseed.append({"eval": tag, "arm": arm, "seed": s, "delta_pp": v})
        # A0 pooled under this eval
        a0 = np.array([pool[(s, "A0_structured")] * 100 for s in SEEDS])
        rows.append({"eval": tag, "arm": "A0_structured", "mean_delta_pp": np.nan,
                     "sd_delta_pp": np.nan, "beats_a0_of_10": np.nan,
                     "a0_pool_mean_pct": a0.mean()})
    dmrows = []
    for tag, wk in (("winsor_eval", wk_w), ("raw_eval", wk_raw)):
        for arm in SENT_ARMS:
            st, pv = [], []
            for s in SEEDS:
                dd = (wk[(s, "A0_structured")] - wk[(s, arm)]).to_numpy()
                a, b = dm_test(dd)
                st.append(a)
                pv.append(b)
            dmrows.append({"eval": tag, "arm": arm, "mean_dm": float(np.mean(st)),
                           "sig_pos_of_10": sum(1 for a, b in zip(st, pv) if b < ALPHA and a > 0)})
    pd.DataFrame(rows).to_csv(OUT / "tableC2_winsor.csv", index=False)
    pd.DataFrame(perseed).to_csv(OUT / "tableC2_winsor_perseed.csv", index=False)
    pd.DataFrame(dmrows).to_csv(OUT / "tableC2_winsor_dm.csv", index=False)
    pd.DataFrame([{"fold": f, "clip_lo": clips[f][0], "clip_hi": clips[f][1]}
                 for f, _, _, _ in FOLDS]).to_csv(OUT / "tableC2_winsor_clips.csv", index=False)

    # ---- Verification: consistency and reproduction checks ----
    print("  wrote tableC2_winsor.csv (+perseed,+dm,+clips)")
    for r in rows:
        if r["arm"] != "A0_structured":
            print(f"    [{r['eval']}] {r['arm']:<12} {r['mean_delta_pp']:+.4f} "
                  f"(sd {r['sd_delta_pp']:.4f}) beats {r['beats_a0_of_10']}/10")
    for r in dmrows:
        print(f"    DM [{r['eval']}] {r['arm']:<12} mean {r['mean_dm']:+.3f} sig+ {r['sig_pos_of_10']}/10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
