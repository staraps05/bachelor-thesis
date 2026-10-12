# %% 38_numbers_calm.py
# Purpose: Table C.2 calm-period rows: the four-arm single-lag run with a fold-1 split that never sees 2020.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: Data/processed/numbers_check/ tableC2_calm.csv, tableC2_calm_perseed.csv, tableC2_calm_diagnostics.csv
# Notes:
# - Loads pipeline/27 via importlib for single_lag_frame, dm_test and the fit config.
# - Only the fold-1 train/validation masks differ from pipeline/26; week sets disjoint.
# - Diagnostics CSV: single-tree fold-1 cells and cells with predictions identical to A0.

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
EARLY_STOP, SEEDS = c27.EARLY_STOP, c27.SEEDS
TARGET_FWD, INPUTS_30 = c27.TARGET_FWD, c27.INPUTS_30
SENT_ARMS, ALPHA = c27.SENT_ARMS, c27.ALPHA


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== 38: calm-period validation xgb {xgb.__version__} ===")
    df = single_lag_frame()
    ARMS = {
        "A0_structured": list(INPUTS_30),
        "A1_lm": list(INPUTS_30) + ["lm_mean", "lm_std", "lm_range", "lm_kurt", "had_news"],
        "A2_finbert": list(INPUTS_30) + ["finbert_mean", "finbert_std", "finbert_range",
                                         "finbert_kurt", "had_news"],
        "A3_fingpt": list(INPUTS_30) + ["fingpt_mean", "fingpt_std", "fingpt_range",
                                        "fingpt_kurt", "had_news"],
    }
    wa = pd.to_datetime(df["week_anchor"])
    yr, mo = wa.dt.year, wa.dt.month
    # fold 1: train 2018 + H1-2019, validate H2-2019, test 2021
    split_masks = {
        1: ((yr == 2018) | ((yr == 2019) & (mo <= 6)), (yr == 2019) & (mo >= 7), yr == 2021),
        2: (yr.isin({2018, 2019, 2020}), yr == 2021, yr == 2022),
        3: (yr.isin({2018, 2019, 2020, 2021}), yr == 2022, yr == 2023),
    }
    for fid, (a, b, c) in split_masks.items():
        w = [set(df.loc[m, "week_anchor"]) for m in (a, b, c)]
    # Verification (in place): fold week sets disjoint (inside loop)
        if w[0] & w[1] or w[0] & w[2] or w[1] & w[2]:
            stop(f"fold {fid}: week sets not disjoint")
    dmats = {}
    for arm, inputs in ARMS.items():
        dmats[arm] = {}
        for fid in (1, 2, 3):
            m_tr, m_va, m_te = split_masks[fid]
            dmats[arm][fid] = (
                xgb.DMatrix(df.loc[m_tr, inputs], label=df.loc[m_tr, TARGET_FWD]),
                xgb.DMatrix(df.loc[m_va, inputs], label=df.loc[m_va, TARGET_FWD]),
                xgb.DMatrix(df.loc[m_te, inputs]))
    te_meta = {fid: df.loc[split_masks[fid][2], ["week_anchor", TARGET_FWD]].reset_index(drop=True)
               for fid in (1, 2, 3)}

    pool, wk_e2, sel, fold_yhat = {}, {}, {}, {}
    for seed in SEEDS:
        for arm, inputs in ARMS.items():
            parts, sels = [], []
            for fid in (1, 2, 3):
                d_tr, d_va, d_te = dmats[arm][fid]
                best = None
                for depth in DEPTH_GRID:
                    params = dict(FIXED_PARAMS, **AMENDMENT, seed=seed, max_depth=depth)
                    bst = xgb.train(params, d_tr, num_boost_round=NUM_ROUND_CAP,
                                    evals=[(d_va, "val")], early_stopping_rounds=EARLY_STOP,
                                    verbose_eval=False)
                    val_mse = float(bst.best_score) ** 2
                    if best is None or val_mse < best["val_mse"]:
                        best = {"bst": bst, "val_mse": val_mse, "depth": depth,
                                "n_est": int(bst.best_iteration) + 1}
                yhat = best["bst"].predict(d_te, iteration_range=(0, best["n_est"]))
                sels.append((fid, best["depth"], best["n_est"]))
                fold_yhat[(seed, arm, fid)] = yhat
                p = te_meta[fid].copy()
                p["yhat"] = yhat
                parts.append(p)
            preds = pd.concat(parts, ignore_index=True)
            e2 = (preds[TARGET_FWD] - preds["yhat"]) ** 2
            pool[(seed, arm)] = 1.0 - float(e2.sum()) / float((preds[TARGET_FWD] ** 2).sum())
            wk_e2[(seed, arm)] = preds.assign(e2=e2).groupby("week_anchor")["e2"].mean().sort_index()
            sel[(seed, arm)] = sels

    # fold-1 single-tree cells (of 40)
    single_f1 = sum(1 for s in SEEDS for arm in ARMS
                    if [x for x in sel[(s, arm)] if x[0] == 1][0][2] == 1)
    # bit-identical-to-A0 cells (of 90 = 10 x 3 x 3)
    ident = sum(1 for s in SEEDS for arm in SENT_ARMS for fid in (1, 2, 3)
                if np.array_equal(fold_yhat[(s, arm, fid)], fold_yhat[(s, "A0_structured", fid)]))
    a0_pool = np.array([pool[(s, "A0_structured")] * 100 for s in SEEDS])

    rows, perseed = [], []
    for arm in SENT_ARMS:
        d = [(pool[(s, arm)] - pool[(s, "A0_structured")]) * 100 for s in SEEDS]
        a = np.array(d)
        st, pv = [], []
        for s in SEEDS:
            dd = (wk_e2[(s, "A0_structured")] - wk_e2[(s, arm)]).to_numpy()
            if np.allclose(dd, 0.0):
                continue
            x, y = dm_test(dd)
            st.append(x)
            pv.append(y)
        rows.append({"arm": arm, "mean_delta_pp": a.mean(), "sd_delta_pp": a.std(ddof=1),
                     "beats_a0_of_10": int((a > 0).sum()),
                     "dm_sig_pos_of_10": sum(1 for x, y in zip(st, pv) if y < ALPHA and x > 0)})
        for s, v in zip(SEEDS, d):
            perseed.append({"arm": arm, "seed": s, "delta_pp": v})
    pd.DataFrame(rows).to_csv(OUT / "tableC2_calm.csv", index=False)
    pd.DataFrame(perseed).to_csv(OUT / "tableC2_calm_perseed.csv", index=False)
    pd.DataFrame([{"metric": "fold1_single_tree_cells_of_40", "value": single_f1},
                 {"metric": "bit_identical_to_A0_cells_of_90", "value": ident},
                 {"metric": "a0_pool_mean_pct_calm", "value": float(a0_pool.mean())},
                 {"metric": "a0_pool_sd_pct_calm", "value": float(a0_pool.std(ddof=1))}
                 ]).to_csv(OUT / "tableC2_calm_diagnostics.csv", index=False)

    # ---- Verification: consistency and reproduction checks ----
    print(f"  fold-1 single-tree cells: {single_f1}/40 | bit-identical-to-A0: {ident}/90")
    print(f"  A0 pooled (calm) mean {a0_pool.mean():+.4f}%  (main spec A0 = -1.2992%)")
    for r in rows:
        print(f"    {r['arm']:<12} {r['mean_delta_pp']:+.4f} (sd {r['sd_delta_pp']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
