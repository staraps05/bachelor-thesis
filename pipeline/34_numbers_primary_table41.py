# %% 34_numbers_primary_table41.py
# Purpose: Reproduce the primary single-lag four-arm XGBoost run and save the
#          full-precision sources behind Table 4.1 (Panel A R2 and SD, Panel B
#          deltas vs A0, DM tests, Sig counts) and Table C.1 (pairwise arm
#          differences).
# Inputs:  Data/processed/model_frame_structured.parquet (via single_lag_frame)
#          Data/processed/features_lm.parquet
#          Data/processed/features_finbert.parquet
#          Data/processed/features_fingpt.parquet
# Outputs: Data/processed/numbers_check/table41_r2_by_seed.csv
#          Data/processed/numbers_check/table41_panelAB_summary.csv
#          Data/processed/numbers_check/table41_dm_by_seed.csv
#          Data/processed/numbers_check/table41_dm_summary.csv
#          Data/processed/numbers_check/tableC1_pairwise.csv
# Notes:
# - Loads pipeline/27_covered_subsample.py via importlib for single_lag_frame,
#   dm_test and the fit configuration (params, depth grid, folds, seeds, input
#   lists), and pipeline/30_per_year_primary.py for YEARS, ARM_ORDER and the
#   reference per-seed R2 of the reproduction gate.
# - Gate: every per-seed pooled R2 (40 seed x arm cells) must match the
#   reference from the pipeline/26 run log within GATE_TOL, else the run stops
#   before anything is written.
# - R2 per GKX eq. 19 with an un-demeaned denominator: per year it uses that
#   year's own sum(y^2); pooled is the ratio of the summed SSE and sum(y^2).
# - DM test: Newey-West (lag 4) t on the weekly cross-sectional mean squared-
#   error difference A0 minus arm, per seed. Sig = seeds with p < ALPHA and
#   stat > 0 (signed); an unsigned count (p < ALPHA only) is stored alongside.
# - Deltas and pairwise differences are within-seed R2 differences in pp,
#   mean and sd (ddof=1) over seeds; "beats_a0_of_10" counts seeds with
#   delta > 0.

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


_c27 = _load("cov27", "27_covered_subsample.py")
_c30 = _load("c30", "30_per_year_primary.py")
single_lag_frame = _c27.single_lag_frame
dm_test = _c27.dm_test
FIXED_PARAMS, AMENDMENT = _c27.FIXED_PARAMS, _c27.AMENDMENT
DEPTH_GRID, NUM_ROUND_CAP = _c27.DEPTH_GRID, _c27.NUM_ROUND_CAP
EARLY_STOP, FOLDS, SEEDS = _c27.EARLY_STOP, _c27.FOLDS, _c27.SEEDS
TARGET_FWD, INPUTS_30 = _c27.TARGET_FWD, _c27.INPUTS_30
SENT_ARMS, ALPHA = _c27.SENT_ARMS, _c27.ALPHA
YEARS = _c30.YEARS
C26_SEED_R2, GATE_TOL, ARM_ORDER = _c30.C26_SEED_R2, _c30.GATE_TOL, _c30.ARM_ORDER


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def stats(v):
    a = np.asarray(v, float)
    return a.mean(), a.std(ddof=1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== 34: PRIMARY single-lag reproduction (xgboost {xgb.__version__}) ===")

    df = single_lag_frame()
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
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {}
    for fid, tr_y, va_y, te_y in FOLDS:
        m_tr, m_va, m_te = yr.isin(tr_y), yr.isin(va_y), yr.isin(te_y)
        split_masks[fid] = (m_tr, m_va, m_te)
    dmats = {}
    for arm, inputs in ARMS.items():
        dmats[arm] = {}
        for fid, _, _, _ in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            dmats[arm][fid] = (
                xgb.DMatrix(df.loc[m_tr, inputs], label=df.loc[m_tr, TARGET_FWD]),
                xgb.DMatrix(df.loc[m_va, inputs], label=df.loc[m_va, TARGET_FWD]),
                xgb.DMatrix(df.loc[m_te, inputs]))
    te_meta = {fid: df.loc[split_masks[fid][2],
                           ["week_anchor", TARGET_FWD]].reset_index(drop=True)
               for fid, _, _, _ in FOLDS}
    te_year = {fid: sorted(te_y)[0] for fid, _, _, te_y in FOLDS}

    res = {}
    max_dev = 0.0
    for seed in SEEDS:
        for arm, inputs in ARMS.items():
            parts = []
            for fid, _, _, _ in FOLDS:
                d_tr, d_va, d_te = dmats[arm][fid]
                best = None
                for depth in DEPTH_GRID:
                    params = dict(FIXED_PARAMS, **AMENDMENT, seed=seed,
                                  max_depth=depth)
                    bst = xgb.train(params, d_tr, num_boost_round=NUM_ROUND_CAP,
                                    evals=[(d_va, "val")],
                                    early_stopping_rounds=EARLY_STOP,
                                    verbose_eval=False)
                    val_mse = float(bst.best_score) ** 2
                    if best is None or val_mse < best["val_mse"]:
                        best = {"bst": bst, "val_mse": val_mse,
                                "n_est": int(bst.best_iteration) + 1}
                yhat = best["bst"].predict(d_te, iteration_range=(0, best["n_est"]))
    # Verification (in place): finite predictions (inside loop)
                if not np.isfinite(yhat).all():
                    stop(f"seed {seed} {arm} fold {fid}: NaN/inf")
                p = te_meta[fid].copy()
                p["yhat"] = yhat
                p["test_year"] = te_year[fid]
                parts.append(p)
            preds = pd.concat(parts, ignore_index=True)
            e2 = (preds[TARGET_FWD] - preds["yhat"]) ** 2
            sse_y, ssy_y, year_r2 = {}, {}, {}
            for y in YEARS:
                m = preds["test_year"] == y
                sse_y[y] = float(e2[m].sum())
                ssy_y[y] = float((preds.loc[m, TARGET_FWD] ** 2).sum())
                year_r2[y] = 1.0 - sse_y[y] / ssy_y[y]
            pool = 1.0 - sum(sse_y.values()) / sum(ssy_y.values())
            wk = (preds.assign(e2=e2).groupby("week_anchor")["e2"]
                  .mean().sort_index())
            res[(seed, arm)] = {"pool": pool, "year": year_r2, "wk": wk}
        # reproduction gate against the per-seed R2 of the pipeline/26 run log
    # Verification (in place): per-seed reproduction gate (inside loop)
        for arm in ARM_ORDER:
            dev = abs(res[(seed, arm)]["pool"] * 100 - C26_SEED_R2[seed][arm])
            max_dev = max(max_dev, dev)
            if dev > GATE_TOL:
                stop(f"GATE FAIL seed {seed} {arm}: dev {dev:.2e} > {GATE_TOL:.0e}")

    # ---- per-seed long table (R2 + delta), scopes pooled/2021/2022/2023 ----
    rows = []
    for seed in SEEDS:
        for arm in ARM_ORDER:
            scopes = [("pooled", res[(seed, arm)]["pool"])] + \
                     [(str(y), res[(seed, arm)]["year"][y]) for y in YEARS]
            for scope, r2 in scopes:
                a0 = (res[(seed, "A0_structured")]["pool"] if scope == "pooled"
                      else res[(seed, "A0_structured")]["year"][int(scope)])
                rows.append({"seed": seed, "arm": arm, "scope": scope,
                             "r2_pct": r2 * 100,
                             "delta_vs_a0_pp": (np.nan if arm == "A0_structured"
                                                else (r2 - a0) * 100)})
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(OUT / "table41_r2_by_seed.csv", index=False)

    # ---- Panel A / Panel B summary ----
    scopes = ["pooled"] + [str(y) for y in YEARS]
    sumrows = []
    for arm in ARM_ORDER:
        for scope in scopes:
            r2v = by_seed[(by_seed.arm == arm) & (by_seed.scope == scope)]["r2_pct"]
            mr, sr = stats(r2v)
            if arm == "A0_structured":
                md = sd = nb = np.nan
            else:
                dv = by_seed[(by_seed.arm == arm) &
                             (by_seed.scope == scope)]["delta_vs_a0_pp"]
                md, sd = stats(dv)
                nb = int((dv > 0).sum())
            sumrows.append({"arm": arm, "scope": scope, "mean_r2_pct": mr,
                            "sd_r2_pct": sr, "mean_delta_pp": md,
                            "sd_delta_pp": sd, "beats_a0_of_10": nb})
    pd.DataFrame(sumrows).to_csv(OUT / "table41_panelAB_summary.csv", index=False)

    # ---- DM + Sig (Panel B, weekly mean squared errors) ----
    dm_rows, dm_summary = [], []
    for arm in SENT_ARMS:
        st_, pv_ = [], []
        for seed in SEEDS:
            d = (res[(seed, "A0_structured")]["wk"] - res[(seed, arm)]["wk"]).to_numpy()
            s, p = dm_test(d)
            st_.append(s)
            pv_.append(p)
            dm_rows.append({"arm": arm, "seed": seed, "dm_stat": s, "p_value": p})
        k_signed = sum(1 for s, p in zip(st_, pv_) if p < ALPHA and s > 0)
        k_unsigned = sum(1 for p in pv_ if p < ALPHA)
        dm_summary.append({"arm": arm, "mean_dm": float(np.mean(st_)),
                           "min_dm": float(np.min(st_)), "max_dm": float(np.max(st_)),
                           "sig_signed_of_10": k_signed,
                           "sig_unsigned_of_10": k_unsigned})
    pd.DataFrame(dm_rows).to_csv(OUT / "table41_dm_by_seed.csv", index=False)
    pd.DataFrame(dm_summary).to_csv(OUT / "table41_dm_summary.csv", index=False)

    # ---- Table C.1 pairwise (within-seed R2 diffs) ----
    pairs = [("A1_lm", "A2_finbert"), ("A1_lm", "A3_fingpt"),
             ("A2_finbert", "A3_fingpt")]
    prows = []
    for a, b in pairs:
        for scope in scopes:
            def r2of(seed, arm, scope):
                return (res[(seed, arm)]["pool"] if scope == "pooled"
                        else res[(seed, arm)]["year"][int(scope)])
            d = [(r2of(s, a, scope) - r2of(s, b, scope)) * 100 for s in SEEDS]
            m, sd = stats(d)
            prows.append({"pair": f"{a}-{b}", "scope": scope,
                          "mean_pp": m, "sd_pp": sd,
                          "abs_mean_over_sd": (abs(m) / sd if sd > 0 else 0.0)})
    pd.DataFrame(prows).to_csv(OUT / "tableC1_pairwise.csv", index=False)

    # ---- Verification: consistency and reproduction checks ----
    print(f"  [PASS] reproduction gate: 40/40 pooled cells match pipeline/26 log "
          f"(max |dev| {max_dev:.2e} pp)")
    print(f"  [PASS] wrote 6 CSVs to {OUT.relative_to(REPO_ROOT)}")
    for arm in SENT_ARMS:
        s = next(r for r in dm_summary if r["arm"] == arm)
        print(f"    {arm:<12} DM mean {s['mean_dm']:+.3f}  Sig {s['sig_signed_of_10']}/10")
    print(f"  max_dev_pp={max_dev:.3e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
