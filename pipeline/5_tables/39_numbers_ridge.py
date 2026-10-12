# %% 39_numbers_ridge.py
# Purpose: Round-1 ridge results on the double-lag frame: Table C.2 ridge rows, Table C.3 and Table C.4.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: Data/processed/numbers_check/ tableC2_ridge_rows.csv, tableC3_ridge_alpha.csv, tableC3_preregistered_valmse.csv, tableC4_ridge_decomp.csv
# Notes:
# - Loads pipeline/3_models/20 via importlib; loaded by pipeline/5_tables/45 (run_grid, weekly_e2, grids).
# - Double-lag pairing as in pipeline/4_analyses/21; pipeline/5_tables/45 is the single-lag counterpart.
# - Closed-form ridge, no seeds; alpha per fold by validation MSE on both grids.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "Data" / "processed" / "numbers_check"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_c20 = _load("seed20", "3_models/20_seed_stability.py")
merge_frame, dm_test = _c20.merge_frame, _c20.dm_test
FOLDS, TARGET_FWD = _c20.FOLDS, _c20.TARGET_FWD
INPUTS_30, YEARS, SENT_ARMS = _c20.INPUTS_30, _c20.YEARS, _c20.SENT_ARMS
r2_oos = _c20.r2_oos

GRID_PRE = [0.1, 1, 10, 100, 1000, 10000]
GRID_EXT = [1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7]


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def weekly_e2(p):
    q = p.copy()
    q["e2"] = (q[TARGET_FWD] - q["yhat"]) ** 2
    return q.groupby("week_anchor")["e2"].mean().sort_index()


def run_grid(df, split_masks, scalers, arms, grid):
    """Closed-form ridge per arm and fold, alpha chosen by minimum validation MSE
    (ties -> smaller alpha). Returns (r2, preds, sel_alpha, valmse[(arm, fid)])."""
    preds, sel_alpha, valmse = {}, {}, {}
    for arm, cols in arms.items():
        parts = []
        for fid, tr_y, va_y, te_y in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            mu, sd = scalers[fid]
            Xtr = ((df.loc[m_tr, cols] - mu[cols]) / sd[cols]).to_numpy(float)
            Xva = ((df.loc[m_va, cols] - mu[cols]) / sd[cols]).to_numpy(float)
            Xte = ((df.loc[m_te, cols] - mu[cols]) / sd[cols]).to_numpy(float)
            ytr = df.loc[m_tr, TARGET_FWD].to_numpy(float)
            yva = df.loc[m_va, TARGET_FWD].to_numpy(float)
            ybar = ytr.mean()
            XtX = Xtr.T @ Xtr
            Xty = Xtr.T @ (ytr - ybar)
            eye = np.eye(len(cols))
            best, vm = None, []
            for a in grid:
                b = np.linalg.solve(XtX + a * eye, Xty)
                vmse = float(((yva - (ybar + Xva @ b)) ** 2).mean())
                vm.append(vmse)
                if best is None or vmse < best["val_mse"]:   # ties -> smaller
                    best = {"alpha": a, "beta": b, "val_mse": vmse}
            sel_alpha[(arm, fid)] = best["alpha"]
            valmse[(arm, fid)] = vm
            p = df.loc[m_te, ["week_anchor", TARGET_FWD]].copy()
            p["yhat"] = ybar + Xte @ best["beta"]
            p["test_year"] = sorted(te_y)[0]
            parts.append(p)
        preds[arm] = pd.concat(parts, ignore_index=True)
        if not np.isfinite(preds[arm]["yhat"]).all():
            stop(f"{arm}: NaN/inf ridge predictions")
    r2 = {arm: {"pool": r2_oos(p[TARGET_FWD], p["yhat"])}
          for arm, p in preds.items()}
    for arm, p in preds.items():
        for y in YEARS:
            s = p[p["test_year"] == y]
            r2[arm][y] = r2_oos(s[TARGET_FWD], s["yhat"])
    return r2, preds, sel_alpha, valmse


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== 39: RIDGE both grids + A0' decomposition (double-lag frame) ===")
    df = merge_frame()
    base = {
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
    a0p = {"A0p_coverage": list(INPUTS_30) + ["had_news"]}
    mean_only = {f"{a}_meanonly": list(INPUTS_30)
                 + [f"{a.split('_', 1)[1]}_mean", "had_news"] for a in SENT_ARMS}
    all_arms = {**base, **a0p, **mean_only}

    ALL_INPUTS = list(dict.fromkeys(
        list(INPUTS_30) + base["A1_lm"][30:] + base["A2_finbert"][30:34]
        + base["A3_fingpt"][30:34]))
    # Verification (in place): no NaN in inputs
    if int(df[ALL_INPUTS].isna().sum().sum()):
        stop("NaN in inputs")
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks, scalers = {}, {}
    for fid, tr_y, va_y, te_y in FOLDS:
        m_tr, m_va, m_te = yr.isin(tr_y), yr.isin(va_y), yr.isin(te_y)
        split_masks[fid] = (m_tr, m_va, m_te)
        mu = df.loc[m_tr, ALL_INPUTS].mean()
        sd = df.loc[m_tr, ALL_INPUTS].std(ddof=0)
    # Verification (in place): non-zero train sd (inside loop)
        if (sd <= 0).any():
            stop(f"fold {fid}: zero train sd")
        scalers[fid] = (mu, sd)

    # ---------- pre-registered grid (arms only) ----------
    r2_pre, preds_pre, sel_pre, vm_pre = run_grid(df, split_masks, scalers,
                                                  base, GRID_PRE)
    wk_pre = {arm: weekly_e2(p) for arm, p in preds_pre.items()}
    # ---------- extended grid (all arms) ----------
    r2_ext, preds_ext, sel_ext, vm_ext = run_grid(df, split_masks, scalers,
                                                  all_arms, GRID_EXT)
    wk_ext = {arm: weekly_e2(p) for arm, p in preds_ext.items()}

    # ---- Table C.2 ridge rows + Table C.3 alphas ----
    c2rows, c3rows = [], []
    for grid_name, r2, sel, wk in (("pre_registered", r2_pre, sel_pre, wk_pre),
                                   ("extended", r2_ext, sel_ext, wk_ext)):
        for arm in SENT_ARMS:
            d = (r2[arm]["pool"] - r2["A0_structured"]["pool"]) * 100
            dd = (wk["A0_structured"] - wk[arm]).to_numpy()
            t, p = dm_test(dd)
            c2rows.append({"grid": grid_name, "arm": arm, "delta_vs_a0_pp": d,
                           "dm_vs_a0": t, "dm_p": p})
        for fid in (1, 2, 3):
            c3rows.append({"grid": grid_name, "fold": fid,
                           "selected_alpha": sel[("A0_structured", fid)]})
    pd.DataFrame(c2rows).to_csv(OUT / "tableC2_ridge_rows.csv", index=False)
    pd.DataFrame(c3rows).to_csv(OUT / "tableC3_ridge_alpha.csv", index=False)

    # ---- pre-registered grid: is validation error still falling at alpha=1e4? ----
    vrows = []
    for fid in (1, 2, 3):
        vm = vm_pre[("A0_structured", fid)]
        falling = vm[-1] == min(vm)            # min at the last (largest) alpha
        strictly = all(vm[i + 1] < vm[i] for i in range(len(vm) - 1))
        vrows.append({"fold": fid,
                      **{f"valmse_alpha_{g:g}": v for g, v in zip(GRID_PRE, vm)},
                      "argmin_alpha": GRID_PRE[int(np.argmin(vm))],
                      "still_falling_at_1e4": bool(falling),
                      "monotone_decreasing": bool(strictly)})
    pd.DataFrame(vrows).to_csv(OUT / "tableC3_preregistered_valmse.csv", index=False)

    # ---- Table C.4 decomposition (extended grid) ----
    d_a0p = (r2_ext["A0p_coverage"]["pool"] - r2_ext["A0_structured"]["pool"]) * 100
    drows = []
    for arm in SENT_ARMS:
        m = f"{arm}_meanonly"
        total = (r2_ext[arm]["pool"] - r2_ext["A0_structured"]["pool"]) * 100
        mean_comp = (r2_ext[m]["pool"] - r2_ext["A0p_coverage"]["pool"]) * 100
        remainder = (r2_ext[arm]["pool"] - r2_ext[m]["pool"]) * 100
        drows.append({"arm": arm, "news_indicator_A0p_minus_A0_pp": d_a0p,
                      "mean_score_beyond_indicator_pp": mean_comp,
                      "remainder_dispersion_pp": remainder,
                      "total_arm_minus_A0_pp": total,
                      "sum_check_pp": d_a0p + mean_comp + remainder})
    pd.DataFrame(drows).to_csv(OUT / "tableC4_ridge_decomp.csv", index=False)

    print("  pre-registered deltas:", {a["arm"].split("_")[0]: round(a["delta_vs_a0_pp"], 4)
                                        for a in c2rows if a["grid"] == "pre_registered"})
    print("  extended deltas:", {a["arm"].split("_")[0]: round(a["delta_vs_a0_pp"], 4)
                                  for a in c2rows if a["grid"] == "extended"})
    print(f"  A0'-A0 (extended) = {d_a0p:+.4f}pp")
    print("  pre-registered A0 selected alphas:",
          [sel_pre[("A0_structured", f)] for f in (1, 2, 3)],
          "| still falling at 1e4:", [r["still_falling_at_1e4"] for r in vrows])
    print("  extended A0 selected alphas:",
          [sel_ext[("A0_structured", f)] for f in (1, 2, 3)])

    # ---- Verification: consistency and reproduction checks ----
    print(f"  [PASS] wrote 4 CSVs to {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
