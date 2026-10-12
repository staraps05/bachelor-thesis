# %% 45_numbers_ridge_singlelag.py
# Purpose: Ridge instrument of pipeline/5_tables/39 under the single-lag pairing: Table C.2 ridge rows, Table C.3, Table C.4.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet, Data/processed/numbers_check/tableC2_ridge_rows.csv, Data/processed/numbers_check/tableC3_ridge_alpha.csv, Data/processed/numbers_check/tableC4_ridge_decomp.csv
# Outputs: Data/processed/numbers_check/ tableC2_ridge_rows_singlelag.csv, tableC3_ridge_alpha_singlelag.csv, tableC3_preregistered_valmse_singlelag.csv, tableC4_ridge_decomp_singlelag.csv
# Notes:
# - Loads 5_tables/39 (run_grid, weekly_e2, grids) and 27 (single_lag_frame) by importlib.
# - Gate 1: the double-lag re-run must reproduce 5_tables/39's CSVs (1e-9, alphas exact).
# - Gate 2: single-lag frame has 157,277 rows, 78,056 test rows, had_news share 0.6620.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
NC = REPO_ROOT / "Data" / "processed" / "numbers_check"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_c39 = _load("ridge39", "5_tables/39_numbers_ridge.py")
_c27 = _load("cov27", "3_models/27_covered_subsample.py")
run_grid, weekly_e2 = _c39.run_grid, _c39.weekly_e2
merge_frame, dm_test = _c39.merge_frame, _c39.dm_test
single_lag_frame = _c27.single_lag_frame
FOLDS, TARGET_FWD = _c39.FOLDS, _c39.TARGET_FWD
INPUTS_30, YEARS, SENT_ARMS = _c39.INPUTS_30, _c39.YEARS, _c39.SENT_ARMS
GRID_PRE, GRID_EXT = _c39.GRID_PRE, _c39.GRID_EXT


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def build_arms():
    """Input lists for the four arms, the A0' coverage arm and the mean-only arms."""
    base = {
        "A0_structured": list(INPUTS_30),
        "A1_lm": list(INPUTS_30) + ["lm_mean", "lm_std", "lm_range", "lm_kurt", "had_news"],
        "A2_finbert": list(INPUTS_30) + ["finbert_mean", "finbert_std", "finbert_range",
                                         "finbert_kurt", "had_news"],
        "A3_fingpt": list(INPUTS_30) + ["fingpt_mean", "fingpt_std", "fingpt_range",
                                        "fingpt_kurt", "had_news"],
    }
    a0p = {"A0p_coverage": list(INPUTS_30) + ["had_news"]}
    mean_only = {f"{a}_meanonly": list(INPUTS_30) + [f"{a.split('_', 1)[1]}_mean", "had_news"]
                 for a in SENT_ARMS}
    return base, a0p, mean_only


def setup(df):
    """Fold masks and train-fold standardisation (mean, sd ddof=0) for df."""
    base, a0p, mean_only = build_arms()
    all_inputs = list(dict.fromkeys(
        list(INPUTS_30) + base["A1_lm"][30:] + base["A2_finbert"][30:34]
        + base["A3_fingpt"][30:34]))
    # Verification (in place): no NaN in inputs (inside setup)
    if int(df[all_inputs].isna().sum().sum()):
        stop("NaN in inputs")
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks, scalers = {}, {}
    for fid, tr_y, va_y, te_y in FOLDS:
        m_tr, m_va, m_te = yr.isin(tr_y), yr.isin(va_y), yr.isin(te_y)
        split_masks[fid] = (m_tr, m_va, m_te)
        mu = df.loc[m_tr, all_inputs].mean()
        sd = df.loc[m_tr, all_inputs].std(ddof=0)
    # Verification (in place): non-zero train sd (inside setup loop)
        if (sd <= 0).any():
            stop(f"fold {fid}: zero train sd")
        scalers[fid] = (mu, sd)
    return base, a0p, mean_only, split_masks, scalers


def compute(df):
    """Run both grids on df; returns the C.2, C.3 (alphas, val-MSE) and C.4 inputs."""
    base, a0p, mean_only, split_masks, scalers = setup(df)
    all_arms = {**base, **a0p, **mean_only}
    r2_pre, _, sel_pre, vm_pre = run_grid(df, split_masks, scalers, base, GRID_PRE)
    wk_pre = {arm: weekly_e2(p) for arm, p in _.items()}
    r2_ext, preds_ext, sel_ext, vm_ext = run_grid(df, split_masks, scalers, all_arms, GRID_EXT)
    wk_ext = {arm: weekly_e2(p) for arm, p in preds_ext.items()}
    # C.2 ridge rows
    c2 = {}
    for gname, r2, wk in (("pre_registered", r2_pre, wk_pre), ("extended", r2_ext, wk_ext)):
        for arm in SENT_ARMS:
            d = (r2[arm]["pool"] - r2["A0_structured"]["pool"]) * 100
            t, p = dm_test((wk["A0_structured"] - wk[arm]).to_numpy())
            c2[(gname, arm)] = (d, t, p, len(wk[arm]))
    # C.3 alphas + valmse
    c3 = {(g, f): (sel_pre if g == "pre_registered" else sel_ext)[("A0_structured", f)]
          for g in ("pre_registered", "extended") for f in (1, 2, 3)}
    vm = {f: vm_pre[("A0_structured", f)] for f in (1, 2, 3)}
    # C.4 decomposition
    d_a0p = (r2_ext["A0p_coverage"]["pool"] - r2_ext["A0_structured"]["pool"]) * 100
    c4 = {}
    for arm in SENT_ARMS:
        m = f"{arm}_meanonly"
        total = (r2_ext[arm]["pool"] - r2_ext["A0_structured"]["pool"]) * 100
        mean_comp = (r2_ext[m]["pool"] - r2_ext["A0p_coverage"]["pool"]) * 100
        remainder = (r2_ext[arm]["pool"] - r2_ext[m]["pool"]) * 100
        c4[arm] = (d_a0p, mean_comp, remainder, total)
    return c2, c3, vm, c4


def main() -> int:
    NC.mkdir(parents=True, exist_ok=True)
    print("=== 45: ridge single-lag re-run ===")

    # ---- gate 1: double-lag re-run must reproduce pipeline/39's CSVs ----
    print("  [GATE 1] reproduce double-lag ridge ...")
    c2_dl, c3_dl, vm_dl, c4_dl = compute(merge_frame())
    # Verification (in place): gate 1: double-lag values reproduce pipeline/39's CSVs
    rr = pd.read_csv(NC / "tableC2_ridge_rows.csv")
    maxdev = 0.0
    for _, row in rr.iterrows():
        d, t, p, _n = c2_dl[(row["grid"], row["arm"])]
        maxdev = max(maxdev, abs(d - row["delta_vs_a0_pp"]), abs(t - row["dm_vs_a0"]),
                     abs(p - row["dm_p"]))
    c3r = pd.read_csv(NC / "tableC3_ridge_alpha.csv")
    for _, row in c3r.iterrows():
        if c3_dl[(row["grid"], int(row["fold"]))] != row["selected_alpha"]:
            stop(f"GATE1 alpha mismatch {row['grid']} f{row['fold']}")
    c4r = pd.read_csv(NC / "tableC4_ridge_decomp.csv").set_index("arm")
    for arm in SENT_ARMS:
        ni, mc, rem, tot = c4_dl[arm]
        maxdev = max(maxdev, abs(ni - c4r.loc[arm, "news_indicator_A0p_minus_A0_pp"]),
                     abs(mc - c4r.loc[arm, "mean_score_beyond_indicator_pp"]),
                     abs(rem - c4r.loc[arm, "remainder_dispersion_pp"]),
                     abs(tot - c4r.loc[arm, "total_arm_minus_A0_pp"]))
    if maxdev > 1e-9:
        stop(f"GATE1 FAIL: double-lag reproduction max dev {maxdev:.3e} > 1e-9")

    # ---- gate 2: single-lag frame row counts and had_news share ----
    df = single_lag_frame()
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    n_test = int(yr.isin(list(YEARS)).sum())
    hn = df["had_news"].mean()
    # Verification (in place): gate 2: single-lag frame rows, test rows, had_news share
    if len(df) != 157_277 or n_test != 78_056 or abs(hn - 0.6620) > 5e-4:
        stop(f"GATE2 FAIL: rows {len(df)}, test {n_test}, had_news {hn:.4f}")

    # ---- single-lag run ----
    c2, c3, vm, c4 = compute(df)

    rows = []
    for gname in ("pre_registered", "extended"):
        for arm in SENT_ARMS:
            d_sl, t_sl, p_sl, nwk = c2[(gname, arm)]
            d_dl, t_dl, p_dl, _ = c2_dl[(gname, arm)]
            rows.append({"grid": gname, "arm": arm,
                         "delta_dl_pp": d_dl, "delta_sl_pp": d_sl,
                         "dm_dl": t_dl, "dm_dl_p": p_dl, "dm_dl_sig5": p_dl < 0.05,
                         "dm_sl": t_sl, "dm_sl_p": p_sl, "dm_sl_sig5": p_sl < 0.05,
                         "n_weeks": nwk})
    pd.DataFrame(rows).to_csv(NC / "tableC2_ridge_rows_singlelag.csv", index=False)

    arows = []
    for gname in ("pre_registered", "extended"):
        for f in (1, 2, 3):
            arows.append({"grid": gname, "fold": f,
                          "alpha_dl": c3_dl[(gname, f)], "alpha_sl": c3[(gname, f)]})
    pd.DataFrame(arows).to_csv(NC / "tableC3_ridge_alpha_singlelag.csv", index=False)

    vrows = []
    for f in (1, 2, 3):
        v = vm[f]
        vrows.append({"fold": f, **{f"valmse_alpha_{g:g}": val for g, val in zip(GRID_PRE, v)},
                      "argmin_alpha": GRID_PRE[int(np.argmin(v))],
                      "still_falling_at_1e4": bool(v[-1] == min(v)),
                      "monotone_decreasing": bool(all(v[i + 1] < v[i] for i in range(len(v) - 1)))})
    pd.DataFrame(vrows).to_csv(NC / "tableC3_preregistered_valmse_singlelag.csv", index=False)

    drows = []
    for arm in SENT_ARMS:
        ni_dl, mc_dl, rem_dl, tot_dl = c4_dl[arm]
        ni, mc, rem, tot = c4[arm]
        drows.append({"arm": arm,
                      "news_ind_dl": ni_dl, "news_ind_sl": ni,
                      "mean_score_dl": mc_dl, "mean_score_sl": mc,
                      "remainder_dl": rem_dl, "remainder_sl": rem,
                      "total_dl": tot_dl, "total_sl": tot})
    pd.DataFrame(drows).to_csv(NC / "tableC4_ridge_decomp_singlelag.csv", index=False)

    print("  wrote *_singlelag.csv (ridge rows, alphas, valmse, decomp)")
    print("  --- C.2 ridge rows (dl -> sl, delta pp | DM[p]) ---")
    for r in rows:
        print(f"    {r['grid']:<14} {r['arm']:<12} {r['delta_dl_pp']:+.4f} -> {r['delta_sl_pp']:+.4f} | "
              f"DM dl {r['dm_dl']:+.3f}[{r['dm_dl_p']:.4f}] sl {r['dm_sl']:+.3f}[{r['dm_sl_p']:.4f}]")
    print("  --- C.3 alphas (dl -> sl) ---")
    for r in arows:
        print(f"    {r['grid']:<14} f{r['fold']}: {r['alpha_dl']:g} -> {r['alpha_sl']:g}")
    print("  pre-reg still falling at 1e4 (sl):", [r["still_falling_at_1e4"] for r in vrows])
    print("  --- C.4 decomp sl (A0'-A0 | mean | remainder | total) ---")
    for r in drows:
        print(f"    {r['arm']:<12} {r['news_ind_sl']:+.4f} | {r['mean_score_sl']:+.4f} | "
              f"{r['remainder_sl']:+.4f} | {r['total_sl']:+.4f}")

    # ---- Verification: consistency and reproduction checks ----
    print(f"  [PASS] GATE 1: double-lag values reproduce exactly (max dev {maxdev:.2e})")
    print(f"  [PASS] GATE 2: single-lag frame {len(df):,} rows, {n_test:,} test rows, "
          f"had_news {hn:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
