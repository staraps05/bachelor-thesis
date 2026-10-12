# %% 21_ridge_exploration.py
# Purpose: Exploratory ridge-regression counterpart to the four XGBoost arms, plus univariate checks
#          of the three mean scores on the test rows.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: prints only
# Notes:
# - Loads 3_models/20 via importlib: merge_frame (SHA-checks inputs), dm_test, folds.
# - Double-lag pairing (ret_excess_fwd vs sentiment at key t); single-lag rerun is 26.

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "seed20", REPO_ROOT / "pipeline" / "3_models" / "20_seed_stability.py")
_c20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c20)
merge_frame, dm_test = _c20.merge_frame, _c20.dm_test
FOLDS, KEYS, TARGET_FWD = _c20.FOLDS, _c20.KEYS, _c20.TARGET_FWD
INPUTS_30, YEARS, SENT_ARMS = _c20.INPUTS_30, _c20.YEARS, _c20.SENT_ARMS
ALPHA = _c20.ALPHA
r2_oos = _c20.r2_oos

ALPHA_GRID = [0.1, 1, 10, 100, 1000, 10000]   # ties -> smaller alpha
SCORERS = ["lm_mean", "finbert_mean", "fingpt_mean"]

# XGBoost-A0 R2 (%) and seed sd (pp) from the pipeline/18 and pipeline/20 logs
XGB_A0_SINGLE = -1.5412
XGB_A0_MULTI = -1.2992
XGB_A0_SD = 0.1657


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def norm_p(t: float) -> float:
    """Two-sided normal p-value of a t statistic."""
    return math.erfc(abs(t) / math.sqrt(2.0))


def main() -> int:
    # ---- merge and arm definitions ----
    print("=== STAGE 0: rebuild (pipeline/19/20 merge) + purpose ===")
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
    ALL_INPUTS = list(INPUTS_30) + [c for c in ARMS["A1_lm"][30:]] + \
        ARMS["A2_finbert"][30:34] + ARMS["A3_fingpt"][30:34]
    ALL_INPUTS = list(dict.fromkeys(ALL_INPUTS))          # 30+5+4+4 = 43
    yr = pd.to_datetime(df["week_anchor"]).dt.year

    # ---- standardisation fitted on train rows per fold ----
    print("\n=== STAGE 1: leakage-safe standardisation (train-fitted only) ===")
    # Verification (in place): no NaN in inputs
    if int(df[ALL_INPUTS].isna().sum().sum()):
        stop("NaN in inputs")
    print(f"  [PASS] 0 NaN in all {len(ALL_INPUTS)} candidate inputs")
    print("  [INFO] 22 chars already rank-normalised [-1,1]; macro raw with "
          "very different scales (e.g. svar ~1e-3) -> z-scoring mandatory "
          "for a penalised linear model. Scaler: mean/sd (ddof=0) fitted on "
          "TRAIN rows of each fold ONLY, applied unchanged to val and test; "
          "target NOT standardised.")
    split_masks, scalers = {}, {}
    for fid, tr_y, va_y, te_y in FOLDS:
        m_tr, m_va, m_te = yr.isin(tr_y), yr.isin(va_y), yr.isin(te_y)
        split_masks[fid] = (m_tr, m_va, m_te)
        mu = df.loc[m_tr, ALL_INPUTS].mean()
        sd = df.loc[m_tr, ALL_INPUTS].std(ddof=0)
    # Verification (in place): non-zero train sd
        if (sd <= 0).any():
            stop(f"fold {fid}: zero train sd in "
                 f"{list(sd.index[sd <= 0])}")
        scalers[fid] = (mu, sd)
        ex = {c: (mu[c], sd[c]) for c in ("svar", "dp", "market_equity",
                                          "lm_mean", "had_news")}
        exs = ", ".join(f"{c} mu={m:.4g}/sd={s:.4g}" for c, (m, s) in
                        ex.items())
        print(f"  fold {fid}: train n={int(m_tr.sum()):,}; sd range across "
              f"inputs [{sd.min():.4g} ({sd.idxmin()}), {sd.max():.4g} "
              f"({sd.idxmax()})]; e.g. {exs}")

    # ---- univariate checks on test rows ----
    print("\n=== STAGE 2: unshrunken checks on TEST rows 2021-23 "
          "(no tuning, no selection) ===")
    m_test = yr.isin(YEARS)
    sub = df.loc[m_test & (df["had_news"] == 1),
                 ["week_anchor", TARGET_FWD] + SCORERS].copy()
    print(f"  test rows with had_news==1: n = {len(sub):,} "
          f"({int(m_test.sum()):,} test rows total)")

    print("  (a) pooled univariate OLS of ret_excess_fwd on score "
          "(clustered-by-week t):")
    weeks_arr = sub["week_anchor"].to_numpy()
    uniq_weeks = pd.unique(weeks_arr)
    for sc in SCORERS:
        x = sub[sc].to_numpy(float)
        y = sub[TARGET_FWD].to_numpy(float)
        Z = np.column_stack([np.ones(len(x)), x])
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
        beta = ZtZ_inv @ (Z.T @ y)
        e = y - Z @ beta
        meat = np.zeros((2, 2))
        for w in uniq_weeks:
            idx = weeks_arr == w
            s = Z[idx].T @ e[idx]
            meat += np.outer(s, s)
        V = ZtZ_inv @ meat @ ZtZ_inv
        t = beta[1] / math.sqrt(V[1, 1])
        print(f"    {sc:<13} b = {beta[1] * 100:+.4f}% per unit score | "
              f"t(week-clustered) = {t:+.2f} | p = {norm_p(t):.4f} | "
              f"n = {len(x):,} ({len(uniq_weeks)} weeks)")

    print("  (b) weekly quintile sorts among had_news==1 firms "
          "(Q5=highest score; NW lag-4 t on the ~155-week spread series):")
    for sc in SCORERS:
        rows = []
        for w, g in sub.groupby("week_anchor"):
            r = g[sc].rank(method="first")
            q = pd.qcut(r, 5, labels=False)
            rows.append((g.loc[q == 4, TARGET_FWD].mean(),
                         g.loc[q == 0, TARGET_FWD].mean()))
        q5 = np.array([a for a, _ in rows])
        q1 = np.array([b for _, b in rows])
        spread = q5 - q1
        t, p = dm_test(spread)   # mean / NW(4) se
        print(f"    {sc:<13} Q5 {q5.mean() * 100:+.4f}%  Q1 "
              f"{q1.mean() * 100:+.4f}%  spread {spread.mean() * 100:+.4f}% "
              f"| NW t = {t:+.2f} | p = {p:.4f} | {len(spread)} weeks")

    # ---- ridge per arm and fold ----
    print("\n=== STAGE 3: ridge arms, same folds/protocol ===")
    print(f"  [INFO] closed-form ridge (numpy solve) -> DETERMINISTIC; no "
          f"seed loop needed - sidesteps the seed instability that dominated "
          f"pipeline/20 (A0 sd {XGB_A0_SD:.4f}pp). alpha grid {ALPHA_GRID} "
          f"(pre-registered; tuned per fold/arm on VALIDATION MSE; ties -> "
          f"smaller alpha; test never touched). Intercept unpenalised "
          f"(y centred on train mean).")
    preds_by_arm, coefs = {}, {}
    for arm, cols in ARMS.items():
        parts = []
        for fid, tr_y, va_y, te_y in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            mu, sd = scalers[fid]
            Xtr = ((df.loc[m_tr, cols] - mu[cols]) / sd[cols]).to_numpy(float)
            Xva = ((df.loc[m_va, cols] - mu[cols]) / sd[cols]).to_numpy(float)
            Xte = ((df.loc[m_te, cols] - mu[cols]) / sd[cols]).to_numpy(float)
    # Verification (in place): standardised train mean is zero
            colmeans = np.abs(Xtr.mean(axis=0))
            if colmeans.max() > 1e-10:
                stop(f"{arm} fold {fid}: standardised train mean != 0")
            ytr = df.loc[m_tr, TARGET_FWD].to_numpy(float)
            yva = df.loc[m_va, TARGET_FWD].to_numpy(float)
            ybar = ytr.mean()
            XtX = Xtr.T @ Xtr
            Xty = Xtr.T @ (ytr - ybar)
            eye = np.eye(len(cols))
            best = None
            for a in ALPHA_GRID:
                b = np.linalg.solve(XtX + a * eye, Xty)
                val_mse = float(((yva - (ybar + Xva @ b)) ** 2).mean())
                if best is None or val_mse < best["val_mse"]:
                    best = {"alpha": a, "beta": b, "val_mse": val_mse}
            print(f"  {arm:<14} fold {fid}: alpha = {best['alpha']:>7} | "
                  f"val MSE {best['val_mse']:.8f}")
            coefs[(arm, fid)] = pd.Series(best["beta"], index=cols)
            p = df.loc[m_te, ["week_anchor", TARGET_FWD]].copy()
            p["yhat"] = ybar + Xte @ best["beta"]
            p["test_year"] = sorted(te_y)[0]
            parts.append(p)
        preds_by_arm[arm] = pd.concat(parts, ignore_index=True)
    # Verification (in place): finite ridge predictions
        if not np.isfinite(preds_by_arm[arm]["yhat"]).all():
            stop(f"{arm}: NaN/inf ridge predictions")

    # ---- R2, DM tests and coefficients ----
    print("\n=== STAGE 4: ridge results + THE coefficients ===")
    r2 = {}
    print(f"  {'arm':<14} {'pooled R2':>10} {'dR2 vs A0':>10} "
          f"{'2021':>9} {'2022':>9} {'2023':>9}")
    for arm, p in preds_by_arm.items():
        r2[arm] = {"pool": r2_oos(p[TARGET_FWD], p["yhat"])}
        for y in YEARS:
            s = p[p["test_year"] == y]
            r2[arm][y] = r2_oos(s[TARGET_FWD], s["yhat"])
    for arm in ARMS:
        d = (r2[arm]["pool"] - r2["A0_structured"]["pool"]) * 100
        print(f"  {arm:<14} {r2[arm]['pool'] * 100:>+9.4f}% {d:>+9.4f}pp "
              f"{r2[arm][2021] * 100:>+8.4f}% {r2[arm][2022] * 100:>+8.4f}% "
              f"{r2[arm][2023] * 100:>+8.4f}%")

    def weekly_e2(p):
        """Mean squared error per week, sorted by week."""
        q = p.copy()
        q["e2"] = (q[TARGET_FWD] - q["yhat"]) ** 2
        return q.groupby("week_anchor")["e2"].mean().sort_index()

    wk = {arm: weekly_e2(p) for arm, p in preds_by_arm.items()}
    print("  adapted DM vs ridge-A0 (NW lag 4; DM > 0 = arm beats A0):")
    dm_res = {}
    for arm in SENT_ARMS:
        d = (wk["A0_structured"] - wk[arm]).to_numpy()
        t, p = dm_test(d)
        dm_res[arm] = (t, p)
        print(f"    {arm:<14} DM = {t:+.4f} | p = {p:.4f} | n = {len(d)}")

    print("\n  KEY OUTPUT - standardised sentiment coefficients per fold "
          "(the measured verdict):")
    sign_stable = {}
    for arm in SENT_ARMS:
        block = ARMS[arm][30:]
        print(f"    {arm}:")
        stable_n = 0
        for f in block:
            v = [coefs[(arm, fid)][f] for fid in (1, 2, 3)]
            st = (all(x > 0 for x in v) or all(x < 0 for x in v))
            stable_n += st
            print(f"      {f:<14} f1 {v[0]:+.6f}  f2 {v[1]:+.6f}  "
                  f"f3 {v[2]:+.6f}   sign {'STABLE' if st else 'unstable'}")
        sign_stable[arm] = stable_n

    # ---- summary ----
    print("\n=== STAGE 5: verdict + HARD STOP ===")
    for arm in SENT_ARMS:
        mx = max(abs(coefs[(arm, fid)][f]) for fid in (1, 2, 3)
                 for f in ARMS[arm][30:])
        top_str = max(abs(coefs[(arm, fid)][f]) for fid in (1, 2, 3)
                      for f in ARMS[arm][:30])
        d = (r2[arm]["pool"] - r2["A0_structured"]["pool"]) * 100
        t, p = dm_res[arm]
        print(f"  {arm:<14} dR2 {d:+.4f}pp | DM p {p:.3f} | sign-stable "
              f"coefs {sign_stable[arm]}/5 | max |sent coef| {mx:.6f} vs "
              f"max |struct coef| {top_str:.6f} "
              f"(ratio {mx / top_str:.2f})")
    print(f"  ridge-A0 pooled {r2['A0_structured']['pool'] * 100:+.4f}% vs "
          f"XGBoost-A0 {XGB_A0_MULTI:+.4f}% (multi-seed mean; single-seed "
          f"original {XGB_A0_SINGLE:+.4f}%) - context only, not a "
          f"competition.")
    print("  Nothing here is promoted to a main result (exploratory, "
          "pre-committed reporting).")
    print("  STOP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
