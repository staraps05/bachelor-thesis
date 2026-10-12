# %% 23_positive_control.py
# Purpose: Positive control for the sentiment merge: test the sentiment features against the contemporaneous week-t return, where a signal must appear if they are attached to the correct firm-week.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: prints only
# Notes:
# - Verification script: checks only; writes nothing to Data/
# - Loads pipeline/3_models/20 via importlib (merge_frame SHA-checks the inputs; dm_test).
# - Prints PASS or FLAG (a FLAG is treated as a bug); pipeline/4_analyses/24 loads its helpers.

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "seed20", REPO_ROOT / "pipeline" / "3_models" / "20_seed_stability.py")
_c20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c20)
merge_frame, dm_test = _c20.merge_frame, _c20.dm_test

TARGET_T = "ret_excess"          # contemporaneous week-t return
SCORERS = ["lm_mean", "finbert_mean", "fingpt_mean"]
T_SIG = 2.0                      # |t| bar for PASS

FLAGS: list[str] = []


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def clustered_ols(y: np.ndarray, x: np.ndarray, weeks: np.ndarray):
    """Univariate OLS with intercept; t on the slope clustered by week."""
    Z = np.column_stack([np.ones(len(x)), x])
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    beta = ZtZ_inv @ (Z.T @ y)
    e = y - Z @ beta
    meat = np.zeros((2, 2))
    for w in pd.unique(weeks):
        idx = weeks == w
        s = Z[idx].T @ e[idx]
        meat += np.outer(s, s)
    V = ZtZ_inv @ meat @ ZtZ_inv
    t = beta[1] / math.sqrt(V[1, 1])
    return beta[1], t


def quintile_spread(frame: pd.DataFrame, score_col: str, ret_col: str):
    """Weekly Q5-Q1 spread of ret_col sorted on score_col (weeks with fewer
    than 20 firms skipped); Newey-West t via dm_test."""
    rows = []
    for _, g in frame.groupby("week_anchor"):
        if len(g) < 20:
            continue
        r = g[score_col].rank(method="first")
        q = pd.qcut(r, 5, labels=False)
        rows.append((g.loc[q == 4, ret_col].mean(),
                     g.loc[q == 0, ret_col].mean()))
    q5 = np.array([a for a, _ in rows])
    q1 = np.array([b for _, b in rows])
    t, _ = dm_test(q5 - q1)
    return q5.mean(), q1.mean(), (q5 - q1).mean(), t, len(rows)


def main() -> int:
    # ---- stage 0: frame + shifted scores ----
    print("=== STAGE 0: rebuild + contemporaneous outcomes ===")
    df = merge_frame()
    if df[TARGET_T].isna().any():
        stop("NaN in week-t ret_excess")
    print(f"  [PASS] contemporaneous outcome = {TARGET_T} (week-t return "
          f"from the frozen panel, still in the frame); ret_excess_fwd NOT "
          f"used - this check is about alignment, not prediction")
    print("  [INFO] NO weekly/contemporaneous volume column exists in the "
          "frozen panel (n_days is a day count; dolvol_126d is a 126-day "
          "average = too smooth per the task). Stated explicitly; the check "
          "runs on return-based outcomes only - no smoothed proxy "
          "substituted.")
    print("  [INFO] DESIGN FACT (frozen S1, neutral statement): the feature "
          "key is 'target_week' with the strict t->t+1 lag - score at key t "
          "= headlines from the Mon-Sun window PRECEDING anchor t; so the "
          "(score_t, ret_t) pairing is prior-window news vs week-t return, "
          "and the same-calendar-window pairing is (score_{t+1}, ret_t). "
          "Stage 2 evaluates the task's pre-registered expectation as "
          "specified.")
    print(f"  full sample used (plumbing check): {len(df):,} rows, "
          f"{df['week_anchor'].nunique()} weeks, 2018-2023")

    # shifted scores via self-merge on the global week index, so spell
    # gaps do not alias neighbouring weeks
    weeks_sorted = sorted(df["week_anchor"].unique())
    w_idx = pd.Series(range(len(weeks_sorted)), index=weeks_sorted)
    df["week_idx"] = df["week_anchor"].map(w_idx)
    base = df[["PERMNO", "week_idx", "had_news"] + SCORERS]
    for s, tag in ((-1, "m1"), (1, "p1")):
        right = base.copy()
        right.columns = ["PERMNO", "week_idx"] + \
            [f"had_news_{tag}"] + [f"{c}_{tag}" for c in SCORERS]
        right["week_idx"] = right["week_idx"] - s
        n0 = len(df)
        df = df.merge(right, on=["PERMNO", "week_idx"], how="left",
                      validate="1:1")
        if len(df) != n0:
            stop(f"shift merge s={s} changed row count")
    print(f"  [PASS] shifted scores attached gap-safely: score(t-1) on "
          f"{int(df['had_news_m1'].notna().sum()):,} rows, score(t+1) on "
          f"{int(df['had_news_p1'].notna().sum()):,} rows (missing = spell "
          f"edges, dropped pairwise)")

    # ---- stage 1: control tests ----
    print("\n=== STAGE 1: control tests (had_news==1 rows unless stated) ===")
    hn = df[df["had_news"] == 1]
    wk_arr = hn["week_anchor"].to_numpy()
    ret = hn[TARGET_T].to_numpy(float)
    print(f"  sample: n = {len(hn):,} had_news rows, "
          f"{len(pd.unique(wk_arr))} weeks")
    print(f"  {'scorer':<13} {'(a) corr':>9} {'t':>7} | {'(b) |corr|':>10} "
          f"{'t':>7} | {'(c) Q5-Q1':>10} {'NW t':>7} {'wks':>4}")
    s1 = {}
    for sc in SCORERS:
        x = hn[sc].to_numpy(float)
        r_a = float(np.corrcoef(x, ret)[0, 1])
        _, t_a = clustered_ols(ret, x, wk_arr)
        r_b = float(np.corrcoef(np.abs(x), np.abs(ret))[0, 1])
        _, t_b = clustered_ols(np.abs(ret), np.abs(x), wk_arr)
        q5, q1, spr, t_c, nw = quintile_spread(hn, sc, TARGET_T)
        s1[sc] = {"r_a": r_a, "t_a": t_a, "r_b": r_b, "t_b": t_b,
                  "spr": spr, "t_c": t_c}
        print(f"  {sc:<13} {r_a:>+9.4f} {t_a:>+7.2f} | {r_b:>+10.4f} "
              f"{t_b:>+7.2f} | {spr * 100:>+9.4f}% {t_c:>+7.2f} {nw:>4}")

    absret = df[TARGET_T].abs().to_numpy(float)
    hn_dummy = df["had_news"].to_numpy(float)
    all_wk = df["week_anchor"].to_numpy()
    m1 = float(absret[hn_dummy == 1].mean())
    m0 = float(absret[hn_dummy == 0].mean())
    _, t_d = clustered_ols(absret, hn_dummy, all_wk)
    print(f"  (d) mean |ret_t|: news weeks {m1 * 100:.4f}% vs no-news "
          f"{m0 * 100:.4f}% | diff {100 * (m1 - m0):+.4f}pp | clustered t = "
          f"{t_d:+.2f} | n = {len(df):,} (all rows)")

    # ---- stage 2: off-by-one probe ----
    print("\n=== STAGE 2: off-by-one probe (score_{t-1} / score_t / "
          "score_{t+1} vs ret_t) ===")
    s2 = {}
    print(f"  {'scorer':<13} {'corr(t-1)':>10} {'corr(t)':>10} "
          f"{'corr(t+1)':>10} | {'sprd(t-1)':>10} {'sprd(t)':>10} "
          f"{'sprd(t+1)':>10}")
    for sc in SCORERS:
        vals, sprs = {}, {}
        for s, tag in ((-1, "m1"), (0, ""), (1, "p1")):
            col = sc if s == 0 else f"{sc}_{tag}"
            hcol = "had_news" if s == 0 else f"had_news_{tag}"
            m = df[hcol] == 1
            g = df[m]
            vals[s] = float(np.corrcoef(g[col].to_numpy(float),
                                        g[TARGET_T].to_numpy(float))[0, 1])
            _, _, spr, t_s, _ = quintile_spread(g, col, TARGET_T)
            sprs[s] = (spr, t_s)
        s2[sc] = (vals, sprs)
        print(f"  {sc:<13} {vals[-1]:>+10.4f} {vals[0]:>+10.4f} "
              f"{vals[1]:>+10.4f} | {sprs[-1][0] * 100:>+9.4f}% "
              f"{sprs[0][0] * 100:>+9.4f}% {sprs[1][0] * 100:>+9.4f}%")
        if vals[1] > vals[0] or vals[-1] > vals[0]:
            FLAGS.append(f"{sc}: score_t is NOT the strongest pairing "
                         f"(corr t-1/t/t+1 = {vals[-1]:+.4f}/{vals[0]:+.4f}/"
                         f"{vals[1]:+.4f}) - per the pre-registered rule "
                         f"this flags a timing misalignment")
    print("  quintile-spread NW t-stats: " + "; ".join(
        f"{sc} t-1/t/t+1 = {s2[sc][1][-1][1]:+.2f}/{s2[sc][1][0][1]:+.2f}/"
        f"{s2[sc][1][1][1]:+.2f}" for sc in SCORERS))

    # ---- stage 3: verdict ----
    print("\n=== STAGE 3: verdict + HARD STOP ===")
    for sc in SCORERS:
        r = s1[sc]
        if not (r["r_a"] > 0 and abs(r["t_a"]) >= T_SIG):
            FLAGS.append(f"{sc}: (a) contemporaneous corr not clearly "
                         f"positive (r {r['r_a']:+.4f}, t {r['t_a']:+.2f})")
        if not (r["spr"] > 0 and abs(r["t_c"]) >= T_SIG):
            FLAGS.append(f"{sc}: (c) quintile spread not clearly positive "
                         f"({r['spr'] * 100:+.4f}%, NW t {r['t_c']:+.2f})")
    if not (m1 > m0 and abs(t_d) >= T_SIG):
        FLAGS.append(f"(d) news weeks not clearly more volatile "
                     f"(diff {100 * (m1 - m0):+.4f}pp, t {t_d:+.2f})")
    if FLAGS:
        print(f"  Verdict: FLAG ({len(FLAGS)}) - per pre-commit these are "
              f"treated as a BUG to hunt, not a finding:")
        for f in FLAGS:
            print(f"    - {f}")
    else:
        print("  Verdict: PASS - sentiment attaches to the correct firm and "
              "week; plumbing confirmed.")
    print("  Nothing promoted (exploratory, read-only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
