# %% sentiment_aggregation.py
# Purpose: Shared aggregation of per-headline sentiment scores to the weekly
#          moment vector (mean, std, range, kurt) per (PERMNO, target_week);
#          the single implementation used identically by the LM, FinBERT and
#          FinGPT arms.
# Inputs:  none
# Outputs: prints only (unit-test results when run as a script)
# Notes:
# - Imported by pipeline/15, 16c and 17c; loaded via importlib by pipeline/29
#   and 42. Importing has no side effects.
# - Running this file as a script executes its unit tests; the scorer scripts
#   call run_unit_tests() before scoring.
# - Conventions: uniform weights; std is population (ddof=0); kurt is excess
#   kurtosis from the centred two-pass fourth moment, set to 0 when the
#   variance is below VAR_GUARD; n = 1 gives std = range = kurt = 0.
# - Any change here changes every arm's features; the near-identical-pair unit
#   test guards the kurtosis numerics (a raw-moment reconstruction fails it).

import numpy as np
import pandas as pd

KEYS = ["PERMNO", "target_week"]
VAR_GUARD = 1e-12          # kurtosis zero-variance guard


def aggregate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-headline scores (columns PERMNO, target_week, s) to one row
    per (PERMNO, target_week) with mean, std, range, kurt, sorted by the keys.

    Uniform weights; std is population (ddof=0) from M2 - M1^2; range is
    max - min; kurt is excess kurtosis mean((s - M1)^4) / var^2 - 3, set to 0
    when var < VAR_GUARD; n = 1 gives std = range = kurt = 0.
    """
    need = set(KEYS + ["s"])
    if not need <= set(df.columns):
        raise ValueError(f"aggregate_scores needs columns {sorted(need)}, "
                         f"got {sorted(df.columns)}")
    s = df["s"].astype("float64")
    if s.isna().any():
        raise ValueError("aggregate_scores: NaN scores are not allowed")

    t = df[KEYS].copy()
    t["s1"], t["s2"] = s, s ** 2
    gm = t.groupby(KEYS)["s1"].transform("mean")   # group mean, broadcast
    t["d4"] = (s - gm) ** 4                        # centred fourth power
    g = t.groupby(KEYS, sort=True)
    a = g.agg(n=("s1", "size"),
              m1=("s1", "mean"), m2=("s2", "mean"),
              m4c=("d4", "mean"),
              smin=("s1", "min"), smax=("s1", "max"))

    m1 = a["m1"]
    var = (a["m2"] - m1 ** 2).clip(lower=0.0)      # fp guard: sigma^2 >= 0
    std = np.sqrt(var)
    rng = a["smax"] - a["smin"]

    m4c = a["m4c"]
    kurt = pd.Series(0.0, index=a.index)
    ok = var >= VAR_GUARD
    kurt[ok] = m4c[ok] / var[ok] ** 2 - 3.0

    single = a["n"] == 1                            # n=1 edge rule
    std[single] = 0.0
    rng[single] = 0.0
    kurt[single] = 0.0

    return (pd.DataFrame({"mean": m1, "std": std, "range": rng, "kurt": kurt})
            .reset_index())


def run_unit_tests(verbose: bool = True) -> bool:
    """Hand-computed unit tests; returns True on success, raises AssertionError
    with detail on failure."""
    wk = pd.Timestamp("2020-01-10")

    def one(scores, permno=1):
        return pd.DataFrame({"PERMNO": permno, "target_week": wk, "s": scores})

    cases = [
        # (input scores, expected (mean, std, range, kurt), note[, tolerance])
        # [0.5, -0.5]: M1=0; M2=0.25 -> var=0.25 -> std=0.5; range=1.0;
        # M4c=(0.5^4+0.5^4)/2=0.0625; kurt=0.0625/0.25^2-3 = 1-3 = -2
        ([0.5, -0.5], (0.0, 0.5, 1.0, -2.0), "two-point symmetric"),
        # single headline -> n=1 edge rule
        ([0.2], (0.2, 0.0, 0.0, 0.0), "n=1 rule"),
        # four identical -> var=0 -> kurt=0 via the guard
        ([0.3, 0.3, 0.3, 0.3], (0.3, 0.0, 0.0, 0.0), "zero-variance guard"),
        # [1,0,0,0]: M1=0.25; M2=0.25 -> var=0.1875 -> std=sqrt(0.1875);
        # range=1; M4c=((0.75)^4+3*(0.25)^4)/4=0.08203125;
        # kurt=0.08203125/0.1875^2-3 = 7/3-3 = -2/3
        ([1.0, 0.0, 0.0, 0.0],
         (0.25, np.sqrt(0.1875), 1.0, -2.0 / 3.0), "asymmetric 4-point"),
        # n=2 near-identical scores: var = ((a-b)/2)^2 = 7.700625e-8, just
        # above the guard. Any two distinct points have excess kurtosis
        # exactly -2; a raw-moment reconstruction of M4c loses this to
        # cancellation, so the tolerance is 1e-6 on all four values.
        ([0.964732, 0.965287],
         ((0.964732 + 0.965287) / 2, 2.775e-4, 5.55e-4, -2.0),
         "REGRESSION: near-identical pair -> kurt exactly -2", 1e-6),
    ]
    for scores, exp, note, *rest in cases:
        tol = rest[0] if rest else 1e-12
        r = aggregate_scores(one(scores)).iloc[0]
        got = (r["mean"], r["std"], r["range"], r["kurt"])
        for name, g_, e_ in zip(["mean", "std", "range", "kurt"], got, exp):
            assert abs(g_ - e_) < tol, \
                f"UNIT TEST FAIL [{note}] {name}: got {g_!r}, expected {e_!r}"
        if verbose:
            print(f"  [PASS] unit test - {note}: scores {scores} -> "
                  f"mean {got[0]:+.4f}, std {got[1]:.4f}, "
                  f"range {got[2]:.4f}, kurt {got[3]:+.4f}")

    # multi-group independence + output shape/order
    df = pd.concat([one([0.5, -0.5], permno=2), one([0.2], permno=1)])
    r = aggregate_scores(df)
    assert list(r.columns) == KEYS + ["mean", "std", "range", "kurt"]
    assert len(r) == 2 and r["PERMNO"].tolist() == [1, 2]      # sorted by keys
    assert abs(r.loc[r["PERMNO"] == 2, "kurt"].iloc[0] + 2.0) < 1e-12
    assert abs(r.loc[r["PERMNO"] == 1, "mean"].iloc[0] - 0.2) < 1e-12
    if verbose:
        print("  [PASS] unit test - multi-group independence, column order, "
              "key sorting")
    return True


if __name__ == "__main__":
    print("=== sentiment_aggregation.py unit tests ===")
    run_unit_tests(verbose=True)
    print("ALL UNIT TESTS PASSED")
