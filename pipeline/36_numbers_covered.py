# %% 36_numbers_covered.py
# Purpose: Four-arm single-lag run on the high-coverage subsample (thresholds 0.50, 0.60, 0.70): Table C.2 covered rows.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet
# Outputs: Data/processed/numbers_check/ tableC2_covered.csv, tableC2_covered_perseed.csv, tableC2_covered_dm060.csv
# Notes:
# - Loads pipeline/27 (single_lag_frame, fit_all, dm_test) via importlib; sets ARMS_DEF.
# - No reproduction gate (different row set); stops if any seed's |A0 R2| exceeds 0.05.
# - DM tests run at the 0.60 threshold only; 0.60 is the primary row.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "Data" / "processed" / "numbers_check"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c27 = _load("cov27", "27_covered_subsample.py")
SEEDS, SENT_ARMS, INPUTS_30 = c27.SEEDS, c27.SENT_ARMS, c27.INPUTS_30
dm_test, ALPHA = c27.dm_test, c27.ALPHA


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== 36: covered subsample 0.50/0.60/0.70 (pipeline/27 reproduction) ===")
    df = c27.single_lag_frame()
    c27.ARMS_DEF = {
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
    te = df[yr.isin({2021, 2022, 2023})]
    share = te.groupby("PERMNO")["had_news"].mean()

    summary, perseed, dmrows = [], [], []
    for thr in (0.50, 0.60, 0.70):
        qual = set(share[share >= thr].index)
        sub = df[df["PERMNO"].isin(qual)].copy()
        res = c27.fit_all(sub, collect_dm=(thr == 0.60), label=f"thr{thr}")
        for arm in SENT_ARMS:
            d = [(res[(s, arm)]["pool"] - res[(s, "A0_structured")]["pool"]) * 100
                 for s in SEEDS]
            a = np.array(d)
            summary.append({"threshold": thr, "arm": arm, "n_firms": len(qual),
                            "n_rows": len(sub), "mean_delta_pp": a.mean(),
                            "sd_delta_pp": a.std(ddof=1), "beats_a0_of_10": int((a > 0).sum())})
            for s, v in zip(SEEDS, d):
                perseed.append({"threshold": thr, "arm": arm, "seed": s, "delta_pp": v})
        # A0 row per threshold
        a0 = np.array([res[(s, "A0_structured")]["pool"] * 100 for s in SEEDS])
        summary.append({"threshold": thr, "arm": "A0_structured", "n_firms": len(qual),
                        "n_rows": len(sub), "mean_delta_pp": np.nan, "sd_delta_pp": np.nan,
                        "beats_a0_of_10": np.nan})
        if thr == 0.60:
            for arm in SENT_ARMS:
                st, pv = [], []
                for s in SEEDS:
                    dd = (res[(s, "A0_structured")]["wk"] - res[(s, arm)]["wk"]).to_numpy()
                    a, b = dm_test(dd)
                    st.append(a)
                    pv.append(b)
                dmrows.append({"arm": arm, "mean_dm": float(np.mean(st)),
                               "sig_signed_of_10": sum(1 for a, b in zip(st, pv)
                                                       if b < ALPHA and a > 0)})
    pd.DataFrame(summary).to_csv(OUT / "tableC2_covered.csv", index=False)
    pd.DataFrame(perseed).to_csv(OUT / "tableC2_covered_perseed.csv", index=False)
    pd.DataFrame(dmrows).to_csv(OUT / "tableC2_covered_dm060.csv", index=False)

    # ---- Verification: consistency and reproduction checks ----
    print("  wrote tableC2_covered.csv (+perseed, +dm060)")
    for r in summary:
        if r["arm"] != "A0_structured":
            print(f"    thr {r['threshold']} {r['arm']:<12} "
                  f"{r['mean_delta_pp']:+.4f} (sd {r['sd_delta_pp']:.4f}) "
                  f"beats {r['beats_a0_of_10']}/10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
