# %% 31_importance_and_triangle.py
# Purpose: Refit the primary single-lag XGBoost spec (4 arms x 3 folds x 10 seeds); print the A0 importance table, sentiment-block shares per arm and the cross-scorer correlation triangle.
# Inputs:  Data/processed/model_frame_structured.parquet, Data/processed/features_lm.parquet, Data/processed/features_finbert.parquet, Data/processed/features_fingpt.parquet, Data/processed/scoring_input.parquet, Data/Loughran-McDonald_MasterDictionary_*.csv, Data/colab/finbert_scores.csv, Data/colab/fingpt_scores.csv
# Outputs: prints only
# Notes:
# - Loads pipeline/30 and pipeline/17 via importlib (fit config, reference R2, blocks).
# - Gate: every seed x arm pooled R2 must match the reference within GATE_TOL, else stop.
# - Triangle values are compared with the recorded ones (Pearson/Spearman, row-set labels)

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "pipeline" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# pipeline/30 re-exports the fit configuration, the 40-cell per-seed
# reference R2 table and the gate tolerance.
_c30 = _load("c30", "30_per_year_primary.py")
merge_frame = _c30.merge_frame
FIXED_PARAMS, AMENDMENT = _c30.FIXED_PARAMS, _c30.AMENDMENT
DEPTH_GRID, NUM_ROUND_CAP = _c30.DEPTH_GRID, _c30.NUM_ROUND_CAP
EARLY_STOP, FOLDS, SEEDS = _c30.EARLY_STOP, _c30.FOLDS, _c30.SEEDS
TARGET_FWD, INPUTS_30 = _c30.TARGET_FWD, _c30.INPUTS_30
SENT_FILES, YEARS, SENT_ARMS = _c30.SENT_FILES, _c30.YEARS, _c30.SENT_ARMS
C26_SEED_R2, C26_POOL_AGG = _c30.C26_SEED_R2, _c30.C26_POOL_AGG
GATE_TOL, ARM_ORDER = _c30.GATE_TOL, _c30.ARM_ORDER

# characteristic/macro block lists from pipeline/17
_c17 = _load("pre17", "17_pre_model_integrity.py")
CHARS_22, MACRO_8 = _c17.CHARS_22, _c17.MACRO_8

# ---- triangle inputs; SHAs from the pipeline/15 header and colab_scores_manifest.md ----
SCORING_FILE = REPO_ROOT / "Data" / "processed" / "scoring_input.parquet"
SCORING_SHA = "eb42e0e3aa05c7c973e407c722125468bb506f1cc51e629042ab4d0a027ccc39"
SCORING_ROWS = 796_038
FB_FILE = REPO_ROOT / "Data" / "colab" / "finbert_scores.csv"
FB_SHA = "e39307f52d262426e4579b315d8c4cf9fee03cd6888d3bdb87e48844987f8731"
FG_FILE = REPO_ROOT / "Data" / "colab" / "fingpt_scores.csv"
FG_SHA = "0daee9398a6714229a83ee53f12ac157469de572725a5b0abaa6ebc632883207"
DICT_GLOB = "Loughran-McDonald_MasterDictionary_*.csv"
LM_ZERO_SHARE = 0.7731          # anchor from the pipeline/15 run log
TOKEN_RE = re.compile(r"[A-Z]+")

# Recorded triangle values, 3 dp, as (ALL, LM-nonzero); Pearson from the
# master H.8 table, Spearman from methodologyCheck.md 6.4; None = not recorded.
PAIRS = (("lm", "finbert"), ("lm", "fingpt"), ("finbert", "fingpt"))
REC_PEARSON = {("lm", "finbert"): (0.393, 0.614),
               ("lm", "fingpt"): (0.339, 0.596),
               ("finbert", "fingpt"): (0.679, 0.757)}
REC_SPEARMAN = {("lm", "finbert"): (None, 0.617),
                ("lm", "fingpt"): (None, 0.607),
                ("finbert", "fingpt"): (0.680, 0.782)}


def stop(msg: str) -> None:
    sys.exit(f"\nSTOP: {msg}")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def share_vec(bst: xgb.Booster, inputs: list) -> np.ndarray:
    """Total-gain share per input feature from a booster (0 if never split)."""
    sc = bst.get_score(importance_type="total_gain")
    tot = sum(sc.values())
    # Verification (in place): non-zero total gain (inside share_vec)
    if tot <= 0:
        stop("booster has zero total gain - cannot form shares")
    return np.array([sc.get(f, 0.0) / tot for f in inputs])


def seed_stats(cells: dict, reduce_folds) -> tuple:
    """cells: (seed, fid) -> value/vector. Returns (mean over all fold x seed
    cells, sd across the 10 per-seed fold-means, ddof=1)."""
    per_seed = np.array([reduce_folds([cells[(sd, fid)]
                                       for fid, _, _, _ in FOLDS])
                         for sd in SEEDS])
    return per_seed.mean(axis=0), per_seed.std(axis=0, ddof=1)


def main() -> int:
    # ---- config consistency checks ----
    print("=== STAGE 0: config + reference-table transcription guard ===")
    # Verification (in place): INPUTS_30 single source
    if INPUTS_30 != CHARS_22 + MACRO_8 or len(MACRO_8) != 8 \
            or len(CHARS_22) != 22:
        stop("INPUTS_30 != CHARS_22 + MACRO_8 - single-source violated")
    # Verification (in place): transcribed 40-cell table consistent with the recorded aggregates
    for arm in ARM_ORDER:
        v = np.array([C26_SEED_R2[sd][arm] for sd in SEEDS])
        got = (v.mean(), v.std(ddof=1), v.min(), v.max())
        if max(abs(g - r) for g, r in zip(got, C26_POOL_AGG[arm])) > 1e-3:
            stop(f"{arm}: 40-cell table inconsistent with the recorded "
                 f"aggregates")
    print("  [PASS] 40-cell pipeline/26 reference table consistent with the "
          "recorded aggregates (tol 1e-3); gate tolerance "
          f"{GATE_TOL:.0e} pp")

    # ---- rebuild the single-lag frame: sentiment re-keyed to week t+1 ----
    print("\n=== STAGE 1: rebuild single-lag frame (verbatim pipeline/26) ===")
    df = merge_frame()
    sent_cols = [f"{a}_{s}" for a in SENT_FILES
                 for s in ("mean", "std", "range", "kurt")]
    df = df.drop(columns=sent_cols + ["had_news"])

    lm = pd.read_parquet(SENT_FILES["lm"][0],
                         columns=["PERMNO", "target_week"])
    grid_anchors = sorted(lm["target_week"].unique())          # 313 anchors
    frame_anchors = sorted(df["week_anchor"].unique())         # 312 anchors
    # Verification (in place): anchor grids 313/312
    if len(grid_anchors) != 313 or frame_anchors != grid_anchors[:312]:
        stop("anchor grids unexpected (frame must be grid minus final week)")
    g_idx = {a: i for i, a in enumerate(grid_anchors)}
    df["week_next"] = df["week_anchor"].map(
        lambda a: grid_anchors[g_idx[a] + 1])
    # Verification (in place): week_next mapping
    if df["week_next"].isna().any():
        stop("week_next mapping produced NaN")

    hn_cols = []
    for arm, (path, _) in SENT_FILES.items():   # SHAs verified in merge_frame
        f = pd.read_parquet(path)
        f = f.rename(columns={"target_week": "week_next",
                              "had_news": f"had_news_{arm}"})
        n0 = len(df)
        df = df.merge(f, on=["PERMNO", "week_next"], how="left",
                      validate="m:1")
    # Verification (in place): re-merge preserves row count (inside loop)
        if len(df) != n0:
            stop(f"re-merge of {arm} changed row count")
        hn_cols.append(f"had_news_{arm}")
    n_nan = int(df[sent_cols + hn_cols].isna().sum().sum())
    n_drop = int(df[hn_cols[0]].isna().sum())
    # Verification (in place): no rows dropped, no NaN after re-merge
    if n_drop or n_nan:
        stop(f"{n_drop} rows without week-(t+1) key / {n_nan} NaN after "
             f"re-merge - pipeline/26 had 0/0")
    # Verification (in place): had_news identical across arms
    if not (df[hn_cols[0]].equals(df[hn_cols[1]])
            and df[hn_cols[0]].equals(df[hn_cols[2]])):
        stop("had_news differs between arms after re-alignment")
    df["had_news"] = df[hn_cols[0]].astype(int)
    df = df.drop(columns=hn_cols + ["week_next"])
    hn_share = df["had_news"].mean()
    # Verification (in place): had_news share
    if abs(hn_share - 0.6620) > 5e-4:
        stop(f"had_news share {hn_share:.4f} != pipeline/26's 0.6620")

    # ---- refit, reproduction gate, collect truncated gain shares ----
    print("\n=== STAGE 2: refit 4 arms x 3 folds x 10 seeds + HARD gate; "
          "collect truncated importances ===")
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
    for arm, cols in ARMS.items():
        want = 30 if arm == "A0_structured" else 35
    # Verification (in place): arm input counts
        if len(cols) != want or len(set(cols)) != want:
            stop(f"{arm}: {len(cols)} inputs, expected {want}")
    yr = pd.to_datetime(df["week_anchor"]).dt.year
    split_masks = {}
    for fid, tr_y, va_y, te_y in FOLDS:
        m_tr, m_va, m_te = yr.isin(tr_y), yr.isin(va_y), yr.isin(te_y)
        w = [set(df.loc[m, "week_anchor"]) for m in (m_tr, m_va, m_te)]
    # Verification (in place): fold feature weeks disjoint (inside loop)
        if w[0] & w[1] or w[0] & w[2] or w[1] & w[2]:
            stop(f"fold {fid}: feature-week sets not disjoint")
        split_masks[fid] = (m_tr, m_va, m_te)
    dmats = {}
    for arm, inputs in ARMS.items():
        dmats[arm] = {}
        for fid, _, _, _ in FOLDS:
            m_tr, m_va, m_te = split_masks[fid]
            dmats[arm][fid] = (
                xgb.DMatrix(df.loc[m_tr, inputs],
                            label=df.loc[m_tr, TARGET_FWD]),
                xgb.DMatrix(df.loc[m_va, inputs],
                            label=df.loc[m_va, TARGET_FWD]),
                xgb.DMatrix(df.loc[m_te, inputs]),
            )
    te_meta = {fid: df.loc[split_masks[fid][2],
                           ["week_anchor", TARGET_FWD]].reset_index(drop=True)
               for fid, _, _, _ in FOLDS}

    a0_trunc, a0_full = {}, {}      # (seed, fid) -> share vector (30,)
    blk_trunc = {arm: {} for arm in SENT_ARMS}   # (seed, fid) -> block share
    overshoot = []                  # trained-past-best rounds, all 120 cells
    max_dev = 0.0
    for seed in SEEDS:
        for arm, inputs in ARMS.items():
            sse = ssy = 0.0
            for fid, _, _, _ in FOLDS:
                d_tr, d_va, d_te = dmats[arm][fid]
                best = None
                for depth in DEPTH_GRID:   # ascending -> ties to lower depth
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
                bst, n_est = best["bst"], best["n_est"]
                yhat = bst.predict(d_te, iteration_range=(0, n_est))
    # Verification (in place): finite predictions (inside loop)
                if not np.isfinite(yhat).all():
                    stop(f"seed {seed} {arm} fold {fid}: NaN/inf predictions")
                y = te_meta[fid][TARGET_FWD].to_numpy()
                sse += float(((y - yhat) ** 2).sum())
                ssy += float((y ** 2).sum())
                overshoot.append(bst.num_boosted_rounds() - n_est)
                v_t = share_vec(bst[0:n_est], inputs)     # truncated booster
                if arm == "A0_structured":
                    a0_trunc[(seed, fid)] = v_t
                    a0_full[(seed, fid)] = share_vec(bst, inputs)
                else:
                    blk_trunc[arm][(seed, fid)] = float(v_t[30:].sum())
    # Verification (in place): per-(seed, arm) reproduction gate (inside loop)
            pool = (1.0 - sse / ssy) * 100
            ref = C26_SEED_R2[seed][arm]
            dev = abs(pool - ref)
            max_dev = max(max_dev, dev)
            if dev > GATE_TOL:
                stop(f"REPRODUCTION GATE FAILED: seed {seed} {arm} pooled "
                     f"{pool:+.4f}% vs pipeline/26 log {ref:+.4f}% "
                     f"(|dev| {dev:.2e} > {GATE_TOL:.0e}) - importances NOT "
                     f"reported")
        print(f"  seed {seed}: all 4 pooled cells match the pipeline/26 log "
              f"(running max |dev| {max_dev:.2e} pp)")
    print(f"  [INFO] overshoot rounds trained past best_iteration "
          f"(excluded by truncation): min {min(overshoot)}, "
          f"median {int(np.median(overshoot))}, max {max(overshoot)} "
          f"(cap {EARLY_STOP}) across all 120 selected boosters")

    # ---- A0 per-feature gain shares ----
    print("\n=== STAGE 3 (a): A0 importance table - total-gain shares, "
          "TRUNCATED boosters, mean over 3 folds x 10 seeds ===")
    mean_t, sd_t = seed_stats(a0_trunc, lambda v: np.mean(v, axis=0))
    mean_f, _ = seed_stats(a0_full, lambda v: np.mean(v, axis=0))
    blk = np.array(["char" if f in CHARS_22 else "macro" for f in INPUTS_30])

    order = np.argsort(mean_t)[::-1]
    print(f"  {'rank':<5}{'feature':<15}{'block':<7}{'mean share':>11}"
          f"{'sd(seeds)':>11}")
    for r, i in enumerate(order[:10], 1):
        print(f"  {r:<5}{INPUTS_30[i]:<15}{blk[i]:<7}{mean_t[i]:>11.4f}"
              f"{sd_t[i]:>11.4f}")
    mac_t = float(mean_t[blk == "macro"].sum())
    ch_t = float(mean_t[blk == "char"].sum())
    mac_sd = float(np.std([np.mean([a0_trunc[(sd_, fid)][blk == "macro"].sum()
                                    for fid, _, _, _ in FOLDS])
                           for sd_ in SEEDS], ddof=1))
    print(f"  block sums (truncated): macro (8 cols) {mac_t:.4f} "
          f"(sd {mac_sd:.4f}) | characteristics (22 cols) {ch_t:.4f}")
    for fid, _, _, _ in FOLDS:
        fm = np.mean([a0_trunc[(sd_, fid)] for sd_ in SEEDS], axis=0)
        top = np.argsort(fm)[::-1][:3]
        print(f"    fold {fid}: macro {fm[blk == 'macro'].sum():.4f} | top-3 "
              + ", ".join(f"{INPUTS_30[i]} {fm[i]:.3f}" for i in top))


    # ---- sentiment-block gain shares ----
    print("\n=== STAGE 4 (b): sentiment-block (5 cols) total-gain shares, "
          "TRUNCATED boosters, primary single-lag spec ===")
    print(f"  {'arm':<14}{'mean':>9}{'sd(seeds)':>11}{'fold1':>9}"
          f"{'fold2':>9}{'fold3':>9}")
    blk_out = {}
    for arm in SENT_ARMS:
        m, s = seed_stats(blk_trunc[arm], np.mean)
        by_fold = {fid: float(np.mean([blk_trunc[arm][(sd_, fid)]
                                       for sd_ in SEEDS]))
                   for fid, _, _, _ in FOLDS}
        blk_out[arm] = (float(m), float(s), by_fold)
        print(f"  {arm:<14}{m:>9.4f}{s:>11.4f}"
              + "".join(f"{by_fold[fid]:>9.4f}" for fid, _, _, _ in FOLDS))

    # ---- scorer-correlation triangle ----
    print("\n=== STAGE 5 (c): cross-scorer triangle - headline-level s, "
          "explicit pair labels and row sets ===")
    # Verification (in place): input file SHA256 (inside loop)
    for path, want in ((SCORING_FILE, SCORING_SHA), (FB_FILE, FB_SHA),
                       (FG_FILE, FG_SHA)):
        got = sha256_of(path)
        if got != want:
            stop(f"{path.name} SHA mismatch: {got}")
        print(f"  [PASS] {path.name}: SHA == frozen ({want[:8]}...)")
    hl = pd.read_parquet(SCORING_FILE)
    # Verification (in place): scoring-input row count
    if len(hl) != SCORING_ROWS:
        stop(f"scoring input has {len(hl):,} rows")

    # LM per-headline s regenerated with the pipeline/15 convention (there is
    # no per-headline LM score file)
    cands = sorted((REPO_ROOT / "Data").glob(DICT_GLOB))
    # Verification (in place): LM dictionary present
    if not cands:
        stop("LM Master Dictionary not found")
    lm_dict = pd.read_csv(cands[-1])
    lm_dict["Word"] = lm_dict["Word"].astype(str).str.upper()
    NEG = set(lm_dict.loc[lm_dict["Negative"] > 0, "Word"])
    POS = set(lm_dict.loc[lm_dict["Positive"] > 0, "Word"])
    # Verification (in place): LM word-list sizes
    if not (300 <= len(POS) <= 400 and 2200 <= len(NEG) <= 2500):
        stop(f"LM word lists unexpected: {len(POS)} pos / {len(NEG)} neg")
    toks = hl["headline"].str.upper().str.findall(TOKEN_RE)
    ex = toks.explode()
    ex = ex[ex.notna()]
    n_pos = (ex.isin(POS).groupby(level=0).sum()
             .reindex(hl.index, fill_value=0).astype("int64"))
    n_neg = (ex.isin(NEG).groupby(level=0).sum()
             .reindex(hl.index, fill_value=0).astype("int64"))
    tot = n_pos + n_neg
    s_lm = np.where(tot > 0, (n_pos - n_neg) / tot.where(tot > 0, 1), 0.0)
    zero_share = float((s_lm == 0).mean())
    # Verification (in place): LM zero share matches pipeline/15
    if abs(zero_share - LM_ZERO_SHARE) > 5e-4:
        stop(f"LM zero share {zero_share:.4f} != pipeline/15's {LM_ZERO_SHARE}")
    print(f"  [PASS] LM headline s regenerated (pipeline/15 convention): zero "
          f"share {zero_share:.4f} == published {LM_ZERO_SHARE}")

    s_neu = {}
    for name, path in (("finbert", FB_FILE), ("fingpt", FG_FILE)):
        sc = pd.read_csv(path).sort_values("hid").reset_index(drop=True)
    # Verification (in place): hid unique and complete (inside loop)
        if not (sc["hid"].is_unique and sc["hid"].min() == 0
                and sc["hid"].max() == SCORING_ROWS - 1
                and len(sc) == SCORING_ROWS):
            stop(f"{path.name}: hid not unique/complete")
        s_neu[name] = (sc["p_pos"] - sc["p_neg"]).to_numpy()

    tri = pd.DataFrame({"lm": s_lm, "finbert": s_neu["finbert"],
                        "fingpt": s_neu["fingpt"]})
    m_nz = tri["lm"] != 0
    print(f"  row sets: ALL = {len(tri):,} scored headlines | LM-nonzero = "
          f"{int(m_nz.sum()):,} ({m_nz.mean():.1%})")

    # Verification (in place): CONFIRMED/CORRECTED against recorded triangle values (inside loop)
    def verdict(x, rec):
        if rec is None:
            return "no recorded value (new)"
        return ("CONFIRMED" if round(x, 3) == rec
                else f"CORRECTED (record {rec:.3f})")

    triangle = {}
    for method, rec_table in (("pearson", REC_PEARSON),
                              ("spearman", REC_SPEARMAN)):
        print(f"\n  {method.upper()} (headline-level s):")
        c_all = tri.corr(method=method)
        c_nz = tri[m_nz].corr(method=method)
        for a, b in PAIRS:
            va, vn = float(c_all.loc[a, b]), float(c_nz.loc[a, b])
            ra, rn = rec_table[(a, b)]
            triangle[(method, a, b)] = (va, vn)
            print(f"    {a:<8}<-> {b:<8} ALL {va:.4f} "
                  f"[{verdict(va, ra)}] | LM-nonzero {vn:.4f} "
                  f"[{verdict(vn, rn)}]")

    # ---- Verification: consistency and reproduction checks ----
    print(f"  [PASS] single-lag frame rebuilt: {len(df):,} rows, 0 dropped, "
          f"had_news share {hn_share:.4f} == pipeline/26")
    print(f"  [PASS] reproduction gate: all 40 seed x arm pooled cells match "
          f"the pipeline/26 log within {GATE_TOL:.0e} pp (max |dev| "
          f"{max_dev:.2e} pp)")
    print("\n  A0 WITHOUT truncation (full early-stopped booster incl. "
          "overshoot rounds - the audited Block 9+10 C11 object, printed "
          "once for the record):")
    order_f = np.argsort(mean_f)[::-1]
    for r, i in enumerate(order_f[:10], 1):
        print(f"  {r:<5}{INPUTS_30[i]:<15}{blk[i]:<7}{mean_f[i]:>11.4f}")
    mac_f, ch_f = (float(mean_f[blk == "macro"].sum()),
                   float(mean_f[blk == "char"].sum()))
    print(f"  block sums (untruncated): macro {mac_f:.4f} | chars {ch_f:.4f}")
    i_dp = INPUTS_30.index("dp")
    print(f"  [INFO] dp share: untruncated {mean_f[i_dp]:.4f} (recorded C11 "
          f"value 0.347) vs truncated {mean_t[i_dp]:.4f} -> delta "
          f"{mean_t[i_dp] - mean_f[i_dp]:+.4f} attributable to the "
          f"overshoot rounds")
    print("  [INFO] folds 2-3 expected ~0: the pipeline/30a census found 56/60 "
          "seed x arm cells in 2022/23 with delta vs A0 exactly 0.0 "
          "(sentiment functionally unused there); fold-1 carries the block.")
    print("  [PASS] neural s = p_pos - p_neg joined on hid (positional index "
          "0..796,037 into the frozen scoring input) for both arms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
