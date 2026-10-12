# 07_verify_characteristic_selection.py
# Purpose: Verify the selected JKP characteristic codes (13 Block-B codes plus the
#          conditional beta_60m) against Data/Fundamentals.csv: each code must exist
#          verbatim as a column, and its NaN share overall, per calendar year and its
#          firm coverage are reported. Does not clean, impute or substitute; writes
#          only the summary CSV.
# Inputs:  Data/Fundamentals.csv
# Outputs: Data/processed/characteristic_selection_verification.csv
# Notes:
# - Code list, thresholds and the expected firm count (681) follow docs/characteristic_selection.md.
# - A code is marked REVIEW when overall NaN > 15% or any single year > 30%; this is a
#   review trigger, not an exclusion.
# - beta_60m is taken pre-computed from JKP because the CRSP daily sample starts in 2015,
#   too short for a 60-month lookback.
# - Missing codes get close-match candidates printed for reference only.
# - Only the ID columns and the present codes are loaded. Deterministic; pandas + stdlib only.
# Verification script: checks only; writes only Data/processed/characteristic_selection_verification.csv

import difflib
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_FILE = BASE / "Data" / "Fundamentals.csv"
OUT_FILE = BASE / "Data" / "processed" / "characteristic_selection_verification.csv"

BLOCK_B = [
    "be_me", "ni_me", "sale_me", "chcsho_12m",   # value
    "ope_be", "niq_be",                            # profitability
    "gp_at",                                       # quality
    "at_gr1", "sale_gr1",                          # investment
    "oaccruals_at",                                # accruals
    "noa_at",                                      # debt issuance
    "at_be",                                       # low leverage
    "niq_su",                                      # profit growth
]
CONDITIONAL = ["beta_60m"]
ALL_CODES = BLOCK_B + CONDITIONAL

ID_COLS = ["permno", "eom"]
PLAUSIBILITY_CODES = ["be_me", "ni_me", "at_be"]

OVERALL_NAN_THRESHOLD = 15.0   # %
YEARLY_NAN_THRESHOLD = 30.0    # %
EXPECTED_FIRMS = 681           # JKP-covered firms in the panel

# ---- check: existence (header only) ----
print(f"Loading column names from {DATA_FILE} ...")
all_columns = list(pd.read_csv(DATA_FILE, nrows=0).columns)
print(f"  {len(all_columns)} columns found.\n")

missing = sorted(set(ALL_CODES) - set(all_columns))
present = [c for c in ALL_CODES if c in all_columns]

print("=" * 70)
print("F3 — EXISTENCE CHECK")
print("=" * 70)
if not missing:
    print(f"All {len(ALL_CODES)} codes exist verbatim as columns. Missing set: {set()}")
else:
    print(f"MISSING CODES: {missing}")
    print("Close-match candidates (report only — NO substitution):")
    for code in missing:
        substr = [c for c in all_columns if code in c or c in code]
        fuzzy = difflib.get_close_matches(code, all_columns, n=5, cutoff=0.6)
        candidates = sorted(set(substr) | set(fuzzy))
        print(f"  {code}: {candidates if candidates else 'no close matches found'}")
print()

# ---- load ID columns and present codes only ----
missing_ids = [c for c in ID_COLS if c not in all_columns]
if missing_ids:
    sys.exit(f"FATAL: ID column(s) {missing_ids} not in file — cannot proceed.")

usecols = ID_COLS + present
print(f"Loading {len(usecols)} columns ({len(present)} codes + {ID_COLS}) ...")
df = pd.read_csv(DATA_FILE, usecols=usecols)
df["eom"] = pd.to_datetime(df["eom"])
df["year"] = df["eom"].dt.year
n_rows = len(df)
n_firms_total = df["permno"].nunique()
years = sorted(df["year"].unique())
print(f"  {n_rows:,} rows, {n_firms_total} distinct PERMNOs, "
      f"{df['eom'].min().date()} → {df['eom'].max().date()}\n")

# ---- check: coverage (NaN share overall and per year) ----
print("=" * 70)
print("F2 — COVERAGE (NaN share overall and per year)")
print("=" * 70)

rows = []
yearly_nan = {}  # code -> Series of NaN % by year

for code in ALL_CODES:
    if code not in present:
        rows.append({
            "code": code, "exists": False, "nan_pct_overall": None,
            "worst_year": None, "worst_year_pct": None,
            "firms_covered": None, "verdict": "MISSING",
        })
        continue

    overall_pct = df[code].isna().mean() * 100
    per_year = df.groupby("year")[code].apply(lambda s: s.isna().mean() * 100)
    yearly_nan[code] = per_year
    worst_year = per_year.idxmax()
    worst_pct = per_year.max()
    firms_covered = df.loc[df[code].notna(), "permno"].nunique()

    flagged = overall_pct > OVERALL_NAN_THRESHOLD or worst_pct > YEARLY_NAN_THRESHOLD
    rows.append({
        "code": code, "exists": True,
        "nan_pct_overall": round(overall_pct, 2),
        "worst_year": int(worst_year), "worst_year_pct": round(worst_pct, 2),
        "firms_covered": firms_covered,
        "verdict": "⚠ REVIEW" if flagged else "OK",
    })

year_table = pd.DataFrame(yearly_nan).T
year_table.columns = [str(y) for y in year_table.columns]
print("\nNaN share per year (%):")
print(year_table.round(1).to_string())
print()

# ---- plausibility peek: raw min / median / max of three ratios ----
print("=" * 70)
print("PLAUSIBILITY PEEK (min / median / max — no cleaning)")
print("=" * 70)
for code in PLAUSIBILITY_CODES:
    if code in present:
        s = df[code].dropna()
        print(f"  {code:8s}: min={s.min():.4f}  median={s.median():.4f}  max={s.max():.4f}")
    else:
        print(f"  {code:8s}: MISSING — no stats")
print()

# ---- summary table (also written to OUT_FILE) ----
summary = pd.DataFrame(rows)
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT_FILE, index=False)
print(f"\nSaved summary to {OUT_FILE}")

# ---- verdict ----
ok_codes = [r["code"] for r in rows if r["verdict"] == "OK"]
review_codes = [r["code"] for r in rows if r["verdict"] == "⚠ REVIEW"]
beta_row = next(r for r in rows if r["code"] == "beta_60m")

print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
print(f"F3: {'PASS' if not missing else 'FAIL'} "
      f"({'all codes exist verbatim' if not missing else 'missing: ' + ', '.join(missing)})")
print(f"F2: {len(ok_codes)} codes OK, {len(review_codes)} flagged REVIEW "
      f"({', '.join(review_codes) if review_codes else 'none'})")
print(f"F1 beta_60m: exists {'yes' if beta_row['exists'] else 'no'}, "
      f"coverage {100 - beta_row['nan_pct_overall']:.2f}% "
      f"(NaN {beta_row['nan_pct_overall']:.2f}%), "
      f"{beta_row['firms_covered']}/{n_firms_total} firms "
      f"(expected panel: {EXPECTED_FIRMS})"
      if beta_row["exists"] else "F1 beta_60m: exists no — see F3 candidates above")
