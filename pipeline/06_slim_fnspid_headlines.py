#!/usr/bin/env python3
# 06_slim_fnspid_headlines.py
# Purpose: Slim the ~23 GB FNSPID CSV to Date, Stock_symbol, Article_title, 2015-2023.
# Inputs:  Data/fnspid_nasdaq_news_raw.csv
# Outputs: Data/processed/fnspid_headlines_2015_2023.csv, Data/processed/fnspid_headlines_2015_2023_TEST.csv (TEST_MODE only)
# Notes:
# - Existing output is deleted first; malformed rows are skipped with a warning.
# - TEST_MODE = True runs TEST_CHUNKS chunks into the *_TEST file; real output untouched.
from pathlib import Path
from collections import Counter
import sys
import pandas as pd

# configuration
TEST_MODE = False         # True -> only first TEST_CHUNKS chunks, *_TEST output
TEST_CHUNKS = 5           # chunks to process when TEST_MODE is True
CHUNKSIZE = 500_000       # rows read per chunk

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "Data" / "fnspid_nasdaq_news_raw.csv"
OUT_REAL = BASE / "Data" / "processed" / "fnspid_headlines_2015_2023.csv"
OUT_TEST = BASE / "Data" / "processed" / "fnspid_headlines_2015_2023_TEST.csv"

# Half-open [START, END): the Date column carries an intraday UTC time, so the
# upper bound is the first instant of 2024 to keep all of 2023-12-31.
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-01-01", tz="UTC")


# helpers
def human_size(num_bytes: int) -> str:
    """Human-readable byte size."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def map_columns(header_cols):
    """Map the (date, ticker, headline) roles to header columns, case-insensitively:
    exact match to the FNSPID name first, then substring fallback; unmapped -> None."""
    lower = {c.lower(): c for c in header_cols}

    def find(primary, fallbacks):
        if primary in lower:
            return lower[primary]
        for token in fallbacks:
            for c in header_cols:
                if token in c.lower():
                    return c
        return None

    date_col = find("date", ["date", "time"])
    ticker_col = find("stock_symbol", ["stock_symbol", "ticker", "symbol"])
    headline_col = find("article_title", ["article_title", "headline", "title"])
    return date_col, ticker_col, headline_col


def main():
    print("=" * 78)
    print("FNSPID headline slimmer")
    print(f"TEST_MODE = {TEST_MODE}  (chunksize = {CHUNKSIZE:,})")
    print("=" * 78)

    # Verification (in place): raw file exists before the header read
    if not RAW.exists():
        sys.exit(f"ERROR: raw file not found: {RAW}")

    # read only the header and map the three target columns
    header_df = pd.read_csv(RAW, nrows=0)
    header_cols = list(header_df.columns)
    # Verification (in place): header column listing, referred to by the STOP below
    print("\nAll column names in the raw file:")
    for i, c in enumerate(header_cols):
        print(f"  [{i}] {c!r}")

    date_col, ticker_col, headline_col = map_columns(header_cols)
    # Verification (in place): column-role mapping, context for the gate below
    print("\nColumn role mapping:")
    print(f"  date     -> {date_col!r}")
    print(f"  ticker   -> {ticker_col!r}")
    print(f"  headline -> {headline_col!r}")

    # Verification (in place): all three target columns mapped before the streaming write
    if None in (date_col, ticker_col, headline_col):
        print("\nSTOP: could not confidently map all three target columns.")
        print("Header shown above — please tell me the correct names.")
        sys.exit(1)

    usecols = [date_col, ticker_col, headline_col]

    # the real file is never touched in TEST_MODE
    out_path = OUT_TEST if TEST_MODE else OUT_REAL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # start clean so appended chunks can't duplicate a previous run's rows
    if out_path.exists():
        out_path.unlink()
    print(f"\nOutput -> {out_path}")

    # stream in chunks, filter by date, append the three columns
    rows_scanned = 0
    rows_kept = 0
    first_write = True

    # running aggregates for the summary
    year_counts = Counter()          # kept rows per calendar year
    unique_tickers = set()           # distinct tickers in the output
    null_headline = 0                # kept rows with empty/null headline
    nonmidnight = 0                  # kept rows whose time-of-day != 00:00:00
    min_date = None                  # earliest kept timestamp (tz-aware)
    max_date = None                  # latest kept timestamp (tz-aware)
    sample_raw_dates = []            # first 5 raw date strings (verbatim)

    reader = pd.read_csv(
        RAW,
        usecols=usecols,
        dtype=str,                   # everything as string -> nothing mis-cast
        chunksize=CHUNKSIZE,
        on_bad_lines="warn",        # FNSPID has messy rows -> warn, don't crash
        encoding="utf-8",
        encoding_errors="replace",   # tolerate stray bytes instead of crashing
    )

    for chunk_no, chunk in enumerate(reader, start=1):
        rows_scanned += len(chunk)

        # unparseable values -> NaT, which fail the range test and are dropped
        dt = pd.to_datetime(chunk[date_col], errors="coerce", utc=True)

        # inclusive 2015-01-01..2023-12-31 via half-open [START, END)
        mask = (dt >= START) & (dt < END)
        kept = chunk.loc[mask, usecols]
        dt_kept = dt.loc[mask]
        n_kept = len(kept)
        rows_kept += n_kept

        # Verification (in place): running aggregates for the summary, inside the chunk loop
        if n_kept:
            # update running aggregates
            year_counts.update(dt_kept.dt.year.astype(int).tolist())
            unique_tickers.update(kept[ticker_col].dropna().unique())
            # empty, whitespace-only or missing headline
            null_headline += int((kept[headline_col].fillna("").str.strip() == "").sum())
            # any non-zero H/M/S means a real intraday timestamp
            nonmidnight += int(
                (dt_kept.dt.hour.ne(0) | dt_kept.dt.minute.ne(0) | dt_kept.dt.second.ne(0)).sum()
            )
            cmin, cmax = dt_kept.min(), dt_kept.max()
            min_date = cmin if min_date is None else min(min_date, cmin)
            max_date = cmax if max_date is None else max(max_date, cmax)
            if len(sample_raw_dates) < 5:
                sample_raw_dates.extend(kept[date_col].head(5 - len(sample_raw_dates)).tolist())

            # append to output (date, ticker, headline)
            out_df = kept[[date_col, ticker_col, headline_col]]
            out_df.to_csv(
                out_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            first_write = False

        print(
            f"  chunk {chunk_no:>3} | scanned {rows_scanned:>12,} | kept {rows_kept:>12,}",
            flush=True,
        )

        # Verification (in place): TEST_MODE early stop, inside the chunk loop
        if TEST_MODE and chunk_no >= TEST_CHUNKS:
            print(f"\nTEST_MODE: stopped after {TEST_CHUNKS} chunks.")
            break

    # ---- Verification: consistency and reproduction checks ----
    print("\n" + "=" * 78)
    print("VERIFICATION SUMMARY")
    print("=" * 78)

    # (1) totals + output file on disk
    print(f"\n[1] Totals")
    print(f"    rows scanned : {rows_scanned:,}")
    print(f"    rows kept    : {rows_kept:,}")
    if out_path.exists():
        print(f"    output file  : {out_path}")
        print(f"    output size  : {human_size(out_path.stat().st_size)}")
    else:
        print(f"    output file  : (none written — 0 rows kept)")

    if rows_kept == 0:
        print("\nNo rows fell inside the date window — nothing more to report.")
        return

    # (2) output date range
    print(f"\n[2] Date range of kept rows")
    print(f"    min : {min_date}")
    print(f"    max : {max_date}")
    in_window = (min_date >= START) and (max_date < END)
    print(f"    within 2015-01-01 .. 2023-12-31 ? {in_window}")

    # (3) per-year coverage
    print(f"\n[3] Rows per year")
    for year in range(2015, 2024):
        print(f"    {year} : {year_counts.get(year, 0):>12,}")
    # surface anything outside the intended span
    extra = sorted(y for y in year_counts if y < 2015 or y > 2023)
    for year in extra:
        print(f"    {year} : {year_counts[year]:>12,}  <-- OUTSIDE 2015-2023")

    # (4) timestamp granularity
    print(f"\n[4] Timestamp granularity")
    print(f"    5 raw date values (verbatim):")
    for v in sample_raw_dates[:5]:
        print(f"      {v!r}")
    pct_intraday = 100.0 * nonmidnight / rows_kept
    if nonmidnight == 0:
        print("    -> DATE-ONLY: every kept timestamp is at 00:00:00 "
              "(no usable intraday time).")
    else:
        print(f"    -> INTRADAY TIME present: {pct_intraday:.2f}% of kept rows "
              f"have a non-midnight HH:MM:SS.")
        print("       (Raw values also carry a 'UTC' marker; parsed as tz-aware UTC.)")

    # (5) headline nulls + unique tickers
    print(f"\n[5] Content checks")
    pct_null = 100.0 * null_headline / rows_kept
    print(f"    empty/null headlines : {null_headline:,} ({pct_null:.4f}% of kept)")
    print(f"    unique tickers       : {len(unique_tickers):,}")

    print("\nDone." + ("  (TEST run — real output untouched.)" if TEST_MODE else ""))


if __name__ == "__main__":
    main()
