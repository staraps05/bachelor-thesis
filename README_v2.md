# News sentiment and the cross-section of weekly S&P 500 returns

This repository holds the code for a bachelor thesis that asks whether headline sentiment adds predictive content to structured firm and macro predictors for weekly S&P 500 excess returns over 2018 to 2023. Four XGBoost arms are compared on identical folds: A0 uses 30 structured inputs (22 firm characteristics and 8 macro predictors), and A1, A2 and A3 add weekly sentiment moments from the Loughran-McDonald dictionary, FinBERT and FinGPT respectively. Models are trained on expanding windows and tested on 2021, 2022 and 2023, with ten seeds (42 to 51) per fit. In one sentence, the sentiment arms do not improve on the structured baseline at the one-week horizon.

Author: Ruben Stark. University of St. Gallen, 2026. Supervisor: Prof. Dr. Despoina Makariou, Institute of Insurance Economics.

## Folder layout

```
bachelor-thesis/
├── archive/                                       # early pilot, not part of the pipeline
│   └── fingpt_sentiment_pilot.py                  # Archived Colab pilot of the FinGPT scorer (LoRA adapter on
├── pipeline/
│   ├── 1_data_build/                              # raw data to the frozen panel
│   │   ├── 01_verify_sp500_membership.py          # Verify the CRSP S&P 500 daily membership export (TSLA
│   │   ├── 06_slim_fnspid_headlines.py            # Slim the ~23 GB FNSPID CSV to Date, Stock_symbol
│   │   ├── 08_build_skeleton.py                   # Skeleton: one (PERMNO, week_anchor) row per S&P 500
│   │   ├── 09_build_target.py                     # Weekly excess-return target ret_excess and n_days
│   │   ├── 10_build_blockA.py                     # Compute the 8 Block-A price/return characteristics from
│   │   ├── 11_merge_blockB_macro.py               # Merge Block B (14 JKP monthly) and 8 Welch-Goyal macro
│   │   ├── 12_preprocess_freeze.py                # Rank-normalise the 22 characteristics week by week to [-1
│   │   └── 13_build_scoring_input.py              # Map FNSPID headlines to PERMNO and target week: input for
│   ├── 2_sentiment/                               # LM, FinBERT, FinGPT scoring
│   │   ├── colab/
│   │   │   ├── 16b_score_finbert.ipynb            # Score the 796,038 scoring-input headlines with FinBERT
│   │   │   └── 17b_score_fingpt.ipynb             # Score the 796,038 scoring-input headlines with FinGPT
│   │   ├── 15_score_lm.py                         # Score headlines with the LM dictionary, aggregate to
│   │   ├── 16a_export_for_colab.py                # Export scoring input as CSV for the Colab FinBERT notebook
│   │   ├── 16c_finbert_aggregate_freeze.py        # Validate the Colab-scored FinBERT probabilities and
│   │   ├── 17c_fingpt_aggregate_freeze.py         # Gate the Colab-scored FinGPT probabilities and aggregate
│   │   └── sentiment_aggregation.py               # Shared aggregation of per-headline sentiment scores to
│   ├── 3_models/                                  # model frame and XGBoost fits
│   │   ├── 17_pre_model_integrity.py              # Build the predictive model frame (ret_excess_fwd = week t+1
│   │   ├── 18_baseline_model.py                   # Fit the structured-only XGBoost baseline (arm A0) with
│   │   ├── 19_four_arm_comparison.py              # Four-arm comparison (A0 structured, A1 +LM, A2 +FinBERT, A3
│   │   ├── 20_seed_stability.py                   # Re-run pipeline/3_models/19 with colsample_bytree 1.0
│   │   ├── 26_single_lag_rerun.py                 # Re-run the four XGBoost arms of pipeline/3_models/20 with
│   │   ├── 27_covered_subsample.py                # Re-run the four-arm single-lag XGBoost comparison on firms
│   │   ├── 30_per_year_primary.py                 # Regenerate the primary single-lag four-arm XGBoost run and
│   │   ├── 30a_zero_delta_addendum.py             # Count how many of the 60 fold-2/3 cells (10 seeds x 3
│   │   ├── 31_importance_and_triangle.py          # Refit the primary single-lag XGBoost spec (4 arms x 3 folds
│   │   ├── 31a_a2_feature_shares.py               # Refit the A2 (FinBERT) arm of the primary single-lag spec
│   │   └── 31b_a2_block_sums.py                   # Refit the A2 (FinBERT) arm of the primary single-lag spec
│   ├── 4_analyses/                                # supporting analyses
│   │   ├── 21_ridge_exploration.py                # Exploratory ridge-regression counterpart to the four
│   │   ├── 22_dispersion_vs_count.py              # Check whether the dispersion features (std, range, kurt)
│   │   ├── 24_coverage_by_year.py                 # Test whether the sentiment null is a news-coverage
│   │   ├── 29_daily_decay.py                      # Univariate diagnostic of how fast per-headline sentiment
│   │   ├── 32_headline_age_bounds.py              # Bound the age of a headline when its target return is
│   │   └── 33_pairwise_arm_differences.py         # Original pairwise differences among the sentiment arms
│   └── 5_tables/                                  # thesis tables as full-precision CSVs
│       ├── 34_numbers_primary_table41.py          # Reproduce the primary single-lag four-arm run and save the
│       ├── 35_numbers_importance_table42.py       # Reproduce the A0 and A2 single-lag fits and save the
│       ├── 36_numbers_covered.py                  # Four-arm single-lag run on the high-coverage subsample
│       ├── 37_numbers_winsor.py                   # Four-arm single-lag run with a winsorised target: Table C.2
│       ├── 38_numbers_calm.py                     # Table C.2 calm-period rows: the four-arm single-lag run
│       ├── 39_numbers_ridge.py                    # Round-1 ridge results on the double-lag frame: Table C.2
│       ├── 40_numbers_a0prime_xgb.py              # Original A0' check (A0 plus had_news): how many of the 30
│       ├── 41_numbers_modelfree_countcorr.py      # Table C.7 (Spearman of each sentiment feature with the
│       ├── 42_numbers_daily_decay.py              # Table C.6: persistence of the sentiment-return relation at
│       ├── 43_numbers_level_share.py              # Table C.8: bias-vs-skill decomposition of the pooled R2
│       ├── 44_numbers_nonews_share.py             # Share of firm-weeks without news (had_news == 0), per year
│       ├── 45_numbers_ridge_singlelag.py          # Ridge instrument of pipeline/5_tables/39 under the
│       ├── 46_numbers_modelfree_singlelag.py      # Table C.5 (univariate slope and Q5-Q1 spread) under the
│       ├── 47_numbers_a0prime_r2delta.py          # A0' (A0 plus had_news) under the single-lag pairing: fits
│       ├── 48_numbers_sentiment_use_variants.py   # Sentiment gain share under two variants of the single-lag
│       └── 49_numbers_data_figures.py             # Recompute the Chapter 3 data figures (lost exit weeks
├── results/                                       # 45 CSV files
├── verification/                                  # read-only checks of inputs and intermediates
│   ├── audit_block11_sessionA/
│   │   └── out24.txt
│   ├── audit_block9_10/
│   │   └── c5_code26_output.log
│   ├── 02_verify_crsp_daily_prices.py             # Verify the WRDS CRSP daily stock file for the S&P 500
│   ├── 03_verify_jkp_fundamentals.py              # Verify the JKP monthly characteristics file for the S&P 500
│   ├── 04_verify_wg_macro.py                      # Verify the Welch-Goyal monthly macro file: monthly
│   ├── 05_verify_ff_factors_daily.py              # Verify the Fama-French 3-factor daily file (source of daily
│   ├── 07_verify_characteristic_selection.py      # Verify the 13 selected JKP characteristic codes plus the
│   ├── 14_ticker_variant_audit.py                 # Check whether panel firms with zero headlines in the
│   ├── 17a_export_for_colab.py                    # Confirm the Colab export finbert_input.csv is unchanged
│   ├── 23_positive_control.py                     # Positive control for the sentiment merge: test the
│   ├── 25_event_spot_check.py                     # Event-level spot check of panel rows against real-world
│   └── 28_closing_checks.py                       # Two closing checks: (1) the 22 panel_final characteristics
├── .gitignore
├── README.md
└── requirements.txt
```

## Pipeline in run order

Scripts load one another by relative path (`REPO_ROOT/pipeline/<stage>/...`), so run them from the repository root. "Hours" marks the multi-seed XGBoost fits; everything else runs in seconds to minutes.

Data build (`pipeline/1_data_build/`)
- `01_verify_sp500_membership.py`: check the CRSP membership export and write the 719-PERMNO universe.
- `06_slim_fnspid_headlines.py`: stream the 22 GB FNSPID file into a slim 2015-2023 headline table.
- `08_build_skeleton.py`: one row per firm and week while the firm is an index member.
- `09_build_target.py`: weekly excess return per skeleton row.
- `10_build_blockA.py`: the 8 price-based characteristics, cross-checked against JKP.
- `11_merge_blockB_macro.py`: merge 14 JKP characteristics and 8 Welch-Goyal macro predictors.
- `12_preprocess_freeze.py`: weekly rank normalisation and median imputation; writes the panel and its manifest.
- `13_build_scoring_input.py`: map headlines to firms point-in-time and to their target week.

Sentiment scoring (`pipeline/2_sentiment/`)
- `15_score_lm.py`: Loughran-McDonald scores and weekly moments (shared code in `sentiment_aggregation.py`).
- `16a_export_for_colab.py`: export the headline table for the notebooks.
- `colab/16b_score_finbert.ipynb` (in `2_sentiment/`, Colab GPU): FinBERT probabilities per headline.
- `16c_finbert_aggregate_freeze.py`: validate the FinBERT scores and aggregate to weekly moments.
- `verification/17a_export_for_colab.py`: confirm the export is unchanged before the FinGPT run.
- `colab/17b_score_fingpt.ipynb` (in `2_sentiment/`, Colab GPU): FinGPT probabilities per headline (Llama-2-7B base, 4-bit).
- `17c_fingpt_aggregate_freeze.py`: validate the FinGPT scores and aggregate to weekly moments.

Model frame and fits (`pipeline/3_models/`)
- `17_pre_model_integrity.py` (in `3_models/`): attach the week t+1 target, scan for leakage and degenerate inputs.
- `18_baseline_model.py`: the structured-only XGBoost baseline A0, single seed.
- `19_four_arm_comparison.py` (hours): the four arms, single seed, writes the prediction files.
- `20_seed_stability.py` (hours): the four arms over ten seeds; supplies folds, seeds and helpers to later scripts.
- `26_single_lag_rerun.py` (hours): the primary specification, sentiment attached at the one-week lag.
- `27_covered_subsample.py` (hours): the primary run on firms with high news coverage.
- `30_per_year_primary.py` (hours): per-year out-of-sample R2 for the primary run; writes `per_year_primary_r2.csv`.
- `30a_zero_delta_addendum.py`: census of fold 2 and 3 cells with a delta of exactly zero.
- `31_importance_and_triangle.py`, `31a_a2_feature_shares.py`, `31b_a2_block_sums.py` (hours): feature importance and the scorer correlation triangle.

Supporting analyses (`pipeline/4_analyses/`)
- `21_ridge_exploration.py`: ridge counterpart to the XGBoost arms.
- `22_dispersion_vs_count.py`: are the dispersion features a proxy for headline count?
- `24_coverage_by_year.py`: coverage profile and per-year tests.
- `29_daily_decay.py`: daily-horizon decay of the sentiment-return relation.
- `32_headline_age_bounds.py`: headline age bounds under the one-week construction.
- `33_pairwise_arm_differences.py`: pairwise differences among the sentiment arms.

Thesis tables (`pipeline/5_tables/`; each writes full-precision CSVs to `Data/processed/numbers_check/`, copied to `results/`)
- `34_numbers_primary_table41.py` (hours): Table 4.1 and Table C.1.
- `35_numbers_importance_table42.py` (hours): Table 4.2.
- `36_numbers_covered.py`, `37_numbers_winsor.py`, `38_numbers_calm.py` (hours): Table C.2 covered, winsorised and calm-period rows.
- `39_numbers_ridge.py`, `45_numbers_ridge_singlelag.py`: Table C.2 ridge rows, Table C.3, Table C.4.
- `41_numbers_modelfree_countcorr.py`, `46_numbers_modelfree_singlelag.py`: Table C.7 and Table C.5.
- `42_numbers_daily_decay.py`: Table C.6.
- `43_numbers_level_share.py`: Table C.8 (seed 42).
- `44_numbers_nonews_share.py`: the no-news share quoted in the text.
- `40_numbers_a0prime_xgb.py`, `47_numbers_a0prime_r2delta.py` (hours): the A0 plus news-indicator check.
- `48_numbers_sentiment_use_variants.py` (hours): sentiment gain share under two variants.
- `49_numbers_data_figures.py`: the Chapter 3 data figures (reads the 22 GB FNSPID file, about 4 minutes).

Verification scripts (`verification/`)
- `02` CRSP daily prices, `03` JKP characteristics, `04` Welch-Goyal file, `05` Fama-French daily factors, `07` characteristic selection, `14` ticker-variant audit, `17a` Colab export, `23` positive control for the sentiment merge, `25` event spot check, `28` closing checks on the panel and the level share.

## How to run

1. Python 3.9.6; `pip install -r requirements.txt`.
2. Place the raw files listed under Data in `Data/` (and `Data/colab/` for the notebook outputs).
3. From the repository root, run `python pipeline/<stage>/<script>.py` in the order above. Each script verifies its inputs by SHA-256 or row count and stops on a mismatch.
4. Colab: upload `Data/colab/finbert_input.csv` (written by `16a`) to Google Drive at `thesis_finbert/finbert_input.csv`, run `pipeline/2_sentiment/colab/16b` on a GPU runtime, download `finbert_scores.csv` into `Data/colab/`, then run `16c`. Run `verification/17a`, then `pipeline/2_sentiment/colab/17b` (the FinGPT notebook reads the same upload and installs transformers, peft, accelerate and bitsandbytes), download `fingpt_scores.csv` into `Data/colab/`, then run `17c`.

## Data

`Data/` is not part of the repository. Raw inputs:

| file | content | source | access |
|---|---|---|---|
| `Data/crsp_daily_prices.csv` | CRSP daily prices, returns, volume, shares | WRDS, CRSP daily stock file | licensed, not included |
| `Data/membership_file.csv` | CRSP S&P 500 daily index membership | WRDS, CRSP index constituents | licensed, not included |
| `Data/Fundamentals.csv` | JKP monthly firm characteristics | jkpfactors.com via WRDS | licensed, not included |
| `Data/F-F_Research_Data_Factors_daily.csv` | Fama-French three daily factors and RF | Kenneth French data library | public download |
| `Data/Macro_data_aggregated.csv` | Welch-Goyal monthly predictors | Amit Goyal's website | public download |
| `Data/fnspid_nasdaq_news_raw.csv` | FNSPID news headlines (22 GB) | FNSPID public dataset | public; SHA-256 `1a7a3eb8e6b97ec19f286f2cfca3371542bddb272ab1eb8f36e33ad98fa5c4da` |
| `Data/Loughran-McDonald_MasterDictionary_1993-2025.csv` | Loughran-McDonald Master Dictionary | sraf.nd.edu | academic licence, not included |

Derived files written by the pipeline into `Data/processed/`: `sp500_permnos.txt`, `fnspid_headlines_2015_2023.csv`, `skeleton.parquet`, `panel_target.parquet`, `panel_blockA.parquet`, `f6_crosscheck.csv`, `panel_structured.parquet`, `panel_final.parquet` and its manifest, `scoring_input.parquet`, `scoring_input_coverage.csv`, `features_lm.parquet`, `features_finbert.parquet`, `features_fingpt.parquet`, `model_frame_structured.parquet` and its manifest, `baseline_predictions.parquet`, `predictions_A0_structured.parquet` to `predictions_A3_fingpt.parquet`, `per_year_primary_r2.csv`, and the `numbers_check/` tables. `Data/colab/` holds `finbert_input.csv`, `finbert_scores.csv` and `fingpt_scores.csv`. `results/` contains only the aggregate tables from `numbers_check/`; it holds no prices, returns or other licensed values; two tables list security identifiers (PERMNO, ticker, SIC code) only.

## Software versions

The pipeline ran with Python 3.9.6, pandas 2.2.3, numpy 2.0.2, pyarrow 21.0.0 and xgboost 2.1.4; the scripts import nothing else beyond the standard library (scikit-learn and statsmodels are not used). The notebooks pin only `pandas<2.3`; FinBERT uses `ProsusAI/finbert` via transformers, and FinGPT uses `NousResearch/Llama-2-7b-hf` with the `FinGPT/fingpt-mt_llama2-7b_lora` adapter, loaded in 4-bit through bitsandbytes with peft and accelerate at the versions Colab installed at run time. Seeds: 42 to 51 for the fits; seed 42 for Table C.8.

## Reproducibility note

Every script checks its inputs against recorded SHA-256 values and row counts, and the numbers-check scripts gate on the recorded per-seed R2 before writing. The FinGPT scores were produced on two GPU types (T4 and A100); bit-exact reproduction of those scores holds per GPU type, and the downstream aggregation is deterministic given the score file.
