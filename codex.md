# KKBox Codebase Notes

## Purpose and scope
- Focus: predict subscription churn for KKBox users using Kaggle challenge data.
- Primary output: a feature table suitable for ML training (DuckDB table and Parquet export).
- Data scale: hundreds of millions of user_log rows, tens of millions of transactions, ~730k users.

## Data sources and domain framing
- Source: Kaggle KKBox Churn Prediction Challenge CSVs (raw_train_v1/v2, raw_transactions_v1/v2, raw_user_logs_v1/v2, raw_members_v3).
- Time window: 2015-01-01 to 2017-03-31.
- Target label: is_churn for March 2017 (class imbalance around 4.5 percent).
- Key domain entities: train (labels), members (profiles), transactions (payments), user_logs (daily usage).

## Repository layout
- `README.md` documents setup, data placement, and pipeline overview.
- `src/main.py` implements the preprocessing pipeline that builds merge tables and prepares data for features.
- `src/utils.py` holds reusable DuckDB operations, transformations, and analysis helpers.
- `src/feature-engineer-for-ml.py` builds ML features from preprocessed tables.
- `src/feature-engineer-for-daily.py` builds a dense daily time series table for per-day features and joins.
- `src/code-guide.md`, `src/structure-guide.md`, `src/feature-engineer-plan-for-ml.md`, `src/feature-formulas.md` capture style and feature specifications.
- Note: README references `src/feature-engineer.py`, but the repo contains `src/feature-engineer-for-ml.py` and `src/feature-engineer-for-daily.py` instead.

## Preprocessing pipeline (main.py)
- CSV ingest: chunked read into DuckDB with progress output and row counts.
- Table naming normalization: add `_v1` suffix for v1 sources.
- Merge strategy:
  - train: v2 overrides v1 by user id.
  - members: `members_v3` is copied to `members_merge`.
  - other tables: v1 and v2 are concatenated with UNION ALL.
- ID mapping: build `user_id_map` from unique `msno` across tables, then replace `msno` with integer `user_id`.
- Filtering steps:
  - keep intersection of ids across core tables.
  - filter by reference table ids.
  - churn transition filter (v1 is_churn = 0, v2 is_churn in {0,1}).
  - remove users with duplicate same-day non-cancel transactions.
  - optional filters for NULL total_secs and out-of-range expire dates.
- Type conversions: gender to int, date columns to DATE, `msno` renamed to `user_id`.
- Analysis outputs: churn transition heatmap, duplicate transaction analysis, feature correlation heatmap.
- Export: selected tables to Parquet and/or CSV.

## Feature engineering for ML (feature-engineer-for-ml.py)
- Phased build in DuckDB:
  - Phase 1: `user_last_txn`, `user_membership_history`, `user_logs_filtered`.
  - Phase 2: `user_logs_features` (21), `transactions_features` (7), `members_features` (1).
  - Phase 3: `ml_features` join and Parquet export.
- Membership windowing:
  - Last transaction defines `membership_period` and the log window.
  - Logs are filtered to [transaction_date, membership_expire_date).
- Listening time normalization:
  - `total_hours` is used instead of `total_secs` and is clipped to 0-24 hours.
- Missing value handling:
  - `fill_nan` toggles NULL retention vs default value fill using COALESCE.
- Feature sets (high level):
  - User logs: averages, gap stats, activity ratios, weekday/weekend ratio, acceleration, max ratio, weekly std, 4 weekly buckets, skip/complete ratios.
  - Transactions: plan_days, last_is_cancel, payment_method_id, membership_duration, tx_seq_length, cancel_exist, had_churn.
  - Members: clipped registration_init as days since 2015-01-01.

## Daily feature pipeline (feature-engineer-for-daily.py)
- Builds a dense user-day grid for a fixed date range with missing days filled with zeros.
- Adds `date_idx` as integer days since base date.
- Supports sorting by user_id and date_idx.
- Joins user_logs with transactions by user_id and date.
- Forward-fill helper for transaction fields with optional expiry check.
- Provides a date lower-bound clipping utility.

## Utility layer (utils.py)
- Common DB ops: drop, copy, rename column, add column, table existence, row counts.
- Data I/O: CSV load with chunking; export to Parquet and CSV.
- Transformations: nullify out-of-range values, conditional transforms, add converted columns, fill null values.
- Analysis: clipping distribution plots, correlation heatmaps, column stats.

## Implementation patterns and conventions
- Strong logging discipline with explicit start and completion banners.
- Functions are small, single-responsibility, and parameterized with explicit names.
- Type hints everywhere, Google-style docstrings.
- Sectioned scripts using 76-char separator blocks.
- Guard clauses for table and column existence before heavy operations.
- DuckDB-first approach with SQL CTEs and full-table replacement patterns:
  - Create temp tables, drop original, then rename.
  - Use PRAGMA threads and optional memory_limit.
  - Use `DESCRIBE`, `SHOW TABLES`, `COPY` extensively.

## Performance and scale considerations
- Large-scale data; pipelines are designed for chunked CSV ingestion and heavy SQL aggregation.
- Multithreaded DuckDB execution and optional memory limit.
- Some workflows are intentionally stepwise and allow skipping via commented calls.

## Data quality assumptions and known issues
- total_secs can overflow; out-of-range values are set to NULL and total_hours is clipped.
- `bd` is age (not birth year) and contains outliers.
- gender uses -1 to indicate missing, and NULLs are common.
- Time leakage is explicitly called out in docs; feature windows must not exceed the prediction cutoff.

## Outputs and artifacts
- DuckDB database at `data/data.duckdb` with `_merge`, `_seq`, and feature tables.
- Parquet outputs under `data/parquet` (notably `ml_features.parquet`).
- Analysis plots under `data/analysis` (heatmaps and feature histograms).

## How to run (documented)
- `uv sync` for environment setup.
- `uv run python src/main.py` for preprocessing.
- `uv run python src/feature-engineer-for-ml.py` for ML feature build.
- `uv run python src/feature-engineer-for-daily.py` for daily pipeline steps.
