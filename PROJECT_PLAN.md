# Learning plan

## Milestone 1 — Ingestion and raw data
- Understand REST request parameters, JSON arrays, Unix milliseconds, pagination, and API errors.
- Run `download` and inspect the first/last records in the saved JSON.
- Explain why the cursor advances by one interval after the last returned open time.

## Milestone 2 — Data quality
- Understand duplicate keys, chronological ordering, timestamp gaps, finite numeric values, and OHLC constraints.
- Run `validate` and inspect the JSON report.
- Decide what should be fixed, dropped, or flagged; do not silently manufacture candles.

## Milestone 3 — Storage and SQL
- Understand columnar storage, compression, Parquet schema, and predicate pushdown.
- Query the Parquet file with DuckDB and record the query outputs.

## Milestone 4 — Features and leakage
- Understand percentage returns, log returns, rolling means, and rolling volatility.
- Trace a row by hand to confirm that the signal is lagged by one candle.
- Explain why using the current candle's closing price to earn the same candle's return would leak information.

## Milestone 5 — Performance
- Compare the same calculation in Pandas, NumPy, and Numba.
- Confirm numerical equivalence before interpreting timing results.
- Separate Numba's first-call compilation time from later execution time.
- Record hardware, Python/library versions, row count, average runtime, throughput, and observed RSS changes.

## Milestone 6 — Portfolio polish
- Add your own charts, benchmark results, design decisions, limitations, and a short explanation of what you personally implemented.
- Commit source code and tests to GitHub; keep the downloaded data out of the repository.
