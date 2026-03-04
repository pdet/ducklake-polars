# Changelog

## v0.3.0

### Highlights

- **Unified Arrow core with Polars + pandas wrappers** — shared PyArrow-based engine powering both DataFrame flavors
- **Package renamed to `ducklake-dataframe`** — reflects multi-framework support beyond Polars
- **Object storage support (S3/GCS/Azure via fsspec)** — read and write Parquet files directly on cloud storage
- **DuckDB catalog backend** — use a local DuckDB database as the DuckLake metadata catalog

### Features

- Concurrent write safety with optimistic concurrency control
- Field-id based column mapping for robust schema evolution
- Case-insensitive table name lookup
- Add files and rewrite data files support

### Improvements

- Performance optimized catalog queries
- Schema evolution edge case handling

### Testing

- Comprehensive test suite (1500+ tests)
