# DuckLake-Polars: Implementation Plan

## Project Summary

`ducklake-polars` is a Python package that provides Polars integration for DuckLake catalogs. It enables reading DuckLake tables as Polars LazyFrames/DataFrames with full support for predicate pushdown, schema evolution, time travel, and delete handling.

---

## Architecture

```
User API:
    scan_ducklake(path, table, ...) -> LazyFrame
    read_ducklake(path, table, ...) -> DataFrame

Internal:
    DuckLakeDataset (PythonDatasetProvider)
        |-- schema()           -> resolve column types from ducklake_column
        |-- to_dataset_scan()  -> resolve file list, stats, deletions
                |
                v
    DuckLakeCatalogReader (metadata access via sqlite3 or psycopg2)
        |-- connect to metadata catalog (SQLite file or PostgreSQL)
        |-- query ducklake_* tables for schema, files, stats
        |
        v
    polars.scan_parquet(
        sources=resolved_parquet_paths,
        _table_statistics=min_max_stats,
        _deletion_files=delete_file_paths,
        missing_columns="insert",
        extra_columns="ignore",
        ...
    )
```

### Dependencies

- `polars` (>= 1.0) — only runtime dependency
- `psycopg2-binary` (>= 2.9) — optional, for PostgreSQL backends (`pip install ducklake-polars[postgres]`)
- `duckdb` — **test-only** dependency (creates catalog fixtures)

---

## Package Structure

```
ducklake-polars/
    pyproject.toml
    README.md
    CLAUDE.md
    PLAN.md
    .github/workflows/tests.yml  - CI: SQLite + PostgreSQL, Python 3.9/3.12
    src/
        ducklake_polars/
            __init__.py          - public API: scan_ducklake, read_ducklake, DuckLakeCatalog
            _backend.py          - backend adapters: SQLiteBackend, PostgreSQLBackend
            _catalog.py          - DuckLakeCatalogReader: metadata queries
            _catalog_api.py      - DuckLakeCatalog: high-level catalog utility functions
            _dataset.py          - DuckLakeDataset: PythonDatasetProvider impl
            _schema.py           - DuckLake -> Polars type mapping
            _stats.py            - Statistics extraction for file pruning
    tests/
        conftest.py              - shared fixtures (parametrized SQLite + PostgreSQL)
        test_backend.py          - backend detection, placeholder, wrapper tests
        test_basic.py            - basic scan/read operations, inlined data, edge cases
        test_catalog_api.py      - DuckLakeCatalog utility function tests
        test_types.py            - all supported type round-trips
        test_schema_evolution.py - ADD/DROP/RENAME COLUMN handling
        test_schema_mapping.py   - unit tests for type string parsing
        test_time_travel.py      - version/timestamp queries
        test_delete.py           - deletion file handling
```

---

## Completed Work

### Phase 1: Core Read Path ✅

- [x] Project setup: `pyproject.toml`, package structure, pytest infrastructure
- [x] Metadata catalog reader (`_catalog.py`): snapshots, tables, columns, files, stats, inlined data
- [x] Type mapping (`_schema.py`): all DuckDB types → Polars types (67+ mappings)
- [x] Dataset provider (`_dataset.py`): PythonDatasetProvider with schema() and to_dataset_scan()
- [x] Public API (`__init__.py`): `scan_ducklake()`, `read_ducklake()` with column selection, time travel, schema override
- [x] Tests: basic operations, all types, multiple inserts, CTAS, filters, nulls, large datasets

### Phase 2: Advanced Read Features ✅

- [x] **Time travel**: `snapshot_version=N` and `snapshot_time="..."` parameters
- [x] **Schema evolution**: add column, drop column (via `missing_columns="insert"`, `extra_columns="ignore"`)
- [x] **Filter pushdown**: min/max column statistics passed via `_table_statistics`
- [x] **Deletion files**: Iceberg-compatible positional deletes via `_deletion_files`
- [x] **Data inlining**: reads inlined data from metadata catalog, combines with Parquet data
- [x] **SQLite + PostgreSQL backends**: thin adapter layer in `_backend.py`, auto-detection from connection string
- [x] **CI**: GitHub Actions with PostgreSQL service container, Python 3.9/3.12 matrix

### Current Test Status

**284 passed, 1 skipped, 10 xfailed** (both SQLite and PostgreSQL backends)

Known xfails: HUGEINT (2), UHUGEINT (2), INTERVAL (2), MAP (2), column RENAME (2) — all due to DuckDB Parquet writer or Polars reader limitations.

---

### Phase 3: Catalog Utility Functions ✅

- [x] **`DuckLakeCatalog` class** (`_catalog_api.py`): high-level catalog inspection API
- [x] **Snapshot functions**: `snapshots()`, `current_snapshot()`
- [x] **Table metadata**: `table_info()`, `list_files(table)`
- [x] **Schema/table listing**: `list_schemas()`, `list_tables()`
- [x] **Catalog options**: `options()`, `settings()`
- [x] **Change data feed**: `table_insertions()`, `table_deletions()`, `table_changes()`
  - Insertions: reads Parquet data files added in snapshot range
  - Deletions: reads delete files, extracts deleted rows from data files
  - Changes: combines insertions + deletions, detects updates (preimage/postimage)
- [x] **Tests**: `tests/test_catalog_api.py` — 21 tests parametrized over SQLite + PostgreSQL

---

## Next Phases

---

### Phase 4: Remaining Read Features

#### 4.1 Column Rename Support
- [ ] Handle `mapping_id` on data files for column name remapping
- [ ] Read `ducklake_column_mapping` / `ducklake_name_mapping` tables
- [ ] Currently xfail in tests

#### 4.2 Partition Pruning
- [ ] Read partition info from `ducklake_partition_info` / `ducklake_partition_column`
- [ ] Read partition values from `ducklake_file_partition_value`
- [ ] Use partition values for file pruning

---

### Phase 5: Write Path (Future)

- [ ] `write_ducklake(df, path, table, *, mode="append"|"overwrite")`
- [ ] Write Parquet files, register in metadata, create snapshots
- [ ] Partitioned writes
- [ ] Schema management (create table, infer schema)

---

## Resolved Design Decisions

### Q1: How do we read the metadata catalog?
**Resolved**: Direct SQLite/PostgreSQL connections via `_backend.py`. No DuckDB runtime dependency. SQLite uses Python stdlib `sqlite3` in read-only mode. PostgreSQL uses `psycopg2` with readonly+autocommit. Auto-detected from connection string.

### Q2: How do we handle deletion files?
**Resolved**: DuckLake uses Iceberg-compatible positional delete Parquet files. Passed directly to Polars via `_deletion_files=("iceberg-position-delete", ...)`. Works correctly.

### Q3: How do we handle inlined data?
**Resolved**: Read from `ducklake_inlined_data_*` tables in the metadata catalog. Combined with Parquet data via `pl.concat(..., how="diagonal_relaxed")`. Type coercion for complex types is fragile but works for basic cases.

### Q4: SQLite vs PostgreSQL?
**Resolved**: Both supported transparently. Backend adapter pattern with `SQLiteBackend` / `PostgreSQLBackend`. Placeholder translation (`?` vs `%s`), connection creation, and error detection abstracted behind the adapter interface.
