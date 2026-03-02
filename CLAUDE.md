# ducklake-dataframe

> **This file is the primary onboarding document for Claude instances working on this project.
> Keep it up to date as the codebase evolves. If you add a module, change an architectural
> decision, discover a new gotcha, or fix a tricky bug, update the relevant section here.**

## What this project is

A pure-Python Polars integration for [DuckLake](https://ducklake.select/) catalogs. It reads
DuckLake metadata from SQLite (via Python's stdlib `sqlite3`) or PostgreSQL (via `psycopg2`) and
scans the underlying Parquet data files through Polars' native Parquet reader. There is **no DuckDB
runtime dependency** -- DuckDB is only used in tests to create catalog fixtures.

Public API: `scan_ducklake()` (LazyFrame), `read_ducklake()` (DataFrame), and `DuckLakeCatalog`
(catalog inspection), exported from `ducklake_polars/__init__.py`.

## Architecture

```
src/ducklake_polars/
    __init__.py      Public API: scan_ducklake(), read_ducklake(), DuckLakeCatalog
    _backend.py      Backend adapters: SQLiteBackend, PostgreSQLBackend, create_backend()
    _catalog.py      Metadata reader (snapshots, tables, columns, files, stats, inlined data)
    _catalog_api.py  DuckLakeCatalog: high-level catalog inspection API
    _dataset.py      Polars PythonDatasetProvider implementation (DuckLakeDataset)
    _schema.py       DuckLake type string -> Polars DataType mapping
    _stats.py        Column statistics builder for file pruning
```

### Data flow

1. User calls `scan_ducklake(path, table)`.
2. `__init__.py` creates a `DuckLakeDataset` dataclass and passes it to Polars'
   private `PyLazyFrame.new_from_dataset_object()`.
3. Polars calls `DuckLakeDataset.schema()` to get the table schema, and
   `DuckLakeDataset.to_dataset_scan()` to get the actual data.
4. `to_dataset_scan()` opens a `DuckLakeCatalogReader` (read-only connection via the
   backend adapter), resolves the snapshot, finds data files/delete files, builds
   statistics, and returns a `scan_parquet(sources, ...)` LazyFrame.
5. Polars handles all query optimization (predicate pushdown, projection pushdown,
   file pruning via statistics, positional deletes via Iceberg-compatible delete files).

### Key Polars internals used

- `polars._plr.PyLazyFrame.new_from_dataset_object(dataset)` -- creates a LazyFrame
  from a Python object implementing the PythonDatasetProvider protocol.
- `scan_parquet(sources, missing_columns="insert", extra_columns="ignore",
  _table_statistics=..., _deletion_files=("iceberg-position-delete", ...))` --
  the underscore-prefixed params are private Polars APIs for statistics and deletes.
- These private APIs may change across Polars versions. If something breaks after
  a Polars upgrade, check these first.

### Column rename support

When `ALTER TABLE RENAME COLUMN` is used, old Parquet files still have the old physical column name.
DuckLake tracks renames via snapshot-versioned rows in `ducklake_column` (same `column_id`, different
`column_name` across snapshot boundaries). The rename logic:

1. `get_column_history()` retrieves all column definitions across all snapshots
2. `_has_renames()` checks if any column_id has multiple distinct names (fast path: no overhead if no renames)
3. If renames exist, files are grouped by their physical column names via `_group_files_by_rename_map()`
4. Each group is scanned separately with `scan_parquet`, collected eagerly, and old-name groups get `.rename()` applied
5. Groups are concatenated, written to a temp Parquet file, and returned as `scan_parquet(tmp_path)`

**Important**: The Polars dataset scan resolver only accepts bare Parquet SCAN nodes from
`to_dataset_scan()`. Returning `.rename()` (WITH_COLUMNS), `pl.concat()` (UNION), or
`df.lazy()` (DF) all fail with "unknown DSL when resolving python dataset scan". The temp
file workaround is the only viable approach. Temp files are cleaned up via `atexit` handlers.

### Partition pruning

DuckLake stores partition metadata in three tables: `ducklake_partition_info`, `ducklake_partition_column`,
`ducklake_file_partition_value`. For identity-transform partitions, partition values supplement the
`_table_statistics` DataFrame as a fallback when `ducklake_file_column_stats` is incomplete.
The logic is in `_build_partition_values_for_stats()` and integrated into `build_table_statistics()`.

### DuckLake metadata schema

The catalog database (SQLite or PostgreSQL) contains these tables (among others):

- `ducklake_metadata` -- key-value pairs, notably `data_path`
- `ducklake_snapshot` -- snapshot_id, schema_version, snapshot_time, next_file_id
- `ducklake_schema` -- schema_id, schema_name, path, begin_snapshot, end_snapshot
- `ducklake_table` -- table_id, table_name, schema_id, path, begin_snapshot, end_snapshot
- `ducklake_column` -- column_id, table_id, column_name, column_type, column_order,
  parent_column, begin_snapshot, end_snapshot, nulls_allowed
- `ducklake_data_file` -- data_file_id, table_id, path, record_count, begin_snapshot, end_snapshot
- `ducklake_delete_file` -- delete_file_id, data_file_id, table_id, path, begin_snapshot, end_snapshot
- `ducklake_file_column_stats` -- data_file_id, column_id, null_count, min_value, max_value
- `ducklake_partition_info` -- partition_id, table_id, begin_snapshot, end_snapshot
- `ducklake_partition_column` -- partition_id, column_id, partition_key_index, transform
- `ducklake_file_partition_value` -- data_file_id, table_id, partition_key_index, partition_value
- `ducklake_inlined_data_tables` -- table_id, table_name, schema_version

Snapshot visibility: a row is visible at snapshot S when `begin_snapshot <= S AND (end_snapshot IS NULL OR end_snapshot > S)`.

Column types for compound types (list, struct, map) use a parent-child hierarchy via `parent_column`.
Top-level columns have `parent_column = NULL`. The `column_type` for compound parents is just
`"list"`, `"struct"`, or `"map"` (lowercase); children have their own types. Scalar types are
stored as lowercase DuckDB internal names like `"int32"`, `"varchar"`, `"boolean"`, `"decimal(18,3)"`.

### Type mapping

`_schema.py` has two paths for resolving types:

1. **`duckdb_type_to_polars(type_str)`** -- string-based parsing for type strings like
   `"BIGINT"`, `"VARCHAR"`, `"STRUCT(a INTEGER, b VARCHAR)"`, `"INTEGER[]"`, `"DECIMAL(18,3)"`.
   Used for the schema mapping unit tests and as a fallback.

2. **`resolve_column_type(column_id, column_type, all_columns)`** -- hierarchy-based resolution
   for the actual catalog data. Walks parent-child relationships for LIST/STRUCT/MAP, falls
   through to `duckdb_type_to_polars()` for scalar types.

**Important type mapping gotchas:**

- `TIMESTAMP_S` maps to `pl.Datetime("us")`, NOT `pl.Datetime("s")`. This is intentional --
  DuckDB writes all timestamps to Parquet as microseconds regardless of the declared precision.
  The schema must match what Polars reads from the Parquet file, not the logical DuckDB type.
- `UUID` and `JSON` map to `pl.Binary()`, not `pl.String()`. DuckDB writes these as binary
  in Parquet. Attempting to use `pl.String()` causes schema mismatch errors.
- `HUGEINT`/`UHUGEINT` map to `pl.Int128`/`pl.UInt128` in the schema, but DuckDB writes them
  as Float64 in Parquet. This is a known DuckDB limitation; tests are marked `xfail`.
- `INTERVAL` cannot be read by Polars from Parquet (month_day_millisecond_interval); `xfail`.
- `MAP` reading is broken in Polars 1.36; `xfail`.
- The LazyFrame returned from `to_dataset_scan()` **must be a bare Parquet SCAN node**.
  Polars rejects WITH_COLUMNS (`.rename()`, `.cast()`), UNION (`pl.concat()`), and DF
  (`df.lazy()`) with "unknown DSL when resolving python dataset scan". This is why the
  schema must exactly match the Parquet physical types, and the rename path uses a temp file.

## How to build and test

```bash
pip install -e ".[dev]"   # installs polars + pytest + pytest-xdist + duckdb (test only)
pytest                     # run all tests
pytest -n auto             # run in parallel
pytest tests/test_types.py -v  # run specific file
```

DuckDB is **only a test dependency**. Tests use DuckDB + the DuckLake extension to create
catalogs (SQLite-backed in temp dirs, PostgreSQL when `DUCKLAKE_PG_DSN` is set), then read
them back with ducklake-dataframe.

Test fixtures are in `tests/conftest.py`:
- `ducklake_catalog` -- data inlining **disabled** (forces Parquet files). Used by most tests.
- `ducklake_catalog_inline` -- data inlining **enabled**. Used by inlined data tests.

Both fixtures are **parametrized over available backends**: always SQLite, plus PostgreSQL when
`DUCKLAKE_PG_DSN` is set. Test IDs include the backend suffix (e.g. `test_basic_read[sqlite]`,
`test_basic_read[postgres]`). PostgreSQL tests are marked `@pytest.mark.postgres`.

The `DuckLakeTestCatalog` helper handles both backends:
- `backend` attribute: `"sqlite"` or `"postgres"`
- `query_metadata(sql, params)`: queries the underlying metadata database directly (after
  DuckDB is closed). Handles `?` → `%s` placeholder translation for PostgreSQL.
- `_cleanup_postgres_tables()`: drops all public tables between tests for isolation.

The caller **must** call `cat.close()` before reading with ducklake-dataframe to release the
SQLite file lock. For tests that need direct metadata access after closing, use
`cat.query_metadata()` instead of opening sqlite3 directly.

Note: PostgreSQL tests should not be run in parallel (`pytest-xdist`) since they share a
single database.

Current test status: **207 passed, 4 xfailed** on SQLite backend (doubled with PostgreSQL).
Known xfails: HUGEINT, UHUGEINT, INTERVAL, MAP — DuckDB Parquet writer or Polars reader limitations.

## Known limitations and future work
- **Non-default schemas** (e.g., `CREATE SCHEMA other`) are supported in the API (`schema="other"`)
  but not tested end-to-end.
- **Inlined data type coercion** -- `read_inlined_data()` returns raw SQLite values without
  casting to the catalog schema. SQLite stores everything as TEXT/INTEGER/REAL/BLOB, so complex
  types (dates, decimals, nested) may have type mismatches when combined with Parquet data.
  The `diagonal_relaxed` concat handles basic cases, but this is fragile.
- **Schema not passed to scan_parquet for data files** -- currently Polars infers the schema
  from Parquet metadata rather than using the catalog schema. This works because
  `missing_columns="insert"` and `extra_columns="ignore"` handle schema evolution, but
  passing an explicit schema could improve robustness.
- **Thread safety** -- `DuckLakeCatalogReader` is not thread-safe. The `data_path` property
  uses a check-then-act pattern without locking.
- **ENUM type** is not handled; will raise `ValueError("Unsupported DuckDB type: ...")`.
- **Rename path temp files** -- when tables have renamed columns, the rename path collects
  data eagerly and writes a temp Parquet file. These are cleaned up via `atexit` but accumulate
  during long-running processes with many `scan_ducklake` calls on renamed tables.
- **DuckLake partition syntax** -- partitioning uses `ALTER TABLE t SET PARTITIONED BY (col)`,
  NOT `CREATE TABLE ... PARTITION BY (col)`. Only identity-transform partitions are currently
  supported for pruning.

## Conventions

- Only runtime dependency is `polars >= 1.0`. PostgreSQL support requires the `postgres` extra
  (`pip install ducklake-dataframe[postgres]`). No DuckDB, no PyArrow, no other dependencies.
- All SQL queries use parameterized `?` placeholders for values (translated to `%s` for PostgreSQL
  by the `_sql()` helper). Identifiers (table names, column names) use double-quote escaping
  with embedded quotes doubled (`"` -> `""`).
- `DuckLakeCatalogReader` supports context manager (`with reader:`) and must be closed
  after use.
- Error messages include the table name, schema name, and snapshot ID for debuggability.
- Tests should always verify actual values, not just shapes. Every assertion on `.shape`
  should be accompanied by value-level checks.

## File-by-file reference

### `__init__.py`
- Public API. Validates `snapshot_version`/`snapshot_time` mutual exclusivity.
- Accepts `str | Path` for path arguments (uses `os.fspath()`).
- Uses Polars private internals: `polars._plr.PyLazyFrame`, `polars._utils.wrap.wrap_ldf`.
- `DuckLakeCatalog` is imported directly from `_catalog_api`.

### `_backend.py`
- `SQLiteBackend` / `PostgreSQLBackend` -- dataclasses handling connection creation, parameter
  placeholder style (`?` vs `%s`), and table-not-found error detection.
- `create_backend(path)` -- auto-detects backend from path/connection string.
- `psycopg2` is imported lazily only when a PostgreSQL backend is used.

### `_catalog.py`
- `DuckLakeCatalogReader` -- delegates connection to the backend adapter.
- `_sql()` helper translates `?` placeholders to the backend's style.
- Key methods: `get_current_snapshot()`, `get_snapshot_at_version()`, `get_snapshot_at_time()`,
  `get_table()`, `get_columns()`, `get_all_columns()`, `get_data_files()`, `get_delete_files()`,
  `get_column_stats()`, `read_inlined_data()`, `get_all_snapshots()`, `get_all_schemas()`,
  `get_all_tables()`, `get_all_metadata()`, `get_data_files_in_range_with_snapshot()`,
  `get_delete_files_in_range()`, `get_data_file_by_id()`, `get_column_history()`,
  `get_partition_info()`, `get_partition_columns()`, `get_file_partition_values()`.
- Dataclasses: `SnapshotInfo`, `TableInfo`, `ColumnInfo`, `FileInfo`, `DeleteFileInfo`,
  `ColumnStats`, `ColumnHistoryEntry`, `PartitionInfo`, `PartitionColumnDef`, `FilePartitionValue`.
- `resolve_data_file_path()` builds full paths: `data_path / schema_path / table_path / file_path`.
- `except Exception` + `backend.is_table_not_found()` in `get_inlined_data_tables`,
  `read_inlined_data`, and partition methods handles both SQLite and PostgreSQL missing-table errors.

### `_catalog_api.py`
- `DuckLakeCatalog` -- high-level catalog inspection class. All methods return `pl.DataFrame`
  (except `current_snapshot()` which returns `int`).
- Creates a fresh `DuckLakeCatalogReader` per method call (stateless).
- **Metadata methods**: `snapshots()`, `current_snapshot()`, `table_info()`, `list_files()`,
  `list_schemas()`, `list_tables()`, `options()`, `settings()`.
- **Change data feed**: `table_insertions(table, start, end)` reads Parquet data files added
  in the snapshot range. `table_deletions(table, start, end)` reads delete files and extracts
  deleted rows from data files. `table_changes(table, start, end)` combines both and detects
  updates (same-snapshot insert + delete → `update_preimage` / `update_postimage`).
- Supports context manager protocol (no-op cleanup since readers are per-method).

### `_dataset.py`
- `DuckLakeDataset` -- dataclass implementing PythonDatasetProvider.
- `schema()` returns the Polars Schema (called by Polars during planning).
- `to_dataset_scan()` returns `(LazyFrame, version_key)` (called by Polars during execution).
- The `limit`, `projection`, `pyarrow_predicate` params are part of the interface but unused.
- Creates a new `DuckLakeCatalogReader` per method call (no caching across schema/scan).
- Module-level helpers for rename detection: `_has_renames()`, `_get_physical_name()`,
  `_get_rename_map()`, `_group_files_by_rename_map()`.
- `_build_partition_values_for_stats()` maps partition values to column IDs for stats supplementation.
- `_safe_unlink()` for robust temp file cleanup in `atexit` handlers.
- `_build_scan_kwargs()` static method extracts shared logic for building `scan_parquet` kwargs per file group.

### `_schema.py`
- `_SIMPLE_TYPE_MAP` -- dict of ~67 entries mapping uppercase type names to Polars DataTypes.
  Both SQL standard names and DuckDB internal names are included.
- `duckdb_type_to_polars()` -- string-based recursive parser. Handles simple types, DECIMAL,
  VARCHAR(N), arrays (`TYPE[]`), LIST, MAP, STRUCT.
- `resolve_column_type()` -- hierarchy-based resolver for DuckLake's parent-child column model.
- `_parse_struct_fields()` / `_parse_single_field()` -- bracket-aware struct field parsing.
  Handles quoted field names with escaped double-quotes.

### `_stats.py`
- `_parse_stat_value()` -- converts stat strings to Python values (int, float, bool, str,
  date, datetime, Decimal). Wrapped in try/except for resilience.
- `build_table_statistics()` -- builds the DataFrame that Polars uses for file pruning.
  Format: `len` (UInt32), `{col}_nc` (UInt32), `{col}_min`, `{col}_max` per filter column.
  Accepts optional `partition_values` dict to supplement missing column stats with partition values.
