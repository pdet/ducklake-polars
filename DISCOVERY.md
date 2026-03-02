# DuckLake-Polars: Discovery & Research

## Overview

This document captures findings from exploring the three sibling repositories:
- **ducklake** - The DuckLake lakehouse format (DuckDB C++ extension)
- **polars** - The Polars DataFrame library
- **ducklake-dataframe** - This repo (empty, greenfield)

---

## 1. What is DuckLake?

DuckLake is an **open lakehouse format** built on SQL and Parquet. Its architecture has two components:

- **Metadata catalog**: A SQL database (DuckDB, SQLite, or PostgreSQL) storing ~22 tables prefixed `ducklake_`
- **Data files**: Parquet files on local filesystem or cloud object stores (S3/MinIO)

### Key Features

| Feature | Description |
|---------|-------------|
| CRUD | Full INSERT, UPDATE, DELETE, MERGE INTO |
| Time Travel | `AT (VERSION => N)` and `AT (TIMESTAMP => ...)` |
| Schema Evolution | ADD/DROP/RENAME COLUMN, type promotion, nested struct field evolution |
| Partitioning | Hive-style with transforms: IDENTITY, YEAR, MONTH, DAY, HOUR |
| Sort Keys | `SET SORTED BY` with configurable direction and null ordering |
| Data Inlining | Small datasets stored directly in metadata catalog |
| Compaction | `merge_adjacent_files()`, `rewrite_data_files()` |
| Snapshot Management | `expire_snapshots()`, `cleanup_old_files()`, `cleanup_orphaned_files()` |
| Change Data Feed | `table_changes()` for CDC |
| Encryption | Per-file encryption keys |
| Transactions | MVCC with snapshot isolation and conflict detection |
| Filter Pushdown | Per-file min/max column statistics |
| Views & Macros | Stored in the metadata catalog |
| Audit | Commit messages (author, message, extra_info) |

### Metadata Schema (Key Tables)

```
ducklake_metadata       - key/value config (version, data_path, encrypted)
ducklake_snapshot       - snapshot history for time travel
ducklake_schema         - schema definitions
ducklake_table          - table definitions
ducklake_column         - column definitions with types and field IDs
ducklake_data_file      - Parquet data file references (path, record_count, file_size)
ducklake_file_column_stats - per-file column statistics (min/max/null_count)
ducklake_delete_file    - deletion tracking files
ducklake_partition_info / ducklake_partition_column / ducklake_file_partition_value
ducklake_column_mapping / ducklake_name_mapping - column name evolution
ducklake_inlined_data_* / ducklake_inlined_delete_* - inlined data tables
ducklake_tag            - comments/tags
ducklake_sort_info / ducklake_sort_column - sort specifications
ducklake_view / ducklake_macro - stored views and macros
```

### Current Version: 0.4

---

## 2. Polars Integration Patterns

Polars already has two analogous lakehouse integrations that serve as **direct templates**:

### Integration Architecture

```
scan_delta() / scan_iceberg()
    |
    v
PythonDatasetProvider (Python @dataclass)
    |-- schema() -> returns table schema
    |-- to_dataset_scan() -> resolves metadata, returns LazyFrame
    |
    v
scan_parquet(paths, _table_statistics=..., _deletion_files=..., _column_mapping=...)
    |
    v
Native Rust Parquet scanner (predicate pushdown, projection pushdown, streaming)
```

### The `PythonDatasetProvider` Interface

Both Delta and Iceberg implement a Python `@dataclass` with two methods:

```python
@dataclass
class SomeDataset:
    def schema(self) -> dict[str, PolarsDataType]:
        """Return the table schema."""
        ...

    def to_dataset_scan(
        self,
        *,
        existing_resolved_version_key: str | None,
        limit: int | None,
        projection: list[str] | None,
        filter_columns: list[str] | None,
        pyarrow_predicate: Any | None,
    ) -> tuple[LazyFrame, str] | None:
        """Resolve metadata and return a scan_parquet() LazyFrame + version key."""
        ...
```

The version key mechanism enables caching: if the table hasn't changed, `to_dataset_scan()` returns `None`.

### Private `scan_parquet()` Parameters

Delta and Iceberg pass resolved metadata into the native Parquet scanner via private `_`-prefixed parameters:

| Parameter | Used By | Purpose |
|-----------|---------|---------|
| `_table_statistics` | Delta, Iceberg | Min/max stats for file pruning |
| `_deletion_files` | Iceberg | Position delete files |
| `_column_mapping` | Iceberg | Field ID to column name mapping |
| `_default_values` | Iceberg | Default values for missing columns |
| `_row_count` | Iceberg | (physical, deleted) tuple for fast COUNT(*) |

### Key Files in Polars

```
py-polars/src/polars/io/delta/
    functions.py        - read_delta(), scan_delta()
    _dataset.py         - DeltaDataset (PythonDatasetProvider)
    _utils.py           - statistics extraction

py-polars/src/polars/io/iceberg/
    functions.py        - scan_iceberg()
    _dataset.py         - IcebergDataset (PythonDatasetProvider)
    _utils.py           - statistics, partitions, predicates
```

### Other Integration Mechanisms

- **`_scan_python_function`** / Anonymous Scan: Generic Python callback, used as fallback
- **`register_io_source()`**: Public IO plugin API for pure-Python sources
- **Native Rust Scan**: For Parquet/CSV/IPC/NDJSON directly

---

## 3. DuckLake Test Infrastructure

### Framework

DuckLake uses **DuckDB SQLLogicTest** format (`.test` / `.test_slow` files). There are **354 test files** across **49 directories**. No Python tests exist.

### Test Setup Pattern

```sql
require ducklake
require parquet

test-env DUCKLAKE_CONNECTION __TEST_DIR__/{UUID}.db
test-env DATA_PATH __TEST_DIR__

ATTACH 'ducklake:${DUCKLAKE_CONNECTION}' AS ducklake (DATA_PATH '${DATA_PATH}/ducklake_files')
```

### Test Configurations

| Config | Backend |
|--------|---------|
| `attach_ducklake.json` | DuckDB core test suite against DuckLake |
| `sqlite.json` | SQLite metadata catalog |
| `postgres.json` | PostgreSQL metadata catalog |
| `no_inline.json` | Data inlining disabled |
| `minio.json` | S3-compatible storage |

### Test Categories (by file count)

| Category | Files | Key Tests |
|----------|-------|-----------|
| Data Inlining | 32 | `basic_data_inlining.test` |
| Add Files | 31 | `add_files.test` |
| Compaction | 28 | `small_insert_compaction.test` |
| Sorted Table | 27 | Various sorted merge/flush tests |
| ALTER/Schema | 25 | `add_column.test`, `struct_evolution.test` |
| Deletion Inlining | 16 | `test_deletion_inlining.test` |
| General | 13 | `attach_at_snapshot.test` |
| Transactions | 12 | `basic_transaction.test`, `transaction_conflicts.test` |
| Types | 11 | `all_types.test` |
| Partitioning | 11 | `basic_partitioning.test` |
| Stats/Filter | 11 | `filter_pushdown.test` |
| Delete | 11 | `basic_delete.test` |
| Macros | 10 | `test_simple_macro.test` |
| Rewrite Files | 10 | `test_rewrite_db.test` |
| Table Changes | 8 | `ducklake_table_changes.test` |
| Views | 8 | `ducklake_view.test` |
| Update | 7 | `basic_update.test` |
| Merge | 6 | `merge_update_insert.test` |
| Concurrent | 5 | `concurrent_insert_conflict.test` |
| Comments | 5 | `comments.test` |
| Time Travel | 2 | `basic_time_travel.test` |

---

## 4. Design Decisions for ducklake-dataframe

### Standalone Package vs. Polars Contribution

**Decision: Standalone Python package** (`ducklake-dataframe`)

Rationale:
- DuckLake is experimental (v0.x), not yet stable enough for Polars core
- Faster iteration cycle as an independent package
- Can still follow the same `PythonDatasetProvider` pattern
- Depends on `polars` and `duckdb` Python packages

### Read vs. Write Support

**Phase 1: Read-only** (`scan_ducklake()` returning `LazyFrame`)
**Phase 2: Write support** (`write_ducklake()` from `DataFrame`)

Rationale: Read path is well-understood from Iceberg/Delta patterns. Write path requires deeper integration with DuckLake's transaction/snapshot system.

### Metadata Access Strategy

**Use `duckdb` Python package** to query the metadata catalog directly.

The metadata catalog is a DuckDB (or SQLite/Postgres) database with well-defined tables. We can:
1. Connect to the catalog database with `duckdb.connect()`
2. Query `ducklake_table`, `ducklake_column`, `ducklake_data_file`, etc.
3. Resolve file paths, schemas, statistics, and deletion info
4. Pass everything to `scan_parquet()` via the private parameters

This avoids depending on the C++ DuckLake extension for reads and gives us full control.
