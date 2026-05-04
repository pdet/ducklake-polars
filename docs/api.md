# API Reference

## ducklake_polars

### Read Operations

#### `scan_ducklake`

```python
scan_ducklake(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    snapshot_version: int | None = None,
    snapshot_time: datetime | str | None = None,
    data_path: str | Path | None = None,
) -> pl.LazyFrame
```

Lazily read a DuckLake table as a Polars LazyFrame. Predicates, projections, and file pruning are pushed down through Polars' native optimizer.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | required | Path to catalog file (`.ducklake` or `.db`), or a PostgreSQL connection string |
| `table` | `str` | required | Name of the table to read |
| `schema` | `str` | `"main"` | Schema name |
| `snapshot_version` | `int \| None` | `None` | Read at a specific snapshot version |
| `snapshot_time` | `datetime \| str \| None` | `None` | Read at a specific timestamp (datetime or ISO string) |
| `data_path` | `str \| Path \| None` | `None` | Override the data path stored in the catalog |

**Returns:** `pl.LazyFrame`

**Raises:** `ValueError` if both `snapshot_version` and `snapshot_time` are specified.

---

#### `read_ducklake` (Polars)

```python
read_ducklake(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    columns: list[str] | None = None,
    snapshot_version: int | None = None,
    snapshot_time: datetime | str | None = None,
    data_path: str | Path | None = None,
) -> pl.DataFrame
```

Eagerly read a DuckLake table into a Polars DataFrame. Convenience wrapper around `scan_ducklake(...).collect()`.

**Additional parameter:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `columns` | `list[str] \| None` | `None` | Columns to select. If `None`, reads all columns |

---

### Write Operations

#### `write_ducklake` (Polars)

```python
write_ducklake(
    df: pl.DataFrame,
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    mode: str = "error",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
    max_retries: int = 3,
    retry_wait_ms: float = 100,
    retry_backoff: float = 2.0,
    schema_evolution: str = "strict",
    union_handling: str = "error",
) -> None
```

Write a Polars DataFrame to a DuckLake table.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pl.DataFrame` | required | DataFrame to write |
| `path` | `str \| Path` | required | Path to catalog file |
| `table` | `str` | required | Table name |
| `schema` | `str` | `"main"` | Schema name |
| `mode` | `str` | `"error"` | Write mode: `"error"`, `"append"`, or `"overwrite"` |
| `data_path` | `str \| Path \| None` | `None` | Override data path |
| `data_inlining_row_limit` | `int` | `0` | Max rows to store inline in the catalog (0 = disabled) |
| `author` | `str \| None` | `None` | Author name for the snapshot |
| `commit_message` | `str \| None` | `None` | Commit message for the snapshot |
| `max_retries` | `int` | `3` | Max transaction retries on conflict |
| `retry_wait_ms` | `float` | `100` | Initial wait between retries (ms) |
| `retry_backoff` | `float` | `2.0` | Backoff multiplier for retries |
| `schema_evolution` | `str` | `"strict"` | `"strict"` to fail on column mismatch; `"merge"` auto-adds new DataFrame columns to the table |
| `union_handling` | `str` | `"error"` | How to handle Polars/Arrow union types when writing |

---

#### `create_table_as_ducklake` (Polars)

```python
create_table_as_ducklake(
    df: pl.DataFrame,
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

Create a new table and insert data in a single atomic snapshot. Equivalent to `CREATE TABLE ... AS SELECT ...`.

---

#### `delete_ducklake` (Polars)

```python
delete_ducklake(
    path: str | Path,
    table: str,
    predicate: pl.Expr,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> int
```

Delete rows matching a predicate from a DuckLake table. Creates Iceberg-compatible position-delete files.

**Returns:** `int` — number of rows deleted.

---

#### `update_ducklake` (Polars)

```python
update_ducklake(
    path: str | Path,
    table: str,
    updates: dict[str, object],
    predicate: pl.Expr,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> int
```

Update rows matching a predicate. Atomically deletes old rows and inserts new rows with updated values in a single snapshot.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `updates` | `dict[str, object]` | Column names → new values. Values can be literals or `pl.Expr` |
| `predicate` | `pl.Expr` | Boolean expression. Rows where `True` are updated |

**Returns:** `int` — number of rows updated.

---

#### `merge_ducklake` (Polars)

```python
merge_ducklake(
    path: str | Path,
    table: str,
    source_df: pl.DataFrame,
    on: str | list[str],
    *,
    when_matched_update: dict[str, object] | bool | None = None,
    when_matched_delete: bool = False,
    when_not_matched_insert: bool = True,
    when_not_matched_by_source_delete: bool = False,
    when_not_matched_by_source_update: dict[str, object] | None = None,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> tuple[int, int]
```

Merge a source DataFrame into an existing DuckLake table (upsert). Implemented as delete + insert in a single snapshot.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `on` | `str \| list[str]` | required | Column(s) to match on |
| `when_matched_update` | `dict \| bool \| None` | `None` | `None` = leave matched rows; `True` = replace; `dict` = update specific columns |
| `when_matched_delete` | `bool` | `False` | Delete target rows that match a source row |
| `when_not_matched_insert` | `bool` | `True` | Insert unmatched source rows |
| `when_not_matched_by_source_delete` | `bool` | `False` | Delete target rows that have no matching source row |
| `when_not_matched_by_source_update` | `dict \| None` | `None` | Update target rows that have no matching source row |

**Returns:** `tuple[int, int]` — `(rows_updated, rows_inserted)`.

---

#### `add_files_ducklake`

```python
add_files_ducklake(
    path: str | Path,
    table: str,
    file_paths: list[str],
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
    max_retries: int = 3,
    retry_wait_ms: float = 100,
    retry_backoff: float = 2.0,
) -> int
```

Register existing Parquet files into a DuckLake table. Files are referenced in-place (not copied). Schema is validated against the table.

**Returns:** `int` — the new snapshot ID.

---

### DDL Operations

#### `create_ducklake_table` (Polars)

```python
create_ducklake_table(
    path: str | Path,
    table: str,
    polars_schema: pl.Schema | dict[str, pl.DataType],
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

Create a new empty table using a Polars schema.

---

#### `drop_ducklake_table`

```python
drop_ducklake_table(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

---

#### `rename_ducklake_table`

```python
rename_ducklake_table(
    path: str | Path,
    old_table: str,
    new_table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

---

#### `alter_ducklake_add_column` (Polars)

```python
alter_ducklake_add_column(
    path: str | Path,
    table: str,
    col_name: str,
    dtype: pl.DataType,
    *,
    default: object = None,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

Add a column to a table. Existing rows get `NULL` (or the specified `default`).

---

#### `alter_ducklake_drop_column`

```python
alter_ducklake_drop_column(
    path: str | Path,
    table: str,
    col_name: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

---

#### `alter_ducklake_rename_column`

```python
alter_ducklake_rename_column(
    path: str | Path,
    table: str,
    old_col_name: str,
    new_col_name: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

Rename a column. Old Parquet files with the old column name are reconciled transparently via column history.

---

#### `alter_ducklake_set_type`

```python
alter_ducklake_set_type(
    path: str | Path,
    table: str,
    column_name: str,
    new_type: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

Change a column's type using a DuckDB type string (e.g., `"BIGINT"`, `"VARCHAR"`, `"DOUBLE"`). Existing Parquet files keep their original types; the reader casts on the fly.

---

#### `alter_ducklake_set_partitioned_by`

```python
alter_ducklake_set_partitioned_by(
    path: str | Path,
    table: str,
    columns: list[str] | list[tuple[str, str]],
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

Set partitioning on a table. Pass either a list of column names (each gets the `identity` transform) or a list of `(column, transform)` tuples. Supported transforms: `"identity"`, `"year"`, `"month"`, `"day"`, `"hour"` — `year`/`month`/`day`/`hour` require a date or timestamp column. Future inserts write one Parquet file per unique combination of partition values (Hive-style layout). DuckDB-DuckLake's `bucket(N, col)` and `truncate(N, col)` transforms are not supported.

---

#### `alter_ducklake_set_sort_keys`

```python
alter_ducklake_set_sort_keys(
    path: str | Path,
    table: str,
    sort_keys: list[str | tuple[str, str] | tuple[str, str, str]],
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

Set sort keys on a table. Each element can be:

- `"col"` — ascending, nulls last
- `("col", "DESC")` — descending, nulls last
- `("col", "ASC", "NULLS_FIRST")` — ascending, nulls first

---

#### `alter_ducklake_reset_sort_keys`

```python
alter_ducklake_reset_sort_keys(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

Remove sort keys from a table.

---

#### `create_ducklake_schema`

```python
create_ducklake_schema(
    path: str | Path,
    schema_name: str,
    *,
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

---

#### `drop_ducklake_schema`

```python
drop_ducklake_schema(
    path: str | Path,
    schema_name: str,
    *,
    cascade: bool = False,
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

Drop a schema. Pass `cascade=True` to drop all tables in the schema first.

---

#### `create_ducklake_view`

```python
create_ducklake_view(
    path: str | Path,
    view_name: str,
    sql: str,
    *,
    schema: str = "main",
    or_replace: bool = False,
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

---

#### `drop_ducklake_view`

```python
drop_ducklake_view(
    path: str | Path,
    view_name: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

---

### Tags

#### `set_ducklake_table_tag`

```python
set_ducklake_table_tag(
    path: str | Path,
    table: str,
    key: str,
    value: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

Set a key-value tag on a table. The `"comment"` key is interoperable with DuckDB's `COMMENT ON TABLE`.

---

#### `set_ducklake_column_tag`

```python
set_ducklake_column_tag(
    path: str | Path,
    table: str,
    column: str,
    key: str,
    value: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

---

#### `delete_ducklake_table_tag`

```python
delete_ducklake_table_tag(
    path: str | Path,
    table: str,
    key: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

---

#### `delete_ducklake_column_tag`

```python
delete_ducklake_column_tag(
    path: str | Path,
    table: str,
    column: str,
    key: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

---

### Maintenance

#### `expire_snapshots`

```python
expire_snapshots(
    path: str | Path,
    *,
    older_than_snapshot: int | None = None,
    keep_last_n: int | None = None,
    data_path: str | Path | None = None,
) -> int
```

Expire old snapshots and clean up associated metadata. This is a metadata-only operation — call `vacuum_ducklake` afterwards to delete orphaned Parquet files.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `older_than_snapshot` | `int \| None` | Expire snapshots with `snapshot_id < older_than_snapshot` |
| `keep_last_n` | `int \| None` | Keep the most recent *n* snapshots, expire the rest |

**Returns:** `int` — number of snapshots expired.

---

#### `vacuum_ducklake`

```python
vacuum_ducklake(
    path: str | Path,
    *,
    data_path: str | Path | None = None,
) -> int
```

Delete orphaned Parquet files not referenced by any catalog entry. Run `expire_snapshots` first.

**Returns:** `int` — number of Parquet files deleted.

---

#### `rewrite_data_files_ducklake`

```python
rewrite_data_files_ducklake(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
    max_retries: int = 3,
    retry_wait_ms: float = 100,
    retry_backoff: float = 2.0,
) -> int
```

Rewrite data files for compaction — merge small files and remove deleted rows. Reads all active data files, writes consolidated Parquet files, and updates the catalog atomically.

**Returns:** `int` — the new snapshot ID, or `-1` if no rewrite was needed.

---

#### `merge_adjacent_files_ducklake`

```python
merge_adjacent_files_ducklake(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    target_file_size: int | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> int
```

Lightweight compaction (DuckLake catalog v1.0+, requires DuckDB ≥ 1.5 for full DuckDB-side interop). Merges already-adjacent small files within a partition without rewriting unrelated files. Retired source files are queued for deletion via `cleanup_old_files_ducklake`.

**Returns:** `int` — new snapshot ID, or `-1` if nothing to merge.

---

#### `cleanup_old_files_ducklake`

```python
cleanup_old_files_ducklake(
    path: str | Path,
    *,
    older_than: datetime | None = None,
    data_path: str | Path | None = None,
) -> int
```

Drain the deletion queue populated by compaction / rewrite. Only physically deletes Parquet files whose scheduled-delete timestamp is before `older_than` (defaults to *now*). Pair with `rewrite_data_files_ducklake` or `merge_adjacent_files_ducklake`.

**Returns:** `int` — files deleted.

---

#### `delete_orphaned_files_ducklake`

```python
delete_orphaned_files_ducklake(
    path: str | Path,
    *,
    data_path: str | Path | None = None,
    dry_run: bool = False,
) -> list[str]
```

Find files in the data directory that the catalog does not reference and delete them (or list them when `dry_run=True`). Differs from `vacuum_ducklake` by scanning the filesystem rather than the catalog.

---

### Change Data Feed

#### `scan_ducklake_changes` (Polars-only)

```python
scan_ducklake_changes(
    path: str | Path,
    table: str,
    *,
    start_version: int,
    end_version: int,
    schema: str = "main",
    data_path: str | Path | None = None,
) -> pl.LazyFrame
```

Lazy variant of `read_ducklake_changes`. Returns rows with a `change_type` column: `"insert"`, `"delete"`, `"update_preimage"`, `"update_postimage"`.

#### `read_ducklake_changes`

```python
read_ducklake_changes(
    path: str | Path,
    table: str,
    *,
    start_version: int,
    end_version: int,
    schema: str = "main",
    data_path: str | Path | None = None,
) -> pl.DataFrame  # or pd.DataFrame in ducklake_pandas
```

Eager change feed between two snapshots. `start_version` is exclusive, `end_version` inclusive. Passing `start > end` raises `ValueError`; an `end` past the current snapshot raises a "snapshot not found" error.

PySpark variant takes `spark` as the first argument and uses `start_snapshot=` / `end_snapshot=`.

---

### Streaming Ingestion

#### `DuckLakeStreamWriter` (Polars and Pandas)

```python
from ducklake_polars import DuckLakeStreamWriter  # or ducklake_pandas

with DuckLakeStreamWriter(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    flush_threshold: int = 10_000,
    compact_on_close: bool = True,
    schema_evolution: str = "strict",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) as writer:
    writer.append(batch)        # auto-flushes at flush_threshold rows
    writer.flush()              # force flush
print(writer.total_rows, writer.flush_count)
```

Buffered micro-batch writer. Each flush is one snapshot. On exception, unflushed rows in the buffer are dropped — already-flushed batches stay visible. Not available in `ducklake_pyspark`.

---

### Macros (DuckLake v1.0+)

```python
create_ducklake_macro(
    path: str | Path,
    name: str,
    sql: str,
    *,
    schema: str = "main",
    or_replace: bool = False,
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None

drop_ducklake_macro(
    path: str | Path,
    name: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

Create/drop a DuckLake macro. Not available in `ducklake_pyspark`. Requires a v1.0 catalog; raises an explicit version error against pre-1.0 catalogs.

---

### Catalog Options (DuckLake v1.0+)

```python
set_ducklake_option(
    path: str | Path,
    key: str,
    value: str | None,
    *,
    scope: str = "global",
    table: str | None = None,
    column: str | None = None,
    schema: str = "main",
    data_path: str | Path | None = None,
) -> None
```

Set a catalog/table/column option. `scope` is one of `"global"`, `"table"`, or `"column"`. Pass `value=None` to clear an option. Not available in `ducklake_pyspark`.

---

### Catalog Migration

```python
from ducklake_polars import migrate_catalog
# also re-exported from ducklake_pandas and ducklake_pyspark

new_version = migrate_catalog(path: str | Path) -> str
```

Bring a v0.3 / v0.4 catalog up to **v1.0** in place. Idempotent. Migration is opt-in — never triggered automatically on read or write. After migration, v1.0-only operations (`merge_adjacent_files_ducklake`, macros, expression sort keys / defaults, custom column tag keys) become available.

---

### Top-level catalog helpers

These return `pl.DataFrame` (Polars), `pd.DataFrame` (Pandas), or `pyspark.sql.DataFrame` (PySpark). They are thin wrappers around the corresponding `DuckLakeCatalog` methods, useful when you don't want to construct a catalog object.

| Function | Description |
|---|---|
| `list_schemas(path, *, snapshot_version=, data_path=)` | All schemas |
| `list_tables(path, *, schema="main", snapshot_version=, data_path=)` | Tables in a schema |
| `list_views(path, *, schema="main", data_path=)` | Views in a schema |
| `get_view(path, name, *, schema="main", data_path=)` | View definition (SQL) |
| `list_snapshots(path, *, data_path=)` | All snapshots |
| `snapshot_changes(path, *, snapshot_id=, data_path=)` | Per-snapshot change summary |
| `catalog_info(path, *, data_path=)` | Backend type, data path, format version |
| `table_info(path, table, *, schema="main", data_path=)` | Per-column metadata for a single table |

---

### Catalog Inspection

#### `DuckLakeCatalog` (Polars)

```python
from ducklake_polars import DuckLakeCatalog

catalog = DuckLakeCatalog(path: str | Path, *, data_path: str | Path | None = None)
```

High-level interface for inspecting DuckLake catalog metadata. All methods return `pl.DataFrame` (except `current_snapshot()` which returns `int`).

| Method | Returns | Description |
|---|---|---|
| `snapshots()` | DataFrame with `snapshot_id`, `snapshot_time`, `schema_version` | All snapshots |
| `current_snapshot()` | `int` | Latest snapshot ID |
| `table_info(schema=)` | DataFrame with `table_name`, `table_id`, `file_count`, `file_size_bytes`, `delete_file_count`, `delete_row_count` | Per-table storage stats |
| `list_files(table, schema=, snapshot_version=)` | DataFrame with `data_file`, `data_file_size_bytes`, `delete_file`, `delete_row_count` | Files for a table |
| `list_schemas(snapshot_version=)` | DataFrame with `schema_id`, `schema_name` | All schemas |
| `list_tables(schema=, snapshot_version=)` | DataFrame with `table_id`, `table_name` | Tables in a schema |
| `options()` | DataFrame with `key`, `value` | Catalog key-value metadata |
| `settings()` | DataFrame with `catalog_type`, `data_path` | Backend type and data path |
| `table_tags(table, schema=)` | DataFrame with `key`, `value` | Table-level tags |
| `column_tags(table, column, schema=)` | DataFrame with `key`, `value` | Column-level tags |
| `sort_keys(table, schema=)` | DataFrame | Active sort keys with direction and null ordering |
| `table_insertions(table, start_version, end_version, schema=)` | DataFrame with `snapshot_id` + table columns | Rows inserted between snapshots |
| `table_deletions(table, start_version, end_version, schema=)` | DataFrame with `snapshot_id` + table columns | Rows deleted between snapshots |
| `table_changes(table, start_version, end_version, schema=)` | DataFrame with `snapshot_id`, `change_type` + table columns | Full change data feed |

The `change_type` column in `table_changes` can be: `"insert"`, `"delete"`, `"update_preimage"`, `"update_postimage"`.

---

## ducklake_pandas

The Pandas API mirrors the Polars API with these key differences:

### Differences from Polars

| Feature | Polars | Pandas |
|---|---|---|
| Lazy evaluation | `scan_ducklake()` returns `LazyFrame` | Not available |
| Read predicates | Pushdown via Polars optimizer | `predicate=` callable on `read_ducklake()` |
| DML predicates | `pl.Expr` (e.g., `pl.col("id") == 2`) | Callable `(df -> Series[bool])` or `True`/`False` |
| Update values | Literal or `pl.Expr` | Literal or callable `(df -> Series)` |
| Table creation schema | `pl.Schema` or `dict[str, pl.DataType]` | `dict[str, str]` of DuckDB type strings |
| Return types | `pl.DataFrame` | `pd.DataFrame` |

### `read_ducklake` (Pandas)

```python
from ducklake_pandas import read_ducklake

read_ducklake(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    columns: list[str] | None = None,
    predicate: Callable[[pd.DataFrame], pd.Series] | None = None,
    snapshot_version: int | None = None,
    snapshot_time: datetime | str | None = None,
    data_path: str | Path | None = None,
) -> pd.DataFrame
```

The `predicate` parameter accepts a callable for partition pruning and stats-based file pruning:

```python
df = read_ducklake("catalog.ducklake", "users", predicate=lambda df: df["region"] == "US")
```

### `write_ducklake` (Pandas)

```python
from ducklake_pandas import write_ducklake

write_ducklake(
    df: pd.DataFrame,
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    mode: str = "error",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> None
```

### `delete_ducklake` (Pandas)

```python
from ducklake_pandas import delete_ducklake

delete_ducklake(
    path: str | Path,
    table: str,
    predicate: Callable[[pd.DataFrame], pd.Series] | bool,
    *,
    schema: str = "main",
    ...
) -> int
```

Predicate is a callable or `True` (delete all rows) / `False` (delete nothing):

```python
deleted = delete_ducklake("catalog.ducklake", "users", lambda df: df["id"] == 2)
```

### `update_ducklake` (Pandas)

```python
from ducklake_pandas import update_ducklake

update_ducklake(
    path: str | Path,
    table: str,
    updates: dict[str, Any],
    predicate: Callable[[pd.DataFrame], pd.Series],
    *,
    schema: str = "main",
    ...
) -> int
```

Update values can be literals or callables:

```python
updated = update_ducklake(
    "catalog.ducklake", "users",
    updates={"region": "APAC"},
    predicate=lambda df: df["name"] == "Eve",
)
```

### `create_ducklake_table` (Pandas)

```python
from ducklake_pandas import create_ducklake_table

create_ducklake_table(
    path: str | Path,
    table: str,
    schema_dict: dict[str, str],
    *,
    schema: str = "main",
    ...
) -> None
```

Uses DuckDB type strings instead of Polars types:

```python
create_ducklake_table("catalog.ducklake", "events", {"ts": "timestamp", "value": "double"})
```

### `DuckLakeCatalog` (Pandas)

```python
from ducklake_pandas import DuckLakeCatalog

catalog = DuckLakeCatalog("catalog.ducklake")
catalog.snapshots()  # returns pd.DataFrame
```

All methods have identical signatures to the Polars version but return `pd.DataFrame` instead of `pl.DataFrame`.

### Shared API

All other functions (DDL, tags, views, maintenance, merge) share the same signatures as the Polars versions. Import from `ducklake_pandas` instead of `ducklake_polars`.

---

## ducklake_pyspark

The PySpark API mirrors the write/DDL surface of the Polars and Pandas wrappers but exposes it on top of a PySpark `SparkSession`.

### Differences from Polars / Pandas

| Feature | Polars / Pandas | PySpark |
|---|---|---|
| Read entry point | `scan_ducklake` / `read_ducklake(path, table)` | `read_ducklake(spark, path, table)` — first arg is a `SparkSession` |
| Lazy evaluation | `scan_ducklake` (Polars) | Spark DataFrames are lazy by default |
| Predicate pushdown | Polars optimizer / Pandas `predicate=` callable | Spark's native Parquet pushdown — no `predicate=` param |
| DML predicates | `pl.Expr` / callable | SQL string (`"id > 10"`) |
| `create_ducklake_table` schema | `pl.Schema` / `dict[str, str]` | `pyspark.sql.types.StructType` |
| Change data feed | `start_version` / `end_version` | `start_snapshot` / `end_snapshot` |
| `DuckLakeCatalog` class | ✅ | ❌ — use the top-level helpers (`list_schemas`, `list_tables`, `table_info`, `list_snapshots`, `snapshot_changes`, `catalog_info`, `get_view`) |
| `DuckLakeStreamWriter` | ✅ | ❌ |
| `merge_adjacent_files_ducklake` | ✅ | ❌ — only `rewrite_data_files_ducklake` |
| `cleanup_old_files_ducklake` / `delete_orphaned_files_ducklake` | ✅ | ❌ |
| `create_ducklake_macro` / `drop_ducklake_macro` | ✅ | ❌ |
| `set_ducklake_option` | ✅ | ❌ |

### `read_ducklake` (PySpark)

```python
from pyspark.sql import SparkSession
from ducklake_pyspark import read_ducklake

read_ducklake(
    spark: SparkSession,
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    columns: list[str] | None = None,
    snapshot_version: int | None = None,
    snapshot_time: datetime | str | None = None,
    data_path: str | Path | None = None,
) -> pyspark.sql.DataFrame
```

### `write_ducklake` (PySpark)

```python
from ducklake_pyspark import write_ducklake

write_ducklake(
    df: pyspark.sql.DataFrame,
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    mode: str = "error",
    data_path: str | Path | None = None,
    ...
) -> None
```

### `delete_ducklake` (PySpark)

```python
from ducklake_pyspark import delete_ducklake

delete_ducklake(
    path: str | Path,
    table: str,
    predicate: str,
    *,
    schema: str = "main",
    ...
) -> int
```

`predicate` is a SQL string evaluated against table columns:

```python
deleted = delete_ducklake("catalog.ducklake", "users", "id > 10")
```

### `create_ducklake_table` (PySpark)

```python
from pyspark.sql.types import StructType, StructField, LongType, StringType
from ducklake_pyspark import create_ducklake_table

create_ducklake_table(
    "catalog.ducklake",
    "events",
    StructType([
        StructField("id", LongType(), True),
        StructField("name", StringType(), True),
    ]),
)
```

### `read_ducklake_changes` (PySpark)

```python
from ducklake_pyspark import read_ducklake_changes

read_ducklake_changes(
    spark, path, table,
    *, start_snapshot: int, end_snapshot: int, schema: str = "main", ...
) -> pyspark.sql.DataFrame
```

### Shared surface

`update_ducklake`, `merge_ducklake`, `create_table_as_ducklake`, `add_files_ducklake`, all `alter_*` functions, `drop_ducklake_table`, `rename_ducklake_table`, `create_ducklake_schema`, `drop_ducklake_schema`, `create_ducklake_view`, `drop_ducklake_view`, the four tag functions, `expire_snapshots`, `vacuum_ducklake`, `rewrite_data_files_ducklake`, and the top-level catalog helpers (`list_schemas`, `list_tables`, `list_views`, `list_snapshots`, `snapshot_changes`, `catalog_info`, `get_view`, `table_info`) all behave the same as in the Polars wrapper. `migrate_catalog` and `TransactionConflictError` are re-exported.

---

## `TransactionConflictError`

```python
from ducklake_polars import TransactionConflictError
# or
from ducklake_pandas import TransactionConflictError
# or
from ducklake_pyspark import TransactionConflictError
```

Raised when a write operation detects a concurrent modification conflict. See [Concurrency](concurrency.md) for details.

## `UnsupportedUnionTypeError`

```python
from ducklake_polars import UnsupportedUnionTypeError
```

Raised when a write would produce a DuckDB `UNION` typed column. The wrappers do not map UNION types — see the [Compatibility page](compatibility.md). Re-exported from `ducklake_polars` only.
