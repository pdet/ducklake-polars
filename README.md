# ducklake-dataframe

> **This project is a proof of concept. It was 100% written by [Claude Code](https://docs.anthropic.com/en/docs/build-with-claude/claude-code/overview) (Anthropic's AI coding agent). It is not intended for production use.**

Pure-Python [Polars](https://pola.rs/), [Pandas](https://pandas.pydata.org/), and [PySpark](https://spark.apache.org/docs/latest/api/python/) integration for [DuckLake](https://ducklake.select/) catalogs — both read and write.

Reads and writes DuckLake metadata directly from SQLite, PostgreSQL, or DuckDB catalog files and scans the underlying Parquet data files through each engine's native Parquet reader or PyArrow. **No DuckDB runtime dependency.** With Polars you get lazy evaluation, predicate pushdown, projection pushdown, file pruning, and all other Polars optimizations out of the box. With Pandas you get familiar DataFrame ergonomics with partition and statistics-based file pruning. With PySpark you get distributed reads and writes with schema evolution and position-delete handling.

## Why DuckLake?

DuckLake's SQL-database-backed catalog (SQLite or PostgreSQL) turns operations that require rewriting JSON manifest files in Iceberg into simple indexed SQL queries. Here's how `ducklake-dataframe` compares to PyIceberg on identical workloads (100K rows, ARM64 4-core server):

| Category | DuckLake | Iceberg | Speedup |
|---|---|---|---|
| **Streaming append** (100 × 1K-row batches) | 1.1s | 9.1s | **8×** |
| **Scan after streaming** (100 small files) | 0.02s | 0.63s | **30×** |
| **Schema evolution — rename** (50 ops) | 0.27s | 27.0s | **100×** |
| **Schema evolution — add column** (50 ops) | 0.26s | 5.0s | **20×** |
| **Upsert / merge** (50% overlap) | 0.64s | 6.9s | **11×** |
| **Time travel** (read snapshot 50/100) | 0.01s | 0.26s | **26×** |
| **Partition pruning** (1 of 20) | 0.01s | 0.04s | **3×** |
| **Baseline read/write** | ~0.1s | ~0.1s | ~1× |

> DuckLake is fastest where catalogs matter most: schema changes, snapshot lookups, streaming ingestion, and merge. Baseline read/write performance is comparable — the bottleneck there is Parquet I/O, not the catalog.
>
> Iceberg wins one benchmark: reads after 50 column renames, where its field-ID-based Parquet metadata avoids name-mapping resolution.

<details>
<summary>Full benchmark details and methodology</summary>

**Hardware:** ARM64, 4 cores, 7.6 GiB RAM · **Software:** Python 3.11, Polars 1.38.1, PyArrow 23.0.1, DuckDB 1.5.2, PyIceberg 0.11.1, ducklake-dataframe v0.4.0

**Streaming benchmark** (100 batches × 1K rows):

| Scenario | DuckLake | Iceberg | Speedup |
|---|---|---|---|
| Streaming append | 1.10s (90K rows/s) | 9.08s (11K rows/s) | 8.2× |
| Read-after-write | 0.32s (62K rows/s) | 2.07s (10K rows/s) | 6.4× |
| Scan + filter | 0.02s (4.7M rows/s) | 0.63s (158K rows/s) | 29.5× |
| Aggregation | 0.02s (5.0M rows/s) | 0.55s (183K rows/s) | 27.3× |
| Compaction | 0.12s (100→1 files) | N/A | — |
| Time travel | 0.003s | 0.011s | 3.6× |

**Schema evolution** (50 ops, 100K rows):

| Scenario | DuckLake | Iceberg | Speedup |
|---|---|---|---|
| Add column (50×) | 0.26s (196 ops/s) | 5.01s (10 ops/s) | 19.6× |
| Read after adds | 0.02s (6.3M rows/s) | 0.04s (2.3M rows/s) | 2.8× |
| Rename column (50×) | 0.27s (184 ops/s) | 27.0s (2 ops/s) | 99.5× |
| Read after renames | 0.07s (1.4M rows/s) | 0.03s (3.6M rows/s) | 0.4× *(Iceberg faster)* |
| Schema churn (50 cycles) | 0.53s (190 ops/s) | 3.73s (27 ops/s) | 7.1× |
| Wide table projection (200→5 cols) | 0.02s (6.5M rows/s) | 0.03s (3.3M rows/s) | 2.0× |

**DML** (100K rows, 20 delete rounds):

| Scenario | DuckLake | Iceberg | Speedup |
|---|---|---|---|
| Selective delete (10%) | 0.08s (1.2M rows/s) | 0.12s (855K rows/s) | 1.4× |
| Read after delete | 0.02s (4.5M rows/s) | 0.02s (3.9M rows/s) | 1.2× |
| Bulk update (20%) | 0.10s (1.0M rows/s) | 0.10s (1.0M rows/s) | ~1.0× |
| Upsert merge (50% overlap) | 0.64s (157K rows/s) | 6.91s (14K rows/s) | 10.9× |
| Delete cascade (20 rounds) | 2.40s (42K rows/s) | 5.53s (18K rows/s) | 2.3× |

**Catalog operations** (100 snapshots, 50K rows):

| Scenario | DuckLake | Iceberg | Speedup |
|---|---|---|---|
| Cold start (create→read) | 0.33s | 0.47s | 1.4× |
| Snapshot history (100) | 0.002s | 0.006s | 2.9× |
| Multi-table list (100) | 0.001s | 0.003s | 2.8× |
| Partition pruning (1/20) | 0.01s (8.9M rows/s) | 0.04s (2.8M rows/s) | 3.2× |
| Time travel (snap 50/100) | 0.01s (2.5M rows/s) | 0.26s (99K rows/s) | 26.0× |

Run the benchmarks yourself: `python benchmarks/bench_streaming.py --batches 100 --batch-size 1000`

See the [full benchmark wiki](https://github.com/pdet/ducklake-dataframe/wiki/Benchmarks) for details.

</details>

## Architecture

```
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  ducklake_polars │  │  ducklake_pandas │  │  ducklake_pyspark │  ← Thin wrappers (API + reader)
└────────┬────────┘  └────────┬────────┘  └────────┬─────────┘
         │                    │                    │
         └────────────┬───────┴────────────────────┘
                      │
             ┌────────▼────────┐
             │   ducklake_core  │   ← Shared engine (catalog, writer, schema, backend)
             └─────────────────┘
```

- **`ducklake_core`** — All catalog I/O, write operations, schema mapping, and backend adapters. Uses [PyArrow](https://arrow.apache.org/docs/python/) as the internal data representation.
- **`ducklake_polars`** — Polars-specific reader (lazy `scan_parquet` via `PythonDatasetProvider`), plus a thin API that converts between Polars types and Arrow.
- **`ducklake_pandas`** — Pandas-specific reader (eager via PyArrow → Pandas conversion), plus a thin API that converts between Pandas types and Arrow.
- **`ducklake_pyspark`** — PySpark-specific reader (distributed via Spark's native Parquet reader), plus a thin API that converts between PySpark types and Arrow.

All wrappers delegate to the shared core for writes, DDL, catalog inspection, and maintenance operations.

## Installation

```bash
pip install ducklake-dataframe[polars]           # Polars engine
pip install ducklake-dataframe[pandas]           # Pandas engine
pip install ducklake-dataframe[pyspark]          # PySpark engine
pip install ducklake-dataframe[polars,pandas]    # Polars + Pandas
pip install ducklake-dataframe[polars,postgres]  # Polars + PostgreSQL catalog
pip install ducklake-dataframe[polars,s3]        # Polars + S3 object storage
pip install ducklake-dataframe[all]              # Everything
```

**Core dependency:** `pyarrow >= 10.0` only. Everything else is optional:
- **Engines:** `polars >= 1.0`, `pandas >= 1.5`, `pyspark >= 3.4` — install at least one
- **Catalogs:** SQLite (built-in), PostgreSQL (`[postgres]`), DuckDB (`[duckdb]`)
- **Storage:** Local (built-in), S3 (`[s3]`), GCS (`[gcs]`), Azure (`[azure]`)

## Tutorial

### 1. Create a catalog and write data

```python
import polars as pl
from ducklake_polars import write_ducklake

# Create a new table (mode="error" fails if it already exists)
df = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Carol"],
    "region": ["US", "EU", "US"],
})
write_ducklake(df, "catalog.ducklake", "users", mode="error")

# Append more rows
more = pl.DataFrame({"id": [4, 5], "name": ["Dave", "Eve"], "region": ["EU", "US"]})
write_ducklake(more, "catalog.ducklake", "users", mode="append")
```

### 2. Query with lazy evaluation

```python
from ducklake_polars import scan_ducklake

# Lazy scan — predicates and projections are pushed down
lf = scan_ducklake("catalog.ducklake", "users")
result = (
    lf.filter(pl.col("region") == "US")
      .select("id", "name")
      .collect()
)
print(result)
# shape: (3, 2)
# ┌─────┬───────┐
# │ id  ┆ name  │
# │ --- ┆ ---   │
# │ i64 ┆ str   │
# ╞═════╪═══════╡
# │ 1   ┆ Alice │
# │ 3   ┆ Carol │
# │ 5   ┆ Eve   │
# └─────┴───────┘
```

### 3. Delete, update, merge

```python
from ducklake_polars import delete_ducklake, update_ducklake, merge_ducklake

# Delete rows
deleted = delete_ducklake("catalog.ducklake", "users", pl.col("id") == 2)

# Update rows
updated = update_ducklake(
    "catalog.ducklake", "users",
    updates={"region": "APAC"},
    predicate=pl.col("name") == "Eve",
)

# Merge (upsert) — atomic delete + insert in one snapshot.
# The source DataFrame must include every column of the target table; pass NULL
# for columns you don't want to set on inserted rows.
source = pl.DataFrame({"id": [1, 6], "name": ["Alice2", "Frank"], "region": ["US", "EU"]})
rows_updated, rows_inserted = merge_ducklake(
    "catalog.ducklake", "users", source, on="id",
    when_matched_update=True,
    when_not_matched_insert=True,
)
```

### 4. Evolve the schema

```python
from ducklake_polars import (
    alter_ducklake_add_column,
    alter_ducklake_rename_column,
    alter_ducklake_set_type,
)

# Add a column — existing rows get NULL
alter_ducklake_add_column("catalog.ducklake", "users", "email", pl.String)

# Rename a column — old Parquet files are reconciled transparently
alter_ducklake_rename_column("catalog.ducklake", "users", "email", "contact")

# Change a column's type — reader casts old data on the fly
alter_ducklake_set_type("catalog.ducklake", "users", "id", "BIGINT")
```

### 5. Time travel

```python
from ducklake_polars import read_ducklake, DuckLakeCatalog

# Read at a specific snapshot — pick the snapshot id from the catalog
catalog = DuckLakeCatalog("catalog.ducklake")
first_data_snapshot = int(catalog.snapshots()["snapshot_id"].sort()[1])  # skip the bootstrap snapshot
df_at_first = read_ducklake("catalog.ducklake", "users", snapshot_version=first_data_snapshot)

# Or read at a specific timestamp — pull one from the catalog so this is reproducible
snapshot_ts = catalog.snapshots()["snapshot_time"].sort()[1]
df_ts = read_ducklake("catalog.ducklake", "users", snapshot_time=snapshot_ts)
```

### 6. Inspect the catalog

```python
from ducklake_polars import DuckLakeCatalog

catalog = DuckLakeCatalog("catalog.ducklake")

catalog.snapshots()                                          # All snapshots
catalog.current_snapshot()                                   # Latest snapshot ID
catalog.table_info()                                         # Per-table file/size stats
catalog.list_files("users")                                  # Data + delete files
catalog.list_schemas()                                       # All schemas
catalog.list_tables()                                        # Tables in default schema
catalog.table_changes("users", start_version=1, end_version=5)  # Change data feed
```

### 7. DuckDB interoperability

Catalogs are fully interoperable — create with DuckDB, read with ducklake-dataframe, or vice versa:

```python
import duckdb

# DuckDB writes data
con = duckdb.connect()
con.execute("INSTALL ducklake; LOAD ducklake")
con.execute("ATTACH 'ducklake:sqlite:catalog.ducklake' AS lake (DATA_PATH 'data/')")
con.execute("CREATE TABLE lake.events (ts TIMESTAMP, value DOUBLE)")
con.execute("INSERT INTO lake.events VALUES ('2025-01-01', 42.0)")
con.close()

# Polars reads it — no DuckDB needed at runtime
from ducklake_polars import scan_ducklake
lf = scan_ducklake("catalog.ducklake", "events")
print(lf.collect())
```

The reverse also works: write with ducklake-dataframe, query with DuckDB SQL.

## API Reference

### Read operations

| Function | Description |
|---|---|
| `scan_ducklake(path, table, ...)` | Polars-only. Returns a `LazyFrame` with full predicate/projection pushdown and file pruning. |
| `read_ducklake(path, table, ...)` | Eager read into `DataFrame` (Polars/PySpark) or `pd.DataFrame` (Pandas). Supports `columns=` for projection. |
| `read_ducklake_changes(path, table, ...)` | CDC: read insertions and deletions between two snapshot versions. Returns rows with `change_type` column. Polars/Pandas use `start_version`/`end_version`; PySpark uses `start_snapshot`/`end_snapshot`. |

All read functions support `snapshot_version=`, `snapshot_time=`, `schema=`, and `data_path=` overrides.

Pandas `read_ducklake` also accepts `predicate=` (a callable `df -> Series[bool]`) for partition and stats-based file pruning.

PySpark `read_ducklake` takes `spark` as the first argument (the active `SparkSession`).

### Write operations

| Function | Description |
|---|---|
| `write_ducklake(df, path, table, mode=...)` | Insert data. Modes: `"error"` (default), `"append"`, `"overwrite"`. Supports `schema_evolution="merge"` for auto-adding new columns. |
| `create_table_as_ducklake(df, path, table)` | Create table + insert data in a single atomic snapshot. |
| `delete_ducklake(path, table, predicate)` | Delete matching rows. Polars: `pl.Expr`; Pandas: callable or `True`; PySpark: SQL string. |
| `update_ducklake(path, table, updates, predicate)` | Atomic delete + insert for matched rows. |
| `merge_ducklake(path, table, source_df, on=...)` | Upsert with `when_matched_update` and `when_not_matched_insert`. |
| `add_files_ducklake(path, table, file_paths)` | Register existing Parquet files into a table without copying. |

All write operations support `author=` and `commit_message=` for snapshot metadata, `data_inlining_row_limit=` for small-data inlining, and OCC retry parameters (`max_retries=`, `retry_wait_ms=`).

### DDL operations

| Function | Description |
|---|---|
| `create_ducklake_table(path, table, schema)` | Create an empty table. Polars: `pl.Schema`; Pandas: `dict[str, str]` of DuckDB types; PySpark: `StructType`. |
| `drop_ducklake_table(path, table)` | Drop a table. |
| `rename_ducklake_table(path, old, new)` | Rename a table. |
| `alter_ducklake_add_column(path, table, col, dtype)` | Add a column (with optional `default=`). |
| `alter_ducklake_drop_column(path, table, col)` | Drop a column. |
| `alter_ducklake_rename_column(path, table, old, new)` | Rename a column. |
| `alter_ducklake_set_type(path, table, col, new_type)` | Change column type (DuckDB type string). |
| `alter_ducklake_set_partitioned_by(path, table, cols)` | Set identity-transform partitioning. |
| `alter_ducklake_set_sort_keys(path, table, keys)` | Set sort keys (`"col"`, `("col", "DESC")`, or `("col", "ASC", "NULLS_FIRST")`). |
| `alter_ducklake_reset_sort_keys(path, table)` | Remove sort keys. |
| `create_ducklake_schema(path, name)` | Create a schema. |
| `drop_ducklake_schema(path, name, cascade=)` | Drop a schema (with optional `cascade=True`). |
| `create_ducklake_view(path, name, sql, or_replace=)` | Create a view. |
| `drop_ducklake_view(path, name)` | Drop a view. |
| `set_ducklake_table_tag(path, table, key, value)` | Set a table tag (key-value metadata). |
| `set_ducklake_column_tag(path, table, col, key, value)` | Set a column tag. |
| `delete_ducklake_table_tag(path, table, key)` | Remove a table tag. |
| `delete_ducklake_column_tag(path, table, col, key)` | Remove a column tag. |

### Catalog inspection

```python
from ducklake_polars import DuckLakeCatalog  # or ducklake_pandas (not available in ducklake_pyspark)

catalog = DuckLakeCatalog("catalog.ducklake")
```

| Method | Description |
|---|---|
| `.snapshots()` | All snapshots with `snapshot_id`, `snapshot_time`, `schema_version`. |
| `.current_snapshot()` | Latest snapshot ID (int). |
| `.list_schemas()` | All schemas. |
| `.list_tables(schema=)` | All tables in a schema. |
| `.table_info(schema=)` | Per-table storage stats (file count, size, delete files). |
| `.list_files(table, schema=)` | Data files and delete files for a table. |
| `.options()` | Catalog key-value metadata. |
| `.settings()` | Backend type and data path. |
| `.table_tags(table)` | Table-level tags. |
| `.column_tags(table, column)` | Column-level tags. |
| `.sort_keys(table)` | Active sort keys with direction and null ordering. |
| `.table_insertions(table, start, end)` | Rows inserted between snapshots. |
| `.table_deletions(table, start, end)` | Rows deleted between snapshots. |
| `.table_changes(table, start, end)` | Full change data feed (`insert`, `delete`, `update_preimage`, `update_postimage`). |

Polars wrapper returns `pl.DataFrame`; Pandas wrapper returns `pd.DataFrame`; core returns `pa.Table`.

### Maintenance

| Function | Description |
|---|---|
| `expire_snapshots(path, keep_last_n=)` | Remove old snapshot metadata. Also accepts `older_than_snapshot=`. |
| `vacuum_ducklake(path)` | Delete orphaned Parquet files not referenced by the catalog. |
| `rewrite_data_files_ducklake(path, table)` | Compact data files — merges small files and applies pending deletes. |

## Optimistic Concurrency Control (OCC)

All write operations use optimistic concurrency control to detect and handle concurrent writes safely. When a transaction commits, it validates against the latest catalog state and raises `TransactionConflictError` if a conflict is detected.

### Conflict detection

Conflicts are detected on:
- **Table-level**: concurrent DDL on the same table (drop, rename, schema changes)
- **File-level**: concurrent writes touching the same data files
- **Partition-level**: concurrent writes to the same partition values

### Automatic retry

Write operations automatically retry on conflict with exponential backoff:

```python
from ducklake_polars import write_ducklake

write_ducklake(
    df, "catalog.ducklake", "users",
    mode="append",
    max_retries=3,          # Retry up to 3 times on conflict (default)
    retry_wait_ms=100,      # Initial wait between retries in ms (default)
    retry_backoff=2.0,      # Exponential backoff multiplier (default)
)
```

The core writer also uses snapshot-level retries for INSERT race conditions on the `ducklake_snapshot` table:
- `max_snapshot_retries=5` — retries on duplicate snapshot ID
- `snapshot_retry_wait_ms=50` — wait between snapshot retries

### Handling conflicts manually

```python
from ducklake_core._writer import TransactionConflictError

try:
    write_ducklake(df, "catalog.ducklake", "users", mode="append", max_retries=0)
except TransactionConflictError as e:
    print(f"Conflict: {e}")
    # Re-read, resolve, retry
```

## Catalog migration (v0.3 / v0.4 → v1.0)

Older catalogs created by earlier DuckDB-ducklake builds can be brought up to v1.0 in place. Migration is opt-in (we don't auto-migrate on read or write) and idempotent:

```python
from ducklake_polars import migrate_catalog  # also re-exported from ducklake_pandas / ducklake_pyspark

new_version = migrate_catalog("legacy.ducklake")  # "1.0"
migrate_catalog("legacy.ducklake")                # idempotent — returns "1.0"
```

After migration, v1.0-only operations become available (`merge_adjacent_files_ducklake`, macros, expression sort keys / defaults, custom column tag keys). Note that v1.0 catalogs are only readable by DuckDB ≥ 1.5 on the DuckDB side.

## Streaming ingestion

The `DuckLakeStreamWriter` (Polars and Pandas wrappers; not available in `ducklake_pyspark`) provides buffered micro-batch ingestion with auto-flush and compaction:

```python
from ducklake_polars import DuckLakeStreamWriter

with DuckLakeStreamWriter("catalog.ducklake", "events", flush_threshold=10_000) as writer:
    for batch in data_source:
        writer.append(batch)  # auto-flushes at threshold
    # auto-compacts on close

print(f"Wrote {writer.total_rows} rows in {writer.flush_count} flushes")
```

Parameters:
- `flush_threshold=10000` — rows before auto-flush
- `compact_on_close=True` — run `rewrite_data_files` on close
- `schema_evolution="strict"` — or `"merge"` to auto-add new columns

**Exception handling:** if the context exits via an exception, any unflushed rows in the buffer are dropped (no partial micro-batch is committed). Already-flushed batches that landed before the exception remain visible — DuckLake gives no cross-flush atomicity, so design downstream consumers accordingly.

## Change data feed

`scan_ducklake_changes` / `read_ducklake_changes` return a row-level diff between two snapshots with a `change_type` column (`insert`, `delete`, `update_preimage`, `update_postimage`):

```python
from ducklake_polars import read_ducklake_changes, DuckLakeCatalog

cat = DuckLakeCatalog("catalog.ducklake")
latest = cat.current_snapshot()
changes = read_ducklake_changes("catalog.ducklake", "users", start_version=0, end_version=latest)

# Just inserts / deletes via the catalog API
cat.table_insertions("users", start_version=0, end_version=latest)
cat.table_deletions("users", start_version=0, end_version=latest)
```

The `start_version` is exclusive and `end_version` inclusive; passing `start > end` raises `ValueError` and an `end` past the current snapshot raises a "snapshot not found" error.

## Compaction

`merge_adjacent_files_ducklake` and `cleanup_old_files_ducklake` are available in the Polars and Pandas wrappers (not in `ducklake_pyspark`); `rewrite_data_files_ducklake` is available in all three. `merge_adjacent_files_ducklake` (DuckLake v1.0+, requires DuckDB ≥ 1.5 for full DuckDB-side interop) merges already-adjacent small files within partitions and is the lightweight option. `rewrite_data_files_ducklake` is the heavier alternative that fully rewrites all of a table's data files (and applies pending positional deletes). Both schedule retired source files for physical deletion via `cleanup_old_files_ducklake`.

```python
from ducklake_polars import (
    rewrite_data_files_ducklake,
    merge_adjacent_files_ducklake,
    cleanup_old_files_ducklake,
)
from datetime import datetime, timedelta, timezone

# After many small appends:
rewrite_data_files_ducklake("catalog.ducklake", "events")

# Drain the deletion queue (only deletes files older than the cutoff):
removed = cleanup_old_files_ducklake(
    "catalog.ducklake",
    older_than=datetime.now(timezone.utc) + timedelta(days=1),
)
```

## Features

### Read path
- **Lazy and eager reads** — `scan_ducklake()` (Polars LazyFrame) / `read_ducklake()` (eager)
- **Predicate and projection pushdown** — through Polars' native optimizer; Pandas supports `columns=` and `predicate=`
- **File pruning** — column-level min/max statistics and partition values
- **Time travel** — by snapshot version or timestamp
- **Delete file handling** — Iceberg-compatible positional deletes (cumulative delete files supported)
- **Schema evolution** — ADD/DROP/RENAME COLUMN handled transparently across file versions
- **Inlined data** — small tables stored directly in catalog metadata, read transparently
- **Column renames** — old Parquet files with old column names seamlessly reconciled via column history
- **Change data capture** — `read_ducklake_changes()` returns insertions/deletions between snapshots

### Write path
- **INSERT** — append, overwrite, or error-on-exists modes
- **DELETE** — predicate-based row deletion with Iceberg position-delete files
- **UPDATE** — atomic delete + insert in a single snapshot
- **MERGE** — upsert with configurable matched/unmatched behavior
- **CREATE TABLE AS** — single-snapshot table creation with data
- **ADD FILES** — register existing Parquet files without copying
- **Data inlining** — small inserts stored as rows in catalog metadata (configurable threshold)
- **Partitioned writes** — Hive-style directory layout per partition key
- **Sort keys** — data sorted before writing Parquet for better row group statistics
- **Author/commit metadata** — `author=` and `commit_message=` on all write operations
- **Schema evolution on write** — `schema_evolution="merge"` auto-adds new columns
- **Streaming ingestion** — `DuckLakeStreamWriter` for buffered micro-batch writes with auto-compaction
- **Optimistic concurrency control** — automatic conflict detection and retry with exponential backoff

### DDL
- **CREATE/DROP TABLE** with full snapshot versioning
- **ADD/DROP/RENAME COLUMN** with schema evolution tracking
- **SET TYPE** — column type changes tracked in schema history
- **CREATE/DROP SCHEMA** with cascade support
- **RENAME TABLE** preserving table identity
- **SET/RESET PARTITIONED BY** — identity-transform partitioning
- **SET/RESET SORTED BY** — with `ASC`/`DESC` and `NULLS_FIRST`/`NULLS_LAST`
- **CREATE/DROP VIEW** with `OR REPLACE` support
- **Tags** — key-value metadata on tables and columns (interoperable with DuckDB's `COMMENT ON`)

### Maintenance
- **`expire_snapshots`** — remove old snapshot metadata
- **`vacuum_ducklake`** — delete orphaned Parquet files
- **`rewrite_data_files_ducklake`** — compact small files and apply pending deletes

### Catalog backends
- **SQLite** — Python stdlib `sqlite3` (zero dependency). Catalogs are auto-flipped to WAL mode on first write so concurrent readers don't collide with an in-flight writer.
- **DuckDB** — `.ducklake` files are SQLite-format, read via `sqlite3`
- **PostgreSQL** — via `psycopg2` (optional `[postgres]` extra)

Catalogs are fully interoperable with DuckDB's native DuckLake extension. New catalogs are bootstrapped at format **v1.0**; **v0.3** and **v0.4** catalogs are read-compatible (basic writes also work). Up-migration is **opt-in** — call `migrate_catalog(path)` (re-exported from `ducklake_polars`, `ducklake_pandas`, and `ducklake_pyspark`) to bring an older catalog up to v1.0 in place. v1.0-only operations (macros, `merge_adjacent_files`, expression sort keys, expression defaults, custom column tag keys) raise an explicit version error against pre-1.0 catalogs.

## Data types

| DuckLake / DuckDB type | Polars type | Notes |
|---|---|---|
| `TINYINT` – `BIGINT` | `Int8` – `Int64` | |
| `UTINYINT` – `UBIGINT` | `UInt8` – `UInt64` | |
| `FLOAT` / `DOUBLE` | `Float32` / `Float64` | |
| `BOOLEAN` | `Boolean` | |
| `VARCHAR` | `String` | |
| `BLOB` | `Binary` | |
| `DATE` | `Date` | |
| `TIME` / `TIMETZ` / `TIME_NS` | `Time` | |
| `TIMESTAMP` (all precisions) | `Datetime("us"/"ms"/"ns")` | `TIMESTAMP_S` maps to `Datetime("us")` since DuckDB writes all timestamps to Parquet as microseconds |
| `TIMESTAMPTZ` | `Datetime("us", "UTC")` | |
| `DECIMAL(p, s)` | `Decimal(p, s)` | |
| `UUID` | `Binary` | 16-byte binary in Parquet |
| `JSON` | `Binary` | Cast to `String` for text access |
| `HUGEINT` / `UHUGEINT` | `Int128` / `UInt128` | ⚠️ DuckDB writes as Float64 in Parquet |
| `INTERVAL` | `Duration("us")` | ⚠️ Polars Parquet reader limitation |
| `LIST(T)` | `List(T)` | Recursive nesting supported |
| `STRUCT(...)` | `Struct(...)` | Recursive nesting, field renames tracked |
| `MAP(K, V)` | `List(Struct(key, value))` | ⚠️ Polars Parquet reader limitation |
| `ENUM` | `String` | Mapped to VARCHAR |
| `GEOMETRY` | `Binary` | |
| `BIT` | `String` | |
| `VARIANT` | `String` | Schema-mapped; binary interop with DuckDB not supported |

## Pandas usage

The `ducklake_pandas` package mirrors the Polars API with Pandas-idiomatic differences:

```python
import pandas as pd
from ducklake_pandas import read_ducklake, write_ducklake, delete_ducklake, DuckLakeCatalog

# Read with optional predicate for file pruning
df = read_ducklake("catalog.ducklake", "users", predicate=lambda df: df["region"] == "US")

# Write
write_ducklake(pd.DataFrame({"id": [1], "name": ["Alice"]}), "catalog.ducklake", "users", mode="append")

# Delete (predicate is a callable, not pl.Expr)
deleted = delete_ducklake("catalog.ducklake", "users", lambda df: df["id"] == 2)

# Table creation uses DuckDB type strings instead of Polars types
from ducklake_pandas import create_ducklake_table
create_ducklake_table("catalog.ducklake", "events", {"ts": "timestamp", "value": "double"})

# Catalog API returns pd.DataFrame
catalog = DuckLakeCatalog("catalog.ducklake")
catalog.snapshots()  # pd.DataFrame
```

Key differences from the Polars wrapper:
- No lazy evaluation (`scan_ducklake` is Polars-only)
- `read_ducklake` accepts a `predicate=` callable for file-level pruning (partition + stats)
- `create_ducklake_table` takes DuckDB type strings (`"int64"`, `"varchar"`) instead of Polars types
- DML predicates are callables (`lambda df: df["col"] > 5`) instead of `pl.Expr`

All DDL operations, catalog inspection, tags, sort keys, views, and maintenance functions share the same signatures.

## PySpark usage

The `ducklake_pyspark` package provides PySpark integration with the same DuckLake catalogs:

```python
from pyspark.sql import SparkSession
from ducklake_pyspark import read_ducklake, write_ducklake, delete_ducklake

spark = SparkSession.builder.getOrCreate()

# Read — returns a PySpark DataFrame
df = read_ducklake(spark, "catalog.ducklake", "users")

# Write
write_ducklake(df, "catalog.ducklake", "users", mode="append")

# Delete (predicate is a SQL string)
deleted = delete_ducklake("catalog.ducklake", "users", "id > 10")

# CDC — read changes between snapshots
from ducklake_pyspark import read_ducklake_changes
changes = read_ducklake_changes(spark, "catalog.ducklake", "users", start_snapshot=1, end_snapshot=5)
```

Key differences from the Polars wrapper:
- `read_ducklake` takes `spark` (SparkSession) as the first argument
- `create_ducklake_table` takes a PySpark `StructType` instead of Polars types
- DML predicates are SQL strings (`"id > 10"`) instead of `pl.Expr`
- Includes `add_files_ducklake` for registering existing Parquet files
- Includes `rewrite_data_files_ducklake` for compaction

See the [PySpark wiki page](https://github.com/pdet/ducklake-dataframe/wiki/PySpark) for complete API documentation and examples.

## Known limitations

- **No UNION type** — DuckDB's UNION type is not mapped.
- **No MySQL backend** — only SQLite, PostgreSQL, and DuckDB.
- **No Parquet encryption** — encrypted DuckLake files cannot be read or written.
- **No automatic inline promotion** — inlined data is only flushed to Parquet on overwrite, not on threshold.
- **VARIANT binary interop** — VARIANT columns are schema-mapped as String, but the binary format used by DuckDB is not interoperable.
- **HUGEINT precision** — DuckDB writes HUGEINT as Float64 in Parquet, causing precision loss for large values.
- **MAP type** — Polars reads MAP columns as `List(Struct(key, value))` due to a Polars Parquet reader limitation.
- **INTERVAL type** — Polars cannot read DuckDB's `month_day_millisecond_interval` Parquet representation; reads of INTERVAL columns are unsupported.

See [docs/compatibility.md](docs/compatibility.md) for the full compatibility matrix versus DuckDB's native DuckLake extension, broken down per wrapper (Polars / Pandas / PySpark).

## Package structure

```
src/
├── ducklake_core/             Shared engine (Arrow-based internals)
│   ├── _backend.py            Backend adapters (SQLite, PostgreSQL)
│   ├── _catalog.py            Metadata reader (snapshots, tables, columns, files, stats)
│   ├── _catalog_api.py        DuckLakeCatalog inspection class (returns pa.Table)
│   ├── _schema.py             DuckDB type ↔ Arrow type mapping
│   └── _writer.py             Catalog writer (all DDL, DML, maintenance, OCC)
├── ducklake_polars/           Polars wrapper
│   ├── __init__.py            Public API (scan/read/write/DDL/DML/streaming)
│   ├── _catalog_api.py        DuckLakeCatalog returning Polars DataFrames
│   ├── _dataset.py            PythonDatasetProvider (lazy scan_parquet)
│   ├── _schema.py             DuckDB type → Polars type mapping
│   └── _stats.py              Column statistics for Polars file pruning
├── ducklake_pandas/           Pandas wrapper
│   ├── __init__.py            Public API (read/write/DDL/DML)
│   ├── _catalog_api.py        DuckLakeCatalog returning Pandas DataFrames
│   └── _writer.py             Thin wrapper over core writer
└── ducklake_pyspark/          PySpark wrapper
    ├── __init__.py            Public API (read/write/DDL/DML/CDC)
    ├── _ddl.py                DDL, maintenance, and catalog read operations
    └── _writer.py             PySpark ↔ Arrow conversion utilities
```

## Development

```bash
git clone https://github.com/pdet/ducklake-dataframe.git
cd ducklake-dataframe
pip install -e ".[dev]"
```

The `[dev]` extra pulls `duckdb` for fixture generation. DuckDB **≥ 1.5** unlocks the v10 catalog features used by some tests (macros, `merge_adjacent_files`, `cleanup_old_files`, custom column-tag keys, expression sort keys, expression defaults, schema-version-per-table, name-mapping reads). On DuckDB 1.4.x those tests are skipped via `_duckdb_supports_v10()`.

### Running tests

```bash
pytest                    # Full suite (SQLite backend)
pytest -n auto            # Parallel execution
pytest -k "test_views"    # Specific pattern

# With PostgreSQL backend
DUCKLAKE_PG_DSN="postgresql://user:pass@localhost/testdb" pytest
```

Test suite: **2,300+ tests** (24 xfailed for known DuckDB/Polars limitations, 4 xpassed). Tests cover all three engines (Polars, Pandas, PySpark) and are parametrized over backends — SQLite always runs; PostgreSQL runs when `DUCKLAKE_PG_DSN` is set. All wrappers are tested for interoperability with DuckDB's native extension.

### Benchmarks

Five benchmark suites for self-tracking regression detection and DuckLake vs [PyIceberg](https://py.iceberg.apache.org/) comparison:

| Benchmark | What it measures |
|---|---|
| [`bench_read_write.py`](benchmarks/bench_read_write.py) | Read/write/filter/agg across Polars, Arrow, and Pandas output formats |
| [`bench_streaming.py`](benchmarks/bench_streaming.py) | Streaming ingestion: many small appends, read-after-write, compaction |
| [`bench_schema_evolution.py`](benchmarks/bench_schema_evolution.py) | DDL cost: add/rename/drop columns, read after evolution, wide table projection |
| [`bench_dml.py`](benchmarks/bench_dml.py) | Delete, update, merge/upsert, delete cascade, read degradation vs compaction |
| [`bench_catalog.py`](benchmarks/bench_catalog.py) | Metadata ops: cold start, snapshot history, multi-table listing, partition pruning, time travel |

```bash
# Quick examples
python benchmarks/bench_read_write.py --rows 100000
python benchmarks/bench_streaming.py --batches 100 --batch-size 1000
python benchmarks/bench_schema_evolution.py --evolutions 50 --rows 100000
python benchmarks/bench_dml.py --rows 100000 --delete-rounds 20
python benchmarks/bench_catalog.py --snapshots 100 --rows 50000
```

All comparison benchmarks use the same data and workloads for both systems. See [`benchmarks/README.md`](benchmarks/README.md) for detailed scenario descriptions and interpretation guidance.

## License

MIT
