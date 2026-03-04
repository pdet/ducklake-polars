# ducklake-dataframe

Pure-Python [Polars](https://pola.rs/) and [Pandas](https://pandas.pydata.org/) integration for [DuckLake](https://ducklake.select/) catalogs — read and write, no DuckDB runtime required.

ducklake-dataframe reads and writes DuckLake metadata directly from SQLite, PostgreSQL, or DuckDB catalog files and scans the underlying Parquet data files through Polars' native Parquet reader or PyArrow. With Polars you get lazy evaluation, predicate pushdown, projection pushdown, file pruning, and all other Polars optimizations out of the box. With Pandas you get familiar DataFrame ergonomics with partition and statistics-based file pruning.

> **Note:** This project is a proof of concept, 100% written by [Claude Code](https://docs.anthropic.com/en/docs/build-with-claude/claude-code/overview). It is not intended for production use.

## Architecture

```
┌─────────────────┐  ┌─────────────────┐
│  ducklake_polars │  │  ducklake_pandas │   ← Thin wrappers (API + reader)
└────────┬────────┘  └────────┬────────┘
         │                    │
         └────────┬───────────┘
                  │
         ┌────────▼────────┐
         │   ducklake_core  │   ← Shared engine (catalog, writer, schema, backend)
         └─────────────────┘
```

- **`ducklake_core`** — All catalog I/O, write operations, schema mapping, and backend adapters. Uses PyArrow as the internal data representation.
- **`ducklake_polars`** — Polars-specific reader (lazy `scan_parquet` via `PythonDatasetProvider`), plus a thin API that converts between Polars types and Arrow.
- **`ducklake_pandas`** — Pandas-specific reader (eager via PyArrow → Pandas conversion), plus a thin API that converts between Pandas types and Arrow.

## Installation

```bash
# Polars engine
pip install ducklake-dataframe[polars]

# Pandas engine
pip install ducklake-dataframe[pandas]

# Both engines
pip install ducklake-dataframe[polars,pandas]

# With PostgreSQL catalog backend
pip install ducklake-dataframe[polars,postgres]

# With S3 object storage
pip install ducklake-dataframe[polars,s3]

# Everything
pip install ducklake-dataframe[all]
```

**Core dependency:** `pyarrow >= 10.0` only. Everything else is optional:

| Extra | What it adds |
|---|---|
| `polars` | `polars >= 1.0` — Polars engine with lazy evaluation |
| `pandas` | `pandas >= 1.5` — Pandas engine |
| `postgres` | `psycopg2` — PostgreSQL catalog backend |
| `duckdb` | `duckdb` — DuckDB catalog backend |
| `s3` | `s3fs` — S3 object storage |
| `gcs` | `gcsfs` — Google Cloud Storage |
| `azure` | `adlfs` — Azure Blob Storage |

## Quick Start

### Create a catalog and write data

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

### Query with lazy evaluation

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

### Delete, update, merge

```python
from ducklake_polars import delete_ducklake, update_ducklake, merge_ducklake

# Delete rows matching a predicate
deleted = delete_ducklake("catalog.ducklake", "users", pl.col("id") == 2)

# Update rows
updated = update_ducklake(
    "catalog.ducklake", "users",
    updates={"region": "APAC"},
    predicate=pl.col("name") == "Eve",
)

# Merge (upsert) — atomic delete + insert in one snapshot
source = pl.DataFrame({"id": [1, 6], "name": ["Alice2", "Frank"], "region": ["US", "EU"]})
rows_updated, rows_inserted = merge_ducklake(
    "catalog.ducklake", "users", source, on="id",
    when_matched_update=True,
    when_not_matched_insert=True,
)
```

### Time travel

```python
from ducklake_polars import read_ducklake

# Read at a specific snapshot version
df_v1 = read_ducklake("catalog.ducklake", "users", snapshot_version=1)

# Read at a specific timestamp
df_ts = read_ducklake("catalog.ducklake", "users", snapshot_time="2025-06-15T10:30:00")
```

### Inspect the catalog

```python
from ducklake_polars import DuckLakeCatalog

catalog = DuckLakeCatalog("catalog.ducklake")

catalog.snapshots()           # All snapshots
catalog.current_snapshot()    # Latest snapshot ID
catalog.table_info()          # Per-table file/size stats
catalog.list_files("users")   # Data + delete files
catalog.list_schemas()        # All schemas
catalog.list_tables()         # Tables in default schema
```

### DuckDB interoperability

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

## Features

### Read path

- **Lazy and eager reads** — `scan_ducklake()` (Polars LazyFrame) / `read_ducklake()` (eager)
- **Predicate and projection pushdown** — through Polars' native optimizer; Pandas supports `columns=` and `predicate=`
- **File pruning** — column-level min/max statistics and partition values
- **Time travel** — by snapshot version or timestamp
- **Delete file handling** — Iceberg-compatible positional deletes (cumulative delete files supported)
- **Schema evolution** — ADD/DROP/RENAME COLUMN handled transparently across file versions
- **Inlined data** — small tables stored directly in catalog metadata, read transparently

### Write path

- **INSERT** — append, overwrite, or error-on-exists modes
- **DELETE** — predicate-based row deletion with Iceberg position-delete files
- **UPDATE** — atomic delete + insert in a single snapshot
- **MERGE** — upsert with configurable matched/unmatched behavior
- **CREATE TABLE AS** — single-snapshot table creation with data
- **Data inlining** — small inserts stored in catalog metadata (configurable threshold)
- **Partitioned writes** — Hive-style directory layout per partition key
- **Sort keys** — data sorted before writing for better row group statistics
- **Author/commit metadata** — `author=` and `commit_message=` on all write operations

### DDL

- **CREATE/DROP TABLE** with full snapshot versioning
- **ADD/DROP/RENAME COLUMN** with schema evolution tracking
- **SET TYPE** — column type changes tracked in schema history
- **CREATE/DROP SCHEMA** with cascade support
- **RENAME TABLE** preserving table identity
- **SET/RESET PARTITIONED BY** — identity-transform partitioning
- **SET/RESET SORTED BY** — with `ASC`/`DESC` and `NULLS_FIRST`/`NULLS_LAST`
- **CREATE/DROP VIEW** with `OR REPLACE` support
- **Tags** — key-value metadata on tables and columns

### Catalog backends

- **SQLite** — Python stdlib `sqlite3` (zero dependency)
- **DuckDB** — `.ducklake` files are SQLite-format, read via `sqlite3`
- **PostgreSQL** — via `psycopg2` (optional `[postgres]` extra)

Both DuckLake catalog format **v0.3** and **v0.4** are supported.
