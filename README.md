# ducklake-polars

A pure-Python [Polars](https://pola.rs/) integration for [DuckLake](https://ducklake.select/) catalogs.

Reads DuckLake metadata directly from SQLite or PostgreSQL (no DuckDB runtime dependency) and scans the underlying Parquet data files through Polars' native Parquet reader. This gives you lazy evaluation, predicate pushdown, projection pushdown, and all other Polars optimizations out of the box.

> **Note:** This project was developed almost entirely by [Claude Code](https://docs.anthropic.com/en/docs/build-with-claude/claude-code/overview) (Anthropic's AI coding agent) with minor human supervision.

## Installation

```bash
pip install ducklake-polars

# With PostgreSQL support
pip install ducklake-polars[postgres]
```

The only runtime dependency is `polars >= 1.0`. SQLite catalogs are read via Python's built-in `sqlite3` module. For PostgreSQL catalogs, install the `postgres` extra which adds `psycopg2`.

## Quick start

```python
from ducklake_polars import scan_ducklake, read_ducklake

# Eager read into a DataFrame
df = read_ducklake("catalog.ducklake", "my_table")

# Lazy scan into a LazyFrame (recommended for large tables)
lf = scan_ducklake("catalog.ducklake", "my_table")
result = lf.filter(pl.col("x") > 100).select("x", "y").collect()

# Select specific columns
df = read_ducklake("catalog.ducklake", "my_table", columns=["x", "y"])

# Time travel to a specific snapshot version
df = read_ducklake("catalog.ducklake", "my_table", snapshot_version=3)

# Time travel to a specific timestamp
df = read_ducklake("catalog.ducklake", "my_table", snapshot_time="2025-01-15T10:30:00")

# Read from a non-default schema
df = read_ducklake("catalog.ducklake", "my_table", schema="analytics")

# Override the data path (useful when catalog has been moved)
df = read_ducklake("catalog.ducklake", "my_table", data_path="/new/data/location")

# Read from a PostgreSQL-backed catalog
df = read_ducklake("postgresql://user:pass@localhost/mydb", "my_table")
```

## Features

- **Lazy and eager reads** via `scan_ducklake()` (LazyFrame) and `read_ducklake()` (DataFrame)
- **Predicate and projection pushdown** through Polars' native scan optimizer
- **Time travel** by snapshot version or timestamp
- **Delete file handling** using Polars' Iceberg-compatible positional delete support
- **Schema evolution** (add column, drop column) via Polars' `missing_columns` / `extra_columns` options
- **Inlined data** support for small tables stored directly in the catalog
- **File pruning** via column-level min/max statistics
- **SQLite and PostgreSQL backends** -- reads metadata from either backend transparently
- **No DuckDB runtime dependency** -- metadata is read directly from SQLite or PostgreSQL

## Creating a DuckLake catalog

DuckLake catalogs are created using DuckDB with the DuckLake extension. Here's an example using a SQLite metadata backend:

```python
import duckdb

con = duckdb.connect()
con.install_extension("ducklake")
con.load_extension("ducklake")
con.install_extension("sqlite_scanner")
con.load_extension("sqlite_scanner")

con.execute("""
    ATTACH 'ducklake:sqlite:my_catalog.ducklake' AS lake
        (DATA_PATH '/path/to/data/files')
""")

con.execute("CREATE TABLE lake.my_table (id INTEGER, name VARCHAR, score DOUBLE)")
con.execute("INSERT INTO lake.my_table VALUES (1, 'Alice', 95.5), (2, 'Bob', 87.3)")
con.close()
```

Then read it with ducklake-polars:

```python
from ducklake_polars import read_ducklake

df = read_ducklake("my_catalog.ducklake", "my_table")
print(df)
# shape: (2, 3)
# ┌─────┬───────┬───────┐
# │ id  ┆ name  ┆ score │
# │ --- ┆ ---   ┆ ---   │
# │ i32 ┆ str   ┆ f64   │
# ╞═════╪═══════╪═══════╡
# │ 1   ┆ Alice ┆ 95.5  │
# │ 2   ┆ Bob   ┆ 87.3  │
# └─────┴───────┴───────┘
```

## Supported data types

| DuckLake type | Polars type | Notes |
|---|---|---|
| `int8` / TINYINT | `Int8` | |
| `int16` / SMALLINT | `Int16` | |
| `int32` / INTEGER | `Int32` | |
| `int64` / BIGINT | `Int64` | |
| `uint8` / UTINYINT | `UInt8` | |
| `uint16` / USMALLINT | `UInt16` | |
| `uint32` / UINTEGER | `UInt32` | |
| `uint64` / UBIGINT | `UInt64` | |
| `float32` / FLOAT | `Float32` | |
| `float64` / DOUBLE | `Float64` | |
| `boolean` | `Boolean` | |
| `varchar` | `String` | |
| `blob` | `Binary` | |
| `date` | `Date` | |
| `time`, `time_ns`, `timetz` | `Time` | |
| `timestamp`, `timestamp_us` | `Datetime("us")` | |
| `timestamp_ms` | `Datetime("ms")` | |
| `timestamp_ns` | `Datetime("ns")` | |
| `timestamp_s` | `Datetime("us")` | Upcast to microseconds |
| `timestamptz` | `Datetime("us", "UTC")` | |
| `decimal(p,s)` | `Decimal(p,s)` | |
| `uuid` | `Binary` | 16-byte binary representation |
| `json` | `Binary` | Cast to `String` for text access |
| `interval` | `Duration("us")` | Limited by Polars Parquet reader |
| `int128` / HUGEINT | `Int128` | Limited by Parquet interop |
| `uint128` / UHUGEINT | `UInt128` | Limited by Parquet interop |
| `list` | `List(...)` | |
| `struct` | `Struct(...)` | |
| `map` | `List(Struct(key, value))` | Limited by Polars Parquet reader |
| `geometry` | `Binary` | |
| `variant` | `String` | |

## Architecture

```
ducklake-polars
├── __init__.py       # Public API: scan_ducklake(), read_ducklake()
├── _backend.py       # Backend adapters (SQLite, PostgreSQL)
├── _catalog.py       # Metadata reader (snapshots, tables, columns, files)
├── _dataset.py       # Polars PythonDatasetProvider implementation
├── _schema.py        # DuckLake type string -> Polars type mapping
└── _stats.py         # Column statistics for file pruning
```

The library reads DuckLake metadata tables (`ducklake_snapshot`, `ducklake_table`, `ducklake_column`, `ducklake_data_file`, etc.) from the catalog database (SQLite or PostgreSQL) through a thin backend adapter layer. It then constructs a Polars `scan_parquet()` call with the resolved file paths, deletion files, and statistics, letting Polars handle all the actual data reading and query optimization.

## Building from source

```bash
git clone https://github.com/your-org/ducklake-polars.git
cd ducklake-polars
pip install -e ".[dev]"
```

This installs the package in editable mode along with dev dependencies (`pytest`, `pytest-xdist`, and `duckdb` for running tests).

## Running tests

```bash
pytest
```

Tests use DuckDB with the DuckLake extension to create catalogs in temporary directories, then read them back with ducklake-polars to verify correctness. DuckDB is only a test dependency -- it is not required at runtime.

By default tests run against SQLite. To also run against PostgreSQL, set the `DUCKLAKE_PG_DSN` environment variable:

```bash
DUCKLAKE_PG_DSN="postgresql://user:pass@localhost/testdb" pytest
```

To run tests in parallel:

```bash
pytest -n auto
```

To run a specific test file:

```bash
pytest tests/test_types.py -v
```

### Test coverage

The test suite covers:

- **Basic operations** -- scan, read, column selection, empty tables, multiple inserts, CTAS, filters, nulls, large datasets, multiple tables
- **Data types** -- all integer widths (signed/unsigned), floats, boolean, string, binary, date, time, timestamps (all precisions), decimal, UUID, JSON, lists, structs, nested types
- **Time travel** -- reading at specific snapshot versions, lazy time travel, invalid version handling
- **Deletes** -- single delete, partial delete, multiple deletes, delete-all, time travel before delete
- **Schema evolution** -- add column, add multiple columns, drop column
- **Type mapping** -- unit tests for all DuckLake type strings (both SQL standard names and internal lowercase names)

## License

MIT
