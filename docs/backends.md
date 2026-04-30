# Catalog Backends

ducklake-dataframe supports three catalog backends for storing DuckLake metadata. The backend is determined by the `path` parameter passed to any API function.

## SQLite (default)

SQLite is the default backend — it uses Python's built-in `sqlite3` module, so there are zero extra dependencies.

```python
from ducklake_polars import write_ducklake, scan_ducklake

# Any file path ending in .ducklake or .db uses SQLite
write_ducklake(df, "catalog.ducklake", "users", mode="error")
lf = scan_ducklake("catalog.ducklake", "users")
```

### File format

DuckLake catalog files (`.ducklake`) are standard SQLite databases. You can inspect them with any SQLite tool:

```bash
sqlite3 catalog.ducklake ".tables"
# ducklake_column_stats      ducklake_data_file         ducklake_metadata
# ducklake_column_tag        ducklake_delete_file       ducklake_partition
# ...
```

### When to use SQLite

- **Single-user workflows** — local analysis, notebooks, scripts
- **Zero-dependency setups** — no external database to manage
- **Development and testing** — catalogs are just files, easy to copy/delete
- **DuckDB interop** — DuckDB's native DuckLake extension reads the same SQLite files

### WAL mode

ducklake-dataframe automatically flips SQLite catalogs to **WAL (Write-Ahead Log)** mode on the first write. This persistent setting allows readers to proceed concurrently with one writer — without it, a reader colliding with an in-flight write transaction can surface as a "disk I/O error" on macOS. The mode is captured in the file header, so all subsequent opens (including from DuckDB's native extension) inherit it. WAL creates `.ducklake-wal` and `.ducklake-shm` sidecar files next to the catalog; they are part of the database and should be kept together.

## PostgreSQL

PostgreSQL provides a shared, network-accessible catalog backend suitable for team environments.

### Installation

```bash
pip install ducklake-dataframe[polars,postgres]
# or
pip install ducklake-dataframe[pandas,postgres]
```

This installs `psycopg2` as the PostgreSQL driver.

### Usage

Pass a PostgreSQL connection string as the `path` parameter:

```python
from ducklake_polars import write_ducklake, scan_ducklake

PG_DSN = "postgresql://user:password@localhost:5432/mydb"

write_ducklake(df, PG_DSN, "users", mode="error")
lf = scan_ducklake(PG_DSN, "users")
```

The connection string format follows the standard PostgreSQL URI syntax:

```
postgresql://[user[:password]@][host][:port][/dbname][?param=value]
```

### Schema setup

DuckLake tables are automatically created in the PostgreSQL database when you first write data. The DuckLake metadata tables (`ducklake_snapshot`, `ducklake_table`, etc.) live alongside your regular PostgreSQL tables.

### When to use PostgreSQL

- **Multi-user environments** — shared catalogs across a team
- **Concurrent access** — PostgreSQL handles locking and transactions
- **Production pipelines** — durable, backed up, monitored
- **Existing infrastructure** — if you already run PostgreSQL

### Running tests with PostgreSQL

```bash
DUCKLAKE_PG_DSN="postgresql://user:pass@localhost/testdb" pytest
```

## DuckDB

DuckDB `.ducklake` files are SQLite-format, so they are read via the SQLite backend. There is no separate DuckDB backend — the same `sqlite3` module reads catalogs created by DuckDB's native DuckLake extension.

### Installation

```bash
pip install ducklake-dataframe[polars,duckdb]
```

The `duckdb` extra is only needed if you want to use DuckDB directly (e.g., for creating catalogs with `ATTACH 'ducklake:...'`). ducklake-dataframe itself reads DuckDB-created catalogs via `sqlite3`.

### Interoperability

Catalogs are fully interoperable between DuckDB and ducklake-dataframe:

```python
import duckdb

# Create with DuckDB
con = duckdb.connect()
con.execute("INSTALL ducklake; LOAD ducklake")
con.execute("ATTACH 'ducklake:sqlite:catalog.ducklake' AS lake (DATA_PATH 'data/')")
con.execute("CREATE TABLE lake.events (ts TIMESTAMP, value DOUBLE)")
con.execute("INSERT INTO lake.events VALUES ('2025-01-01', 42.0)")
con.close()

# Read with ducklake-dataframe
from ducklake_polars import scan_ducklake
lf = scan_ducklake("catalog.ducklake", "events")
print(lf.collect())
```

And vice versa — create with ducklake-dataframe, query with DuckDB SQL.

### Catalog format versions

DuckLake catalog format **v1.0** is the default for new catalogs. Older **v0.3** / **v0.4** catalogs (e.g., those created by older DuckDB-ducklake versions) are read-compatible, but **migration is opt-in** — call `migrate_catalog(path)` (re-exported from `ducklake_polars`, `ducklake_pandas`, and `ducklake_pyspark`) to bring a v0.3 or v0.4 catalog up to v1.0 in place. The function is idempotent. v1.0-only writer features (`merge_adjacent_files`, macros, expression sort keys, expression defaults, custom column tag keys) raise an explicit version error against pre-1.0 catalogs.

## Backend comparison

| Feature | SQLite | PostgreSQL | DuckDB |
|---|---|---|---|
| Dependencies | None (stdlib) | `psycopg2` | `sqlite3` (stdlib) |
| Shared access | Single writer + concurrent readers (WAL) | Multi-user, MVCC | Single process |
| Network access | Local file only | TCP/IP | Local file only |
| Setup | Zero config | Database server | Zero config |
| DuckDB interop | ✅ | ✅ | ✅ |
| Catalog format | v0.3, v0.4, v1.0 | v0.3, v0.4, v1.0 | v0.3, v0.4, v1.0 |

## Data path override

All backends support the `data_path` parameter to override where Parquet data files are stored or read from:

```python
# Catalog metadata in SQLite, data files in a different directory
write_ducklake(df, "catalog.ducklake", "users", mode="error", data_path="/data/warehouse/")

# Read with explicit data path
lf = scan_ducklake("catalog.ducklake", "users", data_path="/data/warehouse/")
```

This is useful when:

- The catalog has been moved to a different location
- Data files are on a different filesystem or mount point
- You want to separate metadata from data storage
