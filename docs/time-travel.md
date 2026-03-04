# Time Travel

DuckLake tracks every change to your tables as immutable snapshots. ducklake-dataframe lets you read any historical version of your data — by snapshot number or by timestamp.

## How snapshots work

Every write operation (insert, delete, update, merge, DDL) creates a new snapshot in the catalog. Each snapshot records:

- **`snapshot_id`** — auto-incrementing integer
- **`snapshot_time`** — timestamp when the snapshot was created
- **`schema_version`** — tracks DDL changes (column adds, renames, type changes)

Snapshots are immutable — old data is never modified, only superseded by new snapshots.

## Reading historical data

### By snapshot version

```python
from ducklake_polars import read_ducklake, scan_ducklake

# Eager read at snapshot version 1
df = read_ducklake("catalog.ducklake", "users", snapshot_version=1)

# Lazy scan at snapshot version 3
lf = scan_ducklake("catalog.ducklake", "users", snapshot_version=3)
result = lf.filter(pl.col("region") == "US").collect()
```

### By timestamp

```python
from datetime import datetime

# ISO format string
df = read_ducklake("catalog.ducklake", "users", snapshot_time="2025-06-15T10:30:00")

# datetime object
ts = datetime(2025, 6, 15, 10, 30, 0)
df = read_ducklake("catalog.ducklake", "users", snapshot_time=ts)
```

When using `snapshot_time`, the catalog returns the most recent snapshot at or before the given timestamp.

### Constraints

You cannot specify both `snapshot_version` and `snapshot_time` — doing so raises a `ValueError`:

```python
# This raises ValueError
df = read_ducklake(
    "catalog.ducklake", "users",
    snapshot_version=1,
    snapshot_time="2025-06-15T10:30:00",
)
```

## Listing snapshots

Use `DuckLakeCatalog` to inspect available snapshots:

```python
from ducklake_polars import DuckLakeCatalog

catalog = DuckLakeCatalog("catalog.ducklake")

# All snapshots
snapshots = catalog.snapshots()
print(snapshots)
# shape: (5, 3)
# ┌─────────────┬──────────────────────┬────────────────┐
# │ snapshot_id ┆ snapshot_time        ┆ schema_version │
# │ ---         ┆ ---                  ┆ ---            │
# │ i64         ┆ str                  ┆ i64            │
# ╞═════════════╪══════════════════════╪════════════════╡
# │ 1           ┆ 2025-06-15T08:00:00  ┆ 1              │
# │ 2           ┆ 2025-06-15T09:15:00  ┆ 1              │
# │ 3           ┆ 2025-06-15T10:30:00  ┆ 2              │
# │ 4           ┆ 2025-06-15T11:00:00  ┆ 2              │
# │ 5           ┆ 2025-06-15T12:00:00  ┆ 2              │
# └─────────────┴──────────────────────┴────────────────┘

# Latest snapshot ID
current = catalog.current_snapshot()
print(current)  # 5
```

## Change data feed

Track what changed between snapshots:

```python
catalog = DuckLakeCatalog("catalog.ducklake")

# Rows inserted between snapshot 1 and 3
insertions = catalog.table_insertions("users", start_version=1, end_version=3)

# Rows deleted between snapshot 1 and 3
deletions = catalog.table_deletions("users", start_version=1, end_version=3)

# Full change data feed (inserts, deletes, updates as pre/post images)
changes = catalog.table_changes("users", start_version=1, end_version=3)
print(changes)
# ┌─────────────┬────────────────────┬─────┬───────┬────────┐
# │ snapshot_id ┆ change_type        ┆ id  ┆ name  ┆ region │
# │ ---         ┆ ---                ┆ --- ┆ ---   ┆ ---    │
# │ i64         ┆ str                ┆ i64 ┆ str   ┆ str    │
# ╞═════════════╪════════════════════╪═════╪═══════╪════════╡
# │ 2           ┆ insert             ┆ 4   ┆ Dave  ┆ EU     │
# │ 3           ┆ update_preimage    ┆ 1   ┆ Alice ┆ US     │
# │ 3           ┆ update_postimage   ┆ 1   ┆ Alice ┆ APAC   │
# │ 3           ┆ delete             ┆ 2   ┆ Bob   ┆ EU     │
# └─────────────┴────────────────────┴─────┴───────┴────────┘
```

The `change_type` column values:

| Value | Meaning |
|---|---|
| `insert` | New row added |
| `delete` | Row removed |
| `update_preimage` | Row before an update (old values) |
| `update_postimage` | Row after an update (new values) |

Updates are detected when both an insertion and a deletion occur in the same snapshot for the same table.

## Schema evolution across time

Time travel handles schema evolution transparently:

- **Added columns** — queries at snapshots before the column was added return `NULL` for that column
- **Dropped columns** — queries at snapshots before the column was dropped still return the column's data
- **Renamed columns** — column history is tracked, so old Parquet files with the old column name are reconciled automatically
- **Type changes** — values from files written before a type change are cast to the new type on the fly

```python
# Column "email" was added at snapshot 3
df_v2 = read_ducklake("catalog.ducklake", "users", snapshot_version=2)
# "email" column is not present (it didn't exist yet)

df_v4 = read_ducklake("catalog.ducklake", "users", snapshot_version=4)
# "email" column is present; rows from before snapshot 3 have NULL
```

## Catalog inspection at specific versions

Several `DuckLakeCatalog` methods accept a `snapshot_version` parameter:

```python
catalog = DuckLakeCatalog("catalog.ducklake")

# List files at a specific snapshot
files_v2 = catalog.list_files("users", snapshot_version=2)

# List schemas at a specific snapshot
schemas_v1 = catalog.list_schemas(snapshot_version=1)

# List tables at a specific snapshot
tables_v1 = catalog.list_tables(snapshot_version=1)
```

## Expiring snapshots

Over time, old snapshots accumulate. Use `expire_snapshots` to clean up metadata and `vacuum_ducklake` to remove orphaned data files:

```python
from ducklake_polars import expire_snapshots, vacuum_ducklake

# Keep only the last 10 snapshots
expired = expire_snapshots("catalog.ducklake", keep_last_n=10)
print(f"Expired {expired} snapshots")

# Or expire by snapshot ID
expired = expire_snapshots("catalog.ducklake", older_than_snapshot=5)

# Then clean up orphaned Parquet files
deleted = vacuum_ducklake("catalog.ducklake")
print(f"Deleted {deleted} orphaned files")
```

!!! warning
    After expiring snapshots, time travel to those versions is no longer possible. Only expire snapshots you're sure you don't need.

## Pandas

All time travel features work identically with the Pandas API:

```python
from ducklake_pandas import read_ducklake, DuckLakeCatalog

df = read_ducklake("catalog.ducklake", "users", snapshot_version=1)
catalog = DuckLakeCatalog("catalog.ducklake")
changes = catalog.table_changes("users", start_version=1, end_version=5)
```
