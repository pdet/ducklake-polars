# Concurrency

ducklake-dataframe provides transaction retry and conflict detection for concurrent write scenarios. This page covers how writes are serialized, how conflicts are detected, and how to configure retry behavior.

## Write model

All write operations in ducklake-dataframe are **snapshot-based**:

1. Read the current snapshot (latest catalog state)
2. Perform the operation (compute new/deleted rows, DDL changes)
3. Write new Parquet files (if applicable)
4. Commit a new snapshot to the catalog

Each snapshot is an atomic unit — either the entire operation succeeds or nothing changes. The catalog backend (SQLite or PostgreSQL) provides the serialization guarantees.

## TransactionConflictError

When two writers try to commit simultaneously and a conflict is detected, a `TransactionConflictError` is raised:

```python
from ducklake_polars import TransactionConflictError, write_ducklake
# or
from ducklake_pandas import TransactionConflictError

try:
    write_ducklake(df, "catalog.ducklake", "users", mode="append")
except TransactionConflictError as e:
    print(f"Conflict detected: {e}")
    # Retry, queue, or handle
```

This exception is importable from both `ducklake_polars` and `ducklake_pandas`.

## Automatic retries

Write operations that support retries will automatically retry on conflict with exponential backoff. The following parameters control retry behavior:

| Parameter | Default | Description |
|---|---|---|
| `max_retries` | `3` | Maximum number of retry attempts |
| `retry_wait_ms` | `100` | Initial wait time between retries (milliseconds) |
| `retry_backoff` | `2.0` | Multiplier for wait time on each subsequent retry |

### Functions with retry support

- `write_ducklake()` — insert, append, overwrite
- `add_files_ducklake()` — register external Parquet files
- `rewrite_data_files_ducklake()` — compaction

### Example

```python
write_ducklake(
    df,
    "catalog.ducklake",
    "users",
    mode="append",
    max_retries=5,          # Try up to 5 times
    retry_wait_ms=200,      # Start with 200ms wait
    retry_backoff=2.0,      # Double the wait each retry: 200ms, 400ms, 800ms, 1600ms
)
```

With these settings, the retry sequence would be:
1. First attempt
2. Wait 200ms → retry
3. Wait 400ms → retry
4. Wait 800ms → retry
5. Wait 1600ms → retry
6. Raise `TransactionConflictError` if still failing

## Backend-specific behavior

### SQLite

SQLite uses file-level locking. Concurrent writes from the same process are serialized by Python's GIL and SQLite's internal locking. Concurrent writes from different processes use SQLite's filesystem locking — one writer proceeds while others wait or get a "database is locked" error.

For single-writer scenarios (the most common case), SQLite works without issues.

### PostgreSQL

PostgreSQL provides row-level locking and MVCC (Multi-Version Concurrency Control). Concurrent writes from multiple clients are handled by PostgreSQL's transaction isolation:

- Snapshot reads are consistent within a transaction
- Conflicting writes are detected at commit time
- The retry mechanism re-reads the latest snapshot before each attempt

PostgreSQL is the recommended backend for multi-user or multi-process write scenarios.

## Patterns

### Single writer (recommended for SQLite)

The simplest and most reliable pattern — one process writes at a time:

```python
# Worker process
write_ducklake(df, "catalog.ducklake", "users", mode="append")
```

### Multiple writers with retries (PostgreSQL)

For concurrent pipelines, use PostgreSQL with retries:

```python
PG_DSN = "postgresql://user:pass@host/db"

# Worker 1
write_ducklake(df1, PG_DSN, "users", mode="append", max_retries=5)

# Worker 2 (concurrent)
write_ducklake(df2, PG_DSN, "users", mode="append", max_retries=5)
```

### Explicit conflict handling

Catch conflicts and implement custom logic:

```python
from ducklake_polars import TransactionConflictError, write_ducklake

def write_with_fallback(df, path, table):
    try:
        write_ducklake(df, path, table, mode="append", max_retries=3)
    except TransactionConflictError:
        # Log, queue for later, alert, etc.
        print(f"Failed to write to {table} after retries")
        raise
```

## Author and commit metadata

All write operations support `author` and `commit_message` parameters for auditing concurrent modifications:

```python
write_ducklake(
    df,
    "catalog.ducklake",
    "users",
    mode="append",
    author="pipeline-worker-1",
    commit_message="Daily ETL load for 2025-06-15",
)
```

These are stored in the snapshot metadata and can be inspected via the catalog API.

## Known limitations

- **No optimistic concurrency control** — ducklake-dataframe does not implement full OCC with conflict resolution. The retry mechanism re-reads and re-applies the entire operation.
- **No merge conflict resolution** — if two writers modify the same rows, the last writer wins (after retry). There is no automatic three-way merge.
- **SQLite concurrent writes** — SQLite's file-level locking means only one writer can proceed at a time. For true concurrent writes, use PostgreSQL.
- **No distributed locking** — there is no external lock manager. Coordination relies entirely on the catalog backend's native transaction support.

For production workloads requiring concurrent writes, PostgreSQL is strongly recommended as the catalog backend.
