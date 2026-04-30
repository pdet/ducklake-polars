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

SQLite catalogs are auto-flipped to **WAL (Write-Ahead Log) mode** on the first write. WAL allows concurrent readers to proceed alongside one writer without colliding (a regular rollback-journal database surfaces "disk I/O error" when a reader and writer race). The mode is persisted in the file header — once flipped, all future opens inherit it.

Single-writer concurrency model: one writer at a time, many concurrent readers. Concurrent writes from different processes are serialised by SQLite's filesystem locking — the second writer waits for the timeout (30s by default) or gets a "database is locked" error. The OCC retry layer above this handles that gracefully.

For single-writer-per-process workloads (the most common case), SQLite works without issues.

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

## Optimistic concurrency control

ducklake-dataframe implements OCC at the snapshot-commit boundary. Each writer:

1. Reads the latest snapshot (`max(snapshot_id)`).
2. Stages new files / DDL changes locally.
3. On commit, attempts to insert the next snapshot row inside an exclusive (`BEGIN IMMEDIATE` on SQLite, default isolation on PostgreSQL) transaction.
4. If a concurrent writer already claimed that snapshot id, validates the staged work against the new state and retries; otherwise raises `TransactionConflictError`.

Conflicts are detected at three granularities:

- **Table-level** — concurrent DDL on the same table (drop, rename, schema changes)
- **File-level** — concurrent writes touching the same data files (e.g., delete vs compact)
- **Partition-level** — concurrent writes to the same identity-transform partition values

The retry layer is the public-facing API (`max_retries=`, `retry_wait_ms=`, `retry_backoff=`). Inside the writer, snapshot-id INSERT races have their own bounded retry (`max_snapshot_retries=5`, `snapshot_retry_wait_ms=50`) to handle the narrow window where two writers picked the same `snapshot_id` simultaneously.

## Known limitations

- **No merge conflict resolution** — if two writers modify the same rows, the last writer wins (after retry). There is no automatic three-way merge.
- **SQLite single-writer** — SQLite serialises writes via filesystem locking. Reads are concurrent (WAL), but only one writer makes progress at a time. For multi-writer pipelines, use PostgreSQL.
- **No distributed locking** — there is no external lock manager. Coordination relies entirely on the catalog backend's native transaction support.

For production workloads requiring concurrent writes, PostgreSQL is strongly recommended as the catalog backend.
