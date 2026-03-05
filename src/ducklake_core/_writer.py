"""DuckLake catalog writer — creates tables and inserts data (Arrow internals)."""

from __future__ import annotations

import functools
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Callable


class TransactionConflictError(Exception):
    """Raised when a write conflicts with a concurrent transaction.

    DuckLake uses optimistic concurrency control: transactions record
    their starting snapshot and check for conflicting changes before
    committing.  This exception is raised when a conflict is detected,
    triggering automatic retry with exponential backoff.
    """

import pyarrow as pa
import pyarrow.compute as pc
import ducklake_core._storage as storage
from ducklake_core._backend import PostgreSQLBackend, SQLiteBackend, create_backend
from ducklake_core._exceptions import SchemaNotFoundError, TableNotFoundError
from ducklake_core._schema import arrow_type_to_duckdb, duckdb_type_to_arrow


def _uuid7() -> str:
    """Generate a UUID version 7 (time-ordered, RFC 9562)."""
    ts_ms = int(time.time() * 1000)
    rand = os.urandom(10)

    # High 64 bits: 48-bit timestamp | 4-bit version (7) | 12-bit random
    high = (ts_ms << 16) | (7 << 12) | (int.from_bytes(rand[0:2], "big") & 0x0FFF)
    # Low 64 bits: 2-bit variant (10) | 62-bit random
    low = ((0x80 | (rand[2] & 0x3F)) << 56) | (
        int.from_bytes(rand[3:10], "big") & 0x00FFFFFFFFFFFFFF
    )

    uuid_int = (high << 64) | low
    h = f"{uuid_int:032x}"
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def _decode_dictionary_columns(table: pa.Table) -> pa.Table:
    """Cast any dictionary-encoded columns to their value types.

    Polars ``Categorical`` / ``Enum`` columns arrive as dictionary-encoded
    Arrow arrays. DuckLake stores these as plain ``varchar``, so we decode
    them before computing statistics or writing Parquet to avoid type
    mismatches on read.
    """
    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            col = table.column(i)
            decoded = col.cast(field.type.value_type)
            table = table.set_column(i, field.with_type(decoded.type), decoded)
    return table


def _read_parquet_footer_size(path: str) -> int:
    """Read the Parquet footer size from a file."""
    try:
        return storage.read_parquet_footer_size(path)
    except Exception:
        return 0


class _PlaceholderConnection:
    """
    Wraps a database connection to translate ``?`` placeholders to ``%s``.

    This allows the writer to use ``?`` consistently everywhere while
    supporting both SQLite (``?``) and PostgreSQL (``%s``) backends.
    """

    __slots__ = ("_con", "_placeholder")

    def __init__(self, con: Any, placeholder: str) -> None:
        self._con = con
        self._placeholder = placeholder

    def execute(self, sql: str, params: Any = None) -> Any:
        if self._placeholder != "?":
            sql = sql.replace("?", self._placeholder)
        if params is not None:
            return self._con.execute(sql, params)
        return self._con.execute(sql)

    def commit(self) -> None:
        if hasattr(self._con, "isolation_level") and self._con.isolation_level is None:
            # Manual transaction mode (SQLite) — explicit COMMIT,
            # then start a new deferred transaction (not IMMEDIATE)
            # so we don't hold the write lock between operations.
            self._con.execute("COMMIT")
            self._con.execute("BEGIN")
        elif hasattr(self._con, "commit"):
            self._con.commit()

    def begin_immediate(self) -> None:
        """Upgrade to an IMMEDIATE transaction (SQLite only)."""
        if hasattr(self._con, "isolation_level") and self._con.isolation_level is None:
            try:
                self._con.execute("COMMIT")
            except Exception:
                pass
            self._con.execute("BEGIN IMMEDIATE")

    def rollback(self) -> None:
        if hasattr(self._con, "isolation_level") and self._con.isolation_level is None:
            try:
                self._con.execute("ROLLBACK")
                self._con.execute("BEGIN")
            except Exception:
                pass
        elif hasattr(self._con, "rollback"):
            self._con.rollback()

    def close(self) -> None:
        self._con.close()


def _stat_value_to_str(value: Any, dtype: pa.DataType) -> str | None:
    """Serialize a Python value to a DuckLake stat string."""
    if value is None:
        return None
    if pa.types.is_boolean(dtype):
        return "true" if value else "false"
    return str(value)


def _contains_nan(arr: pa.ChunkedArray | pa.Array) -> bool:
    """Check if an Arrow array contains NaN values."""
    try:
        result = pc.any(pc.is_nan(arr))
        return bool(result.as_py()) if result.as_py() is not None else False
    except Exception:
        return False


def _parse_stat_value(value: str | None, arrow_type: pa.DataType) -> object:
    """Parse a DuckLake stat value string into a Python value."""
    if value is None or value == "NULL":
        return None

    if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        value = value[1:-1].replace("''", "'")

    try:
        if pa.types.is_integer(arrow_type):
            return int(value)
        if pa.types.is_floating(arrow_type):
            return float(value)
        if pa.types.is_boolean(arrow_type):
            low = value.lower()
            if low in ("true", "1"):
                return True
            if low in ("false", "0"):
                return False
            return None
        if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return value
        if pa.types.is_date(arrow_type):
            return date.fromisoformat(value)
        if pa.types.is_timestamp(arrow_type):
            return datetime.fromisoformat(value)
        if pa.types.is_decimal(arrow_type):
            return Decimal(value)
    except (ValueError, TypeError, ArithmeticError):
        return None

    return None


# ------------------------------------------------------------------
# Arrow helper functions for join / group-by / unique
# ------------------------------------------------------------------

def _group_by_columns(
    table: pa.Table, col_names: list[str],
) -> list[tuple[tuple, pa.Table]]:
    """Group a table by column names, maintaining insertion order."""
    groups: OrderedDict[tuple, list[int]] = OrderedDict()
    n = len(table)

    if len(col_names) == 1:
        col = table.column(col_names[0])
        for i in range(n):
            key = (col[i].as_py(),)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
    else:
        cols = [table.column(name) for name in col_names]
        for i in range(n):
            key = tuple(c[i].as_py() for c in cols)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

    return [
        (key, table.take(pa.array(indices)))
        for key, indices in groups.items()
    ]


def _select_columns(table: pa.Table, col_names: list[str]) -> pa.Table:
    """Select a subset of columns from a table."""
    return pa.table({name: table.column(name) for name in col_names})


def _unique_rows(table: pa.Table) -> pa.Table:
    """Return unique rows of a table."""
    col_names = table.schema.names
    if len(col_names) == 1:
        unique_vals = pc.unique(table.column(col_names[0]))
        return pa.table({col_names[0]: unique_vals})
    seen: set[tuple] = set()
    indices: list[int] = []
    cols = [table.column(name) for name in col_names]
    for i in range(len(table)):
        key = tuple(c[i].as_py() for c in cols)
        if key not in seen:
            seen.add(key)
            indices.append(i)
    return table.take(pa.array(indices))


def _semi_join(
    left: pa.Table, right: pa.Table, on: list[str],
) -> pa.Table:
    """Keep rows from *left* whose key exists in *right*."""
    if len(on) == 1:
        return left.filter(pc.is_in(left.column(on[0]), right.column(on[0])))
    right_keys: set[tuple] = set()
    rcols = [right.column(name) for name in on]
    for i in range(len(right)):
        right_keys.add(tuple(c[i].as_py() for c in rcols))
    lcols = [left.column(name) for name in on]
    mask = [
        tuple(c[i].as_py() for c in lcols) in right_keys
        for i in range(len(left))
    ]
    return left.filter(pa.array(mask))


def _anti_join(
    left: pa.Table, right: pa.Table, on: list[str],
) -> pa.Table:
    """Keep rows from *left* whose key does NOT exist in *right*."""
    if len(on) == 1:
        return left.filter(pc.invert(pc.is_in(left.column(on[0]), right.column(on[0]))))
    right_keys: set[tuple] = set()
    rcols = [right.column(name) for name in on]
    for i in range(len(right)):
        right_keys.add(tuple(c[i].as_py() for c in rcols))
    lcols = [left.column(name) for name in on]
    mask = [
        tuple(c[i].as_py() for c in lcols) not in right_keys
        for i in range(len(left))
    ]
    return left.filter(pa.array(mask))


def _inner_join(
    left: pa.Table, right: pa.Table, on: list[str],
) -> pa.Table:
    """Inner join *left* and *right* on key columns."""
    right_index: dict[tuple, list[int]] = {}
    rcols = [right.column(name) for name in on]
    for i in range(len(right)):
        key = tuple(c[i].as_py() for c in rcols)
        right_index.setdefault(key, []).append(i)

    left_indices: list[int] = []
    right_indices: list[int] = []
    lcols = [left.column(name) for name in on]
    for i in range(len(left)):
        key = tuple(c[i].as_py() for c in lcols)
        if key in right_index:
            for j in right_index[key]:
                left_indices.append(i)
                right_indices.append(j)

    if not left_indices:
        combined: dict[str, pa.Array] = {}
        for name in left.schema.names:
            combined[name] = pa.array([], type=left.schema.field(name).type)
        for name in right.schema.names:
            if name not in combined:
                combined[name] = pa.array([], type=right.schema.field(name).type)
        return pa.table(combined)

    left_taken = left.take(pa.array(left_indices))
    right_taken = right.take(pa.array(right_indices))

    result_cols: dict[str, pa.ChunkedArray] = {}
    for name in left_taken.schema.names:
        result_cols[name] = left_taken.column(name)
    for name in right_taken.schema.names:
        if name not in result_cols:
            result_cols[name] = right_taken.column(name)
    return pa.table(result_cols)


def _empty_like(table: pa.Table) -> pa.Table:
    """Return an empty table with the same schema."""
    return pa.table(
        {name: pa.array([], type=table.schema.field(name).type) for name in table.schema.names}
    )


def _retryable(method):
    """Decorator: retry a write method on :class:`TransactionConflictError`.

    Uses the writer's ``_max_retries``, ``_retry_wait_ms``, and
    ``_retry_backoff`` settings for exponential backoff.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        wait_ms = self._retry_wait_ms
        for attempt in range(self._max_retries + 1):
            try:
                return method(self, *args, **kwargs)
            except TransactionConflictError:
                if attempt < self._max_retries:
                    time.sleep(wait_ms / 1000.0)
                    wait_ms *= self._retry_backoff
                    self._reset_connection()
                else:
                    raise
        # Unreachable, but keeps mypy happy.
        raise RuntimeError("retry loop exited unexpectedly")  # pragma: no cover

    return wrapper


@dataclass
class _ColumnDef:
    """Internal column definition for registration."""

    column_id: int
    column_order: int
    column_name: str
    column_type: str
    parent_column: int | None
    nulls_allowed: bool


class DuckLakeCatalogWriter:
    """
    Writes metadata to a DuckLake catalog (SQLite or PostgreSQL).

    Uses PyArrow as the internal data representation. Handles snapshot
    creation, table/column registration, Parquet file writing, and
    statistics computation. Produces catalogs fully interoperable with
    DuckDB's DuckLake extension.

    The backend is auto-detected from the metadata path: PostgreSQL
    connection strings (``postgresql://...``) use psycopg2; everything
    else is treated as a SQLite file path. SQL placeholder differences
    (``?`` vs ``%s``) are handled transparently.
    """

    def __init__(
        self,
        metadata_path: str,
        *,
        data_path_override: str | None = None,
        data_inlining_row_limit: int = 0,
        author: str | None = None,
        commit_message: str | None = None,
        max_retries: int = 3,
        retry_wait_ms: float = 100,
        retry_backoff: float = 2.0,
    ) -> None:
        self._backend = create_backend(metadata_path)
        self._metadata_path = metadata_path
        self._data_path_override = data_path_override
        self._data_inlining_row_limit = data_inlining_row_limit
        self._author = author
        self._commit_message = commit_message
        self._con: Any = None
        self._catalog_version: str | None = None

        # Optimistic concurrency control settings
        self._max_retries = max_retries
        self._retry_wait_ms = retry_wait_ms
        self._retry_backoff = retry_backoff

        # Per-transaction conflict tracking state
        self._txn_start_snapshot: int | None = None
        self._txn_conflict_tables: dict[int, str] = {}

    @property
    def _is_v04(self) -> bool:
        """True when the catalog is version 0.4 or later."""
        return self._catalog_version is not None and self._catalog_version >= "0.4"

    def _connect(self) -> Any:
        if self._con is None:
            raw = self._backend.connect_writable()
            self._con = _PlaceholderConnection(raw, self._backend.placeholder)
            self._detect_catalog_version()
        return self._con

    def _detect_catalog_version(self) -> None:
        """Read the catalog version from ducklake_metadata."""
        try:
            row = self._con.execute(
                "SELECT value FROM ducklake_metadata WHERE key = 'version'"
            ).fetchone()
            if row is not None:
                self._catalog_version = row[0]
        except Exception:
            pass

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> DuckLakeCatalogWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Optimistic concurrency control
    # ------------------------------------------------------------------

    def _reset_connection(self) -> None:
        """Reset the connection, discarding any uncommitted state.

        Called between retry attempts after a
        :class:`TransactionConflictError`.
        """
        if self._con is not None:
            try:
                self._con.rollback()
            except Exception:
                pass
            try:
                self._con.close()
            except Exception:
                pass
            self._con = None
        self._txn_start_snapshot = None
        self._txn_conflict_tables = {}

    def _acquire_write_lock(self) -> None:
        """Upgrade to IMMEDIATE transaction on SQLite to hold the write lock."""
        self._connect().begin_immediate()

    def _start_write_transaction(self, start_snapshot_id: int) -> None:
        """Begin tracking a write transaction for conflict detection."""
        self._txn_start_snapshot = start_snapshot_id
        self._txn_conflict_tables = {}

    def _track_table_write(self, table_id: int, operation: str) -> None:
        """Track a table being modified for conflict detection.

        Parameters
        ----------
        table_id
            The table being written to.
        operation
            One of ``'insert'``, ``'delete'``, ``'update'``,
            ``'overwrite'``, ``'ddl'``, ``'drop_table'``.
        """
        if table_id not in self._txn_conflict_tables:
            self._txn_conflict_tables[table_id] = operation

    def _get_concurrent_changes(
        self, start_snapshot_id: int,
    ) -> list[tuple[int, str]]:
        """Read snapshot changes committed after *start_snapshot_id*.

        Uses a **separate** read-only connection so we see changes
        committed by other writers, even if we have an open write
        transaction on ``self._con``.

        Returns an empty list if snapshot tables don't exist (v0.3 catalogs).
        """
        # DuckDB doesn't support multiple connections to the same file
        from ducklake_core._backend import DuckDBBackend
        if isinstance(self._backend, DuckDBBackend):
            return []

        raw = self._backend.connect()
        try:
            check_con = _PlaceholderConnection(raw, self._backend.placeholder)
            # Check if tables exist
            if not self._backend.table_exists(raw, "ducklake_snapshot_changes"):
                return []
            rows = check_con.execute(
                "SELECT sc.snapshot_id, sc.changes_made "
                "FROM ducklake_snapshot_changes sc "
                "JOIN ducklake_snapshot s ON sc.snapshot_id = s.snapshot_id "
                "WHERE sc.snapshot_id > ?",
                [start_snapshot_id],
            ).fetchall()
            return [(r[0], r[1]) for r in rows]
        finally:
            try:
                raw.close()
            except Exception:
                pass

    @staticmethod
    def _parse_table_changes(
        changes_made: str,
    ) -> dict[int, set[str]]:
        """Parse a ``changes_made`` string into ``{table_id: {change_types}}``.

        Recognised prefixes:
        ``inserted_into_table:``, ``deleted_from_table:``,
        ``altered_table:``, ``dropped_table:``.
        """
        result: dict[int, set[str]] = {}
        if not changes_made:
            return result
        for change in changes_made.split(","):
            change = change.strip()
            for prefix in (
                "inserted_into_table:",
                "deleted_from_table:",
                "altered_table:",
                "dropped_table:",
            ):
                if change.startswith(prefix):
                    try:
                        tid = int(change[len(prefix):])
                        result.setdefault(tid, set()).add(prefix.rstrip(":"))
                    except ValueError:
                        pass
        return result

    def _check_conflicts(
        self,
        start_snapshot_id: int,
        conflict_tables: dict[int, str],
    ) -> None:
        """Check for conflicts between our pending writes and concurrent commits.

        Raises :class:`TransactionConflictError` if a conflict is found.
        """
        if not conflict_tables:
            return

        changes = self._get_concurrent_changes(start_snapshot_id)
        if not changes:
            return

        # Aggregate all concurrent per-table changes.
        concurrent: dict[int, set[str]] = {}
        for _snap_id, changes_made in changes:
            for tid, ops in self._parse_table_changes(changes_made).items():
                concurrent.setdefault(tid, set()).update(ops)

        for table_id, operation in conflict_tables.items():
            if table_id not in concurrent:
                continue

            concurrent_ops = concurrent[table_id]

            # Table was dropped → always conflicts.
            if "dropped_table" in concurrent_ops:
                raise TransactionConflictError(
                    f"Conflict: table {table_id} was dropped by a concurrent "
                    f"transaction"
                )

            if operation == "insert":
                # Inserts conflict only with DDL, not with other inserts.
                if "altered_table" in concurrent_ops:
                    raise TransactionConflictError(
                        f"Conflict: table {table_id} schema was altered "
                        f"during concurrent insert"
                    )

            elif operation in ("delete", "update"):
                # Deletes/updates conflict with DDL and other deletes.
                if "altered_table" in concurrent_ops:
                    raise TransactionConflictError(
                        f"Conflict: table {table_id} schema was altered "
                        f"during concurrent {operation}"
                    )
                if "deleted_from_table" in concurrent_ops:
                    raise TransactionConflictError(
                        f"Conflict: concurrent deletes on table {table_id} "
                        f"during {operation}"
                    )

            elif operation == "overwrite":
                # Overwrite conflicts with any concurrent DML/DDL.
                conflicting = concurrent_ops & {
                    "altered_table",
                    "inserted_into_table",
                    "deleted_from_table",
                }
                if conflicting:
                    raise TransactionConflictError(
                        f"Conflict: table {table_id} was modified "
                        f"({', '.join(sorted(conflicting))}) during "
                        f"concurrent overwrite"
                    )

            elif operation == "ddl":
                # DDL conflicts with any concurrent change to the table.
                conflicting = concurrent_ops & {
                    "altered_table",
                    "inserted_into_table",
                    "deleted_from_table",
                }
                if conflicting:
                    raise TransactionConflictError(
                        f"Conflict: table {table_id} was modified "
                        f"({', '.join(sorted(conflicting))}) during "
                        f"concurrent DDL operation"
                    )

            elif operation == "drop_table":
                # Drop conflicts with any concurrent change.
                if concurrent_ops:
                    raise TransactionConflictError(
                        f"Conflict: table {table_id} was modified during "
                        f"concurrent drop"
                    )

    def _commit_metadata(self) -> None:
        """Check for conflicts (if tracked) and commit the transaction.

        If ``_start_write_transaction`` was called, the conflict check
        runs against all tables registered via ``_track_table_write``.
        Otherwise this is a plain ``commit()``.
        """
        if self._txn_start_snapshot is not None and self._txn_conflict_tables:
            self._check_conflicts(
                self._txn_start_snapshot, self._txn_conflict_tables,
            )
        self._connect().commit()
        self._txn_start_snapshot = None
        self._txn_conflict_tables = {}

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    def _get_latest_snapshot(self) -> tuple[int, int, int, int] | None:
        """Return (snapshot_id, schema_version, next_catalog_id, next_file_id).

        Returns None if snapshot tables don't exist (v0.3 catalogs without
        snapshot tracking).

        Acquires the write lock first (IMMEDIATE on SQLite) so the
        returned IDs are safe from concurrent races.
        """
        con = self._connect()
        self._acquire_write_lock()
        if not self._backend.table_exists(con._con, "ducklake_snapshot"):
            return None
        row = con.execute(
            "SELECT snapshot_id, schema_version, next_catalog_id, next_file_id "
            "FROM ducklake_snapshot ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return (row[0], row[1], row[2], row[3])

    def _create_snapshot(
        self,
        schema_version: int,
        next_catalog_id: int,
        next_file_id: int,
    ) -> int:
        """Create a new snapshot and return its ID.

        The caller must hold the write lock (BEGIN IMMEDIATE on SQLite)
        to prevent concurrent writers from generating the same ID.
        """
        con = self._connect()
        row = con.execute(
            "SELECT COALESCE(MAX(snapshot_id), -1) + 1 FROM ducklake_snapshot"
        ).fetchone()
        new_id = row[0]
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00")
        con.execute(
            "INSERT INTO ducklake_snapshot "
            "(snapshot_id, snapshot_time, schema_version, next_catalog_id, next_file_id) "
            "VALUES (?, ?, ?, ?, ?)",
            [new_id, now, schema_version, next_catalog_id, next_file_id],
        )
        return new_id

    def _insert_schema_version(
        self, snapshot_id: int, schema_version: int, table_id: int | None = None
    ) -> None:
        """Insert a schema version record, version-aware."""
        con = self._connect()
        if self._is_v04:
            con.execute(
                "INSERT INTO ducklake_schema_versions "
                "(begin_snapshot, schema_version, table_id) VALUES (?, ?, ?)",
                [snapshot_id, schema_version, table_id],
            )
        else:
            con.execute(
                "INSERT INTO ducklake_schema_versions "
                "(begin_snapshot, schema_version) VALUES (?, ?)",
                [snapshot_id, schema_version],
            )

    def _record_change(
        self,
        snapshot_id: int,
        changes_made: str,
    ) -> None:
        """Insert a snapshot_changes record."""
        con = self._connect()
        con.execute(
            "INSERT INTO ducklake_snapshot_changes "
            "(snapshot_id, changes_made, author, commit_message, commit_extra_info) "
            "VALUES (?, ?, ?, ?, NULL)",
            [snapshot_id, changes_made, self._author, self._commit_message],
        )

    # ------------------------------------------------------------------
    # Data inlining helpers
    # ------------------------------------------------------------------

    def _duckdb_type_to_sql_type(self, duckdb_type: str) -> str:
        """Map a DuckDB column type to an appropriate storage type for the backend."""
        t = duckdb_type.lower()
        if t in (
            "int8", "int16", "int32", "int64", "uint8", "uint16",
            "uint32", "uint64", "boolean", "tinyint", "smallint",
            "integer", "bigint",
        ):
            return "BIGINT"
        if t in ("float32", "float64", "float", "double"):
            if isinstance(self._backend, PostgreSQLBackend):
                return "DOUBLE PRECISION"
            return "DOUBLE"
        return "VARCHAR"

    def _get_inlined_table_name(self, table_id: int, schema_version: int) -> str:
        """Return the dynamic inlined data table name."""
        return f"ducklake_inlined_data_{table_id}_{schema_version}"

    def _get_inlined_active_row_count(
        self, table_id: int, snapshot_id: int
    ) -> int:
        """Count active rows across all inlined data tables for a table."""
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT table_name FROM ducklake_inlined_data_tables "
                "WHERE table_id = ?",
                [table_id],
            ).fetchall()
        except Exception:
            return 0

        total = 0
        for (tbl_name,) in rows:
            safe = tbl_name.replace('"', '""')
            try:
                row = con.execute(
                    f'SELECT COUNT(*) FROM "{safe}" '
                    f"WHERE ? >= begin_snapshot "
                    f"AND (? < end_snapshot OR end_snapshot IS NULL)",
                    [snapshot_id, snapshot_id],
                ).fetchone()
                total += row[0]
            except Exception:
                pass
        return total

    def _ensure_inlined_table(
        self,
        table_id: int,
        schema_version: int,
        columns: list[tuple[int, str, str, int | None]],
    ) -> str:
        """Create the inlined data table if it doesn't exist, register it, and return its name."""
        con = self._connect()
        tbl_name = self._get_inlined_table_name(table_id, schema_version)
        safe = tbl_name.replace('"', '""')

        try:
            row = con.execute(
                "SELECT table_name FROM ducklake_inlined_data_tables "
                "WHERE table_id = ? AND schema_version = ?",
                [table_id, schema_version],
            ).fetchone()
            if row is not None:
                return tbl_name
        except Exception:
            pass

        col_defs = [
            '"row_id" BIGINT',
            '"begin_snapshot" BIGINT',
            '"end_snapshot" BIGINT',
        ]
        for _col_id, col_name, col_type, _parent in columns:
            safe_col = col_name.replace('"', '""')
            sqlite_type = self._duckdb_type_to_sql_type(col_type)
            col_defs.append(f'"{safe_col}" {sqlite_type}')

        create_sql = f'CREATE TABLE IF NOT EXISTS "{safe}" ({", ".join(col_defs)})'
        con.execute(create_sql)

        con.execute(
            "INSERT INTO ducklake_inlined_data_tables "
            "(table_id, table_name, schema_version) "
            "VALUES (?, ?, ?)",
            [table_id, tbl_name, schema_version],
        )
        return tbl_name

    def _serialize_value(self, value: Any, col_type: str) -> Any:
        """Convert a Python value to an appropriate storage format for the backend."""
        if value is None:
            return None
        t = col_type.lower()
        if t == "boolean":
            return 1 if value else 0
        if t in ("date",):
            return str(value)
        if t.startswith("timestamp") or t == "time":
            return str(value)
        if t.startswith("decimal"):
            return str(value)
        return value

    def _insert_inlined_rows(
        self,
        df: pa.Table,
        table_id: int,
        schema_version: int,
        columns: list[tuple[int, str, str, int | None]],
        new_snap: int,
        row_id_start: int,
    ) -> None:
        """Insert Arrow table rows into the inlined data table."""
        con = self._connect()
        tbl_name = self._ensure_inlined_table(table_id, schema_version, columns)
        safe_tbl = tbl_name.replace('"', '""')

        col_names = [c[1] for c in columns]
        col_types = {c[1]: c[2] for c in columns}

        all_cols = ["row_id", "begin_snapshot", "end_snapshot"] + [
            f'"{c.replace(chr(34), chr(34) + chr(34))}"' for c in col_names
        ]
        placeholders = ", ".join(["?"] * len(all_cols))
        insert_sql = f'INSERT INTO "{safe_tbl}" ({", ".join(all_cols)}) VALUES ({placeholders})'

        # Get column arrays for efficient iteration
        col_arrays = [df.column(name) for name in col_names]

        for i in range(len(df)):
            row_vals: list[Any] = [row_id_start + i, new_snap, None]
            for j, col_name in enumerate(col_names):
                value = col_arrays[j][i].as_py()
                row_vals.append(
                    self._serialize_value(value, col_types[col_name])
                )
            con.execute(insert_sql, row_vals)

    def _delete_inlined_rows(
        self,
        table_id: int,
        predicate: Callable[[pa.Table], pa.ChunkedArray],
        snapshot_id: int,
        new_snap: int,
        columns: list[tuple[int, str, str, int | None]],
    ) -> int:
        """Set end_snapshot on inlined rows matching the predicate. Returns count of deleted rows."""
        con = self._connect()
        try:
            inlined_tables = con.execute(
                "SELECT table_name FROM ducklake_inlined_data_tables "
                "WHERE table_id = ?",
                [table_id],
            ).fetchall()
        except Exception:
            return 0

        col_names = [c[1] for c in columns]
        total_deleted = 0

        for (tbl_name,) in inlined_tables:
            safe = tbl_name.replace('"', '""')
            cols_sql = '"row_id", ' + ", ".join(
                f'"{c.replace(chr(34), chr(34) + chr(34))}"' for c in col_names
            )
            try:
                rows = con.execute(
                    f'SELECT {cols_sql} FROM "{safe}" '
                    f"WHERE ? >= begin_snapshot "
                    f"AND (? < end_snapshot OR end_snapshot IS NULL)",
                    [snapshot_id, snapshot_id],
                ).fetchall()
            except Exception:
                continue

            if not rows:
                continue

            data = {name: [r[i + 1] for r in rows] for i, name in enumerate(col_names)}
            row_ids = [r[0] for r in rows]
            inline_table = pa.table(data)

            mask = predicate(inline_table)
            mask_list = mask.to_pylist()
            to_delete = [row_ids[i] for i, v in enumerate(mask_list) if v]

            if to_delete:
                for rid in to_delete:
                    con.execute(
                        f'UPDATE "{safe}" SET end_snapshot = ? WHERE row_id = ? AND end_snapshot IS NULL',
                        [new_snap, rid],
                    )
                total_deleted += len(to_delete)

        return total_deleted

    def _end_all_inlined_rows(
        self, table_id: int, snapshot_id: int, new_snap: int
    ) -> None:
        """Mark all active inlined rows as ended (for overwrite)."""
        con = self._connect()
        try:
            inlined_tables = con.execute(
                "SELECT table_name FROM ducklake_inlined_data_tables "
                "WHERE table_id = ?",
                [table_id],
            ).fetchall()
        except Exception:
            return

        for (tbl_name,) in inlined_tables:
            safe = tbl_name.replace('"', '""')
            try:
                con.execute(
                    f'UPDATE "{safe}" SET end_snapshot = ? '
                    f"WHERE ? >= begin_snapshot "
                    f"AND (? < end_snapshot OR end_snapshot IS NULL)",
                    [new_snap, snapshot_id, snapshot_id],
                )
            except Exception:
                pass

    def _should_inline(
        self, table_id: int, snapshot_id: int, new_row_count: int
    ) -> bool:
        """Return True if the insert should be inlined."""
        if self._data_inlining_row_limit <= 0:
            return False
        current = self._get_inlined_active_row_count(table_id, snapshot_id)
        return (current + new_row_count) <= self._data_inlining_row_limit

    def _get_schema_version_for_table(self, table_id: int, snapshot_id: int) -> int:
        """Return the schema_version at a given snapshot."""
        con = self._connect()
        row = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "WHERE snapshot_id = ?",
            [snapshot_id],
        ).fetchone()
        return row[0] if row else 1

    # ------------------------------------------------------------------
    # Column name mapping
    # ------------------------------------------------------------------

    def _get_next_mapping_id(self) -> int:
        """Return the next available mapping_id."""
        con = self._connect()
        row = con.execute(
            "SELECT COALESCE(MAX(mapping_id), -1) + 1 FROM ducklake_column_mapping"
        ).fetchone()
        return row[0]

    def _register_name_mapping(
        self,
        table_id: int,
        columns: list[tuple[int, str, str, int | None]],
    ) -> int:
        """Register a map_by_name column mapping and return the mapping_id."""
        con = self._connect()
        mapping_id = self._get_next_mapping_id()
        con.execute(
            "INSERT INTO ducklake_column_mapping (mapping_id, table_id, type) "
            "VALUES (?, ?, 'map_by_name')",
            [mapping_id, table_id],
        )
        for col_id, col_name, _col_type, parent_col in columns:
            con.execute(
                "INSERT INTO ducklake_name_mapping "
                "(mapping_id, column_id, source_name, target_field_id, "
                "parent_column, is_partition) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [mapping_id, col_id, col_name, col_id, parent_col, False],
            )
        return mapping_id

    # ------------------------------------------------------------------
    # Data path resolution
    # ------------------------------------------------------------------

    @property
    def data_path(self) -> str:
        if self._data_path_override is not None:
            return self._data_path_override
        con = self._connect()
        row = con.execute(
            "SELECT value FROM ducklake_metadata WHERE key = 'data_path'"
        ).fetchone()
        if row is None:
            msg = "No data_path found in ducklake_metadata"
            raise ValueError(msg)
        return row[0]

    def _resolve_schema_info(
        self, schema_name: str, snapshot_id: int
    ) -> tuple[int, str, bool]:
        """Return (schema_id, schema_path, path_is_relative) for a schema."""
        con = self._connect()
        row = con.execute(
            "SELECT schema_id, path, path_is_relative FROM ducklake_schema "
            "WHERE LOWER(schema_name) = LOWER(?) AND begin_snapshot <= ? "
            "AND (end_snapshot IS NULL OR end_snapshot > ?)",
            [schema_name, snapshot_id, snapshot_id],
        ).fetchone()
        if row is None:
            msg = f"Schema '{schema_name}' not found at snapshot {snapshot_id}"
            raise SchemaNotFoundError(msg)
        return (row[0], row[1] or "", bool(row[2]) if row[2] is not None else True)

    # ------------------------------------------------------------------
    # Table existence check
    # ------------------------------------------------------------------

    def _table_exists(
        self, table_name: str, schema_name: str, snapshot_id: int
    ) -> int | None:
        """Return table_id if the table exists at snapshot_id, else None."""
        con = self._connect()
        row = con.execute(
            "SELECT t.table_id FROM ducklake_table t "
            "JOIN ducklake_schema s ON t.schema_id = s.schema_id "
            "WHERE LOWER(t.table_name) = LOWER(?) AND LOWER(s.schema_name) = LOWER(?) "
            "AND ? >= t.begin_snapshot AND (? < t.end_snapshot OR t.end_snapshot IS NULL) "
            "AND ? >= s.begin_snapshot AND (? < s.end_snapshot OR s.end_snapshot IS NULL)",
            [table_name, schema_name, snapshot_id, snapshot_id, snapshot_id, snapshot_id],
        ).fetchone()
        return row[0] if row is not None else None

    # ------------------------------------------------------------------
    # CREATE TABLE
    # ------------------------------------------------------------------

    def _flatten_schema(
        self,
        arrow_schema: dict[str, pa.DataType],
        start_column_id: int = 1,
        parent_column: int | None = None,
        start_order: int = 1,
    ) -> list[_ColumnDef]:
        """Flatten an Arrow schema dict into column definitions, handling nested types."""
        defs: list[_ColumnDef] = []
        col_id = start_column_id
        order = start_order

        for name, dtype in arrow_schema.items():
            duckdb_type = arrow_type_to_duckdb(dtype)
            this_id = col_id
            col_id += 1

            defs.append(
                _ColumnDef(
                    column_id=this_id,
                    column_order=order,
                    column_name=name,
                    column_type=duckdb_type,
                    parent_column=parent_column,
                    nulls_allowed=True,
                )
            )
            order += 1

            # Recurse into compound types
            if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
                inner = dtype.value_type
                if inner is not None:
                    child_type = arrow_type_to_duckdb(inner)
                    child_id = col_id
                    col_id += 1
                    child_def = _ColumnDef(
                        column_id=child_id,
                        column_order=1,
                        column_name="element",
                        column_type=child_type,
                        parent_column=this_id,
                        nulls_allowed=True,
                    )
                    defs.append(child_def)

                    if pa.types.is_struct(inner):
                        child_dict = {field.name: field.type for field in inner}
                        child_defs = self._flatten_schema(
                            child_dict,
                            start_column_id=col_id,
                            parent_column=child_id,
                        )
                        defs.extend(child_defs)
                        col_id += len(child_defs)

            elif pa.types.is_struct(dtype):
                fields = {field.name: field.type for field in dtype}
                child_defs = self._flatten_schema(
                    fields,
                    start_column_id=col_id,
                    parent_column=this_id,
                )
                defs.extend(child_defs)
                col_id += len(child_defs)

        return defs

    @_retryable
    def create_table(
        self,
        table_name: str,
        arrow_schema: pa.Schema | dict[str, pa.DataType],
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Create a new table in the catalog.

        Returns the new table_id.
        """
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        if self._table_exists(table_name, schema_name, snap_id) is not None:
            msg = f"Table '{schema_name}.{table_name}' already exists"
            raise ValueError(msg)

        schema_id, _schema_path, _schema_path_rel = self._resolve_schema_info(
            schema_name, snap_id
        )

        table_id = next_cat_id
        new_next_cat_id = next_cat_id + 1
        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, new_next_cat_id, next_file_id)

        table_uuid = str(uuid.uuid4())
        table_path = f"{table_name}/"
        con.execute(
            "INSERT INTO ducklake_table "
            "(table_id, table_uuid, begin_snapshot, end_snapshot, schema_id, "
            "table_name, path, path_is_relative) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?, ?)",
            [table_id, table_uuid, new_snap, schema_id, table_name, table_path, True],
        )

        if isinstance(arrow_schema, pa.Schema):
            schema_dict = {field.name: field.type for field in arrow_schema}
        else:
            schema_dict = arrow_schema
        col_defs = self._flatten_schema(schema_dict)
        for cd in col_defs:
            con.execute(
                "INSERT INTO ducklake_column "
                "(column_id, begin_snapshot, end_snapshot, table_id, column_order, "
                "column_name, column_type, initial_default, default_value, "
                "nulls_allowed, parent_column" + (
                    ", default_value_type, default_value_dialect) "
                    "VALUES (?, ?, NULL, ?, ?, ?, ?, NULL, NULL, ?, ?, 'literal', NULL)"
                    if self._is_v04 else
                    ") VALUES (?, ?, NULL, ?, ?, ?, ?, NULL, NULL, ?, ?)"
                ),
                [
                    cd.column_id,
                    new_snap,
                    table_id,
                    cd.column_order,
                    cd.column_name,
                    cd.column_type,
                    cd.nulls_allowed,
                    cd.parent_column,
                ],
            )

        self._insert_schema_version(new_snap, new_schema_ver, table_id)

        safe_schema = schema_name.replace('"', '""')
        safe_table = table_name.replace('"', '""')
        self._record_change(
            new_snap, f'created_table:"{safe_schema}"."{safe_table}"'
        )

        self._commit_metadata()
        return table_id

    @_retryable
    def create_table_with_data(
        self,
        table_name: str,
        df: pa.Table,
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Create a new table and insert data in a single snapshot.

        Returns the new snapshot ID.
        """
        df = _decode_dictionary_columns(df)

        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        if self._table_exists(table_name, schema_name, snap_id) is not None:
            msg = f"Table '{schema_name}.{table_name}' already exists"
            raise ValueError(msg)

        schema_id, schema_path, schema_path_rel = self._resolve_schema_info(
            schema_name, snap_id
        )

        table_id = next_cat_id
        new_next_cat_id = next_cat_id + 1
        new_schema_ver = schema_ver + 1

        schema_dict = {field.name: field.type for field in df.schema}
        col_defs = self._flatten_schema(schema_dict)
        top_level_cols = [
            (cd.column_id, cd.column_name, cd.column_type, cd.parent_column)
            for cd in col_defs
            if cd.parent_column is None
        ]

        table_path = f"{table_name}/"

        has_data = len(df) > 0
        inline_data = (
            has_data
            and self._data_inlining_row_limit > 0
            and len(df) <= self._data_inlining_row_limit
        )

        if has_data and not inline_data:
            data_file_id = next_file_id
            new_next_file_id = next_file_id + 1
        else:
            new_next_file_id = next_file_id

        new_snap = self._create_snapshot(
            new_schema_ver, new_next_cat_id, new_next_file_id
        )

        table_uuid = str(uuid.uuid4())
        con.execute(
            "INSERT INTO ducklake_table "
            "(table_id, table_uuid, begin_snapshot, end_snapshot, schema_id, "
            "table_name, path, path_is_relative) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?, ?)",
            [table_id, table_uuid, new_snap, schema_id, table_name, table_path, True],
        )

        for cd in col_defs:
            con.execute(
                "INSERT INTO ducklake_column "
                "(column_id, begin_snapshot, end_snapshot, table_id, column_order, "
                "column_name, column_type, initial_default, default_value, "
                "nulls_allowed, parent_column" + (
                    ", default_value_type, default_value_dialect) "
                    "VALUES (?, ?, NULL, ?, ?, ?, ?, NULL, NULL, ?, ?, 'literal', NULL)"
                    if self._is_v04 else
                    ") VALUES (?, ?, NULL, ?, ?, ?, ?, NULL, NULL, ?, ?)"
                ),
                [
                    cd.column_id,
                    new_snap,
                    table_id,
                    cd.column_order,
                    cd.column_name,
                    cd.column_type,
                    cd.nulls_allowed,
                    cd.parent_column,
                ],
            )

        self._insert_schema_version(new_snap, new_schema_ver, table_id)

        if has_data:
            if inline_data:
                self._insert_inlined_rows(
                    df, table_id, new_schema_ver, top_level_cols, new_snap, 0,
                )
                self._update_table_stats(table_id, len(df), 0)
                col_stats = self._compute_file_column_stats(df, top_level_cols)
                self._update_table_column_stats(
                    table_id, top_level_cols, col_stats,
                )
            else:
                base = self.data_path
                if schema_path_rel:
                    base = storage.join_path(base, schema_path)
                else:
                    base = schema_path
                base = storage.join_path(base, table_path)
                storage.makedirs(base, exist_ok=True)

                file_name = f"ducklake-{_uuid7()}.parquet"
                file_path = storage.join_path(base, file_name)
                storage.write_parquet(df, file_path)

                file_size = storage.get_file_size(file_path)
                footer_size = _read_parquet_footer_size(file_path)

                mapping_id = self._register_name_mapping(table_id, top_level_cols)

                self._register_data_file(
                    data_file_id, table_id, new_snap, file_name,
                    len(df), file_size, footer_size, 0,
                    None, mapping_id,
                )

                col_stats = self._compute_file_column_stats(df, top_level_cols)
                self._register_file_column_stats(
                    data_file_id, table_id, col_stats,
                )
                self._update_table_stats(table_id, len(df), file_size)
                self._update_table_column_stats(
                    table_id, top_level_cols, col_stats,
                )

        safe_schema = schema_name.replace('"', '""')
        safe_table = table_name.replace('"', '""')
        self._record_change(
            new_snap, f'created_table:"{safe_schema}"."{safe_table}"'
        )

        self._commit_metadata()
        return new_snap

    # ------------------------------------------------------------------
    # INSERT DATA
    # ------------------------------------------------------------------

    def _get_table_info(
        self, table_name: str, schema_name: str, snapshot_id: int
    ) -> tuple[int, str, bool, str, bool]:
        """Return (table_id, table_path, table_path_rel, schema_path, schema_path_rel)."""
        con = self._connect()
        row = con.execute(
            "SELECT t.table_id, t.path, t.path_is_relative, s.path, s.path_is_relative "
            "FROM ducklake_table t "
            "JOIN ducklake_schema s ON t.schema_id = s.schema_id "
            "WHERE LOWER(t.table_name) = LOWER(?) AND LOWER(s.schema_name) = LOWER(?) "
            "AND ? >= t.begin_snapshot AND (? < t.end_snapshot OR t.end_snapshot IS NULL) "
            "AND ? >= s.begin_snapshot AND (? < s.end_snapshot OR s.end_snapshot IS NULL)",
            [table_name, schema_name, snapshot_id, snapshot_id, snapshot_id, snapshot_id],
        ).fetchone()
        if row is None:
            msg = f"Table '{schema_name}.{table_name}' not found at snapshot {snapshot_id}"
            raise TableNotFoundError(msg)
        return (
            row[0],
            row[1] or "",
            bool(row[2]) if row[2] is not None else True,
            row[3] or "",
            bool(row[4]) if row[4] is not None else True,
        )

    def _get_columns_for_table(
        self, table_id: int, snapshot_id: int
    ) -> list[tuple[int, str, str, int | None]]:
        """Return [(column_id, column_name, column_type, parent_column)] for top-level columns."""
        con = self._connect()
        rows = con.execute(
            "SELECT column_id, column_name, column_type, parent_column "
            "FROM ducklake_column "
            "WHERE table_id = ? AND ? >= begin_snapshot "
            "AND (? < end_snapshot OR end_snapshot IS NULL) "
            "AND parent_column IS NULL "
            "ORDER BY column_order",
            [table_id, snapshot_id, snapshot_id],
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def _get_table_stats(self, table_id: int) -> tuple[int, int, int] | None:
        """Return (record_count, next_row_id, file_size_bytes) or None."""
        con = self._connect()
        row = con.execute(
            "SELECT record_count, next_row_id, file_size_bytes "
            "FROM ducklake_table_stats WHERE table_id = ?",
            [table_id],
        ).fetchone()
        return (row[0], row[1], row[2]) if row else None

    def _compute_file_column_stats(
        self,
        df: pa.Table,
        columns: list[tuple[int, str, str, int | None]],
    ) -> list[tuple[int, str, int, int | None, str | None, str | None, int | None]]:
        """Compute per-column statistics for an Arrow table.

        Returns list of (column_id, column_name, value_count, null_count,
        min_value, max_value, contains_nan_int).
        """
        results = []
        n = len(df)
        col_names_set = set(df.schema.names)
        for col_id, col_name, col_type, _parent in columns:
            if col_name not in col_names_set:
                continue
            col_array = df.column(col_name)
            # Dictionary-encoded arrays (e.g. from Polars Categorical/Enum)
            # must be decoded before computing min/max statistics.
            if pa.types.is_dictionary(col_array.type):
                col_array = col_array.cast(col_array.type.value_type)
            null_count = col_array.null_count
            value_count = n

            min_val = None
            max_val = None
            try:
                arrow_type = duckdb_type_to_arrow(col_type)
                if (
                    pa.types.is_integer(arrow_type)
                    or pa.types.is_floating(arrow_type)
                    or pa.types.is_boolean(arrow_type)
                    or pa.types.is_string(arrow_type)
                    or pa.types.is_large_string(arrow_type)
                    or pa.types.is_date(arrow_type)
                    or pa.types.is_timestamp(arrow_type)
                    or pa.types.is_decimal(arrow_type)
                ):
                    if null_count < n:
                        raw_min = pc.min(col_array).as_py()
                        raw_max = pc.max(col_array).as_py()
                        min_val = _stat_value_to_str(raw_min, arrow_type)
                        max_val = _stat_value_to_str(raw_max, arrow_type)
            except (ValueError, TypeError):
                pass

            nan_int = None
            if col_type.lower() in ("float32", "float64"):
                nan_int = _contains_nan(col_array)

            results.append(
                (col_id, col_name, value_count, null_count, min_val, max_val, nan_int)
            )
        return results

    def _merge_stat_value(
        self,
        existing: str | None,
        new: str | None,
        col_type: str,
        pick_min: bool,
    ) -> str | None:
        """Merge two stat values, returning the min or max."""
        if existing is None:
            return new
        if new is None:
            return existing
        try:
            arrow_type = duckdb_type_to_arrow(col_type)
            e_val = _parse_stat_value(existing, arrow_type)
            n_val = _parse_stat_value(new, arrow_type)
            if e_val is None:
                return new
            if n_val is None:
                return existing
            if pick_min:
                return existing if e_val <= n_val else new  # type: ignore[operator]
            else:
                return existing if e_val >= n_val else new  # type: ignore[operator]
        except (ValueError, TypeError):
            return new

    def _build_hive_path(
        self,
        partition_col_names: list[str],
        partition_values: list[str],
    ) -> str:
        """Build a Hive-style partition path like ``b=x/c=10``."""
        parts = []
        for name, val in zip(partition_col_names, partition_values):
            safe_name = name.replace("=", "%3D")
            safe_val = str(val).replace("/", "%2F")
            parts.append(f"{safe_name}={safe_val}")
        return "/".join(parts)

    def _register_data_file(
        self,
        data_file_id: int,
        table_id: int,
        new_snap: int,
        rel_path: str,
        record_count: int,
        file_size: int,
        footer_size: int,
        row_id_start: int,
        partition_id: int | None,
        mapping_id: int,
    ) -> None:
        """Register a single data file in the catalog."""
        con = self._connect()
        con.execute(
            "INSERT INTO ducklake_data_file "
            "(data_file_id, table_id, begin_snapshot, end_snapshot, file_order, "
            "path, path_is_relative, file_format, record_count, file_size_bytes, "
            "footer_size, row_id_start, partition_id, encryption_key, "
            + ("mapping_id, partial_max) "
               "VALUES (?, ?, ?, NULL, NULL, ?, ?, 'parquet', ?, ?, ?, ?, ?, NULL, ?, NULL)"
               if self._is_v04 else
               "partial_file_info, mapping_id) "
               "VALUES (?, ?, ?, NULL, NULL, ?, ?, 'parquet', ?, ?, ?, ?, ?, NULL, NULL, ?)"),
            [
                data_file_id,
                table_id,
                new_snap,
                rel_path,
                True,
                record_count,
                file_size,
                footer_size,
                row_id_start,
                partition_id,
                mapping_id,
            ],
        )

    def _register_file_column_stats(
        self,
        data_file_id: int,
        table_id: int,
        col_stats: list[tuple[int, str, int, int | None, str | None, str | None, int | None]],
    ) -> None:
        """Register per-file column statistics."""
        con = self._connect()
        for col_id, _col_name, value_count, null_count, min_val, max_val, nan_int in col_stats:
            con.execute(
                "INSERT INTO ducklake_file_column_stats "
                "(data_file_id, table_id, column_id, column_size_bytes, "
                "value_count, null_count, min_value, max_value, contains_nan, extra_stats) "
                "VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, NULL)",
                [
                    data_file_id,
                    table_id,
                    col_id,
                    value_count,
                    null_count,
                    min_val,
                    max_val,
                    nan_int,
                ],
            )

    def _register_partition_values(
        self,
        data_file_id: int,
        table_id: int,
        partition_key_indices: list[int],
        partition_values: list[str],
    ) -> None:
        """Register partition values for a data file."""
        con = self._connect()
        for key_index, val in zip(partition_key_indices, partition_values):
            con.execute(
                "INSERT INTO ducklake_file_partition_value "
                "(data_file_id, table_id, partition_key_index, partition_value) "
                "VALUES (?, ?, ?, ?)",
                [data_file_id, table_id, key_index, val],
            )

    def _update_table_stats(
        self,
        table_id: int,
        added_records: int,
        added_file_size: int,
    ) -> None:
        """Update aggregate table stats after inserting data."""
        con = self._connect()
        existing_stats = self._get_table_stats(table_id)
        if existing_stats is not None:
            row_id_start = existing_stats[1]
        else:
            row_id_start = 0

        new_record_count = (existing_stats[0] if existing_stats else 0) + added_records
        new_next_row_id = row_id_start + added_records
        new_file_size = (existing_stats[2] if existing_stats else 0) + added_file_size

        if existing_stats is not None:
            con.execute(
                "UPDATE ducklake_table_stats "
                "SET record_count = ?, next_row_id = ?, file_size_bytes = ? "
                "WHERE table_id = ?",
                [new_record_count, new_next_row_id, new_file_size, table_id],
            )
        else:
            con.execute(
                "INSERT INTO ducklake_table_stats "
                "(table_id, record_count, next_row_id, file_size_bytes) "
                "VALUES (?, ?, ?, ?)",
                [table_id, new_record_count, new_next_row_id, new_file_size],
            )

    def _update_table_column_stats(
        self,
        table_id: int,
        columns: list[tuple[int, str, str, int | None]],
        col_stats: list[tuple[int, str, int, int | None, str | None, str | None, int | None]],
    ) -> None:
        """Update aggregate table column stats after inserting data."""
        con = self._connect()
        for col_id, col_name, _vc, null_count, min_val, max_val, nan_int in col_stats:
            col_type = ""
            for c_id, c_name, c_type, _p in columns:
                if c_id == col_id:
                    col_type = c_type
                    break

            existing_col_stat = con.execute(
                "SELECT contains_null, contains_nan, min_value, max_value "
                "FROM ducklake_table_column_stats WHERE table_id = ? AND column_id = ?",
                [table_id, col_id],
            ).fetchone()

            contains_null = bool(null_count and null_count > 0)
            contains_nan = nan_int if nan_int is not None else None

            if existing_col_stat is not None:
                merged_null = bool(existing_col_stat[0] or contains_null)
                merged_nan = None
                if existing_col_stat[1] is not None or contains_nan is not None:
                    merged_nan = bool(existing_col_stat[1] or contains_nan)
                merged_min = self._merge_stat_value(
                    existing_col_stat[2], min_val, col_type, pick_min=True
                )
                merged_max = self._merge_stat_value(
                    existing_col_stat[3], max_val, col_type, pick_min=False
                )
                con.execute(
                    "UPDATE ducklake_table_column_stats "
                    "SET contains_null = ?, contains_nan = ?, "
                    "min_value = ?, max_value = ? "
                    "WHERE table_id = ? AND column_id = ?",
                    [merged_null, merged_nan, merged_min, merged_max, table_id, col_id],
                )
            else:
                con.execute(
                    "INSERT INTO ducklake_table_column_stats "
                    "(table_id, column_id, contains_null, contains_nan, "
                    "min_value, max_value, extra_stats) "
                    "VALUES (?, ?, ?, ?, ?, ?, NULL)",
                    [table_id, col_id, contains_null, contains_nan, min_val, max_val],
                )

    @_retryable
    def insert_data(
        self,
        df: pa.Table,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Insert an Arrow table into an existing table.

        Returns the new snapshot ID.
        """
        df = _decode_dictionary_columns(df)

        if len(df) == 0:
            msg = "Cannot insert empty DataFrame"
            raise ValueError(msg)

        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )
        self._track_table_write(table_id, "insert")

        columns = self._get_columns_for_table(table_id, snap_id)

        record_count = len(df)

        if self._should_inline(table_id, snap_id, record_count):
            return self._insert_inlined(
                df, table_id, columns,
                snap_id, schema_ver, next_cat_id, next_file_id,
            )

        base = self.data_path
        if schema_path_rel:
            base = storage.join_path(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = storage.join_path(base, table_path)
        else:
            base = table_path

        partition_id = self._get_active_partition(table_id, snap_id)

        if partition_id is not None:
            return self._insert_partitioned(
                df, table_id, table_name, schema_name,
                columns, base, partition_id,
                snap_id, schema_ver, next_cat_id, next_file_id,
            )

        storage.makedirs(base, exist_ok=True)

        # Sort by sort keys if defined
        df = self._maybe_sort(df, table_id, snap_id)

        file_name = f"ducklake-{_uuid7()}.parquet"
        file_path = storage.join_path(base, file_name)
        storage.write_parquet(df, file_path)

        file_size = storage.get_file_size(file_path)
        footer_size = _read_parquet_footer_size(file_path)

        existing_stats = self._get_table_stats(table_id)
        row_id_start = existing_stats[1] if existing_stats is not None else 0

        data_file_id = next_file_id
        new_next_file_id = next_file_id + 1

        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        mapping_id = self._register_name_mapping(table_id, columns)

        self._register_data_file(
            data_file_id, table_id, new_snap, file_name,
            record_count, file_size, footer_size, row_id_start,
            None, mapping_id,
        )

        col_stats = self._compute_file_column_stats(df, columns)
        self._register_file_column_stats(data_file_id, table_id, col_stats)

        self._update_table_stats(table_id, record_count, file_size)
        self._update_table_column_stats(table_id, columns, col_stats)

        self._record_change(new_snap, f"inserted_into_table:{table_id}")
        self._commit_metadata()
        return new_snap

    def _insert_inlined(
        self,
        df: pa.Table,
        table_id: int,
        columns: list[tuple[int, str, str, int | None]],
        snap_id: int,
        schema_ver: int,
        next_cat_id: int,
        next_file_id: int,
    ) -> int:
        """Insert data into the inlined data table instead of Parquet."""
        con = self._connect()
        record_count = len(df)

        existing_stats = self._get_table_stats(table_id)
        row_id_start = existing_stats[1] if existing_stats is not None else 0

        new_snap = self._create_snapshot(schema_ver, next_cat_id, next_file_id)

        self._insert_inlined_rows(
            df, table_id, schema_ver, columns, new_snap, row_id_start,
        )

        self._update_table_stats(table_id, record_count, 0)
        self._update_table_column_stats(
            table_id, columns,
            self._compute_file_column_stats(df, columns),
        )

        self._record_change(new_snap, f"inserted_into_table:{table_id}")
        self._commit_metadata()
        return new_snap

    def _insert_partitioned(
        self,
        df: pa.Table,
        table_id: int,
        table_name: str,
        schema_name: str,
        columns: list[tuple[int, str, str, int | None]],
        base_dir: str,
        partition_id: int,
        snap_id: int,
        schema_ver: int,
        next_cat_id: int,
        next_file_id: int,
    ) -> int:
        """Insert data into a partitioned table, one file per partition group."""
        con = self._connect()

        part_cols = self._get_partition_columns(partition_id, table_id)
        col_id_to_name: dict[int, str] = {c[0]: c[1] for c in columns}
        part_col_names: list[str] = []
        part_key_indices: list[int] = []
        for key_index, col_id, transform in part_cols:
            part_col_names.append(col_id_to_name[col_id])
            part_key_indices.append(key_index)

        group_list = _group_by_columns(df, part_col_names)

        n_files = len(group_list)

        existing_stats = self._get_table_stats(table_id)
        row_id_start = existing_stats[1] if existing_stats is not None else 0

        new_next_file_id = next_file_id + n_files

        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        mapping_id = self._register_name_mapping(table_id, columns)

        total_file_size = 0
        total_records = 0
        all_col_stats: list[
            list[tuple[int, str, int, int | None, str | None, str | None, int | None]]
        ] = []

        # Get sort keys for sorting each partition group
        sort_keys = self._get_active_sort_keys(table_id, snap_id)

        current_file_id = next_file_id
        current_row_id = row_id_start

        for group_key, group_df in group_list:
            # Sort partition group by sort keys if defined
            if sort_keys:
                group_df = self._sort_table_by_keys(group_df, sort_keys)

            partition_values = [str(v) for v in group_key]
            hive_subdir = self._build_hive_path(part_col_names, partition_values)
            partition_dir = storage.join_path(base_dir, hive_subdir)
            storage.makedirs(partition_dir, exist_ok=True)

            file_name = f"ducklake-{_uuid7()}.parquet"
            file_path = storage.join_path(partition_dir, file_name)
            storage.write_parquet(group_df, file_path)

            file_size = storage.get_file_size(file_path)
            footer_size = _read_parquet_footer_size(file_path)
            record_count = len(group_df)

            rel_path = f"{hive_subdir}/{file_name}"

            self._register_data_file(
                current_file_id, table_id, new_snap, rel_path,
                record_count, file_size, footer_size, current_row_id,
                partition_id, mapping_id,
            )

            self._register_partition_values(
                current_file_id, table_id, part_key_indices, partition_values,
            )

            col_stats = self._compute_file_column_stats(group_df, columns)
            self._register_file_column_stats(current_file_id, table_id, col_stats)
            all_col_stats.append(col_stats)

            total_file_size += file_size
            total_records += record_count
            current_file_id += 1
            current_row_id += record_count

        self._update_table_stats(table_id, total_records, total_file_size)

        full_col_stats = self._compute_file_column_stats(df, columns)
        self._update_table_column_stats(table_id, columns, full_col_stats)

        self._record_change(new_snap, f"inserted_into_table:{table_id}")
        self._commit_metadata()
        return new_snap

    # ------------------------------------------------------------------
    # OVERWRITE (truncate + insert)
    # ------------------------------------------------------------------

    def _end_all_data_files(self, table_id: int, snapshot_id: int, new_snap: int) -> None:
        """Mark all active data files as ended at new_snap."""
        con = self._connect()
        con.execute(
            "UPDATE ducklake_data_file SET end_snapshot = ? "
            "WHERE table_id = ? AND begin_snapshot <= ? "
            "AND (end_snapshot IS NULL OR end_snapshot > ?)",
            [new_snap, table_id, snapshot_id, snapshot_id],
        )

    def _end_all_delete_files(self, table_id: int, snapshot_id: int, new_snap: int) -> None:
        """Mark all active delete files as ended at new_snap."""
        con = self._connect()
        con.execute(
            "UPDATE ducklake_delete_file SET end_snapshot = ? "
            "WHERE table_id = ? AND begin_snapshot <= ? "
            "AND (end_snapshot IS NULL OR end_snapshot > ?)",
            [new_snap, table_id, snapshot_id, snapshot_id],
        )

    @_retryable
    def overwrite_data(
        self,
        df: pa.Table,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Overwrite all data in a table with a new Arrow table.

        Returns the new snapshot ID.
        """
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )
        self._track_table_write(table_id, "overwrite")

        columns = self._get_columns_for_table(table_id, snap_id)

        base = self.data_path
        if schema_path_rel:
            base = storage.join_path(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = storage.join_path(base, table_path)
        else:
            base = table_path
        storage.makedirs(base, exist_ok=True)

        record_count = len(df)

        part_id = self._get_active_partition(table_id, snap_id)

        if part_id is not None and record_count > 0:
            return self._overwrite_partitioned(
                df, table_id, columns, base, part_id,
                snap_id, schema_ver, next_cat_id, next_file_id,
            )

        data_file_id = next_file_id
        new_next_file_id = next_file_id + 1 if record_count > 0 else next_file_id

        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        self._end_all_data_files(table_id, snap_id, new_snap)
        self._end_all_delete_files(table_id, snap_id, new_snap)
        self._end_all_inlined_rows(table_id, snap_id, new_snap)

        # Check if new data should be inlined
        if record_count > 0 and self._should_inline(table_id, new_snap, record_count):
            self._insert_inlined_rows(
                df, table_id, schema_ver, columns, new_snap, 0,
            )
            existing_stats = self._get_table_stats(table_id)
            if existing_stats is not None:
                con.execute(
                    "UPDATE ducklake_table_stats "
                    "SET record_count = ?, next_row_id = ?, file_size_bytes = 0 "
                    "WHERE table_id = ?",
                    [record_count, record_count, table_id],
                )
            else:
                con.execute(
                    "INSERT INTO ducklake_table_stats "
                    "(table_id, record_count, next_row_id, file_size_bytes) "
                    "VALUES (?, ?, ?, 0)",
                    [table_id, record_count, record_count],
                )
            con.execute(
                "DELETE FROM ducklake_table_column_stats WHERE table_id = ?",
                [table_id],
            )
            col_stats = self._compute_file_column_stats(df, columns)
            for col_id, _col_name, _vc, null_count, min_val, max_val, nan_int in col_stats:
                contains_null = bool(null_count and null_count > 0)
                con.execute(
                    "INSERT INTO ducklake_table_column_stats "
                    "(table_id, column_id, contains_null, contains_nan, "
                    "min_value, max_value, extra_stats) "
                    "VALUES (?, ?, ?, ?, ?, ?, NULL)",
                    [table_id, col_id, contains_null, nan_int, min_val, max_val],
                )
            self._record_change(new_snap, f"inserted_into_table:{table_id}")
            self._commit_metadata()
            return new_snap

        if record_count > 0:
            # Sort by sort keys if defined
            df = self._maybe_sort(df, table_id, snap_id)

            file_name = f"ducklake-{_uuid7()}.parquet"
            file_path = storage.join_path(base, file_name)
            storage.write_parquet(df, file_path)

            file_size = storage.get_file_size(file_path)
            footer_size = _read_parquet_footer_size(file_path)

            mapping_id = self._register_name_mapping(table_id, columns)

            self._register_data_file(
                data_file_id, table_id, new_snap, file_name,
                record_count, file_size, footer_size, 0,
                None, mapping_id,
            )

            col_stats = self._compute_file_column_stats(df, columns)
            self._register_file_column_stats(data_file_id, table_id, col_stats)

            existing_stats = self._get_table_stats(table_id)
            if existing_stats is not None:
                con.execute(
                    "UPDATE ducklake_table_stats "
                    "SET record_count = ?, next_row_id = ?, file_size_bytes = ? "
                    "WHERE table_id = ?",
                    [record_count, record_count, file_size, table_id],
                )
            else:
                con.execute(
                    "INSERT INTO ducklake_table_stats "
                    "(table_id, record_count, next_row_id, file_size_bytes) "
                    "VALUES (?, ?, ?, ?)",
                    [table_id, record_count, record_count, file_size],
                )

            con.execute(
                "DELETE FROM ducklake_table_column_stats WHERE table_id = ?",
                [table_id],
            )
            for col_id, _col_name, _vc, null_count, min_val, max_val, nan_int in col_stats:
                contains_null = bool(null_count and null_count > 0)
                con.execute(
                    "INSERT INTO ducklake_table_column_stats "
                    "(table_id, column_id, contains_null, contains_nan, "
                    "min_value, max_value, extra_stats) "
                    "VALUES (?, ?, ?, ?, ?, ?, NULL)",
                    [table_id, col_id, contains_null, nan_int, min_val, max_val],
                )
        else:
            existing_stats = self._get_table_stats(table_id)
            if existing_stats is not None:
                con.execute(
                    "UPDATE ducklake_table_stats "
                    "SET record_count = 0, next_row_id = 0, file_size_bytes = 0 "
                    "WHERE table_id = ?",
                    [table_id],
                )
            con.execute(
                "DELETE FROM ducklake_table_column_stats WHERE table_id = ?",
                [table_id],
            )

        self._record_change(new_snap, f"inserted_into_table:{table_id}")
        self._commit_metadata()
        return new_snap

    def _overwrite_partitioned(
        self,
        df: pa.Table,
        table_id: int,
        columns: list[tuple[int, str, str, int | None]],
        base_dir: str,
        partition_id: int,
        snap_id: int,
        schema_ver: int,
        next_cat_id: int,
        next_file_id: int,
    ) -> int:
        """Overwrite a partitioned table with new data."""
        con = self._connect()

        part_cols = self._get_partition_columns(partition_id, table_id)
        col_id_to_name: dict[int, str] = {c[0]: c[1] for c in columns}
        part_col_names = [col_id_to_name[col_id] for _, col_id, _ in part_cols]
        part_key_indices = [key_index for key_index, _, _ in part_cols]

        group_list = _group_by_columns(df, part_col_names)

        n_files = len(group_list)
        new_next_file_id = next_file_id + n_files

        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        self._end_all_data_files(table_id, snap_id, new_snap)
        self._end_all_delete_files(table_id, snap_id, new_snap)
        self._end_all_inlined_rows(table_id, snap_id, new_snap)

        mapping_id = self._register_name_mapping(table_id, columns)

        # Get sort keys for sorting each partition group
        sort_keys = self._get_active_sort_keys(table_id, snap_id)

        total_file_size = 0
        total_records = 0
        current_file_id = next_file_id
        current_row_id = 0

        for group_key, group_df in group_list:
            # Sort partition group by sort keys if defined
            if sort_keys:
                group_df = self._sort_table_by_keys(group_df, sort_keys)

            partition_values = [str(v) for v in group_key]
            hive_subdir = self._build_hive_path(part_col_names, partition_values)
            partition_dir = storage.join_path(base_dir, hive_subdir)
            storage.makedirs(partition_dir, exist_ok=True)

            file_name = f"ducklake-{_uuid7()}.parquet"
            file_path = storage.join_path(partition_dir, file_name)
            storage.write_parquet(group_df, file_path)

            file_size = storage.get_file_size(file_path)
            footer_size = _read_parquet_footer_size(file_path)
            record_count = len(group_df)
            rel_path = f"{hive_subdir}/{file_name}"

            self._register_data_file(
                current_file_id, table_id, new_snap, rel_path,
                record_count, file_size, footer_size, current_row_id,
                partition_id, mapping_id,
            )
            self._register_partition_values(
                current_file_id, table_id, part_key_indices, partition_values,
            )

            col_stats = self._compute_file_column_stats(group_df, columns)
            self._register_file_column_stats(current_file_id, table_id, col_stats)

            total_file_size += file_size
            total_records += record_count
            current_file_id += 1
            current_row_id += record_count

        existing_stats = self._get_table_stats(table_id)
        if existing_stats is not None:
            con.execute(
                "UPDATE ducklake_table_stats "
                "SET record_count = ?, next_row_id = ?, file_size_bytes = ? "
                "WHERE table_id = ?",
                [total_records, total_records, total_file_size, table_id],
            )
        else:
            con.execute(
                "INSERT INTO ducklake_table_stats "
                "(table_id, record_count, next_row_id, file_size_bytes) "
                "VALUES (?, ?, ?, ?)",
                [table_id, total_records, total_records, total_file_size],
            )

        con.execute(
            "DELETE FROM ducklake_table_column_stats WHERE table_id = ?",
            [table_id],
        )
        full_col_stats = self._compute_file_column_stats(df, columns)
        for col_id, _col_name, _vc, null_count, min_val, max_val, nan_int in full_col_stats:
            contains_null = bool(null_count and null_count > 0)
            con.execute(
                "INSERT INTO ducklake_table_column_stats "
                "(table_id, column_id, contains_null, contains_nan, "
                "min_value, max_value, extra_stats) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL)",
                [table_id, col_id, contains_null, nan_int, min_val, max_val],
            )

        self._record_change(new_snap, f"inserted_into_table:{table_id}")
        self._commit_metadata()
        return new_snap

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def _get_active_data_files(
        self, table_id: int, snapshot_id: int
    ) -> list[tuple[int, str, bool, int, int]]:
        """Return [(data_file_id, path, path_is_relative, record_count, row_id_start)]."""
        con = self._connect()
        rows = con.execute(
            "SELECT data_file_id, path, path_is_relative, record_count, row_id_start "
            "FROM ducklake_data_file "
            "WHERE table_id = ? AND ? >= begin_snapshot "
            "AND (? < end_snapshot OR end_snapshot IS NULL) "
            "ORDER BY file_order, data_file_id",
            [table_id, snapshot_id, snapshot_id],
        ).fetchall()
        return [
            (r[0], r[1], bool(r[2]) if r[2] is not None else True, r[3], r[4])
            for r in rows
        ]

    def _resolve_file_path(
        self,
        file_path: str,
        file_is_relative: bool,
        table_path: str,
        table_path_rel: bool,
        schema_path: str,
        schema_path_rel: bool,
    ) -> str:
        """Resolve a data file path to an absolute path."""
        if not file_is_relative:
            return file_path
        base = self.data_path
        if schema_path_rel:
            base = storage.join_path(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = storage.join_path(base, table_path)
        else:
            base = table_path
        return storage.join_path(base, file_path)

    def _get_active_delete_positions(
        self,
        data_file_id: int,
        table_id: int,
        snapshot_id: int,
        table_path: str,
        table_path_rel: bool,
        schema_path: str,
        schema_path_rel: bool,
    ) -> set[int]:
        """Return row positions that are deleted for a data file at the given snapshot."""
        con = self._connect()
        rows = con.execute(
            "SELECT path, path_is_relative "
            "FROM ducklake_delete_file "
            "WHERE table_id = ? AND data_file_id = ? "
            "AND begin_snapshot <= ? "
            "AND (end_snapshot IS NULL OR end_snapshot > ?)",
            [table_id, data_file_id, snapshot_id, snapshot_id],
        ).fetchall()

        if not rows:
            return set()

        positions: set[int] = set()
        for del_path, del_is_rel in rows:
            abs_del = self._resolve_file_path(
                del_path,
                bool(del_is_rel) if del_is_rel is not None else True,
                table_path, table_path_rel,
                schema_path, schema_path_rel,
            )
            try:
                del_table = storage.read_parquet(abs_del)
                positions.update(del_table.column("pos").to_pylist())
            except (OSError, FileNotFoundError):
                pass

        return positions

    def _read_active_data_file(
        self,
        data_file_id: int,
        abs_path: str,
        table_id: int,
        snapshot_id: int,
        table_path: str,
        table_path_rel: bool,
        schema_path: str,
        schema_path_rel: bool,
    ) -> pa.Table:
        """Read a data file and exclude rows covered by active delete files."""
        df = storage.read_parquet(abs_path)
        deleted_positions = self._get_active_delete_positions(
            data_file_id, table_id, snapshot_id,
            table_path, table_path_rel, schema_path, schema_path_rel,
        )
        if deleted_positions:
            keep_indices = [i for i in range(len(df)) if i not in deleted_positions]
            df = df.take(pa.array(keep_indices, type=pa.int64()))
        return df

    def _write_cumulative_delete_file(
        self,
        data_file_id: int,
        table_id: int,
        new_snap: int,
        snapshot_id: int,
        new_positions: list[int],
        abs_data_path: str,
        base_dir: str,
        delete_file_id: int,
        table_path: str,
        table_path_rel: bool,
        schema_path: str,
        schema_path_rel: bool,
    ) -> None:
        """Write a cumulative delete file that combines existing + new positions."""
        con = self._connect()

        existing_positions = self._get_active_delete_positions(
            data_file_id, table_id, snapshot_id,
            table_path, table_path_rel, schema_path, schema_path_rel,
        )

        all_positions = sorted(set(existing_positions) | set(new_positions))

        con.execute(
            "UPDATE ducklake_delete_file SET end_snapshot = ? "
            "WHERE table_id = ? AND data_file_id = ? "
            "AND begin_snapshot <= ? "
            "AND (end_snapshot IS NULL OR end_snapshot > ?)",
            [new_snap, table_id, data_file_id, snapshot_id, snapshot_id],
        )

        delete_table = pa.table({
            "file_path": pa.array([abs_data_path] * len(all_positions), type=pa.string()),
            "pos": pa.array(all_positions, type=pa.int64()),
        })
        delete_file_name = f"ducklake-{_uuid7()}-delete.parquet"
        delete_file_path = storage.join_path(base_dir, delete_file_name)
        storage.write_parquet(delete_table, delete_file_path)

        delete_file_size = storage.get_file_size(delete_file_path)
        delete_footer_size = _read_parquet_footer_size(delete_file_path)

        con.execute(
            "INSERT INTO ducklake_delete_file "
            "(delete_file_id, table_id, begin_snapshot, end_snapshot, "
            "data_file_id, path, path_is_relative, format, delete_count, "
            "file_size_bytes, footer_size, encryption_key) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?, 'parquet', ?, ?, ?, NULL)",
            [
                delete_file_id, table_id, new_snap, data_file_id,
                delete_file_name, True, len(all_positions),
                delete_file_size, delete_footer_size,
            ],
        )

    @_retryable
    def delete_data(
        self,
        predicate: Callable[[pa.Table], pa.ChunkedArray],
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Delete rows matching a predicate from a table.

        The predicate is a callable that takes a ``pa.Table`` and returns
        a ``pa.ChunkedArray`` of booleans.

        Returns the number of deleted rows.
        """
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )
        self._track_table_write(table_id, "delete")

        columns = self._get_columns_for_table(table_id, snap_id)
        data_files = self._get_active_data_files(table_id, snap_id)

        inlined_count = self._get_inlined_active_row_count(table_id, snap_id)

        if not data_files and inlined_count == 0:
            return 0

        pending_deletes: list[tuple[int, str, list[int]]] = []
        total_deleted = 0

        if data_files:
            base = self.data_path
            if schema_path_rel:
                base = storage.join_path(base, schema_path)
            else:
                base = schema_path
            if table_path_rel:
                base = storage.join_path(base, table_path)
            else:
                base = table_path
            storage.makedirs(base, exist_ok=True)

            for data_file_id, rel_path, path_is_rel, record_count, row_id_start in data_files:
                abs_path = self._resolve_file_path(
                    rel_path, path_is_rel,
                    table_path, table_path_rel,
                    schema_path, schema_path_rel,
                )
                raw_df = storage.read_parquet(abs_path)
                already_deleted = self._get_active_delete_positions(
                    data_file_id, table_id, snap_id,
                    table_path, table_path_rel, schema_path, schema_path_rel,
                )
                mask = predicate(raw_df)
                mask_list = mask.to_pylist()
                positions = [
                    i for i, v in enumerate(mask_list)
                    if v and i not in already_deleted
                ]
                if positions:
                    pending_deletes.append((data_file_id, abs_path, positions))
                    total_deleted += len(positions)

        has_inlined_data = inlined_count > 0

        if total_deleted == 0 and not has_inlined_data:
            return 0

        current_file_id = next_file_id
        new_next_file_id = next_file_id + len(pending_deletes)

        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        if pending_deletes:
            base = self.data_path
            if schema_path_rel:
                base = storage.join_path(base, schema_path)
            else:
                base = schema_path
            if table_path_rel:
                base = storage.join_path(base, table_path)
            else:
                base = table_path

            for data_file_id, abs_data_path, positions in pending_deletes:
                delete_file_id = current_file_id
                current_file_id += 1

                self._write_cumulative_delete_file(
                    data_file_id, table_id, new_snap, snap_id,
                    positions, abs_data_path, base, delete_file_id,
                    table_path, table_path_rel, schema_path, schema_path_rel,
                )

        inlined_deleted = 0
        if has_inlined_data:
            inlined_deleted = self._delete_inlined_rows(
                table_id, predicate, snap_id, new_snap, columns,
            )
            total_deleted += inlined_deleted

        if total_deleted == 0:
            self._commit_metadata()
            return 0

        self._record_change(new_snap, f"deleted_from_table:{table_id}")
        self._commit_metadata()
        return total_deleted

    # ------------------------------------------------------------------
    # UPDATE (delete + insert in a single snapshot)
    # ------------------------------------------------------------------

    def _get_inlined_matched_rows(
        self,
        table_id: int,
        predicate: Callable[[pa.Table], pa.ChunkedArray],
        snapshot_id: int,
        columns: list[tuple[int, str, str, int | None]],
    ) -> pa.Table | None:
        """Read inlined rows matching a predicate. Returns Arrow table or None."""
        con = self._connect()
        try:
            inlined_tables = con.execute(
                "SELECT table_name FROM ducklake_inlined_data_tables "
                "WHERE table_id = ?",
                [table_id],
            ).fetchall()
        except Exception:
            return None

        col_names = [c[1] for c in columns]
        all_matched: list[pa.Table] = []

        for (tbl_name,) in inlined_tables:
            safe = tbl_name.replace('"', '""')
            cols_sql = '"row_id", ' + ", ".join(
                f'"{c.replace(chr(34), chr(34) + chr(34))}"' for c in col_names
            )
            try:
                rows = con.execute(
                    f'SELECT {cols_sql} FROM "{safe}" '
                    f"WHERE ? >= begin_snapshot "
                    f"AND (? < end_snapshot OR end_snapshot IS NULL)",
                    [snapshot_id, snapshot_id],
                ).fetchall()
            except Exception:
                continue

            if not rows:
                continue

            data = {name: [r[i + 1] for r in rows] for i, name in enumerate(col_names)}
            inline_table = pa.table(data)

            mask = predicate(inline_table)
            matched = inline_table.filter(mask)
            if len(matched) > 0:
                all_matched.append(matched)

        if not all_matched:
            return None
        return pa.concat_tables(all_matched) if len(all_matched) > 1 else all_matched[0]

    @_retryable
    def update_data(
        self,
        updates: dict[str, Any],
        predicate: Callable[[pa.Table], pa.ChunkedArray],
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Update rows matching a predicate.

        ``updates`` maps column names to either literal values or
        ``Callable[[pa.Table], pa.ChunkedArray]`` for computed columns.
        ``predicate`` is ``Callable[[pa.Table], pa.ChunkedArray]``
        returning a boolean mask.

        Returns the number of rows updated.
        """
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )
        self._track_table_write(table_id, "update")

        columns = self._get_columns_for_table(table_id, snap_id)
        data_files = self._get_active_data_files(table_id, snap_id)
        inlined_count = self._get_inlined_active_row_count(table_id, snap_id)

        if not data_files and inlined_count == 0:
            return 0

        base = self.data_path
        if schema_path_rel:
            base = storage.join_path(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = storage.join_path(base, table_path)
        else:
            base = table_path
        storage.makedirs(base, exist_ok=True)

        pending_deletes: list[tuple[int, str, list[int]]] = []
        matched_dfs: list[pa.Table] = []
        total_updated = 0

        for data_file_id, rel_path, path_is_rel, record_count, row_id_start in data_files:
            abs_path = self._resolve_file_path(
                rel_path, path_is_rel,
                table_path, table_path_rel,
                schema_path, schema_path_rel,
            )
            raw_df = storage.read_parquet(abs_path)
            already_deleted = self._get_active_delete_positions(
                data_file_id, table_id, snap_id,
                table_path, table_path_rel, schema_path, schema_path_rel,
            )
            mask = predicate(raw_df)
            mask_list = mask.to_pylist()
            positions = [
                i for i, v in enumerate(mask_list)
                if v and i not in already_deleted
            ]
            if positions:
                pending_deletes.append((data_file_id, abs_path, positions))
                active_matched = raw_df.take(pa.array(positions))
                matched_dfs.append(active_matched)
                total_updated += len(positions)

        inlined_matched = None
        if inlined_count > 0:
            inlined_matched = self._get_inlined_matched_rows(
                table_id, predicate, snap_id, columns,
            )
            if inlined_matched is not None:
                total_updated += len(inlined_matched)
                matched_dfs.append(inlined_matched)

        if total_updated == 0:
            return 0

        all_matched = pa.concat_tables(matched_dfs) if len(matched_dfs) > 1 else matched_dfs[0]

        # Apply updates
        for col_name, value in updates.items():
            if callable(value):
                new_col = value(all_matched)
            else:
                col_type = all_matched.schema.field(col_name).type
                new_col = pa.array([value] * len(all_matched), type=col_type)
            idx = all_matched.schema.get_field_index(col_name)
            all_matched = all_matched.set_column(idx, col_name, new_col)

        updated_df = all_matched

        # Check for active partition spec
        part_id = self._get_active_partition(table_id, snap_id)

        if part_id is not None:
            part_cols = self._get_partition_columns(part_id, table_id)
            col_id_to_name: dict[int, str] = {c[0]: c[1] for c in columns}
            part_col_names = [col_id_to_name[col_id] for _, col_id, _ in part_cols]
            part_key_indices = [key_index for key_index, _, _ in part_cols]

            group_list = _group_by_columns(updated_df, part_col_names)
            n_data_files = len(group_list)
        else:
            n_data_files = 1
            group_list = []
            part_col_names = []
            part_key_indices = []

        n_delete_files = len(pending_deletes)
        first_data_file_id = next_file_id + n_delete_files
        new_next_file_id = first_data_file_id + n_data_files

        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        current_file_id = next_file_id
        for data_file_id, abs_data_path, positions in pending_deletes:
            delete_file_id = current_file_id
            current_file_id += 1

            self._write_cumulative_delete_file(
                data_file_id, table_id, new_snap, snap_id,
                positions, abs_data_path, base, delete_file_id,
                table_path, table_path_rel, schema_path, schema_path_rel,
            )

        if inlined_matched is not None and len(inlined_matched) > 0:
            self._delete_inlined_rows(
                table_id, predicate, snap_id, new_snap, columns,
            )

        existing_stats = self._get_table_stats(table_id)
        row_id_start_new = existing_stats[1] if existing_stats else 0

        mapping_id = self._register_name_mapping(table_id, columns)

        # Get sort keys for sorting updated data
        update_sort_keys = self._get_active_sort_keys(table_id, snap_id)

        if part_id is not None:
            total_file_size = 0
            current_data_file_id = first_data_file_id
            current_row_id = row_id_start_new

            for group_key, group_df in group_list:
                # Sort partition group by sort keys if defined
                if update_sort_keys:
                    group_df = self._sort_table_by_keys(group_df, update_sort_keys)

                partition_values = [str(v) for v in group_key]
                hive_subdir = self._build_hive_path(part_col_names, partition_values)
                partition_dir = storage.join_path(base, hive_subdir)
                storage.makedirs(partition_dir, exist_ok=True)

                file_name = f"ducklake-{_uuid7()}.parquet"
                file_path = storage.join_path(partition_dir, file_name)
                storage.write_parquet(group_df, file_path)

                file_size = storage.get_file_size(file_path)
                footer_size = _read_parquet_footer_size(file_path)
                record_count = len(group_df)
                rel_path = f"{hive_subdir}/{file_name}"

                self._register_data_file(
                    current_data_file_id, table_id, new_snap, rel_path,
                    record_count, file_size, footer_size, current_row_id,
                    part_id, mapping_id,
                )
                self._register_partition_values(
                    current_data_file_id, table_id, part_key_indices, partition_values,
                )

                col_stats = self._compute_file_column_stats(group_df, columns)
                self._register_file_column_stats(current_data_file_id, table_id, col_stats)

                total_file_size += file_size
                current_data_file_id += 1
                current_row_id += record_count

            update_file_size = total_file_size
        else:
            # Sort by sort keys if defined
            if update_sort_keys:
                updated_df = self._sort_table_by_keys(updated_df, update_sort_keys)

            file_name = f"ducklake-{_uuid7()}.parquet"
            file_path = storage.join_path(base, file_name)
            storage.write_parquet(updated_df, file_path)

            file_size = storage.get_file_size(file_path)
            footer_size = _read_parquet_footer_size(file_path)

            self._register_data_file(
                first_data_file_id, table_id, new_snap, file_name,
                total_updated, file_size, footer_size, row_id_start_new,
                None, mapping_id,
            )

            col_stats = self._compute_file_column_stats(updated_df, columns)
            self._register_file_column_stats(first_data_file_id, table_id, col_stats)
            update_file_size = file_size

        self._update_table_stats(table_id, total_updated, update_file_size)

        full_col_stats = self._compute_file_column_stats(updated_df, columns)
        self._update_table_column_stats(table_id, columns, full_col_stats)

        self._record_change(
            new_snap,
            f"inserted_into_table:{table_id},deleted_from_table:{table_id}",
        )

        self._commit_metadata()
        return total_updated

    # ------------------------------------------------------------------
    # MERGE (match on keys, delete matched + insert)
    # ------------------------------------------------------------------

    def _read_all_inlined_active_rows(
        self,
        table_id: int,
        snapshot_id: int,
        columns: list[tuple[int, str, str, int | None]],
    ) -> pa.Table | None:
        """Read all active inlined rows as an Arrow table (no predicate filter)."""
        con = self._connect()
        try:
            inlined_tables = con.execute(
                "SELECT table_name FROM ducklake_inlined_data_tables "
                "WHERE table_id = ?",
                [table_id],
            ).fetchall()
        except Exception:
            return None

        col_names = [c[1] for c in columns]
        all_tables: list[pa.Table] = []

        for (tbl_name,) in inlined_tables:
            safe = tbl_name.replace('"', '""')
            cols_sql = ", ".join(
                f'"{c.replace(chr(34), chr(34) + chr(34))}"' for c in col_names
            )
            try:
                rows = con.execute(
                    f'SELECT {cols_sql} FROM "{safe}" '
                    f"WHERE ? >= begin_snapshot "
                    f"AND (? < end_snapshot OR end_snapshot IS NULL)",
                    [snapshot_id, snapshot_id],
                ).fetchall()
            except Exception:
                continue

            if rows:
                data = {
                    name: [r[i] for r in rows]
                    for i, name in enumerate(col_names)
                }
                all_tables.append(pa.table(data))

        if not all_tables:
            return None
        return pa.concat_tables(all_tables) if len(all_tables) > 1 else all_tables[0]

    @staticmethod
    def _build_key_match_predicate(
        on: list[str],
        matched_keys_table: pa.Table,
    ) -> Callable[[pa.Table], pa.ChunkedArray]:
        """Build an Arrow predicate matching rows whose key columns are in *matched_keys_table*."""
        if len(on) == 1:
            key_col = on[0]
            key_values = matched_keys_table.column(key_col)

            def apply_single(table: pa.Table) -> pa.ChunkedArray:
                return pc.is_in(table.column(key_col), key_values)

            return apply_single

        matched_keys_set: set[tuple] = set()
        mcols = [matched_keys_table.column(col) for col in on]
        for i in range(len(matched_keys_table)):
            matched_keys_set.add(tuple(c[i].as_py() for c in mcols))

        def apply_multi(table: pa.Table) -> pa.ChunkedArray:
            tcols = [table.column(col) for col in on]
            mask = [
                tuple(c[i].as_py() for c in tcols) in matched_keys_set
                for i in range(len(table))
            ]
            return pa.chunked_array([pa.array(mask)])

        return apply_multi

    @_retryable
    def merge_data(
        self,
        source_df: pa.Table,
        table_name: str,
        on: str | list[str],
        *,
        when_matched_update: dict[str, Any] | bool | None = None,
        when_not_matched_insert: bool = True,
        schema_name: str = "main",
    ) -> tuple[int, int]:
        """
        MERGE *source_df* into an existing table.

        Matches rows on the *on* key columns, optionally updates matched
        target rows, and optionally inserts unmatched source rows.

        Parameters
        ----------
        when_matched_update
            - ``None``: matched target rows are left untouched.
            - ``True``: replace matched target rows with source rows.
            - ``dict``: update matched target rows with these values
              (literal or ``Callable[[pa.Table], pa.ChunkedArray]``).
        when_not_matched_insert
            If True (default), source rows that have no match in the
            target are inserted.

        Returns ``(rows_updated, rows_inserted)``.
        """
        if isinstance(on, str):
            on = [on]

        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )
        self._track_table_write(table_id, "update")

        columns = self._get_columns_for_table(table_id, snap_id)
        col_names = [c[1] for c in columns]
        data_files = self._get_active_data_files(table_id, snap_id)
        inlined_count = self._get_inlined_active_row_count(table_id, snap_id)

        source_keys = _unique_rows(_select_columns(source_df, on))

        # ----------------------------------------------------------
        # Phase 1: find matched target rows in Parquet files
        # ----------------------------------------------------------
        pending_deletes: list[tuple[int, str, list[int]]] = []
        matched_target_dfs: list[pa.Table] = []
        all_target_key_dfs: list[pa.Table] = []

        for data_file_id, rel_path, path_is_rel, _rc, _rid in data_files:
            abs_path = self._resolve_file_path(
                rel_path, path_is_rel,
                table_path, table_path_rel,
                schema_path, schema_path_rel,
            )
            raw_df = storage.read_parquet(abs_path)
            already_deleted = self._get_active_delete_positions(
                data_file_id, table_id, snap_id,
                table_path, table_path_rel, schema_path, schema_path_rel,
            )

            # Add row index
            n = len(raw_df)
            idx_col = pa.array(list(range(n)), type=pa.int64())
            file_df = raw_df.append_column("__merge_idx__", idx_col)

            if already_deleted:
                keep_mask = [i not in already_deleted for i in range(n)]
                file_df = file_df.filter(pa.array(keep_mask))

            active_df = file_df.drop_columns(["__merge_idx__"])
            all_target_key_dfs.append(_select_columns(active_df, on))

            if when_matched_update is not None:
                matched = _semi_join(file_df, source_keys, on)
                if len(matched) > 0:
                    positions = sorted(matched.column("__merge_idx__").to_pylist())
                    pending_deletes.append((data_file_id, abs_path, positions))
                    matched_target_dfs.append(matched.drop_columns(["__merge_idx__"]))

        # ----------------------------------------------------------
        # Phase 2: find matched target rows in inlined data
        # ----------------------------------------------------------
        inlined_matched_df: pa.Table | None = None
        if inlined_count > 0:
            inlined_df = self._read_all_inlined_active_rows(
                table_id, snap_id, columns,
            )
            if inlined_df is not None and len(inlined_df) > 0:
                all_target_key_dfs.append(_select_columns(inlined_df, on))
                if when_matched_update is not None:
                    inlined_matched = _semi_join(inlined_df, source_keys, on)
                    if len(inlined_matched) > 0:
                        inlined_matched_df = inlined_matched
                        matched_target_dfs.append(inlined_matched)

        # ----------------------------------------------------------
        # Phase 3: counts and early exit
        # ----------------------------------------------------------
        total_updated = sum(len(p) for _, _, p in pending_deletes)
        if inlined_matched_df is not None:
            total_updated += len(inlined_matched_df)

        if when_not_matched_insert:
            if all_target_key_dfs:
                all_target_keys = _unique_rows(
                    pa.concat_tables(all_target_key_dfs)
                )
                unmatched_source = _anti_join(source_df, all_target_keys, on)
            else:
                unmatched_source = source_df
        else:
            unmatched_source = _empty_like(source_df)

        total_inserted = len(unmatched_source)

        if total_updated == 0 and total_inserted == 0:
            return (0, 0)

        # ----------------------------------------------------------
        # Phase 4: build rows to insert
        # ----------------------------------------------------------
        rows_to_insert_parts: list[pa.Table] = []

        if when_matched_update is not None and matched_target_dfs:
            all_matched = (
                pa.concat_tables(matched_target_dfs)
                if len(matched_target_dfs) > 1
                else matched_target_dfs[0]
            )
            if when_matched_update is True:
                matched_keys = _select_columns(all_matched, on)
                updated_rows = _select_columns(
                    _inner_join(matched_keys, source_df, on), col_names,
                )
                rows_to_insert_parts.append(updated_rows)
            elif isinstance(when_matched_update, dict):
                for col_name, value in when_matched_update.items():
                    if callable(value):
                        new_col = value(all_matched)
                    else:
                        col_type = all_matched.schema.field(col_name).type
                        new_col = pa.array(
                            [value] * len(all_matched), type=col_type,
                        )
                    idx = all_matched.schema.get_field_index(col_name)
                    all_matched = all_matched.set_column(idx, col_name, new_col)
                rows_to_insert_parts.append(all_matched)

        if total_inserted > 0:
            rows_to_insert_parts.append(_select_columns(unmatched_source, col_names))

        if not rows_to_insert_parts:
            return (0, 0)

        insert_df = (
            pa.concat_tables(rows_to_insert_parts)
            if len(rows_to_insert_parts) > 1
            else rows_to_insert_parts[0]
        )

        # ----------------------------------------------------------
        # Phase 5: build output dir
        # ----------------------------------------------------------
        base = self.data_path
        if schema_path_rel:
            base = storage.join_path(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = storage.join_path(base, table_path)
        else:
            base = table_path
        storage.makedirs(base, exist_ok=True)

        # ----------------------------------------------------------
        # Phase 6: allocate IDs and create snapshot
        # ----------------------------------------------------------
        n_delete_files = len(pending_deletes)
        n_data_files = 1 if len(insert_df) > 0 else 0

        part_id = self._get_active_partition(table_id, snap_id)
        if part_id is not None and n_data_files > 0:
            part_cols = self._get_partition_columns(part_id, table_id)
            col_id_to_name: dict[int, str] = {c[0]: c[1] for c in columns}
            part_col_names = [col_id_to_name[cid] for _, cid, _ in part_cols]
            part_key_indices = [ki for ki, _, _ in part_cols]
            group_list: list[tuple[tuple, pa.Table]] = _group_by_columns(
                insert_df, part_col_names,
            )
            n_data_files = len(group_list)
        else:
            part_col_names = []
            part_key_indices = []
            group_list = []

        first_data_file_id = next_file_id + n_delete_files
        new_next_file_id = first_data_file_id + n_data_files

        new_snap = self._create_snapshot(
            schema_ver, next_cat_id, new_next_file_id,
        )

        # ----------------------------------------------------------
        # Phase 7: write delete files for matched Parquet rows
        # ----------------------------------------------------------
        current_file_id = next_file_id
        for data_file_id, abs_data_path, positions in pending_deletes:
            delete_file_id = current_file_id
            current_file_id += 1

            self._write_cumulative_delete_file(
                data_file_id, table_id, new_snap, snap_id,
                positions, abs_data_path, base, delete_file_id,
                table_path, table_path_rel, schema_path, schema_path_rel,
            )

        if inlined_matched_df is not None and len(inlined_matched_df) > 0:
            key_pred = self._build_key_match_predicate(
                on, _unique_rows(_select_columns(inlined_matched_df, on)),
            )
            self._delete_inlined_rows(
                table_id, key_pred, snap_id, new_snap, columns,
            )

        # ----------------------------------------------------------
        # Phase 8: write new data file(s)
        # ----------------------------------------------------------
        existing_stats = self._get_table_stats(table_id)
        row_id_start = existing_stats[1] if existing_stats else 0
        mapping_id = self._register_name_mapping(table_id, columns)

        # Get sort keys for sorting merged data
        merge_sort_keys = self._get_active_sort_keys(table_id, snap_id)

        if part_id is not None and group_list:
            total_file_size = 0
            current_data_file_id = first_data_file_id
            current_row_id = row_id_start

            for gk, gdf in group_list:
                # Sort partition group by sort keys if defined
                if merge_sort_keys:
                    gdf = self._sort_table_by_keys(gdf, merge_sort_keys)

                partition_values = [str(v) for v in gk]
                hive_subdir = self._build_hive_path(
                    part_col_names, partition_values,
                )
                partition_dir = storage.join_path(base, hive_subdir)
                storage.makedirs(partition_dir, exist_ok=True)

                file_name = f"ducklake-{_uuid7()}.parquet"
                file_path = storage.join_path(partition_dir, file_name)
                storage.write_parquet(gdf, file_path)

                file_size = storage.get_file_size(file_path)
                footer_size = _read_parquet_footer_size(file_path)
                rel_path = f"{hive_subdir}/{file_name}"

                self._register_data_file(
                    current_data_file_id, table_id, new_snap, rel_path,
                    len(gdf), file_size, footer_size, current_row_id,
                    part_id, mapping_id,
                )
                self._register_partition_values(
                    current_data_file_id, table_id,
                    part_key_indices, partition_values,
                )
                cs = self._compute_file_column_stats(gdf, columns)
                self._register_file_column_stats(
                    current_data_file_id, table_id, cs,
                )

                total_file_size += file_size
                current_data_file_id += 1
                current_row_id += len(gdf)

            self._update_table_stats(table_id, len(insert_df), total_file_size)
        elif len(insert_df) > 0:
            # Sort by sort keys if defined
            if merge_sort_keys:
                insert_df = self._sort_table_by_keys(insert_df, merge_sort_keys)

            file_name = f"ducklake-{_uuid7()}.parquet"
            file_path = storage.join_path(base, file_name)
            storage.write_parquet(insert_df, file_path)

            file_size = storage.get_file_size(file_path)
            footer_size = _read_parquet_footer_size(file_path)

            self._register_data_file(
                first_data_file_id, table_id, new_snap, file_name,
                len(insert_df), file_size, footer_size, row_id_start,
                None, mapping_id,
            )

            col_stats = self._compute_file_column_stats(insert_df, columns)
            self._register_file_column_stats(
                first_data_file_id, table_id, col_stats,
            )

            self._update_table_stats(table_id, len(insert_df), file_size)

        full_col_stats = self._compute_file_column_stats(insert_df, columns)
        self._update_table_column_stats(table_id, columns, full_col_stats)

        changes: list[str] = []
        if total_updated > 0:
            changes.append(f"deleted_from_table:{table_id}")
        if total_updated > 0 or total_inserted > 0:
            changes.append(f"inserted_into_table:{table_id}")
        self._record_change(new_snap, ",".join(changes))

        self._commit_metadata()
        return (total_updated, total_inserted)

    # ------------------------------------------------------------------
    # ALTER TABLE: ADD COLUMN
    # ------------------------------------------------------------------

    def _get_max_column_id(self, table_id: int) -> int:
        """Return the max column_id for a table, or 0 if no columns exist."""
        con = self._connect()
        row = con.execute(
            "SELECT COALESCE(MAX(column_id), 0) FROM ducklake_column "
            "WHERE table_id = ?",
            [table_id],
        ).fetchone()
        return row[0]

    def _get_max_column_order(self, table_id: int, snapshot_id: int) -> int:
        """Return the max column_order for active top-level columns."""
        con = self._connect()
        row = con.execute(
            "SELECT COALESCE(MAX(column_order), 0) FROM ducklake_column "
            "WHERE table_id = ? AND parent_column IS NULL "
            "AND begin_snapshot <= ? AND (end_snapshot IS NULL OR end_snapshot > ?)",
            [table_id, snapshot_id, snapshot_id],
        ).fetchone()
        return row[0]

    @_retryable
    def add_column(
        self,
        table_name: str,
        column_name: str,
        arrow_dtype: pa.DataType,
        *,
        default: Any = None,
        schema_name: str = "main",
    ) -> None:
        """Add a new column to an existing table."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise TableNotFoundError(msg)
        self._track_table_write(table_id, "ddl")

        columns = self._get_columns_for_table(table_id, snap_id)
        for _, col_name, _, _ in columns:
            if col_name == column_name:
                msg = f"Column '{column_name}' already exists in '{schema_name}.{table_name}'"
                raise ValueError(msg)

        max_col_id = self._get_max_column_id(table_id)
        max_col_order = self._get_max_column_order(table_id, snap_id)
        new_col_id = max_col_id + 1
        new_col_order = max_col_order + 1

        duckdb_type = arrow_type_to_duckdb(arrow_dtype)
        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        default_str = str(default) if default is not None else None

        con.execute(
            "INSERT INTO ducklake_column "
            "(column_id, begin_snapshot, end_snapshot, table_id, column_order, "
            "column_name, column_type, initial_default, default_value, "
            "nulls_allowed, parent_column" + (
                ", default_value_type, default_value_dialect) "
                "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, NULL, 'literal', NULL)"
                if self._is_v04 else
                ") VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, NULL)"
            ),
            [
                new_col_id, new_snap, table_id, new_col_order,
                column_name, duckdb_type, default_str, default_str, True,
            ],
        )

        self._insert_schema_version(new_snap, new_schema_ver, table_id)

        self._record_change(new_snap, f"altered_table:{table_id}")

        self._commit_metadata()

    # ------------------------------------------------------------------
    # ALTER TABLE: RENAME COLUMN
    # ------------------------------------------------------------------

    @_retryable
    def rename_column(
        self,
        table_name: str,
        old_column_name: str,
        new_column_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Rename a column in an existing table."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise TableNotFoundError(msg)
        self._track_table_write(table_id, "ddl")

        columns = self._get_columns_for_table(table_id, snap_id)
        target_col: tuple[int, str, str, int | None] | None = None
        for col_id, col_name, col_type, parent in columns:
            if col_name == old_column_name:
                target_col = (col_id, col_name, col_type, parent)
            if col_name == new_column_name:
                msg = f"Column '{new_column_name}' already exists in '{schema_name}.{table_name}'"
                raise ValueError(msg)

        if target_col is None:
            msg = f"Column '{old_column_name}' not found in '{schema_name}.{table_name}'"
            raise ValueError(msg)

        col_id, _old_name, col_type, parent_column = target_col

        row = con.execute(
            "SELECT column_order, nulls_allowed, initial_default, default_value "
            "FROM ducklake_column "
            "WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
            [table_id, col_id],
        ).fetchone()
        col_order = row[0]
        nulls_allowed = row[1]
        initial_default = row[2]
        default_value = row[3]

        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        con.execute(
            "UPDATE ducklake_column SET end_snapshot = ? "
            "WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
            [new_snap, table_id, col_id],
        )

        con.execute(
            "INSERT INTO ducklake_column "
            "(column_id, begin_snapshot, end_snapshot, table_id, column_order, "
            "column_name, column_type, initial_default, default_value, "
            "nulls_allowed, parent_column" + (
                ", default_value_type, default_value_dialect) "
                "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, 'literal', NULL)"
                if self._is_v04 else
                ") VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            [
                col_id, new_snap, table_id, col_order,
                new_column_name, col_type, initial_default, default_value,
                nulls_allowed, parent_column,
            ],
        )

        self._insert_schema_version(new_snap, new_schema_ver, table_id)

        self._record_change(new_snap, f"altered_table:{table_id}")

        self._commit_metadata()

    # ------------------------------------------------------------------
    # ALTER TABLE: SET COLUMN TYPE
    # ------------------------------------------------------------------

    @_retryable
    def set_column_type(
        self,
        table_name: str,
        column_name: str,
        new_type: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Change the type of a column in an existing table.

        Existing Parquet files keep their original types; the reader must
        cast values when reading files written with an older column type.

        Parameters
        ----------
        table_name
            Name of the table to alter.
        column_name
            Name of the column whose type to change.
        new_type
            DuckDB type string for the new column type (e.g. ``"BIGINT"``).
        schema_name
            Schema name (default: ``"main"``).
        """
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise TableNotFoundError(msg)
        self._track_table_write(table_id, "ddl")

        columns = self._get_columns_for_table(table_id, snap_id)
        target_col: tuple[int, str, str, int | None] | None = None
        for col_id, col_name, col_type, parent in columns:
            if col_name == column_name:
                target_col = (col_id, col_name, col_type, parent)
                break

        if target_col is None:
            msg = f"Column '{column_name}' not found in '{schema_name}.{table_name}'"
            raise ValueError(msg)

        col_id, _name, _old_type, parent_column = target_col

        # Normalize the new type to DuckDB's internal representation
        # by round-tripping through Arrow
        arrow_type = duckdb_type_to_arrow(new_type)
        duckdb_new_type = arrow_type_to_duckdb(arrow_type)

        row = con.execute(
            "SELECT column_order, nulls_allowed, initial_default, default_value "
            "FROM ducklake_column "
            "WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
            [table_id, col_id],
        ).fetchone()
        col_order = row[0]
        nulls_allowed = row[1]
        initial_default = row[2]
        default_value = row[3]

        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        # End the old column entry
        con.execute(
            "UPDATE ducklake_column SET end_snapshot = ? "
            "WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
            [new_snap, table_id, col_id],
        )

        # Create new column entry with the same column_id but new type
        con.execute(
            "INSERT INTO ducklake_column "
            "(column_id, begin_snapshot, end_snapshot, table_id, column_order, "
            "column_name, column_type, initial_default, default_value, "
            "nulls_allowed, parent_column" + (
                ", default_value_type, default_value_dialect) "
                "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, 'literal', NULL)"
                if self._is_v04 else
                ") VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            [
                col_id, new_snap, table_id, col_order,
                column_name, duckdb_new_type, initial_default, default_value,
                nulls_allowed, parent_column,
            ],
        )

        self._insert_schema_version(new_snap, new_schema_ver, table_id)

        self._record_change(new_snap, f"altered_table:{table_id}")

        self._commit_metadata()

    # ------------------------------------------------------------------
    # DROP TABLE
    # ------------------------------------------------------------------

    def _end_all_columns(self, table_id: int, snapshot_id: int, new_snap: int) -> None:
        """Mark all active columns as ended at new_snap."""
        con = self._connect()
        con.execute(
            "UPDATE ducklake_column SET end_snapshot = ? "
            "WHERE table_id = ? AND begin_snapshot <= ? "
            "AND (end_snapshot IS NULL OR end_snapshot > ?)",
            [new_snap, table_id, snapshot_id, snapshot_id],
        )

    @_retryable
    def drop_table(
        self,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Drop a table from the catalog."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise TableNotFoundError(msg)
        self._track_table_write(table_id, "drop_table")

        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        con.execute(
            "UPDATE ducklake_table SET end_snapshot = ? "
            "WHERE table_id = ? AND end_snapshot IS NULL",
            [new_snap, table_id],
        )

        self._end_all_columns(table_id, snap_id, new_snap)
        self._end_all_data_files(table_id, snap_id, new_snap)
        self._end_all_delete_files(table_id, snap_id, new_snap)
        self._end_all_inlined_rows(table_id, snap_id, new_snap)

        self._insert_schema_version(new_snap, new_schema_ver, table_id)

        self._record_change(new_snap, f"dropped_table:{table_id}")

        self._commit_metadata()

    # ------------------------------------------------------------------
    # CREATE SCHEMA
    # ------------------------------------------------------------------

    def _schema_exists(
        self, schema_name: str, snapshot_id: int
    ) -> int | None:
        """Return schema_id if the schema exists at snapshot_id, else None."""
        con = self._connect()
        row = con.execute(
            "SELECT schema_id FROM ducklake_schema "
            "WHERE LOWER(schema_name) = LOWER(?) AND begin_snapshot <= ? "
            "AND (end_snapshot IS NULL OR end_snapshot > ?)",
            [schema_name, snapshot_id, snapshot_id],
        ).fetchone()
        return row[0] if row is not None else None

    @_retryable
    def create_schema(
        self,
        schema_name: str,
    ) -> int:
        """Create a new schema in the catalog. Returns the new schema_id."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        if self._schema_exists(schema_name, snap_id) is not None:
            msg = f"Schema '{schema_name}' already exists"
            raise ValueError(msg)

        schema_id = next_cat_id
        new_next_cat_id = next_cat_id + 1
        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, new_next_cat_id, next_file_id)

        schema_uuid = str(uuid.uuid4())
        schema_path = f"{schema_name}/"
        con.execute(
            "INSERT INTO ducklake_schema "
            "(schema_id, schema_uuid, begin_snapshot, end_snapshot, "
            "schema_name, path, path_is_relative) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?)",
            [schema_id, schema_uuid, new_snap, schema_name, schema_path, True],
        )

        self._insert_schema_version(new_snap, new_schema_ver, None)

        safe_schema = schema_name.replace('"', '""')
        self._record_change(new_snap, f'created_schema:"{safe_schema}"')

        self._commit_metadata()
        return schema_id

    # ------------------------------------------------------------------
    # DROP SCHEMA
    # ------------------------------------------------------------------

    def _get_tables_in_schema(
        self, schema_id: int, snapshot_id: int
    ) -> list[tuple[int, str]]:
        """Return [(table_id, table_name)] for active tables in a schema."""
        con = self._connect()
        rows = con.execute(
            "SELECT table_id, table_name FROM ducklake_table "
            "WHERE schema_id = ? AND begin_snapshot <= ? "
            "AND (end_snapshot IS NULL OR end_snapshot > ?)",
            [schema_id, snapshot_id, snapshot_id],
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    @_retryable
    def drop_schema(
        self,
        schema_name: str,
        *,
        cascade: bool = False,
    ) -> None:
        """Drop a schema from the catalog."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        schema_id = self._schema_exists(schema_name, snap_id)
        if schema_id is None:
            msg = f"Schema '{schema_name}' not found"
            raise SchemaNotFoundError(msg)

        tables = self._get_tables_in_schema(schema_id, snap_id)

        if tables and not cascade:
            table_names = ", ".join(f'"{t[1]}"' for t in tables)
            msg = (
                f"Cannot drop schema \"{schema_name}\" because there are "
                f"entries that depend on it: {table_names}. "
                f"Use cascade=True to drop all dependents."
            )
            raise ValueError(msg)

        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        changes: list[str] = []

        for table_id, _table_name in tables:
            con.execute(
                "UPDATE ducklake_table SET end_snapshot = ? "
                "WHERE table_id = ? AND end_snapshot IS NULL",
                [new_snap, table_id],
            )
            self._end_all_columns(table_id, snap_id, new_snap)
            self._end_all_data_files(table_id, snap_id, new_snap)
            self._end_all_delete_files(table_id, snap_id, new_snap)
            changes.append(f"dropped_table:{table_id}")

        con.execute(
            "UPDATE ducklake_schema SET end_snapshot = ? "
            "WHERE schema_id = ? AND end_snapshot IS NULL",
            [new_snap, schema_id],
        )
        changes.append(f"dropped_schema:{schema_id}")

        self._insert_schema_version(new_snap, new_schema_ver, None)

        schema_changes = [c for c in changes if c.startswith("dropped_schema")]
        table_changes = [c for c in changes if c.startswith("dropped_table")]
        ordered = schema_changes + table_changes
        self._record_change(new_snap, ",".join(ordered))

        self._commit_metadata()

    # ------------------------------------------------------------------
    # RENAME TABLE
    # ------------------------------------------------------------------

    @_retryable
    def rename_table(
        self,
        old_table_name: str,
        new_table_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Rename a table in the catalog."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id = self._table_exists(old_table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{old_table_name}' not found"
            raise TableNotFoundError(msg)
        self._track_table_write(table_id, "ddl")

        if self._table_exists(new_table_name, schema_name, snap_id) is not None:
            msg = f"Table '{schema_name}.{new_table_name}' already exists"
            raise ValueError(msg)

        row = con.execute(
            "SELECT table_uuid, schema_id, path, path_is_relative "
            "FROM ducklake_table "
            "WHERE table_id = ? AND end_snapshot IS NULL",
            [table_id],
        ).fetchone()
        table_uuid = row[0]
        schema_id = row[1]
        table_path = row[2]
        path_is_relative = row[3]

        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        con.execute(
            "UPDATE ducklake_table SET end_snapshot = ? "
            "WHERE table_id = ? AND end_snapshot IS NULL",
            [new_snap, table_id],
        )

        con.execute(
            "INSERT INTO ducklake_table "
            "(table_id, table_uuid, begin_snapshot, end_snapshot, schema_id, "
            "table_name, path, path_is_relative) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?, ?)",
            [table_id, table_uuid, new_snap, schema_id,
             new_table_name, table_path, path_is_relative],
        )

        self._insert_schema_version(new_snap, new_schema_ver, table_id)

        safe_schema = schema_name.replace('"', '""')
        safe_table = new_table_name.replace('"', '""')
        self._record_change(
            new_snap, f'created_table:"{safe_schema}"."{safe_table}"'
        )

        self._commit_metadata()

    # ------------------------------------------------------------------
    # ALTER TABLE: DROP COLUMN
    # ------------------------------------------------------------------

    def _end_descendant_columns(
        self, table_id: int, parent_id: int, new_snap: int
    ) -> None:
        """Recursively end all descendant columns of a parent column."""
        con = self._connect()
        children = con.execute(
            "SELECT column_id FROM ducklake_column "
            "WHERE table_id = ? AND parent_column = ? AND end_snapshot IS NULL",
            [table_id, parent_id],
        ).fetchall()
        for (child_id,) in children:
            con.execute(
                "UPDATE ducklake_column SET end_snapshot = ? "
                "WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
                [new_snap, table_id, child_id],
            )
            self._end_descendant_columns(table_id, child_id, new_snap)

    @_retryable
    def drop_column(
        self,
        table_name: str,
        column_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Drop a column from an existing table."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise TableNotFoundError(msg)
        self._track_table_write(table_id, "ddl")

        columns = self._get_columns_for_table(table_id, snap_id)
        target_col_id = None
        for col_id, col_name, _, _ in columns:
            if col_name == column_name:
                target_col_id = col_id
                break

        if target_col_id is None:
            msg = f"Column '{column_name}' not found in '{schema_name}.{table_name}'"
            raise ValueError(msg)

        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        con.execute(
            "UPDATE ducklake_column SET end_snapshot = ? "
            "WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
            [new_snap, table_id, target_col_id],
        )

        self._end_descendant_columns(table_id, target_col_id, new_snap)

        self._insert_schema_version(new_snap, new_schema_ver, table_id)

        self._record_change(new_snap, f"altered_table:{table_id}")

        self._commit_metadata()

    # ------------------------------------------------------------------
    # ALTER TABLE: SET PARTITIONED BY
    # ------------------------------------------------------------------

    def _get_active_partition(
        self, table_id: int, snapshot_id: int
    ) -> int | None:
        """Return the active partition_id for a table, or None."""
        con = self._connect()
        try:
            row = con.execute(
                "SELECT partition_id FROM ducklake_partition_info "
                "WHERE table_id = ? AND begin_snapshot <= ? "
                "AND (end_snapshot IS NULL OR end_snapshot > ?)",
                [table_id, snapshot_id, snapshot_id],
            ).fetchone()
        except Exception:
            return None
        return row[0] if row is not None else None

    def _get_partition_columns(
        self, partition_id: int, table_id: int
    ) -> list[tuple[int, int, str]]:
        """Return [(partition_key_index, column_id, transform)]."""
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT partition_key_index, column_id, transform "
                "FROM ducklake_partition_column "
                "WHERE partition_id = ? AND table_id = ? "
                "ORDER BY partition_key_index",
                [partition_id, table_id],
            ).fetchall()
        except Exception:
            return []
        return [(r[0], r[1], r[2]) for r in rows]

    @_retryable
    def set_partitioned_by(
        self,
        table_name: str,
        column_names: list[str],
        *,
        schema_name: str = "main",
    ) -> None:
        """Set identity-transform partitioning on an existing table."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise TableNotFoundError(msg)
        self._track_table_write(table_id, "ddl")

        columns = self._get_columns_for_table(table_id, snap_id)
        col_name_to_id: dict[str, int] = {c[1]: c[0] for c in columns}
        partition_col_ids: list[int] = []
        for name in column_names:
            if name not in col_name_to_id:
                msg = f"Column '{name}' not found in '{schema_name}.{table_name}'"
                raise ValueError(msg)
            partition_col_ids.append(col_name_to_id[name])

        partition_id = next_cat_id
        new_next_cat_id = next_cat_id + 1
        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, new_next_cat_id, next_file_id)

        con.execute(
            "INSERT INTO ducklake_partition_info "
            "(partition_id, table_id, begin_snapshot, end_snapshot) "
            "VALUES (?, ?, ?, NULL)",
            [partition_id, table_id, new_snap],
        )

        for key_index, col_id in enumerate(partition_col_ids):
            con.execute(
                "INSERT INTO ducklake_partition_column "
                "(partition_id, table_id, partition_key_index, column_id, transform) "
                "VALUES (?, ?, ?, ?, 'identity')",
                [partition_id, table_id, key_index, col_id],
            )

        self._insert_schema_version(new_snap, new_schema_ver, table_id)

        self._record_change(new_snap, f"altered_table:{table_id}")

        self._commit_metadata()

    # ------------------------------------------------------------------
    # ALTER TABLE: SET SORT KEYS
    # ------------------------------------------------------------------

    def _ensure_sort_tables(self) -> None:
        """Create ducklake_sort_info and ducklake_sort_expression if they don't exist."""
        con = self._connect()
        con.execute(
            "CREATE TABLE IF NOT EXISTS ducklake_sort_info ("
            "sort_id BIGINT, "
            "table_id BIGINT, "
            "begin_snapshot BIGINT, "
            "end_snapshot BIGINT)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS ducklake_sort_expression ("
            "sort_id BIGINT, "
            "table_id BIGINT, "
            "sort_key_index BIGINT, "
            "expression VARCHAR, "
            "dialect VARCHAR, "
            "sort_direction VARCHAR, "
            "null_order VARCHAR)"
        )

    def _sort_tables_exist(self) -> bool:
        """Check if sort tables exist without triggering Postgres rollback."""
        con = self._connect()
        try:
            return self._backend.table_exists(con, "ducklake_sort_info")
        except Exception:
            return False

    def _get_active_sort_keys(
        self, table_id: int, snapshot_id: int
    ) -> list[tuple[str, str, str]] | None:
        """Return active sort keys as [(column_name, direction, null_order)] or None.

        Returns None if no sort keys are defined for the table.
        """
        if not self._sort_tables_exist():
            return None
        con = self._connect()
        row = con.execute(
            "SELECT sort_id FROM ducklake_sort_info "
            "WHERE table_id = ? AND begin_snapshot <= ? "
            "AND (end_snapshot IS NULL OR end_snapshot > ?)",
            [table_id, snapshot_id, snapshot_id],
        ).fetchone()
        if row is None:
            return None

        sort_id = row[0]

        # Get column names from sort expressions
        columns = self._get_columns_for_table(table_id, snapshot_id)
        col_id_to_name: dict[str, str] = {str(c[0]): c[1] for c in columns}
        col_names_set = {c[1] for c in columns}

        try:
            rows = con.execute(
                "SELECT sort_key_index, expression, sort_direction, null_order "
                "FROM ducklake_sort_expression "
                "WHERE sort_id = ? AND table_id = ? "
                "ORDER BY sort_key_index",
                [sort_id, table_id],
            ).fetchall()
        except Exception:
            return None

        result = []
        for _idx, expr, direction, null_order in rows:
            # expression can be column name (v1.5) or column ID (legacy)
            if expr in col_names_set:
                col_name = expr
            else:
                col_name = col_id_to_name.get(expr)
            if col_name is None:
                continue
            direction = (direction or "ASC").upper()
            null_order = (null_order or "NULLS_LAST").upper()
            result.append((col_name, direction, null_order))

        return result if result else None

    def _sort_table_by_keys(
        self, df: pa.Table, sort_keys: list[tuple[str, str, str]]
    ) -> pa.Table:
        """Sort an Arrow table by the given sort keys.

        Each key is ``(column_name, direction, null_order)`` where
        *direction* is ``ASC``/``DESC`` and *null_order* is
        ``NULLS_FIRST``/``NULLS_LAST``.
        """
        sort_cols = []
        for col_name, direction, null_order in sort_keys:
            if col_name in df.schema.names:
                order = "ascending" if direction.upper() == "ASC" else "descending"
                sort_cols.append((col_name, order))
        if not sort_cols:
            return df
        indices = pc.sort_indices(df, sort_keys=sort_cols)
        return df.take(indices)

    def _maybe_sort(
        self, df: pa.Table, table_id: int, snapshot_id: int
    ) -> pa.Table:
        """Sort *df* by the table's sort keys if any are defined."""
        sort_keys = self._get_active_sort_keys(table_id, snapshot_id)
        if sort_keys:
            return self._sort_table_by_keys(df, sort_keys)
        return df

    @_retryable
    def set_sort_keys(
        self,
        table_name: str,
        sort_keys: list[str | tuple[str, str] | tuple[str, str, str]],
        *,
        schema_name: str = "main",
    ) -> None:
        """Set sort keys on an existing table.

        Equivalent to ``ALTER TABLE t SET SORTED BY (col1, col2 DESC, ...)``.

        *sort_keys* accepts:
        - ``"col"`` → ascending, nulls last
        - ``("col", "DESC")`` → descending, nulls last
        - ``("col", "ASC", "NULLS_FIRST")`` → ascending, nulls first

        Future writes will sort data by these columns before writing
        Parquet files, improving filter pushdown via row group statistics.
        """
        con = self._connect()
        self._ensure_sort_tables()

        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise TableNotFoundError(msg)
        self._track_table_write(table_id, "ddl")

        columns = self._get_columns_for_table(table_id, snap_id)
        col_names_set = {c[1] for c in columns}

        # Normalise sort_keys into (name, direction, null_order) triples
        normalised: list[tuple[str, str, str]] = []
        for key in sort_keys:
            if isinstance(key, str):
                name, direction, null_order = key, "ASC", "NULLS_LAST"
            elif len(key) == 2:
                name, direction = key[0], key[1].upper()
                null_order = "NULLS_LAST"
            else:
                name, direction, null_order = key[0], key[1].upper(), key[2].upper()

            if name not in col_names_set:
                msg = f"Column '{name}' not found in '{schema_name}.{table_name}'"
                raise ValueError(msg)
            if direction not in ("ASC", "DESC"):
                msg = f"Invalid sort direction '{direction}'; expected ASC or DESC"
                raise ValueError(msg)
            if null_order not in ("NULLS_FIRST", "NULLS_LAST"):
                msg = f"Invalid null order '{null_order}'; expected NULLS_FIRST or NULLS_LAST"
                raise ValueError(msg)
            normalised.append((name, direction, null_order))

        sort_id = next_cat_id
        new_next_cat_id = next_cat_id + 1
        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, new_next_cat_id, next_file_id)

        # End any existing sort info for this table
        try:
            con.execute(
                "UPDATE ducklake_sort_info SET end_snapshot = ? "
                "WHERE table_id = ? AND end_snapshot IS NULL",
                [new_snap, table_id],
            )
        except Exception:
            pass

        con.execute(
            "INSERT INTO ducklake_sort_info "
            "(sort_id, table_id, begin_snapshot, end_snapshot) "
            "VALUES (?, ?, ?, NULL)",
            [sort_id, table_id, new_snap],
        )

        for key_index, (col_name, direction, null_order) in enumerate(normalised):
            con.execute(
                "INSERT INTO ducklake_sort_expression "
                "(sort_id, table_id, sort_key_index, expression, dialect, "
                "sort_direction, null_order) "
                "VALUES (?, ?, ?, ?, 'duckdb', ?, ?)",
                [sort_id, table_id, key_index, col_name, direction, null_order],
            )

        self._insert_schema_version(new_snap, new_schema_ver, table_id)

        self._record_change(new_snap, f"altered_table:{table_id}")

        self._commit_metadata()

    @_retryable
    def reset_sort_keys(
        self,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Remove sort keys from a table.

        Equivalent to ``ALTER TABLE t RESET SORTED BY``.
        """
        con = self._connect()
        self._ensure_sort_tables()

        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise TableNotFoundError(msg)
        self._track_table_write(table_id, "ddl")

        # Check if there are active sort keys
        existing = self._get_active_sort_keys(table_id, snap_id)
        if existing is None:
            return  # Nothing to reset

        new_schema_ver = schema_ver + 1
        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        con.execute(
            "UPDATE ducklake_sort_info SET end_snapshot = ? "
            "WHERE table_id = ? AND end_snapshot IS NULL",
            [new_snap, table_id],
        )

        self._insert_schema_version(new_snap, new_schema_ver, table_id)

        self._record_change(new_snap, f"altered_table:{table_id}")
        self._commit_metadata()

    # ------------------------------------------------------------------
    # REWRITE DATA FILES (compaction)
    # ------------------------------------------------------------------

    @_retryable
    def rewrite_data_files(
        self,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        """Rewrite data files for compaction — merge small files, remove deleted rows.

        Reads all active data files for a table, respecting deletion
        vectors, and writes a single consolidated Parquet file (or one
        per partition).  Old data files and delete files are marked as
        ended and new files are registered — all within a single
        snapshot.

        Returns the new snapshot ID, or ``-1`` if no rewrite was needed
        (e.g., the table has zero or one active data file with no
        deletion vectors).
        """
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )
        self._track_table_write(table_id, "insert")

        columns = self._get_columns_for_table(table_id, snap_id)
        data_files = self._get_active_data_files(table_id, snap_id)

        # Check if any file has deletion vectors
        has_deletes = False
        for data_file_id, _path, _rel, _rc, _rid in data_files:
            positions = self._get_active_delete_positions(
                data_file_id, table_id, snap_id,
                table_path, table_path_rel, schema_path, schema_path_rel,
            )
            if positions:
                has_deletes = True
                break

        # No rewrite needed: single file without deletes (or no files)
        if len(data_files) <= 1 and not has_deletes:
            return -1

        # Build column rename mapping for schema evolution
        # Get column history: all column definitions across all snapshots
        col_history_rows = con.execute(
            "SELECT column_id, column_name, begin_snapshot, end_snapshot "
            "FROM ducklake_column "
            "WHERE table_id = ? AND parent_column IS NULL "
            "ORDER BY column_id, begin_snapshot",
            [table_id],
        ).fetchall()

        # Current column names (at latest snapshot)
        current_col_names = {c[1] for c in columns}
        current_col_map = {c[0]: c[1] for c in columns}  # col_id -> current_name

        # For each file, get its begin_snapshot
        file_begin_snaps = {}
        for data_file_id, _path, _rel, _rc, _rid in data_files:
            row = con.execute(
                "SELECT begin_snapshot FROM ducklake_data_file WHERE data_file_id = ?",
                [data_file_id],
            ).fetchone()
            if row:
                file_begin_snaps[data_file_id] = row[0]

        def _build_rename_map(file_begin_snap: int) -> dict[str, str]:
            """Build {physical_name -> current_name} for a file written at file_begin_snap."""
            rename: dict[str, str] = {}
            for col_id, current_name in current_col_map.items():
                # Find the column name at file_begin_snap
                for ch_id, ch_name, ch_begin, ch_end in col_history_rows:
                    if ch_id != col_id:
                        continue
                    if ch_begin <= file_begin_snap and (ch_end is None or ch_end > file_begin_snap):
                        if ch_name != current_name:
                            rename[ch_name] = current_name
                        break
            return rename

        # Read all active data, respecting deletion vectors
        all_dfs: list[pa.Table] = []
        for data_file_id, rel_path, path_is_rel, _rc, _rid in data_files:
            abs_path = self._resolve_file_path(
                rel_path, path_is_rel,
                table_path, table_path_rel,
                schema_path, schema_path_rel,
            )
            active_df = self._read_active_data_file(
                data_file_id, abs_path, table_id, snap_id,
                table_path, table_path_rel, schema_path, schema_path_rel,
            )
            if len(active_df) > 0:
                # Apply column rename mapping
                begin_snap = file_begin_snaps.get(data_file_id, 0)
                rename_map = _build_rename_map(begin_snap)
                if rename_map:
                    new_names = [rename_map.get(c, c) for c in active_df.column_names]
                    active_df = active_df.rename_columns(new_names)
                # Drop columns that are no longer in the current schema
                keep_cols = [c for c in active_df.column_names if c in current_col_names]
                if len(keep_cols) < len(active_df.column_names):
                    active_df = active_df.select(keep_cols)
                all_dfs.append(active_df)

        if all_dfs:
            combined = pa.concat_tables(all_dfs, promote_options="default") if len(all_dfs) > 1 else all_dfs[0]
        else:
            combined = pa.table(
                {c[1]: pa.array([], type=pa.string()) for c in columns}
            )

        total_records = len(combined)

        # Resolve the base directory for writing new files
        base = self.data_path
        if schema_path_rel:
            base = storage.join_path(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = storage.join_path(base, table_path)
        else:
            base = table_path
        storage.makedirs(base, exist_ok=True)

        # Check for partitioning
        part_id = self._get_active_partition(table_id, snap_id)

        if part_id is not None and total_records > 0:
            part_cols = self._get_partition_columns(part_id, table_id)
            col_id_to_name: dict[int, str] = {c[0]: c[1] for c in columns}
            part_col_names = [col_id_to_name[col_id] for _, col_id, _ in part_cols]
            part_key_indices = [ki for ki, _, _ in part_cols]
            group_list = _group_by_columns(combined, part_col_names)
            n_new_files = len(group_list)
        else:
            n_new_files = 1 if total_records > 0 else 0
            group_list = []
            part_col_names = []
            part_key_indices = []

        new_next_file_id = next_file_id + n_new_files
        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        # End all old data files and delete files
        self._end_all_data_files(table_id, snap_id, new_snap)
        self._end_all_delete_files(table_id, snap_id, new_snap)

        # Sort by sort keys if defined
        sort_keys = self._get_active_sort_keys(table_id, snap_id)

        if total_records > 0:
            mapping_id = self._register_name_mapping(table_id, columns)

            if part_id is not None and group_list:
                total_file_size = 0
                current_file_id = next_file_id
                current_row_id = 0

                for group_key, group_df in group_list:
                    if sort_keys:
                        group_df = self._sort_table_by_keys(group_df, sort_keys)

                    partition_values = [str(v) for v in group_key]
                    hive_subdir = self._build_hive_path(part_col_names, partition_values)
                    partition_dir = storage.join_path(base, hive_subdir)
                    storage.makedirs(partition_dir, exist_ok=True)

                    file_name = f"ducklake-{_uuid7()}.parquet"
                    file_path = storage.join_path(partition_dir, file_name)
                    storage.write_parquet(group_df, file_path)

                    file_size = storage.get_file_size(file_path)
                    footer_size = _read_parquet_footer_size(file_path)
                    record_count = len(group_df)
                    rel_path = f"{hive_subdir}/{file_name}"

                    self._register_data_file(
                        current_file_id, table_id, new_snap, rel_path,
                        record_count, file_size, footer_size, current_row_id,
                        part_id, mapping_id,
                    )
                    self._register_partition_values(
                        current_file_id, table_id, part_key_indices, partition_values,
                    )

                    col_stats = self._compute_file_column_stats(group_df, columns)
                    self._register_file_column_stats(current_file_id, table_id, col_stats)

                    total_file_size += file_size
                    current_file_id += 1
                    current_row_id += record_count
            else:
                # Non-partitioned: write a single consolidated file
                if sort_keys:
                    combined = self._sort_table_by_keys(combined, sort_keys)

                file_name = f"ducklake-{_uuid7()}.parquet"
                file_path = storage.join_path(base, file_name)
                storage.write_parquet(combined, file_path)

                file_size = storage.get_file_size(file_path)
                footer_size = _read_parquet_footer_size(file_path)

                data_file_id = next_file_id

                self._register_data_file(
                    data_file_id, table_id, new_snap, file_name,
                    total_records, file_size, footer_size, 0,
                    None, mapping_id,
                )

                col_stats = self._compute_file_column_stats(combined, columns)
                self._register_file_column_stats(data_file_id, table_id, col_stats)

                total_file_size = file_size

            # Rebuild table stats from scratch
            con.execute(
                "UPDATE ducklake_table_stats "
                "SET record_count = ?, next_row_id = ?, file_size_bytes = ? "
                "WHERE table_id = ?",
                [total_records, total_records, total_file_size, table_id],
            )

            # Rebuild column stats from scratch
            con.execute(
                "DELETE FROM ducklake_table_column_stats WHERE table_id = ?",
                [table_id],
            )
            full_col_stats = self._compute_file_column_stats(combined, columns)
            for col_id, _col_name, _vc, null_count, min_val, max_val, nan_int in full_col_stats:
                contains_null = bool(null_count and null_count > 0)
                con.execute(
                    "INSERT INTO ducklake_table_column_stats "
                    "(table_id, column_id, contains_null, contains_nan, "
                    "min_value, max_value, extra_stats) "
                    "VALUES (?, ?, ?, ?, ?, ?, NULL)",
                    [table_id, col_id, contains_null, nan_int, min_val, max_val],
                )
        else:
            # All rows were deleted — reset stats
            con.execute(
                "UPDATE ducklake_table_stats "
                "SET record_count = 0, next_row_id = 0, file_size_bytes = 0 "
                "WHERE table_id = ?",
                [table_id],
            )
            con.execute(
                "DELETE FROM ducklake_table_column_stats WHERE table_id = ?",
                [table_id],
            )

        self._record_change(
            new_snap,
            f"inserted_into_table:{table_id},deleted_from_table:{table_id}",
        )
        self._commit_metadata()
        return new_snap

    # ------------------------------------------------------------------
    # EXPIRE SNAPSHOTS
    # ------------------------------------------------------------------

    def expire_snapshots(
        self,
        *,
        older_than_snapshot: int | None = None,
        keep_last_n: int | None = None,
    ) -> int:
        """Expire old snapshots and clean up associated metadata."""
        if older_than_snapshot is not None and keep_last_n is not None:
            msg = "Cannot specify both older_than_snapshot and keep_last_n"
            raise ValueError(msg)
        if older_than_snapshot is None and keep_last_n is None:
            msg = "Must specify either older_than_snapshot or keep_last_n"
            raise ValueError(msg)

        con = self._connect()

        if keep_last_n is not None:
            if keep_last_n < 1:
                msg = "keep_last_n must be >= 1"
                raise ValueError(msg)
            rows = con.execute(
                "SELECT snapshot_id FROM ducklake_snapshot "
                "ORDER BY snapshot_id DESC"
            ).fetchall()
            all_ids = [r[0] for r in rows]
            if len(all_ids) <= keep_last_n:
                return 0
            older_than_snapshot = all_ids[keep_last_n - 1]

        assert older_than_snapshot is not None

        expire_rows = con.execute(
            "SELECT snapshot_id FROM ducklake_snapshot "
            "WHERE snapshot_id < ?",
            [older_than_snapshot],
        ).fetchall()
        expire_ids = [r[0] for r in expire_rows]

        if not expire_ids:
            return 0

        boundary = older_than_snapshot

        con.execute(
            "DELETE FROM ducklake_file_column_stats "
            "WHERE data_file_id IN ("
            "  SELECT data_file_id FROM ducklake_data_file "
            "  WHERE end_snapshot IS NOT NULL AND end_snapshot <= ?"
            ")",
            [boundary],
        )

        try:
            con.execute(
                "DELETE FROM ducklake_file_partition_value "
                "WHERE data_file_id IN ("
                "  SELECT data_file_id FROM ducklake_data_file "
                "  WHERE end_snapshot IS NOT NULL AND end_snapshot <= ?"
                ")",
                [boundary],
            )
        except Exception:
            pass

        con.execute(
            "DELETE FROM ducklake_data_file "
            "WHERE end_snapshot IS NOT NULL AND end_snapshot <= ?",
            [boundary],
        )

        con.execute(
            "DELETE FROM ducklake_delete_file "
            "WHERE end_snapshot IS NOT NULL AND end_snapshot <= ?",
            [boundary],
        )

        for sid in expire_ids:
            con.execute(
                "DELETE FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
                [sid],
            )

        for sid in expire_ids:
            con.execute(
                "DELETE FROM ducklake_snapshot WHERE snapshot_id = ?",
                [sid],
            )

        con.commit()
        return len(expire_ids)

    # ------------------------------------------------------------------
    # VACUUM — delete orphaned Parquet files
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Tags (table and column metadata)
    # ------------------------------------------------------------------

    def set_table_tag(
        self,
        table_name: str,
        key: str,
        value: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Set a tag on a table. Overwrites existing tag with same key."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise TableNotFoundError(msg)

        new_snap = self._create_snapshot(schema_ver, next_cat_id, next_file_id)

        # End any existing tag with the same key
        con.execute(
            "UPDATE ducklake_tag SET end_snapshot = ? "
            "WHERE object_id = ? AND key = ? AND end_snapshot IS NULL",
            [new_snap, table_id, key],
        )

        # Insert the new tag
        con.execute(
            "INSERT INTO ducklake_tag "
            "(object_id, begin_snapshot, end_snapshot, key, value) "
            "VALUES (?, ?, NULL, ?, ?)",
            [table_id, new_snap, key, value],
        )

        safe_schema = schema_name.replace('"', '""')
        safe_table = table_name.replace('"', '""')
        self._record_change(
            new_snap, f'set_table_tag:"{safe_schema}"."{safe_table}"."{key}"'
        )
        self._commit_metadata()

    def delete_table_tag(
        self,
        table_name: str,
        key: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Remove a tag from a table."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise TableNotFoundError(msg)

        # Check that the tag exists
        row = con.execute(
            "SELECT 1 FROM ducklake_tag "
            "WHERE object_id = ? AND key = ? AND end_snapshot IS NULL",
            [table_id, key],
        ).fetchone()
        if row is None:
            msg = f"Tag '{key}' not found on table '{schema_name}.{table_name}'"
            raise ValueError(msg)

        new_snap = self._create_snapshot(schema_ver, next_cat_id, next_file_id)

        con.execute(
            "UPDATE ducklake_tag SET end_snapshot = ? "
            "WHERE object_id = ? AND key = ? AND end_snapshot IS NULL",
            [new_snap, table_id, key],
        )

        safe_schema = schema_name.replace('"', '""')
        safe_table = table_name.replace('"', '""')
        self._record_change(
            new_snap, f'delete_table_tag:"{safe_schema}"."{safe_table}"."{key}"'
        )
        self._commit_metadata()

    def set_column_tag(
        self,
        table_name: str,
        column_name: str,
        key: str,
        value: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Set a tag on a column. Overwrites existing tag with same key."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise TableNotFoundError(msg)

        columns = self._get_columns_for_table(table_id, snap_id)
        col_id = None
        for cid, cname, _ctype, _parent in columns:
            if cname == column_name:
                col_id = cid
                break
        if col_id is None:
            msg = f"Column '{column_name}' not found in table '{schema_name}.{table_name}'"
            raise ValueError(msg)

        new_snap = self._create_snapshot(schema_ver, next_cat_id, next_file_id)

        # End any existing tag with the same key
        con.execute(
            "UPDATE ducklake_column_tag SET end_snapshot = ? "
            "WHERE table_id = ? AND column_id = ? AND key = ? AND end_snapshot IS NULL",
            [new_snap, table_id, col_id, key],
        )

        # Insert the new tag
        con.execute(
            "INSERT INTO ducklake_column_tag "
            "(table_id, column_id, begin_snapshot, end_snapshot, key, value) "
            "VALUES (?, ?, ?, NULL, ?, ?)",
            [table_id, col_id, new_snap, key, value],
        )

        safe_schema = schema_name.replace('"', '""')
        safe_table = table_name.replace('"', '""')
        safe_col = column_name.replace('"', '""')
        self._record_change(
            new_snap,
            f'set_column_tag:"{safe_schema}"."{safe_table}"."{safe_col}"."{key}"',
        )
        self._commit_metadata()

    def delete_column_tag(
        self,
        table_name: str,
        column_name: str,
        key: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Remove a tag from a column."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise TableNotFoundError(msg)

        columns = self._get_columns_for_table(table_id, snap_id)
        col_id = None
        for cid, cname, _ctype, _parent in columns:
            if cname == column_name:
                col_id = cid
                break
        if col_id is None:
            msg = f"Column '{column_name}' not found in table '{schema_name}.{table_name}'"
            raise ValueError(msg)

        # Check that the tag exists
        row = con.execute(
            "SELECT 1 FROM ducklake_column_tag "
            "WHERE table_id = ? AND column_id = ? AND key = ? AND end_snapshot IS NULL",
            [table_id, col_id, key],
        ).fetchone()
        if row is None:
            msg = f"Tag '{key}' not found on column '{column_name}' of table '{schema_name}.{table_name}'"
            raise ValueError(msg)

        new_snap = self._create_snapshot(schema_ver, next_cat_id, next_file_id)

        con.execute(
            "UPDATE ducklake_column_tag SET end_snapshot = ? "
            "WHERE table_id = ? AND column_id = ? AND key = ? AND end_snapshot IS NULL",
            [new_snap, table_id, col_id, key],
        )

        safe_schema = schema_name.replace('"', '""')
        safe_table = table_name.replace('"', '""')
        safe_col = column_name.replace('"', '""')
        self._record_change(
            new_snap,
            f'delete_column_tag:"{safe_schema}"."{safe_table}"."{safe_col}"."{key}"',
        )
        self._commit_metadata()

    def vacuum(self) -> int:
        """Delete orphaned Parquet files not referenced by any catalog entry."""
        con = self._connect()

        referenced: set[str] = set()

        data_base = self.data_path

        data_files = con.execute(
            "SELECT df.path, df.path_is_relative, t.path, t.path_is_relative, "
            "s.path, s.path_is_relative "
            "FROM ducklake_data_file df "
            "JOIN ducklake_table t ON df.table_id = t.table_id "
            "JOIN ducklake_schema s ON t.schema_id = s.schema_id"
        ).fetchall()
        for f_path, f_rel, t_path, t_rel, s_path, s_rel in data_files:
            abs_path = self._resolve_vacuum_path(
                f_path, bool(f_rel) if f_rel is not None else True,
                t_path or "", bool(t_rel) if t_rel is not None else True,
                s_path or "", bool(s_rel) if s_rel is not None else True,
                data_base,
            )
            referenced.add(storage.normalize_path(abs_path))

        delete_files = con.execute(
            "SELECT df.path, df.path_is_relative, t.path, t.path_is_relative, "
            "s.path, s.path_is_relative "
            "FROM ducklake_delete_file df "
            "JOIN ducklake_table t ON df.table_id = t.table_id "
            "JOIN ducklake_schema s ON t.schema_id = s.schema_id"
        ).fetchall()
        for f_path, f_rel, t_path, t_rel, s_path, s_rel in delete_files:
            abs_path = self._resolve_vacuum_path(
                f_path, bool(f_rel) if f_rel is not None else True,
                t_path or "", bool(t_rel) if t_rel is not None else True,
                s_path or "", bool(s_rel) if s_rel is not None else True,
                data_base,
            )
            referenced.add(storage.normalize_path(abs_path))

        deleted_count = 0
        all_parquet_files = storage.list_directory(data_base, suffix=".parquet")
        for full_path in all_parquet_files:
            if storage.normalize_path(full_path) not in referenced:
                storage.delete_file(full_path)
                deleted_count += 1

        return deleted_count

    # ------------------------------------------------------------------
    # CREATE VIEW
    # ------------------------------------------------------------------

    def _view_exists(
        self, view_name: str, schema_name: str, snapshot_id: int
    ) -> int | None:
        """Return view_id if the view exists at snapshot_id, else None."""
        con = self._connect()
        try:
            row = con.execute(
                "SELECT v.view_id FROM ducklake_view v "
                "JOIN ducklake_schema s ON v.schema_id = s.schema_id "
                "WHERE v.view_name = ? AND LOWER(s.schema_name) = LOWER(?) "
                "AND ? >= v.begin_snapshot AND (? < v.end_snapshot OR v.end_snapshot IS NULL) "
                "AND ? >= s.begin_snapshot AND (? < s.end_snapshot OR s.end_snapshot IS NULL)",
                [view_name, schema_name, snapshot_id, snapshot_id, snapshot_id, snapshot_id],
            ).fetchone()
        except Exception as exc:
            if self._backend.is_table_not_found(exc):
                return None
            raise
        return row[0] if row is not None else None

    @_retryable
    def create_view(
        self,
        view_name: str,
        sql: str,
        *,
        schema_name: str = "main",
        or_replace: bool = False,
        column_aliases: str = "",
    ) -> int:
        """Create a view in the catalog. Returns the new view_id."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        existing_view_id = self._view_exists(view_name, schema_name, snap_id)

        if existing_view_id is not None and not or_replace:
            msg = f"View '{schema_name}.{view_name}' already exists"
            raise ValueError(msg)

        schema_id, _schema_path, _schema_path_rel = self._resolve_schema_info(
            schema_name, snap_id
        )

        view_id = next_cat_id
        new_next_cat_id = next_cat_id + 1
        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, new_next_cat_id, next_file_id)

        changes: list[str] = []

        if existing_view_id is not None:
            con.execute(
                "UPDATE ducklake_view SET end_snapshot = ? "
                "WHERE view_id = ? AND end_snapshot IS NULL",
                [new_snap, existing_view_id],
            )
            changes.append(f"dropped_view:{existing_view_id}")

        view_uuid = str(uuid.uuid4())
        con.execute(
            "INSERT INTO ducklake_view "
            "(view_id, view_uuid, begin_snapshot, end_snapshot, schema_id, "
            "view_name, dialect, sql, column_aliases) "
            "VALUES (?, ?, ?, NULL, ?, ?, 'duckdb', ?, ?)",
            [view_id, view_uuid, new_snap, schema_id,
             view_name, sql, column_aliases],
        )

        safe_schema = schema_name.replace('"', '""')
        safe_view = view_name.replace('"', '""')
        changes.append(f'created_view:"{safe_schema}"."{safe_view}"')

        self._insert_schema_version(new_snap, new_schema_ver, None)

        self._record_change(new_snap, ",".join(changes))

        self._commit_metadata()
        return view_id

    # ------------------------------------------------------------------
    # DROP VIEW
    # ------------------------------------------------------------------

    @_retryable
    def drop_view(
        self,
        view_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Drop a view from the catalog."""
        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        view_id = self._view_exists(view_name, schema_name, snap_id)
        if view_id is None:
            msg = f"View '{schema_name}.{view_name}' not found"
            raise ValueError(msg)

        new_schema_ver = schema_ver + 1

        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        con.execute(
            "UPDATE ducklake_view SET end_snapshot = ? "
            "WHERE view_id = ? AND end_snapshot IS NULL",
            [new_snap, view_id],
        )

        self._insert_schema_version(new_snap, new_schema_ver, None)

        self._record_change(new_snap, f"dropped_view:{view_id}")

        self._commit_metadata()

    # ------------------------------------------------------------------
    # ADD FILES (register existing Parquet files)
    # ------------------------------------------------------------------

    @_retryable
    def add_files(
        self,
        table_name: str,
        file_paths: list[str],
        *,
        schema_name: str = "main",
    ) -> int:
        """Register existing Parquet files into a DuckLake table.

        The files are **not** copied or moved — they are referenced
        in-place.  Schema validation is performed by reading the first
        file's schema and comparing against the table's column
        definitions.

        Parameters
        ----------
        table_name
            Name of the target table.
        file_paths
            List of paths to Parquet files (local or object storage).
        schema_name
            Schema name (default: ``"main"``).

        Returns
        -------
        int
            The new snapshot ID.

        Raises
        ------
        ValueError
            If the table does not exist, no file paths are given, or
            the Parquet schema does not match the table schema.
        """
        if not file_paths:
            msg = "file_paths must not be empty"
            raise ValueError(msg)

        con = self._connect()
        snapshot_info = self._get_latest_snapshot()
        if snapshot_info is None:
            snap_id, schema_ver, next_cat_id, next_file_id = -1, 0, 1, 1
        else:
            snap_id, schema_ver, next_cat_id, next_file_id = snapshot_info
        self._start_write_transaction(snap_id)

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )
        self._track_table_write(table_id, "insert")

        columns = self._get_columns_for_table(table_id, snap_id)

        # --- Validate schema from the first file ----------------------
        first_table = storage.read_parquet(file_paths[0])
        first_schema = first_table.schema
        file_col_names = set(first_schema.names)
        catalog_col_names = {c[1] for c in columns}
        if file_col_names != catalog_col_names:
            msg = (
                f"Schema mismatch: file columns {sorted(file_col_names)} "
                f"do not match table columns {sorted(catalog_col_names)}"
            )
            raise ValueError(msg)

        # --- Allocate IDs ---------------------------------------------
        n_files = len(file_paths)
        new_next_file_id = next_file_id + n_files

        existing_stats = self._get_table_stats(table_id)
        row_id_start = existing_stats[1] if existing_stats is not None else 0

        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        mapping_id = self._register_name_mapping(table_id, columns)

        total_records = 0
        total_file_size = 0
        current_file_id = next_file_id
        current_row_id = row_id_start

        for fpath in file_paths:
            df = storage.read_parquet(fpath)
            record_count = len(df)
            file_size = storage.get_file_size(fpath)
            footer_size = _read_parquet_footer_size(fpath)

            # Register the file with an absolute path (path_is_relative=False)
            self._register_data_file_absolute(
                current_file_id, table_id, new_snap, fpath,
                record_count, file_size, footer_size, current_row_id,
                None, mapping_id,
            )

            col_stats = self._compute_file_column_stats(df, columns)
            self._register_file_column_stats(current_file_id, table_id, col_stats)
            self._update_table_column_stats(table_id, columns, col_stats)

            total_records += record_count
            total_file_size += file_size
            current_file_id += 1
            current_row_id += record_count

        self._update_table_stats(table_id, total_records, total_file_size)

        self._record_change(new_snap, f"inserted_into_table:{table_id}")
        self._commit_metadata()
        return new_snap

    def _register_data_file_absolute(
        self,
        data_file_id: int,
        table_id: int,
        new_snap: int,
        abs_path: str,
        record_count: int,
        file_size: int,
        footer_size: int,
        row_id_start: int,
        partition_id: int | None,
        mapping_id: int,
    ) -> None:
        """Register a data file with an absolute (non-relative) path."""
        con = self._connect()
        con.execute(
            "INSERT INTO ducklake_data_file "
            "(data_file_id, table_id, begin_snapshot, end_snapshot, file_order, "
            "path, path_is_relative, file_format, record_count, file_size_bytes, "
            "footer_size, row_id_start, partition_id, encryption_key, "
            + ("mapping_id, partial_max) "
               "VALUES (?, ?, ?, NULL, NULL, ?, ?, 'parquet', ?, ?, ?, ?, ?, NULL, ?, NULL)"
               if self._is_v04 else
               "partial_file_info, mapping_id) "
               "VALUES (?, ?, ?, NULL, NULL, ?, ?, 'parquet', ?, ?, ?, ?, ?, NULL, NULL, ?)"),
            [
                data_file_id,
                table_id,
                new_snap,
                abs_path,
                False,  # path_is_relative = False
                record_count,
                file_size,
                footer_size,
                row_id_start,
                partition_id,
                mapping_id,
            ],
        )

    @staticmethod
    def _resolve_vacuum_path(
        file_path: str,
        file_is_relative: bool,
        table_path: str,
        table_path_rel: bool,
        schema_path: str,
        schema_path_rel: bool,
        data_base: str,
    ) -> str:
        """Resolve a file path for vacuum operations."""
        if not file_is_relative:
            return file_path
        base = data_base
        if schema_path_rel:
            base = storage.join_path(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = storage.join_path(base, table_path)
        else:
            base = table_path
        return storage.join_path(base, file_path)
