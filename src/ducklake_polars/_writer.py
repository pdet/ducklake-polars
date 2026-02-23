"""DuckLake catalog writer — creates tables and inserts data."""

from __future__ import annotations

import os
import struct
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import polars as pl

from ducklake_polars._backend import PostgreSQLBackend, SQLiteBackend, create_backend
from ducklake_polars._schema import polars_type_to_duckdb


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


def _read_parquet_footer_size(path: str) -> int:
    """Read the Parquet footer size from a file."""
    try:
        with open(path, "rb") as f:
            f.seek(-8, 2)
            footer_len = struct.unpack("<I", f.read(4))[0]
        return footer_len
    except (OSError, struct.error):
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
        if hasattr(self._con, "commit"):
            self._con.commit()

    def close(self) -> None:
        self._con.close()


def _stat_value_to_str(value: Any, dtype: pl.DataType) -> str | None:
    """Serialize a Python value to a DuckLake stat string."""
    if value is None:
        return None
    base = type(dtype)
    if base in (pl.Boolean,):
        return "true" if value else "false"
    return str(value)


def _contains_nan(series: pl.Series) -> bool:
    """Check if a Polars Series contains NaN values."""
    try:
        return series.is_nan().any()
    except Exception:
        return False


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

    Handles snapshot creation, table/column registration, Parquet file
    writing, and statistics computation. Produces catalogs fully
    interoperable with DuckDB's DuckLake extension.

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
    ) -> None:
        self._backend = create_backend(metadata_path)
        self._metadata_path = metadata_path
        self._data_path_override = data_path_override
        self._data_inlining_row_limit = data_inlining_row_limit
        self._author = author
        self._commit_message = commit_message
        self._con: Any = None

    def _connect(self) -> Any:
        if self._con is None:
            raw = self._backend.connect_writable()
            self._con = _PlaceholderConnection(raw, self._backend.placeholder)
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> DuckLakeCatalogWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    def _get_latest_snapshot(self) -> tuple[int, int, int, int]:
        """Return (snapshot_id, schema_version, next_catalog_id, next_file_id)."""
        con = self._connect()
        row = con.execute(
            "SELECT snapshot_id, schema_version, next_catalog_id, next_file_id "
            "FROM ducklake_snapshot ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            msg = "No snapshots found — is this a valid DuckLake catalog?"
            raise ValueError(msg)
        return (row[0], row[1], row[2], row[3])

    def _create_snapshot(
        self,
        schema_version: int,
        next_catalog_id: int,
        next_file_id: int,
    ) -> int:
        """Create a new snapshot and return its ID."""
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

    def _record_change(
        self,
        snapshot_id: int,
        changes_made: str,
    ) -> None:
        """Insert a snapshot_changes record.

        Uses the ``author`` and ``commit_message`` stored on the writer
        instance (set via the constructor).
        """
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

        # Check if already registered
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

        # Build CREATE TABLE with row_id, begin_snapshot, end_snapshot + user columns
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

        # Register in ducklake_inlined_data_tables
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
            # Inlined data tables use BIGINT for boolean columns in both
            # SQLite and PostgreSQL, so always store as 0/1 integer.
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
        df: pl.DataFrame,
        table_id: int,
        schema_version: int,
        columns: list[tuple[int, str, str, int | None]],
        new_snap: int,
        row_id_start: int,
    ) -> None:
        """Insert DataFrame rows into the inlined data table."""
        con = self._connect()
        tbl_name = self._ensure_inlined_table(table_id, schema_version, columns)
        safe_tbl = tbl_name.replace('"', '""')

        col_names = [c[1] for c in columns]
        col_types = {c[1]: c[2] for c in columns}

        # Build column list for INSERT
        all_cols = ["row_id", "begin_snapshot", "end_snapshot"] + [
            f'"{c.replace(chr(34), chr(34) + chr(34))}"' for c in col_names
        ]
        placeholders = ", ".join(["?"] * len(all_cols))
        insert_sql = f'INSERT INTO "{safe_tbl}" ({", ".join(all_cols)}) VALUES ({placeholders})'

        for i, row_tuple in enumerate(df.iter_rows()):
            row_vals: list[Any] = [row_id_start + i, new_snap, None]
            for j, col_name in enumerate(col_names):
                row_vals.append(
                    self._serialize_value(row_tuple[j], col_types[col_name])
                )
            con.execute(insert_sql, row_vals)

    def _delete_inlined_rows(
        self,
        table_id: int,
        predicate: pl.Expr,
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
            # Read active rows
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

            # Build DataFrame to evaluate predicate
            data = {name: [r[i + 1] for r in rows] for i, name in enumerate(col_names)}
            row_ids = [r[0] for r in rows]
            inline_df = pl.DataFrame(data)

            # Evaluate predicate
            mask = inline_df.with_columns(predicate.alias("__del__"))["__del__"]
            to_delete = [row_ids[i] for i, v in enumerate(mask.to_list()) if v]

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
        # Check if adding new rows would exceed the limit
        # DuckDB inlines if the total active count (existing inlined + new) <= limit
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
        """Register a map_by_name column mapping and return the mapping_id.

        DuckLake uses Parquet field_id to map columns by default. Since Polars'
        native Parquet writer doesn't set field_ids, we register a ``map_by_name``
        mapping so DuckDB resolves columns by name instead.
        """
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
            "WHERE schema_name = ? AND begin_snapshot <= ? "
            "AND (end_snapshot IS NULL OR end_snapshot > ?)",
            [schema_name, snapshot_id, snapshot_id],
        ).fetchone()
        if row is None:
            msg = f"Schema '{schema_name}' not found at snapshot {snapshot_id}"
            raise ValueError(msg)
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
            "WHERE t.table_name = ? AND s.schema_name = ? "
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
        polars_schema: dict[str, pl.DataType],
        start_column_id: int = 1,
        parent_column: int | None = None,
        start_order: int = 1,
    ) -> list[_ColumnDef]:
        """Flatten a Polars schema into column definitions, handling nested types."""
        defs: list[_ColumnDef] = []
        col_id = start_column_id
        order = start_order

        for name, dtype in polars_schema.items():
            duckdb_type = polars_type_to_duckdb(dtype)
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
            if isinstance(dtype, pl.List):
                inner = dtype.inner  # type: ignore[attr-defined]
                if inner is not None:
                    child_type = polars_type_to_duckdb(inner)
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

                    # If the inner type is also compound, recurse
                    if isinstance(inner, pl.Struct):
                        child_defs = self._flatten_schema(
                            dict(inner.fields) if hasattr(inner, "fields") else {},
                            start_column_id=col_id,
                            parent_column=child_id,
                        )
                        defs.extend(child_defs)
                        col_id += len(child_defs)

            elif isinstance(dtype, pl.Struct):
                fields = {f.name: f.dtype for f in dtype.fields}
                child_defs = self._flatten_schema(
                    fields,
                    start_column_id=col_id,
                    parent_column=this_id,
                )
                defs.extend(child_defs)
                col_id += len(child_defs)

        return defs

    def create_table(
        self,
        table_name: str,
        polars_schema: pl.Schema | dict[str, pl.DataType],
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Create a new table in the catalog.

        Returns the new table_id.
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        # Check if table already exists
        if self._table_exists(table_name, schema_name, snap_id) is not None:
            msg = f"Table '{schema_name}.{table_name}' already exists"
            raise ValueError(msg)

        # Resolve schema
        schema_id, _schema_path, _schema_path_rel = self._resolve_schema_info(
            schema_name, snap_id
        )

        # Allocate table_id
        table_id = next_cat_id
        new_next_cat_id = next_cat_id + 1
        new_schema_ver = schema_ver + 1

        # Create snapshot
        new_snap = self._create_snapshot(new_schema_ver, new_next_cat_id, next_file_id)

        # Insert table
        table_uuid = str(uuid.uuid4())
        table_path = f"{table_name}/"
        con.execute(
            "INSERT INTO ducklake_table "
            "(table_id, table_uuid, begin_snapshot, end_snapshot, schema_id, "
            "table_name, path, path_is_relative) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?, ?)",
            [table_id, table_uuid, new_snap, schema_id, table_name, table_path, True],
        )

        # Flatten schema and insert columns
        schema_dict = dict(polars_schema) if isinstance(polars_schema, pl.Schema) else polars_schema
        col_defs = self._flatten_schema(schema_dict)
        for cd in col_defs:
            con.execute(
                "INSERT INTO ducklake_column "
                "(column_id, begin_snapshot, end_snapshot, table_id, column_order, "
                "column_name, column_type, initial_default, default_value, "
                "nulls_allowed, parent_column) "
                "VALUES (?, ?, NULL, ?, ?, ?, ?, NULL, NULL, ?, ?)",
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

        # Record schema version
        con.execute(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version) "
            "VALUES (?, ?)",
            [new_snap, new_schema_ver],
        )

        # Record change
        safe_schema = schema_name.replace('"', '""')
        safe_table = table_name.replace('"', '""')
        self._record_change(
            new_snap, f'created_table:"{safe_schema}"."{safe_table}"'
        )

        con.commit()
        return table_id

    def create_table_with_data(
        self,
        table_name: str,
        df: pl.DataFrame,
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Create a new table and insert data in a single snapshot.

        Returns the new snapshot ID.
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        # Check table doesn't exist
        if self._table_exists(table_name, schema_name, snap_id) is not None:
            msg = f"Table '{schema_name}.{table_name}' already exists"
            raise ValueError(msg)

        # Resolve schema
        schema_id, schema_path, schema_path_rel = self._resolve_schema_info(
            schema_name, snap_id
        )

        # Allocate IDs
        table_id = next_cat_id
        new_next_cat_id = next_cat_id + 1
        new_schema_ver = schema_ver + 1

        # Flatten schema
        schema_dict = dict(df.schema)
        col_defs = self._flatten_schema(schema_dict)
        top_level_cols = [
            (cd.column_id, cd.column_name, cd.column_type, cd.parent_column)
            for cd in col_defs
            if cd.parent_column is None
        ]

        table_path = f"{table_name}/"

        # Determine data handling
        has_data = not df.is_empty()
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

        # Create single snapshot
        new_snap = self._create_snapshot(
            new_schema_ver, new_next_cat_id, new_next_file_id
        )

        # Insert table
        table_uuid = str(uuid.uuid4())
        con.execute(
            "INSERT INTO ducklake_table "
            "(table_id, table_uuid, begin_snapshot, end_snapshot, schema_id, "
            "table_name, path, path_is_relative) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?, ?)",
            [table_id, table_uuid, new_snap, schema_id, table_name, table_path, True],
        )

        # Insert columns
        for cd in col_defs:
            con.execute(
                "INSERT INTO ducklake_column "
                "(column_id, begin_snapshot, end_snapshot, table_id, column_order, "
                "column_name, column_type, initial_default, default_value, "
                "nulls_allowed, parent_column) "
                "VALUES (?, ?, NULL, ?, ?, ?, ?, NULL, NULL, ?, ?)",
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

        # Record schema version
        con.execute(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version) "
            "VALUES (?, ?)",
            [new_snap, new_schema_ver],
        )

        # Write data
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
                # Build output directory
                base = self.data_path
                if schema_path_rel:
                    base = os.path.join(base, schema_path)
                else:
                    base = schema_path
                base = os.path.join(base, table_path)
                os.makedirs(base, exist_ok=True)

                file_name = f"ducklake-{_uuid7()}.parquet"
                file_path = os.path.join(base, file_name)
                df.write_parquet(file_path)

                file_size = os.path.getsize(file_path)
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

        con.commit()
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
            "WHERE t.table_name = ? AND s.schema_name = ? "
            "AND ? >= t.begin_snapshot AND (? < t.end_snapshot OR t.end_snapshot IS NULL) "
            "AND ? >= s.begin_snapshot AND (? < s.end_snapshot OR s.end_snapshot IS NULL)",
            [table_name, schema_name, snapshot_id, snapshot_id, snapshot_id, snapshot_id],
        ).fetchone()
        if row is None:
            msg = f"Table '{schema_name}.{table_name}' not found at snapshot {snapshot_id}"
            raise ValueError(msg)
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
        df: pl.DataFrame,
        columns: list[tuple[int, str, str, int | None]],
    ) -> list[tuple[int, str, int, int | None, str | None, str | None, int | None]]:
        """Compute per-column statistics for a DataFrame.

        Returns list of (column_id, column_name, value_count, null_count,
        min_value, max_value, contains_nan_int).
        """
        results = []
        n = len(df)
        for col_id, col_name, col_type, _parent in columns:
            if col_name not in df.columns:
                continue
            series = df[col_name]
            null_count = series.null_count()
            value_count = n

            # Compute min/max for scalar types only
            min_val = None
            max_val = None
            try:
                from ducklake_polars._schema import duckdb_type_to_polars

                polars_type = duckdb_type_to_polars(col_type)
                base = polars_type.base_type()
                # Only compute stats for types that support ordering
                if base in (
                    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                    pl.Float32, pl.Float64,
                    pl.Boolean, pl.String, pl.Date, pl.Datetime, pl.Decimal,
                ):
                    if null_count < n:
                        raw_min = series.min()
                        raw_max = series.max()
                        min_val = _stat_value_to_str(raw_min, polars_type)
                        max_val = _stat_value_to_str(raw_max, polars_type)
            except (ValueError, TypeError):
                pass

            # NaN detection for float types
            nan_int = None
            if col_type in ("float32", "float64"):
                nan_int = _contains_nan(series)

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
        # Compare as the appropriate type
        try:
            from ducklake_polars._schema import duckdb_type_to_polars

            polars_type = duckdb_type_to_polars(col_type)
            from ducklake_polars._stats import _parse_stat_value

            e_val = _parse_stat_value(existing, polars_type)
            n_val = _parse_stat_value(new, polars_type)
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
            "partial_file_info, mapping_id) "
            "VALUES (?, ?, ?, NULL, NULL, ?, ?, 'parquet', ?, ?, ?, ?, ?, NULL, NULL, ?)",
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

    def insert_data(
        self,
        df: pl.DataFrame,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Insert a DataFrame into an existing table.

        Writes Parquet file(s) and registers them in the catalog with
        column statistics. For partitioned tables, writes one file per
        unique partition value combination using Hive-style directory
        layout. When data inlining is enabled and the row count is below
        the threshold, data is stored directly in the metadata catalog
        instead of Parquet files. Returns the new snapshot ID.
        """
        if df.is_empty():
            msg = "Cannot insert empty DataFrame"
            raise ValueError(msg)

        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        # Resolve table
        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )

        # Get column definitions
        columns = self._get_columns_for_table(table_id, snap_id)

        record_count = len(df)

        # Check if data should be inlined
        if self._should_inline(table_id, snap_id, record_count):
            return self._insert_inlined(
                df, table_id, columns,
                snap_id, schema_ver, next_cat_id, next_file_id,
            )

        # Build the output directory path
        base = self.data_path
        if schema_path_rel:
            base = os.path.join(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = os.path.join(base, table_path)
        else:
            base = table_path

        # Check for active partition spec
        partition_id = self._get_active_partition(table_id, snap_id)

        if partition_id is not None:
            return self._insert_partitioned(
                df, table_id, table_name, schema_name,
                columns, base, partition_id,
                snap_id, schema_ver, next_cat_id, next_file_id,
            )

        os.makedirs(base, exist_ok=True)

        # Non-partitioned insert (original logic)
        file_name = f"ducklake-{_uuid7()}.parquet"
        file_path = os.path.join(base, file_name)
        df.write_parquet(file_path)

        file_size = os.path.getsize(file_path)
        footer_size = _read_parquet_footer_size(file_path)

        # Get current table stats for row_id_start
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
        con.commit()
        return new_snap

    def _insert_inlined(
        self,
        df: pl.DataFrame,
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

        # Get current table stats for row_id_start
        existing_stats = self._get_table_stats(table_id)
        row_id_start = existing_stats[1] if existing_stats is not None else 0

        # Create snapshot (no new file IDs needed for inlined data)
        new_snap = self._create_snapshot(schema_ver, next_cat_id, next_file_id)

        # Insert rows into the inlined data table
        self._insert_inlined_rows(
            df, table_id, schema_ver, columns, new_snap, row_id_start,
        )

        # Update table stats (file_size=0 for inlined data)
        self._update_table_stats(table_id, record_count, 0)
        self._update_table_column_stats(
            table_id, columns,
            self._compute_file_column_stats(df, columns),
        )

        self._record_change(new_snap, f"inserted_into_table:{table_id}")
        con.commit()
        return new_snap

    def _insert_partitioned(
        self,
        df: pl.DataFrame,
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

        # Get partition column definitions
        part_cols = self._get_partition_columns(partition_id, table_id)
        col_id_to_name: dict[int, str] = {c[0]: c[1] for c in columns}
        part_col_names: list[str] = []
        part_key_indices: list[int] = []
        for key_index, col_id, transform in part_cols:
            part_col_names.append(col_id_to_name[col_id])
            part_key_indices.append(key_index)

        # Group DataFrame by partition columns
        groups = df.group_by(part_col_names, maintain_order=True)
        group_list: list[tuple[tuple, pl.DataFrame]] = []
        for group_key, group_df in groups:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_list.append((group_key, group_df))

        n_files = len(group_list)

        # Get current table stats for row_id_start
        existing_stats = self._get_table_stats(table_id)
        row_id_start = existing_stats[1] if existing_stats is not None else 0

        # Allocate file IDs
        new_next_file_id = next_file_id + n_files

        # Create snapshot
        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        # Register name mapping once (shared across all partition files)
        mapping_id = self._register_name_mapping(table_id, columns)

        total_file_size = 0
        total_records = 0
        all_col_stats: list[
            list[tuple[int, str, int, int | None, str | None, str | None, int | None]]
        ] = []

        current_file_id = next_file_id
        current_row_id = row_id_start

        for group_key, group_df in group_list:
            # Build Hive-style partition path
            partition_values = [str(v) for v in group_key]
            hive_subdir = self._build_hive_path(part_col_names, partition_values)
            partition_dir = os.path.join(base_dir, hive_subdir)
            os.makedirs(partition_dir, exist_ok=True)

            file_name = f"ducklake-{_uuid7()}.parquet"
            file_path = os.path.join(partition_dir, file_name)
            group_df.write_parquet(file_path)

            file_size = os.path.getsize(file_path)
            footer_size = _read_parquet_footer_size(file_path)
            record_count = len(group_df)

            # Relative path from table dir: hive_subdir/filename
            rel_path = f"{hive_subdir}/{file_name}"

            self._register_data_file(
                current_file_id, table_id, new_snap, rel_path,
                record_count, file_size, footer_size, current_row_id,
                partition_id, mapping_id,
            )

            # Register partition values
            self._register_partition_values(
                current_file_id, table_id, part_key_indices, partition_values,
            )

            # Compute and register per-file column stats
            col_stats = self._compute_file_column_stats(group_df, columns)
            self._register_file_column_stats(current_file_id, table_id, col_stats)
            all_col_stats.append(col_stats)

            total_file_size += file_size
            total_records += record_count
            current_file_id += 1
            current_row_id += record_count

        # Update table stats (aggregated across all partition files)
        self._update_table_stats(table_id, total_records, total_file_size)

        # Update table column stats using the full DataFrame (not per-group)
        full_col_stats = self._compute_file_column_stats(df, columns)
        self._update_table_column_stats(table_id, columns, full_col_stats)

        self._record_change(new_snap, f"inserted_into_table:{table_id}")
        con.commit()
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

    def overwrite_data(
        self,
        df: pl.DataFrame,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Overwrite all data in a table with a new DataFrame.

        Ends all existing data/delete files and writes new data.
        Returns the new snapshot ID.
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )

        columns = self._get_columns_for_table(table_id, snap_id)

        # Build output directory
        base = self.data_path
        if schema_path_rel:
            base = os.path.join(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = os.path.join(base, table_path)
        else:
            base = table_path
        os.makedirs(base, exist_ok=True)

        record_count = len(df)

        # Check for active partition spec
        part_id = self._get_active_partition(table_id, snap_id)

        if part_id is not None and record_count > 0:
            return self._overwrite_partitioned(
                df, table_id, columns, base, part_id,
                snap_id, schema_ver, next_cat_id, next_file_id,
            )

        data_file_id = next_file_id
        new_next_file_id = next_file_id + 1 if record_count > 0 else next_file_id

        # Create snapshot
        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        # End all existing files and inlined data
        self._end_all_data_files(table_id, snap_id, new_snap)
        self._end_all_delete_files(table_id, snap_id, new_snap)
        self._end_all_inlined_rows(table_id, snap_id, new_snap)

        # Check if new data should be inlined
        if record_count > 0 and self._should_inline(table_id, new_snap, record_count):
            self._insert_inlined_rows(
                df, table_id, schema_ver, columns, new_snap, 0,
            )
            # Reset table stats for inlined overwrite
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
            con.commit()
            return new_snap

        if record_count > 0:
            file_name = f"ducklake-{_uuid7()}.parquet"
            file_path = os.path.join(base, file_name)
            df.write_parquet(file_path)

            file_size = os.path.getsize(file_path)
            footer_size = _read_parquet_footer_size(file_path)

            mapping_id = self._register_name_mapping(table_id, columns)

            self._register_data_file(
                data_file_id, table_id, new_snap, file_name,
                record_count, file_size, footer_size, 0,
                None, mapping_id,
            )

            col_stats = self._compute_file_column_stats(df, columns)
            self._register_file_column_stats(data_file_id, table_id, col_stats)

            # Reset table stats
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

            # Reset table column stats
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
            # Empty overwrite — clear stats
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
        con.commit()
        return new_snap

    def _overwrite_partitioned(
        self,
        df: pl.DataFrame,
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

        groups = df.group_by(part_col_names, maintain_order=True)
        group_list = []
        for group_key, group_df in groups:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_list.append((group_key, group_df))

        n_files = len(group_list)
        new_next_file_id = next_file_id + n_files

        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        # End all existing files and inlined data
        self._end_all_data_files(table_id, snap_id, new_snap)
        self._end_all_delete_files(table_id, snap_id, new_snap)
        self._end_all_inlined_rows(table_id, snap_id, new_snap)

        mapping_id = self._register_name_mapping(table_id, columns)

        total_file_size = 0
        total_records = 0
        current_file_id = next_file_id
        current_row_id = 0

        for group_key, group_df in group_list:
            partition_values = [str(v) for v in group_key]
            hive_subdir = self._build_hive_path(part_col_names, partition_values)
            partition_dir = os.path.join(base_dir, hive_subdir)
            os.makedirs(partition_dir, exist_ok=True)

            file_name = f"ducklake-{_uuid7()}.parquet"
            file_path = os.path.join(partition_dir, file_name)
            group_df.write_parquet(file_path)

            file_size = os.path.getsize(file_path)
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

        # Reset table stats
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

        # Reset table column stats
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
        con.commit()
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
            base = os.path.join(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = os.path.join(base, table_path)
        else:
            base = table_path
        return os.path.join(base, file_path)

    def delete_data(
        self,
        predicate: pl.Expr,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Delete rows matching a predicate from a table.

        For Parquet-backed data, creates Iceberg-compatible position-delete
        files. For inlined data, sets ``end_snapshot`` on matching rows.
        Returns the number of deleted rows. If no rows match the predicate,
        no snapshot is created and 0 is returned.
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )

        columns = self._get_columns_for_table(table_id, snap_id)
        data_files = self._get_active_data_files(table_id, snap_id)

        # Count inlined rows that match the predicate (pre-check)
        inlined_count = self._get_inlined_active_row_count(table_id, snap_id)

        if not data_files and inlined_count == 0:
            return 0

        # Handle Parquet file deletes
        pending_deletes: list[tuple[int, str, list[int]]] = []
        total_deleted = 0

        if data_files:
            base = self.data_path
            if schema_path_rel:
                base = os.path.join(base, schema_path)
            else:
                base = schema_path
            if table_path_rel:
                base = os.path.join(base, table_path)
            else:
                base = table_path
            os.makedirs(base, exist_ok=True)

            for data_file_id, rel_path, path_is_rel, record_count, row_id_start in data_files:
                abs_path = self._resolve_file_path(
                    rel_path, path_is_rel,
                    table_path, table_path_rel,
                    schema_path, schema_path_rel,
                )
                df = pl.read_parquet(abs_path)
                mask = df.with_columns(predicate.alias("__delete_mask__"))["__delete_mask__"]
                positions = [i for i, v in enumerate(mask.to_list()) if v]
                if positions:
                    pending_deletes.append((data_file_id, abs_path, positions))
                    total_deleted += len(positions)

        # Pre-count inlined deletes (we need to know totals before creating snapshot)
        # We'll do the actual delete after creating the snapshot
        # For now just check if there will be inlined deletes
        has_inlined_data = inlined_count > 0

        if total_deleted == 0 and not has_inlined_data:
            return 0

        # Allocate file IDs for Parquet delete files
        current_file_id = next_file_id
        new_next_file_id = next_file_id + len(pending_deletes)

        # Create snapshot
        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        # Write Parquet delete files
        if pending_deletes:
            base = self.data_path
            if schema_path_rel:
                base = os.path.join(base, schema_path)
            else:
                base = schema_path
            if table_path_rel:
                base = os.path.join(base, table_path)
            else:
                base = table_path

            for data_file_id, abs_data_path, positions in pending_deletes:
                delete_file_id = current_file_id
                current_file_id += 1

                delete_df = pl.DataFrame({
                    "file_path": [abs_data_path] * len(positions),
                    "pos": pl.Series(positions, dtype=pl.Int64),
                })
                delete_file_name = f"ducklake-{_uuid7()}-delete.parquet"
                delete_file_path = os.path.join(base, delete_file_name)
                delete_df.write_parquet(delete_file_path)

                delete_file_size = os.path.getsize(delete_file_path)
                delete_footer_size = _read_parquet_footer_size(delete_file_path)

                con.execute(
                    "INSERT INTO ducklake_delete_file "
                    "(delete_file_id, table_id, begin_snapshot, end_snapshot, "
                    "data_file_id, path, path_is_relative, format, delete_count, "
                    "file_size_bytes, footer_size, encryption_key) "
                    "VALUES (?, ?, ?, NULL, ?, ?, ?, 'parquet', ?, ?, ?, NULL)",
                    [
                        delete_file_id, table_id, new_snap, data_file_id,
                        delete_file_name, True, len(positions),
                        delete_file_size, delete_footer_size,
                    ],
                )

        # Delete from inlined data (set end_snapshot)
        inlined_deleted = 0
        if has_inlined_data:
            inlined_deleted = self._delete_inlined_rows(
                table_id, predicate, snap_id, new_snap, columns,
            )
            total_deleted += inlined_deleted

        if total_deleted == 0:
            # Nothing was actually deleted — but we already created a snapshot.
            # This can happen if inlined data had rows but none matched.
            # Roll back by not committing (the snapshot is harmless though).
            con.commit()
            return 0

        self._record_change(new_snap, f"deleted_from_table:{table_id}")
        con.commit()
        return total_deleted

    # ------------------------------------------------------------------
    # UPDATE (delete + insert in a single snapshot)
    # ------------------------------------------------------------------

    def _get_inlined_matched_rows(
        self,
        table_id: int,
        predicate: pl.Expr,
        snapshot_id: int,
        columns: list[tuple[int, str, str, int | None]],
    ) -> pl.DataFrame | None:
        """Read inlined rows matching a predicate. Returns DataFrame or None."""
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
        all_matched: list[pl.DataFrame] = []

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
            inline_df = pl.DataFrame(data)
            mask = inline_df.with_columns(predicate.alias("__mask__"))["__mask__"]
            matched_indices = [i for i, v in enumerate(mask.to_list()) if v]
            if matched_indices:
                matched = inline_df.filter(predicate)
                all_matched.append(matched)

        if not all_matched:
            return None
        return pl.concat(all_matched) if len(all_matched) > 1 else all_matched[0]

    def update_data(
        self,
        updates: dict[str, Any],
        predicate: pl.Expr,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Update rows matching a predicate.

        Creates position-delete files for old Parquet rows and sets
        ``end_snapshot`` on old inlined rows. Writes updated rows to a
        new Parquet data file. All changes are in a single snapshot.
        Returns the number of rows updated. If no rows match, no
        snapshot is created.
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )

        columns = self._get_columns_for_table(table_id, snap_id)
        data_files = self._get_active_data_files(table_id, snap_id)
        inlined_count = self._get_inlined_active_row_count(table_id, snap_id)

        if not data_files and inlined_count == 0:
            return 0

        # Build output directory
        base = self.data_path
        if schema_path_rel:
            base = os.path.join(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = os.path.join(base, table_path)
        else:
            base = table_path
        os.makedirs(base, exist_ok=True)

        # Evaluate predicate on Parquet data files
        pending_deletes: list[tuple[int, str, list[int]]] = []
        matched_dfs: list[pl.DataFrame] = []
        total_updated = 0

        for data_file_id, rel_path, path_is_rel, record_count, row_id_start in data_files:
            abs_path = self._resolve_file_path(
                rel_path, path_is_rel,
                table_path, table_path_rel,
                schema_path, schema_path_rel,
            )
            df = pl.read_parquet(abs_path)
            mask = df.with_columns(predicate.alias("__mask__"))["__mask__"]
            positions = [i for i, v in enumerate(mask.to_list()) if v]
            if positions:
                pending_deletes.append((data_file_id, abs_path, positions))
                matched_dfs.append(df.filter(predicate))
                total_updated += len(positions)

        # Check inlined data for matches
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

        # Apply updates to matched rows
        all_matched = pl.concat(matched_dfs)
        update_exprs = []
        for col_name, value in updates.items():
            if isinstance(value, pl.Expr):
                update_exprs.append(value.alias(col_name))
            else:
                update_exprs.append(pl.lit(value).alias(col_name))
        updated_df = all_matched.with_columns(update_exprs)

        # Check for active partition spec
        part_id = self._get_active_partition(table_id, snap_id)

        if part_id is not None:
            # Partitioned update: write updated rows as partitioned files
            part_cols = self._get_partition_columns(part_id, table_id)
            col_id_to_name: dict[int, str] = {c[0]: c[1] for c in columns}
            part_col_names = [col_id_to_name[col_id] for _, col_id, _ in part_cols]
            part_key_indices = [key_index for key_index, _, _ in part_cols]

            groups = updated_df.group_by(part_col_names, maintain_order=True)
            group_list = []
            for group_key, group_df in groups:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                group_list.append((group_key, group_df))

            n_data_files = len(group_list)
        else:
            n_data_files = 1

        # Allocate file IDs: delete files first, then data file(s)
        n_delete_files = len(pending_deletes)
        first_data_file_id = next_file_id + n_delete_files
        new_next_file_id = first_data_file_id + n_data_files

        # Create snapshot
        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        # Write delete files
        current_file_id = next_file_id
        for data_file_id, abs_data_path, positions in pending_deletes:
            delete_file_id = current_file_id
            current_file_id += 1

            delete_df = pl.DataFrame({
                "file_path": [abs_data_path] * len(positions),
                "pos": pl.Series(positions, dtype=pl.Int64),
            })
            delete_file_name = f"ducklake-{_uuid7()}-delete.parquet"
            delete_file_path = os.path.join(base, delete_file_name)
            delete_df.write_parquet(delete_file_path)

            delete_file_size = os.path.getsize(delete_file_path)
            delete_footer_size = _read_parquet_footer_size(delete_file_path)

            con.execute(
                "INSERT INTO ducklake_delete_file "
                "(delete_file_id, table_id, begin_snapshot, end_snapshot, "
                "data_file_id, path, path_is_relative, format, delete_count, "
                "file_size_bytes, footer_size, encryption_key) "
                "VALUES (?, ?, ?, NULL, ?, ?, ?, 'parquet', ?, ?, ?, NULL)",
                [
                    delete_file_id, table_id, new_snap, data_file_id,
                    delete_file_name, True, len(positions),
                    delete_file_size, delete_footer_size,
                ],
            )

        # Delete inlined rows (set end_snapshot)
        if inlined_matched is not None and len(inlined_matched) > 0:
            self._delete_inlined_rows(
                table_id, predicate, snap_id, new_snap, columns,
            )

        # Row ID start
        existing_stats = self._get_table_stats(table_id)
        row_id_start_new = existing_stats[1] if existing_stats else 0

        # Register name mapping
        mapping_id = self._register_name_mapping(table_id, columns)

        if part_id is not None:
            # Write partitioned data files
            total_file_size = 0
            current_data_file_id = first_data_file_id
            current_row_id = row_id_start_new

            for group_key, group_df in group_list:
                partition_values = [str(v) for v in group_key]
                hive_subdir = self._build_hive_path(part_col_names, partition_values)
                partition_dir = os.path.join(base, hive_subdir)
                os.makedirs(partition_dir, exist_ok=True)

                file_name = f"ducklake-{_uuid7()}.parquet"
                file_path = os.path.join(partition_dir, file_name)
                group_df.write_parquet(file_path)

                file_size = os.path.getsize(file_path)
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
            # Write single non-partitioned data file
            file_name = f"ducklake-{_uuid7()}.parquet"
            file_path = os.path.join(base, file_name)
            updated_df.write_parquet(file_path)

            file_size = os.path.getsize(file_path)
            footer_size = _read_parquet_footer_size(file_path)

            self._register_data_file(
                first_data_file_id, table_id, new_snap, file_name,
                total_updated, file_size, footer_size, row_id_start_new,
                None, mapping_id,
            )

            col_stats = self._compute_file_column_stats(updated_df, columns)
            self._register_file_column_stats(first_data_file_id, table_id, col_stats)
            update_file_size = file_size

        # Update table stats
        self._update_table_stats(table_id, total_updated, update_file_size)

        # Update table column stats
        full_col_stats = self._compute_file_column_stats(updated_df, columns)
        self._update_table_column_stats(table_id, columns, full_col_stats)

        # Record change: both insert and delete in the same snapshot
        self._record_change(
            new_snap,
            f"inserted_into_table:{table_id},deleted_from_table:{table_id}",
        )

        con.commit()
        return total_updated

    # ------------------------------------------------------------------
    # MERGE (match on keys, delete matched + insert)
    # ------------------------------------------------------------------

    def _read_all_inlined_active_rows(
        self,
        table_id: int,
        snapshot_id: int,
        columns: list[tuple[int, str, str, int | None]],
    ) -> pl.DataFrame | None:
        """Read all active inlined rows as a DataFrame (no predicate filter)."""
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
        all_dfs: list[pl.DataFrame] = []

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
                all_dfs.append(pl.DataFrame(data))

        if not all_dfs:
            return None
        return pl.concat(all_dfs) if len(all_dfs) > 1 else all_dfs[0]

    @staticmethod
    def _build_key_match_predicate(
        on: list[str],
        matched_keys_df: pl.DataFrame,
    ) -> pl.Expr:
        """Build a Polars expression matching rows whose key columns are in *matched_keys_df*."""
        if len(on) == 1:
            return pl.col(on[0]).is_in(matched_keys_df[on[0]])

        # Multiple key columns: OR of per-row AND conditions
        conditions: list[pl.Expr] = []
        for row in matched_keys_df.iter_rows(named=True):
            row_cond: pl.Expr | None = None
            for col in on:
                eq = pl.col(col) == row[col]
                row_cond = eq if row_cond is None else row_cond & eq
            assert row_cond is not None
            conditions.append(row_cond)

        result = conditions[0]
        for c in conditions[1:]:
            result = result | c
        return result

    def merge_data(
        self,
        source_df: pl.DataFrame,
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
        Implemented as delete + insert in a single snapshot.

        Parameters
        ----------
        when_matched_update
            - ``None``: matched target rows are left untouched.
            - ``True``: replace matched target rows with source rows.
            - ``dict``: update matched target rows with these values
              (literal or ``pl.Expr``).
        when_not_matched_insert
            If True (default), source rows that have no match in the
            target are inserted.

        Returns ``(rows_updated, rows_inserted)``.
        """
        if isinstance(on, str):
            on = [on]

        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )

        columns = self._get_columns_for_table(table_id, snap_id)
        col_names = [c[1] for c in columns]
        data_files = self._get_active_data_files(table_id, snap_id)
        inlined_count = self._get_inlined_active_row_count(table_id, snap_id)

        # Unique source keys for matching
        source_keys = source_df.select(on).unique()

        # ----------------------------------------------------------
        # Phase 1: find matched target rows in Parquet files
        # ----------------------------------------------------------
        pending_deletes: list[tuple[int, str, list[int]]] = []
        matched_target_dfs: list[pl.DataFrame] = []
        all_target_key_dfs: list[pl.DataFrame] = []

        for data_file_id, rel_path, path_is_rel, _rc, _rid in data_files:
            abs_path = self._resolve_file_path(
                rel_path, path_is_rel,
                table_path, table_path_rel,
                schema_path, schema_path_rel,
            )
            file_df = pl.read_parquet(abs_path)
            all_target_key_dfs.append(file_df.select(on))

            if when_matched_update is not None:
                file_df_idx = file_df.with_row_index("__merge_idx__")
                matched = file_df_idx.join(source_keys, on=on, how="semi")
                if len(matched) > 0:
                    positions = sorted(matched["__merge_idx__"].to_list())
                    pending_deletes.append((data_file_id, abs_path, positions))
                    matched_target_dfs.append(matched.drop("__merge_idx__"))

        # ----------------------------------------------------------
        # Phase 2: find matched target rows in inlined data
        # ----------------------------------------------------------
        inlined_matched_df: pl.DataFrame | None = None
        if inlined_count > 0:
            inlined_df = self._read_all_inlined_active_rows(
                table_id, snap_id, columns,
            )
            if inlined_df is not None and len(inlined_df) > 0:
                all_target_key_dfs.append(inlined_df.select(on))
                if when_matched_update is not None:
                    inlined_matched = inlined_df.join(
                        source_keys, on=on, how="semi",
                    )
                    if len(inlined_matched) > 0:
                        inlined_matched_df = inlined_matched
                        matched_target_dfs.append(inlined_matched)

        # ----------------------------------------------------------
        # Phase 3: counts and early exit
        # ----------------------------------------------------------
        total_updated = sum(len(p) for _, _, p in pending_deletes)
        if inlined_matched_df is not None:
            total_updated += len(inlined_matched_df)

        # Find unmatched source rows
        unmatched_source: pl.DataFrame
        if when_not_matched_insert:
            if all_target_key_dfs:
                all_target_keys = pl.concat(all_target_key_dfs).unique()
                unmatched_source = source_df.join(
                    all_target_keys, on=on, how="anti",
                )
            else:
                unmatched_source = source_df
        else:
            unmatched_source = source_df.clear()

        total_inserted = len(unmatched_source)

        if total_updated == 0 and total_inserted == 0:
            return (0, 0)

        # ----------------------------------------------------------
        # Phase 4: build rows to insert
        # ----------------------------------------------------------
        rows_to_insert_parts: list[pl.DataFrame] = []

        if when_matched_update is not None and matched_target_dfs:
            all_matched = pl.concat(matched_target_dfs)
            if when_matched_update is True:
                # Replace with source rows
                updated_rows = (
                    all_matched.select(on)
                    .join(source_df, on=on, how="inner")
                    .select(col_names)
                )
                rows_to_insert_parts.append(updated_rows)
            elif isinstance(when_matched_update, dict):
                update_exprs: list[pl.Expr] = []
                for col_name, value in when_matched_update.items():
                    if isinstance(value, pl.Expr):
                        update_exprs.append(value.alias(col_name))
                    else:
                        update_exprs.append(pl.lit(value).alias(col_name))
                rows_to_insert_parts.append(
                    all_matched.with_columns(update_exprs)
                )

        if total_inserted > 0:
            rows_to_insert_parts.append(unmatched_source.select(col_names))

        if not rows_to_insert_parts:
            return (0, 0)

        insert_df = (
            pl.concat(rows_to_insert_parts)
            if len(rows_to_insert_parts) > 1
            else rows_to_insert_parts[0]
        )

        # ----------------------------------------------------------
        # Phase 5: build output dir
        # ----------------------------------------------------------
        base = self.data_path
        if schema_path_rel:
            base = os.path.join(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = os.path.join(base, table_path)
        else:
            base = table_path
        os.makedirs(base, exist_ok=True)

        # ----------------------------------------------------------
        # Phase 6: allocate IDs and create snapshot
        # ----------------------------------------------------------
        n_delete_files = len(pending_deletes)
        n_data_files = 1 if len(insert_df) > 0 else 0

        # Check for partitioned table
        part_id = self._get_active_partition(table_id, snap_id)
        if part_id is not None and n_data_files > 0:
            part_cols = self._get_partition_columns(part_id, table_id)
            col_id_to_name: dict[int, str] = {c[0]: c[1] for c in columns}
            part_col_names = [col_id_to_name[cid] for _, cid, _ in part_cols]
            part_key_indices = [ki for ki, _, _ in part_cols]
            groups = insert_df.group_by(part_col_names, maintain_order=True)
            group_list: list[tuple[tuple, pl.DataFrame]] = []
            for gk, gdf in groups:
                if not isinstance(gk, tuple):
                    gk = (gk,)
                group_list.append((gk, gdf))
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

            delete_df = pl.DataFrame({
                "file_path": [abs_data_path] * len(positions),
                "pos": pl.Series(positions, dtype=pl.Int64),
            })
            delete_file_name = f"ducklake-{_uuid7()}-delete.parquet"
            delete_file_path = os.path.join(base, delete_file_name)
            delete_df.write_parquet(delete_file_path)

            delete_file_size = os.path.getsize(delete_file_path)
            delete_footer_size = _read_parquet_footer_size(delete_file_path)

            con.execute(
                "INSERT INTO ducklake_delete_file "
                "(delete_file_id, table_id, begin_snapshot, end_snapshot, "
                "data_file_id, path, path_is_relative, format, delete_count, "
                "file_size_bytes, footer_size, encryption_key) "
                "VALUES (?, ?, ?, NULL, ?, ?, ?, 'parquet', ?, ?, ?, NULL)",
                [
                    delete_file_id, table_id, new_snap, data_file_id,
                    delete_file_name, True, len(positions),
                    delete_file_size, delete_footer_size,
                ],
            )

        # Delete matched inlined rows
        if inlined_matched_df is not None and len(inlined_matched_df) > 0:
            key_pred = self._build_key_match_predicate(
                on, inlined_matched_df.select(on).unique(),
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

        if part_id is not None and group_list:
            total_file_size = 0
            current_data_file_id = first_data_file_id
            current_row_id = row_id_start

            for gk, gdf in group_list:
                partition_values = [str(v) for v in gk]
                hive_subdir = self._build_hive_path(
                    part_col_names, partition_values,
                )
                partition_dir = os.path.join(base, hive_subdir)
                os.makedirs(partition_dir, exist_ok=True)

                file_name = f"ducklake-{_uuid7()}.parquet"
                file_path = os.path.join(partition_dir, file_name)
                gdf.write_parquet(file_path)

                file_size = os.path.getsize(file_path)
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
            file_name = f"ducklake-{_uuid7()}.parquet"
            file_path = os.path.join(base, file_name)
            insert_df.write_parquet(file_path)

            file_size = os.path.getsize(file_path)
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

        # Record changes
        changes: list[str] = []
        if total_updated > 0:
            changes.append(f"deleted_from_table:{table_id}")
        if total_updated > 0 or total_inserted > 0:
            changes.append(f"inserted_into_table:{table_id}")
        self._record_change(new_snap, ",".join(changes))

        con.commit()
        return (total_updated, total_inserted)

    # ------------------------------------------------------------------
    # ALTER TABLE: ADD COLUMN
    # ------------------------------------------------------------------
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

    def add_column(
        self,
        table_name: str,
        column_name: str,
        polars_dtype: pl.DataType,
        *,
        default: Any = None,
        schema_name: str = "main",
    ) -> None:
        """Add a new column to an existing table."""
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise ValueError(msg)

        # Check column doesn't already exist
        columns = self._get_columns_for_table(table_id, snap_id)
        for _, col_name, _, _ in columns:
            if col_name == column_name:
                msg = f"Column '{column_name}' already exists in '{schema_name}.{table_name}'"
                raise ValueError(msg)

        # Compute new column_id and column_order
        max_col_id = self._get_max_column_id(table_id)
        max_col_order = self._get_max_column_order(table_id, snap_id)
        new_col_id = max_col_id + 1
        new_col_order = max_col_order + 1

        duckdb_type = polars_type_to_duckdb(polars_dtype)
        new_schema_ver = schema_ver + 1

        # Create snapshot
        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        # Default value handling
        default_str = str(default) if default is not None else None

        # Insert column
        con.execute(
            "INSERT INTO ducklake_column "
            "(column_id, begin_snapshot, end_snapshot, table_id, column_order, "
            "column_name, column_type, initial_default, default_value, "
            "nulls_allowed, parent_column) "
            "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, NULL)",
            [
                new_col_id, new_snap, table_id, new_col_order,
                column_name, duckdb_type, default_str, default_str, True,
            ],
        )

        # Record schema version
        con.execute(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version) "
            "VALUES (?, ?)",
            [new_snap, new_schema_ver],
        )

        # Record change
        self._record_change(new_snap, f"altered_table:{table_id}")

        con.commit()

    # ------------------------------------------------------------------
    # ALTER TABLE: RENAME COLUMN
    # ------------------------------------------------------------------

    def rename_column(
        self,
        table_name: str,
        old_column_name: str,
        new_column_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Rename a column in an existing table.

        DuckLake tracks renames by ending the old column row and inserting
        a new row with the same ``column_id`` but a different name. Old
        Parquet files keep the physical old name; the reader uses column
        history to resolve renames.
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise ValueError(msg)

        # Find the source column
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

        # Look up the full column row for column_order and nulls_allowed
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

        # Create snapshot
        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        # End the old column row
        con.execute(
            "UPDATE ducklake_column SET end_snapshot = ? "
            "WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
            [new_snap, table_id, col_id],
        )

        # Insert a new row with the same column_id but new name
        con.execute(
            "INSERT INTO ducklake_column "
            "(column_id, begin_snapshot, end_snapshot, table_id, column_order, "
            "column_name, column_type, initial_default, default_value, "
            "nulls_allowed, parent_column) "
            "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                col_id, new_snap, table_id, col_order,
                new_column_name, col_type, initial_default, default_value,
                nulls_allowed, parent_column,
            ],
        )

        # Record schema version
        con.execute(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version) "
            "VALUES (?, ?)",
            [new_snap, new_schema_ver],
        )

        # Record change
        self._record_change(new_snap, f"altered_table:{table_id}")

        con.commit()

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

    def drop_table(
        self,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Drop a table from the catalog.

        Sets ``end_snapshot`` on the table row, all its columns, and all
        its active data/delete files. Table stats are left as-is
        (matching DuckDB behaviour).
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise ValueError(msg)

        new_schema_ver = schema_ver + 1

        # Create snapshot
        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        # End the table row
        con.execute(
            "UPDATE ducklake_table SET end_snapshot = ? "
            "WHERE table_id = ? AND end_snapshot IS NULL",
            [new_snap, table_id],
        )

        # End all active columns
        self._end_all_columns(table_id, snap_id, new_snap)

        # End all active data files
        self._end_all_data_files(table_id, snap_id, new_snap)

        # End all active delete files
        self._end_all_delete_files(table_id, snap_id, new_snap)

        # End all active inlined rows
        self._end_all_inlined_rows(table_id, snap_id, new_snap)

        # Record schema version
        con.execute(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version) "
            "VALUES (?, ?)",
            [new_snap, new_schema_ver],
        )

        # Record change
        self._record_change(new_snap, f"dropped_table:{table_id}")

        con.commit()

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
            "WHERE schema_name = ? AND begin_snapshot <= ? "
            "AND (end_snapshot IS NULL OR end_snapshot > ?)",
            [schema_name, snapshot_id, snapshot_id],
        ).fetchone()
        return row[0] if row is not None else None

    def create_schema(
        self,
        schema_name: str,
    ) -> int:
        """Create a new schema in the catalog. Returns the new schema_id."""
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        if self._schema_exists(schema_name, snap_id) is not None:
            msg = f"Schema '{schema_name}' already exists"
            raise ValueError(msg)

        schema_id = next_cat_id
        new_next_cat_id = next_cat_id + 1
        new_schema_ver = schema_ver + 1

        # Create snapshot
        new_snap = self._create_snapshot(new_schema_ver, new_next_cat_id, next_file_id)

        # Insert schema
        schema_uuid = str(uuid.uuid4())
        schema_path = f"{schema_name}/"
        con.execute(
            "INSERT INTO ducklake_schema "
            "(schema_id, schema_uuid, begin_snapshot, end_snapshot, "
            "schema_name, path, path_is_relative) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?)",
            [schema_id, schema_uuid, new_snap, schema_name, schema_path, True],
        )

        # Record schema version
        con.execute(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version) "
            "VALUES (?, ?)",
            [new_snap, new_schema_ver],
        )

        # Record change
        safe_schema = schema_name.replace('"', '""')
        self._record_change(new_snap, f'created_schema:"{safe_schema}"')

        con.commit()
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

    def drop_schema(
        self,
        schema_name: str,
        *,
        cascade: bool = False,
    ) -> None:
        """Drop a schema from the catalog.

        Parameters
        ----------
        schema_name
            Name of the schema to drop.
        cascade
            If True, drop all tables in the schema first.
            If False (default), raise if the schema contains tables.
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        schema_id = self._schema_exists(schema_name, snap_id)
        if schema_id is None:
            msg = f"Schema '{schema_name}' not found"
            raise ValueError(msg)

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

        # Create snapshot
        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        # Build change description
        changes: list[str] = []

        # If cascade, drop all tables first
        for table_id, _table_name in tables:
            # End the table row
            con.execute(
                "UPDATE ducklake_table SET end_snapshot = ? "
                "WHERE table_id = ? AND end_snapshot IS NULL",
                [new_snap, table_id],
            )
            self._end_all_columns(table_id, snap_id, new_snap)
            self._end_all_data_files(table_id, snap_id, new_snap)
            self._end_all_delete_files(table_id, snap_id, new_snap)
            changes.append(f"dropped_table:{table_id}")

        # End the schema row
        con.execute(
            "UPDATE ducklake_schema SET end_snapshot = ? "
            "WHERE schema_id = ? AND end_snapshot IS NULL",
            [new_snap, schema_id],
        )
        changes.append(f"dropped_schema:{schema_id}")

        # Record schema version
        con.execute(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version) "
            "VALUES (?, ?)",
            [new_snap, new_schema_ver],
        )

        # Record change (DuckDB puts dropped_schema first when cascade)
        # Re-order: schema first, then tables — matching DuckDB output
        # Actually DuckDB does: dropped_schema:1,dropped_table:2 (schema first)
        # Let's match that order
        schema_changes = [c for c in changes if c.startswith("dropped_schema")]
        table_changes = [c for c in changes if c.startswith("dropped_table")]
        ordered = schema_changes + table_changes
        self._record_change(new_snap, ",".join(ordered))

        con.commit()

    # ------------------------------------------------------------------
    # RENAME TABLE
    # ------------------------------------------------------------------

    def rename_table(
        self,
        old_table_name: str,
        new_table_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Rename a table in the catalog.

        DuckLake tracks renames by ending the old table row and inserting
        a new row with the same ``table_id`` and ``table_uuid`` but a
        different name. The path stays the same (old directory).
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        table_id = self._table_exists(old_table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{old_table_name}' not found"
            raise ValueError(msg)

        if self._table_exists(new_table_name, schema_name, snap_id) is not None:
            msg = f"Table '{schema_name}.{new_table_name}' already exists"
            raise ValueError(msg)

        # Get full table row details
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

        # Create snapshot
        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        # End the old table row
        con.execute(
            "UPDATE ducklake_table SET end_snapshot = ? "
            "WHERE table_id = ? AND end_snapshot IS NULL",
            [new_snap, table_id],
        )

        # Insert new table row with same table_id/uuid but new name
        # Path stays the same (pointing to old directory)
        con.execute(
            "INSERT INTO ducklake_table "
            "(table_id, table_uuid, begin_snapshot, end_snapshot, schema_id, "
            "table_name, path, path_is_relative) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?, ?)",
            [table_id, table_uuid, new_snap, schema_id,
             new_table_name, table_path, path_is_relative],
        )

        # Record schema version
        con.execute(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version) "
            "VALUES (?, ?)",
            [new_snap, new_schema_ver],
        )

        # Record change (DuckDB uses created_table for renames)
        safe_schema = schema_name.replace('"', '""')
        safe_table = new_table_name.replace('"', '""')
        self._record_change(
            new_snap, f'created_table:"{safe_schema}"."{safe_table}"'
        )

        con.commit()

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

    def drop_column(
        self,
        table_name: str,
        column_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Drop a column from an existing table."""
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise ValueError(msg)

        # Find the column
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

        # Create snapshot
        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        # End the target column
        con.execute(
            "UPDATE ducklake_column SET end_snapshot = ? "
            "WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
            [new_snap, table_id, target_col_id],
        )

        # End any descendant columns (for compound types)
        self._end_descendant_columns(table_id, target_col_id, new_snap)

        # Record schema version
        con.execute(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version) "
            "VALUES (?, ?)",
            [new_snap, new_schema_ver],
        )

        # Record change
        self._record_change(new_snap, f"altered_table:{table_id}")

        con.commit()

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

    def set_partitioned_by(
        self,
        table_name: str,
        column_names: list[str],
        *,
        schema_name: str = "main",
    ) -> None:
        """Set identity-transform partitioning on an existing table.

        Equivalent to ``ALTER TABLE t SET PARTITIONED BY (col1, col2, ...)``.
        Future inserts will write one Parquet file per unique combination
        of partition column values, using Hive-style directory layout.
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        table_id = self._table_exists(table_name, schema_name, snap_id)
        if table_id is None:
            msg = f"Table '{schema_name}.{table_name}' not found"
            raise ValueError(msg)

        # Resolve column_ids for the requested partition columns
        columns = self._get_columns_for_table(table_id, snap_id)
        col_name_to_id: dict[str, int] = {c[1]: c[0] for c in columns}
        partition_col_ids: list[int] = []
        for name in column_names:
            if name not in col_name_to_id:
                msg = f"Column '{name}' not found in '{schema_name}.{table_name}'"
                raise ValueError(msg)
            partition_col_ids.append(col_name_to_id[name])

        # Allocate partition_id from next_catalog_id
        partition_id = next_cat_id
        new_next_cat_id = next_cat_id + 1
        new_schema_ver = schema_ver + 1

        # Create snapshot
        new_snap = self._create_snapshot(new_schema_ver, new_next_cat_id, next_file_id)

        # Insert partition_info
        con.execute(
            "INSERT INTO ducklake_partition_info "
            "(partition_id, table_id, begin_snapshot, end_snapshot) "
            "VALUES (?, ?, ?, NULL)",
            [partition_id, table_id, new_snap],
        )

        # Insert partition_column rows
        for key_index, col_id in enumerate(partition_col_ids):
            con.execute(
                "INSERT INTO ducklake_partition_column "
                "(partition_id, table_id, partition_key_index, column_id, transform) "
                "VALUES (?, ?, ?, ?, 'identity')",
                [partition_id, table_id, key_index, col_id],
            )

        # Record schema version
        con.execute(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version) "
            "VALUES (?, ?)",
            [new_snap, new_schema_ver],
        )

        # Record change
        self._record_change(new_snap, f"altered_table:{table_id}")

        con.commit()

    # ------------------------------------------------------------------
    # EXPIRE SNAPSHOTS
    # ------------------------------------------------------------------

    def expire_snapshots(
        self,
        *,
        older_than_snapshot: int | None = None,
        keep_last_n: int | None = None,
    ) -> int:
        """Expire old snapshots and clean up associated metadata.

        Removes snapshot rows, snapshot_changes entries, and metadata
        entries (data files, delete files, column stats, partition
        values) whose ``end_snapshot`` falls within the expired range.
        This is a metadata-only operation — actual Parquet file deletion
        is handled by :meth:`vacuum`.

        Parameters
        ----------
        older_than_snapshot
            Expire all snapshots with ``snapshot_id < older_than_snapshot``.
        keep_last_n
            Keep the most recent *n* snapshots, expire the rest.
            Cannot be combined with *older_than_snapshot*.

        Returns
        -------
        int
            The number of snapshots expired.
        """
        if older_than_snapshot is not None and keep_last_n is not None:
            msg = "Cannot specify both older_than_snapshot and keep_last_n"
            raise ValueError(msg)
        if older_than_snapshot is None and keep_last_n is None:
            msg = "Must specify either older_than_snapshot or keep_last_n"
            raise ValueError(msg)

        con = self._connect()

        # Determine the expiry boundary
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
            # Expire everything before the Nth-from-last
            older_than_snapshot = all_ids[keep_last_n - 1]

        assert older_than_snapshot is not None

        # Find snapshots to expire
        expire_rows = con.execute(
            "SELECT snapshot_id FROM ducklake_snapshot "
            "WHERE snapshot_id < ?",
            [older_than_snapshot],
        ).fetchall()
        expire_ids = [r[0] for r in expire_rows]

        if not expire_ids:
            return 0

        # Clean up metadata for files that ended within the expired range.
        # Files with end_snapshot <= older_than_snapshot are no longer
        # reachable by any remaining snapshot.
        boundary = older_than_snapshot

        # Delete file column stats for expired data files
        con.execute(
            "DELETE FROM ducklake_file_column_stats "
            "WHERE data_file_id IN ("
            "  SELECT data_file_id FROM ducklake_data_file "
            "  WHERE end_snapshot IS NOT NULL AND end_snapshot <= ?"
            ")",
            [boundary],
        )

        # Delete partition values for expired data files
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
            pass  # table may not exist

        # Delete expired data files
        con.execute(
            "DELETE FROM ducklake_data_file "
            "WHERE end_snapshot IS NOT NULL AND end_snapshot <= ?",
            [boundary],
        )

        # Delete expired delete files
        con.execute(
            "DELETE FROM ducklake_delete_file "
            "WHERE end_snapshot IS NOT NULL AND end_snapshot <= ?",
            [boundary],
        )

        # Delete snapshot_changes for expired snapshots
        for sid in expire_ids:
            con.execute(
                "DELETE FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
                [sid],
            )

        # Delete the snapshot rows themselves
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

    def vacuum(self) -> int:
        """Delete orphaned Parquet files not referenced by any catalog entry.

        Scans all data and delete file paths stored in the catalog
        (including files that haven't been expired yet), then walks the
        data directory and removes any ``.parquet`` files that are not
        in the catalog.

        Returns the number of files deleted.
        """
        con = self._connect()

        # Collect all referenced file paths (relative to data_path)
        # from both data files and delete files
        referenced: set[str] = set()

        # Get all schema/table paths for resolving relative file paths
        schemas = con.execute(
            "SELECT DISTINCT path, path_is_relative FROM ducklake_schema"
        ).fetchall()
        tables = con.execute(
            "SELECT DISTINCT t.path, t.path_is_relative, s.path, s.path_is_relative "
            "FROM ducklake_table t "
            "JOIN ducklake_schema s ON t.schema_id = s.schema_id"
        ).fetchall()

        data_base = self.data_path

        # Build full paths for all data files
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
            referenced.add(os.path.normpath(abs_path))

        # Build full paths for all delete files
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
            referenced.add(os.path.normpath(abs_path))

        # Walk the data directory and find all .parquet files
        deleted_count = 0
        for dirpath, _dirnames, filenames in os.walk(data_base):
            for fname in filenames:
                if fname.endswith(".parquet"):
                    full_path = os.path.normpath(os.path.join(dirpath, fname))
                    if full_path not in referenced:
                        os.remove(full_path)
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
                "WHERE v.view_name = ? AND s.schema_name = ? "
                "AND ? >= v.begin_snapshot AND (? < v.end_snapshot OR v.end_snapshot IS NULL) "
                "AND ? >= s.begin_snapshot AND (? < s.end_snapshot OR s.end_snapshot IS NULL)",
                [view_name, schema_name, snapshot_id, snapshot_id, snapshot_id, snapshot_id],
            ).fetchone()
        except Exception as exc:
            if self._backend.is_table_not_found(exc):
                return None
            raise
        return row[0] if row is not None else None

    def create_view(
        self,
        view_name: str,
        sql: str,
        *,
        schema_name: str = "main",
        or_replace: bool = False,
        column_aliases: str = "",
    ) -> int:
        """Create a view in the catalog. Returns the new view_id.

        Parameters
        ----------
        view_name
            Name of the view to create.
        sql
            The SQL definition of the view.
        schema_name
            Schema name (default: "main").
        or_replace
            If True, replace an existing view with the same name.
        column_aliases
            Comma-separated column aliases (empty string if none).
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        existing_view_id = self._view_exists(view_name, schema_name, snap_id)

        if existing_view_id is not None and not or_replace:
            msg = f"View '{schema_name}.{view_name}' already exists"
            raise ValueError(msg)

        # Resolve schema
        schema_id, _schema_path, _schema_path_rel = self._resolve_schema_info(
            schema_name, snap_id
        )

        # Allocate view_id
        view_id = next_cat_id
        new_next_cat_id = next_cat_id + 1
        new_schema_ver = schema_ver + 1

        # Create snapshot
        new_snap = self._create_snapshot(new_schema_ver, new_next_cat_id, next_file_id)

        changes: list[str] = []

        # If replacing, end the old view row
        if existing_view_id is not None:
            con.execute(
                "UPDATE ducklake_view SET end_snapshot = ? "
                "WHERE view_id = ? AND end_snapshot IS NULL",
                [new_snap, existing_view_id],
            )
            changes.append(f"dropped_view:{existing_view_id}")

        # Insert the new view row
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

        # Record schema version
        con.execute(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version) "
            "VALUES (?, ?)",
            [new_snap, new_schema_ver],
        )

        # Record change
        self._record_change(new_snap, ",".join(changes))

        con.commit()
        return view_id

    # ------------------------------------------------------------------
    # DROP VIEW
    # ------------------------------------------------------------------

    def drop_view(
        self,
        view_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        """Drop a view from the catalog.

        Sets ``end_snapshot`` on the view row.
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        view_id = self._view_exists(view_name, schema_name, snap_id)
        if view_id is None:
            msg = f"View '{schema_name}.{view_name}' not found"
            raise ValueError(msg)

        new_schema_ver = schema_ver + 1

        # Create snapshot
        new_snap = self._create_snapshot(new_schema_ver, next_cat_id, next_file_id)

        # End the view row
        con.execute(
            "UPDATE ducklake_view SET end_snapshot = ? "
            "WHERE view_id = ? AND end_snapshot IS NULL",
            [new_snap, view_id],
        )

        # Record schema version
        con.execute(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version) "
            "VALUES (?, ?)",
            [new_snap, new_schema_ver],
        )

        # Record change
        self._record_change(new_snap, f"dropped_view:{view_id}")

        con.commit()

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
            base = os.path.join(base, schema_path)
        else:
            base = schema_path
        if table_path_rel:
            base = os.path.join(base, table_path)
        else:
            base = table_path
        return os.path.join(base, file_path)
