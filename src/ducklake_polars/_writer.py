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

from ducklake_polars._backend import SQLiteBackend, create_backend
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
    Writes metadata to a DuckLake catalog (SQLite only).

    Handles snapshot creation, table/column registration, Parquet file
    writing, and statistics computation. Produces catalogs fully
    interoperable with DuckDB's DuckLake extension.
    """

    def __init__(
        self,
        metadata_path: str,
        *,
        data_path_override: str | None = None,
    ) -> None:
        self._backend = create_backend(metadata_path)
        if not isinstance(self._backend, SQLiteBackend):
            msg = "Write support is currently limited to SQLite backends"
            raise ValueError(msg)
        self._metadata_path = metadata_path
        self._data_path_override = data_path_override
        self._con: Any = None

    def _connect(self) -> Any:
        if self._con is None:
            self._con = self._backend.connect_writable()
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
        """Insert a snapshot_changes record."""
        con = self._connect()
        con.execute(
            "INSERT INTO ducklake_snapshot_changes "
            "(snapshot_id, changes_made, author, commit_message, commit_extra_info) "
            "VALUES (?, ?, NULL, NULL, NULL)",
            [snapshot_id, changes_made],
        )

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
                "VALUES (?, ?, ?, ?, ?, 0)",
                [mapping_id, col_id, col_name, col_id, parent_col],
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
            "VALUES (?, ?, ?, NULL, ?, ?, ?, 1)",
            [table_id, table_uuid, new_snap, schema_id, table_name, table_path],
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
                    1 if cd.nulls_allowed else 0,
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
                nan_int = 1 if _contains_nan(series) else 0

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

    def insert_data(
        self,
        df: pl.DataFrame,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        """
        Insert a DataFrame into an existing table.

        Writes a Parquet file and registers it in the catalog with
        column statistics. Returns the new snapshot ID.
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
        os.makedirs(base, exist_ok=True)

        # Generate Parquet file name (UUID7)
        file_name = f"ducklake-{_uuid7()}.parquet"
        file_path = os.path.join(base, file_name)

        # Write Parquet
        df.write_parquet(file_path)

        file_size = os.path.getsize(file_path)
        footer_size = _read_parquet_footer_size(file_path)
        record_count = len(df)

        # Get current table stats for row_id_start
        existing_stats = self._get_table_stats(table_id)
        if existing_stats is not None:
            row_id_start = existing_stats[1]  # next_row_id
        else:
            row_id_start = 0

        # Allocate file ID
        data_file_id = next_file_id
        new_next_file_id = next_file_id + 1

        # Create snapshot (schema_version unchanged for DML)
        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        # Register name mapping (Polars doesn't write Parquet field_ids,
        # so DuckLake needs a map_by_name mapping to resolve columns)
        mapping_id = self._register_name_mapping(table_id, columns)

        # Register data file
        con.execute(
            "INSERT INTO ducklake_data_file "
            "(data_file_id, table_id, begin_snapshot, end_snapshot, file_order, "
            "path, path_is_relative, file_format, record_count, file_size_bytes, "
            "footer_size, row_id_start, partition_id, encryption_key, "
            "partial_file_info, mapping_id) "
            "VALUES (?, ?, ?, NULL, NULL, ?, 1, 'parquet', ?, ?, ?, ?, NULL, NULL, NULL, ?)",
            [
                data_file_id,
                table_id,
                new_snap,
                file_name,
                record_count,
                file_size,
                footer_size,
                row_id_start,
                mapping_id,
            ],
        )

        # Compute and register per-file column stats
        col_stats = self._compute_file_column_stats(df, columns)
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

        # Update table stats
        new_record_count = (existing_stats[0] if existing_stats else 0) + record_count
        new_next_row_id = row_id_start + record_count
        new_file_size = (existing_stats[2] if existing_stats else 0) + file_size

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

        # Update table column stats (aggregate)
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

            contains_null = 1 if null_count and null_count > 0 else 0
            contains_nan = nan_int if nan_int is not None else None

            if existing_col_stat is not None:
                # Merge
                merged_null = 1 if (existing_col_stat[0] or contains_null) else 0
                merged_nan = None
                if existing_col_stat[1] is not None or contains_nan is not None:
                    merged_nan = 1 if (existing_col_stat[1] or (contains_nan or 0)) else 0
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

        # Record change
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
        data_file_id = next_file_id
        new_next_file_id = next_file_id + 1 if record_count > 0 else next_file_id

        # Create snapshot
        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        # End all existing files
        self._end_all_data_files(table_id, snap_id, new_snap)
        self._end_all_delete_files(table_id, snap_id, new_snap)

        if record_count > 0:
            # Write new Parquet file
            file_name = f"ducklake-{_uuid7()}.parquet"
            file_path = os.path.join(base, file_name)
            df.write_parquet(file_path)

            file_size = os.path.getsize(file_path)
            footer_size = _read_parquet_footer_size(file_path)

            # Register name mapping
            mapping_id = self._register_name_mapping(table_id, columns)

            # Register data file (row_id_start resets to 0)
            con.execute(
                "INSERT INTO ducklake_data_file "
                "(data_file_id, table_id, begin_snapshot, end_snapshot, file_order, "
                "path, path_is_relative, file_format, record_count, file_size_bytes, "
                "footer_size, row_id_start, partition_id, encryption_key, "
                "partial_file_info, mapping_id) "
                "VALUES (?, ?, ?, NULL, NULL, ?, 1, 'parquet', ?, ?, ?, 0, NULL, NULL, NULL, ?)",
                [data_file_id, table_id, new_snap, file_name, record_count, file_size, footer_size, mapping_id],
            )

            # Compute and register per-file column stats
            col_stats = self._compute_file_column_stats(df, columns)
            for col_id, _col_name, value_count, null_count, min_val, max_val, nan_int in col_stats:
                con.execute(
                    "INSERT INTO ducklake_file_column_stats "
                    "(data_file_id, table_id, column_id, column_size_bytes, "
                    "value_count, null_count, min_value, max_value, contains_nan, extra_stats) "
                    "VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, NULL)",
                    [data_file_id, table_id, col_id, value_count, null_count, min_val, max_val, nan_int],
                )

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
                contains_null = 1 if null_count and null_count > 0 else 0
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

        Creates Iceberg-compatible position-delete Parquet files for each
        affected data file. Returns the number of deleted rows. If no rows
        match the predicate, no snapshot is created and 0 is returned.
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )

        data_files = self._get_active_data_files(table_id, snap_id)
        if not data_files:
            return 0

        # Build the output directory for delete files
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

        # For each data file, evaluate the predicate and collect delete positions
        # Each entry: (data_file_id, abs_data_path, local_positions)
        pending_deletes: list[tuple[int, str, list[int]]] = []
        total_deleted = 0

        for data_file_id, rel_path, path_is_rel, record_count, row_id_start in data_files:
            abs_path = self._resolve_file_path(
                rel_path, path_is_rel,
                table_path, table_path_rel,
                schema_path, schema_path_rel,
            )
            df = pl.read_parquet(abs_path)
            # Evaluate predicate to find matching rows
            # Use with_columns (not select) so scalar predicates like pl.lit(True)
            # are broadcast to all rows
            mask = df.with_columns(predicate.alias("__delete_mask__"))["__delete_mask__"]
            positions = [i for i, v in enumerate(mask.to_list()) if v]
            if positions:
                pending_deletes.append((data_file_id, abs_path, positions))
                total_deleted += len(positions)

        if total_deleted == 0:
            return 0

        # Allocate file IDs for all delete files from the shared counter
        current_file_id = next_file_id
        new_next_file_id = next_file_id + len(pending_deletes)

        # Create snapshot
        new_snap = self._create_snapshot(schema_ver, next_cat_id, new_next_file_id)

        # Write delete files and register them
        for data_file_id, abs_data_path, positions in pending_deletes:
            delete_file_id = current_file_id
            current_file_id += 1

            # Build the position-delete DataFrame
            # file_path = absolute resolved path (matches DuckDB behavior)
            # pos = 0-based local position within the file
            delete_df = pl.DataFrame({
                "file_path": [abs_data_path] * len(positions),
                "pos": pl.Series(positions, dtype=pl.Int64),
            })

            # Write delete Parquet file
            delete_file_name = f"ducklake-{_uuid7()}-delete.parquet"
            delete_file_path = os.path.join(base, delete_file_name)
            delete_df.write_parquet(delete_file_path)

            delete_file_size = os.path.getsize(delete_file_path)
            delete_footer_size = _read_parquet_footer_size(delete_file_path)

            # Register delete file in catalog
            con.execute(
                "INSERT INTO ducklake_delete_file "
                "(delete_file_id, table_id, begin_snapshot, end_snapshot, "
                "data_file_id, path, path_is_relative, format, delete_count, "
                "file_size_bytes, footer_size, encryption_key) "
                "VALUES (?, ?, ?, NULL, ?, ?, 1, 'parquet', ?, ?, ?, NULL)",
                [
                    delete_file_id,
                    table_id,
                    new_snap,
                    data_file_id,
                    delete_file_name,
                    len(positions),
                    delete_file_size,
                    delete_footer_size,
                ],
            )

        # Record change (table stats are NOT updated on delete, matching DuckDB)
        self._record_change(new_snap, f"deleted_from_table:{table_id}")
        con.commit()
        return total_deleted

    # ------------------------------------------------------------------
    # UPDATE (delete + insert in a single snapshot)
    # ------------------------------------------------------------------

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

        Creates position-delete files for the old rows and a new data file
        with the updated rows, all in a single snapshot. Returns the number
        of rows updated. If no rows match, no snapshot is created.
        """
        con = self._connect()
        snap_id, schema_ver, next_cat_id, next_file_id = self._get_latest_snapshot()

        table_id, table_path, table_path_rel, schema_path, schema_path_rel = (
            self._get_table_info(table_name, schema_name, snap_id)
        )

        columns = self._get_columns_for_table(table_id, snap_id)
        data_files = self._get_active_data_files(table_id, snap_id)
        if not data_files:
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

        # Evaluate predicate on each data file
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

        # Allocate file IDs: delete files first, then one data file
        n_delete_files = len(pending_deletes)
        new_data_file_id = next_file_id + n_delete_files
        new_next_file_id = new_data_file_id + 1

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
                "VALUES (?, ?, ?, NULL, ?, ?, 1, 'parquet', ?, ?, ?, NULL)",
                [
                    delete_file_id, table_id, new_snap, data_file_id,
                    delete_file_name, len(positions),
                    delete_file_size, delete_footer_size,
                ],
            )

        # Write new data file with updated rows
        file_name = f"ducklake-{_uuid7()}.parquet"
        file_path = os.path.join(base, file_name)
        updated_df.write_parquet(file_path)

        file_size = os.path.getsize(file_path)
        footer_size = _read_parquet_footer_size(file_path)

        # Row ID start
        existing_stats = self._get_table_stats(table_id)
        row_id_start_new = existing_stats[1] if existing_stats else 0

        # Register name mapping
        mapping_id = self._register_name_mapping(table_id, columns)

        # Register data file
        con.execute(
            "INSERT INTO ducklake_data_file "
            "(data_file_id, table_id, begin_snapshot, end_snapshot, file_order, "
            "path, path_is_relative, file_format, record_count, file_size_bytes, "
            "footer_size, row_id_start, partition_id, encryption_key, "
            "partial_file_info, mapping_id) "
            "VALUES (?, ?, ?, NULL, NULL, ?, 1, 'parquet', ?, ?, ?, ?, NULL, NULL, NULL, ?)",
            [
                new_data_file_id, table_id, new_snap, file_name,
                total_updated, file_size, footer_size,
                row_id_start_new, mapping_id,
            ],
        )

        # Compute and register per-file column stats
        col_stats = self._compute_file_column_stats(updated_df, columns)
        for col_id, _col_name, value_count, null_count, min_val, max_val, nan_int in col_stats:
            con.execute(
                "INSERT INTO ducklake_file_column_stats "
                "(data_file_id, table_id, column_id, column_size_bytes, "
                "value_count, null_count, min_value, max_value, contains_nan, extra_stats) "
                "VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, NULL)",
                [
                    new_data_file_id, table_id, col_id,
                    value_count, null_count, min_val, max_val, nan_int,
                ],
            )

        # Update table stats
        new_record_count = (existing_stats[0] if existing_stats else 0) + total_updated
        new_next_row_id = row_id_start_new + total_updated
        new_file_size = (existing_stats[2] if existing_stats else 0) + file_size

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

        # Update table column stats
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

            contains_null = 1 if null_count and null_count > 0 else 0
            contains_nan = nan_int if nan_int is not None else None

            if existing_col_stat is not None:
                merged_null = 1 if (existing_col_stat[0] or contains_null) else 0
                merged_nan = None
                if existing_col_stat[1] is not None or contains_nan is not None:
                    merged_nan = 1 if (existing_col_stat[1] or (contains_nan or 0)) else 0
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

        # Record change: both insert and delete in the same snapshot
        self._record_change(
            new_snap,
            f"inserted_into_table:{table_id},deleted_from_table:{table_id}",
        )

        con.commit()
        return total_updated

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
            "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, 1, NULL)",
            [
                new_col_id, new_snap, table_id, new_col_order,
                column_name, duckdb_type, default_str, default_str,
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
