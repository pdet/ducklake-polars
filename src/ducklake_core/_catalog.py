"""DuckLake metadata catalog reader."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ducklake_core._storage as storage
from ducklake_core._backend import create_backend

import pyarrow as pa

SUPPORTED_DUCKLAKE_VERSIONS = {"0.3", "0.4"}

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class FileInfo:
    """Information about a DuckLake data file."""

    data_file_id: int
    path: str
    path_is_relative: bool
    record_count: int
    file_size_bytes: int
    row_id_start: int
    partition_id: int | None
    mapping_id: int | None
    begin_snapshot: int = 0


@dataclass
class DeleteFileInfo:
    """Information about a DuckLake delete file."""

    delete_file_id: int
    data_file_id: int
    path: str
    path_is_relative: bool
    delete_count: int
    partial_max: int | None = None


@dataclass
class ColumnInfo:
    """Information about a DuckLake column."""

    column_id: int
    column_name: str
    column_type: str
    column_order: int
    parent_column: int | None
    nulls_allowed: bool


@dataclass
class ColumnStats:
    """Per-file column statistics."""

    data_file_id: int
    column_id: int
    null_count: int | None
    min_value: str | None
    max_value: str | None


@dataclass
class TableInfo:
    """Information about a DuckLake table."""

    table_id: int
    table_name: str
    schema_id: int
    table_path: str
    table_path_is_relative: bool
    schema_path: str
    schema_path_is_relative: bool


@dataclass
class SnapshotInfo:
    """DuckLake snapshot information."""

    snapshot_id: int
    schema_version: int
    next_file_id: int


@dataclass
class InlinedDataTableInfo:
    """Information about an inlined data table."""

    table_id: int
    table_name: str
    schema_version: int


@dataclass
class ColumnHistoryEntry:
    """A single column definition across a snapshot range (for rename detection)."""

    column_id: int
    column_name: str
    begin_snapshot: int
    end_snapshot: int | None
    parent_column: int | None = None
    column_type: str = ""
    column_order: int = 0


@dataclass
class PartitionInfo:
    """Partition spec for a table."""

    partition_id: int


@dataclass
class PartitionColumnDef:
    """A column in a partition spec."""

    partition_id: int
    column_id: int
    partition_key_index: int
    transform: str


@dataclass
class FilePartitionValue:
    """A partition value for a specific file and partition key."""

    data_file_id: int
    partition_key_index: int
    partition_value: str | None


@dataclass
class SortKeyDef:
    """A sort key expression for a table."""

    sort_key_index: int
    column_name: str
    sort_direction: str  # "asc" or "desc"
    null_order: str  # "nulls_first" or "nulls_last"


class DuckLakeCatalogReader:
    """
    Reads metadata from a DuckLake catalog database.

    Connects to the DuckLake metadata catalog (a SQLite database file or
    PostgreSQL database) and provides methods to query tables, columns,
    files, and statistics.
    """

    def __init__(
        self,
        metadata_path: str,
        *,
        data_path_override: str | None = None,
    ) -> None:
        self._backend = create_backend(metadata_path)
        self._data_path_override = data_path_override
        self._con: Any = None
        self._data_path: str | None = None
        self._catalog_version: str | None = None

    def _connect(self) -> Any:
        if self._con is None:
            self._con = self._backend.connect()
            self._load_metadata()
        return self._con

    def _load_metadata(self) -> None:
        """Load version and data_path from ducklake_metadata in a single query."""
        rows = self._con.execute(
            "SELECT key, value FROM ducklake_metadata WHERE key IN ('version', 'data_path')"
        ).fetchall()
        meta = {r[0]: r[1] for r in rows}
        version = meta.get("version")
        if version is None:
            msg = "No version found in ducklake_metadata — is this a valid DuckLake catalog?"
            raise ValueError(msg)
        if version not in SUPPORTED_DUCKLAKE_VERSIONS:
            msg = (
                f"Unsupported DuckLake catalog version '{version}'. "
                f"Supported versions: {', '.join(sorted(SUPPORTED_DUCKLAKE_VERSIONS))}"
            )
            raise ValueError(msg)
        self._catalog_version = version
        if self._data_path_override is None and "data_path" in meta:
            self._data_path = meta["data_path"]

    def _sql(self, query: str) -> str:
        """Translate ``?`` placeholders to the backend's parameter style.

        Only use on static SQL strings.  Do not apply to queries built
        with f-strings containing database-sourced identifiers — those
        could contain literal ``?`` characters that would be corrupted.
        Use ``self._backend.placeholder`` directly instead.
        """
        if self._backend.placeholder != "?":
            return query.replace("?", self._backend.placeholder)
        return query

    def __enter__(self) -> DuckLakeCatalogReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
            self._data_path = None

    @property
    def data_path(self) -> str:
        """Get the base data path for resolving relative file paths."""
        if self._data_path is None:
            if self._data_path_override is not None:
                self._data_path = self._data_path_override
            else:
                con = self._connect()
                row = con.execute(
                    self._sql("SELECT value FROM ducklake_metadata WHERE key = 'data_path'")
                ).fetchone()
                if row is None:
                    msg = "No data_path found in ducklake_metadata"
                    raise ValueError(msg)
                self._data_path = row[0]
        return self._data_path

    def resolve_data_file_path(self, file_path: str, file_is_relative: bool, table: TableInfo) -> str:
        """Resolve a data file path including schema and table path components."""
        if not file_is_relative:
            return file_path

        # Build the full path: data_path / schema_path / table_path / file_path
        base = self.data_path

        if table.schema_path_is_relative:
            base = storage.join_path(base, table.schema_path)
        else:
            base = table.schema_path

        if table.table_path_is_relative:
            base = storage.join_path(base, table.table_path)
        else:
            base = table.table_path

        return storage.join_path(base, file_path)

    def get_current_snapshot(self) -> SnapshotInfo:
        """Get the current (latest) snapshot."""
        con = self._connect()
        row = con.execute("""
            SELECT snapshot_id, schema_version, next_file_id
            FROM ducklake_snapshot
            WHERE snapshot_id = (SELECT MAX(snapshot_id) FROM ducklake_snapshot)
        """).fetchone()
        if row is None:
            msg = "No snapshots found in catalog"
            raise ValueError(msg)
        return SnapshotInfo(
            snapshot_id=row[0],
            schema_version=row[1],
            next_file_id=row[2],
        )

    def get_snapshot_at_version(self, version: int) -> SnapshotInfo:
        """Get snapshot info at a specific version."""
        con = self._connect()
        row = con.execute(
            self._sql("""
            SELECT snapshot_id, schema_version, next_file_id
            FROM ducklake_snapshot
            WHERE snapshot_id = ?
            """),
            [version],
        ).fetchone()
        if row is None:
            msg = f"Snapshot version {version} not found"
            raise ValueError(msg)
        return SnapshotInfo(
            snapshot_id=row[0],
            schema_version=row[1],
            next_file_id=row[2],
        )

    def get_snapshot_at_time(self, timestamp: datetime | str) -> SnapshotInfo:
        """Get the snapshot at or before a given timestamp."""
        con = self._connect()
        ts_str = timestamp if isinstance(timestamp, str) else timestamp.isoformat()
        row = con.execute(
            self._sql("""
            SELECT snapshot_id, schema_version, next_file_id
            FROM ducklake_snapshot
            WHERE snapshot_time <= ?
            ORDER BY snapshot_id DESC
            LIMIT 1
            """),
            [ts_str],
        ).fetchone()
        if row is None:
            msg = f"No snapshot found at or before {timestamp}"
            raise ValueError(msg)
        return SnapshotInfo(
            snapshot_id=row[0],
            schema_version=row[1],
            next_file_id=row[2],
        )

    def get_table(self, table_name: str, schema_name: str, snapshot_id: int) -> TableInfo:
        """Get table info for a given table at a specific snapshot."""
        con = self._connect()
        row = con.execute(
            self._sql("""
            SELECT t.table_id, t.table_name, t.schema_id,
                   t.path, t.path_is_relative,
                   s.path, s.path_is_relative
            FROM ducklake_table t
            JOIN ducklake_schema s ON t.schema_id = s.schema_id
            WHERE t.table_name = ?
              AND s.schema_name = ?
              AND ? >= t.begin_snapshot
              AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
              AND ? >= s.begin_snapshot
              AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
            """),
            [table_name, schema_name, snapshot_id, snapshot_id, snapshot_id, snapshot_id],
        ).fetchone()
        if row is None:
            msg = f"Table '{schema_name}.{table_name}' not found at snapshot {snapshot_id}"
            raise ValueError(msg)
        return TableInfo(
            table_id=row[0],
            table_name=row[1],
            schema_id=row[2],
            table_path=row[3] or "",
            table_path_is_relative=bool(row[4]) if row[4] is not None else True,
            schema_path=row[5] or "",
            schema_path_is_relative=bool(row[6]) if row[6] is not None else True,
        )

    def get_columns(self, table_id: int, snapshot_id: int) -> list[ColumnInfo]:
        """Get top-level column definitions for a table at a specific snapshot."""
        all_cols = self.get_all_columns(table_id, snapshot_id)
        return [c for c in all_cols if c.parent_column is None]

    def get_all_columns(self, table_id: int, snapshot_id: int) -> list[ColumnInfo]:
        """Get all column definitions (including nested) for a table at a specific snapshot."""
        con = self._connect()
        rows = con.execute(
            self._sql("""
            SELECT column_id, column_name, column_type, column_order,
                   parent_column, nulls_allowed
            FROM ducklake_column
            WHERE table_id = ?
              AND ? >= begin_snapshot
              AND (? < end_snapshot OR end_snapshot IS NULL)
            ORDER BY column_order
            """),
            [table_id, snapshot_id, snapshot_id],
        ).fetchall()
        return [
            ColumnInfo(
                column_id=r[0],
                column_name=r[1],
                column_type=r[2],
                column_order=r[3],
                parent_column=r[4],
                nulls_allowed=bool(r[5]) if r[5] is not None else True,
            )
            for r in rows
        ]

    def get_data_files(self, table_id: int, snapshot_id: int) -> list[FileInfo]:
        """Get data files for a table at a specific snapshot."""
        con = self._connect()
        rows = con.execute(
            self._sql("""
            SELECT data_file_id, path, path_is_relative, record_count,
                   file_size_bytes, row_id_start, partition_id, mapping_id,
                   begin_snapshot
            FROM ducklake_data_file
            WHERE table_id = ?
              AND ? >= begin_snapshot
              AND (? < end_snapshot OR end_snapshot IS NULL)
            ORDER BY file_order, data_file_id
            """),
            [table_id, snapshot_id, snapshot_id],
        ).fetchall()
        return [
            FileInfo(
                data_file_id=r[0],
                path=r[1],
                path_is_relative=bool(r[2]) if r[2] is not None else True,
                record_count=r[3],
                file_size_bytes=r[4],
                row_id_start=r[5],
                partition_id=r[6],
                mapping_id=r[7],
                begin_snapshot=r[8],
            )
            for r in rows
        ]

    def get_delete_files(self, table_id: int, snapshot_id: int) -> list[DeleteFileInfo]:
        """Get delete files for a table at a specific snapshot."""
        con = self._connect()
        if self._catalog_version is not None and self._catalog_version >= "0.4":
            rows = con.execute(
                self._sql("""
                SELECT delete_file_id, data_file_id, path, path_is_relative,
                       delete_count, partial_max
                FROM ducklake_delete_file
                WHERE table_id = ?
                  AND ? >= begin_snapshot
                  AND (? < end_snapshot OR end_snapshot IS NULL)
                """),
                [table_id, snapshot_id, snapshot_id],
            ).fetchall()
            return [
                DeleteFileInfo(
                    delete_file_id=r[0],
                    data_file_id=r[1],
                    path=r[2],
                    path_is_relative=bool(r[3]) if r[3] is not None else True,
                    delete_count=r[4],
                    partial_max=r[5],
                )
                for r in rows
            ]
        rows = con.execute(
            self._sql("""
            SELECT delete_file_id, data_file_id, path, path_is_relative, delete_count
            FROM ducklake_delete_file
            WHERE table_id = ?
              AND ? >= begin_snapshot
              AND (? < end_snapshot OR end_snapshot IS NULL)
            """),
            [table_id, snapshot_id, snapshot_id],
        ).fetchall()
        return [
            DeleteFileInfo(
                delete_file_id=r[0],
                data_file_id=r[1],
                path=r[2],
                path_is_relative=bool(r[3]) if r[3] is not None else True,
                delete_count=r[4],
            )
            for r in rows
        ]

    def get_column_stats(
        self,
        table_id: int,
        data_file_ids: list[int],
        column_ids: list[int] | None = None,
    ) -> list[ColumnStats]:
        """Get per-file column statistics."""
        if not data_file_ids:
            return []
        if column_ids is not None and not column_ids:
            return []

        con = self._connect()

        ph = self._backend.placeholder
        placeholders = ",".join([ph] * len(data_file_ids))
        params: list[Any] = [table_id, *data_file_ids]

        query = f"""
            SELECT data_file_id, column_id, null_count, min_value, max_value
            FROM ducklake_file_column_stats
            WHERE table_id = {ph}
              AND data_file_id IN ({placeholders})
        """

        if column_ids:
            col_placeholders = ",".join([ph] * len(column_ids))
            query += f" AND column_id IN ({col_placeholders})"
            params.extend(column_ids)

        rows = con.execute(query, params).fetchall()
        return [
            ColumnStats(
                data_file_id=r[0],
                column_id=r[1],
                null_count=r[2],
                min_value=r[3],
                max_value=r[4],
            )
            for r in rows
        ]

    def has_column_changes(self, table_id: int) -> bool:
        """Fast check: has any column ever been dropped, renamed, or re-added?

        Returns True if any column in the table has ``end_snapshot IS NOT NULL``,
        which is a necessary condition for renames, drops, or type changes to
        exist.  When this returns False, ``get_column_history`` + ``_has_renames``
        can be skipped entirely.
        """
        con = self._connect()
        row = con.execute(
            self._sql(
                "SELECT EXISTS(SELECT 1 FROM ducklake_column "
                "WHERE table_id = ? AND end_snapshot IS NOT NULL)"
            ),
            [table_id],
        ).fetchone()
        return bool(row and row[0])

    def get_column_history(self, table_id: int) -> list[ColumnHistoryEntry]:
        """Get all column definitions across all snapshots (for rename detection).

        Includes both top-level and child columns (for struct field rename detection).
        """
        con = self._connect()
        rows = con.execute(
            self._sql("""
            SELECT column_id, column_name, begin_snapshot, end_snapshot,
                   parent_column, column_type, column_order
            FROM ducklake_column
            WHERE table_id = ?
            ORDER BY column_id, begin_snapshot
            """),
            [table_id],
        ).fetchall()
        return [
            ColumnHistoryEntry(
                column_id=r[0],
                column_name=r[1],
                begin_snapshot=r[2],
                end_snapshot=r[3],
                parent_column=r[4],
                column_type=r[5],
                column_order=r[6],
            )
            for r in rows
        ]

    def get_partition_info(self, table_id: int, snapshot_id: int) -> PartitionInfo | None:
        """Get the active partition spec for a table at a given snapshot."""
        con = self._connect()
        try:
            row = con.execute(
                self._sql("""
                SELECT partition_id
                FROM ducklake_partition_info
                WHERE table_id = ?
                  AND ? >= begin_snapshot
                  AND (? < end_snapshot OR end_snapshot IS NULL)
                """),
                [table_id, snapshot_id, snapshot_id],
            ).fetchone()
        except Exception as e:
            if self._backend.is_table_not_found(e):
                return None
            raise
        if row is None:
            return None
        return PartitionInfo(partition_id=row[0])

    def get_partition_columns(self, partition_id: int, table_id: int) -> list[PartitionColumnDef]:
        """Get partition columns for a partition spec."""
        con = self._connect()
        try:
            rows = con.execute(
                self._sql("""
                SELECT partition_id, column_id, partition_key_index, transform
                FROM ducklake_partition_column
                WHERE partition_id = ?
                  AND table_id = ?
                ORDER BY partition_key_index
                """),
                [partition_id, table_id],
            ).fetchall()
        except Exception as e:
            if self._backend.is_table_not_found(e):
                return []
            raise
        return [
            PartitionColumnDef(
                partition_id=r[0],
                column_id=r[1],
                partition_key_index=r[2],
                transform=r[3],
            )
            for r in rows
        ]

    def get_file_partition_values(
        self,
        table_id: int,
        data_file_ids: list[int],
    ) -> list[FilePartitionValue]:
        """Get partition values for a set of data files."""
        if not data_file_ids:
            return []

        con = self._connect()
        ph = self._backend.placeholder
        placeholders = ",".join([ph] * len(data_file_ids))
        try:
            rows = con.execute(
                f"""
                SELECT data_file_id, partition_key_index, partition_value
                FROM ducklake_file_partition_value
                WHERE table_id = {ph}
                  AND data_file_id IN ({placeholders})
                """,
                [table_id, *data_file_ids],
            ).fetchall()
        except Exception as e:
            if self._backend.is_table_not_found(e):
                return []
            raise
        return [
            FilePartitionValue(
                data_file_id=r[0],
                partition_key_index=r[1],
                partition_value=r[2],
            )
            for r in rows
        ]

    def get_sort_keys(self, table_id: int, snapshot_id: int) -> list[SortKeyDef]:
        """Get the active sort keys for a table at a given snapshot.

        Returns an ordered list of :class:`SortKeyDef` entries, or an
        empty list when no sort keys are defined.
        """
        con = self._connect()
        try:
            # First get the active sort_id
            row = con.execute(
                self._sql("""
                SELECT sort_id
                FROM ducklake_sort_info
                WHERE table_id = ?
                  AND ? >= begin_snapshot
                  AND (? < end_snapshot OR end_snapshot IS NULL)
                """),
                [table_id, snapshot_id, snapshot_id],
            ).fetchone()
        except Exception as e:
            if self._backend.is_table_not_found(e):
                return []
            raise
        if row is None:
            return []

        sort_id = row[0]

        try:
            rows = con.execute(
                self._sql("""
                SELECT se.sort_key_index, c.column_name,
                       se.sort_direction, se.null_order
                FROM ducklake_sort_expression se
                JOIN ducklake_column c
                  ON se.table_id = c.table_id
                  AND se.expression = CAST(c.column_id AS VARCHAR)
                WHERE se.sort_id = ?
                  AND se.table_id = ?
                  AND ? >= c.begin_snapshot
                  AND (? < c.end_snapshot OR c.end_snapshot IS NULL)
                ORDER BY se.sort_key_index
                """),
                [sort_id, table_id, snapshot_id, snapshot_id],
            ).fetchall()
        except Exception as e:
            if self._backend.is_table_not_found(e):
                return []
            raise

        return [
            SortKeyDef(
                sort_key_index=r[0],
                column_name=r[1],
                sort_direction=r[2] or "asc",
                null_order=r[3] or "nulls_last",
            )
            for r in rows
        ]

    def get_name_mapping(self, mapping_id: int) -> dict[int, str]:
        """Return ``{target_field_id: source_name}`` for a name mapping.

        Each data file records a ``mapping_id`` that links to the
        ``ducklake_name_mapping`` table.  The mapping tells us the
        physical column name in the Parquet file (``source_name``) and
        the column ID it corresponds to (``target_field_id``).
        """
        con = self._connect()
        rows = con.execute(
            self._sql(
                "SELECT source_name, target_field_id "
                "FROM ducklake_name_mapping "
                "WHERE mapping_id = ?"
            ),
            [mapping_id],
        ).fetchall()
        return {int(r[1]): r[0] for r in rows}

    def get_all_snapshots(self) -> list[tuple]:
        """Get all snapshots ordered by snapshot_id."""
        con = self._connect()
        return con.execute("""
            SELECT snapshot_id, schema_version, snapshot_time
            FROM ducklake_snapshot
            ORDER BY snapshot_id
        """).fetchall()

    def get_all_schemas(self, snapshot_id: int) -> list[tuple]:
        """Get all schemas visible at a given snapshot."""
        con = self._connect()
        return con.execute(
            self._sql("""
            SELECT schema_id, schema_name, path, path_is_relative
            FROM ducklake_schema
            WHERE begin_snapshot <= ?
              AND (end_snapshot IS NULL OR end_snapshot > ?)
            ORDER BY schema_id
            """),
            [snapshot_id, snapshot_id],
        ).fetchall()

    def get_all_tables(self, schema_id: int, snapshot_id: int) -> list[tuple]:
        """Get all tables in a schema visible at a given snapshot."""
        con = self._connect()
        return con.execute(
            self._sql("""
            SELECT table_id, table_name
            FROM ducklake_table
            WHERE schema_id = ?
              AND begin_snapshot <= ?
              AND (end_snapshot IS NULL OR end_snapshot > ?)
            ORDER BY table_id
            """),
            [schema_id, snapshot_id, snapshot_id],
        ).fetchall()

    def get_all_metadata(self) -> list[tuple]:
        """Get all key-value pairs from ducklake_metadata."""
        con = self._connect()
        return con.execute(
            "SELECT key, value FROM ducklake_metadata"
        ).fetchall()

    def get_data_files_in_range_with_snapshot(self, table_id: int, start_snapshot: int, end_snapshot: int) -> list[tuple[FileInfo, int]]:
        """Get data files added between two snapshots, with their begin_snapshot.

        Returns list of (FileInfo, begin_snapshot) tuples.
        """
        con = self._connect()
        rows = con.execute(
            self._sql("""
            SELECT data_file_id, path, path_is_relative, record_count,
                   file_size_bytes, row_id_start, partition_id, mapping_id,
                   begin_snapshot
            FROM ducklake_data_file
            WHERE table_id = ?
              AND begin_snapshot > ?
              AND begin_snapshot <= ?
            ORDER BY file_order, data_file_id
            """),
            [table_id, start_snapshot, end_snapshot],
        ).fetchall()
        return [
            (
                FileInfo(
                    data_file_id=r[0],
                    path=r[1],
                    path_is_relative=bool(r[2]) if r[2] is not None else True,
                    record_count=r[3],
                    file_size_bytes=r[4],
                    row_id_start=r[5],
                    partition_id=r[6],
                    mapping_id=r[7],
                    begin_snapshot=r[8],
                ),
                r[8],
            )
            for r in rows
        ]

    def get_delete_files_in_range(self, table_id: int, start_snapshot: int, end_snapshot: int) -> list[tuple[DeleteFileInfo, int]]:
        """Get delete files added between two snapshots (start exclusive, end inclusive).

        Returns list of (DeleteFileInfo, begin_snapshot) tuples.
        """
        con = self._connect()
        rows = con.execute(
            self._sql("""
            SELECT delete_file_id, data_file_id, path, path_is_relative,
                   delete_count, begin_snapshot
            FROM ducklake_delete_file
            WHERE table_id = ?
              AND begin_snapshot > ?
              AND begin_snapshot <= ?
            ORDER BY delete_file_id
            """),
            [table_id, start_snapshot, end_snapshot],
        ).fetchall()
        return [
            (
                DeleteFileInfo(
                    delete_file_id=r[0],
                    data_file_id=r[1],
                    path=r[2],
                    path_is_relative=bool(r[3]) if r[3] is not None else True,
                    delete_count=r[4],
                ),
                r[5],
            )
            for r in rows
        ]

    def get_data_file_by_id(self, data_file_id: int) -> FileInfo | None:
        """Get a specific data file by its ID, regardless of snapshot visibility.

        This intentionally does not filter on end_snapshot because the change
        data feed needs to read data files that may have been logically removed.
        """
        con = self._connect()
        row = con.execute(
            self._sql("""
            SELECT data_file_id, path, path_is_relative, record_count,
                   file_size_bytes, row_id_start, partition_id, mapping_id,
                   begin_snapshot
            FROM ducklake_data_file
            WHERE data_file_id = ?
            """),
            [data_file_id],
        ).fetchone()
        if row is None:
            return None
        return FileInfo(
            data_file_id=row[0],
            path=row[1],
            path_is_relative=bool(row[2]) if row[2] is not None else True,
            record_count=row[3],
            file_size_bytes=row[4],
            row_id_start=row[5],
            partition_id=row[6],
            mapping_id=row[7],
            begin_snapshot=row[8],
        )

    def get_inlined_data_tables(self, table_id: int) -> list[InlinedDataTableInfo]:
        """Get inlined data table info for a table."""
        con = self._connect()
        try:
            rows = con.execute(
                self._sql("""
                SELECT table_id, table_name, schema_version
                FROM ducklake_inlined_data_tables
                WHERE table_id = ?
                """),
                [table_id],
            ).fetchall()
        except Exception as e:
            if self._backend.is_table_not_found(e):
                return []
            raise
        return [
            InlinedDataTableInfo(
                table_id=r[0],
                table_name=r[1],
                schema_version=r[2],
            )
            for r in rows
        ]

    def get_table_tags(self, table_id: int, snapshot_id: int) -> dict[str, str]:
        """Get all active tags for a table at a specific snapshot.

        Returns a dict mapping tag key to tag value.
        """
        con = self._connect()
        try:
            rows = con.execute(
                self._sql("""
                SELECT key, value
                FROM ducklake_tag
                WHERE object_id = ?
                  AND ? >= begin_snapshot
                  AND (? < end_snapshot OR end_snapshot IS NULL)
                """),
                [table_id, snapshot_id, snapshot_id],
            ).fetchall()
        except Exception as e:
            if self._backend.is_table_not_found(e):
                return {}
            raise
        return {r[0]: r[1] for r in rows}

    def get_column_tags(
        self, table_id: int, column_id: int, snapshot_id: int
    ) -> dict[str, str]:
        """Get all active tags for a column at a specific snapshot.

        Returns a dict mapping tag key to tag value.
        """
        con = self._connect()
        try:
            rows = con.execute(
                self._sql("""
                SELECT key, value
                FROM ducklake_column_tag
                WHERE table_id = ?
                  AND column_id = ?
                  AND ? >= begin_snapshot
                  AND (? < end_snapshot OR end_snapshot IS NULL)
                """),
                [table_id, column_id, snapshot_id, snapshot_id],
            ).fetchall()
        except Exception as e:
            if self._backend.is_table_not_found(e):
                return {}
            raise
        return {r[0]: r[1] for r in rows}

    def read_inlined_data(
        self,
        table_id: int,
        snapshot_id: int,
        column_names: list[str],
    ) -> pa.Table | None:
        """
        Read inlined data from the metadata catalog.

        Returns a PyArrow Table with the inlined data, or None if
        there is no inlined data.
        """
        inlined_tables = self.get_inlined_data_tables(table_id)
        if not inlined_tables:
            return None

        con = self._connect()
        frames: list[pa.Table] = []

        for info in inlined_tables:
            safe_table = info.table_name.replace('"', '""')
            ph = self._backend.placeholder

            # Discover which columns actually exist in this inlined table.
            # Each schema version may have a different set of columns
            # (e.g. after ADD COLUMN).  SQLite silently returns string
            # literals for double-quoted identifiers that don't match a
            # column, so we must probe first.
            try:
                desc_cursor = con.execute(
                    f'SELECT * FROM "{safe_table}" LIMIT 0'
                )
                available_cols = (
                    {d[0] for d in desc_cursor.description}
                    if desc_cursor.description
                    else set()
                )
            except Exception as e:
                if self._backend.is_table_not_found(e):
                    continue
                raise

            existing_cols = [c for c in column_names if c in available_cols]
            missing_cols = [c for c in column_names if c not in available_cols]

            if not existing_cols:
                continue

            cols_sql = ", ".join(
                f'"{c.replace(chr(34), chr(34) + chr(34))}"' for c in existing_cols
            )
            try:
                cursor = con.execute(
                    f"""
                    SELECT {cols_sql}
                    FROM "{safe_table}"
                    WHERE {ph} >= begin_snapshot
                      AND ({ph} < end_snapshot OR end_snapshot IS NULL)
                    """,
                    [snapshot_id, snapshot_id],
                )
                rows = cursor.fetchall()
            except Exception as e:
                if self._backend.is_table_not_found(e):
                    continue
                raise

            if rows:
                data = {
                    name: [row[i] for row in rows]
                    for i, name in enumerate(existing_cols)
                }
                # Fill missing columns with NULLs
                for col in missing_cols:
                    data[col] = [None] * len(rows)
                frames.append(pa.table(data))

        if not frames:
            return None

        return pa.concat_tables(frames, promote_options="permissive")
