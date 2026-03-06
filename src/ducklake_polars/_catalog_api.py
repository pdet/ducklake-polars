"""DuckLake catalog utility functions — Polars wrapper around ducklake_core."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ducklake_core._catalog_api import DuckLakeCatalog as _CoreCatalog

if TYPE_CHECKING:
    from pathlib import Path


class DuckLakeCatalog(_CoreCatalog):
    """
    High-level interface for inspecting DuckLake catalog metadata.

    Provides Python equivalents of DuckLake's DuckDB extension utility
    functions (``ducklake_snapshots``, ``ducklake_table_info``, etc.).
    All methods return ``pl.DataFrame``.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (``.ducklake`` or ``.db``),
        or a PostgreSQL connection string.
    data_path
        Override the data path stored in the catalog.

    Examples
    --------
    >>> catalog = DuckLakeCatalog("catalog.ducklake")
    >>> catalog.snapshots()
    >>> catalog.current_snapshot()
    >>> catalog.table_info()
    >>> catalog.list_files("my_table")
    """

    def __enter__(self) -> DuckLakeCatalog:  # type: ignore[override]
        return self

    # ------------------------------------------------------------------
    # Snapshot functions
    # ------------------------------------------------------------------

    def snapshots(self) -> pl.DataFrame:
        """
        List all snapshots in the catalog.

        Equivalent to ``ducklake_snapshots(catalog)``.

        Returns a DataFrame with columns:
        - ``snapshot_id`` (Int64)
        - ``snapshot_time`` (String)
        - ``schema_version`` (Int64)
        """
        return pl.from_arrow(super().snapshots())

    # ------------------------------------------------------------------
    # Table metadata
    # ------------------------------------------------------------------

    def table_info(self, *, schema: str = "main") -> pl.DataFrame:
        """
        Get per-table storage metadata.

        Equivalent to ``ducklake_table_info(catalog)``.

        Returns a DataFrame with columns:
        - ``table_name`` (String)
        - ``table_id`` (Int64)
        - ``file_count`` (Int64)
        - ``file_size_bytes`` (Int64)
        - ``delete_file_count`` (Int64)
        - ``delete_row_count`` (Int64)
        """
        return pl.from_arrow(super().table_info(schema=schema))

    def list_files(
        self,
        table: str,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pl.DataFrame:
        """
        List data files and delete files for a table.

        Equivalent to ``ducklake_list_files(catalog, table)``.

        Returns a DataFrame with columns:
        - ``data_file`` (String) — resolved path to the data file
        - ``data_file_size_bytes`` (Int64)
        - ``delete_file`` (String or null) — resolved path to delete file
        - ``delete_row_count`` (Int64 or null)
        """
        return pl.from_arrow(super().list_files(table, schema=schema, snapshot_version=snapshot_version))

    # ------------------------------------------------------------------
    # Schema / table listing
    # ------------------------------------------------------------------

    def list_schemas(self, *, snapshot_version: int | None = None) -> pl.DataFrame:
        """
        List all schemas in the catalog.

        Returns a DataFrame with columns:
        - ``schema_id`` (Int64)
        - ``schema_name`` (String)
        """
        return pl.from_arrow(super().list_schemas(snapshot_version=snapshot_version))

    def list_tables(
        self,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pl.DataFrame:
        """
        List all tables in a schema.

        Returns a DataFrame with columns:
        - ``table_id`` (Int64)
        - ``table_name`` (String)
        """
        return pl.from_arrow(super().list_tables(schema=schema, snapshot_version=snapshot_version))

    # ------------------------------------------------------------------
    # Catalog options
    # ------------------------------------------------------------------

    def options(self) -> pl.DataFrame:
        """
        Get catalog key-value options from ducklake_metadata.

        Equivalent to ``ducklake_options(catalog)``.

        Returns a DataFrame with columns:
        - ``key`` (String)
        - ``value`` (String)
        """
        return pl.from_arrow(super().options())

    def settings(self) -> pl.DataFrame:
        """
        Get catalog type and data path.

        Equivalent to ``ducklake_settings(catalog)``.

        Returns a DataFrame with columns:
        - ``catalog_type`` (String) — ``"sqlite"`` or ``"postgresql"``
        - ``data_path`` (String)
        """
        return pl.from_arrow(super().settings())

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def table_tags(self, table: str, *, schema: str = "main") -> pl.DataFrame:
        """
        Get tags for a table.

        Returns a DataFrame with columns:
        - ``key`` (String)
        - ``value`` (String)
        """
        return pl.from_arrow(super().table_tags(table, schema=schema))

    def column_tags(
        self, table: str, column: str, *, schema: str = "main"
    ) -> pl.DataFrame:
        """
        Get tags for a column.

        Returns a DataFrame with columns:
        - ``key`` (String)
        - ``value`` (String)
        """
        return pl.from_arrow(super().column_tags(table, column, schema=schema))

    # ------------------------------------------------------------------
    # Macros
    # ------------------------------------------------------------------

    def list_macros(
        self,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pl.DataFrame:
        """
        List all macros in a schema.

        Returns a DataFrame with columns:
        - ``macro_id`` (Int64)
        - ``macro_name`` (String)
        - ``macro_type`` (String) — ``"scalar"`` or ``"table"``
        """
        return pl.from_arrow(super().list_macros(schema=schema, snapshot_version=snapshot_version))

    def get_macro(
        self,
        name: str,
        *,
        schema: str = "main",
        dialect: str | None = None,
        snapshot_version: int | None = None,
    ) -> pl.DataFrame:
        """
        Get macro definition(s) by name.

        Parameters
        ----------
        name
            Macro name.
        schema
            Schema name (default: "main").
        dialect
            If provided, return only the implementation for this dialect.
        snapshot_version
            Snapshot version to query.

        Returns a DataFrame with columns:
        - ``macro_name`` (String)
        - ``macro_type`` (String)
        - ``dialect`` (String)
        - ``sql`` (String) — the SQL definition
        - ``parameters`` (String) — comma-separated parameter list
        """
        return pl.from_arrow(super().get_macro(name, schema=schema, dialect=dialect, snapshot_version=snapshot_version))

    # ------------------------------------------------------------------
    # Change data feed
    # ------------------------------------------------------------------

    def table_insertions(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pl.DataFrame:
        """
        Get rows inserted between two snapshots.

        Equivalent to ``ducklake_table_insertions(catalog, schema, table, start, end)``.

        Reads the Parquet data files that were added in the snapshot range
        ``(start_version, end_version]``.

        Returns a DataFrame with ``snapshot_id`` column plus all table columns.
        """
        return pl.from_arrow(super().table_insertions(table, start_version, end_version, schema=schema))

    def table_deletions(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pl.DataFrame:
        """
        Get rows deleted between two snapshots.

        Equivalent to ``ducklake_table_deletions(catalog, schema, table, start, end)``.

        Reads delete files added in the snapshot range ``(start_version, end_version]``,
        then reads the corresponding rows from their associated data files.

        Returns a DataFrame with ``snapshot_id`` column plus all table columns.
        """
        return pl.from_arrow(super().table_deletions(table, start_version, end_version, schema=schema))

    def table_changes(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pl.DataFrame:
        """
        Get combined change data feed between two snapshots.

        Equivalent to ``ducklake_table_changes(catalog, schema, table, start, end)``.

        Returns a DataFrame with columns:
        - ``snapshot_id`` (Int64)
        - ``change_type`` (String) — one of ``'insert'``, ``'delete'``,
          ``'update_preimage'``, ``'update_postimage'``
        - All table columns

        Updates are detected when both an insertion and a deletion occur in the
        same snapshot. Deletions in that snapshot become ``update_preimage`` and
        insertions become ``update_postimage``.
        """
        return pl.from_arrow(super().table_changes(table, start_version, end_version, schema=schema))

    # ------------------------------------------------------------------
    # Sort keys
    # ------------------------------------------------------------------

    def sort_keys(
        self, table: str, *, schema: str = "main"
    ) -> pl.DataFrame:
        """
        Get sort keys for a table.

        Returns a DataFrame with columns:
        - ``column_name`` (String)
        - ``sort_order`` (String) — ``'ASC'`` or ``'DESC'``
        """
        return pl.from_arrow(super().sort_keys(table, schema=schema))

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def list_views(
        self, *, schema: str = "main", snapshot_version: int | None = None
    ) -> pl.DataFrame:
        """List views in a schema."""
        return pl.from_arrow(
            super().list_views(schema=schema, snapshot_version=snapshot_version)
        )

    def get_view(
        self,
        name: str,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pl.DataFrame:
        """Get view definition."""
        return pl.from_arrow(
            super().get_view(name, schema=schema, snapshot_version=snapshot_version)
        )

    # ==================================================================
    # Write operations — stateful wrappers around free functions
    # ==================================================================

    def scan(
        self,
        table: str,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
        snapshot_time: str | None = None,
    ) -> pl.LazyFrame:
        """Lazy scan a table. Returns a Polars LazyFrame with pushdown."""
        from ducklake_polars import scan_ducklake

        return scan_ducklake(
            self._metadata_path,
            table,
            schema=schema,
            snapshot_version=snapshot_version,
            snapshot_time=snapshot_time,
            data_path=self._data_path_override,
        )

    def read(
        self,
        table: str,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
        snapshot_time: str | None = None,
    ) -> pl.DataFrame:
        """Read a table eagerly. Returns a Polars DataFrame."""
        from ducklake_polars import read_ducklake

        return read_ducklake(
            self._metadata_path,
            table,
            schema=schema,
            snapshot_version=snapshot_version,
            snapshot_time=snapshot_time,
            data_path=self._data_path_override,
        )

    def write(
        self,
        table: str,
        df: pl.DataFrame,
        *,
        schema: str = "main",
        mode: str = "error",
        data_inlining_row_limit: int = 0,
        author: str | None = None,
        commit_message: str | None = None,
    ) -> None:
        """Write a DataFrame to a table."""
        from ducklake_polars import write_ducklake

        write_ducklake(
            df,
            self._metadata_path,
            table,
            schema=schema,
            mode=mode,
            data_path=self._data_path_override,
            data_inlining_row_limit=data_inlining_row_limit,
            author=author,
            commit_message=commit_message,
        )

    def delete(
        self,
        table: str,
        predicate: pl.Expr,
        *,
        schema: str = "main",
    ) -> int:
        """Delete rows matching predicate. Returns deleted row count."""
        from ducklake_polars import delete_ducklake

        return delete_ducklake(
            self._metadata_path,
            table,
            predicate,
            schema=schema,
            data_path=self._data_path_override,
        )

    def update(
        self,
        table: str,
        updates: dict[str, object],
        predicate: pl.Expr,
        *,
        schema: str = "main",
    ) -> int:
        """Update rows matching predicate. Returns updated row count."""
        from ducklake_polars import update_ducklake

        return update_ducklake(
            self._metadata_path,
            table,
            updates,
            predicate,
            schema=schema,
            data_path=self._data_path_override,
        )

    def merge(
        self,
        table: str,
        source_df: pl.DataFrame,
        on: str | list[str],
        *,
        schema: str = "main",
        when_matched_update: dict[str, object] | bool | None = True,
        when_not_matched_insert: bool = True,
    ) -> dict:
        """Merge (upsert) source into target table."""
        from ducklake_polars import merge_ducklake

        return merge_ducklake(
            self._metadata_path,
            table,
            source_df,
            on,
            schema=schema,
            when_matched_update=when_matched_update,
            when_not_matched_insert=when_not_matched_insert,
            data_path=self._data_path_override,
        )

    def create_table(
        self,
        table: str,
        schema_def: dict[str, pl.DataType],
        *,
        schema: str = "main",
    ) -> None:
        """Create an empty table with the given schema."""
        from ducklake_polars import create_ducklake_table

        create_ducklake_table(
            self._metadata_path,
            table,
            schema_def,
            schema=schema,
            data_path=self._data_path_override,
        )

    def drop_table(
        self,
        table: str,
        *,
        schema: str = "main",
    ) -> None:
        """Drop a table."""
        from ducklake_polars import drop_ducklake_table

        drop_ducklake_table(
            self._metadata_path,
            table,
            schema=schema,
            data_path=self._data_path_override,
        )

    def add_column(
        self,
        table: str,
        column_name: str,
        column_type: pl.DataType,
        *,
        schema: str = "main",
    ) -> None:
        """Add a column to a table."""
        from ducklake_polars import alter_ducklake_add_column

        alter_ducklake_add_column(
            self._metadata_path,
            table,
            column_name,
            column_type,
            schema=schema,
            data_path=self._data_path_override,
        )

    def drop_column(
        self,
        table: str,
        column_name: str,
        *,
        schema: str = "main",
    ) -> None:
        """Drop a column from a table."""
        from ducklake_polars import alter_ducklake_drop_column

        alter_ducklake_drop_column(
            self._metadata_path,
            table,
            column_name,
            schema=schema,
            data_path=self._data_path_override,
        )

    def rename_column(
        self,
        table: str,
        old_name: str,
        new_name: str,
        *,
        schema: str = "main",
    ) -> None:
        """Rename a column."""
        from ducklake_polars import alter_ducklake_rename_column

        alter_ducklake_rename_column(
            self._metadata_path,
            table,
            old_name,
            new_name,
            schema=schema,
            data_path=self._data_path_override,
        )

    def set_partitioned_by(
        self,
        table: str,
        columns: list[str],
        *,
        schema: str = "main",
    ) -> None:
        """Set partition columns."""
        from ducklake_polars import alter_ducklake_set_partitioned_by

        alter_ducklake_set_partitioned_by(
            self._metadata_path,
            table,
            columns,
            schema=schema,
            data_path=self._data_path_override,
        )

    def set_sort_keys(
        self,
        table: str,
        keys: list[tuple[str, str]],
        *,
        schema_name: str = "main",
    ) -> None:
        """Set sort keys. keys = [(col, 'ASC'|'DESC'), ...]."""
        from ducklake_polars import alter_ducklake_set_sort_keys

        alter_ducklake_set_sort_keys(
            self._metadata_path,
            table,
            keys,
            schema=schema_name,
            data_path=self._data_path_override,
        )

    def rewrite_data_files(
        self,
        table: str,
        *,
        schema: str = "main",
    ) -> int:
        """Compact data files. Returns new snapshot ID or -1 if no-op."""
        from ducklake_polars import rewrite_data_files_ducklake

        return rewrite_data_files_ducklake(
            self._metadata_path,
            table,
            schema=schema,
            data_path=self._data_path_override,
        )

    def expire_snapshots(
        self,
        *,
        older_than: str | None = None,
        retain_last: int | None = None,
    ) -> int:
        """Expire old snapshots. Returns number expired."""
        from ducklake_polars import expire_snapshots

        return expire_snapshots(
            self._metadata_path,
            older_than=older_than,
            retain_last=retain_last,
            data_path=self._data_path_override,
        )

    def vacuum(self) -> int:
        """Remove orphaned data files. Returns count removed."""
        from ducklake_polars import vacuum_ducklake

        return vacuum_ducklake(
            self._metadata_path,
            data_path=self._data_path_override,
        )
