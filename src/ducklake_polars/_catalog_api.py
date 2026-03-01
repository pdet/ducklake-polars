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
