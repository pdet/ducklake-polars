"""DuckLake catalog utility functions — Pandas wrapper around ducklake_core."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ducklake_core._catalog_api import DuckLakeCatalog as _CoreCatalog

if TYPE_CHECKING:
    from pathlib import Path


class DuckLakeCatalog(_CoreCatalog):
    """
    High-level interface for inspecting DuckLake catalog metadata.

    Provides Python equivalents of DuckLake's DuckDB extension utility
    functions (``ducklake_snapshots``, ``ducklake_table_info``, etc.).
    All methods return ``pd.DataFrame``.

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

    def snapshots(self) -> pd.DataFrame:
        """
        List all snapshots in the catalog.

        Equivalent to ``ducklake_snapshots(catalog)``.

        Returns a DataFrame with columns:
        - ``snapshot_id`` (int64)
        - ``snapshot_time`` (object/string)
        - ``schema_version`` (int64)
        """
        return super().snapshots().to_pandas()

    # ------------------------------------------------------------------
    # Table metadata
    # ------------------------------------------------------------------

    def table_info(self, *, schema: str = "main") -> pd.DataFrame:
        """
        Get per-table storage metadata.

        Equivalent to ``ducklake_table_info(catalog)``.

        Returns a DataFrame with columns:
        - ``table_name`` (object/string)
        - ``table_id`` (int64)
        - ``file_count`` (int64)
        - ``file_size_bytes`` (int64)
        - ``delete_file_count`` (int64)
        - ``delete_row_count`` (int64)
        """
        return super().table_info(schema=schema).to_pandas()

    def list_files(
        self,
        table: str,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pd.DataFrame:
        """
        List data files and delete files for a table.

        Equivalent to ``ducklake_list_files(catalog, table)``.

        Returns a DataFrame with columns:
        - ``data_file`` (object/string)
        - ``data_file_size_bytes`` (int64)
        - ``delete_file`` (object/string or None)
        - ``delete_row_count`` (int64 or None)
        """
        return super().list_files(table, schema=schema, snapshot_version=snapshot_version).to_pandas()

    # ------------------------------------------------------------------
    # Schema / table listing
    # ------------------------------------------------------------------

    def list_schemas(self, *, snapshot_version: int | None = None) -> pd.DataFrame:
        """
        List all schemas in the catalog.

        Returns a DataFrame with columns:
        - ``schema_id`` (int64)
        - ``schema_name`` (object/string)
        """
        return super().list_schemas(snapshot_version=snapshot_version).to_pandas()

    def list_tables(
        self,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pd.DataFrame:
        """
        List all tables in a schema.

        Returns a DataFrame with columns:
        - ``table_id`` (int64)
        - ``table_name`` (object/string)
        """
        return super().list_tables(schema=schema, snapshot_version=snapshot_version).to_pandas()

    # ------------------------------------------------------------------
    # Catalog options
    # ------------------------------------------------------------------

    def options(self) -> pd.DataFrame:
        """
        Get catalog key-value options from ducklake_metadata.

        Equivalent to ``ducklake_options(catalog)``.

        Returns a DataFrame with columns:
        - ``key`` (object/string)
        - ``value`` (object/string)
        """
        return super().options().to_pandas()

    def settings(self) -> pd.DataFrame:
        """
        Get catalog type and data path.

        Equivalent to ``ducklake_settings(catalog)``.

        Returns a DataFrame with columns:
        - ``catalog_type`` (object/string) — ``"sqlite"`` or ``"postgresql"``
        - ``data_path`` (object/string)
        """
        return super().settings().to_pandas()

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def table_tags(self, table: str, *, schema: str = "main") -> pd.DataFrame:
        """
        Get tags for a table.

        Returns a DataFrame with columns:
        - ``key`` (object/string)
        - ``value`` (object/string)
        """
        return super().table_tags(table, schema=schema).to_pandas()

    def column_tags(
        self, table: str, column: str, *, schema: str = "main"
    ) -> pd.DataFrame:
        """
        Get tags for a column.

        Returns a DataFrame with columns:
        - ``key`` (object/string)
        - ``value`` (object/string)
        """
        return super().column_tags(table, column, schema=schema).to_pandas()

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
    ) -> pd.DataFrame:
        """
        Get rows inserted between two snapshots.

        Equivalent to ``ducklake_table_insertions(catalog, schema, table, start, end)``.

        Returns a DataFrame with ``snapshot_id`` column plus all table columns.
        """
        return super().table_insertions(table, start_version, end_version, schema=schema).to_pandas()

    def table_deletions(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pd.DataFrame:
        """
        Get rows deleted between two snapshots.

        Equivalent to ``ducklake_table_deletions(catalog, schema, table, start, end)``.

        Returns a DataFrame with ``snapshot_id`` column plus all table columns.
        """
        return super().table_deletions(table, start_version, end_version, schema=schema).to_pandas()

    def table_changes(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pd.DataFrame:
        """
        Get combined change data feed between two snapshots.

        Equivalent to ``ducklake_table_changes(catalog, schema, table, start, end)``.

        Returns a DataFrame with columns:
        - ``snapshot_id`` (int64)
        - ``change_type`` (object/string)
        - All table columns
        """
        return super().table_changes(table, start_version, end_version, schema=schema).to_pandas()
