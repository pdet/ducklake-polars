"""DuckLake metadata catalog reader — Polars wrapper around ducklake_core."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

# Re-export everything from core
from ducklake_core._catalog import (  # noqa: F401
    SUPPORTED_DUCKLAKE_VERSIONS,
    ColumnHistoryEntry,
    ColumnInfo,
    ColumnStats,
    DeleteFileInfo,
    FileInfo,
    FilePartitionValue,
    InlinedDataTableInfo,
    PartitionColumnDef,
    PartitionInfo,
    SnapshotInfo,
    TableInfo,
)
from ducklake_core._catalog import DuckLakeCatalogReader as _CoreReader

if TYPE_CHECKING:
    from datetime import datetime


class DuckLakeCatalogReader(_CoreReader):
    """Polars-aware catalog reader.

    Wraps the core reader so that ``read_inlined_data`` returns a
    ``pl.DataFrame`` instead of a ``pa.Table``.
    """

    def read_inlined_data(
        self,
        table_id: int,
        snapshot_id: int,
        column_names: list[str],
    ) -> pl.DataFrame | None:
        arrow_table = super().read_inlined_data(table_id, snapshot_id, column_names)
        if arrow_table is None:
            return None
        return pl.from_arrow(arrow_table)
