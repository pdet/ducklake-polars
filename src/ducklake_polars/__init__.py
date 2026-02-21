"""DuckLake-Polars: Polars integration for DuckLake catalogs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    import polars as pl

from ducklake_polars._catalog_api import DuckLakeCatalog

__all__ = ["scan_ducklake", "read_ducklake", "DuckLakeCatalog"]


def scan_ducklake(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    snapshot_version: int | None = None,
    snapshot_time: datetime | str | None = None,
    data_path: str | Path | None = None,
) -> pl.LazyFrame:
    """
    Lazily read a DuckLake table as a Polars LazyFrame.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db),
        or a PostgreSQL connection string (e.g., "postgresql://user:pass@host/dbname").
    table
        Name of the table to read.
    schema
        Schema name (default: "main").
    snapshot_version
        Read the table at a specific snapshot version. If None, reads latest.
    snapshot_time
        Read the table at a specific timestamp. If None, reads latest.
        Can be a datetime object or an ISO format string.
    data_path
        Override the data path stored in the catalog. Useful when the catalog
        has been moved or the data files are in a different location.

    Returns
    -------
    LazyFrame

    Raises
    ------
    ValueError
        If both snapshot_version and snapshot_time are specified, or if the
        table or snapshot is not found.
    """
    if snapshot_version is not None and snapshot_time is not None:
        msg = "Cannot specify both snapshot_version and snapshot_time"
        raise ValueError(msg)

    from polars._plr import PyLazyFrame
    from polars._utils.wrap import wrap_ldf

    from ducklake_polars._dataset import DuckLakeDataset

    dataset = DuckLakeDataset(
        metadata_path=os.fspath(path),
        table_name=table,
        schema_name=schema,
        snapshot_version=snapshot_version,
        snapshot_time=snapshot_time,
        data_path_override=os.fspath(data_path) if data_path is not None else None,
    )

    return wrap_ldf(PyLazyFrame.new_from_dataset_object(dataset))


def read_ducklake(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    columns: list[str] | None = None,
    snapshot_version: int | None = None,
    snapshot_time: datetime | str | None = None,
    data_path: str | Path | None = None,
) -> pl.DataFrame:
    """
    Read a DuckLake table into a Polars DataFrame.

    This is a convenience function that calls ``scan_ducklake(...).collect()``.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db),
        or a PostgreSQL connection string (e.g., "postgresql://user:pass@host/dbname").
    table
        Name of the table to read.
    schema
        Schema name (default: "main").
    columns
        Columns to select. If None, reads all columns.
    snapshot_version
        Read the table at a specific snapshot version.
    snapshot_time
        Read the table at a specific timestamp.
    data_path
        Override the data path stored in the catalog.

    Returns
    -------
    DataFrame

    Raises
    ------
    ValueError
        If both snapshot_version and snapshot_time are specified, or if the
        table or snapshot is not found.
    """
    lf = scan_ducklake(
        path,
        table,
        schema=schema,
        snapshot_version=snapshot_version,
        snapshot_time=snapshot_time,
        data_path=data_path,
    )

    if columns is not None:
        lf = lf.select(columns)

    return lf.collect()
