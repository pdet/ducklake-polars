"""DuckLake-Polars: Polars integration for DuckLake catalogs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    import polars as pl

from ducklake_polars._catalog_api import DuckLakeCatalog

__all__ = [
    "scan_ducklake",
    "read_ducklake",
    "write_ducklake",
    "create_ducklake_table",
    "delete_ducklake",
    "update_ducklake",
    "alter_ducklake_add_column",
    "alter_ducklake_drop_column",
    "alter_ducklake_rename_column",
    "drop_ducklake_table",
    "create_ducklake_schema",
    "drop_ducklake_schema",
    "rename_ducklake_table",
    "DuckLakeCatalog",
]


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


def write_ducklake(
    df: pl.DataFrame,
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    mode: str = "error",
    data_path: str | Path | None = None,
) -> None:
    """
    Write a Polars DataFrame to a DuckLake table.

    Parameters
    ----------
    df
        DataFrame to write.
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Currently only SQLite backends are supported for writes.
    table
        Name of the table to write to.
    schema
        Schema name (default: "main").
    mode
        Write mode:
        - ``"error"`` (default): Fail if the table already exists.
        - ``"append"``: Append data to an existing table. Creates the
          table if it does not exist.
        - ``"overwrite"``: Replace all data in the table. Creates the
          table if it does not exist.
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If mode is ``"error"`` and the table already exists, or if the
        mode is not recognized.
    """
    if mode not in ("error", "append", "overwrite"):
        msg = f"Invalid write mode '{mode}'. Must be 'error', 'append', or 'overwrite'."
        raise ValueError(msg)

    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        snap_id, _sv, _nci, _nfi = writer._get_latest_snapshot()
        table_id = writer._table_exists(table, schema, snap_id)

        if mode == "error":
            if table_id is not None:
                msg = f"Table '{schema}.{table}' already exists (mode='error')"
                raise ValueError(msg)
            writer.create_table(table, dict(df.schema), schema_name=schema)
            if not df.is_empty():
                writer.insert_data(df, table, schema_name=schema)

        elif mode == "append":
            if table_id is None:
                writer.create_table(table, dict(df.schema), schema_name=schema)
            if not df.is_empty():
                writer.insert_data(df, table, schema_name=schema)

        elif mode == "overwrite":
            if table_id is None:
                writer.create_table(table, dict(df.schema), schema_name=schema)
                if not df.is_empty():
                    writer.insert_data(df, table, schema_name=schema)
            else:
                writer.overwrite_data(df, table, schema_name=schema)


def create_ducklake_table(
    path: str | Path,
    table: str,
    polars_schema: pl.Schema | dict[str, pl.DataType],
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
) -> None:
    """
    Create a new table in a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Currently only SQLite backends are supported for writes.
    table
        Name of the table to create.
    polars_schema
        Schema for the new table, as a Polars Schema or dict.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table already exists.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        writer.create_table(table, polars_schema, schema_name=schema)


def delete_ducklake(
    path: str | Path,
    table: str,
    predicate: pl.Expr,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
) -> int:
    """
    Delete rows matching a predicate from a DuckLake table.

    Creates Iceberg-compatible position-delete files for each affected
    data file. If no rows match the predicate, no snapshot is created.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Currently only SQLite backends are supported for writes.
    table
        Name of the table to delete from.
    predicate
        A Polars expression that evaluates to a boolean mask. Rows where
        the expression is ``True`` will be deleted.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Returns
    -------
    int
        The number of rows deleted.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        return writer.delete_data(predicate, table, schema_name=schema)


def update_ducklake(
    path: str | Path,
    table: str,
    updates: dict[str, object],
    predicate: pl.Expr,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
) -> int:
    """
    Update rows matching a predicate in a DuckLake table.

    Atomically deletes the old rows and inserts new rows with updated
    values in a single snapshot. If no rows match, no snapshot is created.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Currently only SQLite backends are supported for writes.
    table
        Name of the table to update.
    updates
        Dictionary mapping column names to new values. Values can be
        literals (int, str, float, ...) or ``pl.Expr`` for computed updates.
    predicate
        A Polars expression that evaluates to a boolean mask. Rows where
        the expression is ``True`` will be updated.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Returns
    -------
    int
        The number of rows updated.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        return writer.update_data(updates, predicate, table, schema_name=schema)


def alter_ducklake_add_column(
    path: str | Path,
    table: str,
    col_name: str,
    dtype: pl.DataType,
    *,
    default: object = None,
    schema: str = "main",
    data_path: str | Path | None = None,
) -> None:
    """
    Add a column to a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Currently only SQLite backends are supported for writes.
    table
        Name of the table to alter.
    col_name
        Name of the new column.
    dtype
        Polars DataType for the new column.
    default
        Default value for the new column. If None, the column has no default
        and existing rows will have NULL for this column.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table does not exist or the column already exists.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        writer.add_column(table, col_name, dtype, default=default, schema_name=schema)


def alter_ducklake_drop_column(
    path: str | Path,
    table: str,
    col_name: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
) -> None:
    """
    Drop a column from a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Currently only SQLite backends are supported for writes.
    table
        Name of the table to alter.
    col_name
        Name of the column to drop.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table or column does not exist.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        writer.drop_column(table, col_name, schema_name=schema)


def alter_ducklake_rename_column(
    path: str | Path,
    table: str,
    old_col_name: str,
    new_col_name: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
) -> None:
    """
    Rename a column in a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Currently only SQLite backends are supported for writes.
    table
        Name of the table to alter.
    old_col_name
        Current name of the column.
    new_col_name
        New name for the column.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table or column does not exist, or if the new name
        already exists.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        writer.rename_column(
            table, old_col_name, new_col_name, schema_name=schema
        )


def drop_ducklake_table(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
) -> None:
    """
    Drop a table from a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Currently only SQLite backends are supported for writes.
    table
        Name of the table to drop.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table does not exist.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        writer.drop_table(table, schema_name=schema)


def create_ducklake_schema(
    path: str | Path,
    schema_name: str,
    *,
    data_path: str | Path | None = None,
) -> None:
    """
    Create a new schema in a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Currently only SQLite backends are supported for writes.
    schema_name
        Name of the schema to create.
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the schema already exists.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        writer.create_schema(schema_name)


def drop_ducklake_schema(
    path: str | Path,
    schema_name: str,
    *,
    cascade: bool = False,
    data_path: str | Path | None = None,
) -> None:
    """
    Drop a schema from a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Currently only SQLite backends are supported for writes.
    schema_name
        Name of the schema to drop.
    cascade
        If True, drop all tables in the schema first.
        If False (default), raise if the schema contains tables.
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the schema does not exist or contains tables (when cascade=False).
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        writer.drop_schema(schema_name, cascade=cascade)


def rename_ducklake_table(
    path: str | Path,
    old_table: str,
    new_table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
) -> None:
    """
    Rename a table in a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Currently only SQLite backends are supported for writes.
    old_table
        Current name of the table.
    new_table
        New name for the table.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the old table does not exist or if the new name already exists.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        writer.rename_table(old_table, new_table, schema_name=schema)
