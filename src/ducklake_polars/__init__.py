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
    "merge_ducklake",
    "create_table_as_ducklake",
    "alter_ducklake_add_column",
    "alter_ducklake_drop_column",
    "alter_ducklake_rename_column",
    "alter_ducklake_set_type",
    "alter_ducklake_set_partitioned_by",
    "alter_ducklake_set_sort_keys",
    "drop_ducklake_table",
    "create_ducklake_schema",
    "drop_ducklake_schema",
    "rename_ducklake_table",
    "expire_snapshots",
    "vacuum_ducklake",
    "create_ducklake_view",
    "drop_ducklake_view",
    "set_ducklake_table_tag",
    "set_ducklake_column_tag",
    "delete_ducklake_table_tag",
    "delete_ducklake_column_tag",
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
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Write a Polars DataFrame to a DuckLake table.

    Parameters
    ----------
    df
        DataFrame to write.
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
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
    data_inlining_row_limit
        Maximum number of rows to store inline in the metadata catalog
        instead of writing Parquet files. Set to 0 (default) to disable
        inlining. When enabled, small inserts below this threshold are
        stored directly in the catalog database.

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

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
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
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Create a new table in a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
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

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.create_table(table, polars_schema, schema_name=schema)


def delete_ducklake(
    path: str | Path,
    table: str,
    predicate: pl.Expr,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """
    Delete rows matching a predicate from a DuckLake table.

    Creates Iceberg-compatible position-delete files for each affected
    Parquet data file. For inlined data, sets ``end_snapshot`` on matching
    rows. If no rows match the predicate, no snapshot is created.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to delete from.
    predicate
        A Polars expression that evaluates to a boolean mask. Rows where
        the expression is ``True`` will be deleted.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum number of inlined rows. Only affects how the writer
        instance is configured; deletes on inlined data are handled
        regardless of this setting.

    Returns
    -------
    int
        The number of rows deleted.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.delete_data(predicate, table, schema_name=schema)


def update_ducklake(
    path: str | Path,
    table: str,
    updates: dict[str, object],
    predicate: pl.Expr,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """
    Update rows matching a predicate in a DuckLake table.

    Atomically deletes the old rows and inserts new rows with updated
    values in a single snapshot. If no rows match, no snapshot is created.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
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
    data_inlining_row_limit
        Maximum number of inlined rows. Only affects how the writer
        instance is configured.

    Returns
    -------
    int
        The number of rows updated.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.update_data(updates, predicate, table, schema_name=schema)


def merge_ducklake(
    path: str | Path,
    table: str,
    source_df: pl.DataFrame,
    on: str | list[str],
    *,
    when_matched_update: dict[str, object] | bool | None = None,
    when_not_matched_insert: bool = True,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> tuple[int, int]:
    """
    Merge a source DataFrame into an existing DuckLake table.

    Matches rows on the *on* key columns, optionally updates matched
    target rows, and optionally inserts unmatched source rows. Implemented
    as delete + insert in a single snapshot.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the target table.
    source_df
        Source DataFrame to merge from.
    on
        Column name(s) to match on. Can be a single string or a list.
    when_matched_update
        - ``None``: matched target rows are left untouched.
        - ``True``: replace matched target rows with source rows.
        - ``dict``: update matched target rows with these values
          (literal or ``pl.Expr``).
    when_not_matched_insert
        If True (default), source rows that have no match in the
        target are inserted.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum number of inlined rows.

    Returns
    -------
    tuple[int, int]
        ``(rows_updated, rows_inserted)``.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.merge_data(
            source_df,
            table,
            on,
            when_matched_update=when_matched_update,
            when_not_matched_insert=when_not_matched_insert,
            schema_name=schema,
        )


def create_table_as_ducklake(
    df: pl.DataFrame,
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Create a new table and insert data in a single snapshot.

    Equivalent to ``CREATE TABLE ... AS SELECT ...`` — the table schema
    is inferred from the DataFrame and the data is written atomically.

    Parameters
    ----------
    df
        DataFrame whose schema defines the new table and whose rows
        are the initial data.
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to create.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum number of rows to store inline in the metadata catalog.

    Raises
    ------
    ValueError
        If the table already exists.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        writer.create_table_with_data(table, df, schema_name=schema)


def alter_ducklake_add_column(
    path: str | Path,
    table: str,
    col_name: str,
    dtype: pl.DataType,
    *,
    default: object = None,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Add a column to a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
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

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.add_column(table, col_name, dtype, default=default, schema_name=schema)


def alter_ducklake_drop_column(
    path: str | Path,
    table: str,
    col_name: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Drop a column from a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
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

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.drop_column(table, col_name, schema_name=schema)


def alter_ducklake_rename_column(
    path: str | Path,
    table: str,
    old_col_name: str,
    new_col_name: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Rename a column in a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
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

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.rename_column(
            table, old_col_name, new_col_name, schema_name=schema
        )


def alter_ducklake_set_type(
    path: str | Path,
    table: str,
    column_name: str,
    new_type: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Change the type of a column in a DuckLake table.

    Existing Parquet files keep their original types; the reader casts
    values from the old type when reading files written before the change.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to alter.
    column_name
        Name of the column whose type to change.
    new_type
        DuckDB type string for the new column type (e.g., ``"BIGINT"``,
        ``"VARCHAR"``, ``"DOUBLE"``).
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

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.set_column_type(
            table, column_name, new_type, schema_name=schema
        )


def drop_ducklake_table(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Drop a table from a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
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

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.drop_table(table, schema_name=schema)


def create_ducklake_schema(
    path: str | Path,
    schema_name: str,
    *,
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Create a new schema in a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
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

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.create_schema(schema_name)


def drop_ducklake_schema(
    path: str | Path,
    schema_name: str,
    *,
    cascade: bool = False,
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Drop a schema from a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
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

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.drop_schema(schema_name, cascade=cascade)


def rename_ducklake_table(
    path: str | Path,
    old_table: str,
    new_table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Rename a table in a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
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

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.rename_table(old_table, new_table, schema_name=schema)


def alter_ducklake_set_partitioned_by(
    path: str | Path,
    table: str,
    columns: list[str],
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Set identity-transform partitioning on a DuckLake table.

    Equivalent to ``ALTER TABLE t SET PARTITIONED BY (col1, col2, ...)``.
    Future inserts will write one Parquet file per unique combination of
    partition column values, using Hive-style directory layout.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to partition.
    columns
        Column names to partition by (identity transform).
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table or any column does not exist.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.set_partitioned_by(table, columns, schema_name=schema)


def alter_ducklake_set_sort_keys(
    path: str | Path,
    table: str,
    columns: list[str],
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Set sort keys on a DuckLake table.

    Equivalent to ``ALTER TABLE t SET SORTED BY (col1, col2, ...)``.
    Future writes will sort data by these columns before writing Parquet
    files, improving filter pushdown via Parquet row group statistics.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table.
    columns
        Column names to sort by (ascending order).
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table or any column does not exist.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.set_sort_keys(table, columns, schema_name=schema)


def create_ducklake_view(
    path: str | Path,
    view_name: str,
    sql: str,
    *,
    schema: str = "main",
    or_replace: bool = False,
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Create a view in a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db),
        or a PostgreSQL connection string.
    view_name
        Name of the view to create.
    sql
        The SQL definition of the view.
    schema
        Schema name (default: "main").
    or_replace
        If True, replace an existing view with the same name.
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the view already exists and or_replace is False.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.create_view(
            view_name, sql, schema_name=schema, or_replace=or_replace,
        )


def drop_ducklake_view(
    path: str | Path,
    view_name: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Drop a view from a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db),
        or a PostgreSQL connection string.
    view_name
        Name of the view to drop.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the view does not exist.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.drop_view(view_name, schema_name=schema)


def expire_snapshots(
    path: str | Path,
    *,
    older_than_snapshot: int | None = None,
    keep_last_n: int | None = None,
    data_path: str | Path | None = None,
) -> int:
    """
    Expire old snapshots and clean up associated metadata.

    Removes snapshot rows, snapshot_changes entries, and metadata
    entries (data files, delete files, column stats, partition values)
    whose ``end_snapshot`` falls within the expired range. This is a
    metadata-only operation — call :func:`vacuum_ducklake` afterwards
    to delete the actual orphaned Parquet files.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    older_than_snapshot
        Expire all snapshots with ``snapshot_id < older_than_snapshot``.
    keep_last_n
        Keep the most recent *n* snapshots, expire the rest.
        Cannot be combined with *older_than_snapshot*.
    data_path
        Override the data path stored in the catalog.

    Returns
    -------
    int
        The number of snapshots expired.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        return writer.expire_snapshots(
            older_than_snapshot=older_than_snapshot,
            keep_last_n=keep_last_n,
        )


def vacuum_ducklake(
    path: str | Path,
    *,
    data_path: str | Path | None = None,
) -> int:
    """
    Delete orphaned Parquet files not referenced by any catalog entry.

    Scans the data directory for all ``.parquet`` files and removes
    those that are not referenced by any ``ducklake_data_file`` or
    ``ducklake_delete_file`` entry in the catalog. Run
    :func:`expire_snapshots` first to clean up metadata for old
    snapshots, then call this to reclaim disk space.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    data_path
        Override the data path stored in the catalog.

    Returns
    -------
    int
        The number of Parquet files deleted.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        return writer.vacuum()


def set_ducklake_table_tag(
    path: str | Path,
    table: str,
    key: str,
    value: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Set a tag on a DuckLake table.

    Tags are key-value metadata pairs. Setting a tag with an existing key
    overwrites the previous value. The ``comment`` key is interoperable
    with DuckDB's ``COMMENT ON TABLE`` statement.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
    table
        Name of the table.
    key
        Tag key (e.g., ``"comment"``).
    value
        Tag value.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.set_table_tag(table, key, value, schema_name=schema)


def set_ducklake_column_tag(
    path: str | Path,
    table: str,
    column: str,
    key: str,
    value: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Set a tag on a column in a DuckLake table.

    Tags are key-value metadata pairs. Setting a tag with an existing key
    overwrites the previous value. The ``comment`` key is interoperable
    with DuckDB's ``COMMENT ON COLUMN`` statement.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
    table
        Name of the table.
    column
        Name of the column.
    key
        Tag key (e.g., ``"comment"``).
    value
        Tag value.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.set_column_tag(table, column, key, value, schema_name=schema)


def delete_ducklake_table_tag(
    path: str | Path,
    table: str,
    key: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Remove a tag from a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
    table
        Name of the table.
    key
        Tag key to remove.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.delete_table_tag(table, key, schema_name=schema)


def delete_ducklake_column_tag(
    path: str | Path,
    table: str,
    column: str,
    key: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Remove a tag from a column in a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
    table
        Name of the table.
    column
        Name of the column.
    key
        Tag key to remove.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    """
    from ducklake_polars._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.delete_column_tag(table, column, key, schema_name=schema)
