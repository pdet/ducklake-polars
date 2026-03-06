"""DuckLake-PySpark: DDL and catalog metadata operations.

These functions are DataFrame-independent — they operate on the DuckLake
catalog directly via ``ducklake_core``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from pathlib import Path

from ducklake_core._catalog import DuckLakeCatalogReader
from ducklake_core._writer import DuckLakeCatalogWriter

# ------------------------------------------------------------------
# PySpark DataType → Arrow DataType conversion
# ------------------------------------------------------------------


def _spark_type_to_arrow(spark_type: Any) -> pa.DataType:
    """Convert a PySpark DataType to a PyArrow DataType."""
    from pyspark.sql.types import (
        ArrayType,
        BinaryType,
        BooleanType,
        ByteType,
        DateType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        MapType,
        ShortType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    if isinstance(spark_type, ByteType):
        return pa.int8()
    if isinstance(spark_type, ShortType):
        return pa.int16()
    if isinstance(spark_type, IntegerType):
        return pa.int32()
    if isinstance(spark_type, LongType):
        return pa.int64()
    if isinstance(spark_type, FloatType):
        return pa.float32()
    if isinstance(spark_type, DoubleType):
        return pa.float64()
    if isinstance(spark_type, BooleanType):
        return pa.bool_()
    if isinstance(spark_type, StringType):
        return pa.string()
    if isinstance(spark_type, BinaryType):
        return pa.binary()
    if isinstance(spark_type, DateType):
        return pa.date32()
    if isinstance(spark_type, TimestampType):
        return pa.timestamp("us")
    if isinstance(spark_type, DecimalType):
        return pa.decimal128(spark_type.precision, spark_type.scale)
    if isinstance(spark_type, ArrayType):
        return pa.list_(_spark_type_to_arrow(spark_type.elementType))
    if isinstance(spark_type, MapType):
        return pa.map_(
            _spark_type_to_arrow(spark_type.keyType),
            _spark_type_to_arrow(spark_type.valueType),
        )
    if isinstance(spark_type, StructType):
        fields = [
            pa.field(f.name, _spark_type_to_arrow(f.dataType), nullable=f.nullable)
            for f in spark_type.fields
        ]
        return pa.struct(fields)
    # Fallback
    return pa.string()


def _resolve_column_type(col_type: Any) -> pa.DataType:
    """Accept a PySpark DataType or a DuckDB type string -> Arrow DataType."""
    if isinstance(col_type, str):
        from ducklake_pyspark import _duckdb_type_to_spark
        spark_t = _duckdb_type_to_spark(col_type)
        return _spark_type_to_arrow(spark_t)
    return _spark_type_to_arrow(col_type)


# ------------------------------------------------------------------
# Helper: open a core writer context
# ------------------------------------------------------------------

def _open_writer(
    path,
    *,
    data_path=None,
    author=None,
    commit_message=None,
    max_retries=3,
    retry_wait_ms=100,
    retry_backoff=2.0,
    data_inlining_row_limit=0,
):
    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None
    return DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
        max_retries=max_retries,
        retry_wait_ms=retry_wait_ms,
        retry_backoff=retry_backoff,
    )


# ==================================================================
# ALTER TABLE operations
# ==================================================================


def alter_ducklake_add_column(
    path,
    table,
    col_name,
    col_type,
    *,
    default=None,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Add a column to a DuckLake table.

    Parameters
    ----------
    col_type
        Column type — a PySpark ``DataType`` (e.g. ``IntegerType()``) or
        a DuckDB type string (e.g. ``"BIGINT"``).
    """
    arrow_type = _resolve_column_type(col_type)
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.add_column(table, col_name, arrow_type, default=default, schema_name=schema)


def alter_ducklake_drop_column(
    path,
    table,
    col_name,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Drop a column from a DuckLake table."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.drop_column(table, col_name, schema_name=schema)


def alter_ducklake_rename_column(
    path,
    table,
    old_col_name,
    new_col_name,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Rename a column in a DuckLake table."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.rename_column(table, old_col_name, new_col_name, schema_name=schema)


def alter_ducklake_set_type(
    path,
    table,
    column_name,
    new_type,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Change the type of a column in a DuckLake table.

    Parameters
    ----------
    new_type
        DuckDB type string (e.g. ``"BIGINT"``, ``"VARCHAR"``).
    """
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.set_column_type(table, column_name, new_type, schema_name=schema)


def alter_ducklake_set_partitioned_by(
    path,
    table,
    columns,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Set identity-transform partitioning on a DuckLake table."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.set_partitioned_by(table, columns, schema_name=schema)


def alter_ducklake_set_sort_keys(
    path,
    table,
    sort_keys,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Set sort keys on a DuckLake table."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.set_sort_keys(table, sort_keys, schema_name=schema)


def alter_ducklake_reset_sort_keys(
    path,
    table,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Remove sort keys from a DuckLake table."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.reset_sort_keys(table, schema_name=schema)


# ==================================================================
# TABLE operations
# ==================================================================


def drop_ducklake_table(
    path,
    table,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Drop a table from a DuckLake catalog."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.drop_table(table, schema_name=schema)


def rename_ducklake_table(
    path,
    old_table,
    new_table,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Rename a table in a DuckLake catalog."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.rename_table(old_table, new_table, schema_name=schema)


# ==================================================================
# SCHEMA operations
# ==================================================================


def create_ducklake_schema(
    path,
    schema_name,
    *,
    data_path=None,
    author=None,
    commit_message=None,
):
    """Create a new schema in a DuckLake catalog."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.create_schema(schema_name)


def drop_ducklake_schema(
    path,
    schema_name,
    *,
    cascade=False,
    data_path=None,
    author=None,
    commit_message=None,
):
    """Drop a schema from a DuckLake catalog."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.drop_schema(schema_name, cascade=cascade)


# ==================================================================
# VIEW operations
# ==================================================================


def create_ducklake_view(
    path,
    view_name,
    sql,
    *,
    schema="main",
    or_replace=False,
    data_path=None,
    author=None,
    commit_message=None,
):
    """Create a view in a DuckLake catalog."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.create_view(view_name, sql, schema_name=schema, or_replace=or_replace)


def drop_ducklake_view(
    path,
    view_name,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Drop a view from a DuckLake catalog."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.drop_view(view_name, schema_name=schema)


# ==================================================================
# TAG operations
# ==================================================================


def set_ducklake_table_tag(
    path,
    table,
    key,
    value,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Set a tag on a DuckLake table."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.set_table_tag(table, key, value, schema_name=schema)


def set_ducklake_column_tag(
    path,
    table,
    column,
    key,
    value,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Set a tag on a column in a DuckLake table."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.set_column_tag(table, column, key, value, schema_name=schema)


def delete_ducklake_table_tag(
    path,
    table,
    key,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Remove a tag from a DuckLake table."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.delete_table_tag(table, key, schema_name=schema)


def delete_ducklake_column_tag(
    path,
    table,
    column,
    key,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
):
    """Remove a tag from a column in a DuckLake table."""
    with _open_writer(path, data_path=data_path, author=author, commit_message=commit_message) as w:
        w.delete_column_tag(table, column, key, schema_name=schema)


# ==================================================================
# MAINTENANCE operations
# ==================================================================


def expire_snapshots(
    path,
    *,
    older_than_snapshot=None,
    keep_last_n=None,
    data_path=None,
):
    """Expire old snapshots. Returns the number expired."""
    with _open_writer(path, data_path=data_path) as w:
        return w.expire_snapshots(
            older_than_snapshot=older_than_snapshot,
            keep_last_n=keep_last_n,
        )


def vacuum_ducklake(
    path,
    *,
    data_path=None,
):
    """Delete orphaned Parquet files. Returns the number deleted."""
    with _open_writer(path, data_path=data_path) as w:
        return w.vacuum()


def rewrite_data_files_ducklake(
    path,
    table,
    *,
    schema="main",
    data_path=None,
    author=None,
    commit_message=None,
    max_retries=3,
    retry_wait_ms=100,
    retry_backoff=2.0,
):
    """Rewrite data files for compaction. Returns the new snapshot ID or -1."""
    with _open_writer(
        path,
        data_path=data_path,
        author=author,
        commit_message=commit_message,
        max_retries=max_retries,
        retry_wait_ms=retry_wait_ms,
        retry_backoff=retry_backoff,
    ) as w:
        return w.rewrite_data_files(table, schema_name=schema)


# ==================================================================
# CATALOG READ operations
# ==================================================================


def list_schemas(
    path,
    *,
    data_path=None,
):
    """List all schema names in a DuckLake catalog."""
    dp = os.fspath(data_path) if data_path is not None else None
    with DuckLakeCatalogReader(os.fspath(path), data_path_override=dp) as reader:
        return reader.list_schemas()


def list_tables(
    path,
    *,
    schema="main",
    data_path=None,
):
    """List all table names in a DuckLake catalog schema."""
    dp = os.fspath(data_path) if data_path is not None else None
    with DuckLakeCatalogReader(os.fspath(path), data_path_override=dp) as reader:
        return reader.list_tables(schema)


def list_views(
    path,
    *,
    schema="main",
    data_path=None,
):
    """List all view names in a DuckLake catalog schema."""
    dp = os.fspath(data_path) if data_path is not None else None
    with DuckLakeCatalogReader(os.fspath(path), data_path_override=dp) as reader:
        return reader.list_views(schema)


def list_snapshots(
    path,
    *,
    limit=20,
    data_path=None,
):
    """List recent snapshots from a DuckLake catalog."""
    dp = os.fspath(data_path) if data_path is not None else None
    with DuckLakeCatalogReader(os.fspath(path), data_path_override=dp) as reader:
        return reader.list_snapshots(limit=limit)


def snapshot_changes(
    path,
    *,
    snapshot_id=None,
    data_path=None,
):
    """Get changes for a specific snapshot or all snapshots."""
    dp = os.fspath(data_path) if data_path is not None else None
    with DuckLakeCatalogReader(os.fspath(path), data_path_override=dp) as reader:
        return reader.snapshot_changes(snapshot_id=snapshot_id)


def catalog_info(
    path,
    *,
    data_path=None,
):
    """Get catalog summary: version, data_path, table_count, snapshot_count."""
    dp = os.fspath(data_path) if data_path is not None else None
    with DuckLakeCatalogReader(os.fspath(path), data_path_override=dp) as reader:
        return reader.catalog_info()


def get_view(
    path,
    name,
    *,
    schema="main",
    data_path=None,
):
    """Get view definition (SQL, dialect, column aliases)."""
    dp = os.fspath(data_path) if data_path is not None else None
    with DuckLakeCatalogReader(os.fspath(path), data_path_override=dp) as reader:
        return reader.get_view(name, schema)


def table_info(
    path,
    table,
    *,
    schema="main",
    data_path=None,
):
    """Get column info for a table: name, type, nullable, order."""
    dp = os.fspath(data_path) if data_path is not None else None
    with DuckLakeCatalogReader(os.fspath(path), data_path_override=dp) as reader:
        return reader.table_info(table, schema)
