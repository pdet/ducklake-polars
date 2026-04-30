"""DuckLake-PySpark: PySpark integration for DuckLake catalogs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

try:
    import pyspark  # noqa: F401
except ImportError as _e:
    raise ImportError(
        "PySpark is required for ducklake_pyspark. "
        "Install with: pip install ducklake-dataframe[pyspark]"
    ) from _e

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql.types import StructType

from ducklake_core._catalog import DuckLakeCatalogReader
from ducklake_core._migration import migrate_catalog
from ducklake_core._writer import TransactionConflictError
from ducklake_pyspark._ddl import (
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
    alter_ducklake_set_type,
    alter_ducklake_set_partitioned_by,
    alter_ducklake_set_sort_keys,
    alter_ducklake_reset_sort_keys,
    drop_ducklake_table,
    create_ducklake_schema,
    drop_ducklake_schema,
    rename_ducklake_table,
    create_ducklake_view,
    drop_ducklake_view,
    expire_snapshots,
    vacuum_ducklake,
    rewrite_data_files_ducklake,
    set_ducklake_table_tag,
    set_ducklake_column_tag,
    delete_ducklake_table_tag,
    delete_ducklake_column_tag,
    list_schemas,
    list_tables,
    list_views,
    list_snapshots,
    snapshot_changes,
    catalog_info,
    get_view,
    table_info,
)

__all__ = [
    "migrate_catalog",
    "read_ducklake",
    "read_ducklake_changes",
    "TransactionConflictError",
    "write_ducklake",
    "create_ducklake_table",
    "delete_ducklake",
    "update_ducklake",
    "merge_ducklake",
    "create_table_as_ducklake",
    "add_files_ducklake",
    "alter_ducklake_add_column",
    "alter_ducklake_drop_column",
    "alter_ducklake_rename_column",
    "alter_ducklake_set_type",
    "alter_ducklake_set_partitioned_by",
    "alter_ducklake_set_sort_keys",
    "alter_ducklake_reset_sort_keys",
    "drop_ducklake_table",
    "create_ducklake_schema",
    "drop_ducklake_schema",
    "rename_ducklake_table",
    "create_ducklake_view",
    "drop_ducklake_view",
    "expire_snapshots",
    "vacuum_ducklake",
    "rewrite_data_files_ducklake",
    "set_ducklake_table_tag",
    "set_ducklake_column_tag",
    "delete_ducklake_table_tag",
    "delete_ducklake_column_tag",
    "list_schemas",
    "list_tables",
    "list_views",
    "list_snapshots",
    "snapshot_changes",
    "catalog_info",
    "get_view",
    "table_info",
]


def read_ducklake(
    spark: SparkSession,
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    columns: list[str] | None = None,
    snapshot_version: int | None = None,
    snapshot_time: datetime | str | None = None,
    data_path: str | Path | None = None,
) -> DataFrame:
    """
    Read a DuckLake table into a PySpark DataFrame.

    Uses the existing ``ducklake_core`` catalog reader to resolve metadata,
    then reads the underlying Parquet data files through Spark's native
    Parquet reader. Handles schema evolution (column renames, adds, drops),
    delete files (position deletes), and partition pruning.

    Parameters
    ----------
    spark
        Active SparkSession.
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db),
        or a PostgreSQL connection string.
    table
        Name of the table to read.
    schema
        Schema name (default: ``"main"``).
    columns
        List of column names to read. If ``None``, reads all columns.
    snapshot_version
        Read the table at a specific snapshot version.
    snapshot_time
        Read the table at a specific timestamp (datetime or ISO string).
    data_path
        Override the data path stored in the catalog.

    Returns
    -------
    pyspark.sql.DataFrame

    Examples
    --------
    >>> from ducklake_pyspark import read_ducklake
    >>> df = read_ducklake(spark, "catalog.ducklake", "users")
    >>> df.show()

    >>> # Column selection
    >>> df = read_ducklake(spark, "catalog.ducklake", "events", columns=["id", "value"])

    >>> # Time travel
    >>> df = read_ducklake(spark, "catalog.ducklake", "events", snapshot_version=5)
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        BinaryType,
        BooleanType,
        ByteType,
        DateType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        ShortType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    if snapshot_version is not None and snapshot_time is not None:
        msg = "Cannot specify both snapshot_version and snapshot_time"
        raise ValueError(msg)

    metadata_path = os.fspath(path)
    data_path_override = os.fspath(data_path) if data_path is not None else None

    # Open catalog
    reader = DuckLakeCatalogReader(metadata_path, data_path_override=data_path_override)

    # Resolve snapshot
    if snapshot_version is not None:
        snapshot = reader.get_snapshot_at_version(snapshot_version)
    elif snapshot_time is not None:
        snapshot = reader.get_snapshot_at_time(snapshot_time)
    else:
        snapshot = reader.get_current_snapshot()

    # Get table info
    table_info = reader.get_table(table, schema, snapshot.snapshot_id)
    if table_info is None:
        msg = f"Table '{schema}.{table}' not found at snapshot {snapshot.snapshot_id}"
        raise ValueError(msg)

    # Get columns and data files
    all_columns = reader.get_columns(table_info.table_id, snapshot.snapshot_id)
    data_files = reader.get_data_files(table_info.table_id, snapshot.snapshot_id)

    if not data_files and not all_columns:
        msg = f"Table '{schema}.{table}' has no data"
        raise ValueError(msg)

    # Build Spark schema from DuckLake columns
    spark_schema = _build_spark_schema(all_columns)

    if not data_files:
        # Empty table — return empty DataFrame with correct schema
        return spark.createDataFrame([], spark_schema)

    # Resolve file paths
    resolved_paths = {
        f.data_file_id: reader.resolve_data_file_path(f.path, f.path_is_relative, table_info)
        for f in data_files
    }

    # Get delete files
    delete_files = reader.get_delete_files(table_info.table_id, snapshot.snapshot_id)

    # Group delete files by data_file_id
    deletes_by_file: dict[int, list[str]] = {}
    for df in delete_files:
        delete_path = reader.resolve_data_file_path(df.path, df.path_is_relative, table_info)
        deletes_by_file.setdefault(df.data_file_id, []).append(delete_path)

    # Get name mappings for schema evolution (batch fetch only needed IDs)
    mapping_ids = {f.mapping_id for f in data_files if f.mapping_id is not None}
    name_mappings = reader.get_name_mappings_batch(mapping_ids) if mapping_ids else {}

    # Build column ID → current name mapping
    col_id_to_name = {col.column_id: col.column_name for col in all_columns}

    # Group files by their rename mapping for efficient reading
    file_groups: dict[tuple, list] = {}
    for f in data_files:
        # Determine the rename map for this file
        rename_map: dict[str, str] = {}
        if f.mapping_id is not None and f.mapping_id in name_mappings:
            mapping = name_mappings[f.mapping_id]
            for field_id, source_name in mapping.items():
                current_name = col_id_to_name.get(field_id)
                if current_name and source_name != current_name:
                    rename_map[source_name] = current_name

        rename_key = tuple(sorted(rename_map.items()))
        if rename_key not in file_groups:
            file_groups[rename_key] = (rename_map, [])
        file_groups[rename_key][1].append(f)

    # Target column order
    target_cols = [col.column_name for col in all_columns]

    def _apply_renames_and_fill(df_in, rename_map):
        """Apply column renames, add missing columns, select in order."""
        for old_name, new_name in rename_map.items():
            if old_name in df_in.columns:
                df_in = df_in.withColumnRenamed(old_name, new_name)
        for col_info in all_columns:
            if col_info.column_name not in df_in.columns:
                spark_type = _duckdb_type_to_spark(col_info.column_type)
                df_in = df_in.withColumn(col_info.column_name, F.lit(None).cast(spark_type))
        return df_in.select(*[c for c in target_cols if c in df_in.columns])

    # Read each group of files
    result_df = None

    for rename_key, (rename_map, group_files) in file_groups.items():
        # Split files: those with position-deletes must be read individually
        files_no_deletes = [f for f in group_files if f.data_file_id not in deletes_by_file]
        files_with_deletes = [f for f in group_files if f.data_file_id in deletes_by_file]

        # Batch-read files without deletes (faster)
        if files_no_deletes:
            paths = [resolved_paths[f.data_file_id] for f in files_no_deletes]
            batch_df = _apply_renames_and_fill(spark.read.parquet(*paths), rename_map)
            if result_df is None:
                result_df = batch_df
            else:
                result_df = result_df.unionByName(batch_df, allowMissingColumns=True)

        # Read files with deletes individually so row positions are correct
        for f in files_with_deletes:
            file_path = resolved_paths[f.data_file_id]
            file_df = spark.read.parquet(file_path)

            # Assign deterministic 0-based row positions within the file
            from pyspark.sql.window import Window
            file_df = file_df.withColumn(
                "__ducklake_pos__",
                F.row_number().over(
                    Window.orderBy(F.monotonically_increasing_id())
                ) - 1,
            )

            # Apply position-based delete files
            for del_path in deletes_by_file[f.data_file_id]:
                del_df = spark.read.parquet(del_path)
                if "pos" in del_df.columns:
                    del_positions = del_df.select(
                        F.col("pos").cast("long").alias("__ducklake_pos__")
                    )
                    file_df = file_df.join(del_positions, "__ducklake_pos__", "left_anti")

            file_df = file_df.drop("__ducklake_pos__")
            file_df = _apply_renames_and_fill(file_df, rename_map)

            if result_df is None:
                result_df = file_df
            else:
                result_df = result_df.unionByName(file_df, allowMissingColumns=True)

    if result_df is None:
        return spark.createDataFrame([], spark_schema)

    # Apply column selection
    if columns is not None:
        result_df = result_df.select(*columns)

    return result_df


def read_ducklake_changes(
    spark: SparkSession,
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    start_snapshot: int | None = None,
    end_snapshot: int | None = None,
    data_path: str | Path | None = None,
) -> DataFrame:
    """
    Read changes (CDC) between two snapshots as a PySpark DataFrame.

    Returns a DataFrame with additional columns:
    - ``snapshot_id``: the snapshot where the change occurred
    - ``change_type``: ``'insert'`` or ``'delete'``

    Parameters
    ----------
    spark
        Active SparkSession.
    path
        Path to the DuckLake metadata catalog file.
    table
        Table name.
    schema
        Schema name (default: ``"main"``).
    start_snapshot
        Start of the change range (exclusive). If None, starts from 0.
    end_snapshot
        End of the change range (inclusive). If None, reads to latest.
    data_path
        Override data path.

    Returns
    -------
    pyspark.sql.DataFrame
    """
    from pyspark.sql import functions as F

    metadata_path = os.fspath(path)
    data_path_override = os.fspath(data_path) if data_path is not None else None

    reader = DuckLakeCatalogReader(metadata_path, data_path_override=data_path_override)

    current = reader.get_current_snapshot()
    if start_snapshot is None:
        start_snapshot = 0
    if end_snapshot is None:
        end_snapshot = current.snapshot_id

    table_info = reader.get_table(table, schema, end_snapshot)
    if table_info is None:
        msg = f"Table '{schema}.{table}' not found"
        raise ValueError(msg)

    # Get files added and removed in the range
    files_with_snap = reader.get_data_files_in_range_with_snapshot(
        table_info.table_id, start_snapshot, end_snapshot
    )

    all_columns = reader.get_columns(table_info.table_id, end_snapshot)
    col_id_to_name = {col.column_id: col.column_name for col in all_columns}
    # Get name mappings for the changed files
    mapping_ids = {f.mapping_id for f, _ in files_with_snap if f.mapping_id is not None}
    name_mappings = reader.get_name_mappings_batch(mapping_ids) if mapping_ids else {}

    result_df = None

    for file_info, snap_id in files_with_snap:
        file_path = reader.resolve_data_file_path(
            file_info.path, file_info.path_is_relative, table_info
        )

        df = spark.read.parquet(file_path)

        # Apply renames
        if file_info.mapping_id is not None and file_info.mapping_id in name_mappings:
            mapping = name_mappings[file_info.mapping_id]
            for field_id, source_name in mapping.items():
                current_name = col_id_to_name.get(field_id)
                if current_name and source_name != current_name and source_name in df.columns:
                    df = df.withColumnRenamed(source_name, current_name)

        df = df.withColumn("snapshot_id", F.lit(snap_id))
        df = df.withColumn("change_type", F.lit("insert"))

        if result_df is None:
            result_df = df
        else:
            result_df = result_df.unionByName(df, allowMissingColumns=True)

    if result_df is None:
        spark_schema = _build_spark_schema(all_columns)
        from pyspark.sql.types import IntegerType, StringType, StructField
        spark_schema = spark_schema.add(StructField("snapshot_id", IntegerType(), True))
        spark_schema = spark_schema.add(StructField("change_type", StringType(), True))
        return spark.createDataFrame([], spark_schema)

    return result_df




# ------------------------------------------------------------------
# Write operations
# ------------------------------------------------------------------


def _merge_schema(writer, arrow_schema, table: str, schema: str, snap_id: int) -> None:
    """Auto-merge Arrow schema into existing table schema.

    Adds columns present in *arrow_schema* but missing in the table.
    Columns present in the table but missing in the schema are left as-is
    (NULL on write).
    """
    table_id = writer._table_exists(table, schema, snap_id)
    if table_id is None:
        return
    existing_cols = writer._get_columns_for_table(table_id, snap_id)
    existing_names = {col[1] for col in existing_cols}
    for field in arrow_schema:
        if field.name not in existing_names:
            writer.add_column(table, field.name, field.type, schema_name=schema)


def write_ducklake(
    df: DataFrame,
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    mode: str = "error",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
    max_retries: int = 3,
    retry_wait_ms: float = 100,
    retry_backoff: float = 2.0,
    schema_evolution: str = "strict",
) -> None:
    """
    Write a PySpark DataFrame to a DuckLake table.

    Parameters
    ----------
    df
        PySpark DataFrame to write.
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to write to.
    schema
        Schema name (default: ``"main"``).
    mode
        Write mode:

        - ``"error"`` (default): fail if the table already exists.
        - ``"append"``: append data; creates the table if absent.
        - ``"overwrite"``: replace all data; creates the table if absent.
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum number of rows to store inline in the metadata catalog
        instead of writing Parquet files. Set to 0 (default) to disable.
    schema_evolution
        How to handle schema mismatches on append:

        - ``"strict"`` (default): fail if schemas differ.
        - ``"merge"``: auto-add new columns (existing rows get NULL).

    Raises
    ------
    ValueError
        If *mode* is ``"error"`` and the table already exists, or if
        *mode* is not recognized.
    """
    if mode not in ("error", "append", "overwrite"):
        msg = f"Invalid write mode '{mode}'. Must be 'error', 'append', or 'overwrite'."
        raise ValueError(msg)

    from ducklake_pyspark._writer import (
        DuckLakeCatalogWriter,
        _pyspark_df_to_arrow,
    )

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    # Convert PySpark DataFrame -> Arrow once to avoid double Spark collection
    arrow_df = _pyspark_df_to_arrow(df)
    arrow_schema = {field.name: field.type for field in arrow_df.schema}
    is_empty = len(arrow_df) == 0

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
        max_retries=max_retries,
        retry_wait_ms=retry_wait_ms,
        retry_backoff=retry_backoff,
    ) as writer:
        snapshot_info = writer._get_latest_snapshot()
        if snapshot_info is None:
            snap_id = -1
        else:
            snap_id, _sv, _nci, _nfi = snapshot_info
        table_id = writer._table_exists(table, schema, snap_id)

        if mode == "error":
            if table_id is not None:
                msg = f"Table '{schema}.{table}' already exists (mode='error')"
                raise ValueError(msg)
            writer.create_table(table, arrow_schema, schema_name=schema)
            if not is_empty:
                writer.insert_data(arrow_df, table, schema_name=schema)

        elif mode == "append":
            if table_id is None:
                writer.create_table(table, arrow_schema, schema_name=schema)
            elif schema_evolution == "merge":
                _merge_schema(writer, arrow_df.schema, table, schema, snap_id)
            if not is_empty:
                writer.insert_data(arrow_df, table, schema_name=schema)

        elif mode == "overwrite":
            if table_id is None:
                writer.create_table(table, arrow_schema, schema_name=schema)
                if not is_empty:
                    writer.insert_data(arrow_df, table, schema_name=schema)
            else:
                writer.overwrite_data(arrow_df, table, schema_name=schema)


def create_ducklake_table(
    path: str | Path,
    table: str,
    spark_schema: StructType,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Create a new empty table in a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file.
    table
        Name of the table to create.
    spark_schema
        Schema for the new table, as a PySpark ``StructType``.
    schema
        Schema name (default: ``"main"``).
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table already exists.
    """
    from ducklake_pyspark._writer import (
        DuckLakeCatalogWriter,
        _pyspark_schema_to_arrow_dict,
    )

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None
    arrow_schema = _pyspark_schema_to_arrow_dict(spark_schema)

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        author=author,
        commit_message=commit_message,
    ) as writer:
        writer.create_table(table, arrow_schema, schema_name=schema)


def delete_ducklake(
    path: str | Path,
    table: str,
    predicate_sql: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """
    Delete rows matching a SQL predicate from a DuckLake table.

    Creates position-delete files for each affected Parquet data file.
    If no rows match the predicate, no snapshot is created.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file.
    table
        Name of the table to delete from.
    predicate_sql
        A SQL expression that evaluates to a boolean. Rows where the
        expression is ``True`` will be deleted.
        Example: ``"id > 10"`` or ``"region = 'EU' AND score < 50"``.
    schema
        Schema name (default: ``"main"``).
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum number of inlined rows.

    Returns
    -------
    int
        The number of rows deleted.
    """
    from ducklake_pyspark._writer import DuckLakeCatalogWriter, _sql_predicate_to_arrow

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.delete_data(
            _sql_predicate_to_arrow(predicate_sql), table, schema_name=schema
        )


def update_ducklake(
    path: str | Path,
    table: str,
    updates: dict[str, object],
    predicate_sql: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """
    Update rows matching a SQL predicate in a DuckLake table.

    Atomically deletes old rows and inserts new rows with updated
    values in a single snapshot.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file.
    table
        Name of the table to update.
    updates
        Dictionary mapping column names to new literal values.
    predicate_sql
        A SQL expression that evaluates to a boolean. Rows where the
        expression is ``True`` will be updated.
    schema
        Schema name (default: ``"main"``).
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum number of inlined rows.

    Returns
    -------
    int
        The number of rows updated.
    """
    from ducklake_pyspark._writer import DuckLakeCatalogWriter, _sql_predicate_to_arrow

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.update_data(
            updates,
            _sql_predicate_to_arrow(predicate_sql),
            table,
            schema_name=schema,
        )


def merge_ducklake(
    path: str | Path,
    table: str,
    source_df: DataFrame,
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
    Merge a source PySpark DataFrame into an existing DuckLake table.

    Matches rows on the *on* key columns, optionally updates matched
    target rows, and optionally inserts unmatched source rows.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file.
    table
        Name of the target table.
    source_df
        Source PySpark DataFrame to merge from.
    on
        Column name(s) to match on. Single string or list.
    when_matched_update
        - ``None``: matched target rows are left untouched.
        - ``True``: replace matched target rows with source rows.
        - ``dict``: update matched rows with these literal values.
    when_not_matched_insert
        If ``True`` (default), insert unmatched source rows.
    schema
        Schema name (default: ``"main"``).
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum number of inlined rows.

    Returns
    -------
    tuple[int, int]
        ``(rows_updated, rows_inserted)``.
    """
    from ducklake_pyspark._writer import DuckLakeCatalogWriter, _pyspark_df_to_arrow

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    arrow_source = _pyspark_df_to_arrow(source_df)

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.merge_data(
            arrow_source,
            table,
            on,
            when_matched_update=when_matched_update,
            when_not_matched_insert=when_not_matched_insert,
            schema_name=schema,
        )


def create_table_as_ducklake(
    df: DataFrame,
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
    Create a new table and insert data in a single snapshot (CTAS).

    The table schema is inferred from the PySpark DataFrame.

    Parameters
    ----------
    df
        PySpark DataFrame whose schema defines the new table.
    path
        Path to the DuckLake metadata catalog file.
    table
        Name of the table to create.
    schema
        Schema name (default: ``"main"``).
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum rows to store inline in the metadata catalog.

    Raises
    ------
    ValueError
        If the table already exists.
    """
    from ducklake_pyspark._writer import DuckLakeCatalogWriter, _pyspark_df_to_arrow

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    arrow_df = _pyspark_df_to_arrow(df)

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        writer.create_table_with_data(table, arrow_df, schema_name=schema)


def add_files_ducklake(
    path: str | Path,
    table: str,
    file_paths: list[str],
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
    max_retries: int = 3,
    retry_wait_ms: float = 100,
    retry_backoff: float = 2.0,
) -> int:
    """
    Register existing Parquet files into a DuckLake table.

    Files are referenced in-place (not copied). Schema validation is
    performed against the table's column definitions.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file.
    table
        Name of the target table.
    file_paths
        List of paths to Parquet files.
    schema
        Schema name (default: ``"main"``).
    data_path
        Override the data path stored in the catalog.

    Returns
    -------
    int
        The new snapshot ID.

    Raises
    ------
    ValueError
        If the table does not exist or schemas do not match.
    """
    from ducklake_pyspark._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        author=author,
        commit_message=commit_message,
        max_retries=max_retries,
        retry_wait_ms=retry_wait_ms,
        retry_backoff=retry_backoff,
    ) as writer:
        return writer.add_files(table, file_paths, schema_name=schema)


# ---------------------------------------------------------------
# Type mapping helpers
# ---------------------------------------------------------------


def _duckdb_type_to_spark(duckdb_type: str):
    """Convert a DuckDB type string to a PySpark DataType."""
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

    if duckdb_type is None:
        return StringType()

    upper = duckdb_type.upper().strip()

    # Integer types
    if upper in ("TINYINT", "INT1"):
        return ByteType()
    if upper in ("SMALLINT", "INT2", "SHORT"):
        return ShortType()
    if upper in ("INTEGER", "INT4", "INT", "SIGNED"):
        return IntegerType()
    if upper in ("BIGINT", "INT8", "LONG"):
        return LongType()

    # Unsigned → promote
    if upper == "UTINYINT":
        return ShortType()
    if upper == "USMALLINT":
        return IntegerType()
    if upper == "UINTEGER":
        return LongType()
    if upper == "UBIGINT":
        return DecimalType(20, 0)

    # Float
    if upper in ("FLOAT", "FLOAT4", "REAL"):
        return FloatType()
    if upper in ("DOUBLE", "FLOAT8"):
        return DoubleType()

    # Decimal
    if upper.startswith("DECIMAL") or upper.startswith("NUMERIC"):
        return _parse_decimal(upper)

    # Boolean
    if upper in ("BOOLEAN", "BOOL", "LOGICAL"):
        return BooleanType()

    # String
    if upper in ("VARCHAR", "TEXT", "STRING", "CHAR", "BPCHAR", "NAME") or \
       upper.startswith("VARCHAR(") or upper.startswith("CHAR("):
        return StringType()

    # Binary
    if upper in ("BLOB", "BYTEA", "BINARY", "VARBINARY"):
        return BinaryType()

    # Temporal
    if upper == "DATE":
        return DateType()
    if upper.startswith("TIMESTAMP"):
        return TimestampType()
    if upper in ("TIME", "TIMETZ", "INTERVAL"):
        return StringType()

    # UUID, JSON, ENUM
    if upper in ("UUID", "JSON") or upper.startswith("ENUM"):
        return StringType()

    # HUGEINT
    if upper in ("HUGEINT", "UHUGEINT"):
        return DecimalType(38, 0)

    # LIST
    if upper.startswith("LIST(") or upper.endswith("[]"):
        inner = upper[5:-1] if upper.startswith("LIST(") else upper[:-2]
        return ArrayType(_duckdb_type_to_spark(inner.strip()))

    # MAP
    if upper.startswith("MAP("):
        inner = upper[4:-1]
        k, v = _split_map_types(inner)
        return MapType(_duckdb_type_to_spark(k), _duckdb_type_to_spark(v))

    # STRUCT
    if upper.startswith("STRUCT("):
        return _parse_struct(duckdb_type)

    # Fallback
    return StringType()


def _parse_decimal(type_str: str):
    from pyspark.sql.types import DecimalType
    start = type_str.find("(")
    end = type_str.find(")")
    if start < 0 or end < 0:
        return DecimalType(18, 3)
    inner = type_str[start + 1:end]
    parts = inner.split(",")
    precision = int(parts[0].strip())
    scale = int(parts[1].strip()) if len(parts) > 1 else 0
    return DecimalType(precision, scale)


def _split_map_types(inner: str) -> tuple[str, str]:
    depth = 0
    for i, c in enumerate(inner):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            return inner[:i].strip(), inner[i + 1:].strip()
    return "VARCHAR", "VARCHAR"


def _parse_struct(type_str: str):
    from pyspark.sql.types import StructField, StructType
    inner = type_str[7:-1]
    fields = []
    depth = 0
    start = 0
    for i, c in enumerate(inner):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            field_def = inner[start:i].strip()
            name, ftype = field_def.split(" ", 1)
            fields.append(StructField(name.strip(), _duckdb_type_to_spark(ftype.strip()), True))
            start = i + 1
    # Last field
    field_def = inner[start:].strip()
    if field_def:
        name, ftype = field_def.split(" ", 1)
        fields.append(StructField(name.strip(), _duckdb_type_to_spark(ftype.strip()), True))
    return StructType(fields)


def _build_spark_schema(columns):
    """Build a PySpark StructType from DuckLake column definitions."""
    from pyspark.sql.types import StructField, StructType
    fields = [
        StructField(col.column_name, _duckdb_type_to_spark(col.column_type), True)
        for col in columns
    ]
    return StructType(fields)
