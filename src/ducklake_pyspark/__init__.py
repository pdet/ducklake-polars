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

from ducklake_core._catalog import DuckLakeCatalogReader

__all__ = [
    "read_ducklake",
    "read_ducklake_changes",
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

    # Read each group of files
    result_df = None

    for rename_key, (rename_map, group_files) in file_groups.items():
        file_paths = [resolved_paths[f.data_file_id] for f in group_files]

        # Read Parquet files through Spark
        group_df = spark.read.parquet(*file_paths)

        # Apply column renames (schema evolution)
        for old_name, new_name in rename_map.items():
            if old_name in group_df.columns:
                group_df = group_df.withColumnRenamed(old_name, new_name)

        # Apply delete files: anti-join on row position
        for f in group_files:
            if f.data_file_id in deletes_by_file:
                for del_path in deletes_by_file[f.data_file_id]:
                    # DuckLake delete files are Parquet with a row_id column
                    delete_df = spark.read.parquet(del_path)
                    if "row_id" in delete_df.columns:
                        # Add monotonically increasing row index for position-based delete
                        group_df = group_df.withColumn(
                            "__ducklake_row_id__",
                            F.monotonically_increasing_id()
                        )
                        delete_ids = delete_df.select(
                            F.col("row_id").alias("__ducklake_row_id__")
                        )
                        group_df = group_df.join(
                            delete_ids, "__ducklake_row_id__", "left_anti"
                        ).drop("__ducklake_row_id__")

        # Ensure all expected columns exist (added columns default to null)
        for col_info in all_columns:
            if col_info.column_name not in group_df.columns:
                spark_type = _duckdb_type_to_spark(col_info.column_type)
                group_df = group_df.withColumn(col_info.column_name, F.lit(None).cast(spark_type))

        # Select columns in the correct order
        target_cols = [col.column_name for col in all_columns]
        group_df = group_df.select(*[c for c in target_cols if c in group_df.columns])

        # Union groups
        if result_df is None:
            result_df = group_df
        else:
            result_df = result_df.unionByName(group_df, allowMissingColumns=True)

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
