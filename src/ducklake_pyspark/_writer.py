"""DuckLake catalog writer — PySpark utilities and core writer bridge.

Provides conversion functions between PySpark types and Arrow types,
plus a SQL predicate evaluator for Arrow tables using DuckDB.
"""

from __future__ import annotations

from typing import Any, Callable

import pyarrow as pa

from ducklake_core._writer import DuckLakeCatalogWriter  # noqa: F401
from ducklake_core._writer import TransactionConflictError  # noqa: F401


def _pyspark_df_to_arrow(df: Any) -> pa.Table:
    """Convert a PySpark DataFrame to a PyArrow Table.

    Uses ``df.toArrow()`` on Spark 4+ and falls back to
    ``pa.Table.from_pandas(df.toPandas())`` for older versions.
    """
    if hasattr(df, "toArrow"):
        return df.toArrow()
    return pa.Table.from_pandas(df.toPandas())


def _spark_type_to_arrow(dt: Any) -> pa.DataType:
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
        StructType,
        TimestampType,
    )

    if isinstance(dt, ByteType):
        return pa.int8()
    if isinstance(dt, ShortType):
        return pa.int16()
    if isinstance(dt, IntegerType):
        return pa.int32()
    if isinstance(dt, LongType):
        return pa.int64()
    if isinstance(dt, FloatType):
        return pa.float32()
    if isinstance(dt, DoubleType):
        return pa.float64()
    if isinstance(dt, StringType):
        return pa.string()
    if isinstance(dt, BooleanType):
        return pa.bool_()
    if isinstance(dt, DateType):
        return pa.date32()
    if isinstance(dt, TimestampType):
        return pa.timestamp("us")
    if isinstance(dt, DecimalType):
        return pa.decimal128(dt.precision, dt.scale)
    if isinstance(dt, BinaryType):
        return pa.binary()
    if isinstance(dt, ArrayType):
        return pa.list_(_spark_type_to_arrow(dt.elementType))
    if isinstance(dt, MapType):
        return pa.map_(
            _spark_type_to_arrow(dt.keyType),
            _spark_type_to_arrow(dt.valueType),
        )
    if isinstance(dt, StructType):
        return pa.struct(
            [
                pa.field(
                    f.name, _spark_type_to_arrow(f.dataType), nullable=f.nullable,
                )
                for f in dt.fields
            ]
        )
    # TimestampNTZType (Spark 3.4+)
    try:
        from pyspark.sql.types import TimestampNTZType

        if isinstance(dt, TimestampNTZType):
            return pa.timestamp("us")
    except ImportError:
        pass
    return pa.string()  # fallback


def _pyspark_schema_to_arrow_dict(
    spark_schema: Any,
) -> dict[str, pa.DataType]:
    """Convert a PySpark StructType to a dict of Arrow types."""
    return {f.name: _spark_type_to_arrow(f.dataType) for f in spark_schema.fields}


def _sql_predicate_to_arrow(
    sql_pred: str,
) -> Callable[[pa.Table], pa.ChunkedArray]:
    """Convert a SQL predicate string to an Arrow-compatible callable.

    The predicate is evaluated using DuckDB on the Arrow table.
    Column references in the predicate use standard SQL syntax.
    """

    def apply(table: pa.Table) -> pa.ChunkedArray:
        import duckdb

        con = duckdb.connect()
        con.register("__tbl__", table)
        result = con.execute(
            f"SELECT ({sql_pred}) AS __pred__ FROM __tbl__"
        ).fetch_arrow_table()
        return result.column("__pred__")

    return apply
