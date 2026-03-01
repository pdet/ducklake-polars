"""DuckDB SQL type to Polars type mapping."""

from __future__ import annotations

import re

import polars as pl

_SIMPLE_TYPE_MAP: dict[str, pl.DataType] = {
    # SQL standard names (uppercase)
    "BOOLEAN": pl.Boolean(),
    "BOOL": pl.Boolean(),
    "TINYINT": pl.Int8(),
    "SMALLINT": pl.Int16(),
    "INTEGER": pl.Int32(),
    "INT": pl.Int32(),
    "BIGINT": pl.Int64(),
    "HUGEINT": pl.Int128(),
    "UTINYINT": pl.UInt8(),
    "USMALLINT": pl.UInt16(),
    "UINTEGER": pl.UInt32(),
    "UINT": pl.UInt32(),
    "UBIGINT": pl.UInt64(),
    "UHUGEINT": pl.UInt128(),
    "FLOAT": pl.Float32(),
    "REAL": pl.Float32(),
    "DOUBLE": pl.Float64(),
    "VARCHAR": pl.String(),
    "TEXT": pl.String(),
    "STRING": pl.String(),
    "BLOB": pl.Binary(),
    "BYTEA": pl.Binary(),
    "DATE": pl.Date(),
    "TIME": pl.Time(),
    "TIMESTAMP": pl.Datetime("us"),
    "TIMESTAMP_S": pl.Datetime("us"),
    "TIMESTAMP_MS": pl.Datetime("ms"),
    "TIMESTAMP_NS": pl.Datetime("ns"),
    "TIMESTAMP WITH TIME ZONE": pl.Datetime("us", "UTC"),
    "TIMESTAMPTZ": pl.Datetime("us", "UTC"),
    "TIMESTAMP_TZ": pl.Datetime("us", "UTC"),
    "INTERVAL": pl.Duration("us"),
    "UUID": pl.Binary(),
    "JSON": pl.Binary(),
    "BIT": pl.String(),
    "TIME_NS": pl.Time(),
    "TIMESTAMP_US": pl.Datetime("us"),
    "TIMETZ": pl.Time(),
    "TIME_TZ": pl.Time(),
    "TIME WITH TIME ZONE": pl.Time(),
    "GEOMETRY": pl.Binary(),
    "VARIANT": pl.String(),
    "UNKNOWN": pl.String(),
    # DuckDB internal type names (lowercase, as stored in ducklake_column.column_type)
    "INT8": pl.Int8(),
    "INT16": pl.Int16(),
    "INT32": pl.Int32(),
    "INT64": pl.Int64(),
    "INT128": pl.Int128(),
    "UINT8": pl.UInt8(),
    "UINT16": pl.UInt16(),
    "UINT32": pl.UInt32(),
    "UINT64": pl.UInt64(),
    "UINT128": pl.UInt128(),
    "FLOAT32": pl.Float32(),
    "FLOAT16": pl.Float32(),  # Polars has no Float16; upcast to Float32
    "FLOAT64": pl.Float64(),
}


def duckdb_type_to_polars(type_str: str) -> pl.DataType:
    """
    Convert a DuckDB SQL type string to a Polars DataType.

    Parameters
    ----------
    type_str
        DuckDB type string as stored in ducklake_column.column_type.
        Examples: "BIGINT", "VARCHAR", "STRUCT(a INTEGER, b VARCHAR)",
                  "INTEGER[]", "DECIMAL(18,3)"
    """
    t = type_str.strip().upper()

    if t in _SIMPLE_TYPE_MAP:
        return _SIMPLE_TYPE_MAP[t]

    # DECIMAL(precision, scale) / NUMERIC(precision, scale)
    decimal_match = re.match(r"(?:DECIMAL|NUMERIC)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", t)
    if decimal_match:
        precision = int(decimal_match.group(1))
        scale = int(decimal_match.group(2))
        return pl.Decimal(precision, scale)

    # VARCHAR(N) / CHAR(N) / BPCHAR(N) - strip the length constraint
    if re.match(r"(?:VARCHAR|CHAR|BPCHAR|CHARACTER VARYING|CHARACTER)\s*\(\s*\d+\s*\)", t):
        return pl.String()

    # Array types: TYPE[] or TYPE ARRAY
    if t.endswith("[]"):
        inner = type_str.strip()[:-2].strip()
        return pl.List(duckdb_type_to_polars(inner))

    # LIST(TYPE) - use original case for inner type (may contain struct fields)
    list_match = re.match(r"LIST\s*\((.+)\)", type_str.strip(), re.DOTALL | re.IGNORECASE)
    if list_match:
        inner = list_match.group(1).strip()
        return pl.List(duckdb_type_to_polars(inner))

    # MAP(KEY, VALUE) -> List(Struct({key: K, value: V}))
    map_match = re.match(r"MAP\s*\((.+)\)", type_str.strip(), re.DOTALL | re.IGNORECASE)
    if map_match:
        inner = map_match.group(1).strip()
        key_type_str, value_type_str = _split_top_level_args(inner)
        return pl.List(
            pl.Struct(
                {
                    "key": duckdb_type_to_polars(key_type_str),
                    "value": duckdb_type_to_polars(value_type_str),
                }
            )
        )

    # STRUCT(field1 TYPE1, field2 TYPE2, ...)
    # Use original case for struct field names
    struct_match = re.match(r"STRUCT\s*\((.+)\)", type_str.strip(), re.DOTALL | re.IGNORECASE)
    if struct_match:
        inner = struct_match.group(1).strip()
        fields = _parse_struct_fields(inner)
        return pl.Struct({name: duckdb_type_to_polars(ftype) for name, ftype in fields})

    # Fallback: try to handle as-is
    msg = f"Unsupported DuckDB type: {type_str}"
    raise ValueError(msg)


def _split_top_level_args(s: str) -> tuple[str, str]:
    """Split a string like 'INTEGER, VARCHAR' at the top-level comma."""
    depth = 0
    for i, c in enumerate(s):
        if c in ("(", "["):
            depth += 1
        elif c in (")", "]"):
            depth -= 1
        elif c == "," and depth == 0:
            return s[:i].strip(), s[i + 1 :].strip()
    msg = f"Could not split type arguments: {s}"
    raise ValueError(msg)


def _parse_struct_fields(s: str) -> list[tuple[str, str]]:
    """Parse struct field definitions like 'a INTEGER, b VARCHAR'."""
    fields: list[tuple[str, str]] = []
    depth = 0
    current = ""

    for c in s:
        if c in ("(", "["):
            depth += 1
            current += c
        elif c in (")", "]"):
            depth -= 1
            current += c
        elif c == "," and depth == 0:
            fields.append(_parse_single_field(current.strip()))
            current = ""
        else:
            current += c

    if current.strip():
        fields.append(_parse_single_field(current.strip()))

    return fields


def _parse_single_field(field_str: str) -> tuple[str, str]:
    """Parse a single struct field like 'name TYPE' or '"quoted name" TYPE'."""
    field_str = field_str.strip()

    if field_str.startswith('"'):
        # Quoted field name - handle escaped double-quotes (SQL standard: "" -> ")
        i = 1
        name_parts: list[str] = []
        while i < len(field_str):
            if field_str[i] == '"':
                if i + 1 < len(field_str) and field_str[i + 1] == '"':
                    name_parts.append('"')
                    i += 2
                else:
                    break
            else:
                name_parts.append(field_str[i])
                i += 1
        name = "".join(name_parts)
        type_str = field_str[i + 1 :].strip()
    else:
        # Unquoted: first token is name, rest is type
        parts = field_str.split(None, 1)
        if len(parts) != 2:
            msg = f"Cannot parse struct field: {field_str}"
            raise ValueError(msg)
        name = parts[0]
        type_str = parts[1]

    return name, type_str


# Reverse mapping: Polars DataType → DuckDB internal type string (lowercase)
_POLARS_TO_DUCKDB: dict[type, str] = {
    pl.Boolean: "boolean",
    pl.Int8: "int8",
    pl.Int16: "int16",
    pl.Int32: "int32",
    pl.Int64: "int64",
    pl.Int128: "int128",
    pl.UInt8: "uint8",
    pl.UInt16: "uint16",
    pl.UInt32: "uint32",
    pl.UInt64: "uint64",
    pl.UInt128: "uint128",
    pl.Float32: "float32",
    pl.Float64: "float64",
    pl.String: "varchar",
    pl.Utf8: "varchar",
    pl.Binary: "blob",
    pl.Date: "date",
    pl.Time: "time",
    pl.Duration: "interval",
}


def polars_type_to_duckdb(dtype: pl.DataType) -> str:
    """
    Convert a Polars DataType to a DuckDB internal type string.

    Returns lowercase type strings as stored in ``ducklake_column.column_type``.
    For compound types (List, Struct), returns the parent type string;
    children must be registered separately via the column hierarchy.

    Accepts both instantiated types (``pl.String()``) and bare type classes
    (``pl.String``).
    """
    # Handle bare type classes (e.g. pl.String instead of pl.String())
    if isinstance(dtype, type):
        dtype = dtype()  # type: ignore[assignment]

    base = type(dtype)

    # Simple scalar types
    result = _POLARS_TO_DUCKDB.get(base)
    if result is not None:
        return result

    # Decimal
    if base is pl.Decimal:
        p = dtype.precision  # type: ignore[attr-defined]
        s = dtype.scale  # type: ignore[attr-defined]
        if p is None:
            p = 38
        if s is None:
            s = 0
        return f"decimal({p},{s})"

    # Datetime — map based on time unit and timezone
    if base is pl.Datetime:
        tz = dtype.time_zone  # type: ignore[attr-defined]
        if tz is not None:
            return "timestamp with time zone"
        tu = dtype.time_unit  # type: ignore[attr-defined]
        if tu == "ms":
            return "timestamp_ms"
        if tu == "ns":
            return "timestamp_ns"
        return "timestamp"

    # Compound types — return parent type string only
    if base is pl.List:
        return "list"
    if base is pl.Struct:
        return "struct"
    if base is pl.Array:
        return "list"

    # Categorical / Enum → store as varchar (ENUM is catalog-level in DuckDB,
    # not stored in parquet; we flatten to plain strings)
    if base is pl.Categorical:
        return "varchar"
    if base is pl.Enum:
        return "varchar"

    msg = f"Cannot map Polars type {dtype} to DuckDB type"
    raise ValueError(msg)


def resolve_column_type(
    column_id: int,
    column_type: str,
    all_columns: list,
) -> pl.DataType:
    """
    Resolve a column's Polars type using the DuckLake column hierarchy.

    DuckLake stores compound types (list, struct, map) as a parent-child
    hierarchy in the ducklake_column table. This function reconstructs
    the full Polars type by walking the hierarchy.

    Parameters
    ----------
    column_id
        The column_id of this column.
    column_type
        The type string (e.g. "list", "struct", "map", "int32", "varchar").
    all_columns
        All columns for this table (including nested children).
    """
    t = column_type.strip().upper()

    # Find child columns for compound types
    children = [c for c in all_columns if c.parent_column == column_id]

    if t == "LIST":
        if len(children) != 1:
            msg = f"LIST column {column_id} expected 1 child element, found {len(children)}"
            raise ValueError(msg)
        element = children[0]
        return pl.List(resolve_column_type(element.column_id, element.column_type, all_columns))

    if t == "STRUCT":
        if not children:
            msg = f"STRUCT column {column_id} has no child field columns"
            raise ValueError(msg)
        fields = {}
        for child in sorted(children, key=lambda c: c.column_order):
            fields[child.column_name] = resolve_column_type(
                child.column_id, child.column_type, all_columns
            )
        return pl.Struct(fields)

    if t == "MAP":
        if len(children) < 2:
            msg = f"MAP column {column_id} needs key and value children"
            raise ValueError(msg)
        key_col = next((c for c in children if c.column_name == "key"), None)
        val_col = next((c for c in children if c.column_name == "value"), None)
        if key_col is None or val_col is None:
            msg = f"MAP column {column_id} missing 'key' or 'value' child column"
            raise ValueError(msg)
        return pl.List(
            pl.Struct(
                {
                    "key": resolve_column_type(key_col.column_id, key_col.column_type, all_columns),
                    "value": resolve_column_type(val_col.column_id, val_col.column_type, all_columns),
                }
            )
        )

    # Fallback to string-based parsing for any remaining formats
    return duckdb_type_to_polars(column_type)
