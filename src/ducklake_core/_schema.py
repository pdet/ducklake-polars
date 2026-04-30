"""DuckDB SQL type to Arrow type mapping."""

from __future__ import annotations

import re

import pyarrow as pa

_SIMPLE_TYPE_MAP: dict[str, pa.DataType] = {
    # SQL standard names (uppercase)
    "BOOLEAN": pa.bool_(),
    "BOOL": pa.bool_(),
    "TINYINT": pa.int8(),
    "SMALLINT": pa.int16(),
    "INTEGER": pa.int32(),
    "INT": pa.int32(),
    "BIGINT": pa.int64(),
    "HUGEINT": pa.large_binary(),  # Arrow has no int128; store as large_binary
    "UTINYINT": pa.uint8(),
    "USMALLINT": pa.uint16(),
    "UINTEGER": pa.uint32(),
    "UINT": pa.uint32(),
    "UBIGINT": pa.uint64(),
    "UHUGEINT": pa.large_binary(),
    "FLOAT": pa.float32(),
    "REAL": pa.float32(),
    "DOUBLE": pa.float64(),
    "VARCHAR": pa.string(),
    "TEXT": pa.string(),
    "STRING": pa.string(),
    "BLOB": pa.binary(),
    "BYTEA": pa.binary(),
    "DATE": pa.date32(),
    "TIME": pa.time64("us"),
    "TIMESTAMP": pa.timestamp("us"),
    "TIMESTAMP_S": pa.timestamp("s"),
    "TIMESTAMP_MS": pa.timestamp("ms"),
    "TIMESTAMP_NS": pa.timestamp("ns"),
    "TIMESTAMP WITH TIME ZONE": pa.timestamp("us", tz="UTC"),
    "TIMESTAMPTZ": pa.timestamp("us", tz="UTC"),
    "TIMESTAMP_TZ": pa.timestamp("us", tz="UTC"),
    "INTERVAL": pa.duration("us"),
    "UUID": pa.binary(16),
    "JSON": pa.string(),
    "BIT": pa.string(),
    "TIME_NS": pa.time64("ns"),
    "TIMESTAMP_US": pa.timestamp("us"),
    "TIMETZ": pa.time64("us"),
    "TIME_TZ": pa.time64("us"),
    "TIME WITH TIME ZONE": pa.time64("us"),
    "GEOMETRY": pa.binary(),
    "VARIANT": pa.string(),
    "UNKNOWN": pa.string(),
    # Per-shape geometry types (DuckLake spatial extension) — all WKB.
    "POINT": pa.binary(),
    "LINESTRING": pa.binary(),
    "POLYGON": pa.binary(),
    "MULTIPOINT": pa.binary(),
    "MULTILINESTRING": pa.binary(),
    "MULTIPOLYGON": pa.binary(),
    "GEOMETRYCOLLECTION": pa.binary(),
    "POINT_Z": pa.binary(),
    "LINESTRING_Z": pa.binary(),
    "POLYGON_Z": pa.binary(),
    "MULTIPOINT_Z": pa.binary(),
    "MULTILINESTRING_Z": pa.binary(),
    "MULTIPOLYGON_Z": pa.binary(),
    "GEOMETRYCOLLECTION_Z": pa.binary(),
    "POINT_M": pa.binary(),
    "LINESTRING_M": pa.binary(),
    "POLYGON_M": pa.binary(),
    "MULTIPOINT_M": pa.binary(),
    "MULTILINESTRING_M": pa.binary(),
    "MULTIPOLYGON_M": pa.binary(),
    "GEOMETRYCOLLECTION_M": pa.binary(),
    "POINT_ZM": pa.binary(),
    "LINESTRING_ZM": pa.binary(),
    "POLYGON_ZM": pa.binary(),
    "MULTIPOINT_ZM": pa.binary(),
    "MULTILINESTRING_ZM": pa.binary(),
    "MULTIPOLYGON_ZM": pa.binary(),
    "GEOMETRYCOLLECTION_ZM": pa.binary(),
    "WKB_BLOB": pa.binary(),
    # DuckDB internal type names (lowercase, as stored in ducklake_column.column_type)
    "INT8": pa.int8(),
    "INT16": pa.int16(),
    "INT32": pa.int32(),
    "INT64": pa.int64(),
    "INT128": pa.large_binary(),
    "UINT8": pa.uint8(),
    "UINT16": pa.uint16(),
    "UINT32": pa.uint32(),
    "UINT64": pa.uint64(),
    "UINT128": pa.large_binary(),
    "FLOAT32": pa.float32(),
    "FLOAT16": pa.float16(),
    "FLOAT64": pa.float64(),
}


def duckdb_type_to_arrow(type_str: str) -> pa.DataType:
    """
    Convert a DuckDB SQL type string to a PyArrow DataType.

    Parameters
    ----------
    type_str
        DuckDB type string as stored in ducklake_column.column_type.
    """
    t = type_str.strip().upper()

    if t in _SIMPLE_TYPE_MAP:
        return _SIMPLE_TYPE_MAP[t]

    # DECIMAL(precision, scale) / NUMERIC(precision, scale)
    decimal_match = re.match(r"(?:DECIMAL|NUMERIC)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", t)
    if decimal_match:
        precision = int(decimal_match.group(1))
        scale = int(decimal_match.group(2))
        return pa.decimal128(precision, scale)

    # VARCHAR(N) / CHAR(N) / BPCHAR(N)
    if re.match(r"(?:VARCHAR|CHAR|BPCHAR|CHARACTER VARYING|CHARACTER)\s*\(\s*\d+\s*\)", t):
        return pa.string()

    # Array types: TYPE[]
    if t.endswith("[]"):
        inner = type_str.strip()[:-2].strip()
        return pa.list_(duckdb_type_to_arrow(inner))

    # LIST(TYPE)
    list_match = re.match(r"LIST\s*\((.+)\)", type_str.strip(), re.DOTALL | re.IGNORECASE)
    if list_match:
        inner = list_match.group(1).strip()
        return pa.list_(duckdb_type_to_arrow(inner))

    # MAP(KEY, VALUE)
    map_match = re.match(r"MAP\s*\((.+)\)", type_str.strip(), re.DOTALL | re.IGNORECASE)
    if map_match:
        inner = map_match.group(1).strip()
        key_type_str, value_type_str = _split_top_level_args(inner)
        return pa.map_(duckdb_type_to_arrow(key_type_str), duckdb_type_to_arrow(value_type_str))

    # STRUCT(field1 TYPE1, field2 TYPE2, ...)
    struct_match = re.match(r"STRUCT\s*\((.+)\)", type_str.strip(), re.DOTALL | re.IGNORECASE)
    if struct_match:
        inner = struct_match.group(1).strip()
        fields = _parse_struct_fields(inner)
        return pa.struct([pa.field(name, duckdb_type_to_arrow(ftype)) for name, ftype in fields])

    msg = f"Unsupported DuckDB type: {type_str}"
    raise ValueError(msg)


# Reverse mapping: Arrow DataType id → DuckDB internal type string
_ARROW_TO_DUCKDB: dict[int, str] = {
    pa.bool_().id: "boolean",
    pa.int8().id: "int8",
    pa.int16().id: "int16",
    pa.int32().id: "int32",
    pa.int64().id: "int64",
    pa.uint8().id: "uint8",
    pa.uint16().id: "uint16",
    pa.uint32().id: "uint32",
    pa.uint64().id: "uint64",
    pa.float16().id: "float32",
    pa.float32().id: "float32",
    pa.float64().id: "float64",
    pa.string().id: "varchar",
    pa.large_string().id: "varchar",
    pa.binary().id: "blob",
    pa.large_binary().id: "blob",
    pa.date32().id: "date",
    pa.date64().id: "date",
}


def arrow_type_to_duckdb(dtype: pa.DataType) -> str:
    """
    Convert a PyArrow DataType to a DuckDB internal type string.

    For compound types (list, struct), returns the parent type string.
    """
    type_id = dtype.id

    result = _ARROW_TO_DUCKDB.get(type_id)
    if result is not None:
        return result

    # Time types
    if pa.types.is_time(dtype):
        return "time"

    # Duration
    if pa.types.is_duration(dtype):
        return "interval"

    # Decimal
    if pa.types.is_decimal(dtype):
        return f"decimal({dtype.precision},{dtype.scale})"

    # Timestamp
    if pa.types.is_timestamp(dtype):
        if dtype.tz is not None:
            return "timestamp with time zone"
        unit = dtype.unit
        if unit == "ms":
            return "timestamp_ms"
        if unit == "ns":
            return "timestamp_ns"
        return "timestamp"

    # Compound types
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return "list"
    if pa.types.is_struct(dtype):
        return "struct"
    if pa.types.is_map(dtype):
        return "map"

    # Dictionary (e.g. from Polars Categorical/Enum) → store as varchar
    if pa.types.is_dictionary(dtype):
        return "varchar"

    # Fixed-size binary
    if pa.types.is_fixed_size_binary(dtype):
        return "blob"

    # UNION types — not supported by DuckLake upstream
    if pa.types.is_union(dtype):
        from ducklake_core._exceptions import UnsupportedUnionTypeError
        raise UnsupportedUnionTypeError(
            f"Arrow UNION type ({dtype}) cannot be mapped to a DuckDB type. "
            f"DuckLake does not support UNION types (upstream limitation). "
            f"Use union_handling='to_struct' to convert UNION columns to "
            f"STRUCT before writing."
        )

    msg = f"Cannot map Arrow type {dtype} to DuckDB type"
    raise ValueError(msg)


# Map common DuckDB SQL type names → DuckLake-canonical type names. DuckLake
# stores types using its own short names (``int32``, ``int64``, ``varchar``,
# ``float64``, ``timestamp_us``…); when a user passes a SQL alias we accept
# it but persist the canonical form so DuckDB's ducklake reader can parse it.
_DUCKLAKE_TYPE_ALIASES: dict[str, str] = {
    "tinyint": "int8",
    "smallint": "int16",
    "int": "int32",
    "integer": "int32",
    "bigint": "int64",
    "hugeint": "int128",
    "utinyint": "uint8",
    "usmallint": "uint16",
    "uinteger": "uint32",
    "uint": "uint32",
    "ubigint": "uint64",
    "uhugeint": "uint128",
    "real": "float32",
    "float": "float32",
    "double": "float64",
    "double precision": "float64",
    "string": "varchar",
    "text": "varchar",
    "bytea": "blob",
    "bool": "boolean",
    "timestamp_us": "timestamp_us",
    "timestamp without time zone": "timestamp",
    "timestamp with time zone": "timestamptz",
    "datetime": "timestamp",
    "time without time zone": "time",
    "time with time zone": "timetz",
}


def to_ducklake_type(type_str: str) -> str:
    """Canonicalize a DuckDB / SQL type string to its DuckLake form.

    Examples
    --------
    >>> to_ducklake_type("INTEGER")  # 'int32'
    >>> to_ducklake_type("varchar")  # 'varchar' (already canonical)
    """
    s = type_str.strip().lower()
    if s.startswith("decimal"):
        return s  # decimal(p, s) — pass through with original parameters
    return _DUCKLAKE_TYPE_ALIASES.get(s, s)


def resolve_column_type(
    column_id: int,
    column_type: str,
    all_columns: list,
) -> pa.DataType:
    """
    Resolve a column's Arrow type using the DuckLake column hierarchy.

    DuckLake stores compound types (list, struct, map) as a parent-child
    hierarchy in the ducklake_column table. This function reconstructs
    the full Arrow type by walking the hierarchy.
    """
    t = column_type.strip().upper()

    children = [c for c in all_columns if c.parent_column == column_id]

    if t == "LIST":
        if len(children) != 1:
            msg = f"LIST column {column_id} expected 1 child, found {len(children)}"
            raise ValueError(msg)
        element = children[0]
        return pa.list_(resolve_column_type(element.column_id, element.column_type, all_columns))

    if t == "STRUCT":
        if not children:
            msg = f"STRUCT column {column_id} has no child field columns"
            raise ValueError(msg)
        fields = []
        for child in sorted(children, key=lambda c: c.column_order):
            child_type = resolve_column_type(child.column_id, child.column_type, all_columns)
            fields.append(pa.field(child.column_name, child_type))
        return pa.struct(fields)

    if t == "MAP":
        if len(children) < 2:
            msg = f"MAP column {column_id} needs key and value children"
            raise ValueError(msg)
        key_col = next((c for c in children if c.column_name == "key"), None)
        val_col = next((c for c in children if c.column_name == "value"), None)
        if key_col is None or val_col is None:
            msg = f"MAP column {column_id} missing 'key' or 'value' child column"
            raise ValueError(msg)
        return pa.map_(
            resolve_column_type(key_col.column_id, key_col.column_type, all_columns),
            resolve_column_type(val_col.column_id, val_col.column_type, all_columns),
        )

    return duckdb_type_to_arrow(column_type)


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
        parts = field_str.split(None, 1)
        if len(parts) != 2:
            msg = f"Cannot parse struct field: {field_str}"
            raise ValueError(msg)
        name = parts[0]
        type_str = parts[1]

    return name, type_str
