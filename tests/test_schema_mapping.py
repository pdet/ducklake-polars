"""Unit tests for the DuckDB type -> Polars type mapping."""

from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars._schema import duckdb_type_to_polars


class TestSimpleTypes:
    """Test mapping of simple scalar types."""

    @pytest.mark.parametrize(
        "duckdb_type,expected",
        [
            # SQL standard names
            ("BOOLEAN", pl.Boolean()),
            ("TINYINT", pl.Int8()),
            ("SMALLINT", pl.Int16()),
            ("INTEGER", pl.Int32()),
            ("BIGINT", pl.Int64()),
            ("HUGEINT", pl.Int128()),
            ("UTINYINT", pl.UInt8()),
            ("USMALLINT", pl.UInt16()),
            ("UINTEGER", pl.UInt32()),
            ("UBIGINT", pl.UInt64()),
            ("UHUGEINT", pl.UInt128()),
            ("FLOAT", pl.Float32()),
            ("DOUBLE", pl.Float64()),
            ("VARCHAR", pl.String()),
            ("BLOB", pl.Binary()),
            ("DATE", pl.Date()),
            ("TIME", pl.Time()),
            ("TIMESTAMP", pl.Datetime("us")),
            ("TIMESTAMP_S", pl.Datetime("us")),
            ("TIMESTAMP_MS", pl.Datetime("ms")),
            ("TIMESTAMP_NS", pl.Datetime("ns")),
            ("TIMESTAMP WITH TIME ZONE", pl.Datetime("us", "UTC")),
            ("TIMESTAMPTZ", pl.Datetime("us", "UTC")),
            ("INTERVAL", pl.Duration("us")),
            ("UUID", pl.Binary()),
            ("JSON", pl.Binary()),
            # DuckLake internal type names (lowercase)
            ("boolean", pl.Boolean()),
            ("int8", pl.Int8()),
            ("int16", pl.Int16()),
            ("int32", pl.Int32()),
            ("int64", pl.Int64()),
            ("int128", pl.Int128()),
            ("uint8", pl.UInt8()),
            ("uint16", pl.UInt16()),
            ("uint32", pl.UInt32()),
            ("uint64", pl.UInt64()),
            ("uint128", pl.UInt128()),
            ("float32", pl.Float32()),
            ("float64", pl.Float64()),
            ("varchar", pl.String()),
            ("blob", pl.Binary()),
            ("date", pl.Date()),
            ("time", pl.Time()),
            ("time_ns", pl.Time()),
            ("timestamp", pl.Datetime("us")),
            ("timestamp_us", pl.Datetime("us")),
            ("timestamp_s", pl.Datetime("us")),
            ("timestamp_ms", pl.Datetime("ms")),
            ("timestamp_ns", pl.Datetime("ns")),
            ("timestamptz", pl.Datetime("us", "UTC")),
            ("timetz", pl.Time()),
            ("interval", pl.Duration("us")),
            ("uuid", pl.Binary()),
            ("json", pl.Binary()),
            ("geometry", pl.Binary()),
            ("variant", pl.String()),
            ("unknown", pl.String()),
            # Per-shape geometry sub-types (DuckLake spatial extension)
            ("point", pl.Binary()),
            ("linestring", pl.Binary()),
            ("polygon", pl.Binary()),
            ("multipoint", pl.Binary()),
            ("multilinestring", pl.Binary()),
            ("multipolygon", pl.Binary()),
            ("geometrycollection", pl.Binary()),
            ("point_z", pl.Binary()),
            ("linestring_zm", pl.Binary()),
            ("polygon_m", pl.Binary()),
            ("multipoint_z", pl.Binary()),
            ("multipolygon_zm", pl.Binary()),
            ("wkb_blob", pl.Binary()),
        ],
    )
    def test_simple_type_mapping(self, duckdb_type, expected):
        result = duckdb_type_to_polars(duckdb_type)
        assert result == expected


class TestDecimalType:
    """Test DECIMAL type mapping."""

    def test_decimal(self):
        result = duckdb_type_to_polars("DECIMAL(18, 3)")
        assert result == pl.Decimal(18, 3)

    def test_decimal_no_spaces(self):
        result = duckdb_type_to_polars("DECIMAL(10,2)")
        assert result == pl.Decimal(10, 2)


class TestComplexTypes:
    """Test complex/nested type mapping."""

    def test_list_bracket_syntax(self):
        result = duckdb_type_to_polars("INTEGER[]")
        assert result == pl.List(pl.Int32())

    def test_list_function_syntax(self):
        result = duckdb_type_to_polars("LIST(VARCHAR)")
        assert result == pl.List(pl.String())

    def test_struct(self):
        result = duckdb_type_to_polars("STRUCT(a INTEGER, b VARCHAR)")
        assert result == pl.Struct({"a": pl.Int32(), "b": pl.String()})

    def test_map(self):
        result = duckdb_type_to_polars("MAP(VARCHAR, INTEGER)")
        expected = pl.List(pl.Struct({"key": pl.String(), "value": pl.Int32()}))
        assert result == expected

    def test_nested_list_of_structs(self):
        result = duckdb_type_to_polars("STRUCT(x INTEGER, y VARCHAR)[]")
        assert result == pl.List(pl.Struct({"x": pl.Int32(), "y": pl.String()}))

    def test_nested_struct_with_list(self):
        result = duckdb_type_to_polars("STRUCT(a INTEGER, b INTEGER[])")
        assert result == pl.Struct({"a": pl.Int32(), "b": pl.List(pl.Int32())})

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported"):
            duckdb_type_to_polars("COMPLETELY_FAKE_TYPE")

    def test_decimal_lowercase(self):
        """DuckLake stores decimal as lowercase 'decimal(w,s)'."""
        result = duckdb_type_to_polars("decimal(18,3)")
        assert result == pl.Decimal(18, 3)

    def test_numeric_type(self):
        result = duckdb_type_to_polars("NUMERIC(10,2)")
        assert result == pl.Decimal(10, 2)


class TestParameterizedTypes:
    """Test types with length/precision parameters."""

    def test_varchar_with_length(self):
        result = duckdb_type_to_polars("VARCHAR(255)")
        assert result == pl.String()

    def test_char_with_length(self):
        result = duckdb_type_to_polars("CHAR(10)")
        assert result == pl.String()

    def test_bpchar_with_length(self):
        result = duckdb_type_to_polars("BPCHAR(5)")
        assert result == pl.String()
