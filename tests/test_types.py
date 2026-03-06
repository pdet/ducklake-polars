"""Type mapping tests for ducklake-dataframe."""

from __future__ import annotations

import math

import polars as pl
import pytest

from ducklake_polars import read_ducklake, scan_ducklake


class TestScalarTypes:
    """Test reading tables with various scalar DuckDB types."""

    def test_integer_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a TINYINT,
                b SMALLINT,
                c INTEGER,
                d BIGINT
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (1, 100, 10000, 1000000)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {
            "a": pl.Int8,
            "b": pl.Int16,
            "c": pl.Int32,
            "d": pl.Int64,
        }
        assert result.row(0) == (1, 100, 10000, 1000000)

    def test_unsigned_integer_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a UTINYINT,
                b USMALLINT,
                c UINTEGER,
                d UBIGINT
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (1, 100, 10000, 1000000)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {
            "a": pl.UInt8,
            "b": pl.UInt16,
            "c": pl.UInt32,
            "d": pl.UInt64,
        }

    @pytest.mark.xfail(reason="DuckDB writes HUGEINT as Float64 in Parquet, Polars reads Float64")
    def test_hugeint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a HUGEINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (123456789012345678901234)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Int128

    @pytest.mark.xfail(reason="DuckDB writes UHUGEINT as Float64 in Parquet, Polars reads Float64")
    def test_uhugeint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a UHUGEINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (123456789012345678901234)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.UInt128

    def test_float_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a FLOAT, b DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (3.14, 2.718281828)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Float32, "b": pl.Float64}
        assert abs(result["a"][0] - 3.14) < 0.01
        assert abs(result["b"][0] - 2.718281828) < 1e-6

    def test_boolean(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a BOOLEAN)")
        cat.execute("INSERT INTO ducklake.test VALUES (true), (false), (NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Boolean}
        assert result["a"].to_list() == [True, False, None]

    def test_varchar(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES ('hello'), ('world'), ('')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.String}
        assert result["a"].to_list() == ["hello", "world", ""]

    def test_blob(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a BLOB)")
        cat.execute("INSERT INTO ducklake.test VALUES ('\\x01\\x02\\x03'::BLOB)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Binary}
        assert result["a"][0] == b"\x01\x02\x03"

    def test_date(self, ducklake_catalog):
        import datetime

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DATE)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15'), ('2024-12-31')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Date}
        assert result["a"].to_list() == [
            datetime.date(2024, 1, 15),
            datetime.date(2024, 12, 31),
        ]

    def test_timestamp(self, ducklake_catalog):
        import datetime

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Datetime
        assert result["a"][0] == datetime.datetime(2024, 1, 15, 10, 30, 0)

    def test_timestamp_s(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP_S)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Datetime

    def test_timestamp_ms(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP_MS)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00.123')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Datetime

    def test_timestamp_ns(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP_NS)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00.123456789')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Datetime

    def test_timestamptz(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMPTZ)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00+00')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Datetime

    def test_time(self, ducklake_catalog):
        import datetime

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIME)")
        cat.execute("INSERT INTO ducklake.test VALUES ('10:30:00'), ('23:59:59')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Time}
        assert result["a"].to_list() == [
            datetime.time(10, 30, 0),
            datetime.time(23, 59, 59),
        ]

    @pytest.mark.xfail(reason="Polars cannot read Parquet month_day_millisecond_interval type natively")
    def test_interval(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTERVAL)")
        cat.execute("INSERT INTO ducklake.test VALUES (INTERVAL 1 HOUR), (INTERVAL 30 MINUTE)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Duration
        assert result.shape == (2, 1)

    def test_uuid(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a UUID)")
        cat.execute("INSERT INTO ducklake.test VALUES ('550e8400-e29b-41d4-a716-446655440000')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Binary}
        # UUID stored as 16-byte binary in Parquet
        uuid_bytes = result["a"][0]
        assert uuid_bytes is not None
        assert len(uuid_bytes) == 16

    def test_json(self, ducklake_catalog):
        import json

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a JSON)")
        cat.execute("""INSERT INTO ducklake.test VALUES ('{"key": "value"}'), ('[1, 2, 3]')""")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Binary}
        assert result.shape == (2, 1)
        # JSON text is stored as binary in Parquet, can be cast to String
        texts = result["a"].cast(pl.String).to_list()
        parsed_0 = json.loads(texts[0])
        assert parsed_0 == {"key": "value"}
        parsed_1 = json.loads(texts[1])
        assert parsed_1 == [1, 2, 3]

    def test_decimal(self, ducklake_catalog):
        from decimal import Decimal

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DECIMAL(10, 2))")
        cat.execute("INSERT INTO ducklake.test VALUES (123.45), (678.90)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        values = result["a"].to_list()
        assert values == [Decimal("123.45"), Decimal("678.90")]

    def test_decimal_various_precisions(self, ducklake_catalog):
        from decimal import Decimal

        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a DECIMAL(5, 2),
                b DECIMAL(18, 6),
                c DECIMAL(38, 10)
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (123.45, 123456.789012, 12345678901234567890.1234567890)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 3)
        assert result["a"][0] == Decimal("123.45")
        assert result["b"][0] == Decimal("123456.789012")


class TestComplexTypes:
    """Test reading tables with complex/nested DuckDB types."""

    def test_list_type(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER[])")
        cat.execute("INSERT INTO ducklake.test VALUES ([1, 2, 3]), ([4, 5])")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        assert result["a"].to_list() == [[1, 2, 3], [4, 5]]

    def test_list_of_varchar(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR[])")
        cat.execute("INSERT INTO ducklake.test VALUES (['hello', 'world']), (['foo'])")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        assert result["a"].to_list() == [["hello", "world"], ["foo"]]

    def test_struct_type(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(x INTEGER, y VARCHAR))")
        cat.execute("INSERT INTO ducklake.test VALUES ({'x': 1, 'y': 'hello'})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)
        val = result["a"][0]
        assert val == {"x": 1, "y": "hello"}

    @pytest.mark.xfail(reason="MAP type reading from DuckDB Parquet is broken in Polars 1.36")
    def test_map_type(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a MAP(VARCHAR, INTEGER))")
        cat.execute("INSERT INTO ducklake.test VALUES (MAP {'a': 1, 'b': 2})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)

    def test_nested_list_of_structs(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(x INTEGER, y VARCHAR)[])")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "([{'x': 1, 'y': 'a'}, {'x': 2, 'y': 'b'}])"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)
        assert result["a"].to_list() == [[{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]]

    def test_list_of_lists(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER[][])")
        cat.execute("INSERT INTO ducklake.test VALUES ([[1, 2], [3, 4]]), ([[5]])")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        assert result["a"].to_list() == [[[1, 2], [3, 4]], [[5]]]

    def test_struct_with_list_field(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(name VARCHAR, scores INTEGER[]))")
        cat.execute("INSERT INTO ducklake.test VALUES ({'name': 'Alice', 'scores': [90, 85, 92]})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)
        val = result["a"][0]
        assert val["name"] == "Alice"
        assert val["scores"] == [90, 85, 92]


class TestMixedColumns:
    """Test tables with multiple column types."""

    def test_mixed_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                id INTEGER,
                name VARCHAR,
                score DOUBLE,
                active BOOLEAN,
                created DATE
            )
        """)
        cat.execute("""
            INSERT INTO ducklake.test VALUES
                (1, 'Alice', 95.5, true, '2024-01-01'),
                (2, 'Bob', 87.3, false, '2024-01-02'),
                (3, 'Charlie', NULL, true, NULL)
        """)
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 5)
        assert result["id"].to_list() == [1, 2, 3]
        assert result["name"].to_list() == ["Alice", "Bob", "Charlie"]
        assert result["active"].to_list() == [True, False, True]

    def test_all_temporal_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a DATE,
                b TIME,
                c TIMESTAMP
            )
        """)
        cat.execute("""
            INSERT INTO ducklake.test VALUES
                ('2024-01-15', '10:30:00', '2024-01-15 10:30:00')
        """)
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 3)
        assert result.schema["a"] == pl.Date
        assert result.schema["b"] == pl.Time
        assert result.schema["c"].base_type() == pl.Datetime

    def test_all_integer_widths(self, ducklake_catalog):
        """Test all integer types in a single table."""
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a TINYINT,
                b SMALLINT,
                c INTEGER,
                d BIGINT,
                e UTINYINT,
                f USMALLINT,
                g UINTEGER,
                h UBIGINT
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (1, 2, 3, 4, 5, 6, 7, 8)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {
            "a": pl.Int8,
            "b": pl.Int16,
            "c": pl.Int32,
            "d": pl.Int64,
            "e": pl.UInt8,
            "f": pl.UInt16,
            "g": pl.UInt32,
            "h": pl.UInt64,
        }
        assert result.row(0) == (1, 2, 3, 4, 5, 6, 7, 8)


class TestFloatEdgeCases:
    """Test float edge cases: NaN, infinity, and filtering behavior."""

    def test_float_nan(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a FLOAT, b DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES ('NaN'::FLOAT, 'NaN'::DOUBLE)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 2)
        assert math.isnan(result["a"][0])
        assert math.isnan(result["b"][0])

    def test_float_infinity(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a FLOAT, b DOUBLE)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES ('inf'::FLOAT, '-inf'::DOUBLE)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 2)
        assert math.isinf(result["a"][0]) and result["a"][0] > 0
        assert math.isinf(result["b"][0]) and result["b"][0] < 0

    def test_float_nan_filter(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a FLOAT)")
        # Two separate inserts to create two Parquet files
        cat.execute("INSERT INTO ducklake.test VALUES (1.0), (2.0), (3.0)")
        cat.execute("INSERT INTO ducklake.test VALUES ('NaN'::FLOAT), (-1.0)")
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("a") > 0)
            .collect()
        )
        result = result.sort("a")
        # NaN should NOT match > 0
        assert result["a"].to_list() == [1.0, 2.0, 3.0]

    def test_float_infinity_filter(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a FLOAT)")
        # Two separate inserts to create two Parquet files
        cat.execute("INSERT INTO ducklake.test VALUES (1.0), (50.0)")
        cat.execute("INSERT INTO ducklake.test VALUES ('inf'::FLOAT), (99.0)")
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("a") < 100)
            .collect()
        )
        result = result.sort("a")
        # inf should be excluded by < 100
        assert result["a"].to_list() == [1.0, 50.0, 99.0]


class TestTimestampEdgeCases:
    """Test timestamp edge cases: infinity values."""

    def test_timestamp_infinity(self, ducklake_catalog):
        import datetime

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('2024-01-15 10:30:00'), "
            "('infinity'::TIMESTAMP), "
            "('-infinity'::TIMESTAMP)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 1)
        # Polars represents infinity timestamps as extreme microsecond values
        # which can panic when converting to Python datetimes via to_list().
        # Instead, verify using the underlying physical (integer) representation.
        phys = result["a"].to_physical()
        phys_list = sorted(phys.to_list())
        assert len(phys_list) == 3
        # The three values should be: -infinity (min), normal, +infinity (max)
        # The middle value is the normal timestamp; the extremes are the infinities.
        # Verify the extremes are far apart from the middle value (at least 100 years).
        hundred_years_us = 100 * 365 * 24 * 3600 * 1_000_000
        assert phys_list[1] - phys_list[0] > hundred_years_us
        assert phys_list[2] - phys_list[1] > hundred_years_us


class TestNullsInComplexTypes:
    """Test null handling within complex/nested types."""

    def test_list_with_null_elements(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER[])")
        cat.execute("INSERT INTO ducklake.test VALUES ([1, NULL, 3]), (NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        values = result["a"].to_list()
        # First row: list with a null element
        assert values[0] == [1, None, 3]
        # Second row: entire list is null
        assert values[1] is None

    def test_struct_with_null_fields(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(i INTEGER, j INTEGER))")
        cat.execute(
            "INSERT INTO ducklake.test VALUES ({'i': NULL, 'j': 3}), (NULL)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        values = result["a"].to_list()
        # First row: struct with a null field
        assert values[0] == {"i": None, "j": 3}
        # Second row: entire struct is null
        # Polars represents a fully-null struct as None, not {"i": None, "j": None}
        assert values[1] is None

    def test_list_with_null_and_filter(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, a INTEGER[])")
        # Two separate inserts to create two Parquet files
        cat.execute("INSERT INTO ducklake.test VALUES (1, [1, NULL, 3]), (2, NULL)")
        cat.execute("INSERT INTO ducklake.test VALUES (3, [4, 5]), (4, [NULL])")
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("id") >= 2)
            .collect()
        )
        result = result.sort("id")
        assert result.shape == (3, 2)
        assert result["id"].to_list() == [2, 3, 4]
        list_values = result["a"].to_list()
        assert list_values[0] is None
        assert list_values[1] == [4, 5]
        assert list_values[2] == [None]


class TestStringEdgeCases:
    """Test string edge cases: null bytes, empty strings."""

    def test_varchar_null_byte(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        # Insert a string containing a null byte
        cat.execute("INSERT INTO ducklake.test VALUES (chr(0) || 'abc')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)
        value = result["a"][0]
        assert chr(0) in value
        assert "abc" in value

    def test_varchar_empty_and_null(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (''), (NULL), ('hello')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a", nulls_last=True)
        assert result.shape == (3, 1)
        values = result["a"].to_list()
        # Empty string, normal string, and null are all distinct
        assert "" in values
        assert "hello" in values
        assert None in values
        # Verify they are truly distinct
        assert values.count("") == 1
        assert values.count(None) == 1
        assert values.count("hello") == 1


class TestCategoricalEnumTypes:
    """Test Categorical / Enum type handling (mapped to VARCHAR in DuckLake)."""

    def test_read_varchar_as_string(self, ducklake_catalog):
        """VARCHAR columns written by DuckDB are read back as String by Polars."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, color VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'red'), (2, 'green'), (3, 'blue')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"id": pl.Int32, "color": pl.String}
        assert result.sort("id")["color"].to_list() == ["red", "green", "blue"]

    def test_write_categorical_read_string(self, make_write_catalog):
        """Polars Categorical columns are stored as VARCHAR and read back as String."""
        from ducklake_polars import write_ducklake

        cat = make_write_catalog()
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "color": pl.Series(["red", "green", "blue"]).cast(pl.Categorical),
            }
        )

        write_ducklake(df, cat.metadata_path, "test")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 2)
        assert result.schema["color"] == pl.String
        result = result.sort("id")
        assert result["color"].to_list() == ["red", "green", "blue"]

    def test_write_enum_read_string(self, make_write_catalog):
        """Polars Enum columns are stored as VARCHAR and read back as String."""
        from ducklake_polars import write_ducklake

        cat = make_write_catalog()
        color_type = pl.Enum(["red", "green", "blue"])
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "color": pl.Series(["red", "green", "blue"]).cast(color_type),
            }
        )

        write_ducklake(df, cat.metadata_path, "test")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 2)
        assert result.schema["color"] == pl.String
        result = result.sort("id")
        assert result["color"].to_list() == ["red", "green", "blue"]

    def test_categorical_with_nulls(self, make_write_catalog):
        """Categorical columns with null values are handled correctly."""
        from ducklake_polars import write_ducklake

        cat = make_write_catalog()
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "color": pl.Series(["red", None, "blue"]).cast(pl.Categorical),
            }
        )

        write_ducklake(df, cat.metadata_path, "test")

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result["color"].to_list() == ["red", None, "blue"]

    def test_categorical_filter(self, make_write_catalog):
        """Categorical columns can be filtered after reading as String."""
        from ducklake_polars import write_ducklake

        cat = make_write_catalog()
        df = pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "color": pl.Series(["red", "green", "blue", "red"]).cast(
                    pl.Categorical
                ),
            }
        )

        write_ducklake(df, cat.metadata_path, "test")

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("color") == "red")
            .collect()
        )
        assert result.shape == (2, 2)
        assert result.sort("id")["id"].to_list() == [1, 4]


class TestVariantType:
    """Test VARIANT type handling."""

    @pytest.mark.xfail(reason="VARIANT type not supported by Polars")
    def test_variant_type(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARIANT)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), ('hello'), (3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 1)


class TestUnionType:
    """Test UNION type support — DuckDB writes as struct in Parquet."""

    @pytest.mark.xfail(reason="DuckLake does not support UNION type yet (upstream)")
    def test_union_basic(self, ducklake_catalog):
        """Basic UNION type with string and integer members."""
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                id INTEGER,
                u UNION(name VARCHAR, num INTEGER)
            )
        """)
        cat.execute("""
            INSERT INTO ducklake.test VALUES
                (1, 'hello'::UNION(name VARCHAR, num INTEGER)),
                (2, 42::UNION(name VARCHAR, num INTEGER))
        """)
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert len(result) == 2
        assert "u" in result.columns

        # UNION is stored as a struct in Parquet
        # Polars reads it as a Struct type
        u_col = result["u"]
        assert u_col.dtype == pl.Struct

    @pytest.mark.xfail(reason="DuckLake does not support UNION type yet (upstream)")
    def test_union_with_nulls(self, ducklake_catalog):
        """UNION with NULL values."""
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                id INTEGER,
                u UNION(name VARCHAR, num INTEGER)
            )
        """)
        cat.execute("""
            INSERT INTO ducklake.test VALUES
                (1, 'hello'::UNION(name VARCHAR, num INTEGER)),
                (2, NULL),
                (3, 42::UNION(name VARCHAR, num INTEGER))
        """)
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert len(result) == 3

    @pytest.mark.xfail(reason="DuckLake does not support UNION type yet (upstream)")
    def test_union_three_members(self, ducklake_catalog):
        """UNION with three different type members."""
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                id INTEGER,
                u UNION(s VARCHAR, i INTEGER, f DOUBLE)
            )
        """)
        cat.execute("""
            INSERT INTO ducklake.test VALUES
                (1, 'text'::UNION(s VARCHAR, i INTEGER, f DOUBLE)),
                (2, 42::UNION(s VARCHAR, i INTEGER, f DOUBLE)),
                (3, 3.14::UNION(s VARCHAR, i INTEGER, f DOUBLE))
        """)
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert len(result) == 3
        u_col = result["u"]
        assert u_col.dtype == pl.Struct

    @pytest.mark.xfail(reason="DuckLake does not support UNION type yet (upstream)")
    def test_union_read_members(self, ducklake_catalog):
        """Can read individual UNION members from the struct."""
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                id INTEGER,
                u UNION(name VARCHAR, num INTEGER)
            )
        """)
        cat.execute("""
            INSERT INTO ducklake.test VALUES
                (1, 'hello'::UNION(name VARCHAR, num INTEGER)),
                (2, 42::UNION(name VARCHAR, num INTEGER))
        """)
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        # Access struct fields
        u_struct = result["u"]
        names = u_struct.struct.field("name")
        nums = u_struct.struct.field("num")

        # Row 1: name='hello', num=NULL
        assert names[0] == "hello"
        assert nums[0] is None

        # Row 2: name=NULL, num=42
        assert names[1] is None
        assert nums[1] == 42

    @pytest.mark.xfail(reason="DuckLake does not support UNION type yet (upstream)")
    def test_union_time_travel(self, ducklake_catalog):
        """UNION type works with time travel."""
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                id INTEGER,
                u UNION(name VARCHAR, num INTEGER)
            )
        """)
        cat.execute("""
            INSERT INTO ducklake.test VALUES
                (1, 'first'::UNION(name VARCHAR, num INTEGER))
        """)
        cat.execute("""
            INSERT INTO ducklake.test VALUES
                (2, 100::UNION(name VARCHAR, num INTEGER))
        """)
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert len(result) == 2
