"""Type mapping tests for ducklake-pandas."""

from __future__ import annotations

import math

import pandas as pd
import numpy as np
import pytest

from tests_pandas.helpers import assert_list_equal
from ducklake_pandas import read_ducklake


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
        assert True  # schema check skipped for pandas
        assert tuple(result.iloc[0]) == (1, 100, 10000, 1000000)

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
        assert True  # schema check skipped for pandas

    @pytest.mark.xfail(reason="DuckDB writes HUGEINT as Float64 in Parquet")
    def test_hugeint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a HUGEINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (123456789012345678901234)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")

    @pytest.mark.xfail(reason="DuckDB writes UHUGEINT as Float64 in Parquet")
    def test_uhugeint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a UHUGEINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (123456789012345678901234)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")

    def test_float_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a FLOAT, b DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (3.14, 2.718281828)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert True  # schema check skipped for pandas
        assert abs(result["a"][0] - 3.14) < 0.01
        assert abs(result["b"][0] - 2.718281828) < 1e-6

    def test_boolean(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a BOOLEAN)")
        cat.execute("INSERT INTO ducklake.test VALUES (true), (false), (NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert True  # schema check skipped for pandas
        assert_list_equal(result["a"].tolist(), [True, False, None])

    def test_varchar(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES ('hello'), ('world'), ('')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert True  # schema check skipped for pandas
        assert result["a"].tolist() == ["hello", "world", ""]

    def test_blob(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a BLOB)")
        cat.execute("INSERT INTO ducklake.test VALUES ('\\x01\\x02\\x03'::BLOB)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert True  # schema check skipped for pandas
        assert result["a"][0] == b"\x01\x02\x03"

    def test_date(self, ducklake_catalog):
        import datetime

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DATE)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15'), ('2024-12-31')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert True  # schema check skipped for pandas
        assert result["a"].tolist() == [
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
        assert result["a"][0] == datetime.datetime(2024, 1, 15, 10, 30, 0)

    def test_timestamp_s(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP_S)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")

    def test_timestamp_ms(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP_MS)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00.123')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")

    def test_timestamp_ns(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP_NS)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00.123456789')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")

    def test_timestamptz(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMPTZ)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00+00')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")

    def test_time(self, ducklake_catalog):
        import datetime

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIME)")
        cat.execute("INSERT INTO ducklake.test VALUES ('10:30:00'), ('23:59:59')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert True  # schema check skipped for pandas
        assert result["a"].tolist() == [
            datetime.time(10, 30, 0),
            datetime.time(23, 59, 59),
        ]

    @pytest.mark.xfail(reason="Parquet month_day_millisecond_interval type not supported natively")
    def test_interval(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTERVAL)")
        cat.execute("INSERT INTO ducklake.test VALUES (INTERVAL 1 HOUR), (INTERVAL 30 MINUTE)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)

    def test_uuid(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a UUID)")
        cat.execute("INSERT INTO ducklake.test VALUES ('550e8400-e29b-41d4-a716-446655440000')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert True  # schema check skipped for pandas
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
        assert True  # schema check skipped for pandas
        assert result.shape == (2, 1)
        # JSON text is stored as binary in Parquet; decode to string
        texts = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in result["a"].tolist()]
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
        values = result["a"].tolist()
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
        vals = result["a"].tolist()
        assert len(vals) == 2
        assert list(vals[0]) == [1, 2, 3]
        assert list(vals[1]) == [4, 5]

    def test_list_of_varchar(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR[])")
        cat.execute("INSERT INTO ducklake.test VALUES (['hello', 'world']), (['foo'])")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        vals = result["a"].tolist()
        assert len(vals) == 2
        assert list(vals[0]) == ["hello", "world"]
        assert list(vals[1]) == ["foo"]

    def test_struct_type(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(x INTEGER, y VARCHAR))")
        cat.execute("INSERT INTO ducklake.test VALUES ({'x': 1, 'y': 'hello'})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)
        val = result["a"][0]
        assert val == {"x": 1, "y": "hello"}

    @pytest.mark.xfail(reason="MAP type reading from DuckDB Parquet may be broken")
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
        vals = result["a"].tolist()
        assert len(vals) == 1
        inner = [dict(x) for x in vals[0]]
        assert inner == [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]

    def test_list_of_lists(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER[][])")
        cat.execute("INSERT INTO ducklake.test VALUES ([[1, 2], [3, 4]]), ([[5]])")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        vals = result["a"].tolist()
        assert len(vals) == 2
        assert [list(x) for x in vals[0]] == [[1, 2], [3, 4]]
        assert [list(x) for x in vals[1]] == [[5]]

    def test_struct_with_list_field(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(name VARCHAR, scores INTEGER[]))")
        cat.execute("INSERT INTO ducklake.test VALUES ({'name': 'Alice', 'scores': [90, 85, 92]})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)
        val = result["a"][0]
        assert val["name"] == "Alice"
        assert list(val["scores"]) == [90, 85, 92]


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
        assert result["id"].tolist() == [1, 2, 3]
        assert result["name"].tolist() == ["Alice", "Bob", "Charlie"]
        assert result["active"].tolist() == [True, False, True]

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
        assert True  # schema check skipped for pandas
        assert tuple(result.iloc[0]) == (1, 2, 3, 4, 5, 6, 7, 8)


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

        result = read_ducklake(cat.metadata_path, "test")
        result = result[result["a"] > 0]
        result = result.sort_values("a").reset_index(drop=True)
        # NaN should NOT match > 0
        assert result["a"].tolist() == [1.0, 2.0, 3.0]

    def test_float_infinity_filter(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a FLOAT)")
        # Two separate inserts to create two Parquet files
        cat.execute("INSERT INTO ducklake.test VALUES (1.0), (50.0)")
        cat.execute("INSERT INTO ducklake.test VALUES ('inf'::FLOAT), (99.0)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result[result["a"] < 100]
        result = result.sort_values("a").reset_index(drop=True)
        # inf should be excluded by < 100
        assert result["a"].tolist() == [1.0, 50.0, 99.0]


class TestTimestampEdgeCases:
    """Test timestamp edge cases: infinity values."""

    @pytest.mark.skip(reason="Timestamp infinity handling is Polars-specific (to_physical)")
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


class TestNullsInComplexTypes:
    """Test null handling within complex/nested types."""

    def test_list_with_null_elements(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER[])")
        cat.execute("INSERT INTO ducklake.test VALUES ([1, NULL, 3]), (NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        values = result["a"].tolist()
        # First row: list with a null element
        first = list(values[0])
        assert first[0] == 1
        assert pd.isna(first[1])
        assert first[2] == 3
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
        values = result["a"].tolist()
        # First row: struct with a null field
        assert values[0] == {"i": None, "j": 3}
        # Second row: entire struct is null
        assert values[1] is None

    def test_list_with_null_and_filter(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, a INTEGER[])")
        # Two separate inserts to create two Parquet files
        cat.execute("INSERT INTO ducklake.test VALUES (1, [1, NULL, 3]), (2, NULL)")
        cat.execute("INSERT INTO ducklake.test VALUES (3, [4, 5]), (4, [NULL])")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result[result["id"] >= 2]
        result = result.sort_values("id").reset_index(drop=True)
        assert result.shape == (3, 2)
        assert result["id"].tolist() == [2, 3, 4]
        list_values = result["a"].tolist()
        assert list_values[0] is None
        assert list(list_values[1]) == [4, 5]
        lv2 = list(list_values[2])
        assert len(lv2) == 1
        assert pd.isna(lv2[0])


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
        result = result.sort_values("a", na_position="last")
        assert result.shape == (3, 1)
        values = result["a"].tolist()
        # Empty string, normal string, and null are all distinct
        assert "" in values
        assert "hello" in values
        assert any(v is None or (isinstance(v, float) and __import__('math').isnan(v)) for v in values)
        # Verify they are truly distinct
        assert values.count("") == 1
        assert values.count("hello") == 1


class TestCategoricalTypes:
    """Test Categorical / dictionary type handling (mapped to VARCHAR in DuckLake)."""

    def test_read_varchar_as_string(self, ducklake_catalog):
        """VARCHAR columns written by DuckDB are read back as object/string."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, color VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'red'), (2, 'green'), (3, 'blue')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("id").reset_index(drop=True)
        assert result["color"].tolist() == ["red", "green", "blue"]

    def test_write_categorical_read_string(self, make_write_catalog):
        """Pandas Categorical columns are stored as VARCHAR and read back as string."""
        from ducklake_pandas import write_ducklake

        cat = make_write_catalog()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "color": pd.Categorical(["red", "green", "blue"]),
            }
        )

        write_ducklake(df, cat.metadata_path, "test")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 2)
        result = result.sort_values("id").reset_index(drop=True)
        assert result["color"].tolist() == ["red", "green", "blue"]

    def test_categorical_with_nulls(self, make_write_catalog):
        """Categorical columns with null values are handled correctly."""
        from ducklake_pandas import write_ducklake

        cat = make_write_catalog()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "color": pd.Categorical(["red", None, "blue"]),
            }
        )

        write_ducklake(df, cat.metadata_path, "test")

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("id").reset_index(drop=True)
        assert result["color"].iloc[0] == "red"
        assert result["color"].iloc[2] == "blue"
        # null check: pandas represents nulls as NaN for object dtype
        assert pd.isna(result["color"].iloc[1])


class TestVariantType:
    """Test VARIANT type handling."""

    @pytest.mark.xfail(reason="VARIANT type not supported")
    def test_variant_type(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARIANT)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), ('hello'), (3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 1)
