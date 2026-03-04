"""Comprehensive type coverage tests for ducklake-polars.

Tests round-trip (DuckDB write → ducklake_polars read) for every DuckDB type,
including NULL handling.  Uses the SQLite backend for speed and zero setup.
"""

from __future__ import annotations

import datetime
import json
import math
from decimal import Decimal

import polars as pl
import pytest

from ducklake_polars import read_ducklake


# ---------------------------------------------------------------------------
# Signed integer types
# ---------------------------------------------------------------------------


class TestSignedIntegers:
    """TINYINT, SMALLINT, INTEGER, BIGINT, HUGEINT."""

    def test_tinyint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val TINYINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (-128), (0), (127), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.Int8}
        assert df["val"].to_list() == [-128, 0, 127, None]

    def test_smallint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val SMALLINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (-32768), (0), (32767), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.Int16}
        assert df["val"].to_list() == [-32768, 0, 32767, None]

    def test_integer(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (-2147483648), (0), (2147483647), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.Int32}
        assert df["val"].to_list() == [-2147483648, 0, 2147483647, None]

    def test_bigint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val BIGINT)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(-9223372036854775808), (0), (9223372036854775807), (NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.Int64}
        assert df["val"].to_list() == [-9223372036854775808, 0, 9223372036854775807, None]

    @pytest.mark.xfail(reason="DuckDB writes HUGEINT as Float64 in Parquet; Polars reads Float64")
    def test_hugeint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val HUGEINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (123456789012345678901234), (0), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema["val"].base_type() == pl.Int128
        assert df["val"].to_list()[1] == 0
        assert df["val"].to_list()[2] is None


# ---------------------------------------------------------------------------
# Unsigned integer types
# ---------------------------------------------------------------------------


class TestUnsignedIntegers:
    """UTINYINT, USMALLINT, UINTEGER, UBIGINT."""

    def test_utinyint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val UTINYINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (0), (255), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.UInt8}
        assert df["val"].to_list() == [0, 255, None]

    def test_usmallint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val USMALLINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (0), (65535), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.UInt16}
        assert df["val"].to_list() == [0, 65535, None]

    def test_uinteger(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val UINTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (0), (4294967295), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.UInt32}
        assert df["val"].to_list() == [0, 4294967295, None]

    def test_ubigint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val UBIGINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (0), (18446744073709551615), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.UInt64}
        assert df["val"].to_list() == [0, 18446744073709551615, None]


# ---------------------------------------------------------------------------
# Floating-point types
# ---------------------------------------------------------------------------


class TestFloatingPoint:
    """FLOAT, DOUBLE, DECIMAL."""

    def test_float(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val FLOAT)")
        cat.execute("INSERT INTO ducklake.test VALUES (3.14), (-1.5), (0.0), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.Float32}
        vals = df["val"].to_list()
        assert abs(vals[0] - 3.14) < 0.01
        assert vals[1] == -1.5
        assert vals[2] == 0.0
        assert vals[3] is None

    def test_double(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (2.718281828459045), (-1e100), (0.0), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.Float64}
        vals = df["val"].to_list()
        assert abs(vals[0] - 2.718281828459045) < 1e-12
        assert vals[1] == -1e100
        assert vals[2] == 0.0
        assert vals[3] is None

    def test_decimal_18_6(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val DECIMAL(18, 6))")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(123456.789012), (-999999999999.999999), (0.000000), (NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        vals = df["val"].to_list()
        assert vals[0] == Decimal("123456.789012")
        assert vals[1] == Decimal("-999999999999.999999")
        assert vals[2] == Decimal("0.000000")
        assert vals[3] is None


# ---------------------------------------------------------------------------
# String / Binary types
# ---------------------------------------------------------------------------


class TestStringBinary:
    """VARCHAR, BLOB."""

    def test_varchar(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES ('hello'), (''), ('🦆'), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.String}
        assert df["val"].to_list() == ["hello", "", "🦆", None]

    def test_blob(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val BLOB)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('\\x00\\x01\\x02'::BLOB), ('\\xFF'::BLOB), (NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.Binary}
        vals = df["val"].to_list()
        assert vals[0] == b"\x00\x01\x02"
        assert vals[1] == b"\xff"
        assert vals[2] is None


# ---------------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------------


class TestBoolean:
    def test_boolean(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val BOOLEAN)")
        cat.execute("INSERT INTO ducklake.test VALUES (true), (false), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.Boolean}
        assert df["val"].to_list() == [True, False, None]


# ---------------------------------------------------------------------------
# Temporal types
# ---------------------------------------------------------------------------


class TestTemporal:
    """DATE, TIME, TIMESTAMP, TIMESTAMP WITH TIME ZONE, INTERVAL."""

    def test_date(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val DATE)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('2024-01-15'), ('1970-01-01'), ('2099-12-31'), (NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.Date}
        assert df["val"].to_list() == [
            datetime.date(2024, 1, 15),
            datetime.date(1970, 1, 1),
            datetime.date(2099, 12, 31),
            None,
        ]

    def test_time(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val TIME)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('00:00:00'), ('12:30:45'), ('23:59:59.999999'), (NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.Time}
        vals = df["val"].to_list()
        assert vals[0] == datetime.time(0, 0, 0)
        assert vals[1] == datetime.time(12, 30, 45)
        assert vals[2] == datetime.time(23, 59, 59, 999999)
        assert vals[3] is None

    def test_timestamp(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val TIMESTAMP)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('2024-01-15 10:30:00'), "
            "('1970-01-01 00:00:00'), "
            "(NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema["val"].base_type() == pl.Datetime
        vals = df["val"].to_list()
        assert vals[0] == datetime.datetime(2024, 1, 15, 10, 30, 0)
        assert vals[1] == datetime.datetime(1970, 1, 1, 0, 0, 0)
        assert vals[2] is None

    def test_timestamptz(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val TIMESTAMP WITH TIME ZONE)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('2024-01-15 10:30:00+00'), "
            "('2024-06-15 15:00:00+00'), "
            "(NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema["val"].base_type() == pl.Datetime
        vals = df["val"].to_list()
        assert vals[0] is not None
        assert vals[1] is not None
        assert vals[2] is None

    @pytest.mark.xfail(reason="Polars cannot read Parquet month_day_millisecond_interval type natively")
    def test_interval(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val INTERVAL)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(INTERVAL 1 HOUR), "
            "(INTERVAL 30 MINUTE), "
            "(NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema["val"].base_type() == pl.Duration
        assert df.shape == (3, 1)
        assert df["val"].to_list()[2] is None


# ---------------------------------------------------------------------------
# UUID
# ---------------------------------------------------------------------------


class TestUUID:
    def test_uuid(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val UUID)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('550e8400-e29b-41d4-a716-446655440000'), "
            "('00000000-0000-0000-0000-000000000000'), "
            "(NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        # UUID is stored as 16-byte binary in Parquet
        assert df.schema == {"val": pl.Binary}
        vals = df["val"].to_list()
        assert vals[0] is not None
        assert len(vals[0]) == 16
        assert vals[1] is not None
        assert len(vals[1]) == 16
        assert vals[2] is None


# ---------------------------------------------------------------------------
# JSON (stored as VARCHAR in DuckLake)
# ---------------------------------------------------------------------------


class TestJSON:
    def test_json(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val JSON)")
        cat.execute(
            """INSERT INTO ducklake.test VALUES """
            """('{"key": "value"}'), """
            """('[1, 2, 3]'), """
            """('null'), """
            """(NULL)"""
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        # JSON is stored as binary in Parquet
        assert df.schema == {"val": pl.Binary}
        vals = df["val"].to_list()
        # Non-NULL JSON values can be cast to String and parsed
        assert json.loads(vals[0].decode("utf-8") if isinstance(vals[0], bytes) else vals[0]) == {"key": "value"}
        assert json.loads(vals[1].decode("utf-8") if isinstance(vals[1], bytes) else vals[1]) == [1, 2, 3]
        assert json.loads(vals[2].decode("utf-8") if isinstance(vals[2], bytes) else vals[2]) is None
        # SQL NULL is distinct from JSON null
        assert vals[3] is None


# ---------------------------------------------------------------------------
# List / Array types
# ---------------------------------------------------------------------------


class TestListTypes:
    """LIST(INTEGER), LIST(VARCHAR)."""

    def test_list_integer(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val INTEGER[])")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "([1, 2, 3]), ([]), ([NULL, 42]), (NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema["val"] == pl.List(pl.Int32)
        vals = df["val"].to_list()
        assert vals[0] == [1, 2, 3]
        assert vals[1] == []
        assert vals[2] == [None, 42]
        assert vals[3] is None

    def test_list_varchar(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val VARCHAR[])")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(['hello', 'world']), (['🦆']), ([NULL, 'ok']), (NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema["val"] == pl.List(pl.String)
        vals = df["val"].to_list()
        assert vals[0] == ["hello", "world"]
        assert vals[1] == ["🦆"]
        assert vals[2] == [None, "ok"]
        assert vals[3] is None


# ---------------------------------------------------------------------------
# Struct type
# ---------------------------------------------------------------------------


class TestStructType:
    """STRUCT(a INTEGER, b VARCHAR)."""

    def test_struct(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val STRUCT(a INTEGER, b VARCHAR))")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "({'a': 1, 'b': 'hello'}), "
            "({'a': NULL, 'b': 'world'}), "
            "(NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        vals = df["val"].to_list()
        assert vals[0] == {"a": 1, "b": "hello"}
        assert vals[1] == {"a": None, "b": "world"}
        assert vals[2] is None


# ---------------------------------------------------------------------------
# Map type
# ---------------------------------------------------------------------------


class TestMapType:
    """MAP(VARCHAR, INTEGER)."""

    @pytest.mark.xfail(reason="MAP type reading from DuckDB Parquet is broken in Polars 1.36")
    def test_map(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val MAP(VARCHAR, INTEGER))")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(MAP {'a': 1, 'b': 2}), "
            "(MAP {}), "
            "(NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.shape == (3, 1)
        vals = df["val"].to_list()
        assert vals[2] is None


# ---------------------------------------------------------------------------
# Fixed-size Array type
# ---------------------------------------------------------------------------


class TestFixedSizeArray:
    """ARRAY(INTEGER, 3) — fixed-size arrays.

    DuckDB stores fixed-size arrays in DuckLake. In Parquet they are
    written as repeated groups, which Polars reads back as List.
    """

    @pytest.mark.xfail(reason="DuckLake does not support fixed-size ARRAY types yet")
    def test_fixed_array_integer(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val INTEGER[3])")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "([1, 2, 3]), "
            "([10, 20, 30]), "
            "(NULL)"
        )
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.shape == (3, 1)
        vals = df["val"].to_list()
        assert vals[0] == [1, 2, 3]
        assert vals[1] == [10, 20, 30]
        assert vals[2] is None


# ---------------------------------------------------------------------------
# Combined: all types in one table (smoke test)
# ---------------------------------------------------------------------------


class TestAllTypesTable:
    """Write a table with many column types at once; read it back."""

    def test_multi_type_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                col_tinyint    TINYINT,
                col_smallint   SMALLINT,
                col_integer    INTEGER,
                col_bigint     BIGINT,
                col_utinyint   UTINYINT,
                col_usmallint  USMALLINT,
                col_uinteger   UINTEGER,
                col_ubigint    UBIGINT,
                col_float      FLOAT,
                col_double     DOUBLE,
                col_decimal    DECIMAL(18, 6),
                col_varchar    VARCHAR,
                col_blob       BLOB,
                col_boolean    BOOLEAN,
                col_date       DATE,
                col_time       TIME,
                col_timestamp  TIMESTAMP,
                col_list_int   INTEGER[],
                col_struct     STRUCT(a INTEGER, b VARCHAR)
            )
        """)
        # Row with real values
        cat.execute("""
            INSERT INTO ducklake.test VALUES (
                1, 2, 3, 4,
                5, 6, 7, 8,
                3.14, 2.718,
                123.456000,
                'duck', '\\x42'::BLOB,
                true,
                '2024-06-15', '12:00:00', '2024-06-15 12:00:00',
                [1, 2, 3],
                {'a': 42, 'b': 'quack'}
            )
        """)
        # Row with all NULLs
        cat.execute("""
            INSERT INTO ducklake.test VALUES (
                NULL, NULL, NULL, NULL,
                NULL, NULL, NULL, NULL,
                NULL, NULL,
                NULL,
                NULL, NULL,
                NULL,
                NULL, NULL, NULL,
                NULL,
                NULL
            )
        """)
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.shape == (2, 19)

        # First row: non-null values
        row0 = df.row(0)
        assert row0[0] == 1       # tinyint
        assert row0[1] == 2       # smallint
        assert row0[2] == 3       # integer
        assert row0[3] == 4       # bigint
        assert row0[4] == 5       # utinyint
        assert row0[5] == 6       # usmallint
        assert row0[6] == 7       # uinteger
        assert row0[7] == 8       # ubigint
        assert row0[13] is True   # boolean

        # Second row: all nulls
        row1 = df.row(1)
        for val in row1:
            assert val is None, f"Expected None but got {val!r}"


# ---------------------------------------------------------------------------
# NULL-only columns
# ---------------------------------------------------------------------------


class TestNullOnlyColumns:
    """Verify that columns containing only NULLs read back correctly."""

    def test_null_only_integer(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (NULL), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.Int32}
        assert df["val"].to_list() == [None, None]

    def test_null_only_varchar(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (NULL), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df.schema == {"val": pl.String}
        assert df["val"].to_list() == [None, None]

    def test_null_only_list(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val INTEGER[])")
        cat.execute("INSERT INTO ducklake.test VALUES (NULL), (NULL)")
        cat.close()

        df = read_ducklake(cat.metadata_path, "test")
        assert df["val"].to_list() == [None, None]
