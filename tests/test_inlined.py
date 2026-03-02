"""Inlined data tests for ducklake-dataframe."""

from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import read_ducklake, scan_ducklake


class TestInlinedDataTypes:
    """Test reading inlined data with various column types."""

    def test_inlined_integer_types(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute(
            "CREATE TABLE ducklake.test "
            "(ti TINYINT, si SMALLINT, i INTEGER, bi BIGINT)"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 100, 10000, 1000000000), "
            "(2, 200, 20000, 2000000000)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("ti")
        assert result.shape == (2, 4)
        assert result["ti"].to_list() == [1, 2]
        assert result["si"].to_list() == [100, 200]
        assert result["i"].to_list() == [10000, 20000]
        assert result["bi"].to_list() == [1000000000, 2000000000]

    def test_inlined_float_types(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (f FLOAT, d DOUBLE)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1.5, 3.14159265358979), "
            "(2.5, 2.71828182845905)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("f")
        assert result.shape == (2, 2)
        assert result["f"][0] == pytest.approx(1.5, abs=1e-5)
        assert result["f"][1] == pytest.approx(2.5, abs=1e-5)
        assert result["d"][0] == pytest.approx(3.14159265358979, abs=1e-10)
        assert result["d"][1] == pytest.approx(2.71828182845905, abs=1e-10)

    def test_inlined_string_types(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, s VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'hello world'), "
            "(2, ''), "
            "(3, 'special chars: !@#$%')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (3, 2)
        assert result["s"].to_list() == [
            "hello world",
            "",
            "special chars: !@#$%",
        ]

    def test_inlined_date_timestamp(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, d DATE, ts TIMESTAMP)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, '2024-01-15', '2024-01-15 10:30:00'), "
            "(2, '2024-06-30', '2024-06-30 23:59:59')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (2, 3)
        # Verify date values
        from datetime import date, datetime

        assert result["d"].to_list() == [date(2024, 1, 15), date(2024, 6, 30)]
        # Verify timestamp values
        assert result["ts"].to_list() == [
            datetime(2024, 1, 15, 10, 30, 0),
            datetime(2024, 6, 30, 23, 59, 59),
        ]

    def test_inlined_boolean(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b BOOLEAN)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, true), (2, false), (3, NULL)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (3, 2)
        assert result["b"].to_list() == [True, False, None]

    def test_inlined_decimal(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, d DECIMAL(10,2))")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 123.45), (2, 678.90), (3, 0.01)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (3, 2)
        # Decimal values: check they are numerically correct
        d_vals = result["d"].cast(pl.Float64).to_list()
        assert d_vals[0] == pytest.approx(123.45)
        assert d_vals[1] == pytest.approx(678.90)
        assert d_vals[2] == pytest.approx(0.01)


class TestInlinedFilter:
    """Test filtering on inlined data via scan_ducklake."""

    def test_filter_on_inlined_data(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four'), (5, 'five')"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("a") > 3)
            .collect()
        )
        result = result.sort("a")
        assert result.shape[0] == 2
        assert result["a"].to_list() == [4, 5]
        assert result["b"].to_list() == ["four", "five"]

    def test_filter_string_on_inlined(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, name VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'alice'), (2, 'bob'), (3, 'alice'), (4, 'charlie')"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("name") == "alice")
            .collect()
        )
        result = result.sort("id")
        assert result.shape[0] == 2
        assert result["id"].to_list() == [1, 3]
        assert result["name"].to_list() == ["alice", "alice"]


class TestInlinedWithDelete:
    """Test reading inlined data after DELETE operations."""

    def test_read_after_inlined_delete(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four'), (5, 'five')"
        )
        cat.execute("DELETE FROM ducklake.test WHERE a = 3")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape[0] == 4
        assert result["a"].to_list() == [1, 2, 4, 5]
        assert result["b"].to_list() == ["one", "two", "four", "five"]

    def test_read_after_inlined_delete_all(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'one'), (2, 'two'), (3, 'three')"
        )
        cat.execute("DELETE FROM ducklake.test")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0
        # Schema should be preserved
        assert "a" in result.columns
        assert "b" in result.columns


class TestInlinedWithUpdate:
    """Test reading inlined data after UPDATE operations."""

    def test_read_after_inlined_update(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'original'), (2, 'unchanged'), (3, 'unchanged')"
        )
        cat.execute("UPDATE ducklake.test SET b = 'updated' WHERE a = 1")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape[0] == 3
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["updated", "unchanged", "unchanged"]


class TestInlinedMixedWithParquet:
    """Test reading when data comes from both inlined storage and Parquet files."""

    def test_inlined_plus_parquet(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        # Small insert -- should be inlined
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'inlined'), (2, 'inlined')"
        )
        # Large insert -- should exceed inlining limit and go to Parquet
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, 'val' || i::VARCHAR FROM range(0, 5000) t(i)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        # We should have the 2 inlined rows plus 5000 Parquet rows
        assert result.shape[0] == 5002
        # Verify inlined rows are present
        inlined_rows = result.filter(pl.col("b") == "inlined").sort("a")
        assert inlined_rows["a"].to_list() == [1, 2]
        # Verify Parquet rows are present
        parquet_rows = result.filter(pl.col("b").str.starts_with("val"))
        assert parquet_rows.shape[0] == 5000

    def test_inlined_plus_flush(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'first'), (2, 'second'), (3, 'third')"
        )
        # Flush inlined data to Parquet files
        cat.execute("CALL ducklake_flush_inlined_data('ducklake')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape[0] == 3
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["first", "second", "third"]


class TestInlinedSchemaEvolution:
    """Test schema evolution operations with inlined data."""

    def test_inlined_add_column(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'first'), (2, 'second')"
        )

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN c INTEGER")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'third', 42)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (3, 3)
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["first", "second", "third"]
        # Old rows should have c=NULL, new row should have c=42
        assert result["c"].to_list() == [None, None, 42]

    def test_inlined_drop_column(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c INTEGER)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'first', 10), (2, 'second', 20)"
        )

        cat.execute("ALTER TABLE ducklake.test DROP COLUMN b")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 30)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.columns == ["a", "c"]
        assert result.shape == (3, 2)
        assert result["a"].to_list() == [1, 2, 3]
        assert result["c"].to_list() == [10, 20, 30]


class TestInlinedTimeTravel:
    """Test time travel with inlined data."""

    def test_inlined_time_travel(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'first'), (2, 'second')"
        )
        # Capture snapshot after first insert
        v1 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        cat.execute(
            "INSERT INTO ducklake.test VALUES (3, 'third'), (4, 'fourth')"
        )
        cat.close()

        # Read at old snapshot: should have 2 rows
        result_v1 = read_ducklake(
            cat.metadata_path, "test", snapshot_version=v1
        )
        result_v1 = result_v1.sort("a")
        assert result_v1.shape[0] == 2
        assert result_v1["a"].to_list() == [1, 2]
        assert result_v1["b"].to_list() == ["first", "second"]

        # Read latest: should have 4 rows
        result_latest = read_ducklake(cat.metadata_path, "test")
        result_latest = result_latest.sort("a")
        assert result_latest.shape[0] == 4
        assert result_latest["a"].to_list() == [1, 2, 3, 4]
        assert result_latest["b"].to_list() == [
            "first",
            "second",
            "third",
            "fourth",
        ]
