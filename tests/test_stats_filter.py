"""Filter pushdown and statistics tests for ducklake-dataframe."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from ducklake_polars import read_ducklake, scan_ducklake


class TestMultiFileFilterPushdown:
    """Test filter pushdown across multiple Parquet files with non-overlapping ranges."""

    def test_integer_filter_multi_file(self, ducklake_catalog):
        """Three separate INSERTs create 3 files; exact-match filter returns one row."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'a' FROM range(0, 100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'b' FROM range(1000, 1100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'c' FROM range(5000, 5100) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("i") == 1050)
            .collect()
        )
        assert result.shape[0] == 1
        assert result["i"].to_list() == [1050]
        assert result["label"].to_list() == ["b"]

    def test_integer_range_filter(self, ducklake_catalog):
        """Range filter on the middle batch returns exactly 100 rows."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'a' FROM range(0, 100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'b' FROM range(1000, 1100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'c' FROM range(5000, 5100) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter((pl.col("i") >= 1000) & (pl.col("i") < 1100))
            .collect()
        )
        assert result.shape[0] == 100
        assert sorted(result["i"].to_list()) == list(range(1000, 1100))
        assert all(label == "b" for label in result["label"].to_list())

    def test_integer_greater_than(self, ducklake_catalog):
        """Greater-than filter on the last batch returns 99 rows (5001-5099)."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'a' FROM range(0, 100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'b' FROM range(1000, 1100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'c' FROM range(5000, 5100) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("i") > 5000)
            .collect()
        )
        assert result.shape[0] == 99
        assert sorted(result["i"].to_list()) == list(range(5001, 5100))

    def test_integer_less_than(self, ducklake_catalog):
        """Less-than filter on the first batch returns 100 rows (0-99)."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'a' FROM range(0, 100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'b' FROM range(1000, 1100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'c' FROM range(5000, 5100) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("i") < 100)
            .collect()
        )
        assert result.shape[0] == 100
        assert sorted(result["i"].to_list()) == list(range(0, 100))
        assert all(label == "a" for label in result["label"].to_list())


class TestMultiTypeFilter:
    """Test filter pushdown on various data types."""

    def test_date_filter(self, ducklake_catalog):
        """Filter on DATE column with non-overlapping month ranges."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, d DATE)")
        # January 2020: days 1-31
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, '2020-01-01'::DATE + INTERVAL (i) DAY "
            "FROM range(0, 31) t(i)"
        )
        # June 2020: days 1-30
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 100, '2020-06-01'::DATE + INTERVAL (i) DAY "
            "FROM range(0, 30) t(i)"
        )
        # January 2021: days 1-31
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 200, '2021-01-01'::DATE + INTERVAL (i) DAY "
            "FROM range(0, 31) t(i)"
        )
        cat.close()

        target_date = date(2020, 6, 15)
        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("d") == pl.lit(target_date))
            .collect()
        )
        assert result.shape[0] == 1
        assert result["d"].to_list() == [target_date]
        # id = 100 + 14 (June 1 + 14 days = June 15)
        assert result["id"].to_list() == [114]

    def test_varchar_filter(self, ducklake_catalog):
        """Filter on VARCHAR column with non-overlapping string ranges."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, s VARCHAR)")
        # Batch 1: strings starting with 'a'
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, 'a' || chr(CAST(97 + (i % 26) AS INTEGER)) "
            "|| chr(CAST(97 + ((i / 26) % 26) AS INTEGER)) "
            "FROM range(0, 100) t(i)"
        )
        # Batch 2: strings starting with 'm'
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 100, 'm' || chr(CAST(97 + (i % 26) AS INTEGER)) "
            "|| chr(CAST(97 + ((i / 26) % 26) AS INTEGER)) "
            "FROM range(0, 100) t(i)"
        )
        # Batch 3: strings starting with 'z'
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 200, 'z' || chr(CAST(97 + (i % 26) AS INTEGER)) "
            "|| chr(CAST(97 + ((i / 26) % 26) AS INTEGER)) "
            "FROM range(0, 100) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("s") >= "m")
            .collect()
        )
        # Should include batch 2 (m*) and batch 3 (z*), i.e. 200 rows
        assert result.shape[0] == 200
        assert all(s >= "m" for s in result["s"].to_list())
        # Verify that no 'a' prefixed strings are present
        assert all(not s.startswith("a") for s in result["s"].to_list())

    def test_decimal_filter(self, ducklake_catalog):
        """Filter on DECIMAL column with non-overlapping ranges."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, d DECIMAL(10,2))")
        # Batch 1: 0.00 - 99.99
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, CAST(i AS DECIMAL(10,2)) "
            "FROM range(0, 100) t(i)"
        )
        # Batch 2: 1000.00 - 1099.99
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 100, CAST(i + 1000 AS DECIMAL(10,2)) "
            "FROM range(0, 100) t(i)"
        )
        # Batch 3: 5000.00 - 5099.99
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 200, CAST(i + 5000 AS DECIMAL(10,2)) "
            "FROM range(0, 100) t(i)"
        )
        cat.close()

        from decimal import Decimal

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("d") >= 1000)
            .collect()
        )
        # Should include batch 2 and batch 3 = 200 rows
        assert result.shape[0] == 200
        assert result["d"].min() >= Decimal("1000.00")

    def test_boolean_filter(self, ducklake_catalog):
        """Filter on BOOLEAN column: one batch all-true, one batch all-false."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, b BOOLEAN)")
        # Batch 1: all true
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, true FROM range(0, 50) t(i)"
        )
        # Batch 2: all false
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 50, false FROM range(0, 50) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("b") == True)  # noqa: E712
            .collect()
        )
        assert result.shape[0] == 50
        assert all(b is True for b in result["b"].to_list())
        # Verify these are from batch 1 (ids 0-49)
        assert sorted(result["id"].to_list()) == list(range(0, 50))


class TestComplexPredicates:
    """Test complex filter expressions (OR, IN, combined AND/OR, NOT EQUAL)."""

    def test_or_filter(self, ducklake_catalog):
        """OR filter spanning the first and last batches."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'a' FROM range(0, 100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'b' FROM range(1000, 1100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'c' FROM range(5000, 5100) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter((pl.col("i") < 10) | (pl.col("i") > 5090))
            .collect()
        )
        values = sorted(result["i"].to_list())
        expected = list(range(0, 10)) + list(range(5091, 5100))
        assert values == expected
        assert result.shape[0] == len(expected)

    def test_in_filter(self, ducklake_catalog):
        """IS_IN filter picking one value from each batch."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'a' FROM range(0, 100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'b' FROM range(1000, 1100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'c' FROM range(5000, 5100) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("i").is_in([5, 1005, 5005]))
            .collect()
        )
        assert result.shape[0] == 3
        assert sorted(result["i"].to_list()) == [5, 1005, 5005]

    def test_combined_and_or(self, ducklake_catalog):
        """Combined AND filter spanning two batches."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'a' FROM range(0, 100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'b' FROM range(1000, 1100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'c' FROM range(5000, 5100) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter((pl.col("i") > 50) & (pl.col("i") < 1050))
            .collect()
        )
        values = sorted(result["i"].to_list())
        # From batch 1: 51-99, from batch 2: 1000-1049
        expected = list(range(51, 100)) + list(range(1000, 1050))
        assert values == expected
        assert result.shape[0] == len(expected)

    def test_not_equal_filter(self, ducklake_catalog):
        """Not-equal filter on a single small file."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i FROM range(0, 10) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("i") != 5)
            .collect()
        )
        assert result.shape[0] == 9
        assert sorted(result["i"].to_list()) == [0, 1, 2, 3, 4, 6, 7, 8, 9]


class TestNullFiltering:
    """Test filtering on NULL values across multiple files."""

    def test_is_null_filter(self, ducklake_catalog):
        """IS NULL filter returns only null rows from the second batch."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, b INTEGER)")
        # Batch 1: no nulls
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, i * 10 FROM range(0, 50) t(i)"
        )
        # Batch 2: some nulls in column b
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 50, CASE WHEN i % 2 = 0 THEN NULL ELSE i END "
            "FROM range(0, 50) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("b").is_null())
            .collect()
        )
        # 25 even values in range(0, 50) produce NULLs
        assert result.shape[0] == 25
        assert all(b is None for b in result["b"].to_list())
        # Verify these are from batch 2 (ids 50, 52, 54, ...)
        assert sorted(result["id"].to_list()) == list(range(50, 100, 2))

    def test_is_not_null_filter(self, ducklake_catalog):
        """IS NOT NULL filter returns all non-null rows."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, b INTEGER)")
        # Batch 1: no nulls
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, i * 10 FROM range(0, 50) t(i)"
        )
        # Batch 2: some nulls in column b
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 50, CASE WHEN i % 2 = 0 THEN NULL ELSE i END "
            "FROM range(0, 50) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("b").is_not_null())
            .collect()
        )
        # 50 from batch 1 + 25 odd values from batch 2
        assert result.shape[0] == 75
        assert all(b is not None for b in result["b"].to_list())

    def test_null_or_value_filter(self, ducklake_catalog):
        """Combined NULL or value filter."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, b INTEGER)")
        # Batch 1: no nulls, values 0-490 (step 10)
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, i * 10 FROM range(0, 50) t(i)"
        )
        # Batch 2: some nulls in column b
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 50, CASE WHEN i % 2 = 0 THEN NULL ELSE i END "
            "FROM range(0, 50) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("b").is_null() | (pl.col("b") > 50))
            .collect()
        )
        # From batch 1: b values > 50 means i*10 > 50, so i >= 6 (i.e. 44 rows: 6..49)
        # From batch 2: 25 nulls (even i) + odd i > 50 means i in {51,53,...,99} which
        #   maps to b values: odd i values where i > 50, but b = i for odd i in range(0,50)
        #   so b in {1,3,5,...,49} -- none > 50 from batch 2's non-null values
        # Total: 44 from batch 1 (b > 50) + 25 nulls from batch 2 = 69
        values = result["b"].to_list()
        null_count = sum(1 for v in values if v is None)
        non_null_values = [v for v in values if v is not None]
        assert null_count == 25
        assert all(v > 50 for v in non_null_values)
        assert result.shape[0] == 25 + 44


class TestTopNPruning:
    """Test sort + limit (top-N) queries across multiple files."""

    def test_sort_limit(self, ducklake_catalog):
        """Descending sort with head(5) returns the 5 largest values."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i FROM range(0, 100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i FROM range(1000, 1100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i FROM range(5000, 5100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i FROM range(9000, 9100) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .sort("i", descending=True)
            .head(5)
            .collect()
        )
        assert result.shape[0] == 5
        assert result["i"].to_list() == [9099, 9098, 9097, 9096, 9095]

    def test_sort_limit_ascending(self, ducklake_catalog):
        """Ascending sort with head(5) returns the 5 smallest values."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i FROM range(0, 100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i FROM range(1000, 1100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i FROM range(5000, 5100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i FROM range(9000, 9100) t(i)"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .sort("i")
            .head(5)
            .collect()
        )
        assert result.shape[0] == 5
        assert result["i"].to_list() == [0, 1, 2, 3, 4]
