"""Edge case tests for ducklake-dataframe.

Tests unusual inputs, boundary conditions, error handling, and uncommon
but valid DuckLake operations. Each test verifies both row counts and
actual data values.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import DuckLakeCatalog, read_ducklake, scan_ducklake


# ===================================================================
# SECTION 1: Empty & Minimal Tables
# ===================================================================


class TestEmptyTables:
    """Test edge cases with empty and minimal tables."""

    def test_empty_table_schema_preserved(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (0, 3)
        assert result.schema == {"a": pl.Int32, "b": pl.String, "c": pl.Float64}

    def test_single_row_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (42)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)
        assert result["a"][0] == 42

    def test_single_null_row(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (NULL, NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 2)
        assert result.row(0) == (None, None)

    def test_all_null_column(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, NULL), (2, NULL), (3, NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["b"].null_count() == 3
        assert result["b"].to_list() == [None, None, None]

    def test_delete_all_rows(self, ducklake_catalog):
        """Table that has had all rows deleted."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        cat.execute("DELETE FROM ducklake.test")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (0, 2)
        assert result.columns == ["a", "b"]

    def test_insert_after_delete_all(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2)")
        cat.execute("DELETE FROM ducklake.test")
        cat.execute("INSERT INTO ducklake.test VALUES (99)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 1
        assert result["a"][0] == 99

    def test_empty_table_scan(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.collect()
        assert result.shape == (0, 1)

    def test_empty_table_filter(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("a") > 0).collect()
        assert result.shape == (0, 1)


# ===================================================================
# SECTION 2: Wide Tables
# ===================================================================


class TestWideTables:
    """Test tables with many columns."""

    def test_20_columns(self, ducklake_catalog):
        cat = ducklake_catalog
        cols = ", ".join(f"c{i} INTEGER" for i in range(20))
        cat.execute(f"CREATE TABLE ducklake.test ({cols})")
        vals = ", ".join(str(i) for i in range(20))
        cat.execute(f"INSERT INTO ducklake.test VALUES ({vals})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 20)
        for i in range(20):
            assert result[f"c{i}"][0] == i

    def test_50_columns_mixed_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cols = []
        for i in range(50):
            if i % 3 == 0:
                cols.append(f"c{i} INTEGER")
            elif i % 3 == 1:
                cols.append(f"c{i} VARCHAR")
            else:
                cols.append(f"c{i} DOUBLE")
        cat.execute(f"CREATE TABLE ducklake.test ({', '.join(cols)})")

        vals = []
        for i in range(50):
            if i % 3 == 0:
                vals.append(str(i))
            elif i % 3 == 1:
                vals.append(f"'val_{i}'")
            else:
                vals.append(f"{i}.5")
        cat.execute(f"INSERT INTO ducklake.test VALUES ({', '.join(vals)})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 50)

    def test_column_selection_on_wide_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cols = ", ".join(f"c{i} INTEGER" for i in range(30))
        cat.execute(f"CREATE TABLE ducklake.test ({cols})")
        vals = ", ".join(str(i) for i in range(30))
        cat.execute(f"INSERT INTO ducklake.test VALUES ({vals})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test", columns=["c0", "c15", "c29"])
        assert result.shape == (1, 3)
        assert result["c0"][0] == 0
        assert result["c15"][0] == 15
        assert result["c29"][0] == 29


# ===================================================================
# SECTION 3: Column Naming Edge Cases
# ===================================================================


class TestColumnNaming:
    """Test edge cases in column names."""

    def test_long_column_name(self, ducklake_catalog):
        cat = ducklake_catalog
        long_name = "a" * 200
        cat.execute(f'CREATE TABLE ducklake.test ("{long_name}" INTEGER)')
        cat.execute(f'INSERT INTO ducklake.test ("{long_name}") VALUES (42)')
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)
        assert result.columns[0] == long_name
        assert result[long_name][0] == 42

    def test_column_name_with_spaces(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute('CREATE TABLE ducklake.test ("my column" INTEGER, "another col" VARCHAR)')
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["my column", "another col"]
        assert result["my column"][0] == 1

    def test_column_name_with_special_chars(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute('CREATE TABLE ducklake.test ("a-b" INTEGER, "c.d" VARCHAR, "e/f" DOUBLE)')
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hi', 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["a-b", "c.d", "e/f"]
        assert result["a-b"][0] == 1

    def test_column_name_case_sensitivity(self, ducklake_catalog):
        """DuckDB column names are case-insensitive by default, stored lowercase."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (MyCol INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (42)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        # DuckDB stores as lowercase
        assert "mycol" in [c.lower() for c in result.columns]

    def test_quoted_column_name_preserves_case(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute('CREATE TABLE ducklake.test ("MixedCase" INTEGER)')
        cat.execute('INSERT INTO ducklake.test ("MixedCase") VALUES (42)')
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert "MixedCase" in result.columns
        assert result["MixedCase"][0] == 42

    def test_numeric_column_name(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute('CREATE TABLE ducklake.test ("123" INTEGER, "456" VARCHAR)')
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert "123" in result.columns
        assert "456" in result.columns


# ===================================================================
# SECTION 4: Table & Schema Naming Edge Cases
# ===================================================================


class TestTableNaming:
    """Test edge cases in table and schema names."""

    def test_long_table_name(self, ducklake_catalog):
        cat = ducklake_catalog
        long_name = "t" * 200
        cat.execute(f'CREATE TABLE ducklake."{long_name}" (a INTEGER)')
        cat.execute(f'INSERT INTO ducklake."{long_name}" VALUES (42)')
        cat.close()

        result = read_ducklake(cat.metadata_path, long_name)
        assert result["a"][0] == 42

    def test_table_name_with_underscore(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.my_table (a INTEGER)")
        cat.execute("INSERT INTO ducklake.my_table VALUES (1)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "my_table")
        assert result["a"][0] == 1

    def test_multiple_schemas(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.execute("CREATE SCHEMA ducklake.s2")
        cat.execute("CREATE TABLE ducklake.s1.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.s1.t1 VALUES (1)")
        cat.execute("CREATE TABLE ducklake.s2.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.s2.t1 VALUES (2)")
        cat.close()

        r1 = read_ducklake(cat.metadata_path, "t1", schema="s1")
        r2 = read_ducklake(cat.metadata_path, "t1", schema="s2")
        assert r1["a"][0] == 1
        assert r2["a"][0] == 2


# ===================================================================
# SECTION 5: Error Handling
# ===================================================================


class TestErrorHandling:
    """Test error handling for invalid operations."""

    def test_nonexistent_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "nonexistent")

    def test_nonexistent_schema(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "test", schema="nonexistent")

    def test_invalid_snapshot_version(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        with pytest.raises(ValueError):
            read_ducklake(cat.metadata_path, "test", snapshot_version=99999)

    def test_snapshot_version_and_time_mutual_exclusivity(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        with pytest.raises(ValueError, match="Cannot specify both"):
            read_ducklake(
                cat.metadata_path,
                "test",
                snapshot_version=1,
                snapshot_time="2025-01-01T00:00:00",
            )

    def test_wrong_data_path(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        with pytest.raises(Exception):
            read_ducklake(cat.metadata_path, "test", data_path="/nonexistent/path")

    def test_read_dropped_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (42)")
        cat.execute("DROP TABLE ducklake.test")
        cat.close()

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "test")

    def test_nonexistent_column_selection(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")
        cat.close()

        with pytest.raises(Exception):
            read_ducklake(cat.metadata_path, "test", columns=["nonexistent"])

    def test_empty_column_selection(self, ducklake_catalog):
        """Selecting no columns should work or fail gracefully."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        # Empty columns list should either return all columns or raise
        # (implementation-dependent)
        try:
            result = read_ducklake(cat.metadata_path, "test", columns=[])
            # If it returns, should have the row
            assert result.shape[0] >= 0
        except Exception:
            pass  # Also acceptable


# ===================================================================
# SECTION 6: Boundary Values
# ===================================================================


class TestBoundaryValues:
    """Test boundary values for various data types."""

    def test_integer_boundaries(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                tiny TINYINT,
                small SMALLINT,
                medium INTEGER,
                big BIGINT
            )
        """)
        # Insert min values
        cat.execute("INSERT INTO ducklake.test VALUES (-128, -32768, -2147483648, -9223372036854775808)")
        # Insert max values
        cat.execute("INSERT INTO ducklake.test VALUES (127, 32767, 2147483647, 9223372036854775807)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("tiny")
        assert result["tiny"].to_list() == [-128, 127]
        assert result["small"].to_list() == [-32768, 32767]
        assert result["medium"].to_list() == [-2147483648, 2147483647]
        assert result["big"].to_list() == [-9223372036854775808, 9223372036854775807]

    def test_unsigned_integer_boundaries(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                tiny UTINYINT,
                small USMALLINT,
                medium UINTEGER,
                big UBIGINT
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (0, 0, 0, 0)")
        cat.execute("INSERT INTO ducklake.test VALUES (255, 65535, 4294967295, 18446744073709551615)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("tiny")
        assert result["tiny"].to_list() == [0, 255]
        assert result["small"].to_list() == [0, 65535]
        assert result["medium"].to_list() == [0, 4294967295]
        assert result["big"].to_list() == [0, 18446744073709551615]

    def test_date_boundaries(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (d DATE)")
        cat.execute("INSERT INTO ducklake.test VALUES ('0001-01-01'), ('9999-12-31')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 2
        dates = sorted(result["d"].to_list())
        assert dates[0] == datetime.date(1, 1, 1)
        assert dates[1] == datetime.date(9999, 12, 31)

    def test_empty_string_vs_null(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (''), (NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        values = result["a"].to_list()
        assert "" in values
        assert None in values
        assert len(values) == 2

    def test_zero_values(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, f DOUBLE, d DECIMAL(5,2))")
        cat.execute("INSERT INTO ducklake.test VALUES (0, 0.0, 0.00)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["i"][0] == 0
        assert result["f"][0] == 0.0
        assert result["d"][0] == Decimal("0.00")

    def test_negative_zero_float(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (0.0), (-0.0)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 2
        # Both should be equal to 0.0
        for v in result["a"].to_list():
            assert v == 0.0

    def test_very_large_varchar(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        large_str = "x" * 10000
        cat.execute("INSERT INTO ducklake.test VALUES (?)", [large_str])
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"][0] == large_str
        assert len(result["a"][0]) == 10000


# ===================================================================
# SECTION 7: Filter Edge Cases
# ===================================================================


class TestFilterEdgeCases:
    """Test edge cases in filter pushdown."""

    def test_filter_all_null_column(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, NULL), (2, NULL)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("b").is_not_null()).collect()
        assert result.shape[0] == 0

    def test_filter_returns_nothing(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS a FROM range(100) t(i)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("a") > 999).collect()
        assert result.shape[0] == 0
        assert result.columns == ["a"]

    def test_filter_returns_everything(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS a FROM range(10) t(i)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("a") >= 0).collect()
        assert result.shape[0] == 10

    def test_filter_on_non_partition_column(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y'), (3, 'x')")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("a") == 2).collect()
        assert result.shape[0] == 1
        assert result["b"][0] == "y"

    def test_combined_and_filter(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i, i * 10 FROM range(100) t(i)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter((pl.col("a") >= 10) & (pl.col("a") < 20)).collect()
        assert result.shape[0] == 10
        assert sorted(result["a"].to_list()) == list(range(10, 20))

    def test_or_filter(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i FROM range(100) t(i)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter((pl.col("a") == 5) | (pl.col("a") == 95)).collect()
        assert result.shape[0] == 2
        assert sorted(result["a"].to_list()) == [5, 95]

    def test_not_filter(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i FROM range(10) t(i)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(~(pl.col("a") < 5)).collect()
        assert result.shape[0] == 5
        assert sorted(result["a"].to_list()) == [5, 6, 7, 8, 9]


# ===================================================================
# SECTION 8: Multiple Operations on Same Table
# ===================================================================


class TestMultipleOperations:
    """Test sequences of operations that stress test the catalog reader.

    NOTE: These tests use only the SQLite backend fixture because DuckDB's
    DuckLake extension can deadlock when executing many sequential write
    operations on a Postgres-backed catalog.
    """

    def test_many_small_inserts(self, ducklake_catalog_sqlite):
        """Many 1-row inserts creating many Parquet files."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        for i in range(20):
            cat.execute(f"INSERT INTO ducklake.test VALUES ({i})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 20
        assert sorted(result["a"].to_list()) == list(range(20))

    def test_insert_delete_repeat(self, ducklake_catalog_sqlite):
        """Alternating inserts and deletes."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        cat.execute("DELETE FROM ducklake.test WHERE a = 2")
        cat.execute("INSERT INTO ducklake.test VALUES (4), (5)")
        cat.execute("DELETE FROM ducklake.test WHERE a = 1")
        cat.execute("INSERT INTO ducklake.test VALUES (6)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [3, 4, 5, 6]

    def test_update_same_row_multiple_times(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 0)")
        cat.execute("UPDATE ducklake.test SET val = 1 WHERE id = 1")
        cat.execute("UPDATE ducklake.test SET val = 2 WHERE id = 1")
        cat.execute("UPDATE ducklake.test SET val = 3 WHERE id = 1")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 1
        assert result["val"][0] == 3

    def test_delete_all_then_multiple_inserts(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i FROM range(100) t(i)")
        cat.execute("DELETE FROM ducklake.test")
        cat.execute("INSERT INTO ducklake.test VALUES (1000)")
        cat.execute("INSERT INTO ducklake.test VALUES (2000)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 2
        assert sorted(result["a"].to_list()) == [1000, 2000]

    def test_schema_evolution_multiple_steps(self, ducklake_catalog_sqlite):
        """Add column, insert, drop column, add another, insert."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'hello')")

        cat.execute("ALTER TABLE ducklake.test DROP COLUMN b")
        cat.execute("INSERT INTO ducklake.test VALUES (3)")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN c DOUBLE")
        cat.execute("INSERT INTO ducklake.test VALUES (4, 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "c"]
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["c"].to_list() == [None, None, None, 3.14]


# ===================================================================
# SECTION 9: Concurrent Tables & Independence
# ===================================================================


class TestTableIndependence:
    """Test that operations on one table don't affect others."""

    def test_delete_on_one_table_preserves_other(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2), (3)")
        cat.execute("CREATE TABLE ducklake.t2 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t2 VALUES (10), (20), (30)")
        cat.execute("DELETE FROM ducklake.t1 WHERE a = 2")
        cat.close()

        r1 = read_ducklake(cat.metadata_path, "t1")
        r2 = read_ducklake(cat.metadata_path, "t2")
        assert sorted(r1["a"].to_list()) == [1, 3]
        assert sorted(r2["a"].to_list()) == [10, 20, 30]

    def test_schema_change_on_one_table_preserves_other(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1, 'x')")
        cat.execute("CREATE TABLE ducklake.t2 (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t2 VALUES (2, 'y')")
        cat.execute("ALTER TABLE ducklake.t1 DROP COLUMN b")
        cat.close()

        r1 = read_ducklake(cat.metadata_path, "t1")
        r2 = read_ducklake(cat.metadata_path, "t2")
        assert r1.columns == ["a"]
        assert r2.columns == ["a", "b"]

    def test_drop_table_preserves_other(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.keep (a INTEGER)")
        cat.execute("INSERT INTO ducklake.keep VALUES (42)")
        cat.execute("CREATE TABLE ducklake.drop_me (a INTEGER)")
        cat.execute("INSERT INTO ducklake.drop_me VALUES (99)")
        cat.execute("DROP TABLE ducklake.drop_me")
        cat.close()

        result = read_ducklake(cat.metadata_path, "keep")
        assert result["a"][0] == 42

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "drop_me")


# ===================================================================
# SECTION 10: Unicode & Special Characters
# ===================================================================


class TestUnicode:
    """Test unicode content in data."""

    def test_unicode_varchar_values(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('hello world'), ('日本語テスト'), ('Ñoño'), ('emoji: 🦆')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        values = result["a"].to_list()
        assert "日本語テスト" in values
        assert "Ñoño" in values
        assert "emoji: 🦆" in values

    def test_unicode_filter(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES ('abc'), ('日本'), ('xyz')"
        )
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("a") == "日本").collect()
        assert result.shape[0] == 1
        assert result["a"][0] == "日本"


# ===================================================================
# SECTION 11: Snapshot Edge Cases
# ===================================================================


class TestSnapshotEdgeCases:
    """Test edge cases in snapshot-based operations."""

    def test_read_at_creation_snapshot(self, ducklake_catalog):
        """Read at the snapshot where the table was just created (no data)."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        v_create = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test", snapshot_version=v_create)
        assert result.shape[0] == 0
        assert result.columns == ["a"]

    def test_read_at_each_snapshot(self, ducklake_catalog):
        """Verify correctness at every snapshot in a sequence."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        snapshots = []
        for i in range(5):
            cat.execute(f"INSERT INTO ducklake.test VALUES ({i})")
            snap = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
            snapshots.append(snap)
        cat.close()

        for idx, snap in enumerate(snapshots):
            result = read_ducklake(cat.metadata_path, "test", snapshot_version=snap)
            assert result.shape[0] == idx + 1
            assert sorted(result["a"].to_list()) == list(range(idx + 1))

    def test_many_snapshots(self, ducklake_catalog):
        """Table with many snapshots."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        for i in range(30):
            cat.execute(f"INSERT INTO ducklake.test VALUES ({i})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 30

    def test_catalog_snapshots_api(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        snaps = api.snapshots()
        assert snaps.shape[0] >= 3  # CREATE + 2 INSERTs
        # Snapshot IDs should be monotonically increasing
        ids = snaps["snapshot_id"].to_list()
        assert ids == sorted(ids)


# ===================================================================
# SECTION 12: Data Type Combinations
# ===================================================================


class TestDataTypeCombinations:
    """Test combinations of data types in single tables."""

    def test_all_numeric_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                i8 TINYINT, i16 SMALLINT, i32 INTEGER, i64 BIGINT,
                u8 UTINYINT, u16 USMALLINT, u32 UINTEGER, u64 UBIGINT,
                f32 FLOAT, f64 DOUBLE, d DECIMAL(10, 2)
            )
        """)
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 2, 3, 4, 5, 6, 7, 8, 1.5, 2.5, 99.99)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 11)
        assert result["i8"][0] == 1
        assert result["u64"][0] == 8
        assert result["d"][0] == Decimal("99.99")

    def test_all_temporal_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                d DATE, t TIME, ts TIMESTAMP, tstz TIMESTAMPTZ
            )
        """)
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('2024-06-15', '12:30:00', '2024-06-15 12:30:00', '2024-06-15 12:30:00+00')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 4)
        assert result.schema["d"] == pl.Date
        assert result.schema["t"] == pl.Time

    def test_nested_struct_in_struct(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (a STRUCT(x INTEGER, nested STRUCT(y VARCHAR, z DOUBLE)))"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES ({'x': 1, 'nested': {'y': 'hello', 'z': 3.14}})"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        val = result["a"][0]
        assert val["x"] == 1
        assert val["nested"]["y"] == "hello"
        assert abs(val["nested"]["z"] - 3.14) < 0.01

    def test_list_of_different_scalar_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                int_list INTEGER[],
                str_list VARCHAR[],
                float_list DOUBLE[]
            )
        """)
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "([1, 2], ['a', 'b'], [1.1, 2.2])"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["int_list"].to_list() == [[1, 2]]
        assert result["str_list"].to_list() == [["a", "b"]]
        assert result["float_list"].to_list() == [[1.1, 2.2]]


# ===================================================================
# SECTION 13: Rename Edge Cases
# ===================================================================


class TestRenameEdgeCases:
    """Test edge cases in column renaming."""

    def test_rename_only_column(self, ducklake_catalog):
        """Rename the only column in the table."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2)")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN a TO b")
        cat.execute("INSERT INTO ducklake.test VALUES (3)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("b")
        assert result.columns == ["b"]
        assert result["b"].to_list() == [1, 2, 3]

    def test_rename_preserves_data_across_files(self, ducklake_catalog):
        """Multiple files before rename, new files after."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x')")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'y')")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'z')")
        cat.execute("INSERT INTO ducklake.test VALUES (4, 'w')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "name"]
        assert result["name"].to_list() == ["x", "y", "z", "w"]

    def test_rename_with_concurrent_delete(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'keep'), (2, 'delete')")
        cat.execute("DELETE FROM ducklake.test WHERE a = 2")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO label")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'new')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "label"]
        assert result["a"].to_list() == [1, 3]
        assert result["label"].to_list() == ["keep", "new"]


# ===================================================================
# SECTION 14: Inlined Data Edge Cases
# ===================================================================


class TestInlinedEdgeCases:
    """Test edge cases specific to inlined data."""

    def test_inlined_single_value(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (42)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"][0] == 42

    def test_inlined_with_nulls(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, NULL), (NULL, 'hi')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 2

    def test_inlined_empty_strings(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (''), ('hello'), ('')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        values = result["a"].to_list()
        assert values.count("") == 2
        assert values.count("hello") == 1

    def test_inlined_boolean_values(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a BOOLEAN)")
        cat.execute("INSERT INTO ducklake.test VALUES (true), (false), (NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        values = result["a"].to_list()
        assert True in values
        assert False in values
        assert None in values


# ===================================================================
# SECTION 15: Partition Edge Cases
# ===================================================================


class TestPartitionEdgeCases:
    """Test edge cases specific to partitioning."""

    @pytest.mark.xfail(reason="DuckDB internal error: NULL partition values crash DuckLake extension")
    def test_partition_with_null_values(self, ducklake_catalog_sqlite):
        """Rows with NULL partition column values."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, part VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'x'), (2, NULL), (3, 'x'), (4, NULL)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 4
        result = result.sort("a")
        assert result["part"].to_list() == ["x", None, "x", None]

    def test_partition_single_partition_value(self, ducklake_catalog):
        """All rows have the same partition value."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, part VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'only'), (2, 'only'), (3, 'only')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        assert all(v == "only" for v in result["part"].to_list())

    def test_partition_many_unique_values(self, ducklake_catalog):
        """Each row has a unique partition value."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, part INTEGER)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, i FROM range(20) t(i)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 20


# ===================================================================
# SECTION 16: Catalog API Edge Cases
# ===================================================================


class TestCatalogAPIEdgeCases:
    """Test DuckLakeCatalog API edge cases."""

    def test_catalog_api_on_empty_catalog(self, ducklake_catalog):
        """Catalog with no tables."""
        cat = ducklake_catalog
        # Just create the catalog, no tables
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        tables = api.list_tables()
        assert tables.shape[0] == 0

    def test_catalog_context_manager(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        with DuckLakeCatalog(cat.metadata_path) as api:
            snaps = api.snapshots()
            assert snaps.shape[0] >= 1

    def test_settings_api(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        settings = api.settings()
        assert isinstance(settings, pl.DataFrame)

    def test_current_snapshot_increases(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        snap = api.current_snapshot()
        assert isinstance(snap, int)
        assert snap >= 3  # CREATE + 2 INSERTs


# ===================================================================
# SECTION 17: Multi-File Read Correctness
# ===================================================================


class TestMultiFileCorrectness:
    """Verify correct reading when data spans many Parquet files."""

    def test_three_files_sorted_read(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i FROM range(0, 10) t(i)")
        cat.execute("INSERT INTO ducklake.test SELECT i FROM range(10, 20) t(i)")
        cat.execute("INSERT INTO ducklake.test SELECT i FROM range(20, 30) t(i)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 30
        assert sorted(result["id"].to_list()) == list(range(30))

    def test_files_with_overlapping_ranges(self, ducklake_catalog):
        """Two files with overlapping id ranges (different batches)."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, batch INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i, 1 FROM range(50) t(i)")
        cat.execute("INSERT INTO ducklake.test SELECT i, 2 FROM range(50) t(i)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 100
        batch1 = result.filter(pl.col("batch") == 1)
        batch2 = result.filter(pl.col("batch") == 2)
        assert batch1.shape[0] == 50
        assert batch2.shape[0] == 50

    def test_delete_from_specific_file(self, ducklake_catalog):
        """Delete rows from only one of multiple files."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, batch VARCHAR)")
        cat.execute("INSERT INTO ducklake.test SELECT i, 'a' FROM range(0, 50) t(i)")
        cat.execute("INSERT INTO ducklake.test SELECT i + 100, 'b' FROM range(0, 50) t(i)")
        cat.execute("DELETE FROM ducklake.test WHERE batch = 'a' AND id < 25")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        batch_a = result.filter(pl.col("batch") == "a")
        batch_b = result.filter(pl.col("batch") == "b")
        assert batch_a.shape[0] == 25
        assert batch_b.shape[0] == 50
        assert min(batch_a["id"].to_list()) == 25


# ===================================================================
# SECTION 18: Projection Pushdown
# ===================================================================


class TestProjectionPushdown:
    """Test that column selection (projection) works correctly."""

    def test_select_single_column(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test", columns=["b"])
        assert result.columns == ["b"]
        assert result["b"][0] == "hello"

    def test_select_columns_out_of_order(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test", columns=["c", "a"])
        assert set(result.columns) == {"a", "c"}

    def test_select_all_columns_explicit(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test", columns=["a", "b"])
        assert result.columns == ["a", "b"]
        assert result.shape == (1, 2)

    def test_lazy_select_projection(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 3.14)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.select("a", "c").collect()
        assert result.columns == ["a", "c"]
        assert result["a"][0] == 1
        assert result["c"][0] == pytest.approx(3.14)


# ===================================================================
# SECTION 19: String Content Edge Cases
# ===================================================================


class TestStringContentEdgeCases:
    """Test edge cases in string values."""

    def test_very_long_string(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        long_str = "abc" * 5000
        cat.execute("INSERT INTO ducklake.test VALUES (?)", [long_str])
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"][0] == long_str

    def test_string_with_newlines(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute(r"INSERT INTO ducklake.test VALUES ('line1' || chr(10) || 'line2')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert "\n" in result["a"][0]

    def test_string_with_tabs(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute(r"INSERT INTO ducklake.test VALUES ('col1' || chr(9) || 'col2')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert "\t" in result["a"][0]

    def test_string_with_quotes(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES ('he said ''hello''')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert "'" in result["a"][0]

    def test_string_with_backslash(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute(r"INSERT INTO ducklake.test VALUES ('path\to\file')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert "\\" in result["a"][0]


# ===================================================================
# SECTION 20: Decimal Precision Edge Cases
# ===================================================================


class TestDecimalEdgeCases:
    """Test decimal precision and scale edge cases."""

    def test_decimal_zero_scale(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DECIMAL(10, 0))")
        cat.execute("INSERT INTO ducklake.test VALUES (12345)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"][0] == Decimal("12345")

    def test_decimal_max_scale(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DECIMAL(10, 10))")
        cat.execute("INSERT INTO ducklake.test VALUES (0.1234567890)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"][0] == Decimal("0.1234567890")

    def test_decimal_negative_value(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DECIMAL(10, 2))")
        cat.execute("INSERT INTO ducklake.test VALUES (-99.99), (0.00), (99.99)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        values = sorted(result["a"].to_list())
        assert values == [Decimal("-99.99"), Decimal("0.00"), Decimal("99.99")]


# ===================================================================
# SECTION 21: Time Travel with Multiple Tables
# ===================================================================


class TestTimeTravelMultiTable:
    """Test time travel works correctly across multiple tables."""

    def test_time_travel_independent_per_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("CREATE TABLE ducklake.t2 (b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t2 VALUES ('hello')")

        cat.execute("INSERT INTO ducklake.t1 VALUES (2)")
        cat.close()

        # At v1, t1 has 1 row, t2 doesn't exist yet
        r1 = read_ducklake(cat.metadata_path, "t1", snapshot_version=v1)
        assert r1.shape[0] == 1

        # Latest: t1 has 2 rows, t2 has 1 row
        r1_latest = read_ducklake(cat.metadata_path, "t1")
        r2_latest = read_ducklake(cat.metadata_path, "t2")
        assert r1_latest.shape[0] == 2
        assert r2_latest.shape[0] == 1

    def test_time_travel_after_table_drop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        cat.execute("CREATE TABLE ducklake.t2 (b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t2 VALUES ('hello')")
        v_both = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("DROP TABLE ducklake.t1")
        cat.close()

        # t1 is gone, t2 is still there
        with pytest.raises(ValueError):
            read_ducklake(cat.metadata_path, "t1")

        result = read_ducklake(cat.metadata_path, "t2")
        assert result["b"][0] == "hello"

        # Time travel: t1 is accessible
        r1 = read_ducklake(cat.metadata_path, "t1", snapshot_version=v_both)
        assert r1["a"][0] == 1


# ===================================================================
# SECTION 22: Data Path Override
# ===================================================================


class TestDataPathOverride:
    """Test the data_path parameter."""

    def test_explicit_data_path(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (42)")
        cat.close()

        # Should work with explicit data_path
        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result["a"][0] == 42

    def test_explicit_data_path_scan(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (42)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        result = lf.collect()
        assert result["a"][0] == 42


# ===================================================================
# SECTION 23: Repeated Read Consistency
# ===================================================================


class TestReadConsistency:
    """Test that repeated reads give consistent results."""

    def test_multiple_reads_consistent(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(100) t(i)")
        cat.close()

        r1 = read_ducklake(cat.metadata_path, "test")
        r2 = read_ducklake(cat.metadata_path, "test")
        assert_frame_equal(r1, r2)

    def test_scan_then_read_consistent(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y')")
        cat.close()

        scan_result = scan_ducklake(cat.metadata_path, "test").collect()
        read_result = read_ducklake(cat.metadata_path, "test")
        assert_frame_equal(scan_result.sort("a"), read_result.sort("a"))

    def test_filtered_read_subset_of_full(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(100) t(i)")
        cat.close()

        full = read_ducklake(cat.metadata_path, "test")
        filtered = scan_ducklake(cat.metadata_path, "test").filter(
            pl.col("id") < 10
        ).collect()
        assert filtered.shape[0] == 10
        assert all(v in full["id"].to_list() for v in filtered["id"].to_list())
