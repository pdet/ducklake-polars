"""Tests for case-insensitive table name lookup."""

from __future__ import annotations

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import read_ducklake, write_ducklake


class TestCaseInsensitiveTableLookup:
    """Table names stored by DuckDB are lowercase (unquoted).

    The ducklake reader/writer must find them regardless of the case
    used in the Python API call.
    """

    def test_lowercase_created_uppercase_read(self, make_write_catalog):
        """Create table with lowercase name, read with uppercase."""
        cat = make_write_catalog()
        df = pl.DataFrame({"x": pl.Series([1, 2, 3], dtype=pl.Int32)})

        # DuckDB stores unquoted names as lowercase
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        source = cat.attach_source()
        con.execute(f"ATTACH '{source}' AS ducklake (DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)")
        con.execute("CREATE TABLE ducklake.main.mytable (x INTEGER)")
        con.execute("INSERT INTO ducklake.main.mytable VALUES (1), (2), (3)")
        con.close()

        # Read with uppercase — should work
        result = read_ducklake(cat.metadata_path, "MYTABLE")
        assert result.shape[0] == 3
        assert_frame_equal(result, df)

    def test_mixedcase_created_lowercase_read(self, make_write_catalog):
        """Create table with MixedCase (unquoted → stored as lowercase), read with lowercase."""
        cat = make_write_catalog()
        df = pl.DataFrame({"x": pl.Series([10, 20], dtype=pl.Int32)})

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        source = cat.attach_source()
        con.execute(f"ATTACH '{source}' AS ducklake (DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)")
        # Unquoted MixedCase → DuckDB stores as 'mixedtable'
        con.execute("CREATE TABLE ducklake.main.MixedTable (x INTEGER)")
        con.execute("INSERT INTO ducklake.main.MixedTable VALUES (10), (20)")
        con.close()

        # Read with lowercase — should work
        result = read_ducklake(cat.metadata_path, "mixedtable")
        assert_frame_equal(result, df)

        # Read with original MixedCase — should also work
        result2 = read_ducklake(cat.metadata_path, "MixedTable")
        assert_frame_equal(result2, df)

    def test_mytable_all_case_variants(self, make_write_catalog):
        """Create 'MyTable', insert data, read with 'mytable' and 'MYTABLE' → same data."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int32), "b": ["hello", "world"]})

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        source = cat.attach_source()
        con.execute(f"ATTACH '{source}' AS ducklake (DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)")
        # Unquoted MyTable → stored as 'mytable'
        con.execute("CREATE TABLE ducklake.main.MyTable (a INTEGER, b VARCHAR)")
        con.execute("INSERT INTO ducklake.main.MyTable VALUES (1, 'hello'), (2, 'world')")
        con.close()

        result_lower = read_ducklake(cat.metadata_path, "mytable")
        result_upper = read_ducklake(cat.metadata_path, "MYTABLE")
        result_mixed = read_ducklake(cat.metadata_path, "MyTable")

        assert_frame_equal(result_lower, df)
        assert_frame_equal(result_upper, df)
        assert_frame_equal(result_mixed, df)

    def test_quoted_name_preserves_case(self, make_write_catalog):
        """Quoted name stays as-is (DuckDB preserves case for quoted identifiers)."""
        cat = make_write_catalog()
        df = pl.DataFrame({"x": pl.Series([42], dtype=pl.Int32)})

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        source = cat.attach_source()
        con.execute(f"ATTACH '{source}' AS ducklake (DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)")
        # Quoted → DuckDB stores as 'CaseSensitive' (preserves case)
        con.execute('CREATE TABLE ducklake.main."CaseSensitive" (x INTEGER)')
        con.execute('INSERT INTO ducklake.main."CaseSensitive" VALUES (42)')
        con.close()

        # Exact quoted name works
        result = read_ducklake(cat.metadata_path, "CaseSensitive")
        assert_frame_equal(result, df)

        # Case-insensitive lookup also works (LOWER match)
        result2 = read_ducklake(cat.metadata_path, "casesensitive")
        assert_frame_equal(result2, df)

    def test_write_case_insensitive(self, make_write_catalog):
        """Writing to a table with different case than stored should work."""
        cat = make_write_catalog()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        source = cat.attach_source()
        con.execute(f"ATTACH '{source}' AS ducklake (DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)")
        con.execute("CREATE TABLE ducklake.main.mytable (x INTEGER)")
        con.close()

        # Write using uppercase name
        df = pl.DataFrame({"x": pl.Series([100, 200], dtype=pl.Int32)})
        write_ducklake(df, cat.metadata_path, "MYTABLE", mode="append")

        # Read back
        result = read_ducklake(cat.metadata_path, "mytable")
        assert_frame_equal(result, df)
