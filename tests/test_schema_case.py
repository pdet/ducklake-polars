"""Tests for case-insensitive schema name lookup."""

from __future__ import annotations

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    create_ducklake_schema,
    read_ducklake,
    write_ducklake,
)


class TestSchemaCaseInsensitive:
    """Schema names stored by DuckDB are lowercase (unquoted).

    The ducklake reader/writer must find them regardless of the case
    used in the Python API call.
    """

    def test_create_schema_mixed_case_read_lowercase(self, make_write_catalog):
        """Create schema 'MySchema' via DuckDB (stored as 'myschema'), read with lowercase."""
        cat = make_write_catalog()
        df = pl.DataFrame({"x": pl.Series([1, 2, 3], dtype=pl.Int32)})

        # DuckDB stores unquoted schema names as lowercase
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        source = cat.attach_source()
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE SCHEMA ducklake.MySchema")
        con.execute("CREATE TABLE ducklake.MySchema.test_tbl (x INTEGER)")
        con.execute("INSERT INTO ducklake.MySchema.test_tbl VALUES (1), (2), (3)")
        con.close()

        # Read with lowercase schema — should work
        result = read_ducklake(cat.metadata_path, "test_tbl", schema="myschema")
        assert_frame_equal(result, df)

    def test_create_schema_mixed_case_read_uppercase(self, make_write_catalog):
        """Create schema 'MySchema' via DuckDB (stored as 'myschema'), read with MixedCase."""
        cat = make_write_catalog()
        df = pl.DataFrame({"x": pl.Series([10, 20], dtype=pl.Int32)})

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        source = cat.attach_source()
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE SCHEMA ducklake.MySchema")
        con.execute("CREATE TABLE ducklake.MySchema.test_tbl (x INTEGER)")
        con.execute("INSERT INTO ducklake.MySchema.test_tbl VALUES (10), (20)")
        con.close()

        # Read with MixedCase schema — should work
        result = read_ducklake(cat.metadata_path, "test_tbl", schema="MySchema")
        assert_frame_equal(result, df)

        # Read with UPPERCASE schema — should also work
        result2 = read_ducklake(cat.metadata_path, "test_tbl", schema="MYSCHEMA")
        assert_frame_equal(result2, df)

    def test_write_to_schema_case_insensitive(self, make_write_catalog):
        """Create schema via API with lowercase, write using MixedCase."""
        cat = make_write_catalog()

        create_ducklake_schema(cat.metadata_path, "myschema")

        df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int64)})

        # Write using MixedCase schema name
        write_ducklake(df, cat.metadata_path, "tbl", schema="MySchema", mode="error")

        # Read back with lowercase
        result = read_ducklake(cat.metadata_path, "tbl", schema="myschema")
        assert_frame_equal(result, df)

        # Read back with UPPERCASE
        result2 = read_ducklake(cat.metadata_path, "tbl", schema="MYSCHEMA")
        assert_frame_equal(result2, df)

    def test_default_schema_no_regression(self, make_write_catalog):
        """Create table in default 'main' schema — no regression."""
        cat = make_write_catalog()
        df = pl.DataFrame({"v": pl.Series([100, 200], dtype=pl.Int64)})

        write_ducklake(df, cat.metadata_path, "default_tbl", mode="error")

        result = read_ducklake(cat.metadata_path, "default_tbl")
        assert_frame_equal(result, df)

        # Also read with explicit "main"
        result2 = read_ducklake(cat.metadata_path, "default_tbl", schema="main")
        assert_frame_equal(result2, df)

        # And with "MAIN"
        result3 = read_ducklake(cat.metadata_path, "default_tbl", schema="MAIN")
        assert_frame_equal(result3, df)

    def test_multiple_schemas_mixed_case(self, make_write_catalog):
        """Multiple schemas created with mixed case — each resolves correctly."""
        cat = make_write_catalog()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        source = cat.attach_source()
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE SCHEMA ducklake.alpha")
        con.execute("CREATE TABLE ducklake.alpha.tbl (x INTEGER)")
        con.execute("INSERT INTO ducklake.alpha.tbl VALUES (1)")

        con.execute("CREATE SCHEMA ducklake.beta")
        con.execute("CREATE TABLE ducklake.beta.tbl (x INTEGER)")
        con.execute("INSERT INTO ducklake.beta.tbl VALUES (2)")
        con.close()

        # Read with various cases
        r1 = read_ducklake(cat.metadata_path, "tbl", schema="Alpha")
        assert r1["x"].to_list() == [1]

        r2 = read_ducklake(cat.metadata_path, "tbl", schema="BETA")
        assert r2["x"].to_list() == [2]

        r3 = read_ducklake(cat.metadata_path, "tbl", schema="alpha")
        assert r3["x"].to_list() == [1]

        r4 = read_ducklake(cat.metadata_path, "tbl", schema="beta")
        assert r4["x"].to_list() == [2]

    def test_create_schema_duplicate_case_insensitive(self, make_write_catalog):
        """Creating a schema with a different case than an existing one should raise."""
        cat = make_write_catalog()

        create_ducklake_schema(cat.metadata_path, "myschema")

        # Creating with different case should detect the duplicate
        with pytest.raises(ValueError, match="already exists"):
            create_ducklake_schema(cat.metadata_path, "MYSCHEMA")

    def test_append_to_schema_case_insensitive(self, make_write_catalog):
        """Write to a schema with one case, then append with another."""
        cat = make_write_catalog()

        create_ducklake_schema(cat.metadata_path, "analytics")

        df1 = pl.DataFrame({"val": pl.Series([1], dtype=pl.Int64)})
        write_ducklake(df1, cat.metadata_path, "data", schema="analytics", mode="error")

        df2 = pl.DataFrame({"val": pl.Series([2], dtype=pl.Int64)})
        write_ducklake(df2, cat.metadata_path, "data", schema="ANALYTICS", mode="append")

        result = read_ducklake(cat.metadata_path, "data", schema="Analytics")
        assert result.shape[0] == 2
        assert sorted(result["val"].to_list()) == [1, 2]
