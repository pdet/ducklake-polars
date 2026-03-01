"""Tests for DuckDB as metadata catalog backend."""

from __future__ import annotations

import os
import tempfile

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    DuckLakeCatalog,
    create_ducklake_table,
    drop_ducklake_table,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
    delete_ducklake,
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
)


@pytest.fixture
def duckdb_catalog():
    """Create a DuckLake catalog backed by DuckDB."""
    tmp = tempfile.mkdtemp()
    meta = os.path.join(tmp, "catalog.duckdb")
    data = os.path.join(tmp, "data")

    # Initialize catalog schema using DuckDB's ducklake extension
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:duckdb:{meta}' AS ducklake "
        f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.close()

    yield meta, data


class TestDuckDBBackendRead:
    """Test reading from a DuckDB-backed catalog."""

    def test_read_basic(self, duckdb_catalog):
        meta, data = duckdb_catalog
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:duckdb:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        con.close()

        result = read_ducklake(meta, "test", data_path=data)
        assert result.shape == (2, 2)
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == ["hello", "world"]

    def test_scan_basic(self, duckdb_catalog):
        meta, data = duckdb_catalog
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:duckdb:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y'), (3, 'z')")
        con.close()

        result = scan_ducklake(meta, "test", data_path=data).collect()
        assert result.shape == (3, 2)

    def test_read_with_column_selection(self, duckdb_catalog):
        meta, data = duckdb_catalog
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:duckdb:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'x', 3.14)")
        con.close()

        result = read_ducklake(meta, "test", data_path=data, columns=["a", "c"])
        assert list(result.columns) == ["a", "c"]


class TestDuckDBBackendWrite:
    """Test writing to a DuckDB-backed catalog."""

    def test_create_and_write(self, duckdb_catalog):
        meta, data = duckdb_catalog
        df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}).cast({"x": pl.Int32})
        write_ducklake(df, meta, "mytest", mode="error", data_path=data)

        result = read_ducklake(meta, "mytest", data_path=data)
        assert_frame_equal(result, df)

    def test_append(self, duckdb_catalog):
        meta, data = duckdb_catalog
        df1 = pl.DataFrame({"x": [1, 2]}).cast({"x": pl.Int32})
        df2 = pl.DataFrame({"x": [3, 4]}).cast({"x": pl.Int32})

        write_ducklake(df1, meta, "mytest", mode="error", data_path=data)
        write_ducklake(df2, meta, "mytest", mode="append", data_path=data)

        result = read_ducklake(meta, "mytest", data_path=data)
        assert result["x"].sort().to_list() == [1, 2, 3, 4]

    def test_delete(self, duckdb_catalog):
        meta, data = duckdb_catalog
        df = pl.DataFrame({"x": [1, 2, 3, 4]}).cast({"x": pl.Int32})
        write_ducklake(df, meta, "test", mode="error", data_path=data)
        delete_ducklake(meta, "test", pl.col("x") > 2, data_path=data)

        result = read_ducklake(meta, "test", data_path=data)
        assert result["x"].sort().to_list() == [1, 2]


class TestDuckDBBackendDDL:
    """Test DDL operations with DuckDB backend."""

    def test_create_and_drop_table(self, duckdb_catalog):
        meta, data = duckdb_catalog
        create_ducklake_table(
            meta, "ddl_test", {"a": pl.Int32, "b": pl.String}, data_path=data
        )
        result = read_ducklake(meta, "ddl_test", data_path=data)
        assert result.shape == (0, 2)

        drop_ducklake_table(meta, "ddl_test", data_path=data)
        with pytest.raises(Exception):
            read_ducklake(meta, "ddl_test", data_path=data)

    def test_alter_add_column(self, duckdb_catalog):
        meta, data = duckdb_catalog
        df = pl.DataFrame({"a": [1, 2]}).cast({"a": pl.Int32})
        write_ducklake(df, meta, "test", mode="error", data_path=data)

        alter_ducklake_add_column(meta, "test", "b", pl.String, data_path=data)
        result = read_ducklake(meta, "test", data_path=data)
        assert "b" in result.columns

    def test_alter_rename_column(self, duckdb_catalog):
        meta, data = duckdb_catalog
        df = pl.DataFrame({"old_name": [1, 2]}).cast({"old_name": pl.Int32})
        write_ducklake(df, meta, "test", mode="error", data_path=data)

        alter_ducklake_rename_column(meta, "test", "old_name", "new_name", data_path=data)
        result = read_ducklake(meta, "test", data_path=data)
        assert "new_name" in result.columns
        assert "old_name" not in result.columns


class TestDuckDBBackendInterop:
    """Test interop: write with DuckDB backend, read with DuckDB engine and vice versa."""

    def test_we_write_duckdb_reads(self, duckdb_catalog):
        meta, data = duckdb_catalog
        df = pl.DataFrame({"a": [10, 20, 30], "b": ["x", "y", "z"]}).cast({"a": pl.Int32})
        write_ducklake(df, meta, "interop", mode="error", data_path=data)

        # Read with DuckDB engine
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:duckdb:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        result = con.execute("SELECT * FROM ducklake.interop ORDER BY a").fetchall()
        con.close()

        assert result == [(10, "x"), (20, "y"), (30, "z")]

    def test_duckdb_writes_we_read(self, duckdb_catalog):
        meta, data = duckdb_catalog
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:duckdb:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.interop (a INTEGER, b VARCHAR)")
        con.execute("INSERT INTO ducklake.interop VALUES (1, 'a'), (2, 'b')")
        con.close()

        result = read_ducklake(meta, "interop", data_path=data)
        assert result.shape == (2, 2)

    def test_mixed_writes(self, duckdb_catalog):
        """DuckDB writes, we append, DuckDB reads all."""
        meta, data = duckdb_catalog
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:duckdb:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.mixed (id INTEGER, val VARCHAR)")
        con.execute("INSERT INTO ducklake.mixed VALUES (1, 'duckdb')")
        con.close()

        # We append
        write_ducklake(
            pl.DataFrame({"id": [2], "val": ["polars"]}).cast({"id": pl.Int32}),
            meta, "mixed", mode="append", data_path=data,
        )

        # DuckDB reads all
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:duckdb:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        result = con.execute("SELECT * FROM ducklake.mixed ORDER BY id").fetchall()
        con.close()
        assert result == [(1, "duckdb"), (2, "polars")]


class TestDuckDBCatalogAPI:
    """Test catalog API with DuckDB backend."""

    def test_catalog_snapshots(self, duckdb_catalog):
        meta, data = duckdb_catalog
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:duckdb:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER)")
        con.execute("INSERT INTO ducklake.test VALUES (1)")
        con.close()

        cat = DuckLakeCatalog(meta)
        snaps = cat.snapshots()
        assert len(snaps) >= 1
