"""Tests for catalog bootstrapping — creating catalogs without DuckDB."""

from __future__ import annotations

import os
import sqlite3

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_core._bootstrap import _needs_bootstrap, bootstrap_catalog


class TestNeedsBootstrap:
    """Test the _needs_bootstrap detection logic."""

    def test_nonexistent_file(self, tmp_path):
        path = str(tmp_path / "missing.ducklake")
        assert _needs_bootstrap(path) is True

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.ducklake")
        open(path, "w").close()
        assert os.path.getsize(path) == 0
        assert _needs_bootstrap(path) is True

    def test_existing_sqlite_without_tables(self, tmp_path):
        path = str(tmp_path / "bare.ducklake")
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE foo (x INTEGER)")
        con.commit()
        con.close()
        assert _needs_bootstrap(path) is True

    def test_bootstrapped_catalog(self, tmp_path):
        path = str(tmp_path / "ok.ducklake")
        bootstrap_catalog(path)
        assert _needs_bootstrap(path) is False


class TestBootstrapCatalog:
    """Test the bootstrap_catalog function itself."""

    def test_creates_all_tables(self, tmp_path):
        path = str(tmp_path / "test.ducklake")
        bootstrap_catalog(path)

        con = sqlite3.connect(path)
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        con.close()

        expected_tables = sorted([
            "ducklake_metadata",
            "ducklake_snapshot",
            "ducklake_snapshot_changes",
            "ducklake_schema",
            "ducklake_schema_versions",
            "ducklake_table",
            "ducklake_column",
            "ducklake_data_file",
            "ducklake_delete_file",
            "ducklake_file_column_stats",
            "ducklake_file_partition_value",
            "ducklake_file_variant_stats",
            "ducklake_files_scheduled_for_deletion",
            "ducklake_inlined_data_tables",
            "ducklake_column_mapping",
            "ducklake_name_mapping",
            "ducklake_partition_column",
            "ducklake_partition_info",
            "ducklake_table_stats",
            "ducklake_table_column_stats",
            "ducklake_tag",
            "ducklake_column_tag",
            "ducklake_view",
            "ducklake_sort_info",
            "ducklake_sort_expression",
            "ducklake_macro",
            "ducklake_macro_impl",
            "ducklake_macro_parameters",
        ])
        assert tables == expected_tables

    def test_metadata_rows(self, tmp_path):
        path = str(tmp_path / "test.ducklake")
        bootstrap_catalog(path)

        con = sqlite3.connect(path)
        rows = con.execute(
            "SELECT key, value FROM ducklake_metadata ORDER BY key"
        ).fetchall()
        con.close()

        keys = {r[0]: r[1] for r in rows}
        assert keys["version"] == "0.4"
        assert keys["created_by"] == "ducklake-dataframe"
        assert keys["encrypted"] == "false"
        assert "data_path" in keys

    def test_snapshot_seeded(self, tmp_path):
        path = str(tmp_path / "test.ducklake")
        bootstrap_catalog(path)

        con = sqlite3.connect(path)
        snap = con.execute("SELECT * FROM ducklake_snapshot").fetchone()
        con.close()
        assert snap is not None
        assert snap[0] == 0  # snapshot_id

    def test_main_schema_created(self, tmp_path):
        path = str(tmp_path / "test.ducklake")
        bootstrap_catalog(path)

        con = sqlite3.connect(path)
        schema = con.execute("SELECT * FROM ducklake_schema").fetchone()
        con.close()
        assert schema is not None
        assert schema[0] == 0  # schema_id
        assert schema[4] == "main"  # schema_name

    def test_idempotent(self, tmp_path):
        path = str(tmp_path / "test.ducklake")
        bootstrap_catalog(path)
        bootstrap_catalog(path)  # should be no-op

        con = sqlite3.connect(path)
        count = con.execute("SELECT COUNT(*) FROM ducklake_metadata").fetchone()[0]
        con.close()
        assert count == 4  # version, created_by, encrypted, data_path

    def test_custom_data_path(self, tmp_path):
        path = str(tmp_path / "test.ducklake")
        custom_data = str(tmp_path / "my_data")
        bootstrap_catalog(path, data_path=custom_data)

        con = sqlite3.connect(path)
        row = con.execute(
            "SELECT value FROM ducklake_metadata WHERE key = 'data_path'"
        ).fetchone()
        con.close()
        assert row[0] == os.path.abspath(custom_data) + "/"
        assert os.path.isdir(custom_data)

    def test_creates_parent_directory(self, tmp_path):
        path = str(tmp_path / "subdir" / "deep" / "test.ducklake")
        bootstrap_catalog(path)
        assert os.path.exists(path)


class TestDuckDBInterop:
    """Test that a bootstrapped catalog is readable by DuckDB."""

    def test_duckdb_can_attach(self, tmp_path):
        path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")
        bootstrap_catalog(path, data_path=data_path)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{path}' AS ducklake "
            f"(DATA_PATH '{data_path}')"
        )
        # Should be able to list tables (empty)
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_catalog = 'ducklake'"
        ).fetchall()
        con.close()
        assert tables == []

    def test_duckdb_can_create_table_on_bootstrapped(self, tmp_path):
        path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")
        bootstrap_catalog(path, data_path=data_path)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{path}' AS ducklake "
            f"(DATA_PATH '{data_path}')"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")
        result = con.execute("SELECT * FROM ducklake.test").fetchall()
        con.close()
        assert result == [(1, "hello")]

    def test_write_then_duckdb_read(self, tmp_path):
        """The core use case: write with ducklake-dataframe, read with DuckDB."""
        from ducklake_polars import read_ducklake, write_ducklake

        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")

        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        write_ducklake(df, path, "users", mode="error", data_path=data_path)

        # Read back with DuckDB
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{path}' AS ducklake "
            f"(DATA_PATH '{data_path}')"
        )
        result = con.execute("SELECT * FROM ducklake.users ORDER BY id").fetchall()
        con.close()
        assert result == [(1, "a"), (2, "b"), (3, "c")]

    def test_duckdb_write_then_dataframe_read(self, tmp_path):
        """Verify round-trip: DuckDB writes, ducklake-dataframe reads."""
        from ducklake_polars import read_ducklake

        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{path}' AS ducklake "
            f"(DATA_PATH '{data_path}')"
        )
        con.execute("CREATE TABLE ducklake.products (id INTEGER, price DOUBLE)")
        con.execute("INSERT INTO ducklake.products VALUES (1, 9.99), (2, 19.99)")
        con.close()

        result = read_ducklake(path, "products")
        expected = pl.DataFrame({
            "id": pl.Series([1, 2], dtype=pl.Int32),
            "price": [9.99, 19.99],
        })
        assert_frame_equal(result, expected)


class TestWriteDucklakeBootstrap:
    """Test that write_ducklake works on fresh catalogs."""

    def test_write_error_mode_fresh_catalog(self, tmp_path):
        """The exact README example."""
        from ducklake_polars import write_ducklake

        path = str(tmp_path / "catalog.ducklake")
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        write_ducklake(df, path, "users", mode="error")
        assert os.path.exists(path)

    def test_write_append_mode_fresh_catalog(self, tmp_path):
        from ducklake_polars import read_ducklake, write_ducklake

        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")

        df1 = pl.DataFrame({"x": [1, 2]})
        write_ducklake(df1, path, "t", mode="append", data_path=data_path)

        df2 = pl.DataFrame({"x": [3, 4]})
        write_ducklake(df2, path, "t", mode="append", data_path=data_path)

        result = read_ducklake(path, "t")
        assert len(result) == 4
        assert sorted(result["x"].to_list()) == [1, 2, 3, 4]

    def test_write_overwrite_mode_fresh_catalog(self, tmp_path):
        from ducklake_polars import read_ducklake, write_ducklake

        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")

        df = pl.DataFrame({"v": [10, 20]})
        write_ducklake(df, path, "t", mode="overwrite", data_path=data_path)

        result = read_ducklake(path, "t")
        assert result["v"].to_list() == [10, 20]

    def test_write_empty_dataframe_fresh_catalog(self, tmp_path):
        """Create table from empty DataFrame."""
        from ducklake_polars import write_ducklake

        path = str(tmp_path / "catalog.ducklake")
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
        write_ducklake(df, path, "empty_t", mode="error")

    def test_write_no_explicit_data_path(self, tmp_path):
        """When no data_path is given, bootstrap should derive one."""
        from ducklake_polars import read_ducklake, write_ducklake

        path = str(tmp_path / "catalog.ducklake")
        df = pl.DataFrame({"id": [1]})
        write_ducklake(df, path, "t", mode="error")

        result = read_ducklake(path, "t")
        assert result["id"].to_list() == [1]

    def test_multiple_tables_fresh_catalog(self, tmp_path):
        from ducklake_polars import read_ducklake, write_ducklake

        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")

        write_ducklake(
            pl.DataFrame({"a": [1]}), path, "t1",
            mode="error", data_path=data_path,
        )
        write_ducklake(
            pl.DataFrame({"b": ["x"]}), path, "t2",
            mode="error", data_path=data_path,
        )

        r1 = read_ducklake(path, "t1")
        r2 = read_ducklake(path, "t2")
        assert r1["a"].to_list() == [1]
        assert r2["b"].to_list() == ["x"]
