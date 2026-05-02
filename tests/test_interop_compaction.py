"""Compaction interop tests: verify rewrite_data_files results are readable by DuckDB.

Tests both directions:
  - Polars compacts → DuckDB reads
  - DuckDB writes many files → Polars compacts → DuckDB reads
  - Correctness preserved through compaction with schema evolution
"""

from __future__ import annotations

import os

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    DuckLakeCatalog,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
    rewrite_data_files_ducklake,
)


def _reopen_duckdb(metadata_path, data_path, backend="sqlite"):
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    if backend == "postgres":
        source = f"ducklake:postgres:{metadata_path}"
    elif backend == "duckdb":
        source = f"ducklake:duckdb:{metadata_path}"
    else:
        source = f"ducklake:sqlite:{metadata_path}"
    con.execute(
        f"ATTACH '{source}' AS ducklake "
        f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
    )
    return con


class TestPolarsCompactsDuckDBReads:
    """Polars creates/appends data, compacts, DuckDB reads."""

    def test_basic_compaction_interop(self, ducklake_catalog):
        """Multiple appends → compact → DuckDB reads correctly."""
        cat = ducklake_catalog
        meta, data = cat.metadata_path, cat.data_path
        backend = cat.backend

        # Write with DuckDB (creates multiple files)
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val VARCHAR)")
        for batch in range(5):
            cat.execute(
                f"INSERT INTO ducklake.test "
                f"SELECT i + {batch * 100}, 'batch_{batch}' "
                f"FROM range(100) t(i)"
            )
        duckdb_count_before = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        cat.close()

        # Verify pre-compaction
        assert duckdb_count_before == 500

        # Compact with Polars
        api = DuckLakeCatalog(meta)
        files_before = api.list_files("test")
        assert files_before.shape[0] >= 5

        rewrite_data_files_ducklake(meta, "test")

        files_after = api.list_files("test")
        assert files_after.shape[0] < files_before.shape[0]

        # DuckDB reads the compacted data
        con = _reopen_duckdb(meta, data, getattr(cat, "backend", "sqlite"))
        duckdb_count_after = con.execute("SELECT COUNT(*) FROM ducklake.test").fetchone()[0]
        duckdb_sum = con.execute("SELECT SUM(id) FROM ducklake.test").fetchone()[0]
        con.close()

        assert duckdb_count_after == 500

        # Polars also reads correctly
        polars_result = read_ducklake(meta, "test")
        assert polars_result.shape[0] == 500
        assert polars_result["id"].sum() == duckdb_sum

    def test_compaction_preserves_values(self, ducklake_catalog):
        """All individual values survive compaction."""
        cat = ducklake_catalog
        meta, data = cat.metadata_path, cat.data_path
        backend = cat.backend

        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        for i in range(10):
            cat.execute(f"INSERT INTO ducklake.test VALUES ({i * 7 + 3})")

        # Get exact values from DuckDB before compaction
        duckdb_before = sorted(
            r[0] for r in cat.fetchall("SELECT id FROM ducklake.test")
        )
        cat.close()

        rewrite_data_files_ducklake(meta, "test")

        # Read back with DuckDB after compaction
        con = _reopen_duckdb(meta, data, getattr(cat, "backend", "sqlite"))
        duckdb_after = sorted(
            r[0] for r in con.execute("SELECT id FROM ducklake.test").fetchall()
        )
        con.close()

        assert duckdb_before == duckdb_after

    def test_compaction_with_schema_evolution(self, ducklake_catalog):
        """Compact table that has undergone schema evolution."""
        cat = ducklake_catalog
        meta, data = cat.metadata_path, cat.data_path
        backend = cat.backend

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2)")
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'hello')")
        cat.execute("INSERT INTO ducklake.test VALUES (4, 'world')")

        duckdb_before = cat.fetchall("SELECT * FROM ducklake.test ORDER BY a")
        cat.close()

        rewrite_data_files_ducklake(meta, "test")

        con = _reopen_duckdb(meta, data, getattr(cat, "backend", "sqlite"))
        duckdb_after = con.execute("SELECT * FROM ducklake.test ORDER BY a").fetchall()
        con.close()

        assert duckdb_before == duckdb_after

    def test_compaction_with_deletes(self, ducklake_catalog):
        """Compact after deletes — dead rows should be garbage collected."""
        cat = ducklake_catalog
        meta, data = cat.metadata_path, cat.data_path
        backend = cat.backend

        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(1000) t(i)")
        cat.execute("DELETE FROM ducklake.test WHERE id % 3 = 0")
        duckdb_remaining = cat.fetchall("SELECT id FROM ducklake.test ORDER BY id")
        cat.close()

        rewrite_data_files_ducklake(meta, "test")

        con = _reopen_duckdb(meta, data, getattr(cat, "backend", "sqlite"))
        duckdb_after = con.execute("SELECT id FROM ducklake.test ORDER BY id").fetchall()
        con.close()

        assert duckdb_remaining == duckdb_after

    def test_compaction_with_column_rename(self, ducklake_catalog):
        """Compact after column rename — DuckDB sees renamed columns."""
        cat = ducklake_catalog
        meta, data = cat.metadata_path, cat.data_path
        backend = cat.backend

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'old')")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'new')")

        duckdb_before = cat.fetchall("SELECT * FROM ducklake.test ORDER BY a")
        cat.close()

        rewrite_data_files_ducklake(meta, "test")

        con = _reopen_duckdb(meta, data, getattr(cat, "backend", "sqlite"))
        cols = [
            d[0] for d in con.execute("SELECT * FROM ducklake.test LIMIT 0").description
        ]
        duckdb_after = con.execute("SELECT * FROM ducklake.test ORDER BY a").fetchall()
        con.close()

        assert "name" in cols
        assert "b" not in cols
        assert duckdb_before == duckdb_after

    def test_compaction_time_travel_still_works(self, ducklake_catalog_sqlite):
        """Time travel to pre-compaction snapshot still returns correct data."""
        cat = ducklake_catalog_sqlite
        meta, data = cat.metadata_path, cat.data_path
        backend = cat.backend

        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        snap_before = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]
        cat.execute("INSERT INTO ducklake.test VALUES (4), (5)")
        cat.close()

        # Compact
        rewrite_data_files_ducklake(meta, "test")

        # Latest should have all 5
        result = read_ducklake(meta, "test")
        assert sorted(result["id"].to_list()) == [1, 2, 3, 4, 5]

        # Pre-compaction snapshot should still have 3
        result_old = read_ducklake(meta, "test", snapshot_version=snap_before)
        assert sorted(result_old["id"].to_list()) == [1, 2, 3]

    def test_duckdb_writes_after_compaction(self, ducklake_catalog):
        """DuckDB can continue writing after Polars compacts."""
        cat = ducklake_catalog
        meta, data = cat.metadata_path, cat.data_path
        backend = cat.backend

        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        for i in range(5):
            cat.execute(f"INSERT INTO ducklake.test VALUES ({i})")
        cat.close()

        rewrite_data_files_ducklake(meta, "test")

        # DuckDB writes more data
        con = _reopen_duckdb(meta, data, getattr(cat, "backend", "sqlite"))
        con.execute("INSERT INTO ducklake.test VALUES (100), (200)")
        count = con.execute("SELECT COUNT(*) FROM ducklake.test").fetchone()[0]
        con.close()

        assert count == 7

        # Polars also reads correctly
        result = read_ducklake(meta, "test")
        assert result.shape[0] == 7
        assert 100 in result["id"].to_list()
        assert 200 in result["id"].to_list()

    def test_compaction_partitioned_table(self, ducklake_catalog):
        """Compact a partitioned table — DuckDB reads partitions correctly."""
        cat = ducklake_catalog
        meta, data = cat.metadata_path, cat.data_path
        backend = cat.backend

        cat.execute("CREATE TABLE ducklake.test (val INTEGER, part VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part)")
        for batch in range(4):
            cat.execute(
                f"INSERT INTO ducklake.test VALUES "
                f"({batch * 10}, 'a'), ({batch * 10 + 1}, 'b')"
            )
        duckdb_before = cat.fetchall("SELECT * FROM ducklake.test ORDER BY val")
        cat.close()

        rewrite_data_files_ducklake(meta, "test")

        con = _reopen_duckdb(meta, data, getattr(cat, "backend", "sqlite"))
        duckdb_after = con.execute("SELECT * FROM ducklake.test ORDER BY val").fetchall()
        # Partition filter
        duckdb_filtered = con.execute(
            "SELECT val FROM ducklake.test WHERE part = 'a' ORDER BY val"
        ).fetchall()
        con.close()

        assert duckdb_before == duckdb_after
        assert all(r[0] % 10 == 0 for r in duckdb_filtered)


class TestPolarsWriteCompactDuckDBReads:
    """Polars writes → Polars compacts → DuckDB reads."""

    def test_polars_write_compact_duckdb_read(self, ducklake_catalog):
        cat = ducklake_catalog
        meta, data = cat.metadata_path, cat.data_path
        backend = cat.backend
        cat.close()  # We only need the initialized catalog

        # Polars writes multiple batches
        for batch in range(5):
            df = pl.DataFrame({
                "id": list(range(batch * 50, (batch + 1) * 50)),
                "label": [f"b{batch}"] * 50,
            })
            mode = "error" if batch == 0 else "append"
            write_ducklake(df, meta, "test", mode=mode)

        # Compact
        rewrite_data_files_ducklake(meta, "test")

        # DuckDB reads
        con = _reopen_duckdb(meta, data, getattr(cat, "backend", "sqlite"))
        count = con.execute("SELECT COUNT(*) FROM ducklake.test").fetchone()[0]
        duckdb_sum = con.execute("SELECT SUM(id) FROM ducklake.test").fetchone()[0]
        con.close()

        assert count == 250
        expected_sum = sum(range(250))
        assert duckdb_sum == expected_sum

    def test_polars_write_evolve_compact_duckdb_read(self, ducklake_catalog):
        """Polars writes, evolves schema, writes more, compacts, DuckDB reads."""
        from ducklake_polars import alter_ducklake_add_column

        cat = ducklake_catalog
        meta, data = cat.metadata_path, cat.data_path
        backend = cat.backend
        cat.close()

        df1 = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df1, meta, "test", mode="error")

        alter_ducklake_add_column(meta, "test", "b", pl.String())

        df2 = pl.DataFrame({"a": [4, 5], "b": ["x", "y"]})
        write_ducklake(df2, meta, "test", mode="append")

        rewrite_data_files_ducklake(meta, "test")

        con = _reopen_duckdb(meta, data, getattr(cat, "backend", "sqlite"))
        rows = con.execute("SELECT * FROM ducklake.test ORDER BY a").fetchall()
        con.close()

        assert len(rows) == 5
        assert rows[0] == (1, None)
        assert rows[3] == (4, "x")
        assert rows[4] == (5, "y")
