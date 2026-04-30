"""Tests for the stateful DuckLakeCatalog write API.

The DuckLakeCatalog object wraps all free functions so users don't need
to pass metadata_path/data_path to every call.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import DuckLakeCatalog


@pytest.fixture
def catalog(tmp_path):
    """Create a fresh DuckLakeCatalog backed by SQLite."""
    import duckdb
    import os

    meta = str(tmp_path / "test.ducklake")
    data = str(tmp_path / "data")
    os.makedirs(data, exist_ok=True)

    # Initialize via DuckDB
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{meta}' AS ducklake "
        f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.close()

    return DuckLakeCatalog(meta, data_path=data)


class TestCatalogReadWrite:
    """Basic read/write through catalog object."""

    def test_write_and_read(self, catalog):
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        catalog.write("t", df)

        result = catalog.read("t")
        assert len(result) == 3
        assert_frame_equal(result.sort("id"), df.sort("id"))

    def test_write_append(self, catalog):
        df1 = pl.DataFrame({"id": [1, 2]})
        catalog.write("t", df1)

        df2 = pl.DataFrame({"id": [3, 4]})
        catalog.write("t", df2, mode="append")

        result = catalog.read("t")
        assert set(result["id"].to_list()) == {1, 2, 3, 4}

    def test_write_overwrite(self, catalog):
        catalog.write("t", pl.DataFrame({"id": [1, 2, 3]}))
        catalog.write("t", pl.DataFrame({"id": [10]}), mode="overwrite")

        result = catalog.read("t")
        assert result["id"].to_list() == [10]

    def test_scan_lazy(self, catalog):
        catalog.write("t", pl.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]}))

        lf = catalog.scan("t")
        assert isinstance(lf, pl.LazyFrame)

        result = lf.filter(pl.col("id") > 1).collect()
        assert len(result) == 2


class TestCatalogDML:
    """Delete, update, merge through catalog."""

    def test_delete(self, catalog):
        catalog.write("t", pl.DataFrame({"id": [1, 2, 3]}))
        catalog.delete("t", pl.col("id") == 2)

        result = catalog.read("t")
        assert set(result["id"].to_list()) == {1, 3}

    def test_update(self, catalog):
        catalog.write("t", pl.DataFrame({"id": [1, 2], "name": ["a", "b"]}))
        catalog.update("t", {"name": pl.lit("X")}, pl.col("id") == 1)

        result = catalog.read("t")
        row = result.filter(pl.col("id") == 1)
        assert row["name"][0] == "X"

    def test_merge_upsert(self, catalog):
        catalog.write("t", pl.DataFrame({"id": [1, 2], "val": ["a", "b"]}))
        source = pl.DataFrame({"id": [2, 3], "val": ["B", "c"]})
        catalog.merge("t", source, "id", when_matched_update=True)

        result = catalog.read("t").sort("id")
        assert result["val"].to_list() == ["a", "B", "c"]


class TestCatalogDDL:
    """Schema operations through catalog."""

    def test_create_and_drop_table(self, catalog):
        catalog.create_table("empty", {"id": pl.Int64, "name": pl.String})
        tables = catalog.list_tables()
        assert "empty" in tables["table_name"].to_list()

        catalog.drop_table("empty")
        tables2 = catalog.list_tables()
        assert "empty" not in tables2["table_name"].to_list()

    def test_add_drop_rename_column(self, catalog):
        catalog.write("t", pl.DataFrame({"id": [1], "a": ["x"]}))

        catalog.add_column("t", "b", pl.Int64)
        result = catalog.read("t")
        assert "b" in result.columns

        catalog.rename_column("t", "a", "name")
        result = catalog.read("t")
        assert "name" in result.columns
        assert "a" not in result.columns

        catalog.drop_column("t", "b")
        result = catalog.read("t")
        assert "b" not in result.columns


class TestCatalogMaintenance:
    """Compaction and maintenance through catalog."""

    def test_rewrite_data_files(self, catalog):
        for i in range(3):
            catalog.write("t", pl.DataFrame({"id": [i]}), mode="append")

        snap = catalog.rewrite_data_files("t")
        assert snap > 0

        result = catalog.read("t")
        assert len(result) == 3

    def test_rewrite_idempotent(self, catalog):
        for i in range(3):
            catalog.write("t", pl.DataFrame({"id": [i]}), mode="append")

        catalog.rewrite_data_files("t")
        snap2 = catalog.rewrite_data_files("t")
        assert snap2 == -1


class TestCatalogObjectAdvanced:
    """Forms exposed only via free functions before — now reachable on
    the ``DuckLakeCatalog`` object."""

    def test_set_partitioned_by_year_transform(self, tmp_path):
        import datetime
        import sqlite3

        from ducklake_polars._catalog_api import DuckLakeCatalog

        path = str(tmp_path / "c.ducklake")
        data_path = str(tmp_path / "data")
        cat = DuckLakeCatalog(path, data_path=data_path)
        cat.write(
            "t",
            pl.DataFrame({
                "id": [1],
                "ts": [datetime.datetime(2024, 1, 1)],
            }),
        )
        cat.set_partitioned_by("t", [("ts", "year")])

        con = sqlite3.connect(path)
        try:
            row = con.execute(
                "SELECT pc.transform FROM ducklake_partition_column pc "
                "JOIN ducklake_partition_info pi "
                "  ON pc.partition_id = pi.partition_id "
                "WHERE pi.end_snapshot IS NULL"
            ).fetchone()
        finally:
            con.close()
        assert row == ("year",)

    def test_set_sort_keys_with_expression(self, tmp_path):
        import sqlite3

        from ducklake_polars._catalog_api import DuckLakeCatalog

        path = str(tmp_path / "c.ducklake")
        data_path = str(tmp_path / "data")
        cat = DuckLakeCatalog(path, data_path=data_path)
        cat.write("t", pl.DataFrame({"id": [1, 2]}))
        cat.set_sort_keys(
            "t",
            [{"expression": "id + 1", "direction": "DESC"}],
        )

        con = sqlite3.connect(path)
        try:
            row = con.execute(
                "SELECT se.expression, se.sort_direction "
                "FROM ducklake_sort_expression se "
                "JOIN ducklake_sort_info si ON se.sort_id = si.sort_id "
                "WHERE si.end_snapshot IS NULL"
            ).fetchone()
        finally:
            con.close()
        assert row == ("id + 1", "DESC")

    def test_expire_snapshots_keeps_latest(self, tmp_path):
        import sqlite3

        from ducklake_polars._catalog_api import DuckLakeCatalog

        path = str(tmp_path / "c.ducklake")
        data_path = str(tmp_path / "data")
        cat = DuckLakeCatalog(path, data_path=data_path)
        for i in range(5):
            cat.write(
                "t", pl.DataFrame({"id": [i]}),
                mode="append" if i else "error",
            )

        # Keep only the most recent snapshot.
        cat.expire_snapshots(retain_last=1)

        con = sqlite3.connect(path)
        try:
            n = con.execute(
                "SELECT COUNT(*) FROM ducklake_snapshot"
            ).fetchone()[0]
        finally:
            con.close()
        assert n == 1
        # The surviving snapshot still exposes the full data.
        df = cat.read("t").sort("id")
        assert df["id"].to_list() == [0, 1, 2, 3, 4]


class TestCatalogMetadata:
    """Metadata queries through catalog."""

    def test_snapshots(self, catalog):
        catalog.write("t", pl.DataFrame({"id": [1]}))
        snaps = catalog.snapshots()
        assert len(snaps) >= 1

    def test_current_snapshot(self, catalog):
        catalog.write("t", pl.DataFrame({"id": [1]}))
        snap = catalog.current_snapshot()
        assert snap >= 1

    def test_table_info(self, catalog):
        catalog.write("t", pl.DataFrame({"id": [1]}))
        info = catalog.table_info()
        assert len(info) >= 1

    def test_list_files(self, catalog):
        catalog.write("t", pl.DataFrame({"id": [1]}))
        files = catalog.list_files("t")
        assert len(files) >= 1


class TestCatalogContextManager:
    """Catalog works as context manager."""

    def test_context_manager(self, catalog):
        with catalog as cat:
            cat.write("t", pl.DataFrame({"id": [1]}))
            result = cat.read("t")
            assert len(result) == 1
