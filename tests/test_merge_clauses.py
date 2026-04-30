"""Tests for extended ``MERGE INTO`` clauses.

DuckLake 1.0 / DuckDB grammar adds:

* ``WHEN MATCHED THEN DELETE``
* ``WHEN NOT MATCHED BY SOURCE THEN DELETE``
* ``WHEN NOT MATCHED BY SOURCE THEN UPDATE SET ...``
"""

from __future__ import annotations

import duckdb
import polars as pl
import pytest

from ducklake_polars import (
    merge_ducklake,
    read_ducklake,
    write_ducklake,
)


def _duckdb_supports_v10() -> bool:
    parts = duckdb.__version__.split(".")
    return len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (1, 5)


def _make_table(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    df = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "value": ["a", "b", "c", "d"],
    })
    write_ducklake(df, path, "events", mode="error", data_path=data_path)
    return path, data_path


def _read_sorted(path, data_path):
    return read_ducklake(path, "events", data_path=data_path).sort("id")


# ------------------------------------------------------------------
# WHEN MATCHED THEN DELETE
# ------------------------------------------------------------------


class TestWhenMatchedDelete:
    def test_deletes_matched_rows_only(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        source = pl.DataFrame({"id": [2, 4]})
        merge_ducklake(
            path, "events", source, "id",
            when_matched_delete=True,
            when_not_matched_insert=False,
            data_path=data_path,
        )
        df = _read_sorted(path, data_path)
        assert df["id"].to_list() == [1, 3]
        assert df["value"].to_list() == ["a", "c"]

    def test_matched_delete_with_insert_for_new_keys(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        source = pl.DataFrame({"id": [2, 5], "value": ["x", "e"]})
        merge_ducklake(
            path, "events", source, "id",
            when_matched_delete=True,
            when_not_matched_insert=True,
            data_path=data_path,
        )
        df = _read_sorted(path, data_path)
        # id=2 deleted, id=5 inserted
        assert df["id"].to_list() == [1, 3, 4, 5]
        assert df.filter(pl.col("id") == 5)["value"][0] == "e"

    def test_update_and_delete_are_mutually_exclusive(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        with pytest.raises(ValueError, match="mutually exclusive"):
            merge_ducklake(
                path, "events", pl.DataFrame({"id": [1]}), "id",
                when_matched_update=True,
                when_matched_delete=True,
                data_path=data_path,
            )


# ------------------------------------------------------------------
# WHEN NOT MATCHED BY SOURCE THEN DELETE
# ------------------------------------------------------------------


class TestWhenNotMatchedBySourceDelete:
    def test_deletes_only_unmatched(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        source = pl.DataFrame({"id": [1, 3]})
        merge_ducklake(
            path, "events", source, "id",
            when_not_matched_by_source_delete=True,
            when_not_matched_insert=False,
            data_path=data_path,
        )
        df = _read_sorted(path, data_path)
        # Only ids in source survive.
        assert df["id"].to_list() == [1, 3]

    def test_combined_with_matched_update(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        source = pl.DataFrame({"id": [1, 3], "value": ["X", "Y"]})
        merge_ducklake(
            path, "events", source, "id",
            when_matched_update=True,
            when_not_matched_by_source_delete=True,
            when_not_matched_insert=False,
            data_path=data_path,
        )
        df = _read_sorted(path, data_path)
        assert df["id"].to_list() == [1, 3]
        assert df["value"].to_list() == ["X", "Y"]


# ------------------------------------------------------------------
# WHEN NOT MATCHED BY SOURCE THEN UPDATE
# ------------------------------------------------------------------


class TestWhenNotMatchedBySourceUpdate:
    def test_marks_unmatched_rows(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        source = pl.DataFrame({"id": [2]})
        merge_ducklake(
            path, "events", source, "id",
            when_not_matched_by_source_update={"value": "missing"},
            when_not_matched_insert=False,
            data_path=data_path,
        )
        df = _read_sorted(path, data_path)
        # Rows 1, 3, 4 had no match in source → flagged.
        assert df["id"].to_list() == [1, 2, 3, 4]
        assert df["value"].to_list() == ["missing", "b", "missing", "missing"]

    def test_delete_and_update_bysource_mutually_exclusive(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        with pytest.raises(ValueError, match="mutually exclusive"):
            merge_ducklake(
                path, "events", pl.DataFrame({"id": [1]}), "id",
                when_not_matched_by_source_delete=True,
                when_not_matched_by_source_update={"value": "x"},
                data_path=data_path,
            )


# ------------------------------------------------------------------
# DuckDB roundtrip — verify that DuckDB's MERGE produces the same
# result against our catalog (parity check).
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_reads_after_matched_delete(tmp_path):
    path, data_path = _make_table(tmp_path)
    source = pl.DataFrame({"id": [2, 4]})
    merge_ducklake(
        path, "events", source, "id",
        when_matched_delete=True,
        when_not_matched_insert=False,
        data_path=data_path,
    )

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{path}' AS d (DATA_PATH '{data_path}')"
    )
    rows = con.execute(
        "SELECT id, value FROM d.events ORDER BY id"
    ).fetchall()
    con.close()
    assert rows == [(1, "a"), (3, "c")]
