"""Reader applies ``ducklake_name_mapping`` for added Parquet files."""

from __future__ import annotations

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ducklake_polars import (
    add_files_ducklake,
    alter_ducklake_rename_column,
    create_ducklake_table,
    read_ducklake,
    write_ducklake,
)


def _duckdb_supports_v10() -> bool:
    parts = duckdb.__version__.split(".")
    return len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (1, 5)


def _make_added_file(tmp_path) -> tuple[str, str, str]:
    metadata_path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    create_ducklake_table(
        metadata_path, "events",
        pl.Schema({"a": pl.Int64(), "b": pl.String(), "c": pl.Float64()}),
        data_path=data_path,
    )
    external = str(tmp_path / "external.parquet")
    pq.write_table(
        pa.table({"a": [1, 2], "b": ["x", "y"], "c": [1.5, 2.5]}),
        external,
    )
    add_files_ducklake(metadata_path, "events", [external], data_path=data_path)
    return metadata_path, data_path, external


def test_simple_add_files_read(tmp_path):
    metadata_path, data_path, _ = _make_added_file(tmp_path)
    df = read_ducklake(metadata_path, "events", data_path=data_path).sort("a")
    assert df["a"].to_list() == [1, 2]
    assert df["b"].to_list() == ["x", "y"]
    assert df["c"].to_list() == [1.5, 2.5]


def test_rename_one_column_after_add_files(tmp_path):
    metadata_path, data_path, _ = _make_added_file(tmp_path)
    alter_ducklake_rename_column(
        metadata_path, "events", "a", "id", data_path=data_path,
    )
    df = read_ducklake(metadata_path, "events", data_path=data_path).sort("id")
    assert "id" in df.columns
    assert "a" not in df.columns
    assert df["id"].to_list() == [1, 2]
    assert df["b"].to_list() == ["x", "y"]


def test_rename_multiple_columns_after_add_files(tmp_path):
    metadata_path, data_path, _ = _make_added_file(tmp_path)
    alter_ducklake_rename_column(
        metadata_path, "events", "a", "id", data_path=data_path,
    )
    alter_ducklake_rename_column(
        metadata_path, "events", "b", "label", data_path=data_path,
    )
    alter_ducklake_rename_column(
        metadata_path, "events", "c", "score", data_path=data_path,
    )
    df = read_ducklake(metadata_path, "events", data_path=data_path).sort("id")
    assert set(df.columns) == {"id", "label", "score"}
    assert df["id"].to_list() == [1, 2]
    assert df["label"].to_list() == ["x", "y"]
    assert df["score"].to_list() == [1.5, 2.5]


def test_rename_then_insert_more_data(tmp_path):
    """After a rename, further inserts use the new name; reads merge both."""
    metadata_path, data_path, _ = _make_added_file(tmp_path)
    alter_ducklake_rename_column(
        metadata_path, "events", "a", "id", data_path=data_path,
    )

    write_ducklake(
        pl.DataFrame({"id": [3, 4], "b": ["p", "q"], "c": [3.5, 4.5]}),
        metadata_path, "events", mode="append", data_path=data_path,
    )

    df = read_ducklake(metadata_path, "events", data_path=data_path).sort("id")
    assert df["id"].to_list() == [1, 2, 3, 4]
    assert df["b"].to_list() == ["x", "y", "p", "q"]


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_reads_renamed_added_file(tmp_path):
    """DuckDB reads a renamed-column added file via the same name mapping."""
    metadata_path, data_path, _ = _make_added_file(tmp_path)
    alter_ducklake_rename_column(
        metadata_path, "events", "a", "id", data_path=data_path,
    )

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{metadata_path}' AS d "
        f"(DATA_PATH '{data_path}')"
    )
    rows = con.execute("SELECT id, b FROM d.events ORDER BY id").fetchall()
    con.close()
    assert rows == [(1, "x"), (2, "y")]
