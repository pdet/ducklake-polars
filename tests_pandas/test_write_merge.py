"""Tests for ducklake-pandas MERGE and CREATE TABLE AS support."""

from __future__ import annotations

import duckdb
import pandas as pd
import numpy as np
import pytest

from ducklake_pandas import (
    create_table_as_ducklake,
    merge_ducklake,
    read_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# CREATE TABLE AS
# ---------------------------------------------------------------------------


class TestCreateTableAs:
    """Test create_table_as_ducklake (single-snapshot create + insert)."""

    def test_create_table_as_basic(self, make_write_catalog):
        """Create a table from a DataFrame in one operation."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        create_table_as_ducklake(df, cat.metadata_path, "test")

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["x", "y", "z"]

    def test_create_table_as_single_snapshot(self, make_write_catalog):
        """Only one snapshot should be created (not create + insert)."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3]})

        snap_before = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot"
        )[0]

        create_table_as_ducklake(df, cat.metadata_path, "test")

        snap_after = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot"
        )[0]
        assert snap_after == snap_before + 1

    def test_create_table_as_empty_df(self, make_write_catalog):
        """Create a table with an empty DataFrame (schema only, no data)."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": pd.array([], dtype="int64"), "b": pd.Series([], dtype="string")})
        create_table_as_ducklake(df, cat.metadata_path, "test")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0
        assert len(result.columns) == 2

    def test_create_table_as_already_exists(self, make_write_catalog):
        """Should raise if the table already exists."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})
        create_table_as_ducklake(df, cat.metadata_path, "test")

        with pytest.raises(ValueError, match="already exists"):
            create_table_as_ducklake(df, cat.metadata_path, "test")

    def test_create_table_as_multiple_types(self, make_write_catalog):
        """Create a table with various column types."""
        cat = make_write_catalog()
        df = pd.DataFrame({
            "i": [1, 2, 3],
            "f": [1.5, 2.5, 3.5],
            "s": ["a", "b", "c"],
            "b": [True, False, True],
        })
        create_table_as_ducklake(df, cat.metadata_path, "multi")

        result = read_ducklake(cat.metadata_path, "multi").sort_values("i").reset_index(drop=True)
        assert result["i"].tolist() == [1, 2, 3]
        assert result["f"].tolist() == [1.5, 2.5, 3.5]
        assert result["s"].tolist() == ["a", "b", "c"]
        assert result["b"].tolist() == [True, False, True]

    def test_create_table_as_duckdb_reads(self, make_write_catalog):
        """DuckDB can read a table created with create_table_as_ducklake."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [10, 20, 30], "b": ["x", "y", "z"]})
        create_table_as_ducklake(df, cat.metadata_path, "test")

        pdf = cat.read_with_duckdb("test").sort_values("a").reset_index(drop=True)
        assert pdf["a"].tolist() == [10, 20, 30]
        assert pdf["b"].tolist() == ["x", "y", "z"]

    def test_create_table_as_then_append(self, make_write_catalog):
        """Create a table, then append more data."""
        cat = make_write_catalog()
        df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        create_table_as_ducklake(df1, cat.metadata_path, "test")

        df2 = pd.DataFrame({"a": [3, 4], "b": ["p", "q"]})
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3, 4]
        assert result["b"].tolist() == ["x", "y", "p", "q"]


# ---------------------------------------------------------------------------
# MERGE: basic operations
# ---------------------------------------------------------------------------


class TestMergeBasic:
    """Test basic merge operations."""

    def test_merge_insert_only(self, make_write_catalog):
        """Merge with no matches should insert all source rows."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        source = pd.DataFrame({"id": [3, 4], "val": ["c", "d"]})
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_not_matched_insert=True,
        )
        assert updated == 0
        assert inserted == 2

        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["id"].tolist() == [1, 2, 3, 4]
        assert result["val"].tolist() == ["a", "b", "c", "d"]

    def test_merge_update_only(self, make_write_catalog):
        """Merge with all matches should update, no inserts."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        source = pd.DataFrame({"id": [2, 3], "val": ["B", "C"]})
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_matched_update=True,
            when_not_matched_insert=False,
        )
        assert updated == 2
        assert inserted == 0

        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["id"].tolist() == [1, 2, 3]
        assert result["val"].tolist() == ["a", "B", "C"]

    def test_merge_upsert(self, make_write_catalog):
        """Classic upsert: update matched, insert unmatched."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        source = pd.DataFrame({"id": [2, 4], "val": ["B", "D"]})
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_matched_update=True,
            when_not_matched_insert=True,
        )
        assert updated == 1
        assert inserted == 1

        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["id"].tolist() == [1, 2, 3, 4]
        assert result["val"].tolist() == ["a", "B", "c", "D"]

    def test_merge_no_op(self, make_write_catalog):
        """Merge with no changes creates no snapshot."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        snap_before = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot"
        )[0]

        source = pd.DataFrame({"id": [1, 2], "val": ["X", "Y"]})
        # No update (when_matched_update=None), and keys already exist so
        # when_not_matched_insert finds nothing
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_matched_update=None,
            when_not_matched_insert=True,
        )
        assert updated == 0
        assert inserted == 0

        snap_after = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot"
        )[0]
        assert snap_after == snap_before

    def test_merge_with_dict_update(self, make_write_catalog):
        """Merge with dict-based update: set specific column values."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"], "score": [10, 20, 30]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        source = pd.DataFrame({"id": [2, 3], "val": ["B", "C"], "score": [200, 300]})
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_matched_update={"val": "MERGED"},
            when_not_matched_insert=False,
        )
        assert updated == 2
        assert inserted == 0

        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["val"].tolist() == ["a", "MERGED", "MERGED"]
        # score should be unchanged (dict update only modified val)
        assert result["score"].tolist() == [10, 20, 30]


# ---------------------------------------------------------------------------
# MERGE: multi-key columns
# ---------------------------------------------------------------------------


class TestMergeMultiKey:
    """Test merge with composite key columns."""

    def test_merge_composite_key(self, make_write_catalog):
        """Merge matching on two key columns."""
        cat = make_write_catalog()
        df = pd.DataFrame({
            "k1": [1, 1, 2, 2],
            "k2": ["a", "b", "a", "b"],
            "val": ["v1", "v2", "v3", "v4"],
        })
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        source = pd.DataFrame({
            "k1": [1, 2, 3],
            "k2": ["b", "c", "a"],
            "val": ["V2", "V_new", "V_new2"],
        })
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, ["k1", "k2"],
            when_matched_update=True,
            when_not_matched_insert=True,
        )
        assert updated == 1  # (1, "b") matched
        assert inserted == 2  # (2, "c") and (3, "a") unmatched

        result = read_ducklake(cat.metadata_path, "test").sort_values(["k1", "k2"]).reset_index(drop=True)
        assert result["val"].tolist() == ["v1", "V2", "v3", "v4", "V_new", "V_new2"]


# ---------------------------------------------------------------------------
# MERGE: multi-file
# ---------------------------------------------------------------------------


class TestMergeMultiFile:
    """Test merge across multiple data files."""

    def test_merge_across_two_files(self, make_write_catalog):
        """Merge where target spans two data files."""
        cat = make_write_catalog()
        df1 = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        df2 = pd.DataFrame({"id": [4, 5, 6], "val": ["d", "e", "f"]})
        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        source = pd.DataFrame({"id": [2, 5, 7], "val": ["B", "E", "G"]})
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_matched_update=True,
            when_not_matched_insert=True,
        )
        assert updated == 2
        assert inserted == 1

        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["id"].tolist() == [1, 2, 3, 4, 5, 6, 7]
        assert result["val"].tolist() == ["a", "B", "c", "d", "E", "f", "G"]


# ---------------------------------------------------------------------------
# MERGE: time travel
# ---------------------------------------------------------------------------


class TestMergeTimeTravel:
    """Test time travel works after merge."""

    def test_time_travel_before_after_merge(self, make_write_catalog):
        """Read at snapshot before and after merge."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        snap_before = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        merge_ducklake(
            cat.metadata_path, "test",
            pd.DataFrame({"id": [2, 4], "val": ["B", "D"]}),
            "id",
            when_matched_update=True,
        )

        # Latest: merged
        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["id"].tolist() == [1, 2, 3, 4]
        assert result["val"].tolist() == ["a", "B", "c", "D"]

        # Before merge: original
        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before
        ).sort_values("id").reset_index(drop=True)
        assert result_old["id"].tolist() == [1, 2, 3]
        assert result_old["val"].tolist() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# MERGE: DuckDB interop
# ---------------------------------------------------------------------------


class TestMergeDuckDBInterop:
    """Verify merged catalogs are readable by DuckDB."""

    def test_pandas_merge_duckdb_reads(self, make_write_catalog):
        """Write + merge with ducklake-pandas, read with DuckDB."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        merge_ducklake(
            cat.metadata_path, "test",
            pd.DataFrame({"id": [2, 4], "val": ["B", "D"]}),
            "id",
            when_matched_update=True,
        )

        pdf = cat.read_with_duckdb("test").sort_values("id").reset_index(drop=True)
        assert pdf["id"].tolist() == [1, 2, 3, 4]
        assert pdf["val"].tolist() == ["a", "B", "c", "D"]

    def test_duckdb_write_pandas_merge_duckdb_reads(self, make_write_catalog):
        """DuckDB creates data, pandas merges, DuckDB reads result."""
        cat = make_write_catalog()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        if cat.backend == "sqlite":
            attach_source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            attach_source = f"ducklake:postgres:{cat.metadata_path}"
        con.execute(
            f"ATTACH '{attach_source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (id INTEGER, val VARCHAR)")
        con.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'one'), (2, 'two'), (3, 'three')"
        )
        con.close()

        # DuckDB INTEGER maps to Int32 in Parquet — source must match
        source = pd.DataFrame({
            "id": pd.array([2, 3, 4], dtype="int32"),
            "val": ["TWO", "THREE", "FOUR"],
        })
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_matched_update=True,
        )
        assert updated == 2
        assert inserted == 1

        # Read with ducklake-pandas
        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["val"].tolist() == ["one", "TWO", "THREE", "FOUR"]

        # Read with DuckDB
        pdf = cat.read_with_duckdb("test").sort_values("id").reset_index(drop=True)
        assert pdf["val"].tolist() == ["one", "TWO", "THREE", "FOUR"]

    def test_create_table_as_duckdb_interop(self, make_write_catalog):
        """DuckDB can read a table created via create_table_as, then pandas can merge."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        create_table_as_ducklake(df, cat.metadata_path, "test")

        # Verify DuckDB can read
        pdf = cat.read_with_duckdb("test").sort_values("id").reset_index(drop=True)
        assert pdf["id"].tolist() == [1, 2]

        # Now merge
        source = pd.DataFrame({"id": [2, 3], "val": ["B", "C"]})
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_matched_update=True,
        )
        assert updated == 1
        assert inserted == 1

        # Both readers agree
        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        pdf = cat.read_with_duckdb("test").sort_values("id").reset_index(drop=True)
        assert result["val"].tolist() == ["a", "B", "C"]
        assert pdf["val"].tolist() == ["a", "B", "C"]


# ---------------------------------------------------------------------------
# MERGE: metadata verification
# ---------------------------------------------------------------------------


class TestMergeMetadata:
    """Verify merge metadata is correctly stored."""

    def test_merge_creates_single_snapshot(self, make_write_catalog):
        """Merge should create exactly one snapshot for the whole operation."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        snap_before = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot"
        )[0]

        merge_ducklake(
            cat.metadata_path, "test",
            pd.DataFrame({"id": [2, 4], "val": ["B", "D"]}),
            "id",
            when_matched_update=True,
        )

        snap_after = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot"
        )[0]
        assert snap_after == snap_before + 1

    def test_merge_snapshot_changes(self, make_write_catalog):
        """Changes record should have both insert and delete."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        merge_ducklake(
            cat.metadata_path, "test",
            pd.DataFrame({"id": [2, 4], "val": ["B", "D"]}),
            "id",
            when_matched_update=True,
        )

        row = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert row is not None
        assert "inserted_into_table" in row[0]
        assert "deleted_from_table" in row[0]

    def test_merge_delete_file_created(self, make_write_catalog):
        """Merge with updates should create delete files for matched rows."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        merge_ducklake(
            cat.metadata_path, "test",
            pd.DataFrame({"id": [2], "val": ["B"]}),
            "id",
            when_matched_update=True,
        )

        # Should have 1 delete file for the matched row
        row = cat.query_one("SELECT COUNT(*) FROM ducklake_delete_file")
        assert row[0] >= 1

        # Should have 2 data files: original + new insert
        row = cat.query_one("SELECT COUNT(*) FROM ducklake_data_file")
        assert row[0] == 2


# ---------------------------------------------------------------------------
# MERGE: edge cases
# ---------------------------------------------------------------------------


class TestMergeEdgeCases:
    """Edge cases for the merge path."""

    def test_merge_insert_only_no_update(self, make_write_catalog):
        """when_matched_update=None, when_not_matched_insert=True."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        source = pd.DataFrame({"id": [2, 3], "val": ["B", "C"]})
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_matched_update=None,
            when_not_matched_insert=True,
        )
        assert updated == 0
        assert inserted == 1

        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["id"].tolist() == [1, 2, 3]
        assert result["val"].tolist() == ["a", "b", "C"]

    def test_merge_update_all_rows(self, make_write_catalog):
        """Merge where all target rows match."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        source = pd.DataFrame({"id": [1, 2, 3], "val": ["X", "Y", "Z"]})
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_matched_update=True,
        )
        assert updated == 3
        assert inserted == 0

        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["val"].tolist() == ["X", "Y", "Z"]

    def test_merge_with_expr_update(self, make_write_catalog):
        """Merge with expression-based dict update."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        source = pd.DataFrame({"id": [2, 3], "val": [200, 300]})
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_matched_update={"val": lambda df: df["val"] + 1000},
            when_not_matched_insert=False,
        )
        assert updated == 2

        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["val"].tolist() == [10, 1020, 1030]

    def test_merge_preserves_unmatched_target(self, make_write_catalog):
        """Unmatched target rows should remain unchanged."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "val": ["a", "b", "c", "d", "e"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        source = pd.DataFrame({"id": [3], "val": ["C"]})
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_matched_update=True,
            when_not_matched_insert=False,
        )
        assert updated == 1
        assert inserted == 0

        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["val"].tolist() == ["a", "b", "C", "d", "e"]

    def test_merge_on_string_single_key(self, make_write_catalog):
        """Merge with 'on' passed as a single string (not list)."""
        cat = make_write_catalog()
        df = pd.DataFrame({"key": ["a", "b"], "val": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        source = pd.DataFrame({"key": ["b", "c"], "val": [20, 30]})
        updated, inserted = merge_ducklake(
            cat.metadata_path, "test", source, "key",
            when_matched_update=True,
        )
        assert updated == 1
        assert inserted == 1

        result = read_ducklake(cat.metadata_path, "test").sort_values("key").reset_index(drop=True)
        assert result["key"].tolist() == ["a", "b", "c"]
        assert result["val"].tolist() == [1, 20, 30]
