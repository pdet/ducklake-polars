"""Tests for add_files — registering existing Parquet files into DuckLake."""

from __future__ import annotations

import os

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    add_files_ducklake,
    create_ducklake_table,
    read_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_parquet_file(path: str, df: pl.DataFrame) -> str:
    """Write a Polars DataFrame to a Parquet file and return its path."""
    arrow_table = df.to_arrow()
    pq.write_table(arrow_table, path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAddFiles:
    """Test add_files functionality."""

    def test_add_single_file(self, ducklake_catalog):
        """Add a single Parquet file to an existing table."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (a BIGINT, b VARCHAR)")
        catalog.close()

        # Write a Parquet file outside the catalog
        parquet_path = os.path.join(
            os.path.dirname(catalog.metadata_path)
            if catalog.backend == "sqlite"
            else str(ducklake_catalog.data_path),
            "external_data.parquet",
        )
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        _write_parquet_file(parquet_path, df)

        # Add the file
        add_files_ducklake(
            catalog.metadata_path,
            "t1",
            [parquet_path],
            data_path=catalog.data_path,
        )

        # Read back and verify
        result = read_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path
        )
        assert len(result) == 3
        assert sorted(result["a"].to_list()) == [1, 2, 3]

    def test_add_multiple_files(self, ducklake_catalog):
        """Add multiple Parquet files to an existing table."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (x BIGINT, y DOUBLE)")
        catalog.close()

        base_dir = os.path.dirname(catalog.metadata_path) if catalog.backend == "sqlite" else catalog.data_path

        # Write two Parquet files
        df1 = pl.DataFrame({"x": [1, 2], "y": [1.0, 2.0]})
        df2 = pl.DataFrame({"x": [3, 4], "y": [3.0, 4.0]})
        path1 = _write_parquet_file(os.path.join(base_dir, "part1.parquet"), df1)
        path2 = _write_parquet_file(os.path.join(base_dir, "part2.parquet"), df2)

        add_files_ducklake(
            catalog.metadata_path,
            "t1",
            [path1, path2],
            data_path=catalog.data_path,
        )

        result = read_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path
        )
        assert len(result) == 4
        assert sorted(result["x"].to_list()) == [1, 2, 3, 4]

    def test_read_after_add_files(self, ducklake_catalog):
        """Data is visible after adding files."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (id BIGINT, name VARCHAR)")
        catalog.close()

        base_dir = os.path.dirname(catalog.metadata_path) if catalog.backend == "sqlite" else catalog.data_path

        df = pl.DataFrame({"id": [10, 20, 30], "name": ["alice", "bob", "carol"]})
        parquet_path = _write_parquet_file(
            os.path.join(base_dir, "people.parquet"), df
        )

        add_files_ducklake(
            catalog.metadata_path,
            "t1",
            [parquet_path],
            data_path=catalog.data_path,
        )

        result = read_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path
        )
        expected = df.sort("id")
        actual = result.sort("id")
        assert_frame_equal(actual, expected)

    def test_schema_mismatch_raises_error(self, ducklake_catalog):
        """Adding a file with mismatched schema raises ValueError."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (a BIGINT, b VARCHAR)")
        catalog.close()

        base_dir = os.path.dirname(catalog.metadata_path) if catalog.backend == "sqlite" else catalog.data_path

        # Parquet file with different columns
        df_bad = pl.DataFrame({"a": [1], "c": [2]})
        bad_path = _write_parquet_file(
            os.path.join(base_dir, "bad.parquet"), df_bad
        )

        with pytest.raises(ValueError, match="Schema mismatch"):
            add_files_ducklake(
                catalog.metadata_path,
                "t1",
                [bad_path],
                data_path=catalog.data_path,
            )

    def test_add_files_then_insert(self, ducklake_catalog):
        """Adding files then inserting more data — both visible."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.close()

        base_dir = os.path.dirname(catalog.metadata_path) if catalog.backend == "sqlite" else catalog.data_path

        # Add a parquet file
        df_file = pl.DataFrame({"v": [100, 200]})
        parquet_path = _write_parquet_file(
            os.path.join(base_dir, "added.parquet"), df_file
        )
        add_files_ducklake(
            catalog.metadata_path,
            "t1",
            [parquet_path],
            data_path=catalog.data_path,
        )

        # Insert more data via write_ducklake
        df_insert = pl.DataFrame({"v": [300, 400]})
        write_ducklake(
            df_insert,
            catalog.metadata_path,
            "t1",
            mode="append",
            data_path=catalog.data_path,
        )

        result = read_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path
        )
        assert len(result) == 4
        assert sorted(result["v"].to_list()) == [100, 200, 300, 400]
