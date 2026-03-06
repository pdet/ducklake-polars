"""Tests for DuckLakeStreamWriter — buffered streaming ingestion."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import DuckLakeStreamWriter, read_ducklake


@pytest.fixture
def write_cat(make_write_catalog):
    return make_write_catalog()


class TestStreamWriterBasic:
    """Basic streaming writer behavior."""

    def test_context_manager_writes(self, write_cat):
        cat = write_cat
        with DuckLakeStreamWriter(cat.metadata_path, "t",
                                  data_path=cat.data_path,
                                  flush_threshold=100) as writer:
            for i in range(5):
                writer.append(pl.DataFrame({"id": [i]}))

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 5

    def test_auto_flush_at_threshold(self, write_cat):
        cat = write_cat
        with DuckLakeStreamWriter(cat.metadata_path, "t",
                                  data_path=cat.data_path,
                                  flush_threshold=10) as writer:
            # Write 25 rows in batches of 5
            for i in range(5):
                writer.append(pl.DataFrame({
                    "id": list(range(i * 5, (i + 1) * 5)),
                }))

            # Should have auto-flushed at least twice (10, 20)
            assert writer.flush_count >= 2

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 25

    def test_empty_append_ignored(self, write_cat):
        cat = write_cat
        with DuckLakeStreamWriter(cat.metadata_path, "t",
                                  data_path=cat.data_path,
                                  flush_threshold=100) as writer:
            writer.append(pl.DataFrame({"id": [1]}))
            writer.append(pl.DataFrame({"id": pl.Series([], dtype=pl.Int64)}))
            writer.append(pl.DataFrame({"id": [2]}))

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 2


class TestStreamWriterCompaction:
    """Auto-compaction on close."""

    def test_compact_on_close(self, write_cat):
        cat = write_cat
        with DuckLakeStreamWriter(cat.metadata_path, "t",
                                  data_path=cat.data_path,
                                  flush_threshold=5,
                                  compact_on_close=True) as writer:
            for i in range(20):
                writer.append(pl.DataFrame({"id": [i]}))

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 20

    def test_no_compact_when_disabled(self, write_cat):
        cat = write_cat
        with DuckLakeStreamWriter(cat.metadata_path, "t",
                                  data_path=cat.data_path,
                                  flush_threshold=5,
                                  compact_on_close=False) as writer:
            for i in range(20):
                writer.append(pl.DataFrame({"id": [i]}))

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 20


class TestStreamWriterProperties:
    """Writer properties: total_rows, flush_count, buffer_rows."""

    def test_properties(self, write_cat):
        cat = write_cat
        writer = DuckLakeStreamWriter(cat.metadata_path, "t",
                                      data_path=cat.data_path,
                                      flush_threshold=10)
        writer.append(pl.DataFrame({"id": list(range(7))}))
        assert writer.buffer_rows == 7
        assert writer.flush_count == 0
        assert writer.total_rows == 7

        writer.append(pl.DataFrame({"id": list(range(7, 15))}))
        # Should have auto-flushed (15 > 10)
        assert writer.flush_count >= 1
        assert writer.total_rows == 15

        writer.close()


class TestStreamWriterSchemaEvolution:
    """Streaming writer with schema_evolution='merge'."""

    def test_evolving_schema(self, write_cat):
        cat = write_cat
        with DuckLakeStreamWriter(cat.metadata_path, "t",
                                  data_path=cat.data_path,
                                  flush_threshold=5,
                                  schema_evolution="merge") as writer:
            # First batch: id only
            writer.append(pl.DataFrame({"id": [1, 2, 3, 4, 5]}))

            # Second batch: id + name (new column)
            writer.append(pl.DataFrame({"id": [6, 7, 8, 9, 10], "name": ["a", "b", "c", "d", "e"]}))

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 10
        assert "name" in result.columns
        # Old rows have NULL for name
        old = result.filter(pl.col("id") <= 5)
        assert old["name"].null_count() == 5
