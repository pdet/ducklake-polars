"""Configuration / settings tests for DuckLake catalog.

Covers:
- Data inlining threshold behaviour (small data inlined, large data → Parquet)
- Reading tables with both inlined and Parquet data
- Different data paths
"""

from __future__ import annotations

import os

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    read_ducklake,
    write_ducklake,
    create_ducklake_table,
    scan_ducklake,
)


# ------------------------------------------------------------------
# Data inlining threshold
# ------------------------------------------------------------------


class TestDataInliningThreshold:
    """Verify that data below the inlining row limit is inlined,
    while data above goes to Parquet files."""

    def test_small_data_is_inlined(self, make_write_catalog):
        """Insert below threshold → no Parquet files, data in inlined tables."""
        cat = make_write_catalog(inline=True, inline_limit=100)

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(
            df, cat.metadata_path, "small",
            data_path=cat.data_path,
            data_inlining_row_limit=100,
        )

        # No Parquet data files
        row = cat.query_one("SELECT COUNT(*) FROM ducklake_data_file")
        assert row[0] == 0

        # Inlined data table exists
        inlined = cat.query_all("SELECT table_name FROM ducklake_inlined_data_tables")
        assert len(inlined) >= 1

        # Data reads back correctly
        result = read_ducklake(cat.metadata_path, "small", data_path=cat.data_path)
        assert_frame_equal(result.sort("a"), df.sort("a"))

    def test_large_data_goes_to_parquet(self, make_write_catalog):
        """Insert above threshold → data written to Parquet file."""
        cat = make_write_catalog(inline=True, inline_limit=10)

        # 50 rows, well above the 10-row inline limit
        df = pl.DataFrame({
            "id": list(range(50)),
            "val": [f"row_{i}" for i in range(50)],
        })
        write_ducklake(
            df, cat.metadata_path, "large",
            data_path=cat.data_path,
            data_inlining_row_limit=10,
        )

        # Should have Parquet data file(s)
        row = cat.query_one("SELECT COUNT(*) FROM ducklake_data_file")
        assert row[0] >= 1

        # Data reads back correctly
        result = read_ducklake(cat.metadata_path, "large", data_path=cat.data_path)
        assert result.shape[0] == 50
        result = result.sort("id")
        assert result["id"].to_list() == list(range(50))

    def test_mixed_inlined_and_parquet(self, make_write_catalog):
        """First insert small (inlined), then large (Parquet) — both readable."""
        cat = make_write_catalog(inline=True, inline_limit=20)

        # Small insert → inlined
        df_small = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(
            df_small, cat.metadata_path, "mixed",
            data_path=cat.data_path,
            data_inlining_row_limit=20,
        )

        # Large insert → Parquet
        df_large = pl.DataFrame({
            "a": list(range(100, 200)),
            "b": [f"v{i}" for i in range(100, 200)],
        })
        write_ducklake(
            df_large, cat.metadata_path, "mixed",
            mode="append",
            data_path=cat.data_path,
            data_inlining_row_limit=20,
        )

        result = read_ducklake(cat.metadata_path, "mixed", data_path=cat.data_path)
        assert result.shape[0] == 103  # 3 inlined + 100 Parquet

        # Check both portions present
        small_rows = result.filter(pl.col("a") < 10).sort("a")
        assert small_rows["a"].to_list() == [1, 2, 3]

        large_rows = result.filter(pl.col("a") >= 100).sort("a")
        assert large_rows.shape[0] == 100

    def test_inlining_disabled_always_parquet(self, make_write_catalog):
        """With inlining disabled (limit=0), even tiny inserts go to Parquet."""
        cat = make_write_catalog(inline=False)

        df = pl.DataFrame({"a": [1], "b": ["only"]})
        write_ducklake(
            df, cat.metadata_path, "noninline",
            data_path=cat.data_path,
            data_inlining_row_limit=0,
        )

        # Should have a Parquet file
        row = cat.query_one("SELECT COUNT(*) FROM ducklake_data_file")
        assert row[0] >= 1

        result = read_ducklake(cat.metadata_path, "noninline", data_path=cat.data_path)
        assert result.shape[0] == 1
        assert result["a"][0] == 1


# ------------------------------------------------------------------
# Different data paths
# ------------------------------------------------------------------


class TestDataPaths:
    """Test that data_path override works correctly."""

    def test_custom_data_path(self, tmp_path, make_write_catalog):
        """Writing with a custom data path stores files in that directory."""
        cat = make_write_catalog(inline=False)

        df = pl.DataFrame({"x": [10, 20, 30]})
        write_ducklake(
            df, cat.metadata_path, "pathtest",
            data_path=cat.data_path,
            data_inlining_row_limit=0,
        )

        # Data directory should have files
        data_files = []
        for root, dirs, files in os.walk(cat.data_path):
            for f in files:
                if f.endswith(".parquet"):
                    data_files.append(os.path.join(root, f))
        assert len(data_files) >= 1

        # Can read with explicit data_path
        result = read_ducklake(
            cat.metadata_path, "pathtest", data_path=cat.data_path
        )
        assert result.shape[0] == 3
        assert sorted(result["x"].to_list()) == [10, 20, 30]

    def test_data_path_override_on_read(self, tmp_path, make_write_catalog):
        """data_path override on read allows reading from a relocated catalog."""
        cat = make_write_catalog(inline=False)

        df = pl.DataFrame({"val": [100, 200]})
        write_ducklake(
            df, cat.metadata_path, "relocated",
            data_path=cat.data_path,
            data_inlining_row_limit=0,
        )

        # Read with explicit data_path override (same location, but explicit)
        result = read_ducklake(
            cat.metadata_path, "relocated", data_path=cat.data_path
        )
        assert result.shape[0] == 2
        assert sorted(result["val"].to_list()) == [100, 200]

    def test_scan_with_data_path(self, make_write_catalog):
        """scan_ducklake works with data_path override."""
        cat = make_write_catalog(inline=False)

        df = pl.DataFrame({"k": [1, 2, 3, 4, 5], "v": ["a", "b", "c", "d", "e"]})
        write_ducklake(
            df, cat.metadata_path, "scanpath",
            data_path=cat.data_path,
            data_inlining_row_limit=0,
        )

        result = (
            scan_ducklake(cat.metadata_path, "scanpath", data_path=cat.data_path)
            .filter(pl.col("k") > 3)
            .collect()
        )
        result = result.sort("k")
        assert result["k"].to_list() == [4, 5]
        assert result["v"].to_list() == ["d", "e"]
