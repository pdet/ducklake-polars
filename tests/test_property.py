"""Property-based tests for ducklake-polars using Hypothesis.

Stress-tests the roundtrip: write random data → read it back → verify it matches.
"""

from __future__ import annotations

import os
import string
import tempfile

import polars as pl
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from ducklake_polars import (
    alter_ducklake_add_column,
    alter_ducklake_set_partitioned_by,
    delete_ducklake,
    read_ducklake,
    write_ducklake,
)
from ducklake_polars._catalog_api import DuckLakeCatalog


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

col_names = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=10)
int_values = st.integers(min_value=-(2**31), max_value=2**31 - 1)
str_values = st.text(min_size=0, max_size=100)
float_values = st.floats(allow_nan=False, allow_infinity=False)
bool_values = st.booleans()
row_counts = st.integers(min_value=1, max_value=100)


def _make_int_df(n_rows: int, col_name: str, values: list[int]) -> pl.DataFrame:
    """Build a single-column Int32 DataFrame."""
    return pl.DataFrame({col_name: pl.Series(values, dtype=pl.Int32)})


def _make_mixed_df(
    n_rows: int,
    int_vals: list[int],
    str_vals: list[str],
    float_vals: list[float],
    bool_vals: list[bool],
) -> pl.DataFrame:
    """Build a multi-column DataFrame with mixed types."""
    return pl.DataFrame(
        {
            "icol": pl.Series(int_vals, dtype=pl.Int32),
            "scol": str_vals,
            "fcol": pl.Series(float_vals, dtype=pl.Float64),
            "bcol": bool_vals,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TmpCatalog:
    """Context manager that provides a fresh DuckLake catalog in a temp dir.

    Initialises the catalog via DuckDB's ducklake extension so the metadata
    tables are created before we attempt any Python-side writes.
    """

    def __init__(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def __enter__(self) -> "_TmpCatalog":
        import duckdb

        self.dir = self._tmpdir.__enter__()
        self.metadata = os.path.join(self.dir, "test.ducklake")
        self.data = os.path.join(self.dir, "data")
        os.makedirs(self.data, exist_ok=True)

        # Bootstrap the catalog using DuckDB
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{self.metadata}' AS ducklake "
            f"(DATA_PATH '{self.data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.close()

        return self

    def __exit__(self, *args):
        self._tmpdir.__exit__(*args)


# ---------------------------------------------------------------------------
# 1. Roundtrip integrity
# ---------------------------------------------------------------------------


class TestRoundtripIntegrity:
    """For any DataFrame with supported types, write then read returns the same data."""

    @given(
        n_rows=row_counts,
        data=st.data(),
    )
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_int_roundtrip(self, n_rows: int, data):
        values = data.draw(st.lists(int_values, min_size=n_rows, max_size=n_rows))
        df = _make_int_df(n_rows, "val", values)

        with _TmpCatalog() as cat:
            write_ducklake(df, cat.metadata, "t", data_path=cat.data)
            result = read_ducklake(cat.metadata, "t", data_path=cat.data)
            assert_frame_equal(result.sort("val"), df.sort("val"))

    @given(
        n_rows=row_counts,
        data=st.data(),
    )
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_string_roundtrip(self, n_rows: int, data):
        values = data.draw(st.lists(str_values, min_size=n_rows, max_size=n_rows))
        df = pl.DataFrame({"val": values})

        with _TmpCatalog() as cat:
            write_ducklake(df, cat.metadata, "t", data_path=cat.data)
            result = read_ducklake(cat.metadata, "t", data_path=cat.data)
            assert_frame_equal(result.sort("val"), df.sort("val"))

    @given(
        n_rows=row_counts,
        data=st.data(),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_mixed_roundtrip(self, n_rows: int, data):
        int_vals = data.draw(st.lists(int_values, min_size=n_rows, max_size=n_rows))
        str_vals = data.draw(st.lists(str_values, min_size=n_rows, max_size=n_rows))
        float_vals = data.draw(
            st.lists(float_values, min_size=n_rows, max_size=n_rows)
        )
        bool_vals = data.draw(st.lists(bool_values, min_size=n_rows, max_size=n_rows))

        df = _make_mixed_df(n_rows, int_vals, str_vals, float_vals, bool_vals)

        with _TmpCatalog() as cat:
            write_ducklake(df, cat.metadata, "t", data_path=cat.data)
            result = read_ducklake(cat.metadata, "t", data_path=cat.data)
            assert_frame_equal(
                result.sort("icol", "scol"), df.sort("icol", "scol")
            )

    @given(
        n_rows=row_counts,
        data=st.data(),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_bool_roundtrip(self, n_rows: int, data):
        values = data.draw(st.lists(bool_values, min_size=n_rows, max_size=n_rows))
        df = pl.DataFrame({"val": values})

        with _TmpCatalog() as cat:
            write_ducklake(df, cat.metadata, "t", data_path=cat.data)
            result = read_ducklake(cat.metadata, "t", data_path=cat.data)
            # Sort both consistently
            df_sorted = df.with_row_index("_idx").sort("val", "_idx").drop("_idx")
            result_sorted = result.with_row_index("_idx").sort("val", "_idx").drop("_idx")
            assert result_sorted.shape == df_sorted.shape
            assert result_sorted["val"].to_list() == df_sorted["val"].to_list()


# ---------------------------------------------------------------------------
# 2. Append consistency
# ---------------------------------------------------------------------------


class TestAppendConsistency:
    """Writing in multiple appends produces same result as writing once."""

    @given(
        n_rows=st.integers(min_value=2, max_value=50),
        data=st.data(),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_append_equals_single_write(self, n_rows: int, data):
        values = data.draw(st.lists(int_values, min_size=n_rows, max_size=n_rows))
        df = _make_int_df(n_rows, "val", values)
        split = n_rows // 2
        df1 = df.head(split)
        df2 = df.tail(n_rows - split)

        # Single write
        with _TmpCatalog() as cat_single:
            write_ducklake(df, cat_single.metadata, "t", data_path=cat_single.data)
            single = read_ducklake(cat_single.metadata, "t", data_path=cat_single.data)

        # Two appends
        with _TmpCatalog() as cat_append:
            write_ducklake(df1, cat_append.metadata, "t", data_path=cat_append.data)
            write_ducklake(
                df2, cat_append.metadata, "t", mode="append", data_path=cat_append.data
            )
            appended = read_ducklake(
                cat_append.metadata, "t", data_path=cat_append.data
            )

        assert_frame_equal(single.sort("val"), appended.sort("val"))


# ---------------------------------------------------------------------------
# 3. Delete correctness
# ---------------------------------------------------------------------------


class TestDeleteCorrectness:
    """After delete, remaining rows don't match the predicate."""

    @given(
        n_rows=st.integers(min_value=5, max_value=80),
        threshold=int_values,
        data=st.data(),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_delete_removes_matching_rows(self, n_rows: int, threshold: int, data):
        values = data.draw(st.lists(int_values, min_size=n_rows, max_size=n_rows))
        df = _make_int_df(n_rows, "val", values)

        with _TmpCatalog() as cat:
            write_ducklake(df, cat.metadata, "t", data_path=cat.data)
            delete_ducklake(
                cat.metadata, "t", pl.col("val") > threshold, data_path=cat.data
            )
            result = read_ducklake(cat.metadata, "t", data_path=cat.data)

            # No remaining row should match the delete predicate
            assert result.filter(pl.col("val") > threshold).height == 0

            # Remaining rows should be exactly the ones <= threshold
            expected = df.filter(pl.col("val") <= threshold)
            assert_frame_equal(result.sort("val"), expected.sort("val"))


# ---------------------------------------------------------------------------
# 4. Schema evolution roundtrip
# ---------------------------------------------------------------------------


class TestSchemaEvolutionRoundtrip:
    """Add column, insert, read — new column has nulls for old rows."""

    @given(
        n_rows=st.integers(min_value=1, max_value=50),
        new_col=col_names,
        data=st.data(),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_add_column_nulls_old_rows(self, n_rows: int, new_col: str, data):
        # Avoid column name collision
        if new_col == "val":
            new_col = "extra"

        values = data.draw(st.lists(int_values, min_size=n_rows, max_size=n_rows))
        df = _make_int_df(n_rows, "val", values)

        with _TmpCatalog() as cat:
            write_ducklake(df, cat.metadata, "t", data_path=cat.data)
            alter_ducklake_add_column(
                cat.metadata, "t", new_col, pl.Int32(), data_path=cat.data
            )

            # Insert new row with the new column populated
            new_row = pl.DataFrame(
                {
                    "val": pl.Series([42], dtype=pl.Int32),
                    new_col: pl.Series([99], dtype=pl.Int32),
                }
            )
            write_ducklake(
                new_row, cat.metadata, "t", mode="append", data_path=cat.data
            )

            result = read_ducklake(cat.metadata, "t", data_path=cat.data)

            # Old rows have null for the new column
            old_rows = result.filter(pl.col("val") != 42)
            assert old_rows[new_col].null_count() == old_rows.height

            # New row has the value
            new_rows = result.filter(pl.col("val") == 42)
            assert new_rows[new_col].to_list() == [99]


# ---------------------------------------------------------------------------
# 5. Partition roundtrip
# ---------------------------------------------------------------------------


class TestPartitionRoundtrip:
    """Partitioned writes then reads return all data."""

    @given(
        data=st.data(),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_partitioned_roundtrip(self, data):
        # Generate data with a partition column (small cardinality)
        n_rows = data.draw(st.integers(min_value=5, max_value=50))
        part_vals = data.draw(
            st.lists(
                st.sampled_from(["a", "b", "c"]),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        int_vals = data.draw(st.lists(int_values, min_size=n_rows, max_size=n_rows))

        df = pl.DataFrame(
            {
                "part": part_vals,
                "val": pl.Series(int_vals, dtype=pl.Int32),
            }
        )

        with _TmpCatalog() as cat:
            # Create table without partitioning first, then set it
            write_ducklake(df, cat.metadata, "t", data_path=cat.data)
            # Overwrite with partition
            alter_ducklake_set_partitioned_by(
                cat.metadata, "t", ["part"], data_path=cat.data
            )
            # Append more data (now partitioned)
            extra = pl.DataFrame(
                {
                    "part": ["a"],
                    "val": pl.Series([999], dtype=pl.Int32),
                }
            )
            write_ducklake(
                extra, cat.metadata, "t", mode="append", data_path=cat.data
            )

            result = read_ducklake(cat.metadata, "t", data_path=cat.data)

            # All original data + extra row should be present
            assert result.height == n_rows + 1
            assert 999 in result["val"].to_list()


# ---------------------------------------------------------------------------
# 6. Idempotent metadata
# ---------------------------------------------------------------------------


class TestIdempotentMetadata:
    """Reading catalog info multiple times returns same result."""

    @given(
        n_rows=st.integers(min_value=1, max_value=30),
        data=st.data(),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_catalog_snapshots_idempotent(self, n_rows: int, data):
        values = data.draw(st.lists(int_values, min_size=n_rows, max_size=n_rows))
        df = _make_int_df(n_rows, "val", values)

        with _TmpCatalog() as cat:
            write_ducklake(df, cat.metadata, "t", data_path=cat.data)

            with DuckLakeCatalog(cat.metadata, data_path=cat.data) as api:
                snap1 = api.snapshots()
                snap2 = api.snapshots()
                assert_frame_equal(snap1, snap2)

                info1 = api.table_info()
                info2 = api.table_info()
                assert_frame_equal(info1, info2)

                files1 = api.list_files("t")
                files2 = api.list_files("t")
                assert_frame_equal(files1, files2)

    @given(
        n_rows=st.integers(min_value=1, max_value=30),
        data=st.data(),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_read_idempotent(self, n_rows: int, data):
        values = data.draw(st.lists(int_values, min_size=n_rows, max_size=n_rows))
        df = _make_int_df(n_rows, "val", values)

        with _TmpCatalog() as cat:
            write_ducklake(df, cat.metadata, "t", data_path=cat.data)

            r1 = read_ducklake(cat.metadata, "t", data_path=cat.data)
            r2 = read_ducklake(cat.metadata, "t", data_path=cat.data)
            assert_frame_equal(r1, r2)
