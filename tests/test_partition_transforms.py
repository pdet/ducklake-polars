"""Tests for non-identity partition transforms (year/month/day/hour).

DuckLake v1.0 spec: docs/stable/specification/tables/ducklake_partition_column.md
"""

from __future__ import annotations

import datetime
import os
import sqlite3

import duckdb
import polars as pl
import pyarrow as pa
import pytest

from ducklake_polars import (
    alter_ducklake_set_partitioned_by,
    read_ducklake,
    write_ducklake,
)


def _duckdb_supports_v10() -> bool:
    parts = duckdb.__version__.split(".")
    return len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (1, 5)


# ------------------------------------------------------------------
# Catalog metadata: transform string is persisted
# ------------------------------------------------------------------


class TestSetPartitionedByTransform:
    def _make_table(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "ts": [
                datetime.datetime(2024, 1, 15, 10, 0),
                datetime.datetime(2024, 6, 20, 14, 30),
                datetime.datetime(2025, 3, 1, 9, 0),
            ],
        })
        write_ducklake(df, path, "events", mode="error", data_path=data_path)
        return path, data_path

    def test_identity_transform_default(self, tmp_path):
        """Bare list of column names defaults to identity (back-compat)."""
        path, data_path = self._make_table(tmp_path)
        alter_ducklake_set_partitioned_by(
            path, "events", ["id"], data_path=data_path,
        )
        con = sqlite3.connect(path)
        try:
            row = con.execute(
                "SELECT pc.transform FROM ducklake_partition_column pc "
                "JOIN ducklake_partition_info pi ON pc.partition_id = pi.partition_id "
                "WHERE pi.end_snapshot IS NULL ORDER BY pc.partition_key_index"
            ).fetchall()
        finally:
            con.close()
        assert row == [("identity",)]

    def test_year_transform_recorded(self, tmp_path):
        path, data_path = self._make_table(tmp_path)
        alter_ducklake_set_partitioned_by(
            path, "events", [("ts", "year")], data_path=data_path,
        )
        con = sqlite3.connect(path)
        try:
            row = con.execute(
                "SELECT pc.transform FROM ducklake_partition_column pc "
                "JOIN ducklake_partition_info pi ON pc.partition_id = pi.partition_id "
                "WHERE pi.end_snapshot IS NULL"
            ).fetchall()
        finally:
            con.close()
        assert row == [("year",)]

    @pytest.mark.parametrize("transform", ["year", "month", "day", "hour"])
    def test_all_transforms_recorded(self, tmp_path, transform):
        path, data_path = self._make_table(tmp_path)
        alter_ducklake_set_partitioned_by(
            path, "events", [("ts", transform)], data_path=data_path,
        )
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
        assert row == (transform,)

    def test_invalid_transform_rejected(self, tmp_path):
        path, data_path = self._make_table(tmp_path)
        with pytest.raises(ValueError, match="Unsupported partition transform"):
            alter_ducklake_set_partitioned_by(
                path, "events", [("ts", "bucket")], data_path=data_path,
            )

    def test_year_on_non_timestamp_column_rejected(self, tmp_path):
        path, data_path = self._make_table(tmp_path)
        with pytest.raises(ValueError, match="requires a date/timestamp column"):
            alter_ducklake_set_partitioned_by(
                path, "events", [("id", "year")], data_path=data_path,
            )

    def test_hour_on_date_column_rejected(self, tmp_path):
        """``hour`` requires a timestamp, not a plain date."""
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df = pl.DataFrame({
            "d": [datetime.date(2024, 1, 1), datetime.date(2024, 6, 1)],
        })
        write_ducklake(df, path, "t", mode="error", data_path=data_path)

        with pytest.raises(ValueError, match="requires a timestamp column"):
            alter_ducklake_set_partitioned_by(
                path, "t", [("d", "hour")], data_path=data_path,
            )


# ------------------------------------------------------------------
# Insert path: data is partitioned by the transformed value
# ------------------------------------------------------------------


class TestPartitionedInsertWithTransform:
    def test_year_transform_groups_by_year(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")

        # Create a table, partition it by year(ts), then insert mixed years.
        df_init = pl.DataFrame({
            "v": [1],
            "ts": [datetime.datetime(2020, 1, 1)],
        })
        write_ducklake(df_init, path, "events", mode="error", data_path=data_path)

        alter_ducklake_set_partitioned_by(
            path, "events", [("ts", "year")], data_path=data_path,
        )

        df = pl.DataFrame({
            "v": [1, 2, 3, 4],
            "ts": [
                datetime.datetime(2024, 1, 15),
                datetime.datetime(2024, 12, 31),
                datetime.datetime(2025, 6, 1),
                datetime.datetime(2025, 7, 4),
            ],
        })
        write_ducklake(df, path, "events", mode="append", data_path=data_path)

        # Two new files (2024, 2025) — plus the original unpartitioned file.
        con = sqlite3.connect(path)
        try:
            new_files = con.execute(
                "SELECT data_file_id FROM ducklake_data_file "
                "WHERE partition_id IS NOT NULL AND end_snapshot IS NULL"
            ).fetchall()
            partition_values = sorted(
                con.execute(
                    "SELECT data_file_id, partition_value "
                    "FROM ducklake_file_partition_value "
                    "ORDER BY data_file_id"
                ).fetchall()
            )
        finally:
            con.close()
        assert len(new_files) == 2
        # Year transform stores (year - 1970) per the spec.
        # 2024 → 54, 2025 → 55
        assert {row[1] for row in partition_values} == {"54", "55"}

        # All original data still readable
        result = read_ducklake(path, "events", data_path=data_path)
        assert result.shape[0] == 5
        assert sorted(result["v"].to_list()) == [1, 1, 2, 3, 4]

    def test_month_transform_values(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df_init = pl.DataFrame({
            "v": [0],
            "ts": [datetime.datetime(1969, 12, 1)],
        })
        write_ducklake(df_init, path, "events", mode="error", data_path=data_path)

        alter_ducklake_set_partitioned_by(
            path, "events", [("ts", "month")], data_path=data_path,
        )

        df = pl.DataFrame({
            "v": [1, 2],
            "ts": [
                # 1970-01 → 0; 1970-02 → 1
                datetime.datetime(1970, 1, 15),
                datetime.datetime(1970, 2, 28),
            ],
        })
        write_ducklake(df, path, "events", mode="append", data_path=data_path)

        con = sqlite3.connect(path)
        try:
            partition_values = sorted(
                con.execute(
                    "SELECT partition_value FROM ducklake_file_partition_value"
                ).fetchall()
            )
        finally:
            con.close()
        assert {pv[0] for pv in partition_values} == {"0", "1"}

    def test_day_transform_values(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df_init = pl.DataFrame({
            "v": [0],
            "d": [datetime.date(1969, 12, 31)],
        })
        write_ducklake(df_init, path, "events", mode="error", data_path=data_path)

        alter_ducklake_set_partitioned_by(
            path, "events", [("d", "day")], data_path=data_path,
        )

        df = pl.DataFrame({
            "v": [1, 2],
            "d": [datetime.date(1970, 1, 1), datetime.date(1970, 1, 5)],
        })
        write_ducklake(df, path, "events", mode="append", data_path=data_path)

        con = sqlite3.connect(path)
        try:
            partition_values = sorted(
                con.execute(
                    "SELECT partition_value FROM ducklake_file_partition_value"
                ).fetchall()
            )
        finally:
            con.close()
        # Day transform: days from 1970-01-01
        assert {pv[0] for pv in partition_values} == {"0", "4"}

    def test_hour_transform_values(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df_init = pl.DataFrame({
            "v": [0],
            "ts": [datetime.datetime(1969, 12, 31, 23)],
        })
        write_ducklake(df_init, path, "events", mode="error", data_path=data_path)

        alter_ducklake_set_partitioned_by(
            path, "events", [("ts", "hour")], data_path=data_path,
        )

        df = pl.DataFrame({
            "v": [1, 2],
            "ts": [
                datetime.datetime(1970, 1, 1, 0, 30),  # hour 0
                datetime.datetime(1970, 1, 1, 5, 0),   # hour 5
            ],
        })
        write_ducklake(df, path, "events", mode="append", data_path=data_path)

        con = sqlite3.connect(path)
        try:
            partition_values = sorted(
                con.execute(
                    "SELECT partition_value FROM ducklake_file_partition_value"
                ).fetchall()
            )
        finally:
            con.close()
        assert {pv[0] for pv in partition_values} == {"0", "5"}


# ------------------------------------------------------------------
# DuckDB roundtrip — when DuckDB supports v1.0 catalogs
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_reads_year_partitioned_table(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")

    df = pl.DataFrame({
        "v": [1, 2, 3],
        "ts": [
            datetime.datetime(2023, 1, 1),
            datetime.datetime(2024, 1, 1),
            datetime.datetime(2025, 1, 1),
        ],
    })
    write_ducklake(df, path, "events", mode="error", data_path=data_path)
    alter_ducklake_set_partitioned_by(
        path, "events", [("ts", "year")], data_path=data_path,
    )
    write_ducklake(df, path, "events", mode="append", data_path=data_path)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{path}' AS d (DATA_PATH '{data_path}')"
    )
    rows = con.execute(
        "SELECT v FROM d.events ORDER BY v"
    ).fetchall()
    con.close()
    # 3 original + 3 partitioned = 6
    assert sorted(r[0] for r in rows) == [1, 1, 2, 2, 3, 3]


# ------------------------------------------------------------------
# Read-side: partition pruning on year/month/day/hour transforms
# ------------------------------------------------------------------


class TestPartitionPruning:
    """Filter pushdown should restrict the file set when querying a
    partitioned table — even with non-identity transforms.

    The polars dataset uses ``ducklake_file_partition_value`` to
    skip files whose transformed bucket cannot satisfy the predicate.
    """

    def _make_yearly(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        # Write three batches in distinct years.
        for year in (2023, 2024, 2025):
            write_ducklake(
                pl.DataFrame({
                    "id": [year],
                    "ts": [datetime.datetime(year, 6, 15, 12)],
                }),
                path, "events",
                mode="append" if year != 2023 else "error",
                data_path=data_path,
            )
            if year == 2023:
                alter_ducklake_set_partitioned_by(
                    path, "events", [("ts", "year")], data_path=data_path,
                )
        return path, data_path

    def test_year_partitioned_full_scan(self, tmp_path):
        path, data_path = self._make_yearly(tmp_path)
        df = read_ducklake(path, "events", data_path=data_path).sort("id")
        assert df["id"].to_list() == [2023, 2024, 2025]

    def test_year_partitioned_filter_returns_only_matching(self, tmp_path):
        path, data_path = self._make_yearly(tmp_path)
        df = (
            read_ducklake(path, "events", data_path=data_path)
            .filter(pl.col("ts") >= datetime.datetime(2024, 1, 1))
            .sort("id")
        )
        assert df["id"].to_list() == [2024, 2025]
