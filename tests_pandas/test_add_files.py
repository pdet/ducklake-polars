"""Pandas parity tests for add_files_ducklake."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ducklake_pandas import (
    add_files_ducklake,
    create_ducklake_table,
    read_ducklake,
)


def test_pandas_add_files_then_read(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")

    create_ducklake_table(
        path, "t",
        {"a": "int64", "b": "string"},
        data_path=data_path,
    )

    external = str(tmp_path / "external.parquet")
    pq.write_table(
        pa.table({"a": [1, 2], "b": ["x", "y"]}),
        external,
    )
    new_snap = add_files_ducklake(
        path, "t", [external], data_path=data_path,
    )
    assert new_snap >= 0

    df = read_ducklake(path, "t", data_path=data_path)
    df = df.sort_values("a").reset_index(drop=True)
    assert df["a"].tolist() == [1, 2]
    assert df["b"].tolist() == ["x", "y"]
