#!/usr/bin/env python3
"""Profile the DuckLake read path to find bottlenecks."""

import os
import shutil
import tempfile
import time

import duckdb
import numpy as np
import polars as pl

CATEGORIES = ["electronics", "books", "clothing", "food", "toys", "sports"]


def generate_df(n):
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "id": list(range(n)),
        "name": [f"item_{i}" for i in range(n)],
        "value": rng.uniform(0, 1000, n).tolist(),
        "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(n)],
        "ts": [f"2024-01-01T{(i % 24):02d}:{(i % 60):02d}:00+00:00" for i in range(n)],
    }).with_columns(pl.col("ts").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"))


def setup_catalog(base_dir):
    meta = os.path.join(base_dir, "profile.ducklake")
    data = os.path.join(base_dir, "profile_data")
    os.makedirs(data, exist_ok=True)
    attach = f"ducklake:sqlite:{meta}"
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(f"ATTACH '{attach}' AS ducklake (DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)")
    return meta, data, con


def profile_read(n=10000):
    tmp = tempfile.mkdtemp()
    try:
        meta, data, con = setup_catalog(tmp)
        df = generate_df(n)
        con.execute(
            "CREATE TABLE ducklake.bench "
            "(id INTEGER, name VARCHAR, value DOUBLE, category VARCHAR, ts TIMESTAMP WITH TIME ZONE)"
        )
        con.execute("INSERT INTO ducklake.bench SELECT * FROM df")
        con.close()

        # Profile individual catalog operations
        from ducklake_polars._catalog import DuckLakeCatalogReader

        times = {}

        # Full read path
        reader = DuckLakeCatalogReader(meta, data_path_override=data)
        reader._connect()

        t0 = time.perf_counter()
        snapshot = reader.get_current_snapshot()
        times["get_current_snapshot"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        table = reader.get_table("bench", "main", snapshot.snapshot_id)
        times["get_table"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        all_columns = reader.get_all_columns(table.table_id, snapshot.snapshot_id)
        times["get_all_columns"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        columns = [c for c in all_columns if c.parent_column is None]
        column_names = [c.column_name for c in columns]
        times["filter_columns"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        data_files = reader.get_data_files(table.table_id, snapshot.snapshot_id)
        times["get_data_files"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        delete_files = reader.get_delete_files(table.table_id, snapshot.snapshot_id)
        times["get_delete_files"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        history = reader.get_column_history(table.table_id)
        times["get_column_history"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        inlined = reader.read_inlined_data(table.table_id, snapshot.snapshot_id, column_names)
        times["read_inlined_data"] = time.perf_counter() - t0

        # Stats (filter pushdown path)
        filter_columns = ["id", "value"]
        t0 = time.perf_counter()
        file_ids = [f.data_file_id for f in data_files]
        col_ids = [c.column_id for c in columns if c.column_name in filter_columns]
        stats = reader.get_column_stats(table.table_id, file_ids, col_ids)
        times["get_column_stats"] = time.perf_counter() - t0

        # Resolve file paths
        t0 = time.perf_counter()
        sources = [reader.resolve_data_file_path(f.path, f.path_is_relative, table) for f in data_files]
        times["resolve_paths"] = time.perf_counter() - t0

        # Build stats table
        from ducklake_polars._stats import build_table_statistics
        t0 = time.perf_counter()
        table_stats = build_table_statistics(data_files, stats, columns, filter_columns)
        times["build_table_statistics"] = time.perf_counter() - t0

        # Actual scan_parquet
        from polars.io.parquet.functions import scan_parquet
        t0 = time.perf_counter()
        lf = scan_parquet(sources, missing_columns="insert", extra_columns="ignore",
                         cast_options=pl.ScanCastOptions(integer_cast="upcast", float_cast="upcast",
                                                         missing_struct_fields="insert", extra_struct_fields="ignore",
                                                         categorical_to_string="allow"))
        result = lf.collect()
        times["scan_parquet+collect"] = time.perf_counter() - t0

        reader.close()

        # Full end-to-end
        from ducklake_polars import read_ducklake
        t0 = time.perf_counter()
        read_ducklake(meta, "bench")
        times["full_read_ducklake"] = time.perf_counter() - t0

        total_catalog = sum(v for k, v in times.items()
                           if k not in ("scan_parquet+collect", "full_read_ducklake", "build_table_statistics", "resolve_paths", "filter_columns"))
        times["total_catalog_queries"] = total_catalog

        print(f"\nProfile for {n} rows ({len(data_files)} data files):")
        print("=" * 60)
        for name, t in sorted(times.items(), key=lambda x: -x[1]):
            print(f"  {name:30s}: {t*1000:8.2f} ms")
        print(f"\n  Catalog queries as % of full read: {total_catalog / times['full_read_ducklake'] * 100:.1f}%")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for n in [1000, 10000, 100000]:
        profile_read(n)
