#!/usr/bin/env python3
"""Detailed profile of the read path including connection overhead and multi-file scenarios."""

import cProfile
import os
import pstats
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


def setup_catalog_multifile(base_dir, n_rows=10000, n_files=10):
    """Create catalog with multiple data files."""
    meta = os.path.join(base_dir, "profile.ducklake")
    data = os.path.join(base_dir, "profile_data")
    os.makedirs(data, exist_ok=True)
    attach = f"ducklake:sqlite:{meta}"
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(f"ATTACH '{attach}' AS ducklake (DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)")
    con.execute(
        "CREATE TABLE ducklake.bench "
        "(id INTEGER, name VARCHAR, value DOUBLE, category VARCHAR, ts TIMESTAMP WITH TIME ZONE)"
    )

    rows_per_file = n_rows // n_files
    for i in range(n_files):
        start = i * rows_per_file
        end = start + rows_per_file
        chunk = generate_df(n_rows).slice(start, rows_per_file)
        con.execute("INSERT INTO ducklake.bench SELECT * FROM chunk")

    con.close()
    return meta, data


def profile_multifile():
    """Profile read with multiple files."""
    tmp = tempfile.mkdtemp()
    try:
        for n_files in [1, 5, 10, 20]:
            meta, data = setup_catalog_multifile(tmp + f"_{n_files}", n_rows=10000, n_files=n_files)

            from ducklake_polars import read_ducklake, scan_ducklake

            # Warm up
            read_ducklake(meta, "bench")

            # Time it
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                read_ducklake(meta, "bench")
                times.append(time.perf_counter() - t0)
            times.sort()
            median = times[len(times) // 2]

            # Time with scan (lazy)
            scan_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                scan_ducklake(meta, "bench").collect()
                scan_times.append(time.perf_counter() - t0)
            scan_times.sort()
            scan_median = scan_times[len(scan_times) // 2]

            print(f"  {n_files:3d} files: read={median*1000:.1f}ms  scan+collect={scan_median*1000:.1f}ms")

    finally:
        for suffix in ["_1", "_5", "_10", "_20"]:
            shutil.rmtree(tmp + suffix, ignore_errors=True)


def profile_cprofile():
    """Use cProfile to find exact hotspots."""
    tmp = tempfile.mkdtemp()
    try:
        meta, data = setup_catalog_multifile(tmp, n_rows=100000, n_files=10)

        from ducklake_polars import read_ducklake

        # Warm up
        read_ducklake(meta, "bench")

        # Profile 10 reads
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(10):
            read_ducklake(meta, "bench")
        pr.disable()

        print("\n\nTop 30 by cumulative time (10 reads of 100K rows, 10 files):")
        print("=" * 100)
        stats = pstats.Stats(pr)
        stats.sort_stats("cumulative")
        stats.print_stats(30)

        print("\n\nTop 20 by total (self) time:")
        print("=" * 100)
        stats.sort_stats("tottime")
        stats.print_stats(20)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("Multi-file read profile (10K rows):")
    print("=" * 60)
    profile_multifile()
    profile_cprofile()
