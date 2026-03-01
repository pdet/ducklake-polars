#!/usr/bin/env python3
"""
DuckLake read/write benchmarks.

Compares ducklake-polars, ducklake-pandas, and raw DuckDB for:
  - Write performance (1K, 10K, 100K, 1M rows)
  - Read performance
  - Arrow conversion overhead (pl.DataFrame ↔ pa.Table)
  - Filter pushdown (predicate vs full scan)

Schema: (id INTEGER, name VARCHAR, value DOUBLE, category VARCHAR, ts TIMESTAMP)

Usage:
    python benchmarks/bench_read_write.py [--sizes 1000,10000] [--runs 3]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from ducklake_polars import read_ducklake, scan_ducklake, write_ducklake
from ducklake_pandas import (
    read_ducklake as pd_read_ducklake,
    write_ducklake as pd_write_ducklake,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORIES = ["electronics", "books", "clothing", "food", "toys", "sports"]


def _generate_polars_df(n: int) -> pl.DataFrame:
    """Generate a Polars DataFrame with n rows."""
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "id": list(range(1, n + 1)),
            "name": [f"item_{i}" for i in range(n)],
            "value": rng.uniform(0, 1000, n).tolist(),
            "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(n)],
            "ts": [
                datetime(2024, 1, 1, tzinfo=timezone.utc)
                for _ in range(n)
            ],
        }
    )


def _generate_pandas_df(n: int) -> pd.DataFrame:
    """Generate a Pandas DataFrame with n rows."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": list(range(1, n + 1)),
            "name": [f"item_{i}" for i in range(n)],
            "value": rng.uniform(0, 1000, n),
            "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(n)],
            "ts": pd.Timestamp("2024-01-01", tz="UTC"),
        }
    )


def _setup_ducklake(tmp_dir: str, name: str) -> tuple[str, str]:
    """Create a DuckLake catalog and return (metadata_path, data_path)."""
    meta = os.path.join(tmp_dir, f"{name}.ducklake")
    data = os.path.join(tmp_dir, f"{name}_data")
    os.makedirs(data, exist_ok=True)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{meta}' AS ducklake "
        f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.close()
    return meta, data


def _setup_duckdb_ducklake(tmp_dir: str, name: str) -> tuple[str, str, str]:
    """Create DuckLake catalog for raw DuckDB benchmark.

    Returns (metadata_path, data_path, attach_source).
    """
    meta = os.path.join(tmp_dir, f"{name}.ducklake")
    data = os.path.join(tmp_dir, f"{name}_data")
    os.makedirs(data, exist_ok=True)
    return meta, data, f"ducklake:sqlite:{meta}"


def _timeit(fn: Any, runs: int = 3) -> float:
    """Time a callable over `runs` iterations, return median seconds."""
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def bench_write(
    sizes: list[int], runs: int
) -> list[dict[str, Any]]:
    """Benchmark write performance across methods and sizes."""
    results: list[dict[str, Any]] = []

    for n in sizes:
        pl_df = _generate_polars_df(n)
        pd_df = _generate_pandas_df(n)

        # --- ducklake-polars ---
        tmp = tempfile.mkdtemp()
        try:
            meta, data = _setup_ducklake(tmp, "polars")

            def write_polars():
                tbl = f"bench_{n}_{int(time.time()*1e6)}"
                write_ducklake(pl_df, meta, tbl)

            t = _timeit(write_polars, runs)
            results.append(
                {"operation": "write", "method": "ducklake-polars", "rows": n, "time_s": t}
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        # --- ducklake-pandas ---
        tmp = tempfile.mkdtemp()
        try:
            meta, data = _setup_ducklake(tmp, "pandas")

            def write_pandas():
                tbl = f"bench_{n}_{int(time.time()*1e6)}"
                pd_write_ducklake(pd_df, meta, tbl)

            t = _timeit(write_pandas, runs)
            results.append(
                {"operation": "write", "method": "ducklake-pandas", "rows": n, "time_s": t}
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        # --- raw DuckDB ---
        tmp = tempfile.mkdtemp()
        try:
            meta, data, attach = _setup_duckdb_ducklake(tmp, "duckdb")

            def write_duckdb():
                con = duckdb.connect()
                con.install_extension("ducklake")
                con.load_extension("ducklake")
                con.execute(
                    f"ATTACH '{attach}' AS ducklake "
                    f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
                )
                tbl = f"bench_{n}_{int(time.time()*1e6)}"
                arrow_tbl = pl_df.to_arrow()
                con.execute(
                    f'CREATE TABLE ducklake."{tbl}" '
                    f"(id INTEGER, name VARCHAR, value DOUBLE, "
                    f"category VARCHAR, ts TIMESTAMP WITH TIME ZONE)"
                )
                con.execute(
                    f'INSERT INTO ducklake."{tbl}" SELECT * FROM arrow_tbl'
                )
                con.close()

            t = _timeit(write_duckdb, runs)
            results.append(
                {"operation": "write", "method": "raw-duckdb", "rows": n, "time_s": t}
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return results


def bench_read(
    sizes: list[int], runs: int
) -> list[dict[str, Any]]:
    """Benchmark read performance across methods and sizes."""
    results: list[dict[str, Any]] = []

    for n in sizes:
        pl_df = _generate_polars_df(n)
        pd_df = _generate_pandas_df(n)

        # Setup: write data with DuckDB
        tmp = tempfile.mkdtemp()
        try:
            meta, data, attach = _setup_duckdb_ducklake(tmp, "read")
            con = duckdb.connect()
            con.install_extension("ducklake")
            con.load_extension("ducklake")
            con.execute(
                f"ATTACH '{attach}' AS ducklake "
                f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
            )
            con.execute(
                "CREATE TABLE ducklake.bench "
                "(id INTEGER, name VARCHAR, value DOUBLE, "
                "category VARCHAR, ts TIMESTAMP WITH TIME ZONE)"
            )
            con.execute("INSERT INTO ducklake.bench SELECT * FROM pl_df")
            con.close()

            # --- ducklake-polars read ---
            t = _timeit(lambda: read_ducklake(meta, "bench"), runs)
            results.append(
                {"operation": "read", "method": "ducklake-polars", "rows": n, "time_s": t}
            )

            # --- ducklake-pandas read ---
            t = _timeit(lambda: pd_read_ducklake(meta, "bench"), runs)
            results.append(
                {"operation": "read", "method": "ducklake-pandas", "rows": n, "time_s": t}
            )

            # --- raw DuckDB read ---
            def read_duckdb():
                c = duckdb.connect()
                c.install_extension("ducklake")
                c.load_extension("ducklake")
                c.execute(
                    f"ATTACH '{attach}' AS ducklake "
                    f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
                )
                c.execute("SELECT * FROM ducklake.bench").fetchall()
                c.close()

            t = _timeit(read_duckdb, runs)
            results.append(
                {"operation": "read", "method": "raw-duckdb", "rows": n, "time_s": t}
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return results


def bench_arrow_conversion(
    sizes: list[int], runs: int
) -> list[dict[str, Any]]:
    """Benchmark Arrow ↔ Polars DataFrame conversion overhead."""
    results: list[dict[str, Any]] = []

    for n in sizes:
        pl_df = _generate_polars_df(n)
        arrow_table = pl_df.to_arrow()

        # Polars → Arrow
        t = _timeit(lambda: pl_df.to_arrow(), runs)
        results.append(
            {"operation": "pl_to_arrow", "method": "conversion", "rows": n, "time_s": t}
        )

        # Arrow → Polars
        t = _timeit(lambda: pl.from_arrow(arrow_table), runs)
        results.append(
            {"operation": "arrow_to_pl", "method": "conversion", "rows": n, "time_s": t}
        )

    return results


def bench_filter_pushdown(
    sizes: list[int], runs: int
) -> list[dict[str, Any]]:
    """Benchmark filter pushdown: predicate scan vs full scan."""
    results: list[dict[str, Any]] = []

    for n in sizes:
        pl_df = _generate_polars_df(n)

        tmp = tempfile.mkdtemp()
        try:
            meta, data, attach = _setup_duckdb_ducklake(tmp, "filter")
            con = duckdb.connect()
            con.install_extension("ducklake")
            con.load_extension("ducklake")
            con.execute(
                f"ATTACH '{attach}' AS ducklake "
                f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
            )
            con.execute(
                "CREATE TABLE ducklake.bench "
                "(id INTEGER, name VARCHAR, value DOUBLE, "
                "category VARCHAR, ts TIMESTAMP WITH TIME ZONE)"
            )
            con.execute("INSERT INTO ducklake.bench SELECT * FROM pl_df")
            con.close()

            # Full scan
            t_full = _timeit(
                lambda: scan_ducklake(meta, "bench").collect(),
                runs,
            )
            results.append(
                {"operation": "full_scan", "method": "ducklake-polars", "rows": n, "time_s": t_full}
            )

            # Filtered scan (id > 50% of rows)
            threshold = n // 2
            t_filter = _timeit(
                lambda: (
                    scan_ducklake(meta, "bench")
                    .filter(pl.col("id") > threshold)
                    .collect()
                ),
                runs,
            )
            results.append(
                {"operation": "filtered_scan", "method": "ducklake-polars", "rows": n, "time_s": t_filter}
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _format_time(t: float) -> str:
    """Format a time value for display."""
    if t < 0.001:
        return f"{t * 1_000_000:.0f}µs"
    if t < 1.0:
        return f"{t * 1_000:.1f}ms"
    return f"{t:.2f}s"


def _format_rows(n: int) -> str:
    """Format row count for display."""
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def print_results(all_results: list[dict[str, Any]]) -> str:
    """Print results as a markdown table and return the string."""
    lines: list[str] = []

    # Group by operation
    operations = sorted(set(r["operation"] for r in all_results))

    for op in operations:
        op_results = [r for r in all_results if r["operation"] == op]

        lines.append(f"\n### {op.replace('_', ' ').title()}\n")
        lines.append("| Rows | Method | Time | Rows/s |")
        lines.append("|------|--------|------|--------|")

        for r in sorted(op_results, key=lambda x: (x["rows"], x["method"])):
            rows_str = _format_rows(r["rows"])
            time_str = _format_time(r["time_s"])
            rows_per_sec = r["rows"] / r["time_s"] if r["time_s"] > 0 else float("inf")
            if rows_per_sec >= 1_000_000:
                rps_str = f"{rows_per_sec / 1_000_000:.1f}M"
            elif rows_per_sec >= 1_000:
                rps_str = f"{rows_per_sec / 1_000:.0f}K"
            else:
                rps_str = f"{rows_per_sec:.0f}"
            lines.append(f"| {rows_str} | {r['method']} | {time_str} | {rps_str} |")

    output = "\n".join(lines)
    print(output)
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="DuckLake read/write benchmarks")
    parser.add_argument(
        "--sizes",
        default="1000,10000,100000",
        help="Comma-separated row counts (default: 1000,10000,100000)",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs per benchmark (default: 3)"
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    runs = args.runs

    print(f"DuckLake Benchmarks — sizes={sizes}, runs={runs}")
    print("=" * 60)

    all_results: list[dict[str, Any]] = []

    print("\n⏱  Write benchmarks...")
    all_results.extend(bench_write(sizes, runs))

    print("⏱  Read benchmarks...")
    all_results.extend(bench_read(sizes, runs))

    print("⏱  Arrow conversion benchmarks...")
    all_results.extend(bench_arrow_conversion(sizes, runs))

    print("⏱  Filter pushdown benchmarks...")
    all_results.extend(bench_filter_pushdown(sizes, runs))

    print("\n" + "=" * 60)
    print_results(all_results)


if __name__ == "__main__":
    main()
