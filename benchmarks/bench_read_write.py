#!/usr/bin/env python3
"""
DuckLake read/write performance benchmarks.

Self-tracking benchmarks for ducklake-polars and ducklake-pandas.
Use to detect regressions and measure improvement over time.

Measures:
  - Write performance (ducklake-polars vs ducklake-pandas)
  - Read performance (ducklake-polars vs ducklake-pandas)
  - Arrow conversion overhead (pl.DataFrame ↔ pa.Table, pd.DataFrame ↔ pa.Table)
  - Filter pushdown: filtered scan vs full scan (polars only)

Schema: (id INTEGER, name VARCHAR, value DOUBLE, category VARCHAR, ts TIMESTAMP)

Usage:
    python benchmarks/bench_read_write.py [--sizes 1000,10000] [--runs 3]
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
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
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "id": list(range(n)),
            "name": [f"item_{i}" for i in range(n)],
            "value": rng.uniform(0, 1000, n).tolist(),
            "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(n)],
            "ts": [
                f"2024-01-01T{(i % 24):02d}:{(i % 60):02d}:00+00:00"
                for i in range(n)
            ],
        }
    ).with_columns(pl.col("ts").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"))


def _generate_pandas_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": list(range(n)),
            "name": [f"item_{i}" for i in range(n)],
            "value": rng.uniform(0, 1000, n),
            "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(n)],
            "ts": pd.to_datetime(
                [
                    f"2024-01-01T{(i % 24):02d}:{(i % 60):02d}:00+00:00"
                    for i in range(n)
                ]
            ),
        }
    )


def _setup_catalog(base_dir: str, label: str) -> tuple[str, str]:
    """Create a DuckLake catalog and return (metadata_path, data_path)."""
    meta = os.path.join(base_dir, f"{label}.ducklake")
    data = os.path.join(base_dir, f"{label}_data")
    os.makedirs(data, exist_ok=True)
    attach = f"ducklake:sqlite:{meta}"
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH '{attach}' AS ducklake "
        f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.close()
    return meta, data


def _timeit(fn, runs: int = 3) -> float:
    """Run fn `runs` times and return the median time in seconds."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_write(sizes: list[int], runs: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for n in sizes:
        pl_df = _generate_polars_df(n)
        pd_df = _generate_pandas_df(n)

        # ducklake-polars write
        tmp = tempfile.mkdtemp()
        try:
            meta, data = _setup_catalog(tmp, "polars")

            def write_polars():
                tbl = f"bench_{int(time.time()*1e6)}"
                write_ducklake(pl_df, meta, tbl, mode="error", data_path=data)

            t = _timeit(write_polars, runs)
            results.append({"operation": "write", "method": "ducklake-polars", "rows": n, "time_s": t})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        # ducklake-pandas write
        tmp = tempfile.mkdtemp()
        try:
            meta, data = _setup_catalog(tmp, "pandas")

            def write_pandas():
                tbl = f"bench_{int(time.time()*1e6)}"
                pd_write_ducklake(pd_df, meta, tbl, mode="error", data_path=data)

            t = _timeit(write_pandas, runs)
            results.append({"operation": "write", "method": "ducklake-pandas", "rows": n, "time_s": t})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return results


def bench_read(sizes: list[int], runs: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for n in sizes:
        pl_df = _generate_polars_df(n)

        # Write data once with DuckDB for consistent baseline
        tmp = tempfile.mkdtemp()
        try:
            meta, data = _setup_catalog(tmp, "read")
            attach = f"ducklake:sqlite:{meta}"
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

            # ducklake-polars read
            t = _timeit(lambda: read_ducklake(meta, "bench"), runs)
            results.append({"operation": "read", "method": "ducklake-polars", "rows": n, "time_s": t})

            # ducklake-pandas read
            t = _timeit(lambda: pd_read_ducklake(meta, "bench"), runs)
            results.append({"operation": "read", "method": "ducklake-pandas", "rows": n, "time_s": t})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return results


def bench_arrow_conversion(sizes: list[int], runs: int) -> list[dict[str, Any]]:
    """Benchmark Arrow conversion overhead — the cost of the wrapper layer."""
    results: list[dict[str, Any]] = []

    for n in sizes:
        pl_df = _generate_polars_df(n)
        pd_df = _generate_pandas_df(n)
        arrow_table = pl_df.to_arrow()

        # Polars → Arrow
        t = _timeit(lambda: pl_df.to_arrow(), runs)
        results.append({"operation": "pl_to_arrow", "method": "polars", "rows": n, "time_s": t})

        # Arrow → Polars
        t = _timeit(lambda: pl.from_arrow(arrow_table), runs)
        results.append({"operation": "arrow_to_pl", "method": "polars", "rows": n, "time_s": t})

        # Pandas → Arrow
        t = _timeit(lambda: pa.Table.from_pandas(pd_df), runs)
        results.append({"operation": "pd_to_arrow", "method": "pandas", "rows": n, "time_s": t})

        # Arrow → Pandas
        t = _timeit(lambda: arrow_table.to_pandas(), runs)
        results.append({"operation": "arrow_to_pd", "method": "pandas", "rows": n, "time_s": t})

    return results


def bench_filter_pushdown(sizes: list[int], runs: int) -> list[dict[str, Any]]:
    """Benchmark filter pushdown: scan_ducklake with predicate vs full read."""
    results: list[dict[str, Any]] = []

    for n in sizes:
        pl_df = _generate_polars_df(n)

        tmp = tempfile.mkdtemp()
        try:
            meta, data = _setup_catalog(tmp, "filter")
            attach = f"ducklake:sqlite:{meta}"
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

            # Full scan (polars)
            t = _timeit(lambda: scan_ducklake(meta, "bench").collect(), runs)
            results.append({"operation": "full_scan", "method": "ducklake-polars", "rows": n, "time_s": t})

            # Filtered scan (polars) — select ~50%
            threshold = n // 2
            t = _timeit(
                lambda: scan_ducklake(meta, "bench").filter(pl.col("id") > threshold).collect(),
                runs,
            )
            results.append({"operation": "filtered_scan", "method": "ducklake-polars", "rows": n, "time_s": t})

            # Full read (pandas)
            t = _timeit(lambda: pd_read_ducklake(meta, "bench"), runs)
            results.append({"operation": "full_scan", "method": "ducklake-pandas", "rows": n, "time_s": t})

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _format_time(t: float) -> str:
    if t < 0.001:
        return f"{t * 1_000_000:.0f}µs"
    if t < 1.0:
        return f"{t * 1_000:.1f}ms"
    return f"{t:.2f}s"


def _format_rows(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def print_results(all_results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    operations = sorted(set(r["operation"] for r in all_results))

    for op in operations:
        op_results = [r for r in all_results if r["operation"] == op]
        lines.append(f"\n### {op.replace('_', ' ').title()}\n")
        lines.append("| Rows | Method | Time | Rows/s |")
        lines.append("|------|--------|------|--------|")

        for r in sorted(op_results, key=lambda x: (x["rows"], x["method"])):
            rows_str = _format_rows(r["rows"])
            time_str = _format_time(r["time_s"])
            rps = r["rows"] / r["time_s"] if r["time_s"] > 0 else float("inf")
            if rps >= 1_000_000:
                rps_str = f"{rps / 1_000_000:.1f}M"
            elif rps >= 1_000:
                rps_str = f"{rps / 1_000:.0f}K"
            else:
                rps_str = f"{rps:.0f}"
            lines.append(f"| {rows_str} | {r['method']} | {time_str} | {rps_str} |")

    output = "\n".join(lines)
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="DuckLake performance benchmarks (self-tracking)")
    parser.add_argument("--sizes", default="1000,10000,100000", help="Comma-separated row counts")
    parser.add_argument("--runs", type=int, default=3, help="Runs per benchmark (median used)")
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

    print("⏱  Arrow conversion overhead...")
    all_results.extend(bench_arrow_conversion(sizes, runs))

    print("⏱  Filter pushdown...")
    all_results.extend(bench_filter_pushdown(sizes, runs))

    print("\n" + "=" * 60)
    print_results(all_results)


if __name__ == "__main__":
    main()
