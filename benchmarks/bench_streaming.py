"""
DuckLake vs Iceberg — Streaming Benchmark

Simulates a streaming ingestion workload: many small appends followed by
reads, scans, and compaction. Compares ducklake-dataframe against
pyiceberg, both using Polars as the DataFrame engine.

Scenarios:
  1. Streaming Append: N batches of M rows each
  2. Read-After-Write: append then immediately read back
  3. Scan + Filter after streaming ingestion
  4. Compaction: merge small files after streaming
  5. Time Travel: read historical snapshots

Usage:
    python benchmarks/bench_streaming.py [--batches 100] [--batch-size 1000]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass

import duckdb
import polars as pl
import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ducklake_polars import (
    DuckLakeCatalog,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
    rewrite_data_files_ducklake,
)


@dataclass
class BenchResult:
    name: str
    system: str  # "ducklake" or "iceberg"
    operation: str
    total_rows: int
    elapsed_s: float
    rows_per_sec: float = 0.0
    notes: str = ""

    def __post_init__(self):
        if self.elapsed_s > 0:
            self.rows_per_sec = self.total_rows / self.elapsed_s


def _make_batch(batch_id: int, batch_size: int) -> pl.DataFrame:
    """Generate a streaming batch of sensor-like data."""
    offset = batch_id * batch_size
    return pl.DataFrame({
        "event_id": list(range(offset, offset + batch_size)),
        "timestamp": [1709600000 + offset + i for i in range(batch_size)],
        "sensor_id": [f"sensor_{(offset + i) % 50}" for i in range(batch_size)],
        "temperature": [20.0 + ((offset + i) % 100) * 0.1 for i in range(batch_size)],
        "humidity": [50.0 + ((offset + i) % 80) * 0.5 for i in range(batch_size)],
        "status": [
            "ok" if (offset + i) % 20 != 0 else "alert"
            for i in range(batch_size)
        ],
    })


def _make_batch_arrow(batch_id: int, batch_size: int) -> pa.Table:
    """Generate the same batch as a PyArrow Table for Iceberg."""
    df = _make_batch(batch_id, batch_size)
    return df.to_arrow()


# ===================================================================
# DuckLake benchmarks
# ===================================================================


class DuckLakeBench:
    def __init__(self, base_dir: str):
        self.meta = os.path.join(base_dir, "ducklake.ducklake")
        self.data = os.path.join(base_dir, "ducklake_data")
        os.makedirs(self.data, exist_ok=True)
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{self.meta}' AS ducklake "
            f"(DATA_PATH '{self.data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.close()

    def bench_streaming_append(
        self, batches: int, batch_size: int
    ) -> BenchResult:
        total = batches * batch_size
        start = time.perf_counter()
        for b in range(batches):
            df = _make_batch(b, batch_size)
            mode = "error" if b == 0 else "append"
            write_ducklake(df, self.meta, "events", mode=mode)
        elapsed = time.perf_counter() - start
        return BenchResult(
            "streaming_append", "ducklake", "write",
            total, elapsed, notes=f"{batches}x{batch_size}",
        )

    def bench_read_after_write(
        self, batches: int, batch_size: int
    ) -> BenchResult:
        total = batches * batch_size
        start = time.perf_counter()
        for b in range(batches):
            df = _make_batch(b + 10000, batch_size)
            write_ducklake(df, self.meta, "events_raw", mode="append" if b > 0 else "error")
            result = read_ducklake(self.meta, "events_raw")
        elapsed = time.perf_counter() - start
        return BenchResult(
            "read_after_write", "ducklake", "read_write",
            total, elapsed, notes=f"{batches}x{batch_size}",
        )

    def bench_scan_filter(self) -> BenchResult:
        start = time.perf_counter()
        result = (
            scan_ducklake(self.meta, "events")
            .filter(pl.col("status") == "alert")
            .collect()
        )
        elapsed = time.perf_counter() - start
        total = read_ducklake(self.meta, "events").shape[0]
        return BenchResult(
            "scan_filter", "ducklake", "scan_filter",
            total, elapsed, notes=f"matched={result.shape[0]}",
        )

    def bench_aggregation(self) -> BenchResult:
        start = time.perf_counter()
        result = (
            scan_ducklake(self.meta, "events")
            .group_by("sensor_id")
            .agg(
                pl.col("temperature").mean().alias("avg_temp"),
                pl.col("humidity").mean().alias("avg_hum"),
                pl.len().alias("count"),
            )
            .sort("sensor_id")
            .collect()
        )
        elapsed = time.perf_counter() - start
        total = read_ducklake(self.meta, "events").shape[0]
        return BenchResult(
            "aggregation", "ducklake", "agg",
            total, elapsed, notes=f"groups={result.shape[0]}",
        )

    def bench_compaction(self) -> BenchResult:
        api = DuckLakeCatalog(self.meta)
        files_before = api.list_files("events").shape[0]
        start = time.perf_counter()
        rewrite_data_files_ducklake(self.meta, "events")
        elapsed = time.perf_counter() - start
        files_after = api.list_files("events").shape[0]
        total = read_ducklake(self.meta, "events").shape[0]
        return BenchResult(
            "compaction", "ducklake", "compaction",
            total, elapsed,
            notes=f"files: {files_before} -> {files_after}",
        )

    def bench_time_travel(self) -> BenchResult:
        start = time.perf_counter()
        result = read_ducklake(self.meta, "events", snapshot_version=1)
        elapsed = time.perf_counter() - start
        return BenchResult(
            "time_travel", "ducklake", "read",
            result.shape[0], elapsed, notes="snapshot=1",
        )


# ===================================================================
# Iceberg benchmarks
# ===================================================================


class IcebergBench:
    def __init__(self, base_dir: str):
        from pyiceberg.catalog.sql import SqlCatalog
        self.warehouse = os.path.join(base_dir, "iceberg_warehouse")
        os.makedirs(self.warehouse, exist_ok=True)
        self.catalog = SqlCatalog("bench", **{
            "uri": f"sqlite:///{base_dir}/iceberg.db",
            "warehouse": f"file://{self.warehouse}",
        })
        self.catalog.create_namespace("default")
        self._table = None
        self._raw_table = None
        self._schema = pa.schema([
            ("event_id", pa.int64()),
            ("timestamp", pa.int64()),
            ("sensor_id", pa.string()),
            ("temperature", pa.float64()),
            ("humidity", pa.float64()),
            ("status", pa.string()),
        ])

    def _get_or_create(self, name: str) -> "pyiceberg.table.Table":
        from pyiceberg.schema import Schema
        from pyiceberg.types import (
            LongType,
            DoubleType,
            StringType,
            NestedField,
        )
        try:
            return self.catalog.load_table(f"default.{name}")
        except Exception:
            schema = Schema(
                NestedField(1, "event_id", LongType(), required=False),
                NestedField(2, "timestamp", LongType(), required=False),
                NestedField(3, "sensor_id", StringType(), required=False),
                NestedField(4, "temperature", DoubleType(), required=False),
                NestedField(5, "humidity", DoubleType(), required=False),
                NestedField(6, "status", StringType(), required=False),
            )
            return self.catalog.create_table(f"default.{name}", schema=schema)

    def bench_streaming_append(
        self, batches: int, batch_size: int
    ) -> BenchResult:
        total = batches * batch_size
        table = self._get_or_create("events")
        start = time.perf_counter()
        for b in range(batches):
            arrow_batch = _make_batch_arrow(b, batch_size)
            table.append(arrow_batch)
        elapsed = time.perf_counter() - start
        self._table = table
        return BenchResult(
            "streaming_append", "iceberg", "write",
            total, elapsed, notes=f"{batches}x{batch_size}",
        )

    def bench_read_after_write(
        self, batches: int, batch_size: int
    ) -> BenchResult:
        total = batches * batch_size
        table = self._get_or_create("events_raw")
        start = time.perf_counter()
        for b in range(batches):
            arrow_batch = _make_batch_arrow(b + 10000, batch_size)
            table.append(arrow_batch)
            scan = table.scan()
            result = pl.from_arrow(scan.to_arrow())
        elapsed = time.perf_counter() - start
        self._raw_table = table
        return BenchResult(
            "read_after_write", "iceberg", "read_write",
            total, elapsed, notes=f"{batches}x{batch_size}",
        )

    def bench_scan_filter(self) -> BenchResult:
        table = self._table or self._get_or_create("events")
        start = time.perf_counter()
        scan = table.scan(row_filter="status = 'alert'")
        arrow_result = scan.to_arrow()
        result = pl.from_arrow(arrow_result)
        elapsed = time.perf_counter() - start
        total_scan = table.scan()
        total = total_scan.to_arrow().num_rows
        return BenchResult(
            "scan_filter", "iceberg", "scan_filter",
            total, elapsed, notes=f"matched={result.shape[0]}",
        )

    def bench_aggregation(self) -> BenchResult:
        table = self._table or self._get_or_create("events")
        start = time.perf_counter()
        scan = table.scan()
        arrow_table = scan.to_arrow()
        df = pl.from_arrow(arrow_table)
        result = (
            df.group_by("sensor_id")
            .agg(
                pl.col("temperature").mean().alias("avg_temp"),
                pl.col("humidity").mean().alias("avg_hum"),
                pl.len().alias("count"),
            )
            .sort("sensor_id")
        )
        elapsed = time.perf_counter() - start
        return BenchResult(
            "aggregation", "iceberg", "agg",
            arrow_table.num_rows, elapsed,
            notes=f"groups={result.shape[0]}",
        )

    def bench_time_travel(self) -> BenchResult:
        table = self._table or self._get_or_create("events")
        snapshots = list(table.metadata.snapshots)
        if len(snapshots) < 2:
            return BenchResult(
                "time_travel", "iceberg", "read", 0, 0.0,
                notes="skipped (< 2 snapshots)",
            )
        first_snap = snapshots[0].snapshot_id
        start = time.perf_counter()
        scan = table.scan(snapshot_id=first_snap)
        result = pl.from_arrow(scan.to_arrow())
        elapsed = time.perf_counter() - start
        return BenchResult(
            "time_travel", "iceberg", "read",
            result.shape[0], elapsed,
            notes=f"snapshot_id={first_snap}",
        )


# ===================================================================
# Runner
# ===================================================================


def run_benchmark(batches: int, batch_size: int, output: str | None = None):
    total_rows = batches * batch_size
    print(f"\n{'='*70}")
    print(f"DuckLake vs Iceberg — Streaming Benchmark")
    print(f"  {batches} batches × {batch_size} rows = {total_rows:,} total rows")
    print(f"  polars {pl.__version__}  pyarrow {pa.__version__}")
    print(f"{'='*70}\n")

    tmpdir = tempfile.mkdtemp(prefix="streaming_bench_")
    results: list[BenchResult] = []

    try:
        dl = DuckLakeBench(os.path.join(tmpdir, "ducklake"))
        ice = IcebergBench(os.path.join(tmpdir, "iceberg"))

        benchmarks = [
            ("Streaming Append", [
                dl.bench_streaming_append(batches, batch_size),
                ice.bench_streaming_append(batches, batch_size),
            ]),
            ("Read-After-Write (per batch)", [
                dl.bench_read_after_write(
                    min(batches, 20), batch_size
                ),
                ice.bench_read_after_write(
                    min(batches, 20), batch_size
                ),
            ]),
            ("Scan + Filter (status = 'alert')", [
                dl.bench_scan_filter(),
                ice.bench_scan_filter(),
            ]),
            ("Aggregation (group by sensor_id)", [
                dl.bench_aggregation(),
                ice.bench_aggregation(),
            ]),
            ("Compaction", [
                dl.bench_compaction(),
                BenchResult(
                    "compaction", "iceberg", "compaction", 0, 0.0,
                    notes="not available in pyiceberg",
                ),
            ]),
            ("Time Travel (first snapshot)", [
                dl.bench_time_travel(),
                ice.bench_time_travel(),
            ]),
        ]

        for group_name, group_results in benchmarks:
            print(f"--- {group_name} ---")
            valid = [r for r in group_results if r.elapsed_s > 0]
            fastest = min(r.elapsed_s for r in valid) if valid else 1.0

            for r in group_results:
                results.append(r)
                if r.elapsed_s <= 0:
                    print(f"  {r.system:>10s}: {'N/A':>10s}  ({r.notes})")
                    continue
                ratio = r.elapsed_s / fastest if fastest > 0 else 0
                tag = "" if ratio < 1.05 else f"  ({ratio:.1f}x slower)"
                notes_str = f"  [{r.notes}]" if r.notes else ""
                print(
                    f"  {r.system:>10s}: {r.elapsed_s:8.3f}s "
                    f"({r.rows_per_sec:>12,.0f} rows/s){tag}{notes_str}"
                )
            print()

        # Summary
        print(f"{'='*70}")
        print(f"{'Name':<25s} {'System':<10s} {'Time(s)':<10s} {'Rows/s':<15s}")
        print("-" * 60)
        for r in results:
            if r.elapsed_s > 0:
                print(
                    f"{r.name:<25s} {r.system:<10s} "
                    f"{r.elapsed_s:<10.3f} {r.rows_per_sec:<15,.0f}"
                )
            else:
                print(
                    f"{r.name:<25s} {r.system:<10s} "
                    f"{'N/A':<10s} {r.notes}"
                )

        if output:
            with open(output, "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2)
            print(f"\nResults saved to {output}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="DuckLake vs Iceberg — Streaming Benchmark"
    )
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_benchmark(args.batches, args.batch_size, args.output)


if __name__ == "__main__":
    main()
