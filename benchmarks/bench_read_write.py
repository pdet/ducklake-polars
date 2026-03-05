"""
DuckLake-Polars Benchmarks: Polars vs Arrow vs Pandas

Compare ducklake-dataframe reading performance across output formats:
  - Polars (native pl.DataFrame / LazyFrame)
  - Arrow (pa.Table via .to_arrow())
  - Pandas (pd.DataFrame via .to_pandas())

Usage:
    python benchmarks/bench_read_write.py [--rows 100000] [--output results.json]
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
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ducklake_polars import (
    read_ducklake,
    scan_ducklake,
    write_ducklake,
)


@dataclass
class BenchResult:
    name: str
    format: str  # "polars", "arrow", "pandas"
    operation: str
    rows: int
    columns: int
    elapsed_s: float
    rows_per_sec: float = 0.0
    notes: str = ""

    def __post_init__(self):
        if self.elapsed_s > 0:
            self.rows_per_sec = self.rows / self.elapsed_s


class DuckLakeBenchmark:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.results: list[BenchResult] = []

    def _make_catalog(self, name: str) -> tuple[str, str]:
        meta = os.path.join(self.base_dir, f"{name}.ducklake")
        data = os.path.join(self.base_dir, f"{name}_data")
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

    def _seed_table(
        self, name: str, rows: int, cols: int, *, num_files: int = 1
    ) -> tuple[str, str]:
        meta, data = self._make_catalog(name)
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )

        col_defs = []
        col_exprs = []
        for i in range(cols):
            ctype = i % 4
            if ctype == 0:
                col_defs.append(f"int_{i} BIGINT")
                col_exprs.append(f"(i + offset) AS int_{i}")
            elif ctype == 1:
                col_defs.append(f"float_{i} DOUBLE")
                col_exprs.append(f"CAST(i + offset AS DOUBLE) * 1.1 AS float_{i}")
            elif ctype == 2:
                col_defs.append(f"str_{i} VARCHAR")
                col_exprs.append(f"'value_' || ((i + offset) % 1000) AS str_{i}")
            else:
                col_defs.append(f"bool_{i} BOOLEAN")
                col_exprs.append(f"((i + offset) % 2 = 0) AS bool_{i}")

        con.execute(f"CREATE TABLE ducklake.bench ({', '.join(col_defs)})")
        batch_size = rows // num_files
        for f in range(num_files):
            offset = f * batch_size
            exprs = ", ".join(
                e.replace("offset", str(offset)) for e in col_exprs
            )
            con.execute(
                f"INSERT INTO ducklake.bench SELECT {exprs} "
                f"FROM range({batch_size}) t(i)"
            )
        con.close()
        return meta, data

    def _build_data(self, rows: int, cols: int) -> dict:
        d = {}
        for i in range(cols):
            ctype = i % 4
            if ctype == 0:
                d[f"int_{i}"] = list(range(rows))
            elif ctype == 1:
                d[f"float_{i}"] = [float(x) * 1.1 for x in range(rows)]
            elif ctype == 2:
                d[f"str_{i}"] = [f"value_{x % 1000}" for x in range(rows)]
            else:
                d[f"bool_{i}"] = [x % 2 == 0 for x in range(rows)]
        return d

    # ---------------------------------------------------------------
    # Write benchmarks
    # ---------------------------------------------------------------

    def bench_write_polars(self, rows: int, cols: int) -> BenchResult:
        meta, _ = self._make_catalog(f"write_pl_{rows}_{cols}")
        df = pl.DataFrame(self._build_data(rows, cols))
        start = time.perf_counter()
        write_ducklake(df, meta, "bench", mode="error")
        elapsed = time.perf_counter() - start
        r = BenchResult("write", "polars", "write", rows, cols, elapsed)
        self.results.append(r)
        return r

    def bench_write_arrow(self, rows: int, cols: int) -> BenchResult:
        meta, _ = self._make_catalog(f"write_ar_{rows}_{cols}")
        arrow_table = pa.table(self._build_data(rows, cols))
        start = time.perf_counter()
        df = pl.from_arrow(arrow_table)
        write_ducklake(df, meta, "bench", mode="error")
        elapsed = time.perf_counter() - start
        r = BenchResult("write", "arrow", "write", rows, cols, elapsed)
        self.results.append(r)
        return r

    def bench_write_pandas(self, rows: int, cols: int) -> BenchResult:
        meta, _ = self._make_catalog(f"write_pd_{rows}_{cols}")
        pdf = pd.DataFrame(self._build_data(rows, cols))
        start = time.perf_counter()
        df = pl.from_pandas(pdf)
        write_ducklake(df, meta, "bench", mode="error")
        elapsed = time.perf_counter() - start
        r = BenchResult("write", "pandas", "write", rows, cols, elapsed)
        self.results.append(r)
        return r

    # ---------------------------------------------------------------
    # Read benchmarks
    # ---------------------------------------------------------------

    def bench_read_polars(self, rows: int, cols: int) -> BenchResult:
        meta, _ = self._seed_table(f"read_pl_{rows}_{cols}", rows, cols)
        start = time.perf_counter()
        df = read_ducklake(meta, "bench")
        elapsed = time.perf_counter() - start
        assert df.shape[0] == rows
        r = BenchResult("read", "polars", "read", rows, cols, elapsed)
        self.results.append(r)
        return r

    def bench_read_arrow(self, rows: int, cols: int) -> BenchResult:
        meta, _ = self._seed_table(f"read_ar_{rows}_{cols}", rows, cols)
        start = time.perf_counter()
        df = read_ducklake(meta, "bench")
        table = df.to_arrow()
        elapsed = time.perf_counter() - start
        assert table.num_rows == rows
        r = BenchResult("read", "arrow", "read", rows, cols, elapsed)
        self.results.append(r)
        return r

    def bench_read_pandas(self, rows: int, cols: int) -> BenchResult:
        meta, _ = self._seed_table(f"read_pd_{rows}_{cols}", rows, cols)
        start = time.perf_counter()
        df = read_ducklake(meta, "bench")
        pdf = df.to_pandas()
        elapsed = time.perf_counter() - start
        assert len(pdf) == rows
        r = BenchResult("read", "pandas", "read", rows, cols, elapsed)
        self.results.append(r)
        return r

    # ---------------------------------------------------------------
    # Scan + filter
    # ---------------------------------------------------------------

    def _seed_filter_table(self, name: str, rows: int) -> str:
        meta, data = self._make_catalog(name)
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute(
            "CREATE TABLE ducklake.bench "
            "(id BIGINT, category VARCHAR, value DOUBLE)"
        )
        batch = rows // 5
        for b in range(5):
            off = b * batch
            con.execute(
                f"INSERT INTO ducklake.bench "
                f"SELECT i + {off}, "
                f"CASE WHEN (i + {off}) % 10 = 0 THEN 'target' ELSE 'other' END, "
                f"CAST(i + {off} AS DOUBLE) FROM range({batch}) t(i)"
            )
        con.close()
        return meta

    def bench_filter_polars(self, rows: int) -> BenchResult:
        meta = self._seed_filter_table(f"filt_pl_{rows}", rows)
        start = time.perf_counter()
        result = (
            scan_ducklake(meta, "bench")
            .filter(pl.col("category") == "target")
            .collect()
        )
        elapsed = time.perf_counter() - start
        r = BenchResult(
            "scan_filter", "polars", "scan_filter", rows, 3, elapsed,
            notes=f"matched={result.shape[0]}",
        )
        self.results.append(r)
        return r

    def bench_filter_arrow(self, rows: int) -> BenchResult:
        meta = self._seed_filter_table(f"filt_ar_{rows}", rows)
        import pyarrow.compute as pc
        start = time.perf_counter()
        df = read_ducklake(meta, "bench")
        table = df.to_arrow()
        filtered = table.filter(pc.equal(table.column("category"), "target"))
        elapsed = time.perf_counter() - start
        r = BenchResult(
            "scan_filter", "arrow", "scan_filter", rows, 3, elapsed,
            notes=f"matched={filtered.num_rows}",
        )
        self.results.append(r)
        return r

    def bench_filter_pandas(self, rows: int) -> BenchResult:
        meta = self._seed_filter_table(f"filt_pd_{rows}", rows)
        start = time.perf_counter()
        df = read_ducklake(meta, "bench")
        pdf = df.to_pandas()
        filtered = pdf[pdf["category"] == "target"]
        elapsed = time.perf_counter() - start
        r = BenchResult(
            "scan_filter", "pandas", "scan_filter", rows, 3, elapsed,
            notes=f"matched={len(filtered)}",
        )
        self.results.append(r)
        return r

    # ---------------------------------------------------------------
    # Aggregation
    # ---------------------------------------------------------------

    def _seed_agg_table(self, name: str, rows: int) -> str:
        meta, data = self._make_catalog(name)
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute(
            f"CREATE TABLE ducklake.bench AS "
            f"SELECT i AS id, i % 100 AS group_id, "
            f"CAST(i AS DOUBLE) * 1.5 AS value "
            f"FROM range({rows}) t(i)"
        )
        con.close()
        return meta

    def bench_agg_polars(self, rows: int) -> BenchResult:
        meta = self._seed_agg_table(f"agg_pl_{rows}", rows)
        start = time.perf_counter()
        result = (
            scan_ducklake(meta, "bench")
            .group_by("group_id")
            .agg(
                pl.col("value").sum().alias("total"),
                pl.col("value").mean().alias("avg"),
                pl.len().alias("cnt"),
            )
            .sort("group_id")
            .collect()
        )
        elapsed = time.perf_counter() - start
        r = BenchResult("aggregation", "polars", "agg", rows, 3, elapsed)
        self.results.append(r)
        return r

    def bench_agg_arrow(self, rows: int) -> BenchResult:
        meta = self._seed_agg_table(f"agg_ar_{rows}", rows)
        start = time.perf_counter()
        df = read_ducklake(meta, "bench")
        table = df.to_arrow()
        result = table.group_by("group_id").aggregate([
            ("value", "sum"), ("value", "mean"), ("value", "count"),
        ])
        elapsed = time.perf_counter() - start
        r = BenchResult("aggregation", "arrow", "agg", rows, 3, elapsed)
        self.results.append(r)
        return r

    def bench_agg_pandas(self, rows: int) -> BenchResult:
        meta = self._seed_agg_table(f"agg_pd_{rows}", rows)
        start = time.perf_counter()
        df = read_ducklake(meta, "bench")
        pdf = df.to_pandas()
        result = pdf.groupby("group_id")["value"].agg(["sum", "mean", "count"])
        elapsed = time.perf_counter() - start
        r = BenchResult("aggregation", "pandas", "agg", rows, 3, elapsed)
        self.results.append(r)
        return r

    # ---------------------------------------------------------------
    # Multi-file read
    # ---------------------------------------------------------------

    def bench_multifile_polars(self, rows: int, nf: int) -> BenchResult:
        meta, _ = self._seed_table(f"mf_pl_{rows}_{nf}", rows, 4, num_files=nf)
        start = time.perf_counter()
        df = read_ducklake(meta, "bench")
        elapsed = time.perf_counter() - start
        assert df.shape[0] == rows
        r = BenchResult(f"multifile_{nf}f", "polars", "read", rows, 4, elapsed)
        self.results.append(r)
        return r

    def bench_multifile_arrow(self, rows: int, nf: int) -> BenchResult:
        meta, _ = self._seed_table(f"mf_ar_{rows}_{nf}", rows, 4, num_files=nf)
        start = time.perf_counter()
        df = read_ducklake(meta, "bench")
        table = df.to_arrow()
        elapsed = time.perf_counter() - start
        assert table.num_rows == rows
        r = BenchResult(f"multifile_{nf}f", "arrow", "read", rows, 4, elapsed)
        self.results.append(r)
        return r

    def bench_multifile_pandas(self, rows: int, nf: int) -> BenchResult:
        meta, _ = self._seed_table(f"mf_pd_{rows}_{nf}", rows, 4, num_files=nf)
        start = time.perf_counter()
        df = read_ducklake(meta, "bench")
        pdf = df.to_pandas()
        elapsed = time.perf_counter() - start
        assert len(pdf) == rows
        r = BenchResult(f"multifile_{nf}f", "pandas", "read", rows, 4, elapsed)
        self.results.append(r)
        return r

    # ---------------------------------------------------------------
    # Run all
    # ---------------------------------------------------------------

    def _print_group(self, name: str, results: list[BenchResult]):
        print(f"--- {name} ---")
        fastest = min(r.elapsed_s for r in results)
        for r in results:
            ratio = r.elapsed_s / fastest if fastest > 0 else 0
            tag = "" if ratio < 1.05 else f"  ({ratio:.1f}x slower)"
            notes = f"  [{r.notes}]" if r.notes else ""
            print(
                f"  {r.format:>8s}: {r.elapsed_s:8.4f}s "
                f"({r.rows_per_sec:>12,.0f} rows/s){tag}{notes}"
            )
        print()

    def run_all(self, rows: int = 100_000, cols: int = 10):
        print(f"\n{'='*70}")
        print(f"DuckLake Benchmark: Polars vs Arrow vs Pandas")
        print(f"  Rows: {rows:,}  Columns: {cols}")
        print(
            f"  polars {pl.__version__}  "
            f"pyarrow {pa.__version__}  "
            f"pandas {pd.__version__}"
        )
        print(f"{'='*70}\n")

        groups = [
            ("Write (from each format into DuckLake)", [
                self.bench_write_polars(rows, cols),
                self.bench_write_arrow(rows, cols),
                self.bench_write_pandas(rows, cols),
            ]),
            ("Read (DuckLake into each format)", [
                self.bench_read_polars(rows, cols),
                self.bench_read_arrow(rows, cols),
                self.bench_read_pandas(rows, cols),
            ]),
            ("Scan + Filter (5 files, 10% selectivity)", [
                self.bench_filter_polars(rows),
                self.bench_filter_arrow(rows),
                self.bench_filter_pandas(rows),
            ]),
            ("Aggregation (group by 100 keys)", [
                self.bench_agg_polars(rows),
                self.bench_agg_arrow(rows),
                self.bench_agg_pandas(rows),
            ]),
            ("Multi-file Read (10 files)", [
                self.bench_multifile_polars(rows, 10),
                self.bench_multifile_arrow(rows, 10),
                self.bench_multifile_pandas(rows, 10),
            ]),
            ("Multi-file Read (50 files)", [
                self.bench_multifile_polars(rows, 50),
                self.bench_multifile_arrow(rows, 50),
                self.bench_multifile_pandas(rows, 50),
            ]),
        ]

        for group_name, group_results in groups:
            self._print_group(group_name, group_results)

    def save_results(self, path: str):
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"Results saved to {path}")

    def print_summary(self):
        print(f"\n{'='*70}")
        print(f"{'Name':<25s} {'Format':<8s} {'Time(s)':<10s} {'Rows/s':<15s}")
        print("-" * 60)
        for r in self.results:
            print(
                f"{r.name:<25s} {r.format:<8s} "
                f"{r.elapsed_s:<10.4f} {r.rows_per_sec:<15,.0f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="DuckLake Benchmark: Polars vs Arrow vs Pandas"
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="ducklake_bench_")
    try:
        bench = DuckLakeBenchmark(tmpdir)
        bench.run_all(rows=args.rows, cols=args.cols)
        bench.print_summary()
        if args.output:
            bench.save_results(args.output)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
