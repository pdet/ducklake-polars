"""DuckLake catalog utility functions — Python equivalents of DuckLake's DuckDB extension functions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import polars as pl

from ducklake_polars._catalog import DuckLakeCatalogReader
from ducklake_polars._schema import resolve_column_type

if TYPE_CHECKING:
    from pathlib import Path


class DuckLakeCatalog:
    """
    High-level interface for inspecting DuckLake catalog metadata.

    Provides Python equivalents of DuckLake's DuckDB extension utility
    functions (``ducklake_snapshots``, ``ducklake_table_info``, etc.).
    All methods return ``pl.DataFrame``.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (``.ducklake`` or ``.db``),
        or a PostgreSQL connection string.
    data_path
        Override the data path stored in the catalog.

    Examples
    --------
    >>> catalog = DuckLakeCatalog("catalog.ducklake")
    >>> catalog.snapshots()
    >>> catalog.current_snapshot()
    >>> catalog.table_info()
    >>> catalog.list_files("my_table")
    """

    def __init__(self, path: str | Path, *, data_path: str | Path | None = None) -> None:
        self._metadata_path = os.fspath(path)
        self._data_path_override = os.fspath(data_path) if data_path else None

    @staticmethod
    def _build_cdf_schema(all_columns: list, extra_prefix: list[tuple[str, pl.DataType]] | None = None) -> dict[str, pl.DataType]:
        """Build a schema dict for change data feed DataFrames from catalog columns."""
        schema_dict: dict[str, pl.DataType] = {}
        if extra_prefix:
            for name, dtype in extra_prefix:
                schema_dict[name] = dtype
        top_level = [c for c in all_columns if c.parent_column is None]
        for col in top_level:
            try:
                schema_dict[col.column_name] = resolve_column_type(
                    col.column_id, col.column_type, all_columns
                )
            except ValueError:
                schema_dict[col.column_name] = pl.String
        return schema_dict

    def _reader(self) -> DuckLakeCatalogReader:
        return DuckLakeCatalogReader(self._metadata_path, data_path_override=self._data_path_override)

    def __enter__(self) -> DuckLakeCatalog:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    # ------------------------------------------------------------------
    # Snapshot functions
    # ------------------------------------------------------------------

    def snapshots(self) -> pl.DataFrame:
        """
        List all snapshots in the catalog.

        Equivalent to ``ducklake_snapshots(catalog)``.

        Returns a DataFrame with columns:
        - ``snapshot_id`` (Int64)
        - ``snapshot_time`` (String)
        - ``schema_version`` (Int64)
        """
        with self._reader() as reader:
            rows = reader.get_all_snapshots()
        if not rows:
            return pl.DataFrame(
                schema={"snapshot_id": pl.Int64, "snapshot_time": pl.String, "schema_version": pl.Int64}
            )
        return pl.DataFrame(
            {
                "snapshot_id": [r[0] for r in rows],
                "snapshot_time": [str(r[2]) for r in rows],
                "schema_version": [r[1] for r in rows],
            },
            schema={"snapshot_id": pl.Int64, "snapshot_time": pl.String, "schema_version": pl.Int64},
        )

    def current_snapshot(self) -> int:
        """
        Get the current (latest) snapshot ID.

        Equivalent to ``ducklake_current_snapshot(catalog)``
        and ``ducklake_last_committed_snapshot(catalog)``.
        """
        with self._reader() as reader:
            snap = reader.get_current_snapshot()
        return snap.snapshot_id

    # ------------------------------------------------------------------
    # Table metadata
    # ------------------------------------------------------------------

    def table_info(self, *, schema: str = "main") -> pl.DataFrame:
        """
        Get per-table storage metadata.

        Equivalent to ``ducklake_table_info(catalog)``.

        Returns a DataFrame with columns:
        - ``table_name`` (String)
        - ``table_id`` (Int64)
        - ``file_count`` (Int64)
        - ``file_size_bytes`` (Int64)
        - ``delete_file_count`` (Int64)
        - ``delete_row_count`` (Int64)
        """
        with self._reader() as reader:
            snap = reader.get_current_snapshot()
            schemas = reader.get_all_schemas(snap.snapshot_id)
            schema_row = None
            for s in schemas:
                if s[1] == schema:
                    schema_row = s
                    break
            if schema_row is None:
                return pl.DataFrame(
                    schema={
                        "table_name": pl.String,
                        "table_id": pl.Int64,
                        "file_count": pl.Int64,
                        "file_size_bytes": pl.Int64,
                        "delete_file_count": pl.Int64,
                        "delete_row_count": pl.Int64,
                    }
                )

            tables = reader.get_all_tables(schema_row[0], snap.snapshot_id)
            result_rows = []
            for table_id, table_name in tables:
                data_files = reader.get_data_files(table_id, snap.snapshot_id)
                delete_files = reader.get_delete_files(table_id, snap.snapshot_id)
                file_count = len(data_files)
                file_size = sum(f.file_size_bytes for f in data_files)
                del_count = len(delete_files)
                del_size = sum(f.delete_count for f in delete_files)
                result_rows.append((table_name, table_id, file_count, file_size, del_count, del_size))

        if not result_rows:
            return pl.DataFrame(
                schema={
                    "table_name": pl.String,
                    "table_id": pl.Int64,
                    "file_count": pl.Int64,
                    "file_size_bytes": pl.Int64,
                    "delete_file_count": pl.Int64,
                    "delete_row_count": pl.Int64,
                }
            )

        return pl.DataFrame(
            {
                "table_name": [r[0] for r in result_rows],
                "table_id": [r[1] for r in result_rows],
                "file_count": [r[2] for r in result_rows],
                "file_size_bytes": [r[3] for r in result_rows],
                "delete_file_count": [r[4] for r in result_rows],
                "delete_row_count": [r[5] for r in result_rows],
            },
            schema={
                "table_name": pl.String,
                "table_id": pl.Int64,
                "file_count": pl.Int64,
                "file_size_bytes": pl.Int64,
                "delete_file_count": pl.Int64,
                "delete_row_count": pl.Int64,
            },
        )

    def list_files(
        self,
        table: str,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pl.DataFrame:
        """
        List data files and delete files for a table.

        Equivalent to ``ducklake_list_files(catalog, table)``.

        Returns a DataFrame with columns:
        - ``data_file`` (String) — resolved path to the data file
        - ``data_file_size_bytes`` (Int64)
        - ``delete_file`` (String or null) — resolved path to delete file
        - ``delete_row_count`` (Int64 or null)
        """
        with self._reader() as reader:
            if snapshot_version is not None:
                snap = reader.get_snapshot_at_version(snapshot_version)
            else:
                snap = reader.get_current_snapshot()

            table_info = reader.get_table(table, schema, snap.snapshot_id)
            data_files = reader.get_data_files(table_info.table_id, snap.snapshot_id)
            delete_files = reader.get_delete_files(table_info.table_id, snap.snapshot_id)

            # Build delete file mapping: data_file_id -> list of delete file paths
            del_map: dict[int, list[tuple[str, int]]] = {}
            for df in delete_files:
                path = reader.resolve_data_file_path(df.path, df.path_is_relative, table_info)
                del_map.setdefault(df.data_file_id, []).append((path, df.delete_count))

            rows: list[tuple[str, int, str | None, int | None]] = []
            for f in data_files:
                data_path = reader.resolve_data_file_path(f.path, f.path_is_relative, table_info)
                del_entries = del_map.get(f.data_file_id, [])
                if del_entries:
                    for del_path, del_count in del_entries:
                        rows.append((data_path, f.file_size_bytes, del_path, del_count))
                else:
                    rows.append((data_path, f.file_size_bytes, None, None))

        if not rows:
            return pl.DataFrame(
                schema={
                    "data_file": pl.String,
                    "data_file_size_bytes": pl.Int64,
                    "delete_file": pl.String,
                    "delete_row_count": pl.Int64,
                }
            )

        return pl.DataFrame(
            {
                "data_file": [r[0] for r in rows],
                "data_file_size_bytes": [r[1] for r in rows],
                "delete_file": [r[2] for r in rows],
                "delete_row_count": [r[3] for r in rows],
            },
            schema={
                "data_file": pl.String,
                "data_file_size_bytes": pl.Int64,
                "delete_file": pl.String,
                "delete_row_count": pl.Int64,
            },
        )

    # ------------------------------------------------------------------
    # Schema / table listing
    # ------------------------------------------------------------------

    def list_schemas(self, *, snapshot_version: int | None = None) -> pl.DataFrame:
        """
        List all schemas in the catalog.

        Returns a DataFrame with columns:
        - ``schema_id`` (Int64)
        - ``schema_name`` (String)
        """
        with self._reader() as reader:
            if snapshot_version is not None:
                snap = reader.get_snapshot_at_version(snapshot_version)
            else:
                snap = reader.get_current_snapshot()
            rows = reader.get_all_schemas(snap.snapshot_id)

        if not rows:
            return pl.DataFrame(schema={"schema_id": pl.Int64, "schema_name": pl.String})

        return pl.DataFrame(
            {
                "schema_id": [r[0] for r in rows],
                "schema_name": [r[1] for r in rows],
            },
            schema={"schema_id": pl.Int64, "schema_name": pl.String},
        )

    def list_tables(
        self,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pl.DataFrame:
        """
        List all tables in a schema.

        Returns a DataFrame with columns:
        - ``table_id`` (Int64)
        - ``table_name`` (String)
        """
        with self._reader() as reader:
            if snapshot_version is not None:
                snap = reader.get_snapshot_at_version(snapshot_version)
            else:
                snap = reader.get_current_snapshot()

            schemas = reader.get_all_schemas(snap.snapshot_id)
            schema_row = None
            for s in schemas:
                if s[1] == schema:
                    schema_row = s
                    break
            if schema_row is None:
                return pl.DataFrame(schema={"table_id": pl.Int64, "table_name": pl.String})

            tables = reader.get_all_tables(schema_row[0], snap.snapshot_id)

        if not tables:
            return pl.DataFrame(schema={"table_id": pl.Int64, "table_name": pl.String})

        return pl.DataFrame(
            {
                "table_id": [r[0] for r in tables],
                "table_name": [r[1] for r in tables],
            },
            schema={"table_id": pl.Int64, "table_name": pl.String},
        )

    # ------------------------------------------------------------------
    # Catalog options
    # ------------------------------------------------------------------

    def options(self) -> pl.DataFrame:
        """
        Get catalog key-value options from ducklake_metadata.

        Equivalent to ``ducklake_options(catalog)``.

        Returns a DataFrame with columns:
        - ``key`` (String)
        - ``value`` (String)
        """
        with self._reader() as reader:
            rows = reader.get_all_metadata()

        if not rows:
            return pl.DataFrame(schema={"key": pl.String, "value": pl.String})

        return pl.DataFrame(
            {
                "key": [r[0] for r in rows],
                "value": [r[1] for r in rows],
            },
            schema={"key": pl.String, "value": pl.String},
        )

    def settings(self) -> pl.DataFrame:
        """
        Get catalog type and data path.

        Equivalent to ``ducklake_settings(catalog)``.

        Returns a DataFrame with columns:
        - ``catalog_type`` (String) — ``"sqlite"`` or ``"postgresql"``
        - ``data_path`` (String)
        """
        with self._reader() as reader:
            catalog_type = reader._backend.__class__.__name__
            if "sqlite" in catalog_type.lower():
                cat_type = "sqlite"
            else:
                cat_type = "postgresql"
            data_path = reader.data_path

        return pl.DataFrame(
            {
                "catalog_type": [cat_type],
                "data_path": [data_path],
            },
            schema={"catalog_type": pl.String, "data_path": pl.String},
        )

    # ------------------------------------------------------------------
    # Change data feed
    # ------------------------------------------------------------------

    def table_insertions(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pl.DataFrame:
        """
        Get rows inserted between two snapshots.

        Equivalent to ``ducklake_table_insertions(catalog, schema, table, start, end)``.

        Reads the Parquet data files that were added in the snapshot range
        ``(start_version, end_version]``.

        Returns a DataFrame with ``snapshot_id`` column plus all table columns.
        """
        with self._reader() as reader:
            snap = reader.get_snapshot_at_version(end_version)
            table_info = reader.get_table(table, schema, snap.snapshot_id)
            all_columns = reader.get_all_columns(table_info.table_id, snap.snapshot_id)
            column_names = [c.column_name for c in all_columns if c.parent_column is None]

            files_with_snap = reader.get_data_files_in_range_with_snapshot(
                table_info.table_id, start_version, end_version
            )

            if not files_with_snap:
                return pl.DataFrame(
                    schema=self._build_cdf_schema(
                        all_columns, [("snapshot_id", pl.Int64)]
                    )
                )

            frames: list[pl.DataFrame] = []
            for file_info, begin_snap in files_with_snap:
                path = reader.resolve_data_file_path(
                    file_info.path, file_info.path_is_relative, table_info
                )
                df = pl.read_parquet(path)
                # Only keep top-level columns that exist in the current schema
                available = [c for c in column_names if c in df.columns]
                df = df.select(available)
                df = df.with_columns(pl.lit(begin_snap).alias("snapshot_id").cast(pl.Int64))
                frames.append(df)

        result = pl.concat(frames, how="diagonal_relaxed")
        # Reorder: snapshot_id first
        cols = ["snapshot_id"] + [c for c in result.columns if c != "snapshot_id"]
        return result.select(cols)

    def table_deletions(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pl.DataFrame:
        """
        Get rows deleted between two snapshots.

        Equivalent to ``ducklake_table_deletions(catalog, schema, table, start, end)``.

        Reads delete files added in the snapshot range ``(start_version, end_version]``,
        then reads the corresponding rows from their associated data files.

        Returns a DataFrame with ``snapshot_id`` column plus all table columns.
        """
        with self._reader() as reader:
            snap = reader.get_snapshot_at_version(end_version)
            table_info = reader.get_table(table, schema, snap.snapshot_id)
            all_columns = reader.get_all_columns(table_info.table_id, snap.snapshot_id)
            column_names = [c.column_name for c in all_columns if c.parent_column is None]
            empty_schema = self._build_cdf_schema(
                all_columns, [("snapshot_id", pl.Int64)]
            )

            del_files = reader.get_delete_files_in_range(
                table_info.table_id, start_version, end_version
            )

            if not del_files:
                return pl.DataFrame(schema=empty_schema)

            frames: list[pl.DataFrame] = []
            for del_info, begin_snap in del_files:
                # Read the delete file (position-delete format: file_path, pos columns)
                del_path = reader.resolve_data_file_path(
                    del_info.path, del_info.path_is_relative, table_info
                )
                del_df = pl.read_parquet(del_path)

                # Get the corresponding data file
                data_file = reader.get_data_file_by_id(del_info.data_file_id)
                if data_file is None:
                    continue

                data_path = reader.resolve_data_file_path(
                    data_file.path, data_file.path_is_relative, table_info
                )
                data_df = pl.read_parquet(data_path)

                # Extract deleted row positions
                # DuckLake delete files use Iceberg position-delete format
                # with row_id_start offset and "pos" column
                if "pos" in del_df.columns:
                    positions = del_df["pos"].to_list()
                    row_id_start = data_file.row_id_start
                    local_positions = [p - row_id_start for p in positions]
                    # Filter to valid positions
                    local_positions = [p for p in local_positions if 0 <= p < len(data_df)]
                    if local_positions:
                        deleted_rows = data_df[local_positions]
                        available = [c for c in column_names if c in deleted_rows.columns]
                        deleted_rows = deleted_rows.select(available)
                        deleted_rows = deleted_rows.with_columns(
                            pl.lit(begin_snap).alias("snapshot_id").cast(pl.Int64)
                        )
                        frames.append(deleted_rows)

        if not frames:
            return pl.DataFrame(schema=empty_schema)

        result = pl.concat(frames, how="diagonal_relaxed")
        cols = ["snapshot_id"] + [c for c in result.columns if c != "snapshot_id"]
        return result.select(cols)

    def table_changes(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pl.DataFrame:
        """
        Get combined change data feed between two snapshots.

        Equivalent to ``ducklake_table_changes(catalog, schema, table, start, end)``.

        Returns a DataFrame with columns:
        - ``snapshot_id`` (Int64)
        - ``change_type`` (String) — one of ``'insert'``, ``'delete'``,
          ``'update_preimage'``, ``'update_postimage'``
        - All table columns

        Updates are detected when both an insertion and a deletion occur in the
        same snapshot. Deletions in that snapshot become ``update_preimage`` and
        insertions become ``update_postimage``.
        """
        insertions = self.table_insertions(table, start_version, end_version, schema=schema)
        deletions = self.table_deletions(table, start_version, end_version, schema=schema)

        has_ins = len(insertions) > 0
        has_del = len(deletions) > 0

        if not has_ins and not has_del:
            # Derive schema from the (empty) insertions/deletions frames
            source = insertions if len(insertions.columns) >= len(deletions.columns) else deletions
            schema_dict: dict[str, pl.DataType] = {"snapshot_id": pl.Int64, "change_type": pl.String}
            for c in source.columns:
                if c != "snapshot_id":
                    schema_dict[c] = source.schema[c]
            return pl.DataFrame(schema=schema_dict)

        # Find snapshots that have both insertions and deletions (potential updates)
        ins_snapshots = set(insertions["snapshot_id"].to_list()) if has_ins else set()
        del_snapshots = set(deletions["snapshot_id"].to_list()) if has_del else set()
        update_snapshots = ins_snapshots & del_snapshots

        frames: list[pl.DataFrame] = []

        if has_ins:
            if update_snapshots:
                # Split insertions into pure inserts and update_postimage
                pure_ins = insertions.filter(~pl.col("snapshot_id").is_in(list(update_snapshots)))
                update_post = insertions.filter(pl.col("snapshot_id").is_in(list(update_snapshots)))

                if len(pure_ins) > 0:
                    frames.append(pure_ins.with_columns(pl.lit("insert").alias("change_type")))
                if len(update_post) > 0:
                    frames.append(update_post.with_columns(pl.lit("update_postimage").alias("change_type")))
            else:
                frames.append(insertions.with_columns(pl.lit("insert").alias("change_type")))

        if has_del:
            if update_snapshots:
                # Split deletions into pure deletes and update_preimage
                pure_del = deletions.filter(~pl.col("snapshot_id").is_in(list(update_snapshots)))
                update_pre = deletions.filter(pl.col("snapshot_id").is_in(list(update_snapshots)))

                if len(pure_del) > 0:
                    frames.append(pure_del.with_columns(pl.lit("delete").alias("change_type")))
                if len(update_pre) > 0:
                    frames.append(update_pre.with_columns(pl.lit("update_preimage").alias("change_type")))
            else:
                frames.append(deletions.with_columns(pl.lit("delete").alias("change_type")))

        if not frames:
            schema_dict = {"snapshot_id": pl.Int64, "change_type": pl.String}
            return pl.DataFrame(schema=schema_dict)

        result = pl.concat(frames, how="diagonal_relaxed")
        # Reorder: snapshot_id, change_type, then rest
        other_cols = [c for c in result.columns if c not in ("snapshot_id", "change_type")]
        return result.select(["snapshot_id", "change_type"] + other_cols)
