"""DuckLake dataset provider for Polars."""

from __future__ import annotations

import atexit
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from ducklake_polars._catalog import DuckLakeCatalogReader
from ducklake_polars._schema import resolve_column_type
from ducklake_polars._stats import build_table_statistics

if TYPE_CHECKING:
    from datetime import datetime

    from polars import LazyFrame
    from polars.schema import Schema

    from ducklake_polars._catalog import (
        ColumnHistoryEntry,
        ColumnInfo,
        DeleteFileInfo,
        FileInfo,
        FilePartitionValue,
        PartitionColumnDef,
        TableInfo,
    )


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _is_active_at(entry: ColumnHistoryEntry, snapshot: int) -> bool:
    """Check if a column history entry was active at a given snapshot."""
    return entry.begin_snapshot <= snapshot and (
        entry.end_snapshot is None or entry.end_snapshot > snapshot
    )


def _top_level_history(history: list[ColumnHistoryEntry]) -> list[ColumnHistoryEntry]:
    """Filter history to top-level columns only."""
    return [e for e in history if e.parent_column is None]


def _has_renames(
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
) -> bool:
    """Check if any column has been renamed, has a drop+re-add conflict, or struct field rename."""
    top_history = _top_level_history(history)
    top_current = [c for c in current_columns if c.parent_column is None]
    current_names = {c.column_id: c.column_name for c in top_current}
    current_name_set = set(current_names.values())

    # Check top-level renames and drop+re-add conflicts
    for entry in top_history:
        if entry.column_id in current_names:
            if entry.column_name != current_names[entry.column_id]:
                return True
        elif entry.column_name in current_name_set:
            return True

    # Check struct field renames: a struct child that ended at the same snapshot
    # another child with the same parent began → field rename
    struct_parents = {c.column_id for c in current_columns if c.column_type == "struct"}
    if struct_parents:
        child_history = [
            e for e in history
            if e.parent_column in struct_parents
        ]
        # Build lookup: (parent_column, begin_snapshot) → column entries
        begin_lookup: dict[tuple[int, int], list[ColumnHistoryEntry]] = defaultdict(list)
        for entry in child_history:
            begin_lookup[(entry.parent_column, entry.begin_snapshot)].append(entry)
        for entry in child_history:
            if entry.end_snapshot is not None:
                for other in begin_lookup.get((entry.parent_column, entry.end_snapshot), ()):
                    if other.column_id != entry.column_id and other.column_name != entry.column_name:
                        return True

    return False


def _get_physical_name(
    column_id: int,
    file_begin_snapshot: int,
    history: list[ColumnHistoryEntry],
) -> str | None:
    """Get the physical column name that was active when the file was written."""
    for entry in history:
        if entry.column_id != column_id:
            continue
        if _is_active_at(entry, file_begin_snapshot):
            return entry.column_name
    return None


def _get_rename_map(
    file_begin_snapshot: int,
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
) -> dict[str, str]:
    """Get {physical_name -> target_name} for top-level columns that differ.

    Handles two cases:
    1. Column renames: physical name differs from current name (same column_id).
    2. Drop+re-add conflicts: a dropped column's physical name collides with a
       new column's name (different column_id, same name). The old physical name
       is mapped to a dummy so it becomes an "extra" column that gets ignored.
    """
    top_history = _top_level_history(history)
    top_current = [c for c in current_columns if c.parent_column is None]
    rename_map: dict[str, str] = {}
    current_ids = {c.column_id for c in top_current}
    current_name_set = {c.column_name for c in top_current}

    # Case 1: column renames
    for col in top_current:
        physical = _get_physical_name(col.column_id, file_begin_snapshot, top_history)
        if physical is not None and physical != col.column_name:
            rename_map[physical] = col.column_name

    # Case 2: dropped column name conflicts
    for entry in top_history:
        if entry.column_id in current_ids:
            continue  # Not dropped
        if _is_active_at(entry, file_begin_snapshot):
            if entry.column_name in current_name_set and entry.column_name not in rename_map:
                rename_map[entry.column_name] = f"__ducklake_dropped_{entry.column_id}__"

    return rename_map


def _get_struct_field_renames(
    file_begin_snapshot: int,
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
) -> dict[str, list[str]] | None:
    """Get struct columns needing field renames for files written at a given snapshot.

    Returns ``{struct_col_name: [current_field_names_in_order]}`` for struct columns
    where the physical field names differ from the current field names, or ``None``
    if no struct field renames are needed.

    The returned field name lists are used with ``struct.rename_fields()``.
    """
    child_history = [e for e in history if e.parent_column is not None]
    if not child_history:
        return None

    # Find struct columns (top-level only)
    struct_cols = {
        c.column_id: c
        for c in current_columns
        if c.column_type == "struct" and c.parent_column is None
    }
    if not struct_cols:
        return None

    result: dict[str, list[str]] = {}
    for struct_id, struct_col in struct_cols.items():
        # Current children (ordered)
        current_children = sorted(
            [c for c in current_columns if c.parent_column == struct_id],
            key=lambda c: c.column_order,
        )
        # Physical children at file's snapshot (ordered)
        physical_children = sorted(
            [e for e in child_history if e.parent_column == struct_id and _is_active_at(e, file_begin_snapshot)],
            key=lambda e: e.column_order,
        )

        if len(current_children) != len(physical_children):
            continue  # Different field count — handled by missing/extra_struct_fields

        physical_names = [e.column_name for e in physical_children]
        current_names = [c.column_name for c in current_children]

        if physical_names != current_names:
            result[struct_col.column_name] = current_names

    return result if result else None


def _group_files_by_rename_map(
    files: list[FileInfo],
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
) -> list[tuple[dict[str, str], Any, list[FileInfo]]]:
    """Group files by their rename map and struct field renames.

    Returns list of (rename_map, struct_field_renames, files_list) tuples.
    struct_field_renames is ``dict[str, list[str]] | None``.
    """
    groups: dict[tuple, list[FileInfo]] = defaultdict(list)
    rename_maps: dict[tuple, dict[str, str]] = {}
    struct_renames_map: dict[tuple, Any] = {}

    for f in files:
        rmap = _get_rename_map(f.begin_snapshot, history, current_columns)
        srenames = _get_struct_field_renames(f.begin_snapshot, history, current_columns)

        rmap_key = frozenset(rmap.items())
        sren_key = None
        if srenames:
            sren_key = frozenset((k, tuple(v)) for k, v in srenames.items())
        key = (rmap_key, sren_key)

        groups[key].append(f)
        if key not in rename_maps:
            rename_maps[key] = rmap
            struct_renames_map[key] = srenames

    return [
        (rename_maps[k], struct_renames_map[k], group)
        for k, group in groups.items()
    ]


def _build_partition_values_for_stats(
    partition_columns: list[PartitionColumnDef],
    file_partition_values: list[FilePartitionValue],
) -> dict[int, dict[int, str | None]]:
    """Build a lookup of {data_file_id: {column_id: partition_value}}.

    Only includes partition columns that use the identity transform.
    """
    # Map partition_key_index -> column_id
    identity_key_to_col: dict[int, int] = {}
    for pc in partition_columns:
        if pc.transform == "identity":
            identity_key_to_col[pc.partition_key_index] = pc.column_id

    if not identity_key_to_col:
        return {}

    result: dict[int, dict[int, str | None]] = {}
    for fpv in file_partition_values:
        if fpv.partition_key_index in identity_key_to_col:
            col_id = identity_key_to_col[fpv.partition_key_index]
            result.setdefault(fpv.data_file_id, {})[col_id] = fpv.partition_value

    return result


@dataclass
class DuckLakeDataset:
    """
    Dataset provider for DuckLake tables.

    Implements the PythonDatasetProvider interface expected by
    Polars' PyLazyFrame.new_from_dataset_object().
    """

    metadata_path: str
    table_name: str
    schema_name: str
    snapshot_version: int | None = None
    snapshot_time: datetime | str | None = None
    data_path_override: str | None = None

    def __post_init__(self) -> None:
        if self.snapshot_version is not None and self.snapshot_time is not None:
            msg = "Cannot specify both snapshot_version and snapshot_time"
            raise ValueError(msg)

    def _get_reader(self) -> DuckLakeCatalogReader:
        return DuckLakeCatalogReader(
            self.metadata_path,
            data_path_override=self.data_path_override,
        )

    def _resolve_snapshot(self, reader: DuckLakeCatalogReader) -> Any:
        if self.snapshot_version is not None:
            return reader.get_snapshot_at_version(self.snapshot_version)
        if self.snapshot_time is not None:
            return reader.get_snapshot_at_time(self.snapshot_time)
        return reader.get_current_snapshot()

    #
    # PythonDatasetProvider interface
    #

    @staticmethod
    def _build_schema_from_columns(all_columns: list[ColumnInfo]) -> dict[str, pl.DataType]:
        """Build the Polars schema dict from the column hierarchy."""
        top_level = [c for c in all_columns if c.parent_column is None]

        schema_dict: dict[str, pl.DataType] = {}
        for col in top_level:
            schema_dict[col.column_name] = resolve_column_type(
                col.column_id, col.column_type, all_columns
            )
        return schema_dict

    @staticmethod
    def _apply_physical_struct_fields(
        schema: dict[str, pl.DataType],
        struct_field_renames: dict[str, list[str]],
        file_begin_snapshot: int,
        history: list[ColumnHistoryEntry],
        all_columns: list[ColumnInfo],
        rename_map: dict[str, str] | None = None,
    ) -> dict[str, pl.DataType]:
        """Replace struct column dtypes with physical field names.

        For struct columns with field renames, builds a Struct dtype using the
        physical field names (matching the Parquet file) but the current field types.

        ``rename_map`` is the top-level column rename map; when a struct column
        was also renamed at the top level, the schema key is the *physical* name
        while ``struct_field_renames`` uses the *current* name.
        """
        result = dict(schema)
        # Reverse rename_map: current_name -> physical_name (schema key)
        reverse = {v: k for k, v in rename_map.items()} if rename_map else {}

        for col_name, _current_names in struct_field_renames.items():
            # col_name is the current name; schema key may be the physical name
            schema_key = reverse.get(col_name, col_name)
            if schema_key not in result:
                continue
            current_dtype = result[schema_key]
            if not isinstance(current_dtype, pl.Struct):
                continue
            # Find the struct column by current name in all_columns
            # (all_columns is snapshot-filtered, so no risk of matching dropped cols)
            struct_col = next(
                (c for c in all_columns if c.column_name == col_name
                 and c.parent_column is None and c.column_type == "struct"),
                None,
            )
            if struct_col is None:
                continue
            physical_children = sorted(
                [
                    e for e in history
                    if e.parent_column == struct_col.column_id
                    and _is_active_at(e, file_begin_snapshot)
                ],
                key=lambda e: e.column_order,
            )
            current_fields = list(current_dtype.fields)
            if len(physical_children) != len(current_fields):
                continue
            # Build physical struct type with physical field names but current types
            physical_struct = pl.Struct([
                pl.Field(phys.column_name, field.dtype)
                for phys, field in zip(physical_children, current_fields)
            ])
            result[schema_key] = physical_struct
        return result

    def schema(self) -> Schema:
        """Return the table schema as a Polars Schema."""
        with self._get_reader() as reader:
            snapshot = self._resolve_snapshot(reader)
            table = reader.get_table(self.table_name, self.schema_name, snapshot.snapshot_id)
            all_columns = reader.get_all_columns(table.table_id, snapshot.snapshot_id)
            return pl.Schema(self._build_schema_from_columns(all_columns))

    @staticmethod
    def _build_scan_kwargs(
        group_files: list[FileInfo],
        delete_files: list[DeleteFileInfo],
        reader: DuckLakeCatalogReader,
        table: TableInfo,
        columns: list[ColumnInfo],
        filter_columns: list[str] | None,
        partition_values: dict[int, dict[int, str | None]] | None,
    ) -> dict[str, Any]:
        """Build scan_parquet kwargs for a group of files."""
        kwargs: dict[str, Any] = {
            "missing_columns": "insert",
            "extra_columns": "ignore",
            "cast_options": pl.ScanCastOptions(
                integer_cast="upcast",
                float_cast="upcast",
                missing_struct_fields="insert",
                extra_struct_fields="ignore",
            ),
        }

        # Build deletion files mapping for this group
        if delete_files:
            file_id_to_idx = {
                f.data_file_id: i for i, f in enumerate(group_files)
            }
            deletion_files_map: dict[int, list[str]] = {}
            for df in delete_files:
                idx = file_id_to_idx.get(df.data_file_id)
                if idx is not None:
                    path = reader.resolve_data_file_path(df.path, df.path_is_relative, table)
                    deletion_files_map.setdefault(idx, []).append(path)
            if deletion_files_map:
                kwargs["_deletion_files"] = (
                    "iceberg-position-delete",
                    deletion_files_map,
                )

        # Build table statistics for this group
        if filter_columns:
            file_ids = [f.data_file_id for f in group_files]
            col_ids = [
                c.column_id
                for c in columns
                if c.column_name in filter_columns
            ]
            stats = reader.get_column_stats(table.table_id, file_ids, col_ids)
            table_statistics = build_table_statistics(
                group_files, stats, columns, filter_columns,
                partition_values=partition_values,
            )
            if table_statistics is not None:
                kwargs["_table_statistics"] = table_statistics

        return kwargs

    def to_dataset_scan(
        self,
        *,
        existing_resolved_version_key: str | None = None,
        limit: int | None = None,
        projection: list[str] | None = None,
        filter_columns: list[str] | None = None,
        pyarrow_predicate: str | None = None,
    ) -> tuple[LazyFrame, str] | None:
        """
        Resolve metadata and construct a scan_parquet LazyFrame.

        Returns (LazyFrame, version_key) or None if the version hasn't changed.

        Note: ``limit``, ``projection``, and ``pyarrow_predicate`` are part of the
        PythonDatasetProvider interface but are not used here; Polars handles
        projection and filtering on the resulting LazyFrame.
        """
        from polars.io.parquet.functions import scan_parquet

        with self._get_reader() as reader:
            snapshot = self._resolve_snapshot(reader)
            version_key = str(snapshot.snapshot_id)

            # Short-circuit if version hasn't changed
            if (
                existing_resolved_version_key is not None
                and existing_resolved_version_key == version_key
            ):
                return None

            table = reader.get_table(
                self.table_name, self.schema_name, snapshot.snapshot_id
            )
            all_columns = reader.get_all_columns(table.table_id, snapshot.snapshot_id)
            columns = [c for c in all_columns if c.parent_column is None]
            column_names = [c.column_name for c in columns]

            # Get data files
            data_files = reader.get_data_files(table.table_id, snapshot.snapshot_id)

            if not data_files:
                # No data files - check for inlined data
                inlined = reader.read_inlined_data(
                    table.table_id,
                    snapshot.snapshot_id,
                    column_names,
                )
                if inlined is not None and not inlined.is_empty():
                    return inlined.lazy(), version_key
                # Empty table - return scan_parquet with empty list
                schema_dict = self._build_schema_from_columns(all_columns)
                return scan_parquet(
                    [],
                    schema=schema_dict,
                ), version_key

            # Get delete files
            delete_files = reader.get_delete_files(table.table_id, snapshot.snapshot_id)

            # Fetch partition values for statistics supplementation
            partition_values: dict[int, dict[int, str | None]] | None = None
            if filter_columns:
                try:
                    part_info = reader.get_partition_info(table.table_id, snapshot.snapshot_id)
                    if part_info is not None:
                        part_cols = reader.get_partition_columns(part_info.partition_id, table.table_id)
                        # Check if any partition columns overlap with filter columns
                        col_id_to_name = {c.column_id: c.column_name for c in columns}
                        part_col_names = {col_id_to_name[pc.column_id] for pc in part_cols if pc.column_id in col_id_to_name}
                        if part_col_names & set(filter_columns):
                            file_ids = [f.data_file_id for f in data_files]
                            fpvs = reader.get_file_partition_values(table.table_id, file_ids)
                            partition_values = _build_partition_values_for_stats(
                                part_cols, fpvs,
                            )
                except Exception as e:
                    if not reader._backend.is_table_not_found(e):
                        raise

            # Detect column renames and struct field renames
            history = reader.get_column_history(table.table_id)
            has_rename = _has_renames(history, all_columns)

            if not has_rename:
                # Fast path: no renames, existing code
                sources = [
                    reader.resolve_data_file_path(f.path, f.path_is_relative, table)
                    for f in data_files
                ]

                kwargs = self._build_scan_kwargs(
                    data_files, delete_files, reader, table,
                    columns, filter_columns, partition_values,
                )

                lf = scan_parquet(sources, **kwargs)
            else:
                # Rename path: the dataset scan resolver only accepts bare
                # Parquet SCAN nodes.  We collect each file group eagerly,
                # apply renames, write the combined result to a temporary
                # Parquet file, and return scan_parquet on that file.
                groups = _group_files_by_rename_map(data_files, history, all_columns)
                catalog_schema = self._build_schema_from_columns(all_columns)

                group_dfs: list[pl.DataFrame] = []
                for rename_map, struct_field_renames, group_files_list in groups:
                    sources = [
                        reader.resolve_data_file_path(f.path, f.path_is_relative, table)
                        for f in group_files_list
                    ]

                    kwargs = self._build_scan_kwargs(
                        group_files_list, delete_files, reader, table,
                        columns, filter_columns, partition_values,
                    )

                    # Build a per-group schema so Polars knows about all
                    # current columns (even if the first file lacks some).
                    group_schema = dict(catalog_schema)

                    if rename_map:
                        # For groups with renames, build a physical schema:
                        # swap current names for their physical names.
                        # Columns whose current name is a rename_map KEY are
                        # drop+re-add conflicts — the physical column is a
                        # different (dropped) column with possibly a different
                        # type.  Exclude them; extra_columns="ignore" skips the
                        # physical column, and diagonal_relaxed concat fills the
                        # current column with NULL.
                        reverse = {v: k for k, v in rename_map.items()}
                        phys_schema: dict[str, pl.DataType] = {}
                        for name, dtype in group_schema.items():
                            if name in reverse:
                                phys_name = reverse[name]
                                if not phys_name.startswith("__ducklake_dropped_"):
                                    phys_schema[phys_name] = dtype
                            elif name in rename_map:
                                pass  # drop+re-add conflict — skip
                            else:
                                phys_schema[name] = dtype
                        group_schema = phys_schema

                    if struct_field_renames:
                        # Replace struct dtypes with physical field names so
                        # Polars matches the Parquet file layout.  After
                        # collection we rename fields back to current names.
                        group_schema = self._apply_physical_struct_fields(
                            group_schema,
                            struct_field_renames,
                            group_files_list[0].begin_snapshot,
                            history,
                            all_columns,
                            rename_map=rename_map or None,
                        )

                    kwargs["schema"] = group_schema

                    df = scan_parquet(sources, **kwargs).collect()

                    # Apply struct field renames
                    if struct_field_renames:
                        rename_exprs = [
                            pl.col(col_name).struct.rename_fields(new_names)
                            for col_name, new_names in struct_field_renames.items()
                            if col_name in df.columns
                        ]
                        if rename_exprs:
                            df = df.with_columns(rename_exprs)

                    # Apply top-level column renames
                    if rename_map:
                        applicable = {k: v for k, v in rename_map.items() if k in df.columns}
                        if applicable:
                            df = df.rename(applicable)

                    # Drop columns not in the current schema
                    keep = [n for n in column_names if n in df.columns]
                    if len(keep) < len(df.columns):
                        df = df.select(keep)
                    group_dfs.append(df)

                if len(group_dfs) == 1:
                    combined = group_dfs[0]
                else:
                    combined = pl.concat(group_dfs, how="diagonal_relaxed")

                # Write to temp file and scan it (resolver needs Parquet SCAN)
                fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
                os.close(fd)
                atexit.register(lambda p=tmp_path: _safe_unlink(p))
                combined.write_parquet(tmp_path)

                lf = scan_parquet(
                    tmp_path,
                    missing_columns="insert",
                    extra_columns="ignore",
                    cast_options=pl.ScanCastOptions(
                        integer_cast="upcast",
                        float_cast="upcast",
                        missing_struct_fields="insert",
                        extra_struct_fields="ignore",
                    ),
                )

            # If there's inlined data, we need to combine it
            inlined = reader.read_inlined_data(
                table.table_id,
                snapshot.snapshot_id,
                column_names,
            )
            if inlined is not None and not inlined.is_empty():
                lf = pl.concat([lf, inlined.lazy()], how="diagonal_relaxed")

            return lf, version_key
