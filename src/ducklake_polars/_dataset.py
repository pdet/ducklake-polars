"""DuckLake dataset provider for Polars."""

from __future__ import annotations

import atexit
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

import ducklake_core._storage as storage
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


def _filter_delete_file_by_snapshot(path: str, snapshot_id: int) -> str | None:
    """Filter a cumulative delete file to only include entries up to *snapshot_id*.

    DuckDB v1.5 (catalog v0.4) writes cumulative delete files that contain
    a ``_ducklake_internal_snapshot_id`` column tagging each position with
    the snapshot that deleted it.  When time-traveling, we must strip out
    positions from later snapshots.

    Returns the path to a filtered temp file, the original *path* when no
    filtering is needed, or ``None`` when no positions remain after filtering.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    tbl = storage.read_parquet(path)
    if "_ducklake_internal_snapshot_id" not in tbl.column_names:
        return path

    mask = pc.less_equal(
        tbl.column("_ducklake_internal_snapshot_id"), snapshot_id
    )
    filtered = tbl.filter(mask)

    if filtered.num_rows == 0:
        return None

    if filtered.num_rows == tbl.num_rows:
        # All entries are within the target snapshot — use original file
        # but still drop the internal column for Polars compatibility.
        pass

    # Drop the internal column; Polars expects only file_path + pos
    keep_cols = [
        c for c in filtered.column_names
        if c != "_ducklake_internal_snapshot_id"
    ]
    filtered = filtered.select(keep_cols)

    fd, tmp_path = tempfile.mkstemp(suffix="-delete-filtered.parquet")
    os.close(fd)
    atexit.register(lambda p=tmp_path: _safe_unlink(p))
    pq.write_table(filtered, tmp_path)
    return tmp_path


def _cast_inlined_to_schema(
    df: pl.DataFrame, schema: dict[str, pl.DataType]
) -> pl.DataFrame:
    """Cast inlined data from SQLite types to the catalog schema.

    SQLite stores all integers as BIGINT (Int64), booleans as 0/1 integers,
    dates/timestamps as strings, and decimals as strings.  This function
    casts each column to the expected Polars type so the temp Parquet file
    matches the dataset schema reported by ``schema()``.
    """
    cast_exprs: list[pl.Expr] = []
    for col_name in df.columns:
        if col_name not in schema:
            cast_exprs.append(pl.col(col_name))
            continue
        target = schema[col_name]
        current = df[col_name].dtype
        if current == target:
            cast_exprs.append(pl.col(col_name))
        elif isinstance(target, pl.Boolean) and current in (pl.Int64, pl.Int32, pl.Int8):
            cast_exprs.append(pl.col(col_name).cast(pl.Boolean))
        elif isinstance(target, pl.Datetime):
            if current in (pl.String, pl.Utf8):
                # SQLite stores timestamps as strings — parse them
                cast_exprs.append(
                    pl.col(col_name).str.to_datetime(time_unit="us", strict=False)
                )
            else:
                # DuckDB writes all timestamps as microseconds in Parquet
                cast_exprs.append(pl.col(col_name).cast(pl.Datetime("us")))
        elif isinstance(target, pl.Date) and current in (pl.String, pl.Utf8):
            cast_exprs.append(pl.col(col_name).str.to_date(strict=False))
        elif isinstance(target, pl.Time) and current in (pl.String, pl.Utf8):
            cast_exprs.append(pl.col(col_name).str.to_time(strict=False))
        else:
            try:
                cast_exprs.append(pl.col(col_name).cast(target))
            except Exception:
                cast_exprs.append(pl.col(col_name))
    return df.select(cast_exprs)


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
    """Check if any column has been renamed, has a type change, drop+re-add conflict, or struct field rename."""
    top_history = _top_level_history(history)
    top_current = [c for c in current_columns if c.parent_column is None]
    current_names = {c.column_id: c.column_name for c in top_current}
    current_types = {c.column_id: c.column_type for c in top_current}
    current_name_set = set(current_names.values())

    # Check top-level renames, type changes, and drop+re-add conflicts
    for entry in top_history:
        if entry.column_id in current_names:
            if entry.column_name != current_names[entry.column_id]:
                return True
            # Check type changes (same column_id, different type)
            if entry.column_type and entry.column_id in current_types:
                if entry.column_type != current_types[entry.column_id]:
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


def _get_type_cast_map(
    file_begin_snapshot: int,
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
    all_columns: list[ColumnInfo],
) -> dict[str, tuple[pl.DataType, pl.DataType]]:
    """Get {column_name: (old_polars_type, current_polars_type)} for columns with type changes.

    Only returns entries where the column type at file write time differs from
    the current column type.
    """
    top_history = _top_level_history(history)
    top_current = [c for c in current_columns if c.parent_column is None]
    current_types: dict[int, str] = {c.column_id: c.column_type for c in top_current}
    current_names: dict[int, str] = {c.column_id: c.column_name for c in top_current}

    result: dict[str, tuple[pl.DataType, pl.DataType]] = {}
    for entry in top_history:
        if entry.column_id not in current_types:
            continue
        if not _is_active_at(entry, file_begin_snapshot):
            continue
        if entry.column_type and entry.column_type != current_types[entry.column_id]:
            col_name = current_names[entry.column_id]
            old_pl_type = resolve_column_type(
                entry.column_id, entry.column_type, all_columns
            )
            cur_pl_type = resolve_column_type(
                entry.column_id, current_types[entry.column_id], all_columns
            )
            result[col_name] = (old_pl_type, cur_pl_type)
    return result


def _get_physical_type_key(
    file_begin_snapshot: int,
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
) -> frozenset[tuple[int, str]]:
    """Get a hashable key representing the physical column types at a given snapshot.

    Returns a frozenset of (column_id, physical_type) for columns whose type
    at the file's snapshot differs from the current type.
    """
    top_history = _top_level_history(history)
    top_current = [c for c in current_columns if c.parent_column is None]
    current_types: dict[int, str] = {c.column_id: c.column_type for c in top_current}

    diffs: list[tuple[int, str]] = []
    for entry in top_history:
        if entry.column_id not in current_types:
            continue
        if not _is_active_at(entry, file_begin_snapshot):
            continue
        if entry.column_type and entry.column_type != current_types[entry.column_id]:
            diffs.append((entry.column_id, entry.column_type))
    return frozenset(diffs)


def _group_files_by_rename_map(
    files: list[FileInfo],
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
) -> list[tuple[dict[str, str], Any, list[FileInfo]]]:
    """Group files by their rename map, struct field renames, and type changes.

    Returns list of (rename_map, struct_field_renames, files_list) tuples.
    struct_field_renames is ``dict[str, list[str]] | None``.
    """
    groups: dict[tuple, list[FileInfo]] = defaultdict(list)
    rename_maps: dict[tuple, dict[str, str]] = {}
    struct_renames_map: dict[tuple, Any] = {}

    for f in files:
        rmap = _get_rename_map(f.begin_snapshot, history, current_columns)
        srenames = _get_struct_field_renames(f.begin_snapshot, history, current_columns)
        type_key = _get_physical_type_key(f.begin_snapshot, history, current_columns)

        rmap_key = frozenset(rmap.items())
        sren_key = None
        if srenames:
            sren_key = frozenset((k, tuple(v)) for k, v in srenames.items())
        key = (rmap_key, sren_key, type_key)

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
        snapshot_id: int | None = None,
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
                categorical_to_string="allow",
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
                    # Filter cumulative delete files when time-traveling.
                    # v0.4 catalogs set partial_max to the highest snapshot
                    # stored in the delete file; when partial_max > target
                    # snapshot we must strip out future positions.
                    if (
                        df.partial_max is not None
                        and snapshot_id is not None
                        and df.partial_max > snapshot_id
                    ):
                        path = _filter_delete_file_by_snapshot(path, snapshot_id)
                        if path is None:
                            continue  # no deletes at this snapshot
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
                    # The dataset scan resolver only accepts bare Parquet
                    # SCAN nodes; df.lazy() is a DF node which is rejected.
                    # Write inlined data to a temp Parquet file.
                    # Cast to the catalog schema first — SQLite stores all
                    # ints as Int64, dates as strings, etc.
                    schema_dict = self._build_schema_from_columns(all_columns)
                    inlined = _cast_inlined_to_schema(inlined, schema_dict)
                    fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
                    os.close(fd)
                    atexit.register(lambda p=tmp_path: _safe_unlink(p))
                    inlined.write_parquet(tmp_path)
                    return scan_parquet(
                        tmp_path,
                        missing_columns="insert",
                        extra_columns="ignore",
                    ), version_key
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
                    snapshot_id=snapshot.snapshot_id,
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
                        snapshot_id=snapshot.snapshot_id,
                    )

                    # Build a per-group schema so Polars knows about all
                    # current columns (even if the first file lacks some).
                    group_schema = dict(catalog_schema)

                    # Compute type casts for this group early so we can
                    # adjust the physical schema before scanning.
                    type_casts = _get_type_cast_map(
                        group_files_list[0].begin_snapshot,
                        history, all_columns, all_columns,
                    )

                    # Replace current types with old (physical) types in the
                    # schema so scan_parquet matches the Parquet file layout.
                    if type_casts:
                        for col_name, (old_type, _new_type) in type_casts.items():
                            if col_name in group_schema:
                                group_schema[col_name] = old_type

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

                    # Apply type casts for columns whose type changed after the file was written
                    if type_casts:
                        cast_exprs = []
                        for col_name, (_old_type, new_type) in type_casts.items():
                            if col_name in df.columns:
                                cast_exprs.append(pl.col(col_name).cast(new_type))
                        if cast_exprs:
                            df = df.with_columns(cast_exprs)

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

            # If there's inlined data, combine via temp file (the scan
            # resolver only accepts bare Parquet SCAN nodes, so we must
            # collect the Parquet scan, append inlined rows, and write
            # everything to a single temp file).
            inlined = reader.read_inlined_data(
                table.table_id,
                snapshot.snapshot_id,
                column_names,
            )
            if inlined is not None and not inlined.is_empty():
                schema_dict = self._build_schema_from_columns(all_columns)
                inlined = _cast_inlined_to_schema(inlined, schema_dict)
                combined = pl.concat(
                    [lf.collect(), inlined], how="diagonal_relaxed"
                )
                fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
                os.close(fd)
                atexit.register(lambda p=tmp_path: _safe_unlink(p))
                combined.write_parquet(tmp_path)
                lf = scan_parquet(
                    tmp_path,
                    missing_columns="insert",
                    extra_columns="ignore",
                )

            return lf, version_key
