"""Module with union_full geooperations."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias, get_args

import geofileops as gfo
from geofileops import GeometryType, LayerInfo, fileops
from geofileops.helpers import _general_helper
from geofileops.util import _io_util, _ogr_sql_util
from geofileops.util._geofileinfo import GeofileInfo
from geofileops.util._geoops_sql import (
    _subdivide_layer,
    _validate_params_single_layer,
    delete_duplicate_geometries,
    difference,
    intersection,
    join_by_location,
    select,
)

UnionFullSelfTypes: TypeAlias = Literal[
    "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS",
    "NO_INTERSECTIONS_ATTRIBUTE_LISTS",
    "REPEATED_INTERSECTIONS",
]


def union_full_self(
    input_path: Path,
    output_path: Path,
    union_type: str,
    input_layer: str | LayerInfo | None = None,
    output_layer: str | None = None,
    columns: list[str] | None = None,
    explodecollections: bool = False,
    gridsize: float = 0.0,
    where_post: str | None = None,
    nb_parallel: int = -1,
    batchsize: int = -1,
    subdivide_coords: int = 2000,
    force: bool = False,
    output_with_spatial_index: bool | None = None,
):
    # Because the calculations of the intermediate results will be towards temp files,
    # we need to do some additional init + checks here...
    if subdivide_coords < 0:
        raise ValueError("subdivide_coords < 0 is not allowed")
    if union_type not in get_args(UnionFullSelfTypes):
        raise ValueError(f"union_type should be one of {get_args(UnionFullSelfTypes)}")

    operation_name = "union_full_self"
    logger = logging.getLogger(f"geofileops.{operation_name}")

    if _io_util.output_exists(path=output_path, remove_if_exists=force):
        return

    input_layer, output_layer = _validate_params_single_layer(
        input_path=input_path,
        output_path=output_path,
        input_layer=input_layer,
        output_layer=output_layer,
        operation_name=operation_name,
    )

    # Determine output_geometrytype
    force_output_geometrytype = input_layer.geometrytype
    if explodecollections:
        force_output_geometrytype = force_output_geometrytype.to_singletype
    elif force_output_geometrytype is not GeometryType.POINT:
        # If explodecollections is False and the input type is not point, force the
        # output type to multi, because difference can cause eg. polygons to be split to
        # multipolygons.
        force_output_geometrytype = force_output_geometrytype.to_multitype

    start_time = datetime.now()
    with _general_helper.create_gfo_tmp_dir(operation_name) as tmp_dir:
        # Prepare the input files
        logger.info("Step 1 of 4: prepare input file")
        input_subdivided_path = _subdivide_layer(
            path=input_path,
            layer=input_layer,
            output_path=tmp_dir / "subdivided/input_layer.gpkg",
            subdivide_coords=subdivide_coords,
            nb_parallel=nb_parallel,
            batchsize=batchsize,
            operation_prefix=f"{operation_name}/",
            tmp_basedir=tmp_dir,
        )
        if input_subdivided_path is None:
            # Hardcoded optimization: root means that no subdivide was needed
            input_subdivided_path = Path("/")

        # Loop until all intersections are gone...
        logger.info("Step 2 of 4: create a 'flat union' layer (=no intersections)")
        non_intersecting_path = tmp_dir / "union_self_non_intersecting.gpkg"
        if union_type == "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS":
            columns_loop = list(input_layer.columns) if columns is None else columns
        else:
            columns_loop = []

        loop_id = 0
        input_loop_path = input_path
        input_loop_layer: str | LayerInfo = input_layer
        input_subdivided_cur_path: Path | None = input_subdivided_path
        while True:
            logger.info(f"  -> create 'flat union' layer, pass {loop_id}")
            if loop_id == 0:
                # In the first loop, include the original fid column
                input1_columns = columns_loop
                input2_columns = input1_columns
                input1_columns_prefix = f"is{loop_id}_"
                input2_columns_prefix = f"is{loop_id + 1}_"

            elif loop_id == 1:
                # In the second loop, we need to include the original fid column
                input1_columns = None  # all columns
                input2_columns = [
                    f"is{loop_id}_{col}" for col in columns_loop
                ]  # only the "right" columns
                input1_columns_prefix = ""
                input2_columns_prefix = f"is{loop_id + 1}_"

                # After the 0th loop, we cannot use the subdivided input anymore
                input_subdivided_cur_path = None

                # The input_loop layer becomes the output layer after 1st loop
                input_loop_layer = output_layer

            else:
                # In the next loops, no use to keep columns anymore
                input1_columns = []
                input2_columns = []

            # Parts of geometries that don't intersect in the input layer are ready for
            # the output.
            # Remark: subdivide_coords=0, because the input is already subdivided.
            diff_output_path = tmp_dir / f"diff_output_{loop_id}.gpkg"
            difference(
                input1_path=input_loop_path,
                input2_path=input_loop_path,
                output_path=diff_output_path,
                overlay_self=True,
                input1_layer=input_loop_layer,
                input1_columns=input1_columns,
                input_columns_prefix=input1_columns_prefix,
                input2_layer=input_loop_layer,
                output_layer=output_layer,
                explodecollections=explodecollections,
                gridsize=gridsize,
                where_post=where_post,
                nb_parallel=nb_parallel,
                batchsize=batchsize,
                subdivide_coords=0,
                force=force,
                output_with_spatial_index=False,
                operation_prefix=f"{operation_name}/",
                tmp_basedir=tmp_dir,
                input1_subdivided_path=input_subdivided_cur_path,
                input2_subdivided_path=input_subdivided_cur_path,
            )

            if loop_id == 0:
                # First loop, so we can just rename
                gfo.move(diff_output_path, non_intersecting_path)
            else:
                # Append
                fileops.copy_layer(
                    src=diff_output_path,
                    dst=non_intersecting_path,
                    src_layer=output_layer,
                    dst_layer=output_layer,
                    write_mode="append_add_fields",
                    force_output_geometrytype=force_output_geometrytype,
                )

            # Determine the parts of geometries in the input layer that intersect.
            # Remark: subdivide_coords=0, because the input is already subdivided.
            intersection_output_path = tmp_dir / f"intersection_output_{loop_id}.gpkg"
            intersection(
                input1_path=input_loop_path,
                input2_path=input_loop_path,
                output_path=intersection_output_path,
                overlay_self=True,
                include_duplicates=False,
                input1_layer=input_loop_layer,
                input1_columns=input1_columns,
                input1_columns_prefix=input1_columns_prefix,
                input2_layer=input_loop_layer,
                input2_columns=input2_columns,
                input2_columns_prefix=input2_columns_prefix,
                output_layer=output_layer,
                explodecollections=explodecollections,
                gridsize=gridsize,
                where_post=None,
                nb_parallel=nb_parallel,
                batchsize=batchsize,
                subdivide_coords=0,
                force=force,
                output_with_spatial_index=False,
                operation_prefix=f"{operation_name}/",
                tmp_basedir=tmp_dir,
                input1_subdivided_path=input_subdivided_cur_path,
                input2_subdivided_path=input_subdivided_cur_path,
            )

            # If the intersection output is empty, we are ready...
            inters_info = gfo.get_layerinfo(path=intersection_output_path)
            if inters_info.featurecount == 0:
                attributes_in_flat_union = True if loop_id <= 1 else False
                break

            # Delete duplicates from the intersections before starting next loop.
            deldups_path = tmp_dir / f"{intersection_output_path.stem}_no-dups.gpkg"
            delete_duplicate_geometries(
                input_path=intersection_output_path,
                output_path=deldups_path,
                input_layer=output_layer,
                output_layer=output_layer,
                columns=None,
                priority_column=None,
                priority_ascending=True,
                explodecollections=False,
                keep_empty_geoms=False,
                where_post=None,
                nb_parallel=nb_parallel,
                batchsize=batchsize,
                force=force,
                operation_prefix=f"{operation_name}/",
                tmp_basedir=tmp_dir,
            )
            intersection_output_path = deldups_path

            # Init for the next loop
            # if intersection_output_prev_path is not None:
            #    gfo.remove(intersection_output_prev_path)
            input_loop_path = intersection_output_path
            loop_id += 1

        logger.info("Step 3 of 4: combine the attributes with the 'flat union' layer")
        if union_type == "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS" and columns == []:
            # No output attribute columns needed -> ready
            output_tmp_path = non_intersecting_path

        elif (
            union_type == "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS"
            and attributes_in_flat_union
        ):
            # If the attributes are already in the result of the "flat union", not
            # needed to join them.
            output_tmp_path = non_intersecting_path

        else:
            # Join the "flat union" with the original input layer to:
            #   - add the attributes of the input layer
            #   - duplicate the polygon parts as many times as they overlap in the
            #     input layer
            union_multirow_path = tmp_dir / "union_multirow.gpkg"
            join_by_location(
                input1_path=non_intersecting_path,
                input2_path=input_path,
                output_path=union_multirow_path,
                spatial_relations_query="intersects is True and touches is False",
                input1_layer=output_layer,
                input2_layer=input_layer,
                output_layer=output_layer,
                input1_columns=["fid"],
                input2_columns=columns,
                input1_columns_prefix="union_",
                input2_columns_prefix="",
                explodecollections=explodecollections,
                gridsize=gridsize,
                where_post=where_post,
                nb_parallel=nb_parallel,
                batchsize=batchsize,
                force=False,
                output_with_spatial_index=False,
                operation_prefix=f"{operation_name}/",
                tmp_basedir=tmp_dir,
            )
            output_tmp_path = union_multirow_path

            if union_type in (
                "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS",
                "NO_INTERSECTIONS_ATTRIBUTE_LISTS",
            ):
                columns_local = (
                    list(input_layer.columns) if columns is None else columns
                )
                sql_stmt = _get_union_full_attr_sql_stmt(
                    union_multirow_path=output_tmp_path,
                    union_type=union_type,
                    columns=columns_local,
                )
                select_output_path = tmp_dir / "union_with_attributes.gpkg"
                select(
                    input_path=output_tmp_path,
                    output_path=select_output_path,
                    sql_stmt=sql_stmt,
                    input_layer=output_layer,
                    output_layer=output_layer,
                    preserve_fid=False,
                    nb_parallel=nb_parallel,
                    batchsize=batchsize,
                    operation_prefix=f"{operation_name}/",
                    batch_filter_column="union_fid",
                    tmp_basedir=tmp_dir,
                )
                output_tmp_path = select_output_path

        # Convert or add spatial index
        logger.info("Step 4 of 4: finalize")

        if output_tmp_path.suffix != output_path.suffix:
            # Output file should be in different format, so convert
            output_tmp2_path = tmp_dir / output_path.name
            gfo.copy_layer(src=output_tmp_path, dst=output_tmp2_path)
            output_tmp_path = output_tmp2_path
        elif GeofileInfo(output_tmp_path).default_spatial_index:
            gfo.create_spatial_index(
                path=output_tmp_path, layer=output_layer, exist_ok=True
            )

        # Now we are ready to move the result to the final spot...
        gfo.move(output_tmp_path, output_path)

    logger.info(f"Ready, full {operation_name} took {datetime.now() - start_time}")


def _get_union_full_attr_sql_stmt(
    union_multirow_path: Path, union_type: UnionFullSelfTypes | str, columns: list[str]
) -> str:
    """Create a sql statement to aggregate attributes based on a multi-row-union file.

    The file should include a "union_fid" column to group on.

    Args:
        union_multirow_path (Path): path to the union file with multiple rows per
            intersection
        union_type (str): type of union_full_self
        columns (List[str]): list of columns to aggregate

    Returns:
        str: sql statement
    """
    if union_type == "NO_INTERSECTIONS_ATTRIBUTE_LISTS":
        if len(columns) > 0:
            columns_list = []
            for col in columns:
                if col.lower() == "fid":
                    # The fid column will be aliased already in union_multirow_path!
                    alias = _ogr_sql_util.get_unique_fid_alias(col, columns)
                    column_str = f'json_group_array("{alias}") AS "{alias}"'
                else:
                    column_str = f'json_group_array("{col}") AS "{col}"'
                columns_list.append(column_str)

            columns_str = f", {', '.join(columns_list)}"

        else:
            columns_str = ""

        # An index on union_fid does not speed this up/decrease memory usage
        sql_stmt = f"""
            SELECT layer.{{geometrycolumn}}
                  ,COUNT(*) - 1 AS nb_intersections
                  {columns_str}
              FROM "{{input_layer}}" layer
             WHERE 1=1
               {{batch_filter}}
             GROUP BY union_fid
        """

    elif union_type == "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS":
        # An index on union_fid does not speed up/decrease memory usage for
        # following queries
        if len(columns) > 0:
            # First determine the maximum number of intersections
            sql_stmt_max = """
                SELECT MAX(counts.count) AS max_intersections
                  FROM  ( SELECT COUNT(*) AS count
                            FROM "{input_layer}" layer
                           GROUP BY union_fid
                        ) counts
            """
            df = fileops.read_file(union_multirow_path, sql_stmt=sql_stmt_max)
            max_intersections = df.iloc[0]["max_intersections"]

            # Now create the columns for in the select
            columns_list = []
            for is_id in range(max_intersections):
                for input_col in columns:
                    if input_col.lower() == "fid":
                        input_col = "fid_1"
                        col_alias = f"is{is_id}_fid"
                    else:
                        col_alias = f"is{is_id}_{input_col}"
                    columns_list.append(
                        f'MIN(CASE WHEN rn = {is_id + 1} THEN "{input_col}" END'
                        f') AS "{col_alias}"'
                    )

            columns_str = f", {', '.join(columns_list)}"

        else:
            columns_str = ""

        sql_stmt = f"""
            SELECT data.{{geometrycolumn}}
                  {columns_str}
              FROM (
                SELECT layer.{{geometrycolumn}}
                      ,layer.union_fid
                      {{columns_to_select_str}}
                      ,row_number() OVER (PARTITION BY union_fid) AS rn
                  FROM "{{input_layer}}" layer
                 WHERE 1=1
                   {{batch_filter}}
                ) data
             WHERE 1=1
             GROUP BY data.union_fid
        """

    else:
        raise ValueError(f"Unsupported union_type: {union_type}")

    return sql_stmt
