from concurrent import futures
import datetime
from geofileops.util import general_util
import logging
import logging.config
import multiprocessing
import os
from pathlib import Path
import shutil
from typing import List, Optional, Tuple, Union

from geofileops import geofile
from . import io_util
from . import ogr_util as ogr_util

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# Operations on one layer
################################################################################

def buffer(
        input_path: Path,
        output_path: Path,
        distance: float,
        quadrantsegments: int = 5,
        input_layer: str = None,
        output_layer: str = None,
        columns: List[str] = None,
        explodecollections: bool = False,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    # If buffer distance < 0, necessary to apply a make_valid to evade invalid geometries 
    if distance < 0:
        # A negative buffer is only relevant for polygon types, so only keep polygon results
        # Negative buffer creates invalid stuff, and the st_simplify(geom, 0) seems the only function fixing this!
        #geom_operation_sqlite = f"Collectionextract(ST_makevalid(ST_simplify(ST_buffer({{geometrycolumn}}, {distance}, {quadrantsegments}), 0)), 3) AS geom"
        
        sql_template = f'''
            SELECT Collectionextract(ST_buffer({{geometrycolumn}}, {distance}, {quadrantsegments}), 3) AS geom
                  {{columns_to_select_str}} 
              FROM "{{input_layer}}"
             WHERE 1=1 
               {{batch_filter}}'''
    else:
        sql_template = f'''
            SELECT ST_Buffer({{geometrycolumn}}, {distance}, {quadrantsegments}) AS geom
                  {{columns_to_select_str}} 
              FROM "{{input_layer}}"
             WHERE 1=1 
               {{batch_filter}}'''

    # Buffer operation always results in polygons...
    force_output_geometrytype = None
    if explodecollections is True:
        force_output_geometrytype = 'POLYGON'
    else:
        force_output_geometrytype = 'MULTIPOLYGON'
            
    return _single_layer_vector_operation(
            input_path=input_path,
            output_path=output_path,
            sql_template=sql_template,
            operation_name='buffer',
            input_layer=input_layer,
            output_layer=output_layer,
            columns=columns,
            explodecollections=explodecollections,
            force_output_geometrytype=force_output_geometrytype,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def isvalid(
        input_path: Path,
        output_path: Path,
        input_layer: str = None,        
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False) -> bool:

    # Prepare sql template for this operation 
    sql_template = f'''
            SELECT ST_IsValidDetail({{geometrycolumn}}) AS geom
                  ,ST_IsValid({{geometrycolumn}}) AS isvalid
                  ,ST_IsValidReason({{geometrycolumn}}) AS isvalidreason
                  {{columns_to_select_str}} 
              FROM "{{input_layer}}"
             WHERE 1=1 
                {{batch_filter}}'''

    _single_layer_vector_operation(
            input_path=input_path,
            output_path=output_path,
            sql_template=sql_template,
            operation_name='isvalid',
            input_layer=input_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)
    
    # If no invalid geoms are found, there won't be an output file and we can return True
    if not output_path.exists():
        return True
    else:
        layerinfo = geofile.getlayerinfo(output_path)
        logger.info(f"Found {layerinfo.featurecount} invalid geometries in {output_path}")
        return False

def convexhull(
        input_path: Path,
        output_path: Path,
        input_layer: str = None,
        output_layer: str = None,
        columns: Optional[List[str]] = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    # Prepare sql template for this operation 
    sql_template = f'''
            SELECT ST_ConvexHull({{geometrycolumn}}) AS geom
                  {{columns_to_select_str}} 
              FROM "{{input_layer}}"
             WHERE 1=1 
               {{batch_filter}}'''

    return _single_layer_vector_operation(
            input_path=input_path,
            output_path=output_path,
            sql_template=sql_template,
            operation_name='convexhull',
            input_layer=input_layer,
            output_layer=output_layer,
            columns=columns,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def makevalid(
        input_path: Path,
        output_path: Path,
        input_layer: str = None,        
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    # Prepare sql template for this operation 
    sql_template = f'''
            SELECT ST_MakeValid({{geometrycolumn}}) AS geom
                  {{columns_to_select_str}} 
              FROM "{{input_layer}}"
             WHERE 1=1 
               {{batch_filter}}'''

    return _single_layer_vector_operation(
            input_path=input_path,
            output_path=output_path,
            sql_template=sql_template,
            operation_name='make_valid',
            input_layer=input_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def select(
        input_path: Path,
        output_path: Path,
        sql_stmt: str,
        sql_dialect: str = None,
        input_layer: str = None,        
        output_layer: str = None,
        explodecollections: bool = False,
        force_output_geometrytype: str = None,
        nb_parallel: int = 1,
        verbose: bool = False,
        force: bool = False):

    # If parallel processing asked...
    if nb_parallel == -1 or nb_parallel > 1:
        return _single_layer_vector_operation(
                input_path=input_path,
                output_path=output_path,
                sql_template=sql_stmt,
                operation_name='select',
                input_layer=input_layer,
                output_layer=output_layer,
                explodecollections=explodecollections,
                force_output_geometrytype=force_output_geometrytype,
                nb_parallel=nb_parallel,
                verbose=verbose,
                force=force)

    ##### Init #####
    start_time = datetime.datetime.now()
    if output_path.exists():
        if force is False:
            logger.info(f"Stop select: output exists already {output_path}")
            return
        else:
            geofile.remove(output_path)

    if input_layer is None:
        input_layer = geofile.get_only_layer(input_path)
    if output_layer is None:
        output_layer = geofile.get_default_layer(output_path)
    if force_output_geometrytype is None:
        input_layer_info = geofile.getlayerinfo(input_path, input_layer)
        force_output_geometrytype = input_layer_info.geometrytypename

    ##### Exec #####
    translate_description = f"Select on {input_path}"
    ogr_util.vector_translate(
            input_path=input_path,
            output_path=output_path,
            translate_description=translate_description,
            output_layer=output_layer,
            sql_stmt=sql_stmt,
            sql_dialect=sql_dialect,
            force_output_geometrytype=force_output_geometrytype,
            verbose=verbose)

    logger.info(f"Processing ready, took {datetime.datetime.now()-start_time}!")

def simplify(
        input_path: Path,
        output_path: Path,
        tolerance: float,        
        input_layer: str = None,        
        output_layer: str = None,
        columns: Optional[List[str]] = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    # Prepare sql template for this operation 
    sql_template = f'''
            SELECT ST_Simplify({{geometrycolumn}}, {tolerance}) AS geom
                  {{columns_to_select_str}} 
              FROM "{{input_layer}}"
             WHERE 1=1 
               {{batch_filter}}'''

    return _single_layer_vector_operation(
            input_path=input_path,
            output_path=output_path,
            sql_template=sql_template,
            operation_name='simplify',
            input_layer=input_layer,
            output_layer=output_layer,
            columns=columns,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def _single_layer_vector_operation(
        input_path: Path,
        output_path: Path,
        sql_template: str,
        operation_name: str,
        input_layer: str = None,        
        output_layer: str = None,
        columns: List[str] = None,
        explodecollections: bool = False,
        force_output_geometrytype: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    ##### Init #####
    start_time = datetime.datetime.now()
    if output_path.exists():
        if force is False:
            logger.info(f"Stop {operation_name}: output exists already {output_path}")
            return
        else:
            geofile.remove(output_path)
    
    # Check if spatialite is properly installed to execute this query
    if input_path.suffix.lower() == '.gpkg':
        ogr_util.check_gdal_spatialite_install(sql_template)

    # Get layer info
    if input_layer is None:
        input_layer = geofile.get_only_layer(input_path)
    if output_layer is None:
        output_layer = geofile.get_default_layer(output_path)

    ##### Calculate #####
    tempdir = io_util.create_tempdir(operation_name.replace(' ', '_'))
    
    try:
        input_tmp_path = input_path

        ##### Calculate #####
        # Calculating can be done in parallel, but only one process can write to 
        # the same file at the time... 
        if(nb_parallel == -1):
            nb_parallel = multiprocessing.cpu_count()
            if nb_parallel > 4:
                nb_parallel -= 1

        nb_batches = nb_parallel*4
        nb_done = 0
        with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

            # Prepare columns to select
            layerinfo = geofile.getlayerinfo(input_path, input_layer)  
            columns_to_select_str = ''
            if columns is not None:
                columns_quoted = [f'"{col}"' for col in columns] 
                columns_to_select_str = f", {', '.join(columns_quoted)}"
            elif len(layerinfo.columns) > 0:
                columns_quoted = [f'"{col}"' for col in layerinfo.columns]
                columns_to_select_str = f", {', '.join(columns_quoted)}"
            
            # Calculate the number of features per thread
            nb_rows_input_layer = layerinfo.featurecount
            row_limit = int(nb_rows_input_layer/nb_batches)
            row_offset = 0
            # Prepare output filename
            tmp_output_path = tempdir / output_path.name

            translate_jobs = {}    
            future_to_batch_id = {}
            for batch_id in range(nb_batches):

                translate_jobs[batch_id] = {}
                translate_jobs[batch_id]['layer'] = output_layer

                output_tmp_partial_path = tempdir / f"{output_path.stem}_{batch_id}{output_path.suffix}"
                translate_jobs[batch_id]['tmp_partial_output_path'] = output_tmp_partial_path

                # For the last batch_id, take all rowid's left...
                if batch_id < nb_batches-1:
                    batch_filter = f"AND (rowid >= {row_offset} AND rowid < {row_offset + row_limit})"
                else:
                    batch_filter = f"AND rowid >= {row_offset}"

                # Now we have everything to format sql statement
                sql_stmt = sql_template.format(
                        geometrycolumn=layerinfo.geometrycolumn,
                        columns_to_select_str=columns_to_select_str,
                        input_layer=input_layer,
                        batch_filter=batch_filter)

                # Make sure no NULL geoms are outputted...
                sql_stmt = f'''
                        SELECT sub.*
                          FROM
                            ( {sql_stmt}
                            ) sub
                         WHERE sub.geom IS NOT NULL'''

                translate_jobs[batch_id]['sql_stmt'] = sql_stmt
                translate_description = f"Async {operation_name} {batch_id} of {nb_batches}"
                # Remark: this temp file doesn't need spatial index
                translate_info = ogr_util.VectorTranslateInfo(
                        input_path=input_tmp_path,
                        output_path=output_tmp_partial_path,
                        translate_description=translate_description,
                        output_layer=output_layer,
                        sql_stmt=sql_stmt,
                        sql_dialect='SQLITE',
                        create_spatial_index=False,
                        explodecollections=explodecollections,
                        force_output_geometrytype=force_output_geometrytype,
                        verbose=verbose)
                future = ogr_util.vector_translate_async(
                        concurrent_pool=calculate_pool, info=translate_info)
                future_to_batch_id[future] = batch_id
                row_offset += row_limit
            
            # Loop till all parallel processes are ready, but process each one that is ready already
            for future in futures.as_completed(future_to_batch_id):
                try:
                    _ = future.result()
                except Exception as ex:
                    batch_id = future_to_batch_id[future]
                    raise Exception(f"Error executing {translate_jobs[batch_id]}") from ex

                # Start copy of the result to a common file
                # Remark: give higher priority, because this is the slowest factor
                batch_id = future_to_batch_id[future]
                tmp_partial_output_path = translate_jobs[batch_id]['tmp_partial_output_path']
                
                if tmp_partial_output_path.exists():
                    geofile.append_to(
                            src=tmp_partial_output_path, 
                            dst=tmp_output_path, 
                            create_spatial_index=False)
                    geofile.remove(tmp_partial_output_path)
                else:
                    if verbose:
                        logger.info(f"Result file {tmp_partial_output_path} was empty")

                # Log the progress and prediction speed
                nb_done += 1
                general_util.report_progress(start_time, nb_done, nb_batches, operation_name)

        ##### Round up and clean up ##### 
        # Now create spatial index and move to output location
        if tmp_output_path.exists():
            geofile.create_spatial_index(path=tmp_output_path, layer=output_layer)
            geofile.move(tmp_output_path, output_path)
    finally:
        # Clean tmp dir
        shutil.rmtree(tempdir)
        logger.info(f"Processing ready, took {datetime.datetime.now()-start_time}!")

################################################################################
# Operations on two layers
################################################################################

def intersect(
        input1_path: Path,
        input2_path: Path,
        output_path: Path,
        input1_layer: str = None,
        input1_columns: List[str] = None,
        input2_layer: str = None,
        input2_columns: List[str] = None,
        output_layer: str = None,
        explodecollections: bool = False,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    # In the query, important to only extract the geometry types that are expected 
    input1_layer_info = geofile.getlayerinfo(input1_path, input1_layer)
    input2_layer_info = geofile.getlayerinfo(input2_path, input2_layer)
    collection_extract_typeid = min(geofile.to_generaltypeid(input1_layer_info.geometrytypename), 
                                    geofile.to_generaltypeid(input2_layer_info.geometrytypename))

    # Prepare sql template for this operation 
    sql_template = f'''
        SELECT sub.geom
             {{layer1_columns_in_select_str}}
             {{layer2_columns_in_select_str}} 
          FROM
            ( SELECT ST_Multi(Collectionextract(ST_Intersection(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}), {collection_extract_typeid})) as geom
                    {{layer1_columns_in_subselect_str}}
                    {{layer2_columns_in_subselect_str}}
                FROM "{{input1_tmp_layer}}" layer1
                JOIN "rtree_{{input1_tmp_layer}}_{{input1_geometrycolumn}}" layer1tree ON layer1.fid = layer1tree.id
                JOIN "{{input2_tmp_layer}}" layer2
                JOIN "rtree_{{input2_tmp_layer}}_{{input2_geometrycolumn}}" layer2tree ON layer2.fid = layer2tree.id
               WHERE 1=1
                 {{batch_filter}}
                 AND layer1tree.minx <= layer2tree.maxx AND layer1tree.maxx >= layer2tree.minx
                 AND layer1tree.miny <= layer2tree.maxy AND layer1tree.maxy >= layer2tree.miny
                 AND ST_Intersects(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}) = 1
                 AND ST_Touches(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}) = 0
            ) sub
         WHERE sub.geom IS NOT NULL
        '''
    operation_name = "intersect"

    # Go!
    return _two_layer_vector_operation(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            sql_template=sql_template,
            operation_name=operation_name,
            input1_layer=input1_layer,
            input1_columns=input1_columns,
            input2_layer=input2_layer,
            input2_columns=input2_columns,
            output_layer=output_layer,
            explodecollections=explodecollections,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def erase(
        input_path: Path,
        erase_path: Path,
        output_path: Path,
        input_layer: str = None,
        input_columns: List[str] = None,
        erase_layer: str = None,
        output_layer: str = None,
        explodecollections: bool = False,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    # Init
    # In the query, important to only extract the geometry types that are expected 
    input_layer_info = geofile.getlayerinfo(input_path, input_layer)
    collection_extract_typeid = geofile.to_generaltypeid(input_layer_info.geometrytypename)

    # To be safe, if explodecollections is False, force the MULTI version of 
    # the input layer as output type, because erase can cause eg. polygons to 
    # be split to multipolygons...
    if explodecollections is True: # and input_layer_info.geometrytypename in ['POLYGON']:
        force_output_geometrytype = input_layer_info.geometrytypename
    else:
        force_output_geometrytype = geofile.to_multi_type(input_layer_info.geometrytypename)

    # Prepare sql template for this operation 
    # Remarks:
    #   - ST_difference(geometry , NULL) gives NULL as result! -> hence the CASE 
    #   - use of the with instead of an inline view is a lot faster
    #   - WHERE geom IS NOT NULL to evade rows with a NULL geom, they give issues in later operations
    sql_template = f'''
          SELECT * FROM (
            WITH layer2_unioned AS (
              SELECT layer1.rowid AS layer1_rowid
                    ,ST_union(layer2.{{input2_geometrycolumn}}) AS geom
                FROM "{{input1_tmp_layer}}" layer1
                JOIN "rtree_{{input1_tmp_layer}}_{{input1_geometrycolumn}}" layer1tree ON layer1.fid = layer1tree.id
                JOIN "{{input2_tmp_layer}}" layer2
                JOIN "rtree_{{input2_tmp_layer}}_{{input2_geometrycolumn}}" layer2tree ON layer2.fid = layer2tree.id
               WHERE layer1tree.minx <= layer2tree.maxx AND layer1tree.maxx >= layer2tree.minx
                 AND layer1tree.miny <= layer2tree.maxy AND layer1tree.maxy >= layer2tree.miny
                 AND ST_Intersects(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}) = 1
                 AND ST_Touches(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}) = 0
               GROUP BY layer1.rowid
            )
            SELECT CASE WHEN layer2_unioned.geom IS NULL THEN layer1.{{input1_geometrycolumn}}
                        ELSE CollectionExtract(ST_difference(layer1.{{input1_geometrycolumn}}, layer2_unioned.geom), {collection_extract_typeid})
                   END as geom
                  {{layer1_columns_in_subselect_str}}
              FROM "{{input1_tmp_layer}}" layer1
              LEFT JOIN layer2_unioned ON layer1.rowid = layer2_unioned.layer1_rowid
             WHERE 1=1
               {{batch_filter}}
          )
          WHERE geom IS NOT NULL
            '''
    
    # Go!
    return _two_layer_vector_operation(
            input1_path=input_path,
            input2_path=erase_path,
            output_path=output_path,
            sql_template=sql_template,
            operation_name='erase',
            input1_layer=input_layer,
            input1_columns=input_columns,
            input2_layer=erase_layer,
            input2_columns=None,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            explodecollections=explodecollections,
            force_output_geometrytype=force_output_geometrytype,
            verbose=verbose,
            force=force)

def export_by_location(
        input_to_select_from_path: Path,
        input_to_compare_with_path: Path,
        output_path: Path,
        min_area_intersect: Optional[float] = None,
        area_inters_column_name: Optional[str] = 'area_inters',
        input1_layer: str = None,
        input1_columns: List[str] = None,
        input2_layer: str = None,
        input2_columns: List[str] = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    
    # Prepare sql template for this operation 
    # TODO: test performance difference between the following two queries
    sql_template = f'''
            SELECT layer1.{{input1_geometrycolumn}} AS geom 
                  {{layer1_columns_in_subselect_str}}
              FROM "{{input1_tmp_layer}}" layer1
              JOIN "rtree_{{input1_tmp_layer}}_{{input1_geometrycolumn}}" layer1tree ON layer1.fid = layer1tree.id
             WHERE 1=1
               {{batch_filter}}
               AND EXISTS (
                  SELECT 1 
                    FROM "{{input2_tmp_layer}}" layer2
                    JOIN "rtree_{{input2_tmp_layer}}_{{input2_geometrycolumn}}" layer2tree ON layer2.fid = layer2tree.id
                   WHERE layer1tree.minx <= layer2tree.maxx AND layer1tree.maxx >= layer2tree.minx
                     AND layer1tree.miny <= layer2tree.maxy AND layer1tree.maxy >= layer2tree.miny
                     AND ST_intersects(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}) = 1
                     AND ST_touches(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}) = 0)
            '''
    # Calculate intersect area if necessary
    area_inters_column_expression = None
    if area_inters_column_name is not None or min_area_intersect is not None:
        if area_inters_column_name is None:
            area_inters_column_name = 'area_inters'
        area_inters_column_expression = f",ST_area(ST_intersection(ST_union(layer1.{{input1_geometrycolumn}}), ST_union(layer2.{{input2_geometrycolumn}}))) as {area_inters_column_name}"
    
    # Prepare sql template for this operation 
    sql_template = f'''
            SELECT ST_union(layer1.{{input1_geometrycolumn}}) as geom
                  {{layer1_columns_in_groupby_str}}
                  {area_inters_column_expression}
              FROM "{{input1_tmp_layer}}" layer1
              JOIN "rtree_{{input1_tmp_layer}}_{{input1_geometrycolumn}}" layer1tree ON layer1.fid = layer1tree.id
              JOIN "{{input2_tmp_layer}}" layer2
              JOIN "rtree_{{input2_tmp_layer}}_{{input2_geometrycolumn}}" layer2tree ON layer2.fid = layer2tree.id
             WHERE 1=1
               {{batch_filter}}
               AND layer1tree.minx <= layer2tree.maxx AND layer1tree.maxx >= layer2tree.minx
               AND layer1tree.miny <= layer2tree.maxy AND layer1tree.maxy >= layer2tree.miny
               AND ST_Intersects(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}) = 1
               AND ST_Touches(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}) = 0
             GROUP BY layer1.rowid {{layer1_columns_in_groupby_str}}
            '''

    # Filter on intersect area if necessary
    if min_area_intersect is not None:
        sql_template = f'''
                SELECT sub.* 
                  FROM 
                    ( {sql_template}
                    ) sub
                 WHERE sub.{area_inters_column_name} >= {min_area_intersect}'''

    # Go!
    input_layer_info = geofile.getlayerinfo(input_to_select_from_path, input1_layer)
    return _two_layer_vector_operation(
            input1_path=input_to_select_from_path,
            input2_path=input_to_compare_with_path,
            output_path=output_path,
            sql_template=sql_template,
            operation_name='export_by_location',
            input1_layer=input1_layer,
            input1_columns=input1_columns,
            input2_layer=input2_layer,
            input2_columns=input2_columns,
            output_layer=output_layer,
            force_output_geometrytype=input_layer_info.geometrytypename,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def export_by_distance(
        input_to_select_from_path: Path,
        input_to_compare_with_path: Path,
        output_path: Path,
        max_distance: float,
        input1_layer: str = None,
        input2_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    # Prepare sql template for this operation 
    sql_template = f'''
            SELECT geom
                  {{layer1_columns_in_subselect_str}}
                FROM "{{input1_tmp_layer}}" layer1
                JOIN "rtree_{{input1_tmp_layer}}_{{input1_geometrycolumn}}" layer1tree ON layer1.fid = layer1tree.id
                WHERE 1=1
                  {{batch_filter}}
                  AND EXISTS (
                      SELECT 1 
                        FROM "{{input2_tmp_layer}}" layer2
                        JOIN "rtree_{{input2_tmp_layer}}_{{input2_geometrycolumn}}" layer2tree ON layer2.fid = layer2tree.id
                        WHERE (layer1tree.minx-{max_distance}) <= layer2tree.maxx 
                          AND (layer1tree.maxx+{max_distance}) >= layer2tree.minx
                          AND (layer1tree.miny-{max_distance}) <= layer2tree.maxy 
                          AND (layer1tree.maxy+{max_distance}) >= layer2tree.miny
                          AND ST_distance(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}) <= {max_distance})'''

    input_layer_info = geofile.getlayerinfo(input_to_select_from_path, input1_layer)

    # Go!
    return _two_layer_vector_operation(
            input1_path=input_to_select_from_path,
            input2_path=input_to_compare_with_path,
            output_path=output_path,
            sql_template=sql_template,
            operation_name='export_by_distance',
            input1_layer=input1_layer,
            input2_layer=input2_layer,
            output_layer=output_layer,
            force_output_geometrytype=input_layer_info.geometrytypename,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def _two_layer_vector_operation(
        input1_path: Path,
        input2_path: Path,
        output_path: Path,
        sql_template: str,
        operation_name: str,
        input1_layer: str = None,
        input1_columns: List[str] = None,
        input2_layer: str = None,
        input2_columns: List[str] = None,
        output_layer: str = None,
        explodecollections: bool = False,
        force_output_geometrytype: str = 'MULTIPOLYGON',
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    # TODO: think about whether MULTIPOLYGON is a good default value for force_output_geometrytype!
    """
    ...
    
    Args:
        input1_path (str): the file to export features from
        input2_path (str): the file to check intersections with
        output_path (str): output file
        input1_layer (str, optional): [description]. Defaults to None.
        input2_layer (str, optional): [description]. Defaults to None.
        output_layer (str, optional): [description]. Defaults to None.
        explodecollections (bool, optional): Explode collecions in output. Defaults to False.
        force_output_geometrytype (str, optional): Defaults to 'MULTIPOLYGON'.
        nb_parallel (int, optional): [description]. Defaults to -1.
        force (bool, optional): [description]. Defaults to False.
    
    Raises:
        Exception: [description]
    """
    # TODO: test the following syntax, maybe copying to the same gpkg becomes
    # unnecessary that way:
    #  ogr2ogr diff.shp file1.shp -dialect sqlite \
    #      -sql "SELECT ST_Difference(a.Geometry, b.Geometry) AS Geometry, a.id \
    #            FROM file1 a LEFT JOIN 'file2.shp'.file2 b USING (id) WHERE a.Geometry != b.Geometry"

    ##### Init #####
    if output_path.exists():
        if force is False:
            logger.info(f"Stop {operation_name}: output exists already {output_path}")
            return
        else:
            geofile.remove(output_path)

    # Check if spatialite is properly installed to execute this query
    ogr_util.check_gdal_spatialite_install(sql_template)

    # Init layer info
    start_time = datetime.datetime.now()
    if input1_layer is None:
        input1_layer = geofile.get_only_layer(input1_path)
    if input2_layer is None:
        input2_layer = geofile.get_only_layer(input2_path)
    if output_layer is None:
        output_layer = geofile.get_default_layer(output_path)

    # Prepare tmp layer/file names
    tempdir = io_util.create_tempdir(operation_name)
    if(input1_layer != input2_layer):
        input1_tmp_layer = input1_layer
        input2_tmp_layer = input2_layer
    else:
        input1_tmp_layer = 'l1_' + input1_layer
        input2_tmp_layer = 'l2_' + input2_layer
    input_tmp_path = tempdir / "input_layers.gpkg" 

    ##### Prepare tmp files #####
    logger.info(f"Prepare temp input files for {operation_name} in {tempdir}")

    try:
        # Get input2 data to temp gpkg file. 
        # If it is the only layer in the input file, just copy file
        if(input2_path.suffix.lower() == '.gpkg' 
           and len(geofile.listlayers(input2_path)) == 1):
            logger.debug(f"Copy {input2_path} to {input_tmp_path}")
            geofile.copy(input2_path, input_tmp_path)

            # If needed, rename layer
            if input2_layer != input2_tmp_layer:
                geofile.rename_layer(input_tmp_path, input2_layer, input2_tmp_layer)           
        else:
            # Copy the layer needed to a new gpkg
            ogr_util.vector_translate(
                    input_path=input2_path,
                    output_path=input_tmp_path,
                    input_layers=input2_layer,
                    output_layer=input2_tmp_layer,
                    verbose=verbose)
        
        # Spread input1 data over different layers to be able to calculate in parallel
        if nb_parallel == -1:
            nb_parallel = multiprocessing.cpu_count()
            if nb_parallel > 4:
                nb_parallel -= 1

        nb_batches = nb_parallel * 4
        batches = _split_layer_features(
                input_path=input1_path,
                input_layer=input1_layer,
                output_path=input_tmp_path,
                output_baselayer=input1_tmp_layer,
                nb_batches=nb_batches,
                verbose=verbose)
        if len(batches) < nb_parallel:
            nb_parallel = len(batches)
        
        ##### Calculate! #####
        # We need the input1 column names to format the select
        input1_tmp_layerinfo = geofile.getlayerinfo(input_tmp_path, batches[0]['layer'])
        if input1_columns is not None:
            layer1_columns = input1_columns
        else:
            layer1_columns = input1_tmp_layerinfo.columns
        layer1_columns_in_subselect_str = ''
        layer1_columns_in_select_str = ''
        layer1_columns_in_groupby_str = ''
        if len(layer1_columns) > 0:
            layer1_columns_in_subselect = [f'layer1."{column}" "l1_{column}"' for column in layer1_columns]
            layer1_columns_in_subselect_str = ',' + ", ".join(layer1_columns_in_subselect)
            layer1_columns_in_select = [f'sub."l1_{column}"' for column in layer1_columns]
            layer1_columns_in_select_str = ',' + ", ".join(layer1_columns_in_select)
            layer1_columns_in_groupby = [f'layer1."{column}"' for column in layer1_columns]
            layer1_columns_in_groupby_str = ',' + ", ".join(layer1_columns_in_groupby)

        # We need the input2 column names to format the select
        input2_tmp_layerinfo = geofile.getlayerinfo(input_tmp_path, input2_tmp_layer)
        if input2_columns is not None:
            layer2_columns = input2_columns
        else:
            layer2_columns = input2_tmp_layerinfo.columns
        layer2_columns_in_subselect_str = ''
        layer2_columns_in_select_str = ''
        layer2_columns_in_groupby_str = ''
        if len(layer2_columns) > 0:
            layer2_columns_in_subselect = [f'layer2."{column}" "l2_{column}"' for column in layer2_columns]
            layer2_columns_in_subselect_str = ',' + ", ".join(layer2_columns_in_subselect)
            layer2_columns_in_select = [f'sub."l2_{column}"' for column in layer2_columns]
            layer2_columns_in_select_str = ',' + ", ".join(layer2_columns_in_select)
            layer2_columns_in_groupby = [f'layer2."{column}"' for column in layer2_columns]
            layer2_columns_in_groupby_str = ',' + ", ".join(layer2_columns_in_groupby)        

        # Fill out the geometry column name in geom_operation_sqlite
        # TODO: check if geom column is always geom
        #geom_operation_sqlite = geom_operation_sqlite.format(
        #        geometrycolumn=layerinfo['geometry_column'])
        # Calculate the number of features per thread
        #nb_rows_input_layer = layerinfo['featurecount']
        #row_limit = int(nb_rows_input_layer/nb_batches)
        #row_offset = 0

        # Prepare output filename
        tmp_output_path = tempdir / output_path.name
        
        ##### Calculate #####
        logger.info(f"Start {operation_name} in {nb_parallel} parallel processes")

        # Calculating can be done in parallel, but only one process can write to 
        # the same file at the time... 
        with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

            # Start looping
            translate_jobs = {}    
            future_to_batch_id = {}
            for batch_id in batches:

                translate_jobs[batch_id] = {}
                translate_jobs[batch_id]['layer'] = output_layer

                tmp_output_partial_path = tempdir / f"{output_path.stem}_{batch_id}{output_path.suffix}"
                translate_jobs[batch_id]['tmp_partial_output_path'] = tmp_output_partial_path

                input1_tmp_curr_layer = batches[batch_id]['layer']
                sql_stmt = sql_template.format(
                        layer1_columns_in_select_str=layer1_columns_in_select_str,
                        layer1_columns_in_subselect_str=layer1_columns_in_subselect_str,
                        input1_tmp_layer=input1_tmp_curr_layer,
                        input1_geometrycolumn=input1_tmp_layerinfo.geometrycolumn,
                        layer2_columns_in_select_str=layer2_columns_in_select_str,
                        layer2_columns_in_subselect_str=layer2_columns_in_subselect_str,
                        input2_tmp_layer=input2_tmp_layer,
                        input2_geometrycolumn=input2_tmp_layerinfo.geometrycolumn,
                        layer1_columns_in_groupby_str=layer1_columns_in_groupby_str,
                        batch_filter=batches[batch_id]['batch_filter'])

                translate_jobs[batch_id]['sqlite_stmt'] = sql_stmt
                translate_description = f"Calculate export_by_location between {input_tmp_path} and {tmp_output_partial_path}"
                # Remark: this temp file doesn't need spatial index nor journalling mode
                translate_info = ogr_util.VectorTranslateInfo(
                        input_path=input_tmp_path,
                        output_path=tmp_output_partial_path,
                        translate_description=translate_description,
                        output_layer=output_layer,
                        sql_stmt=sql_stmt,
                        sql_dialect='SQLITE',
                        create_spatial_index=False,
                        explodecollections=explodecollections,
                        sqlite_journal_mode='OFF',
                        force_output_geometrytype=force_output_geometrytype,
                        verbose=verbose)
                future = ogr_util.vector_translate_async(
                        concurrent_pool=calculate_pool, info=translate_info)
                future_to_batch_id[future] = batch_id
            
            # Loop till all parallel processes are ready, but process each one that is ready already
            nb_done = 0

            general_util.report_progress(start_time, nb_done, nb_batches, operation_name)
            for future in futures.as_completed(future_to_batch_id):
                try:
                    _ = future.result()

                    # Start copy of the result to a common file
                    # Remark: give higher priority, because this is the slowest factor
                    batch_id = future_to_batch_id[future]
                    tmp_partial_output_path = translate_jobs[batch_id]['tmp_partial_output_path']

                    # If there wasn't an exception, but the output file doesn't exist, the result was empty, so just skip.
                    if not tmp_partial_output_path.exists():
                        if verbose:
                            logger.info(f"Temporary partial file was empty: {translate_jobs[batch_id]}")
                        continue
                    
                    translate_description = f"Copy result {batch_id} of {nb_batches} to {output_layer}"
                    translate_info = ogr_util.VectorTranslateInfo(
                            input_path=tmp_partial_output_path,
                            output_path=tmp_output_path,
                            translate_description=translate_description,
                            output_layer=output_layer,
                            append=True,
                            update=True,
                            create_spatial_index=False,
                            explodecollections=explodecollections,
                            force_output_geometrytype=force_output_geometrytype,
                            priority_class='NORMAL',
                            force_py=True,
                            verbose=verbose)
                    ogr_util.vector_translate_by_info(info=translate_info)
                    future_to_batch_id[future] = batch_id
                    tmp_partial_output_path = translate_jobs[batch_id]['tmp_partial_output_path']
                    geofile.remove(tmp_partial_output_path)
                except Exception as ex:
                    batch_id = future_to_batch_id[future]
                    raise Exception(f"Error executing {translate_jobs[batch_id]}") from ex

                # Log the progress and prediction speed
                nb_done += 1
                general_util.report_progress(start_time, nb_done, nb_batches, operation_name)

        ##### Round up and clean up ##### 
        # Now create spatial index and move to output location
        geofile.create_spatial_index(path=tmp_output_path, layer=output_layer)
        geofile.move(tmp_output_path, output_path)
        shutil.rmtree(tempdir)
        logger.info(f"{operation_name} ready, took {datetime.datetime.now()-start_time}!")
    except Exception as ex:
        logger.exception(f"{operation_name} ready with ERROR, took {datetime.datetime.now()-start_time}!")

def _split_layer_features(
        input_path: Path,
        input_layer: str,
        output_path: Path,
        output_baselayer: str,
        nb_batches: int,
        verbose: bool = False) -> dict:

    ##### Init #####
    split_random = True
    split_to_seperate_layers = True
    seperate_working_file_created = False
    try:
        # If we want to split to seperate layers using random allocation, 
        # we need to create a temporary copy of the file to work on
        if split_to_seperate_layers:

            # If the batches should be split by random, we need a temp file to add
            # the batch_id column to...
            if split_random is True:

                temp_path = output_path.parent / f"{input_path.stem}.gpkg"
                seperate_working_file_created = True

                if input_path.suffix.lower() == '.gpkg':
                    geofile.copy(input_path, temp_path)
                else:
                    ogr_util.vector_translate(
                            input_path=input_path, 
                            output_path=temp_path,
                            output_layer=input_layer,
                            transaction_size=200000,
                            force_py=True,
                            verbose=verbose)
            else:
                # Else we can just use the input file to copy the data from
                temp_path = input_path
        else:
            # Else we can just add the layer tot the output path
            temp_path = output_path
            geofile.append_to(
                    src=input_path, 
                    dst=temp_path,
                    dst_layer=input_layer,
                    verbose=verbose)

        # TODO: test if it is still needed to copy data to seperate layers if index is present on batch_id
        # TODO: maybe if a sequential number is added as column this can be used to create the batches,
        #       because that way the number of items in a batch will be more correct...
        # TODO: if the rowid is already +- sequential, use rowid filtering instead of a seperate column?
        #       remark: for dissolve, it will be necessary (especially if multiple passes were used) that
        #               the output file doen't group large objects as it does now...

        # Get column names and info
        layerinfo = geofile.getlayerinfo(temp_path, input_layer)
        columns_to_select_str = ''
        if len(layerinfo.columns) > 0:
            columns_to_select = [f'"{column}"' for column in layerinfo.columns]
            columns_to_select_str = "," + ", ".join(columns_to_select)
        if layerinfo.geometrycolumn == 'geom':
            geometry_column_for_select = 'geom'
        else:
            geometry_column_for_select = f"{layerinfo.geometrycolumn} geom"

        # Check number of batches + appoint rows to batches
        nb_rows_input_layer = layerinfo.featurecount
        if nb_batches > int(nb_rows_input_layer/10):
            nb_batches = int(nb_rows_input_layer/10)
        nb_rows_per_batch = int(nb_rows_input_layer / nb_batches)

        ##### Split to x batches/layers #####
        # If needed, randomly determine the batch to be used for calculation in parallel 
        # + add index for (big)increase of performance 
        if split_random is True:
            geofile.add_column(path=temp_path, layer=input_layer, 
                    name='batch_id', type='INTEGER', expression=f"ABS(RANDOM() % {nb_batches})")
            sqlite_stmt = f'CREATE INDEX idx_batch_id ON "{input_layer}"(batch_id)' 
            ogr_util.vector_info(path=temp_path, sql_stmt=sqlite_stmt, sql_dialect='SQLITE', readonly=False)

        # Remark: adding data to a file in parallel using ogr2ogr gives locking 
        # issues on the sqlite file, so needs to be done sequential!
        logger.debug(f"Split the input file to {nb_batches} batches")
        batches = {}
        offset = 0
        for batch_id in range(nb_batches):
            
            # If each batch should get its seperate layer... 
            if split_to_seperate_layers is True:
                output_baselayer_stripped = output_baselayer.strip("'\"")
                output_layer_curr = f"{output_baselayer_stripped}_{batch_id}"
                
                # If random split, use the batch_id
                if split_random is True:
                    sql_stmt = f'''
                            SELECT {geometry_column_for_select}{columns_to_select_str}  
                              FROM "{input_layer}" layer1
                             WHERE layer1.batch_id = {batch_id}'''
                else:
                    # If not random, use rowid filtering
                    if batch_id < nb_batches:
                        sql_stmt = f'''
                            SELECT {geometry_column_for_select}{columns_to_select_str}  
                              FROM "{input_layer}" layer1
                             WHERE layer1.rowid >= {offset} 
                               AND layer1.rowid < {offset+nb_rows_per_batch}'''
                        
                    else:
                        sql_stmt = f'''
                            SELECT {geometry_column_for_select}{columns_to_select_str}  
                              FROM "{input_layer}" layer1
                             WHERE layer1.rowid >= {offset}'''

                translate_description=f"Copy data from {input_path}.{input_layer} to {output_path}.{output_layer_curr}"
                ogr_util.vector_translate(
                        input_path=temp_path,
                        output_path=output_path,
                        translate_description=translate_description,
                        output_layer=output_layer_curr,
                        sql_stmt=sql_stmt,
                        sql_dialect='SQLITE',
                        transaction_size=200000,
                        append=True,
                        force_py=True,
                        verbose=verbose)

                # If items were actually added to the layer, add it to the list of jobs
                if output_layer_curr in geofile.listlayers(output_path):
                    batches[batch_id] = {}
                    batches[batch_id]['layer'] = output_layer_curr
                    batches[batch_id]['batch_filter'] = ''
                else:
                    logger.debug(f"Layer {output_layer_curr} is empty in geofile {output_path}")
            else:
                # If they are all still in the same layer
                batches[batch_id] = {}
                batches[batch_id]['layer'] = input_layer
                if split_random is True:
                    batches[batch_id]['batch_filter'] = f"AND batch_id = {batch_id}"
                else:
                    # If not random, use rowid filtering
                    if batch_id < nb_batches:
                        batches[batch_id]['batch_filter'] = f"AND (layer1.rowid >= {offset} AND layer1.rowid < {offset+nb_rows_per_batch})"
                        offset += nb_rows_per_batch
                    else:
                        batches[batch_id]['batch_filter'] = f"AND layer1.rowid >= {offset}"
    
    finally:
        # Cleanup
        if seperate_working_file_created is True and temp_path.exists():
            geofile.remove(temp_path)
    
    return batches

def dissolve(
        input_path: Path,
        output_path: Path,
        groupby_columns: List[str] = None,
        explodecollections: bool = False,
        input_layer: str = None,        
        output_layer: str = None,
        verbose: bool = False,
        force: bool = False):

    ##### Init #####
    start_time = datetime.datetime.now()
    if output_path.exists():
        if force is False:
            logger.info(f"Stop dissolve: Output exists already {output_path}")
            return
        else:
            geofile.remove(output_path)

    if input_layer is None:
        input_layer = geofile.get_only_layer(input_path)
    if output_layer is None:
        output_layer = geofile.get_default_layer(output_path)

    # Prepare the strings to use in the select statement
    if groupby_columns is not None:
        # Because the query uses a subselect, the groupby columns need to be prefixed
        columns_with_prefix = [f't."{column}"' for column in groupby_columns]
        groupby_columns_str = ", ".join(columns_with_prefix)
        groupby_columns_for_groupby_str = groupby_columns_str
        groupby_columns_for_select_str = ", " + groupby_columns_str
    else:
        # Even if no groupby is provided, we still need to use a groupby clause, otherwise 
        # ST_union doesn't seem to work
        groupby_columns_for_groupby_str = "'1'"
        groupby_columns_for_select_str = ""

    # Remark: calculating the area in the enclosing selects halves the processing time

    sql_stmt = f"""
            SELECT sub.*, ST_area(sub.geom) AS area 
              FROM (SELECT ST_union(t.geom) AS geom{groupby_columns_for_select_str}
                      FROM {input_layer} t
                     GROUP BY {groupby_columns_for_groupby_str}) sub"""
    sql_stmt = f"""
            SELECT ST_union(t.geom) AS geom{groupby_columns_for_select_str}
              FROM {input_layer} t
             GROUP BY {groupby_columns_for_groupby_str}) sub"""
    sql_stmt = f"""
            SELECT ST_UnaryUnion(ST_Collect(t.geom)) AS geom{groupby_columns_for_select_str}
              FROM \"{input_layer}\" t
             GROUP BY {groupby_columns_for_groupby_str}"""
    sql_stmt = f"""
            SELECT ST_Collect(t.geom) AS geom{groupby_columns_for_select_str}
              FROM \"{input_layer}\" t"""
    sql_stmt = f"""
            SELECT ST_union(t.geom) AS geom{groupby_columns_for_select_str}
              FROM \"{input_layer}\" t"""

    sql_stmt = f"""
        SELECT ST_union(t.geom) AS geom{groupby_columns_for_select_str}
            FROM \"{input_layer}\" t
            GROUP BY {groupby_columns_for_groupby_str}"""

    translate_description = f"Dissolve {input_path}"
    ogr_util.vector_translate(
            input_path=input_path,
            output_path=output_path,
            translate_description=translate_description,
            output_layer=output_layer,
            sql_stmt=sql_stmt,
            sql_dialect='SQLITE',
            force_output_geometrytype='MULTIPOLYGON',
            explodecollections=explodecollections,
            verbose=verbose)

    logger.info(f"Processing ready, took {datetime.datetime.now()-start_time}!")

def dissolve_cardsheets(
        input_path: Path,
        input_cardsheets_path: Path,
        output_path: Path,
        groupby_columns: List[str] = None,
        explodecollections: bool = False,
        input_layer: str = None,        
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    ##### Init #####
    start_time = datetime.datetime.now()
    if output_path.exists():
        if force is False:
            logger.info(f"Stop dissolve_cardsheets: output exists already {output_path}, so stop")
            return
        else:
            geofile.remove(output_path)
    if nb_parallel == -1:
        nb_parallel = multiprocessing.cpu_count()
        if nb_parallel > 4:
            nb_parallel -= 1

    # Get input data to temp gpkg file
    tempdir = io_util.create_tempdir("dissolve_cardsheets")
    input_tmp_path = tempdir / "input_layers.gpkg"
    if(input_path.suffix.lower() == '.gpkg'):
        logger.info(f"Copy {input_path} to {input_tmp_path}")
        geofile.copy(input_path, input_tmp_path)
        logger.debug("Copy ready")
    else:
        # Remark: this temp file doesn't need spatial index
        logger.info(f"Copy {input_path} to {input_tmp_path} using ogr2ogr")
        ogr_util.vector_translate(
                input_path=input_path,
                output_path=input_tmp_path,
                create_spatial_index=False,
                output_layer=input_layer,
                verbose=verbose)
        logger.debug("Copy ready")

    if input_layer is None:
        input_layer = geofile.get_only_layer(input_tmp_path)
    if output_layer is None:
        output_layer = geofile.get_default_layer(output_path)

    ##### Prepare tmp files #####

    # Prepare the strings to use in the select statement
    if groupby_columns is not None:
        # Because the query uses a subselect, the groupby columns need to be prefixed
        columns_with_prefix = [f"t.{column}" for column in groupby_columns]
        groupby_columns_str = ", ".join(columns_with_prefix)
        groupby_columns_for_groupby_str = groupby_columns_str
        groupby_columns_for_select_str = ", " + groupby_columns_str
    else:
        # Even if no groupby is provided, we still need to use a groupby clause, otherwise 
        # ST_union doesn't seem to work
        groupby_columns_for_groupby_str = "'1'"
        groupby_columns_for_select_str = ""

    # Load the cardsheets we want the dissolve to be bound on
    cardsheets_gdf = geofile.read_file(input_cardsheets_path)

    try:
        # Start calculation of intersections in parallel
        logger.info(f"Start calculation of dissolves in file {input_tmp_path} to partial files")
        tmp_output_path = tempdir / output_path.name

        with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

            translate_jobs = {}    
            future_to_batch_id = {}    
            nb_batches = len(cardsheets_gdf)
            for batch_id, cardsheet in enumerate(cardsheets_gdf.itertuples()):
        
                translate_jobs[batch_id] = {}
                translate_jobs[batch_id]['layer'] = output_layer

                output_tmp_partial_path = tempdir / f"{output_path.stem}_{batch_id}{output_path.suffix}"
                translate_jobs[batch_id]['tmp_partial_output_path'] = output_tmp_partial_path

                # Remarks: 
                #   - calculating the area in the enclosing selects halves the processing time
                #   - ST_union() gives same performance as ST_unaryunion(ST_collect())!
                bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = cardsheet.geometry.bounds  
                bbox_wkt = f"POLYGON (({bbox_xmin} {bbox_ymin}, {bbox_xmax} {bbox_ymin}, {bbox_xmax} {bbox_ymax}, {bbox_xmin} {bbox_ymax}, {bbox_xmin} {bbox_ymin}))"
                sql_stmt = f"""
                        SELECT ST_union(ST_intersection(t.geom, ST_GeomFromText('{bbox_wkt}'))) AS geom{groupby_columns_for_select_str}
                          FROM {input_layer} t
                          JOIN rtree_{input_layer}_geom t_tree ON t.fid = t_tree.id
                         WHERE t_tree.minx <= {bbox_xmax} AND t_tree.maxx >= {bbox_xmin}
                           AND t_tree.miny <= {bbox_ymax} AND t_tree.maxy >= {bbox_ymin}
                           AND ST_Intersects(t.geom, ST_GeomFromText('{bbox_wkt}')) = 1
                           AND ST_Touches(t.geom, ST_GeomFromText('{bbox_wkt}')) = 0
                         GROUP BY {groupby_columns_for_groupby_str}"""
                if explodecollections is True:
                    force_output_geometrytype = 'POLYGON'
                else:
                    force_output_geometrytype = 'MULTIPOLYGON'

                translate_jobs[batch_id]['sqlite_stmt'] = sql_stmt
                translate_description = f"Async dissolve {batch_id} of {nb_batches}, bounds: {cardsheet.geometry.bounds}"
                # Remark: this temp file doesn't need spatial index
                translate_info = ogr_util.VectorTranslateInfo(
                        input_path=input_tmp_path,
                        output_path=output_tmp_partial_path,
                        translate_description=translate_description,
                        output_layer=output_layer,
                        #clip_bounds=cardsheet.geometry.bounds,
                        sql_stmt=sql_stmt,
                        sql_dialect='SQLITE',
                        append=True,
                        update=True,
                        explodecollections=True,
                        force_output_geometrytype=force_output_geometrytype,
                        verbose=verbose)
                future = ogr_util.vector_translate_async(
                        concurrent_pool=calculate_pool, info=translate_info)
                future_to_batch_id[future] = batch_id
            
            # Loop till all parallel processes are ready, but process each one that is ready already
            for future in futures.as_completed(future_to_batch_id):
                try:
                    _ = future.result()

                    # Start copy of the result to a common file
                    # Remark: give higher priority, because this is the slowest factor
                    batch_id = future_to_batch_id[future]
                    # If the calculate gave results, copy to output
                    tmp_partial_output_path = translate_jobs[batch_id]['tmp_partial_output_path']
                    if tmp_partial_output_path.exists():
                        translate_description = f"Copy result {batch_id} of {nb_batches} to {output_layer}"
                        translate_info = ogr_util.VectorTranslateInfo(
                                input_path=tmp_partial_output_path,
                                output_path=tmp_output_path,
                                translate_description=translate_description,
                                output_layer=output_layer,
                                transaction_size=200000,
                                append=True,
                                update=True,
                                create_spatial_index=False,
                                force_output_geometrytype='MULTIPOLYGON',
                                priority_class='NORMAL',
                                verbose=verbose)
                        ogr_util.vector_translate_by_info(info=translate_info)
                        geofile.remove(tmp_partial_output_path)
                except Exception as ex:
                    batch_id = future_to_batch_id[future]
                    #calculate_pool.shutdown()
                    logger.error(f"Error executing {translate_jobs[batch_id]}: {ex}")

        ##### Round up and clean up ##### 
        # Now create spatial index and move to output location
        geofile.create_spatial_index(path=tmp_output_path, layer=output_layer)
        geofile.move(tmp_output_path, output_path)
    finally:
        # Clean tmp dir
        shutil.rmtree(tempdir)
        logger.info(f"Processing ready, took {datetime.datetime.now()-start_time}!")

if __name__ == '__main__':
    raise Exception("Not implemented!")
