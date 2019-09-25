from concurrent import futures
import datetime
import logging
import logging.config
import multiprocessing
import os
import shutil
from io import StringIO
import re
import tempfile
import time

# TODO: on windows, the init of this doensn't seem to work properly... should be solved somewhere else?
if os.name == 'nt':
    os.environ["GDAL_DATA"] = r"C:\Tools\anaconda3\envs\geofile_ops\Library\share\gdal"
    os.environ["PATH"] += os.pathsep + r"C:\tools\anaconda3\envs\geofile_ops\Library\bin"

import fiona
import geopandas as gpd
from osgeo import gdal

from .util import io_util
from .util import geofile_util
from .util import ogr_util

################################################################################
# Some init
################################################################################

gdal.UseExceptions()        # Enable exceptions
logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

def getfileinfo(
        path: str,
        verbose: bool = False) -> dict:
            
    # Get info
    info_str = ogr_util.vector_info(
            path=path, 
            readonly=True,
            verbose=verbose)

    # Prepare result
    result_dict = {}
    result_dict['info_str'] = info_str
    result_dict['layers'] = []
    info_strio = StringIO(info_str)
    for line in info_strio.readlines():
        line = line.strip()
        if re.match(r"\A\d: ", line):
            # It is a layer, now extract only the layer name
            logger.debug(f"This is a layer: {line}")
            layername_with_geomtype = re.sub(r"\A\d: ", "", line)
            layername = re.sub(r" [(][a-zA-Z ]+[)]\Z", "", layername_with_geomtype)
            result_dict['layers'].append(layername)

    return result_dict

def getlayerinfo(
        path: str,
        layer: str = None,
        verbose: bool = False) -> dict:
        
    ##### Init #####
    if layer is None:
        layer = get_default_layer(path)
    
    ##### Get summary info #####
    info_str = ogr_util.vector_info(
            path=path, 
            layer=layer,
            readonly=True,
            report_summary=True,
            verbose=verbose)

    # Prepare result
    result_dict = {}
    result_dict['info_str'] = info_str
    info_strio = StringIO(info_str)
    for line in info_strio.readlines():
        if line.startswith("Feature Count: "):
            line_value = line.strip().replace("Feature Count: ", "")
            result_dict['featurecount'] = int(line_value)
        elif line.startswith("Geometry Column = "):
            line_value = line.strip().replace("Geometry Column = ", "")
            result_dict['geometry_column'] = line_value

    # If no geometry_column info found, check if specific type of file
    if 'geometry_column' not in result_dict:
        _, ext = os.path.splitext(path)
        ext_lower = ext.lower()
        if ext_lower == '.shp':
            result_dict['geometry_column'] = 'geometry'

    ##### Get (non geometry) columns #####
    with fiona.open(path) as layer:
        result_dict['columns'] = layer.schema['properties'].keys()

    return result_dict

def create_spatial_index(
        path: str,
        layer: str = None,
        geometry_column: str = 'geom',
        verbose: bool = False):

    if layer is None:
        layer = get_only_layer(path)
    sqlite_stmt = f"SELECT CreateSpatialIndex('{layer}', '{geometry_column}')"
    ogr_util.vector_info(path=path, sqlite_stmt=sqlite_stmt)

def rename_layer(
        path: str,
        layer: str,
        new_layer: str,
        verbose: bool = False):

    if layer is None:
        layer = get_only_layer(path)
    sqlite_stmt = f'ALTER TABLE "{layer}" RENAME TO "{new_layer}"'
    ogr_util.vector_info(path=path, sqlite_stmt=sqlite_stmt)

def add_column(
        path: str,
        column_name: str,
        column_type: str = None,
        layer: str = None):

    ##### Init #####
    column_name = column_name.lower()
    if layer is None:
        layer = get_only_layer(path)
    if column_name not in ('area'):
        raise Exception(f"Unsupported column type: {column_type}")
    if column_type is None:
        if column_name == 'area':
            column_type = 'real'
        else:
            raise Exception(f"Columns type should be specified for colum name: {column_name}")

    ##### Go! #####
    sqlite_stmt = f'ALTER TABLE "{layer}" ADD COLUMN "{column_name}" {column_type}'

    try:
        ogr_util.vector_info(path=path, sqlite_stmt=sqlite_stmt, readonly=False)

        if column_name == 'area':
            sqlite_stmt = f'UPDATE "{layer}" SET "{column_name}" = ST_area(geom)'
            ogr_util.vector_info(path=path, sqlite_stmt=sqlite_stmt)
    except Exception as ex:
        # If the column exists already, just print warning
        if 'duplicate column name:' in str(ex):
            logger.warning(f"Column {column_name} existed already in {path}")
        else:
            raise ex

def select(
        input_path: str,
        output_path: str,
        sqlite_stmt: str,
        input_layer: str = None,        
        output_layer: str = None,
        verbose: bool = False,
        force: bool = False):

    ##### Init #####
    start_time = datetime.datetime.now()
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop select: output exists already {output_path}")
            return
        else:
            os.remove(output_path)

    if input_layer is None:
        input_layer = get_only_layer(input_path)
    if output_layer is None:
        output_layer = get_default_layer(output_path)

    ##### Exec #####
    translate_description = f"Select on {input_path}"
    ogr_util.vector_translate(
            input_path=input_path,
            output_path=output_path,
            translate_description=translate_description,
            output_layer=output_layer,
            sqlite_stmt=sqlite_stmt,
            verbose=verbose)

    logger.info(f"Processing ready, took {datetime.datetime.now()-start_time}!")

def convexhull(
        input_path: str,
        output_path: str,
        input_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    geom_operation_sqlite = f"ST_ConvexHull({{geom_column}})"
    geom_operation_description = "convexhull"

    return _single_layer_vector_operation(
            input_path=input_path,
            output_path=output_path,
            geom_operation_sqlite=geom_operation_sqlite,
            geom_operation_description=geom_operation_description,
            input_layer=input_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def buffer(
        input_path: str,
        output_path: str,
        buffer: float,
        quadrantsegments: int = 5,
        input_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    geom_operation_sqlite = f"ST_Buffer({{geom_column}}, {buffer}, {quadrantsegments})"
    geom_operation_description = "buffer"

    return _single_layer_vector_operation(
            input_path=input_path,
            output_path=output_path,
            geom_operation_sqlite=geom_operation_sqlite,
            geom_operation_description=geom_operation_description,
            input_layer=input_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def simplify(
        input_path: str,
        output_path: str,
        tolerance: float,        
        input_layer: str = None,        
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    geom_operation_sqlite = f"ST_Simplify({{geom_column}}, {tolerance})"
    geom_operation_description = "simplify"

    return _single_layer_vector_operation(
            input_path=input_path,
            output_path=output_path,
            geom_operation_sqlite=geom_operation_sqlite,
            geom_operation_description=geom_operation_description,
            input_layer=input_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def _single_layer_vector_operation(
        input_path: str,
        output_path: str,
        geom_operation_sqlite: str,
        geom_operation_description: str,
        input_layer: str = None,        
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    ##### Init #####
    start_time = datetime.datetime.now()
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop {geom_operation_description}: output exists already {output_path}")
            return
        else:
            os.remove(output_path)

    if input_layer is None:
        input_layer = get_only_layer(input_path)
    if output_layer is None:
        output_layer = get_default_layer(output_path)

    ##### Prepare tmp files #####
    tempdir = create_tempdir(geom_operation_description.replace(' ', '_'))
    logger.info(f"Start preparation of the temp files to calculate on in {tempdir}")

    try:
        input_tmp_path = input_path
        '''
        # Get input data to temp gpkg file
        input_tmp_path = os.path.join(tempdir, f"input_layers.gpkg")        
        _, input_ext = os.path.splitext(input_path)
        if(input_ext == '.gpkg'):
            logger.info("Input is already gpkg, just use it")
            input_tmp_path = input_path
            #logger.info(f"Copy {input_path} to {input_tmp_path}")
            #io_util.copyfile(input_path, input_tmp_path)
            #logger.debug("Copy ready")
        else:
            # Remark: this temp file doesn't need spatial index
            logger.info(f"Copy {input_path} to {input_tmp_path} using ogr2ogr")
            ogr_util.vector_translate(
                    input_path=input_path,
                    output_path=input_tmp_path,
                    create_spatial_index=False,
                    output_layer=input_layer)
            logger.debug("Copy ready")
        '''

        ##### Calculate #####
        logger.info(f"Start {geom_operation_description} on file {input_tmp_path} to partial files")
        # Calculating can be done in parallel, but only one process can write to 
        # the same file at the time... 
        if(nb_parallel == -1):
            nb_parallel = multiprocessing.cpu_count()
        nb_batches = nb_parallel*4
        with futures.ThreadPoolExecutor(nb_parallel) as calculate_pool, \
             futures.ThreadPoolExecutor(1) as write_result_pool:
            # Prepare columns to select
            layerinfo = getlayerinfo(input_path, input_layer)
            columns_to_select_str = ''
            if len(layerinfo['columns']) > 0:
                columns_to_select_str = f", {','.join(layerinfo['columns'])}"
            # Fill out the geometry column name in geom_operation_sqlite
            geom_operation_sqlite = geom_operation_sqlite.format(
                    geom_column=layerinfo['geometry_column'])
            # Calculate the number of features per thread
            nb_rows_input_layer = layerinfo['featurecount']
            row_limit = int(nb_rows_input_layer/nb_batches)
            row_offset = 0
            # Prepare output filename
            _, output_filename = os.path.split(output_path) 
            output_filename_noext, output_ext = os.path.splitext(output_filename) 
            tmp_output_path = os.path.join(tempdir, output_filename)

            translate_jobs = {}    
            future_to_translate_id = {}
            for translate_id in range(nb_batches):

                translate_jobs[translate_id] = {}
                translate_jobs[translate_id]['layer'] = output_layer

                output_tmp_partial_path = os.path.join(tempdir, f"{output_filename_noext}_{translate_id}{output_ext}")
                translate_jobs[translate_id]['tmp_partial_output_path'] = output_tmp_partial_path

                # For the last translate_id, take all rowid's left...
                if translate_id < nb_batches-1:
                    sqlite_stmt = f'''
                            SELECT {geom_operation_sqlite} AS geom{columns_to_select_str} 
                              FROM "{input_layer}"
                             WHERE rowid >= {row_offset}
                               AND rowid < {row_offset + row_limit}'''
                else:
                    sqlite_stmt = f'''
                            SELECT {geom_operation_sqlite} AS geom{columns_to_select_str} 
                              FROM "{input_layer}"
                             WHERE rowid >= {row_offset}'''
                translate_jobs[translate_id]['sqlite_stmt'] = sqlite_stmt
                translate_jobs[translate_id]['task_type'] = 'CALCULATE'
                translate_description = f"Async {geom_operation_description} {translate_id} of {nb_batches}"
                # Remark: this temp file doesn't need spatial index
                translate_info = ogr_util.VectorTranslateInfo(
                        input_path=input_tmp_path,
                        output_path=output_tmp_partial_path,
                        translate_description=translate_description,
                        output_layer=output_layer,
                        sqlite_stmt=sqlite_stmt,
                        create_spatial_index=False,
                        force_output_geometrytype='MULTIPOLYGON',
                        verbose=verbose)
                future = ogr_util.vector_translate_async(
                        concurrent_pool=calculate_pool, info=translate_info)
                future_to_translate_id[future] = translate_id
                row_offset += row_limit
            
            # Wait till all parallel processes are ready
            for future in futures.as_completed(future_to_translate_id):
                try:
                    _ = future.result()

                    # Start copy of the result to a common file
                    # Remark: give higher priority, because this is the slowest factor
                    translate_id = future_to_translate_id[future]
                    if translate_jobs[translate_id]['task_type'] == 'CALCULATE':
                        tmp_partial_output_path = translate_jobs[translate_id]['tmp_partial_output_path']
                        sqlite_stmt = f'SELECT * FROM "{output_layer}"'                   
                        translate_description = f"Copy result {translate_id} of {nb_batches} to {output_layer}"
                        
                        translate_info = ogr_util.VectorTranslateInfo(
                                input_path=tmp_partial_output_path,
                                output_path=tmp_output_path,
                                translate_description=translate_description,
                                output_layer=output_layer,
                                sqlite_stmt=sqlite_stmt,
                                transaction_size=200000,
                                append=True,
                                update=True,
                                create_spatial_index=False,
                                force_output_geometrytype='MULTIPOLYGON',
                                priority_class='NORMAL',
                                verbose=verbose)
                        translate_jobs[translate_id]['task_type'] = 'WRITE_RESULTS'
                        ogr_util.vector_translate_by_info(info=translate_info)
                        future_to_translate_id[future] = translate_id
                    elif translate_jobs[translate_id]['task_type'] == 'WRITE_RESULTS':
                        tmp_partial_output_path = translate_jobs[translate_id]['tmp_partial_output_path']
                        os.remove(tmp_partial_output_path)
                except Exception as ex:
                    translate_id = future_to_translate_id[future]
                    raise Exception(f"Error executing {translate_jobs[translate_id]}") from ex

        ##### Round up and clean up ##### 
        # Now create spatial index and move to output location
        create_spatial_index(path=tmp_output_path, layer=output_layer)
        shutil.move(tmp_output_path, output_path, copy_function=io_util.copyfile)
    finally:
        # Clean tmp dir
        shutil.rmtree(tempdir)
        logger.info(f"Processing ready, took {datetime.datetime.now()-start_time}!")
    
def intersect(
        input1_path: str,
        input2_path: str,
        output_path: str,
        input1_layer: str = None,
        input2_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    """
    Calculate the pairwise intersection of alle features in input1 with all 
    features in input2.
    
    Args:
        input1_path (str): [description]
        input2_path (str): [description]
        output_path (str): [description]
        input1_layer (str, optional): [description]. Defaults to None.
        input2_layer (str, optional): [description]. Defaults to None.
        output_layer (str, optional): [description]. Defaults to None.
        nb_parallel (int, optional): [description]. Defaults to -1.
        force (bool, optional): [description]. Defaults to False.
    
    Raises:
        Exception: [description]
    """

    ##### Init #####
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop intersect: output file exists already {output_path}, so stop")
            return
        else:
            os.remove(output_path)

    start_time = datetime.datetime.now()
    if input1_layer is None:
        input1_layer = get_only_layer(input1_path)
    if input2_layer is None:
        input2_layer = get_only_layer(input2_path)
    if output_layer is None:
        output_layer = get_default_layer(output_path)
    if(nb_parallel == -1):
        nb_parallel = multiprocessing.cpu_count()

    # Prepare tmp layer/file names
    tempdir = create_tempdir("intersect")
    if(input1_layer != input2_layer):
        input1_tmp_layer = input1_layer
        input2_tmp_layer = input2_layer
    else:
        input1_tmp_layer = 'l1_' + input1_layer
        input2_tmp_layer = 'l2_' + input2_layer
    input_tmp_path = os.path.join(tempdir, f"input_layers.gpkg")  

    ##### Prepare tmp files #####
    logger.info(f"Start preparation of the temp files to calculate on in {tempdir}")

    try:
        # Get input1 data to temp gpkg file
        _, input1_ext = os.path.splitext(input1_path)
        if(input1_ext == '.gpkg'):
            logger.debug(f"Copy {input1_path} to {input_tmp_path}")
            io_util.copyfile(input1_path, input_tmp_path)
        else:
            ogr_util.vector_translate(
                    input_path=input1_path,
                    output_path=input_tmp_path,
                    output_layer=input1_tmp_layer,
                    verbose=verbose)
        
        # Spread input2 data over different layers to be able to calculate in parallel
        split_jobs = split_layer_features(
                input_path=input2_path,
                input_layer=input2_layer,
                output_path=input_tmp_path,
                output_baselayer=input2_tmp_layer,
                nb_parts=nb_parallel,
                verbose=verbose)
        
        ##### Calculate intersections! #####
        # We need the input1 column names to format the select
        with fiona.open(input1_path) as layer:
            layer1_columns = layer.schema['properties'].keys()
        layer1_columns_in_subselect = [f"layer1.{column} l1_{column}" for column in layer1_columns]
        layer1_columns_in_subselect_str = ", ".join(layer1_columns_in_subselect)
        layer1_columns_in_select = [f"sub.l1_{column}" for column in layer1_columns]
        layer1_columns_in_select_str = ", ".join(layer1_columns_in_select)

        # We need the input2 column names to format the select
        with fiona.open(input2_path) as layer:
            layer2_columns = layer.schema['properties'].keys()
        layer2_columns_in_subselect = [f"layer2.{column} l2_{column}" for column in layer2_columns]
        layer2_columns_in_subselect_str = ", ".join(layer2_columns_in_subselect)
        layer2_columns_in_select = [f"sub.l2_{column}" for column in layer2_columns]
        layer2_columns_in_select_str = ", ".join(layer2_columns_in_select)

        # Start calculation of intersections in parallel
        logger.info(f"Start calculation of intersections in file {input_tmp_path} to partial files")
        _, output_filename = os.path.split(output_path) 
        output_filename_noext, output_ext = os.path.splitext(output_filename) 
        
        intersect_jobs = []
        for split_id in split_jobs:

            tmp_partial_output_path = os.path.join(tempdir, f"{output_filename_noext}_{split_id}{output_ext}")
            tmp_partial_output_layer = get_default_layer(tmp_partial_output_path)
            input2_tmp_curr_layer = split_jobs[split_id]['layer']
            sqlite_stmt = f"""
                    SELECT sub.geom, ST_area(sub.geom) area_inter
                          ,{layer1_columns_in_select_str}
                          ,{layer2_columns_in_select_str}
                        FROM (SELECT ST_Multi(ST_Intersection(layer1.geom, layer2.geom)) AS geom
                                    ,{layer1_columns_in_subselect_str}
                                    ,{layer2_columns_in_subselect_str}
                                FROM \"{input1_tmp_layer}\" layer1
                                JOIN \"rtree_{input1_tmp_layer}_geom\" layer1tree ON layer1.fid = layer1tree.id
                                JOIN \"{input2_tmp_curr_layer}\" layer2
                                JOIN \"rtree_{input2_tmp_curr_layer}_geom\" layer2tree ON layer2.fid = layer2tree.id
                               WHERE 1=1
                                 AND layer1tree.minx <= layer2tree.maxx AND layer1tree.maxx >= layer2tree.minx
                                 AND layer1tree.miny <= layer2tree.maxy AND layer1tree.maxy >= layer2tree.miny
                                 AND ST_Intersects(layer1.geom, layer2.geom) = 1
                                 AND ST_Touches(layer1.geom, layer2.geom) = 0
                            ) sub
                        WHERE GeometryType(sub.geom) IN ('POLYGON', 'MULTIPOLYGON')"""

            translate_description = f"Calculate intersect between {input_tmp_path} and {tmp_partial_output_path}"
            intersect_info = ogr_util.VectorTranslateInfo(
                    input_path=input_tmp_path,
                    output_path=tmp_partial_output_path,
                    translate_description=translate_description,
                    output_layer=tmp_partial_output_layer,
                    sqlite_stmt=sqlite_stmt,
                    #append=True,
                    force_output_geometrytype='MULTIPOLYGON',
                    transaction_size=5000,
                    verbose=verbose)
            intersect_jobs.append(intersect_info)

        # Start calculation in parallel!
        ogr_util.vector_translate_parallel(intersect_jobs, nb_parallel)

        ##### Round up and clean up ##### 
        # Combine all partial results
        logger.info(f"Start copy from partial temp files to one temp output file")
        tmp_output_path = os.path.join(tempdir, output_filename)
        for intersect_job in intersect_jobs:
            tmp_partial_output_path = intersect_job.output_path
            tmp_partial_output_layer = intersect_job.output_layer
            sqlite_stmt = f"SELECT * FROM \"{tmp_partial_output_layer}\""
            translate_description = f"Copy data from {tmp_partial_output_path} to file {tmp_output_path}"
            ogr_util.vector_translate(
                    input_path=tmp_partial_output_path,
                    output_path=tmp_output_path,
                    translate_description=translate_description,
                    output_layer=output_layer,
                    sqlite_stmt=sqlite_stmt,
                    append=True,
                    update=True,
                    force_output_geometrytype='MULTIPOLYGON',
                    verbose=verbose)
        shutil.move(tmp_output_path, output_path, copy_function=io_util.copyfile)

    finally:
        # Clean tmp dir
        shutil.rmtree(tempdir)
        logger.info(f"Processing ready, took {datetime.datetime.now()-start_time}!")

def export_by_location(
        input_to_select_from_path: str,
        input_to_compare_with_path: str,
        output_path: str,
        input1_layer: str = None,
        input2_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    sql_template = f'''
            SELECT geom, {{layer1_columns_in_subselect_str}}
                FROM "{{input1_tmp_layer}}" layer1
                JOIN "rtree_{{input1_tmp_layer}}_geom" layer1tree ON layer1.fid = layer1tree.id
                WHERE 1=1
                AND EXISTS (
                    SELECT 1 
                        FROM "{{input2_tmp_layer}}" layer2
                        JOIN "rtree_{{input2_tmp_layer}}_geom" layer2tree ON layer2.fid = layer2tree.id
                        WHERE layer1tree.minx <= layer2tree.maxx AND layer1tree.maxx >= layer2tree.minx
                        AND layer1tree.miny <= layer2tree.maxy AND layer1tree.maxy >= layer2tree.miny
                        AND ST_Intersects(layer1.geom, layer2.geom) = 1
                        AND ST_Touches(layer1.geom, layer2.geom) = 0)
            '''
    sql_template = f'''
        SELECT ST_union(layer1.geom) as geom
              ,{{layer1_columns_in_subselect_str}}
              ,ST_area(ST_intersection(ST_union(layer1.geom), ST_union(layer2.geom))) as area_inters
            FROM "{{input1_tmp_layer}}" layer1
            JOIN "rtree_{{input1_tmp_layer}}_geom" layer1tree ON layer1.fid = layer1tree.id
            JOIN "{{input2_tmp_layer}}" layer2
            JOIN "rtree_{{input2_tmp_layer}}_geom" layer2tree ON layer2.fid = layer2tree.id
           WHERE 1=1
             AND layer1tree.minx <= layer2tree.maxx AND layer1tree.maxx >= layer2tree.minx
             AND layer1tree.miny <= layer2tree.maxy AND layer1tree.maxy >= layer2tree.miny
             AND ST_Intersects(layer1.geom, layer2.geom) = 1
             AND ST_Touches(layer1.geom, layer2.geom) = 0
           GROUP BY {{layer1_columns_in_groupby_str}}
        '''
    geom_operation_description = "export_by_location"

    return _two_layer_vector_operation(
            input1_path=input_to_select_from_path,
            input2_path=input_to_compare_with_path,
            output_path=output_path,
            sql_template=sql_template,
            geom_operation_description=geom_operation_description,
            input1_layer=input1_layer,
            input2_layer=input2_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def export_by_distance(
        input_to_select_from_path: str,
        input_to_compare_with_path: str,
        output_path: str,
        max_distance: float,
        input1_layer: str = None,
        input2_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    sql_template = f'''
            SELECT geom, {{layer1_columns_in_subselect_str}}
                FROM "{{input1_tmp_layer}}" layer1
                JOIN "rtree_{{input1_tmp_layer}}_geom" layer1tree ON layer1.fid = layer1tree.id
                WHERE 1=1
                AND EXISTS (
                    SELECT 1 
                        FROM "{{input2_tmp_layer}}" layer2
                        JOIN "rtree_{{input2_tmp_layer}}_geom" layer2tree ON layer2.fid = layer2tree.id
                        WHERE (layer1tree.minx-{max_distance}) <= layer2tree.maxx 
                          AND (layer1tree.maxx+{max_distance}) >= layer2tree.minx
                          AND (layer1tree.miny-{max_distance}) <= layer2tree.maxy 
                          AND (layer1tree.maxy+{max_distance}) >= layer2tree.miny
                          AND ST_distance(layer1.geom, layer2.geom) <= {max_distance})
            '''
    geom_operation_description = "export_by_distance"

    return _two_layer_vector_operation(
            input1_path=input_to_select_from_path,
            input2_path=input_to_compare_with_path,
            output_path=output_path,
            sql_template=sql_template,
            geom_operation_description=geom_operation_description,
            input1_layer=input1_layer,
            input2_layer=input2_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def _two_layer_vector_operation(
        input1_path: str,
        input2_path: str,
        output_path: str,
        sql_template: str,
        geom_operation_description: str,
        input1_layer: str = None,
        input2_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    """
    Export the features that intersect with elements of another layer.

    Similar to the typical "select by location" tool.
    
    Args:
        input1_path (str): the file to export features from
        input2_path (str): the file to check intersections with
        output_path (str): output file
        input1_layer (str, optional): [description]. Defaults to None.
        input2_layer (str, optional): [description]. Defaults to None.
        output_layer (str, optional): [description]. Defaults to None.
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
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop _two_layer_vector_operation: output exists already {output_path}")
            return
        else:
            os.remove(output_path)

    start_time = datetime.datetime.now()
    if input1_layer is None:
        input1_layer = get_only_layer(input1_path)
    if input2_layer is None:
        input2_layer = get_only_layer(input2_path)
    if output_layer is None:
        output_layer = get_default_layer(output_path)

    # Prepare tmp layer/file names
    tempdir = create_tempdir("export_by_location")
    if(input1_layer != input2_layer):
        input1_tmp_layer = input1_layer
        input2_tmp_layer = input2_layer
    else:
        input1_tmp_layer = 'l1_' + input1_layer
        input2_tmp_layer = 'l2_' + input2_layer
    input_tmp_path = os.path.join(tempdir, f"input_layers.gpkg") 

    ##### Prepare tmp files #####
    logger.info(f"Start preparation of the temp files to calculate on in {tempdir}")

    try:
        # Get input2 data to temp gpkg file
        _, input2_ext = os.path.splitext(input2_path)
        if(input2_ext == '.gpkg'):
            logger.debug(f"Copy {input2_path} to {input_tmp_path}")
            io_util.copyfile(input2_path, input_tmp_path)
        else:
            ogr_util.vector_translate(
                    input_path=input2_path,
                    output_path=input_tmp_path,
                    output_layer=input2_tmp_layer,
                    verbose=verbose)
        
        # Spread input1 data over different layers to be able to calculate in parallel
        if(nb_parallel == -1):
            nb_parallel = multiprocessing.cpu_count()
        nb_batches = nb_parallel*4
        split_jobs = split_layer_features(
                input_path=input1_path,
                input_layer=input1_layer,
                output_path=input_tmp_path,
                output_baselayer=input1_tmp_layer,
                nb_parts=nb_batches,
                verbose=verbose)
        
        ##### Calculate! #####
        # We need the input1 column names to format the select
        with fiona.open(input1_path) as layer:
            layer1_columns = layer.schema['properties'].keys()
        layer1_columns_in_subselect = [f"layer1.{column} l1_{column}" for column in layer1_columns]
        layer1_columns_in_subselect_str = ", ".join(layer1_columns_in_subselect)
        layer1_columns_in_select = [f"sub.l1_{column}" for column in layer1_columns]
        layer1_columns_in_select_str = ", ".join(layer1_columns_in_select)
        layer1_columns_in_groupby = [f"layer1.{column}" for column in layer1_columns]
        layer1_columns_in_groupby_str = ", ".join(layer1_columns_in_groupby)

        # We need the input2 column names to format the select
        with fiona.open(input2_path) as layer:
            layer2_columns = layer.schema['properties'].keys()
        layer2_columns_in_subselect = [f"layer2.{column} l2_{column}" for column in layer2_columns]
        layer2_columns_in_subselect_str = ", ".join(layer2_columns_in_subselect)
        layer2_columns_in_select = [f"sub.l2_{column}" for column in layer2_columns]
        layer2_columns_in_select_str = ", ".join(layer2_columns_in_select)
        layer2_columns_in_groupby = [f"layer2.{column}" for column in layer2_columns]
        layer2_columns_in_groupby_str = ", ".join(layer2_columns_in_groupby)        

        # Fill out the geometry column name in geom_operation_sqlite
        # TODO: check if geom column is always geom
        #geom_operation_sqlite = geom_operation_sqlite.format(
        #        geom_column=layerinfo['geometry_column'])
        # Calculate the number of features per thread
        #nb_rows_input_layer = layerinfo['featurecount']
        #row_limit = int(nb_rows_input_layer/nb_batches)
        #row_offset = 0
        # Prepare output filename
        _, output_filename = os.path.split(output_path) 
        output_filename_noext, output_ext = os.path.splitext(output_filename) 
        tmp_output_path = os.path.join(tempdir, output_filename)
        
        ##### Calculate #####
        logger.info(f"Start {geom_operation_description} on file {input_tmp_path} to partial files")
        # Calculating can be done in parallel, but only one process can write to 
        # the same file at the time... 
        with futures.ThreadPoolExecutor(nb_parallel) as calculate_pool, \
             futures.ThreadPoolExecutor(1) as write_result_pool:

            # Start looping
            translate_jobs = {}    
            future_to_translate_id = {}
            #for translate_id in range(nb_batches):
            for translate_id in split_jobs:

                translate_jobs[translate_id] = {}
                translate_jobs[translate_id]['layer'] = output_layer

                tmp_output_partial_path = os.path.join(tempdir, f"{output_filename_noext}_{translate_id}{output_ext}")
                translate_jobs[translate_id]['tmp_partial_output_path'] = tmp_output_partial_path

                input1_tmp_curr_layer = split_jobs[translate_id]['layer']
                sqlite_stmt = sql_template.format(
                        layer1_columns_in_subselect_str=layer1_columns_in_subselect_str,
                        input1_tmp_layer=input1_tmp_curr_layer,
                        input2_tmp_layer=input2_tmp_layer,
                        layer1_columns_in_groupby_str=layer1_columns_in_groupby_str)

                translate_jobs[translate_id]['sqlite_stmt'] = sqlite_stmt
                translate_jobs[translate_id]['task_type'] = 'CALCULATE'
                translate_description = f"Calculate export_by_location between {input_tmp_path} and {tmp_output_partial_path}"
                # Remark: this temp file doesn't need spatial index
                translate_info = ogr_util.VectorTranslateInfo(
                        input_path=input_tmp_path,
                        output_path=tmp_output_partial_path,
                        translate_description=translate_description,
                        output_layer=output_layer,
                        sqlite_stmt=sqlite_stmt,
                        create_spatial_index=False,
                        force_output_geometrytype='MULTIPOLYGON',
                        verbose=verbose)
                future = ogr_util.vector_translate_async(
                        concurrent_pool=calculate_pool, info=translate_info)
                future_to_translate_id[future] = translate_id
                #row_offset += row_limit

            # Wait till all parallel processes are ready
            for future in futures.as_completed(future_to_translate_id):
                try:
                    _ = future.result()

                    # Start copy of the result to a common file
                    # Remark: give higher priority, because this is the slowest factor
                    translate_id = future_to_translate_id[future]
                    if translate_jobs[translate_id]['task_type'] == 'CALCULATE':
                        tmp_partial_output_path = translate_jobs[translate_id]['tmp_partial_output_path']
                        sqlite_stmt = f'SELECT * FROM "{output_layer}"'                   
                        translate_description = f"Copy result {translate_id} of {nb_batches} to {output_layer}"
                        
                        translate_info = ogr_util.VectorTranslateInfo(
                                input_path=tmp_partial_output_path,
                                output_path=tmp_output_path,
                                translate_description=translate_description,
                                output_layer=output_layer,
                                sqlite_stmt=sqlite_stmt,
                                transaction_size=200000,
                                append=True,
                                update=True,
                                create_spatial_index=False,
                                force_output_geometrytype='MULTIPOLYGON',
                                priority_class='NORMAL',
                                verbose=verbose)
                        translate_jobs[translate_id]['task_type'] = 'WRITE_RESULTS'
                        ogr_util.vector_translate_by_info(info=translate_info)
                        future_to_translate_id[future] = translate_id
                    elif translate_jobs[translate_id]['task_type'] == 'WRITE_RESULTS':
                        tmp_partial_output_path = translate_jobs[translate_id]['tmp_partial_output_path']
                        os.remove(tmp_partial_output_path)
                except Exception as ex:
                    translate_id = future_to_translate_id[future]
                    raise Exception(f"Error executing {translate_jobs[translate_id]}") from ex

        ##### Round up and clean up ##### 
        # Now create spatial index and move to output location
        create_spatial_index(path=tmp_output_path, layer=output_layer)
        shutil.move(tmp_output_path, output_path, copy_function=io_util.copyfile)
    finally:
        # Clean tmp dir
        shutil.rmtree(tempdir)
        logger.info(f"Processing ready, took {datetime.datetime.now()-start_time}!")

def dissolve(
        input_path: str,
        output_path: str,
        groupby_columns: [] = None,
        explodecollections: bool = False,
        input_layer: str = None,        
        output_layer: str = None,
        verbose: bool = False,
        force: bool = False):

    ##### Init #####
    start_time = datetime.datetime.now()
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop dissolve: Output exists already {output_path}")
            return
        else:
            os.remove(output_path)

    if input_layer is None:
        input_layer = get_only_layer(input_path)
    if output_layer is None:
        output_layer = get_default_layer(output_path)

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

    # Remark: calculating the area in the enclosing selects halves the processing time
    '''
    sqlite_stmt = f"""
            SELECT sub.*, ST_area(sub.geom) AS area 
              FROM (SELECT ST_union(t.geom) AS geom{groupby_columns_for_select_str}
                      FROM {input_layer} t
                     GROUP BY {groupby_columns_for_groupby_str}) sub"""
    sqlite_stmt = f"""
            SELECT ST_union(t.geom) AS geom{groupby_columns_for_select_str}
              FROM {input_layer} t
             GROUP BY {groupby_columns_for_groupby_str}) sub"""
    sqlite_stmt = f"""
            SELECT ST_UnaryUnion(ST_Collect(t.geom)) AS geom{groupby_columns_for_select_str}
              FROM \"{input_layer}\" t
             GROUP BY {groupby_columns_for_groupby_str}"""
    sqlite_stmt = f"""
            SELECT ST_Collect(t.geom) AS geom{groupby_columns_for_select_str}
              FROM \"{input_layer}\" t"""
    sqlite_stmt = f"""
            SELECT ST_union(t.geom) AS geom{groupby_columns_for_select_str}
              FROM \"{input_layer}\" t"""
    '''
    sqlite_stmt = f"""
        SELECT ST_union(t.geom) AS geom{groupby_columns_for_select_str}
            FROM \"{input_layer}\" t
            GROUP BY {groupby_columns_for_groupby_str}"""

    translate_description = f"Dissolve {input_path}"
    ogr_util.vector_translate(
            input_path=input_path,
            output_path=output_path,
            translate_description=translate_description,
            output_layer=output_layer,
            sqlite_stmt=sqlite_stmt,
            force_output_geometrytype='MULTIPOLYGON',
            explodecollections=explodecollections,
            verbose=verbose)

    logger.info(f"Processing ready, took {datetime.datetime.now()-start_time}!")

def dissolve_cardsheets(
        input_path: str,
        input_cardsheets_path: str,
        output_path: str,
        groupby_columns: [] = None,
        explodecollections: bool = False,
        input_layer: str = None,        
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    ##### Init #####
    start_time = datetime.datetime.now()
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop dissolve_cardsheets: output exists already {output_path}, so stop")
            return
        else:
            os.remove(output_path)

    if input_layer is None:
        input_layer = get_only_layer(input_path)
    if output_layer is None:
        output_layer = get_default_layer(output_path)

    # Prepare tmp layer/file names
    tempdir = create_tempdir("dissolve_cardsheets")
    input_tmp_path = os.path.join(tempdir, f"input_layers.gpkg")
    _, output_filename = os.path.split(output_path)
    output_tmp_path = os.path.join(tempdir, output_filename)
    if(nb_parallel == -1):
        nb_parallel = multiprocessing.cpu_count()

    ##### Prepare tmp files #####
    logger.info(f"Start preparation of the temp files to calculate on in {tempdir}")

    # Get input data to temp gpkg file
    _, input_ext = os.path.splitext(input_path)
    if(input_ext == '.gpkg'):
        logger.info(f"Copy {input_path} to {input_tmp_path}")
        io_util.copyfile(input_path, input_tmp_path)
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
    cardsheets_gdf = geofile_util.read_file(input_cardsheets_path)

    # Start calculation of intersections in parallel
    logger.info(f"Start calculation of dissolves in file {input_tmp_path} to partial files")
    _, output_filename = os.path.split(output_path) 
    output_filename_noext, output_ext = os.path.splitext(output_filename)

    translate_jobs = []    
    for translate_id, cardsheet in enumerate(cardsheets_gdf.itertuples()):
        
        translate_description = f"Dissolve cardsheet id: {translate_id}, bounds: {cardsheet.geometry.bounds}"
        output_tmp_partial_path = os.path.join(tempdir, f"{output_filename_noext}_{translate_id}{output_ext}")

        # Remarks: 
        #   - calculating the area in the enclosing selects halves the processing time
        #   - ST_union() gives same performance as ST_unaryunion(ST_collect())!
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = cardsheet.geometry.bounds  
        bbox_wkt = f"POLYGON (({bbox_xmin} {bbox_ymin}, {bbox_xmax} {bbox_ymin}, {bbox_xmax} {bbox_ymax}, {bbox_xmin} {bbox_ymax}, {bbox_xmin} {bbox_ymin}))"
        sqlite_stmt = f"""
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

        translate_info = ogr_util.VectorTranslateInfo(
                input_path=input_path,
                output_path=output_tmp_partial_path,
                translate_description=translate_description,
                output_layer=output_layer,
                #clip_bounds=cardsheet.geometry.bounds,
                sqlite_stmt=sqlite_stmt,
                append=True,
                update=True,
                explodecollections=True,
                force_output_geometrytype=force_output_geometrytype,
                verbose=verbose)
        translate_jobs.append(translate_info)

    ogr_util.vector_translate_parallel(translate_jobs)

    ##### Round up and clean up ##### 
    # Combine all partial results
    logger.info(f"Start copy from partial temp files to one temp output file")
    tmp_output_path = os.path.join(tempdir, output_filename)
    for translate_id in translate_jobs:
        tmp_partial_output_path = translate_jobs[translate_id].output_path
        sqlite_stmt = f"SELECT * FROM {output_layer}"
        logger.debug(f"Copy data from {tmp_partial_output_path} to {tmp_output_path}.{output_layer}")
        # First create without index, this add the index once all data is there
        ogr_util.vector_translate(
                input_path=tmp_partial_output_path,
                output_path=tmp_output_path,
                output_layer=output_layer,
                sqlite_stmt=sqlite_stmt,
                append=True,
                update=True,
                #force_output_geometrytype='MULTIPOLYGON'
                verbose=verbose)

    # Now create spatial index and move to output location
    shutil.move(tmp_output_path, output_path, copy_function=io_util.copyfile)

    # Clean tmp dir
    shutil.rmtree(tempdir)
    logger.info(f"Processing ready, took {datetime.datetime.now()-start_time}!")

def create_tempdir(base_dirname: str) -> str:
    #base_tempdir = os.path.join(tempfile.gettempdir(), base_dirname)
    base_tempdir = os.path.join(r"C:\temp", base_dirname)

    tempdir_ok = False
    for i in range(1, 9999):
        try:
            tempdir = f"{base_tempdir}_{i:04d}"
            os.mkdir(tempdir)
            tempdir_ok = True
            break
        except FileExistsError:
            continue

    if tempdir_ok:
        return tempdir
    else:
        raise Exception(f"Wasn't able to create a temporary dir with basedir: {base_tempdir}")

def split_layer_features(
        input_path: str,
        input_layer: str,
        output_path: str,
        output_baselayer: str,
        nb_parts: int,
        verbose: bool = False) -> dict:

    ##### Init #####
    # Get column names
    with fiona.open(input_path) as layer:
        columns = layer.schema['properties'].keys()
    columns_to_select_str = f", {','.join(columns)}"

    ##### Split to x files/layers #####
    # Get input2 data to temp file, but divide it over several parts
    # Remark: adding data to a file in parallel using ogr2ogr gives locking 
    # issues on the sqlite file, so needs to be done sequential!
    split_jobs = {}
    layerinfo = getlayerinfo(input_path, input_layer)
    nb_rows_input_layer = layerinfo['featurecount']
    row_limit = int(nb_rows_input_layer/nb_parts)
    row_offset = 0

    if layerinfo['geometry_column'] == 'geom':
        geometry_column_for_select = 'geom'
    else:
        geometry_column_for_select = f"{layerinfo['geometry_column']} geom"

    for split_job_id in range(nb_parts):
        # Prepare destination layer name
        split_jobs[split_job_id] = {}
        output_baselayer_stripped = output_baselayer.strip("'\"")
        output_layer_curr = f"{output_baselayer_stripped}_{split_job_id}"
        split_jobs[split_job_id]['layer'] = output_layer_curr
        
        # For the last batch, don't limit anymore
        if split_job_id >= nb_parts-1:
            row_limit = -1       

        sqlite_stmt = f"""
                SELECT {geometry_column_for_select}{columns_to_select_str} FROM \"{input_layer}\"
                 LIMIT {row_limit}
                OFFSET {row_offset}"""
        
        translate_description=f"Copy data from {input_path}.{input_layer} to {output_path}.{output_layer_curr}"
        ogr_util.vector_translate(
                input_path=input_path,
                output_path=output_path,
                translate_description=translate_description,
                output_layer=output_layer_curr,
                sqlite_stmt=sqlite_stmt,
                append=True,
                force_output_geometrytype='MULTIPOLYGON',
                verbose=verbose)
        row_offset += row_limit

    return split_jobs

def get_only_layer(path: str):
    
    fileinfo = getfileinfo(path)
    nb_layers = len(fileinfo['layers'])
    if nb_layers == 1:
        return fileinfo['layers'][0]
    elif nb_layers == 0:
        raise Exception(f"Error: No layers found in {path}")
    else:
        raise Exception(f"Error: More than 1 layer found in {path}: {fileinfo['layers']}")

def get_default_layer(path: str):
    _, filename = os.path.split(path)
    layername, _ = os.path.splitext(filename)
    return layername

if __name__ == '__main__':
    raise Exception("Not implemented!")
