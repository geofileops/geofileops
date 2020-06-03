from concurrent import futures
import datetime
from io import StringIO
import logging
import logging.config
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, AnyStr, List, Optional, Tuple, Union

# TODO: on windows, the init of this doensn't seem to work properly... should be solved somewhere else?
#if os.name == 'nt':
#    os.environ['GDAL_DATA'] = r'C:\Tools\anaconda3\envs\geofileops\Library\share\gdal'
#    os.environ['PATH'] += os.pathsep + r'C:\tools\anaconda3\envs\geofileops\Library\bin'

import fiona
import geopandas as gpd
from osgeo import gdal
import pandas as pd
import psutil
import shapely.geometry as sh_geom

from .util import io_util
from .util import geofile_util
from .util import ogr_util as ogr_util
from .util import ogr_util_direct as ogr_util_direct

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
    datasource = gdal.OpenEx(path)
    if layer is not None:
        datasource_layer = datasource.GetLayer(layer)
    elif datasource.GetLayerCount() == 1:
        datasource_layer = datasource.GetLayerByIndex(0)
    else:
        raise Exception(f"No layer specified, and file has <> 1 layer: {path}")
    
    # Prepare result
    result_dict = {}
    result_dict['featurecount'] = datasource_layer.GetFeatureCount()
    result_dict['geometry_column'] = datasource_layer.GetGeometryColumn()

    # Get column names
    columns = []
    layer_defn = datasource_layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        columns.append(layer_defn.GetFieldDefn(i).GetName())
    result_dict['columns'] = columns
    '''
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
    with fiona.open(path) as fio_layer:
        result_dict['columns'] = fio_layer.schema['properties'].keys()
    '''

    return result_dict

def create_spatial_index(
        path: str,
        layer: str = None,
        geometry_column: str = 'geom',
        verbose: bool = False):

    if layer is None:
        layer = get_only_layer(path)

    '''
    sqlite_stmt = f"SELECT CreateSpatialIndex('{layer}', '{geometry_column}')"
    ogr_util.vector_info(path=path, sqlite_stmt=sqlite_stmt)
    '''
    #driver = ogr.GetDriverByName('SQLite')    
    #data_source = driver.CreateDataSource('db.sqlite', ['SPATIALITE=YES'])    
    data_source = gdal.OpenEx(path, nOpenFlags=gdal.OF_UPDATE)
    #layer = data_source.CreateLayer('the_table', None, ogr.wkbLineString25D, ['SPATIAL_INDEX=NO'])
    #geometry_column = layer.GetGeometryColumn()
    data_source.ExecuteSQL(f"SELECT CreateSpatialIndex('{layer}', '{geometry_column}')") 

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

    #geom_operation_sqlite = f"ST_Buffer({{geom_column}}, {buffer}, {quadrantsegments})"
    geom_operation_sqlite = f"ST_Buffer({{geom_column}}, {buffer})"
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

def buffer_gpd(
        input_path: str,
        output_path: str,
        buffer: float,
        quadrantsegments: int = 5,
        input_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    operation = "buffer"
    if input_layer is None:
        input_layer = get_only_layer(input_path)

    ##### Init #####
    start_time = datetime.datetime.now()
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path}")
            return
        else:
            os.remove(output_path)

    if input_layer is None:
        input_layer = get_only_layer(input_path)
    if output_layer is None:
        output_layer = get_default_layer(output_path)

    ##### Prepare tmp files #####
    tempdir = create_tempdir(operation.replace(' ', '_'))
    logger.info(f"Start calculation to temp files in {tempdir}")

    try:
        ##### Calculate #####
        # Remark: calculating can be done in parallel, but only one process 
        # can write to the same output file at the time...
        
        # Calculate the best number of parallel processes and batches for 
        # the available resources
        if(nb_parallel == -1):
            nb_parallel = multiprocessing.cpu_count()

        memory_basefootprint = 50*1024*1024
        memory_per_row = 100*1024*1024/30000   # Memory usage per row
        min_rows_per_batch = 5000
        memory_min_per_process = memory_basefootprint + memory_per_row * min_rows_per_batch
        memory_usable = psutil.virtual_memory().available * 0.9
        
        # If the available memory is very small, check if we can use more swap 
        if memory_usable < 1024*1024:
            memory_usable = min(psutil.swap_memory().free, 1024*1024)
        logger.info(f"memory_usable: {formatbytes(memory_usable)} with mem.available: {formatbytes(psutil.virtual_memory().available)} and swap.free: {formatbytes(psutil.swap_memory().free)}") 

        # If not enough memory for the amount of parallellism asked, reduce
        if (nb_parallel * memory_min_per_process) > memory_usable:
            nb_parallel = int(memory_usable/memory_min_per_process)
            logger.info(f"Nb_parallel reduced to {nb_parallel} to evade excessive memory usage")

        # Calculate the number of rows per batch
        layerinfo = getlayerinfo(input_path, input_layer)
        nb_rows_input_layer = layerinfo['featurecount']

        # Optimal number of batches and rows per batch 
        nb_batches = int((nb_rows_input_layer*memory_per_row*nb_parallel)/(memory_usable-memory_basefootprint*nb_parallel))
        if nb_batches < nb_parallel:
            nb_batches = nb_parallel
        
        rows_per_batch = int(nb_rows_input_layer/nb_batches)
        mem_predicted = (memory_basefootprint + rows_per_batch*memory_per_row)*nb_batches

        logger.info(f"nb_batches: {nb_batches}, rows_per_batch: {rows_per_batch} for nb_rows_input_layer: {nb_rows_input_layer} will result in mem_predicted: {formatbytes(mem_predicted)}")   

        with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

            # Prepare output filename
            _, output_filename = os.path.split(output_path) 
            output_filename_noext, output_ext = os.path.splitext(output_filename) 
            tmp_output_path = os.path.join(tempdir, output_filename)

            row_limit = rows_per_batch
            row_offset = 0
            jobs = {}    
            future_to_job_id = {}
            nb_todo = nb_batches
            nb_done = 0
            if verbose:
                logger.info(f"Start calculation on {nb_rows_input_layer} rows in {nb_batches} batches, so {row_limit} per batch")

            for job_id in range(nb_batches):

                jobs[job_id] = {}
                jobs[job_id]['layer'] = output_layer

                output_tmp_partial_path = os.path.join(tempdir, f"{output_filename_noext}_{job_id}{output_ext}")
                jobs[job_id]['tmp_partial_output_path'] = output_tmp_partial_path

                # For the last translate_id, take all rowid's left...
                if job_id < nb_batches-1:
                    rows = slice(row_offset, row_offset + row_limit)
                else:
                    rows = slice(row_offset, nb_rows_input_layer)
                jobs[job_id]['task_type'] = 'CALCULATE'
                # Remark: this temp file doesn't need spatial index
                future = calculate_pool.submit(
                        _buffer_gpd,
                        input_path=input_path,
                        output_path=output_tmp_partial_path,
                        buffer=buffer,
                        quadrantsegments=quadrantsegments,
                        input_layer=input_layer,        
                        output_layer=output_layer,
                        rows=rows,
                        verbose=verbose,
                        force=force)
                future_to_job_id[future] = job_id
                row_offset += row_limit
            
            # Loop till all parallel processes are ready, but process each one that is ready already
            for future in futures.as_completed(future_to_job_id):
                try:
                    result = future.result()

                    if result is not None and verbose is True:
                        logger.info(result)

                    # Start copy of the result to a common file
                    job_id = future_to_job_id[future]

                    # If the calculate gave results, copy to output
                    tmp_partial_output_path = jobs[job_id]['tmp_partial_output_path']
                    if os.path.exists(tmp_partial_output_path):

                        # TODO: append not yet supported in geopandas 0.7, but will be supported in next version
                        """
                        partial_output_gdf = geofile_util.read_file(tmp_partial_output_path)
                        geofile_util.to_file(partial_output_gdf, tmp_output_path, mode='a')
                        """
                        sqlite_stmt = None                  
                        translate_description = f"Copy result {job_id} of {nb_todo} to {output_layer}"
                        translate_info = ogr_util_direct.VectorTranslateInfo(
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
                        ogr_util_direct.vector_translate_by_info(info=translate_info)
                        os.remove(tmp_partial_output_path)
                    else:
                        logger.info(f"Result file {tmp_partial_output_path} was empty")

                except Exception as ex:
                    job_id = future_to_job_id[future]
                    #calculate_pool.shutdown()
                    logger.error(f"Error executing {jobs[job_id]}: {ex}")

                # Log the progress and prediction speed
                nb_done += 1
                report_progress(start_time, nb_done, nb_todo, operation)

        ##### Round up and clean up ##### 
        # Now create spatial index and move to output location
        create_spatial_index(path=tmp_output_path, layer=output_layer)
        shutil.move(tmp_output_path, output_path, copy_function=io_util.copyfile)

    finally:
        # Clean tmp dir
        #shutil.rmtree(tempdir)
        logger.info(f"{operation} ready, took {datetime.datetime.now()-start_time}!")

def _buffer_gpd(
        input_path: str,
        output_path: str,
        buffer: float,
        quadrantsegments: int = 5,
        input_layer: str = None,
        output_layer: str = None,
        rows = None,
        verbose: bool = False,
        force: bool = False) -> Optional[str]:
    
    ##### Init #####
    start_time = datetime.datetime.now()
    operation = 'buffer'
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path}")
            return None
        else:
            os.remove(output_path)

    data_gdf = geofile_util.read_file(filepath=input_path, layer=input_layer, rows=rows)
    if len(data_gdf) == 0:
        logger.info(f"No input geometries found for rows: {rows} in layer: {input_layer} in input_path: {input_path}")
        return None

    data_gdf.geometry = data_gdf.geometry.buffer(distance=buffer, resolution=quadrantsegments)

    if len(data_gdf) > 0:
        geofile_util.to_file(gdf=data_gdf, filepath=output_path, layer=output_layer)

    message = f"Took {datetime.datetime.now()-start_time} for {len(data_gdf)} rows ({rows})!"
    logger.info(message)

    return message

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
        # Calculating can be done in parallel, but only one process can write to 
        # the same file at the time... 
        if(nb_parallel == -1):
            nb_parallel = multiprocessing.cpu_count()
        nb_batches = nb_parallel*4
        with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

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
                        sqlite_stmt = None #f'SELECT * FROM "{output_layer}"'                   
                        translate_description = f"Copy result {translate_id} of {nb_batches} to {output_layer}"
                        
                        translate_info = ogr_util_direct.VectorTranslateInfo(
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
                        ogr_util_direct.vector_translate_by_info(info=translate_info)
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
        with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool: #, \
             #futures.ProcessPoolExecutor(1) as write_result_pool:

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

                        # If there wasn't an exception, but the output file doesn't exist, the result was empty, so just skip.
                        if not os.path.exists(tmp_partial_output_path):
                            continue
                        
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
        shutil.rmtree(tempdir)
        logger.info(f"Processing ready, took {datetime.datetime.now()-start_time}!")
    except Exception as ex:
        logger.exception(f"Processing ready with ERROR, took {datetime.datetime.now()-start_time}!")
   
def dissolve(
        input_path: Union[str, 'os.PathLike[Any]'],  
        output_path: Union[str, 'os.PathLike[Any]'],
        groupby_columns: List[str],
        aggfunc: str = None,
        explodecollections: bool = False,
        keep_cardsheets: bool = False,
        input_layer: str = None,        
        output_layer: str = None,
        input_cardsheets_path: Union[str, 'os.PathLike[Any]'] = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    """
    Function that applies a dissolve on the input file.

    Args:
        input_path (PathLike): path to the input file
        output_path (PathLike): path to the output file
        groupby_columns (List[str]): columns to group on
        aggfunc (str, optional): aggregation function to apply to columns not 
                grouped on. Defaults to None.
        explodecollections (bool, optional): after dissolving, evade having 
                multiparts in the output. Defaults to False.
        keep_cardsheets (bool, optional): if True, the result will only be 
                dissolved on the cardsheet level and not on the entire 
                dataset. Only available if no groupby_columns specified. 
                Defaults to False.
        input_layer (str, optional): input layername. If not specified, 
                there should be only one layer in the input file.
        output_layer (str, optional): output layername. If not specified, 
                then the filename is used as layer name.
        input_cardsheets_path (PathLike, optional): a file with the tiles/
                cardsheets to be used. If not specified, a tiling scheme 
                will be generated.
        nb_parallel (int, optional): number of parallel threads to use. If not
                specified, all available CPU's will be maximally used.
        verbose (bool, optional): output more detailed logging. Defaults to 
                False.
        force (bool, optional): overwrite result file if it exists already. 
                Defaults to False.
    """

    ##### Init #####
    input_path = str(input_path)
    output_path = str(output_path)
    operation = 'dissolve'
    start_time = datetime.datetime.now()
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path} and force is false")
            return
        else:
            os.remove(output_path)
    if nb_parallel == -1:
        nb_cpu = multiprocessing.cpu_count()
        nb_parallel = int(1.25 * nb_cpu)
        logger.debug(f"Nb cpus found: {nb_cpu}, nb_parallel: {nb_parallel}")

    # Get input data to temp gpkg file
    # TODO: still necessary to copy locally?
    tempdir = create_tempdir(operation)

    input_tmp_path = input_path
    '''
    input_tmp_path = os.path.join(tempdir, "input_layers.gpkg")
    _, input_ext = os.path.splitext(input_path)
    if(input_ext == '.gpkg'):
        logger.debug(f"Copy {input_path} to {input_tmp_path}")
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
    '''

    # Get the cardsheets we want the dissolve to be bound on to be able to parallelize
    if input_cardsheets_path is not None:
        input_cardsheets_path = str(input_cardsheets_path)
        cardsheets_gdf = geofile_util.read_file(input_cardsheets_path)
    else:
        # TODO: implement heuristic to choose a grid in a smart way
        cardsheets_gdf = None
        raise Exception("Not implemented!")

    try:
        # Start calculation in parallel
        logger.info(f"Start {operation} on file {input_tmp_path}")
        _, output_filename = os.path.split(output_path) 
        output_filename_noext, output_ext = os.path.splitext(output_filename)
        tmp_output_path = os.path.join(tempdir, output_filename)
        if output_layer is None:
            output_layer = get_default_layer(output_path)
        
        with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

            jobs = {}    
            future_to_job_id = {}    
            nb_todo = len(cardsheets_gdf)
            nb_done = 0
            for job_id, cardsheet in enumerate(cardsheets_gdf.itertuples()):
        
                jobs[job_id] = {}
                jobs[job_id]['layer'] = output_layer

                output_tmp_partial_path = os.path.join(tempdir, f"{output_filename_noext}_{job_id}{output_ext}")
                jobs[job_id]['tmp_partial_output_path'] = output_tmp_partial_path
                future = calculate_pool.submit(
                        _dissolve,
                        input_path=input_path,
                        output_path=output_tmp_partial_path,
                        groupby_columns=groupby_columns,
                        aggfunc=aggfunc,
                        explodecollections=explodecollections,
                        input_layer=input_layer,        
                        output_layer=output_layer,
                        bbox=cardsheet.geometry.bounds,
                        verbose=verbose,
                        force=force)
                future_to_job_id[future] = job_id
            
            # Loop till all parallel processes are ready, but process each one that is ready already
            for future in futures.as_completed(future_to_job_id):
                try:
                    _ = future.result()

                    # Start copy of the result to a common file
                    job_id = future_to_job_id[future]

                    # If the calculate gave results, copy to output
                    tmp_partial_output_path = jobs[job_id]['tmp_partial_output_path']
                    if os.path.exists(tmp_partial_output_path):

                        # TODO: append not yet supported in geopandas 0.7, but will be supported in next version
                        """
                        partial_output_gdf = geofile_util.read_file(tmp_partial_output_path)
                        geofile_util.to_file(partial_output_gdf, tmp_output_path, mode='a')
                        """
                        sqlite_stmt = None #f'SELECT * FROM "{output_layer}"'                   
                        translate_description = f"Copy result {job_id} of {nb_todo} to {output_layer}"
                        translate_info = ogr_util_direct.VectorTranslateInfo(
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
                        ogr_util_direct.vector_translate_by_info(info=translate_info)
                        os.remove(tmp_partial_output_path)

                except Exception as ex:
                    job_id = future_to_job_id[future]
                    #calculate_pool.shutdown()
                    logger.exception(f"Error executing {jobs[job_id]}: {ex}")

                # Log the progress and prediction speed
                nb_done += 1
                report_progress(start_time, nb_done, nb_todo, operation)

        # Now dissolve a second time to find elements on the border of the tiles that should 
        # still be dissolved
        if not keep_cardsheets:
            if groupby_columns is not None:
                logger.info("Now dissolve the entire file to get final result")
            
                _dissolve(
                        input_path=tmp_output_path,
                        output_path=output_path,
                        groupby_columns=groupby_columns,
                        aggfunc=aggfunc,
                        explodecollections=explodecollections,
                        input_layer=input_layer,        
                        output_layer=output_layer,
                        verbose=verbose,
                        force=force)
                # Now create spatial index
                create_spatial_index(path=tmp_output_path, layer=output_layer)
            else:
                logger.info("Now dissolve the elements on the borders as well to get final result")

                # First copy all elements that don't overlap with the borders of the tiles
                input_gdf = geofile_util.read_file(tmp_output_path)

                import shapely
                from shapely.geometry import MultiPolygon, Point

                cardsheets_lines = []
                for cardsheet_poly in cardsheets_gdf.itertuples():
                    cardsheet_boundary = cardsheet_poly.geometry.boundary
                    if cardsheet_boundary.type == 'MultiLineString':
                        for line in cardsheet_boundary:
                            cardsheets_lines.append(line)
                    else:
                        cardsheets_lines.append(cardsheet_boundary)
                
                cardsheets_lines_gdf = gpd.GeoDataFrame(geometry=cardsheets_lines)
                logger.info(f"Number of lines in cardsheets_lines_gdf: {len(cardsheets_lines_gdf)}")
                intersecting_gdf = gpd.sjoin(input_gdf, cardsheets_lines_gdf, op='intersects')
                logger.info(intersecting_gdf)
                geofile_util.to_file(intersecting_gdf, tmp_output_path + '_inters.gpkg')

        else:
            # Now create spatial index and move to output location
            create_spatial_index(path=tmp_output_path, layer=output_layer)
            shutil.move(tmp_output_path, output_path, copy_function=io_util.copyfile)

    finally:
        # Clean tmp dir
        #shutil.rmtree(tempdir)
        logger.info(f"{operation} ready, took {datetime.datetime.now()-start_time}!")

def _dissolve(
        input_path: str,
        output_path: str,
        groupby_columns: List[str] = None,
        aggfunc: str = None,
        explodecollections: bool = False,
        input_layer: str = None,        
        output_layer: str = None,
        bbox: Tuple[float, float, float, float] = None,
        verbose: bool = False,
        force: bool = False):

    ##### Init #####
    start_time = datetime.datetime.now()
    operation = 'dissolve'
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path}")
            return
        else:
            os.remove(output_path)

    # Read all records that are in the bbox
    retry_count = 0
    while True:
        try:
            input_gdf = geofile_util.read_file(filepath=input_path, layer=input_layer, bbox=bbox)
            if len(input_gdf) == 0:
                logger.info("No input geometries found")
                return 
            break
        except Exception as ex:
            if str(ex) == 'database is locked':
                if retry_count < 10:
                    retry_count += 1
                    time.sleep(1)
                else:
                    raise Exception("retried 10 times, database still locked") from ex
            else:
                raise ex

    # Now the real processing
    # If no groupby is filled out, perform unary_union 
    if groupby_columns is None:
        
        # unary union...
        union_geom = input_gdf['geometry'].unary_union
        # TODO: also support other geometry types (points and lines) 
        union_polygons = extract_polygons_from_list(union_geom)
        diss_gdf = gpd.GeoDataFrame(geometry=union_polygons)

        # Clip the result on the borders of the bbox not to have overlaps
        # between the different cards
        if bbox is not None:
            polygon = sh_geom.Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])])
            bbox_gdf = gpd.GeoDataFrame([1], geometry=[polygon])
            # keep_geom_type=True gives errors, so replace by own implementation
            diss_gdf = gpd.clip(diss_gdf, bbox_gdf)
            diss_gdf = extract_polygons_from_gdf(diss_gdf)
    else:
        
        # For a dissolve with a groupby it is important to only process every geometry once 
        # to evade duplicated rows. Because it is possible that a 
        # geometry intersects with multiple cardsheets, retain only geometries 
        # where the first point of the geometry is in the bbox.
        # Geometries that don't comply now will be treated in another tile.
        representative_point_gs = input_gdf.geometry.representative_point()
        input_gdf['representative_point_x'] = representative_point_gs.x
        input_gdf['representative_point_y'] = representative_point_gs.y
        input_gdf = input_gdf.loc[
                (input_gdf['representative_point_x'] >= bbox[0]) &
                (input_gdf['representative_point_y'] >= bbox[1]) &
                (input_gdf['representative_point_x'] < bbox[2]) &
                (input_gdf['representative_point_y'] < bbox[3])].copy() 
        input_gdf.drop(['representative_point_x', 'representative_point_y'], axis=1, inplace=True)

        diss_gdf = input_gdf.dissolve(by=groupby_columns, aggfunc=aggfunc)
        diss_gdf.geometry = [sh_geom.MultiPolygon([feature]) 
                                if type(feature) == sh_geom.Polygon 
                                else feature for feature in diss_gdf.geometry]

        if explodecollections:
            diss_gdf = diss_gdf.explode().reset_index()
            # TODO: reset_index???

    # TODO: Cleanup!
    nonpoly_gdf = diss_gdf.copy().reset_index()
    nonpoly_gdf['geomtype'] = nonpoly_gdf.geometry.geom_type
    nonpoly_gdf = nonpoly_gdf.loc[~nonpoly_gdf['geomtype'].isin(['Polygon', 'MultiPolygon'])]
    if len(nonpoly_gdf) > 0:
       raise Exception(f"3_Found {len(nonpoly_gdf)} non-(multi)polygons, eg.: {nonpoly_gdf}")

    if len(diss_gdf) > 0:
        geofile_util.to_file(gdf=diss_gdf, filepath=output_path, layer=output_layer)
    logger.info(f"{operation} ready, took {datetime.datetime.now()-start_time}!")

def unaryunion_cardsheets(
        input_path: Union[str, 'os.PathLike[Any]'],  
        output_path: Union[str, 'os.PathLike[Any]'],
        input_cardsheets_path: Union[str, 'os.PathLike[Any]'] = None,
        input_layer: str = None,        
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    """
    Function that applies a unaryunion on all geometries in the input a file
    and outputs the unary union clipped on the geographic tiles (card sheets)
    as specified by the input_cardsheets_path.

    Args:
        input_path (PathLike): path to the input file
        output_path (PathLike): path to the output file
        input_cardsheets_path (PathLike, optional): [description]
        input_layer (str, optional): [description]. Defaults to None.
        output_layer (str, optional): [description]. Defaults to None.
        nb_parallel (int, optional): [description]. Defaults to -1.
        verbose (bool, optional): [description]. Defaults to False.
        force (bool, optional): [description]. Defaults to False.
    """

    ##### Init #####
    input_path = str(input_path)
    input_cardsheets_path = str(input_cardsheets_path)
    output_path = str(output_path)
    operation = 'unaryunion_cardsheets'
    start_time = datetime.datetime.now()
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path} and force is false")
            return
        else:
            os.remove(output_path)
    if nb_parallel == -1:
        nb_cpu = multiprocessing.cpu_count()
        nb_parallel = int(1.25 * nb_cpu)
        logger.debug(f"Nb cpus found: {nb_cpu}, nb_parallel: {nb_parallel}")

    # Get input data to temp gpkg file
    tempdir = create_tempdir(operation)
    input_tmp_path = os.path.join(tempdir, "input_layers.gpkg")
    _, input_ext = os.path.splitext(input_path)
    if(input_ext == '.gpkg'):
        logger.debug(f"Copy {input_path} to {input_tmp_path}")
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

    # Load the cardsheets we want the unaryunion to be bound on
    cardsheets_gdf = geofile_util.read_file(input_cardsheets_path)

    try:
        # Start calculation in parallel
        logger.info(f"Start {operation} on file {input_tmp_path}")
        _, output_filename = os.path.split(output_path) 
        output_filename_noext, output_ext = os.path.splitext(output_filename)
        tmp_output_path = os.path.join(tempdir, output_filename)
        if output_layer is None:
            output_layer = get_default_layer(output_path)
        
        with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

            jobs = {}    
            future_to_job_id = {}    
            nb_todo = len(cardsheets_gdf)
            nb_done = 0
            for job_id, cardsheet in enumerate(cardsheets_gdf.itertuples()):
        
                jobs[job_id] = {}
                jobs[job_id]['layer'] = output_layer

                output_tmp_partial_path = os.path.join(tempdir, f"{output_filename_noext}_{job_id}{output_ext}")
                jobs[job_id]['tmp_partial_output_path'] = output_tmp_partial_path
                future = calculate_pool.submit(
                        _unaryunion,
                        input_path=input_path,
                        output_path=output_tmp_partial_path,
                        input_layer=input_layer,        
                        output_layer=output_layer,
                        bbox=cardsheet.geometry.bounds,
                        verbose=verbose,
                        force=force)
                future_to_job_id[future] = job_id
            
            # Loop till all parallel processes are ready, but process each one that is ready already
            for future in futures.as_completed(future_to_job_id):
                try:
                    _ = future.result()

                    # Start copy of the result to a common file
                    job_id = future_to_job_id[future]

                    # If the calculate gave results, copy to output
                    tmp_partial_output_path = jobs[job_id]['tmp_partial_output_path']
                    if os.path.exists(tmp_partial_output_path):

                        # TODO: append not yet supported in geopandas 0.7, but will be supported in next version
                        """
                        partial_output_gdf = geofile_util.read_file(tmp_partial_output_path)
                        geofile_util.to_file(partial_output_gdf, tmp_output_path, mode='a')
                        """
                        sqlite_stmt = None #f'SELECT * FROM "{output_layer}"'                   
                        translate_description = f"Copy result {job_id} of {nb_todo} to {output_layer}"
                        translate_info = ogr_util_direct.VectorTranslateInfo(
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
                        ogr_util_direct.vector_translate_by_info(info=translate_info)
                        os.remove(tmp_partial_output_path)

                except Exception as ex:
                    job_id = future_to_job_id[future]
                    #calculate_pool.shutdown()
                    logger.error(f"Error executing {jobs[job_id]}: {ex}")

                # Log the progress and prediction speed
                nb_done += 1
                report_progress(start_time, nb_done, nb_todo, operation)

        ##### Round up and clean up ##### 
        # Now create spatial index and move to output location
        create_spatial_index(path=tmp_output_path, layer=output_layer)
        shutil.move(tmp_output_path, output_path, copy_function=io_util.copyfile)

    finally:
        # Clean tmp dir
        #shutil.rmtree(tempdir)
        logger.info(f"{operation} ready, took {datetime.datetime.now()-start_time}!")

def report_progress(
        start_time: datetime.datetime,
        nb_done: int,
        nb_todo: int,
        operation: str):

    # 
    time_passed = (datetime.datetime.now()-start_time).total_seconds()
    if time_passed > 0 and nb_done > 0:
        processed_per_hour = (nb_done/time_passed) * 3600
        hours_to_go = (int)((nb_todo - nb_done)/processed_per_hour)
        min_to_go = (int)((((nb_todo - nb_done)/processed_per_hour)%1)*60)
        print(f"\r{hours_to_go:3d}:{min_to_go:2d} left to do {operation} on {(nb_todo-nb_done):6d} of {nb_todo}", 
              end="", flush=True)

def _unaryunion(
        input_path: str,
        output_path: str,
        groupby_columns: List[str] = None,
        explodecollections: bool = False,
        input_layer: str = None,        
        output_layer: str = None,
        bbox: Tuple[float, float, float, float] = None,
        verbose: bool = False,
        force: bool = False):

    ##### Init #####
    start_time = datetime.datetime.now()
    operation = 'dissolve'
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path}")
            return
        else:
            os.remove(output_path)

    input_gdf = geofile_util.read_file(filepath=input_path, layer=input_layer, bbox=bbox)
    if len(input_gdf) == 0:
        logger.info("No input geometries found")
        return 

    if groupby_columns is None:
        union_geom = input_gdf['geometry'].unary_union
        union_polygons = extract_polygons_from_list(union_geom)
        diss_gdf = gpd.GeoDataFrame(geometry=union_polygons)
    else:
        raise Exception("Not implemented")

    if explodecollections:
        diss_gdf = diss_gdf.explode()
    if bbox is not None:
        polygon = sh_geom.Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])])
        bbox_gdf = gpd.GeoDataFrame([1], geometry=[polygon])
        diss_gdf = gpd.clip(diss_gdf, bbox_gdf)

    if len(diss_gdf) > 0:
        geofile_util.to_file(gdf=diss_gdf, filepath=output_path, layer=output_layer)
    logger.info(f"{operation} ready, took {datetime.datetime.now()-start_time}!")
            
def extract_polygons_from_list(
        in_geom: list) -> list:
    """
    Extracts all polygons from the input geom and returns them as a list.
    """
    
    # Extract the polygons from the multipolygon
    geoms = []
    if in_geom.geom_type == 'MultiPolygon':
        geoms = list(in_geom)
    elif in_geom.geom_type == 'Polygon':
        geoms.append(in_geom)
    elif in_geom.geom_type == 'GeometryCollection':
        for geom in in_geom:
            if geom.geom_type in ('MultiPolygon', 'Polygon'):
                geoms.append(geom)
            else:
                logger.debug(f"Found {geom.geom_type}, ignore!")
    else:
        raise IOError(f"in_geom is of an unsupported type: {in_geom.geom_type}")
    
    return geoms

def extract_polygons_from_gdf(
        in_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    # Extract only polygons
    poly_gdf = in_gdf.loc[(in_gdf.geometry.geom_type == 'Polygon')].copy()
    multipoly_gdf = in_gdf.loc[(in_gdf.geometry.geom_type == 'MultiPolygon')].copy()
    collection_gdf = in_gdf.loc[(in_gdf.geometry.geom_type == 'GeometryCollection')].copy()
    collection_polys_gdf = None
    
    if len(collection_gdf) > 0:
        collection_polygons = []
        for collection_geom in collection_gdf.geometry:
            collection_polygons.extend(extract_polygons_from_list(collection_geom))
        if len(collection_polygons) > 0:
            collection_polys_gdf = gpd.GeoDataFrame(geometry=collection_polygons)

    # Only keep the polygons...
    ret_gdf = poly_gdf
    if len(multipoly_gdf) > 0:
        ret_gdf = ret_gdf.append(multipoly_gdf.explode(), ignore_index=True)
    if collection_polys_gdf is not None:
        ret_gdf = ret_gdf.append(collection_polys_gdf, ignore_index=True)
    
    return ret_gdf

def create_tempdir(base_dirname: str) -> str:
    #base_tempdir = os.path.join(tempfile.gettempdir(), base_dirname)
    base_tempdir = os.path.join(r"C:\temp", base_dirname)

    for i in range(1, 9999):
        try:
            tempdir = f"{base_tempdir}_{i:04d}"
            os.mkdir(tempdir)
            return tempdir
        except FileExistsError:
            continue

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
        
        '''
        # For the last batch, don't limit anymore
        if split_job_id >= nb_parts-1:
            row_limit = -1       

        sqlite_stmt = f"""
                SELECT {geometry_column_for_select}{columns_to_select_str} FROM \"{input_layer}\"
                 LIMIT {row_limit}
                OFFSET {row_offset}"""
        '''

        # For the last translate_id, take all rowid's left...
        if split_job_id < nb_parts-1:
            sqlite_stmt = f'''
                    SELECT {geometry_column_for_select}{columns_to_select_str}  
                        FROM "{input_layer}"
                        WHERE rowid >= {row_offset}
                        AND rowid < {row_offset + row_limit}'''
        else:
            sqlite_stmt = f'''
                    SELECT {geometry_column_for_select}{columns_to_select_str}  
                        FROM "{input_layer}"
                        WHERE rowid >= {row_offset}'''
                        
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
    
    #layers = getfileinfo(path)['layers']
    layers = fiona.listlayers(path)
    nb_layers = len(layers)
    if nb_layers == 1:
        return layers[0]
    elif nb_layers == 0:
        raise Exception(f"Error: No layers found in {path}")
    else:
        raise Exception(f"Error: More than 1 layer found in {path}: {layers}")

def get_default_layer(path: str):
    _, filename = os.path.split(path)
    layername, _ = os.path.splitext(filename)
    return layername

'''
OLD CODE

def dissolve_ogr(
        input_path: str,
        output_path: str,
        groupby_columns: List[str] = None,
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

def dissolve_cardsheets_ogr(
        input_path: str,
        input_cardsheets_path: str,
        output_path: str,
        groupby_columns: List[str] = None,
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
    if nb_parallel == -1:
        nb_parallel = multiprocessing.cpu_count()
        logger.info(f"Nb cpus found: {nb_parallel}")

    # Get input data to temp gpkg file
    tempdir = create_tempdir("dissolve_cardsheets")
    input_tmp_path = os.path.join(tempdir, f"input_layers.gpkg")
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

    if input_layer is None:
        input_layer = get_only_layer(input_tmp_path)
    if output_layer is None:
        output_layer = get_default_layer(output_path)

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
    cardsheets_gdf = geofile_util.read_file(input_cardsheets_path)

    try:
        # Start calculation of intersections in parallel
        logger.info(f"Start calculation of dissolves in file {input_tmp_path} to partial files")
        _, output_filename = os.path.split(output_path) 
        output_filename_noext, output_ext = os.path.splitext(output_filename)
        tmp_output_path = os.path.join(tempdir, output_filename)

        with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

            translate_jobs = {}    
            future_to_translate_id = {}    
            nb_batches = len(cardsheets_gdf)
            for translate_id, cardsheet in enumerate(cardsheets_gdf.itertuples()):
        
                translate_jobs[translate_id] = {}
                translate_jobs[translate_id]['layer'] = output_layer

                output_tmp_partial_path = os.path.join(tempdir, f"{output_filename_noext}_{translate_id}{output_ext}")
                translate_jobs[translate_id]['tmp_partial_output_path'] = output_tmp_partial_path

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

                translate_jobs[translate_id]['sqlite_stmt'] = sqlite_stmt
                translate_jobs[translate_id]['task_type'] = 'CALCULATE'
                translate_description = f"Async dissolve {translate_id} of {nb_batches}, bounds: {cardsheet.geometry.bounds}"
                # Remark: this temp file doesn't need spatial index
                translate_info = ogr_util.VectorTranslateInfo(
                        input_path=input_tmp_path,
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
                future = ogr_util.vector_translate_async(
                        concurrent_pool=calculate_pool, info=translate_info)
                future_to_translate_id[future] = translate_id
            
            # Loop till all parallel processes are ready, but process each one that is ready already
            for future in futures.as_completed(future_to_translate_id):
                try:
                    _ = future.result()

                    # Start copy of the result to a common file
                    # Remark: give higher priority, because this is the slowest factor
                    translate_id = future_to_translate_id[future]
                    if translate_jobs[translate_id]['task_type'] == 'CALCULATE':
                        # If the calculate gave results, copy to output
                        tmp_partial_output_path = translate_jobs[translate_id]['tmp_partial_output_path']
                        if os.path.exists(tmp_partial_output_path):
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
                            ogr_util.vector_translate_by_info(info=translate_info)
                            os.remove(tmp_partial_output_path)
                except Exception as ex:
                    translate_id = future_to_translate_id[future]
                    #calculate_pool.shutdown()
                    logger.error(f"Error executing {translate_jobs[translate_id]}: {ex}")

        ##### Round up and clean up ##### 
        # Now create spatial index and move to output location
        create_spatial_index(path=tmp_output_path, layer=output_layer)
        shutil.move(tmp_output_path, output_path, copy_function=io_util.copyfile)
    finally:
        # Clean tmp dir
        shutil.rmtree(tempdir)
        logger.info(f"Processing ready, took {datetime.datetime.now()-start_time}!")
'''

def formatbytes(bytes: float):
    """
    Return the given bytes as a human friendly KB, MB, GB, or TB string
    """

    bytes_float = float(bytes)
    KB = float(1024)
    MB = float(KB ** 2) # 1,048,576
    GB = float(KB ** 3) # 1,073,741,824
    TB = float(KB ** 4) # 1,099,511,627,776

    if bytes_float < KB:
        return '{0} {1}'.format(bytes_float,'Bytes' if 0 == bytes_float > 1 else 'Byte')
    elif KB <= bytes_float < MB:
        return '{0:.2f} KB'.format(bytes_float/KB)
    elif MB <= bytes_float < GB:
        return '{0:.2f} MB'.format(bytes_float/MB)
    elif GB <= bytes_float < TB:
        return '{0:.2f} GB'.format(bytes_float/GB)
    elif TB <= bytes_float:
        return '{0:.2f} TB'.format(bytes_float/TB)
      
if __name__ == '__main__':
    raise Exception("Not implemented!")
