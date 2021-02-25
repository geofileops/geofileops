# -*- coding: utf-8 -*-
"""
Module containing the implementation of Geofile operations using GeoPandas.
"""

from concurrent import futures
import datetime
import enum
import logging
import logging.config
import multiprocessing
from pathlib import Path
import time
import shutil
from typing import List, Optional, Tuple

import geopandas as gpd
import shapely.geometry as sh_geom

from geofileops import geofile
from geofileops.util import geofileops_ogr
from geofileops.util import ogr_util
from . import general_util
from . import io_util
from . import vector_util
from .vector_util import GeometryType 

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

class GeoOperation(enum.Enum):
    SIMPLIFY = 'simplify'
    BUFFER = 'buffer'
    CONVEXHULL = 'convexhull'

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
    # Init
    operation_params = {
            'distance': distance,
            'quadrantsegments': quadrantsegments
        }
    
    # Buffer operation always results in polygons...
    # TODO: test is some buffers don't result in geometrycollection, line, point,...
    force_output_geometrytype = GeometryType.MULTIPOLYGON

    # Go!
    return _apply_geooperation_to_layer(
            input_path=input_path,
            output_path=output_path,
            operation=GeoOperation.BUFFER,
            operation_params=operation_params,
            input_layer=input_layer,
            output_layer=output_layer,
            columns=columns,
            explodecollections=explodecollections,
            force_output_geometrytype=force_output_geometrytype,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def convexhull(
        input_path: Path,
        output_path: Path,
        input_layer: str = None,
        output_layer: str = None,
        columns: List[str] = None,
        explodecollections: bool = False,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    # Init
    operation_params = {}
    
    # Go!
    return _apply_geooperation_to_layer(
            input_path=input_path,
            output_path=output_path,
            operation=GeoOperation.CONVEXHULL,
            operation_params=operation_params,
            input_layer=input_layer,
            output_layer=output_layer,
            columns=columns,
            nb_parallel=nb_parallel,
            explodecollections=explodecollections,
            verbose=verbose,
            force=force)

def simplify(
        input_path: Path,
        output_path: Path,
        tolerance: float,
        algorithm: vector_util.SimplifyAlgorithm = vector_util.SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER,
        lookahead: int = 8,
        input_layer: str = None,
        output_layer: str = None,
        columns: List[str] = None,
        explodecollections: bool = False,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    # Init
    operation_params = {
            'tolerance': tolerance,
            'algorithm': algorithm,
            'step': lookahead
        }
    
    # Go!
    return _apply_geooperation_to_layer(
            input_path=input_path,
            output_path=output_path,
            operation=GeoOperation.SIMPLIFY,
            operation_params=operation_params,
            input_layer=input_layer,
            output_layer=output_layer,
            columns=columns,
            explodecollections=explodecollections,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def _apply_geooperation_to_layer(
        input_path: Path,
        output_path: Path,
        operation: GeoOperation,
        operation_params: dict,
        input_layer: str = None,
        columns: List[str] = None,
        output_layer: str = None,
        explodecollections: bool = False,
        force_output_geometrytype: GeometryType = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    """
    Applies a geo operation on a layer.

    The operation to apply can be one of the the following:
      - BUFFER: apply a buffer. Operation parameters:
          - distance: distance to buffer
          - quadrantsegments: number of points used to represent 1/4 of a circle
      - CONVEXHULL: appy a convex hull.
      - SIMPLIFY: simplify the geometry. Operation parameters:
          - algorithm: vector_util.SimplifyAlgorithm
          - tolerance: maximum distance to simplify.
          - lookahead: for LANG, the number of points to forward-look

    Args:
        input_path (Path): [description]
        output_path (Path): [description]
        operation (GeoOperation): the geo operation to apply.
        operation_params (dict, optional): the parameters for the geo operation. 
            Defaults to None. 
        input_layer (str, optional): [description]. Defaults to None.
        output_layer (str, optional): [description]. Defaults to None.
        columns (List[str], optional): If not None, only output the columns 
            specified. Defaults to None.
        explodecollections (bool, optional): True to convert all multi-geometries to 
            singular ones during the geooperation. Defaults to False.
        force_output_geometrytype (GeometryType, optional): Geometrytype to 
            force output to. Defaults to None.
        nb_parallel (int, optional): [description]. Defaults to -1.
        verbose (bool, optional): [description]. Defaults to False.
        force (bool, optional): [description]. Defaults to False.
    """
    ##### Init #####
    start_time = datetime.datetime.now()
    if input_layer is None:
        input_layer = geofile.get_only_layer(input_path)
    if output_path.exists():
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path}")
            return
        else:
            geofile.remove(output_path)
    if input_layer is None:
        input_layer = geofile.get_only_layer(input_path)
    if output_layer is None:
        output_layer = geofile.get_default_layer(output_path)

    ##### Prepare tmp files #####
    tempdir = io_util.create_tempdir(operation.value)
    logger.info(f"Start calculation to temp files in {tempdir}")

    try:
        ##### Calculate #####
        # Remark: calculating can be done in parallel, but only one process 
        # can write to the same output file at the time...

        # Calculate the best number of parallel processes and batches for 
        # the available resources
        layerinfo = geofile.get_layerinfo(input_path, input_layer)
        nb_rows_total = layerinfo.featurecount
        if(nb_parallel == -1):
            nb_parallel = multiprocessing.cpu_count()
        nb_batches = nb_parallel
        batch_size = int(nb_rows_total/nb_batches)
        #nb_parallel, nb_batches, batch_size = general_util.get_parallellisation_params(
        #        nb_rows_total=nb_rows_total,
        #        nb_parallel=nb_parallel,
        #        verbose=verbose)

        with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

            # Prepare output filename
            output_tmp_path = tempdir / output_path.name

            row_limit = batch_size
            row_offset = 0
            batches = {}    
            future_to_batch_id = {}
            nb_done = 0
            if verbose:
                logger.info(f"Start calculation on {nb_rows_total} rows in {nb_batches} batches, so {row_limit} per batch")

            for batch_id in range(nb_batches):

                batches[batch_id] = {}
                batches[batch_id]['layer'] = output_layer

                output_tmp_partial_path = tempdir / f"{output_path.stem}_{batch_id}{output_path.suffix}"
                batches[batch_id]['tmp_partial_output_path'] = output_tmp_partial_path

                # For the last translate_id, take all rowid's left...
                if batch_id < nb_batches-1:
                    rows = slice(row_offset, row_offset + row_limit)
                else:
                    rows = slice(row_offset, nb_rows_total)

                # Remark: this temp file doesn't need spatial index
                future = calculate_pool.submit(
                        _apply_geooperation,
                        input_path=input_path,
                        output_path=output_tmp_partial_path,
                        operation=operation,
                        operation_params=operation_params,
                        input_layer=input_layer,
                        columns=columns,     
                        output_layer=output_layer,
                        rows=rows,
                        explodecollections=explodecollections,
                        verbose=verbose,
                        force=force)
                future_to_batch_id[future] = batch_id
                row_offset += row_limit
            
            # Loop till all parallel processes are ready, but process each one that is ready already
            for future in futures.as_completed(future_to_batch_id):
                try:
                    result = future.result()

                    if result is not None and verbose is True:
                        logger.info(result)

                    # Start copy of the result to a common file
                    batch_id = future_to_batch_id[future]

                    # If the calculate gave results, copy to output
                    tmp_partial_output_path = batches[batch_id]['tmp_partial_output_path']
                    if tmp_partial_output_path.exists() and tmp_partial_output_path.stat().st_size > 0:
                        geofile.append_to(
                                src=tmp_partial_output_path, 
                                dst=output_tmp_path, 
                                force_output_geometrytype=force_output_geometrytype,
                                create_spatial_index=False)
                        geofile.remove(tmp_partial_output_path)
                    else:
                        if verbose:
                            logger.info(f"Result file {tmp_partial_output_path} was empty")

                except Exception as ex:
                    batch_id = future_to_batch_id[future]
                    #calculate_pool.shutdown()
                    logger.error(f"Error executing {batches[batch_id]}: {ex}")

                # Log the progress and prediction speed
                nb_done += 1
                general_util.report_progress(start_time, nb_done, nb_batches, operation.value)

        ##### Round up and clean up ##### 
        # Now create spatial index and move to output location
        if output_tmp_path.exists():
            geofile.create_spatial_index(path=output_tmp_path, layer=output_layer)
            geofile.move(output_tmp_path, output_path)
        else:
            logger.warning(f"Result of {operation} was empty!f")

    finally:
        # Clean tmp dir
        shutil.rmtree(tempdir)
        logger.info(f"{operation} ready, took {datetime.datetime.now()-start_time}!")

def _apply_geooperation(
        input_path: Path,
        output_path: Path,
        operation: GeoOperation,
        operation_params: dict,
        input_layer: str = None,
        output_layer: str = None,
        columns: List[str] = None,
        rows = None,
        explodecollections: bool = False,
        verbose: bool = False,
        force: bool = False) -> str:
    
    ##### Init #####
    if output_path.exists():
        if force is False:
            message = f"Stop {operation}: output exists already {output_path}"
            return message
        else:
            geofile.remove(output_path)

    ##### Now go! #####
    start_time = datetime.datetime.now()
    data_gdf = geofile.read_file(path=input_path, layer=input_layer, columns=columns, rows=rows)
    if len(data_gdf) == 0:
        message = f"No input geometries found for rows: {rows} in layer: {input_layer} in input_path: {input_path}"
        return message

    if operation is GeoOperation.BUFFER:
        data_gdf.geometry = data_gdf.geometry.buffer(
                distance=operation_params['distance'], 
                resolution=operation_params['quadrantsegments'])
        #data_gdf['geometry'] = [sh_geom.Polygon(sh_geom.mapping(x)['coordinates']) for x in data_gdf.geometry]
    elif operation is GeoOperation.CONVEXHULL:
        data_gdf.geometry = data_gdf.geometry.convex_hull
    elif operation is GeoOperation.SIMPLIFY:
        # If ramer-douglas-peucker, use standard geopandas algorithm
        if operation_params['algorithm'] is vector_util.SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER:
            data_gdf.geometry = data_gdf.geometry.simplify(
                    tolerance=operation_params['tolerance'])
        else:
            # For other algorithms, use vector_util.simplify_ext()
            data_gdf.geometry = data_gdf.geometry.apply(
                    lambda geom: vector_util.simplify_ext(
                            geom, algorithm=operation_params['algorithm'], 
                            tolerance=operation_params['tolerance'], 
                            lookahead=operation_params['step']))
    else:
        raise Exception(f"Operation not supported: {operation}")     

    # Remove rows where geom is empty
    data_gdf = data_gdf[~data_gdf.is_empty] 
    data_gdf = data_gdf[~data_gdf.isna()] 
    
    if explodecollections:
        data_gdf = data_gdf.explode().reset_index(drop=True)

    if len(data_gdf) > 0:
        # assert to evade pyLance warning
        assert isinstance(data_gdf, gpd.GeoDataFrame)
        geofile.to_file(gdf=data_gdf, path=output_path, layer=output_layer, index=False)

    message = f"Took {datetime.datetime.now()-start_time} for {len(data_gdf)} rows ({rows})!"
    return message

def dissolve(
        input_path: Path,  
        output_path: Path,
        groupby_columns: List[str] = [],
        columns: Optional[List[str]] = [],
        aggfunc: str = 'first',
        explodecollections: bool = True,
        tiles_path: Path = None,
        nb_squarish_tiles: int = 1,
        clip_on_tiles: bool = False,
        input_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False) -> dict:
    """
    Function that applies a dissolve on the input file.

    More detailed documentation in module geofileops!
    """

    ##### Init #####
    start_time = datetime.datetime.now()
    operation = 'dissolve'
    result_info = {}
    
    # Check input parameters
    if aggfunc != 'first':
        raise NotImplementedError(f"aggfunc != 'first' is not implemented")
    if groupby_columns is None and explodecollections == False:
        raise NotImplementedError(f"The combination of groupby_columns is None AND explodecollections == False is not implemented")
    if clip_on_tiles is True and tiles_path is None and nb_squarish_tiles == 1:
        raise Exception(f"For clip_on_tiles to be True, tiles_path or nb_squarish_tiles needs to be specified")
    if not input_path.exists():
        raise Exception(f"input_path does not exist: {input_path}")
    if output_path.exists():
        if force is False:
            result_info['message'] = f"Stop {operation}: output exists already {output_path} and force is false"
            logger.info(result_info['message'])
            return result_info
        else:
            geofile.remove(output_path)
    
    # Prepare columns to retain
    layerinfo = None
    if columns is None:
        # If no columns specified, keep all columns
        layerinfo = geofile.get_layerinfo(input_path, input_layer)
        columns_to_retain = layerinfo.columns
    else:
        # If columns specified, add groupby_columns if they are not specified
        columns_to_retain = columns.copy()
        for column in groupby_columns:
            if column not in columns_to_retain:
                columns_to_retain.append(column)

    # If a tiles_path is specified, read those tiles... 
    result_tiles_gdf = None
    if tiles_path is not None:
        result_tiles_gdf = geofile.read_file(tiles_path)
        if nb_parallel == -1:
            nb_cpu = multiprocessing.cpu_count()
            nb_parallel = nb_cpu #int(1.25 * nb_cpu)
            logger.debug(f"Nb cpus found: {nb_cpu}, nb_parallel: {nb_parallel}")
    else:
        # Else, create a grid based on the number of tiles wanted as result
        layerinfo = geofile.get_layerinfo(input_path, input_layer)
        result_tiles_gdf = vector_util.create_grid2(layerinfo.total_bounds, nb_squarish_tiles, layerinfo.crs)
        if len(result_tiles_gdf) > 1:
            geofile.to_file(result_tiles_gdf, output_path.parent / f"{output_path.stem}_tiles.gpkg")

    ##### Now start dissolving... #####
    # The dissolve is done in several passes, and after the first pass, only the 
    # 'onborder' features are further dissolved, as the 'notonborder' features 
    # are already OK.  
    pass_input_path = input_path
    if output_layer is None:
        output_layer = geofile.get_default_layer(output_path)
    tempdir = io_util.create_tempdir(operation)
    output_tmp_path = tempdir / f"{output_path.stem}.gpkg"
    prev_nb_batches = None
    last_pass = False
    pass_id = 0
    current_clip_on_tiles = False
    logger.info(f"Start dissolve on file {input_path}")
    start_time = datetime.datetime.now()

    try:
        while True:
            
            # Get some info of the file that needs to be dissolved
            layerinfo = geofile.get_layerinfo(pass_input_path, input_layer)
            nb_rows_total = layerinfo.featurecount
            
            # Calculate the best number of parallel processes and batches for 
            # the available resources
            nb_parallel, nb_batches_recommended, _ = general_util.get_parallellisation_params(
                    nb_rows_total=nb_rows_total,
                    nb_parallel=nb_parallel,
                    prev_nb_batches=prev_nb_batches,
                    verbose=verbose)

            # If the ideal number of batches is close to the nb. result tiles asked,  
            # dissolve towards the asked result!
            # If not, a temporary result is created using smaller tiles 
            if nb_batches_recommended <= len(result_tiles_gdf)*1.1:
                tiles_gdf = result_tiles_gdf
                current_clip_on_tiles = clip_on_tiles
                last_pass = True
            elif len(result_tiles_gdf) == 1:
                # Create a grid based on the ideal number of batches
                tiles_gdf = vector_util.create_grid2(layerinfo.total_bounds, nb_batches_recommended, layerinfo.crs)
            else:
                # If a grid is specified already, add extra columns/rows instead of 
                # creating new one...
                tiles_gdf = vector_util.split_tiles(
                    result_tiles_gdf, nb_batches_recommended)
            geofile.to_file(tiles_gdf, tempdir / f"{output_path.stem}_{pass_id}_tiles.gpkg")

            # The notonborder rows are final immediately
            # The onborder parcels will need extra processing still... 
            output_tmp_onborder_path = tempdir / f"{output_path.stem}_{pass_id}_onborder.gpkg"
            
            result = _dissolve_pass(
                    input_path=pass_input_path,
                    output_notonborder_path=output_tmp_path,
                    output_onborder_path=output_tmp_onborder_path,
                    explodecollections=explodecollections,
                    groupby_columns=groupby_columns,
                    columns=columns_to_retain,
                    aggfunc=aggfunc,
                    tiles_gdf=tiles_gdf,
                    clip_on_tiles=current_clip_on_tiles,
                    input_layer=input_layer,        
                    output_layer=output_layer,
                    nb_parallel=nb_parallel,
                    verbose=verbose,
                    force=force)

            # If we are ready...
            if last_pass is True:
                break
            
            # Prepare the next pass...
            # The input path are the onborder rows...
            prev_nb_batches = len(tiles_gdf)
            pass_input_path = output_tmp_onborder_path
            pass_id += 1

        ##### Calculation ready! Now finalise output! #####
        # If there is a result on border, append it to the rest
        if output_tmp_onborder_path.exists():
            geofile.append_to(output_tmp_onborder_path, output_tmp_path, dst_layer=output_layer)

        # If there is a result...
        if output_tmp_path.exists():
            # Now move tmp file to output location, but order the rows randomly
            # to evade having all complex geometries together...   
            # Add column to use for random ordering
            geofile.add_column(path=output_tmp_path, layer=output_layer, 
                    name='temp_ordering_id', type='REAL', expression=f"RANDOM()", force_update=True)
            sqlite_stmt = f'CREATE INDEX idx_batch_id ON "{output_layer}"(temp_ordering_id)' 
            ogr_util.vector_info(path=output_tmp_path, sql_stmt=sqlite_stmt, sql_dialect='SQLITE', readonly=False)

            # Get columns to keep and write final stuff
            columns_str = ''
            for column in columns_to_retain:
                columns_str += f',"{column}"'
            sql_stmt = f'''
                    SELECT {{geometrycolumn}} 
                        {columns_str} 
                    FROM "{output_layer}" 
                    ORDER BY temp_ordering_id'''
            geofileops_ogr.select(output_tmp_path, output_path, sql_stmt, output_layer=output_layer)

    finally:
        # Clean tmp dir if it exists...
        if tempdir.exists():
            shutil.rmtree(tempdir)

    # Return result info
    result_info['message'] = f"Dissolve completely ready, took {datetime.datetime.now()-start_time}!"
    logger.info(result_info['message'])
    return result_info

def _dissolve_pass(
        input_path: Path,  
        output_notonborder_path: Path,
        output_onborder_path: Path,
        explodecollections: bool,
        groupby_columns: List[str],
        columns: List[str],
        aggfunc: str,
        tiles_gdf: gpd.GeoDataFrame,
        clip_on_tiles: bool,
        input_layer: Optional[str],        
        output_layer: Optional[str],
        nb_parallel: int,
        verbose: bool,
        force: bool) -> dict:

    # Start calculation in parallel
    start_time = datetime.datetime.now()
    result_info = {}
    start_time = datetime.datetime.now()
    layerinfo = geofile.get_layerinfo(input_path, input_layer)
    nb_rows_total = layerinfo.featurecount
    force_output_geometrytype = geofile.to_multigeometrytype(layerinfo.geometrytype)

    logger.info(f"Start dissolve pass to {len(tiles_gdf)} tiles (nb_parallel: {nb_parallel})")       
    with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

        batches = {}    
        future_to_batch_id = {}    
        nb_rows_done = 0
        for batch_id, tile in enumerate(tiles_gdf.itertuples()):
    
            batches[batch_id] = {}
            batches[batch_id]['layer'] = output_layer
            
            future = calculate_pool.submit(
                    _dissolve,
                    input_path=input_path,
                    output_notonborder_path=output_notonborder_path,
                    output_onborder_path=output_onborder_path,
                    explodecollections=explodecollections,
                    groupby_columns=groupby_columns,
                    columns=columns,
                    aggfunc=aggfunc,
                    force_output_geometrytype=force_output_geometrytype,
                    clip_on_tiles=clip_on_tiles,
                    input_layer=input_layer,        
                    output_layer=output_layer,
                    bbox=tile.geometry.bounds,
                    verbose=verbose)
            future_to_batch_id[future] = batch_id
        
        # Loop till all parallel processes are ready, but process each one that is ready already
        for future in futures.as_completed(future_to_batch_id):
            try:
                # If the calculate gave results
                batch_id = future_to_batch_id[future]
                result = future.result()
                if result is not None:
                    nb_rows_done += result['nb_rows_done']
                    if verbose and result['nb_rows_done'] > 0 and result['total_time'] > 0:
                        rows_per_sec = round(result['nb_rows_done']/result['total_time'])
                        logger.info(f"Batch {batch_id} ready, processed {result['nb_rows_done']} rows in {rows_per_sec} rows/sec")
                        if 'perfstring' in result:
                            logger.info(f"Perfstring: {result['perfstring']}")
            except Exception as ex:
                batch_id = future_to_batch_id[future]
                message = f"Error executing {batches[batch_id]}: {ex}"
                logger.exception(message)
                calculate_pool.shutdown()
                raise Exception(message) from ex

            # Log the progress and prediction speed
            general_util.report_progress(start_time, nb_rows_done, nb_rows_total, 'dissolve')

    logger.info(f"Dissolve pass ready, took {datetime.datetime.now()-start_time}!")
                
    return result_info

def _dissolve(
        input_path: Path,
        output_notonborder_path: Path,
        output_onborder_path: Path,
        explodecollections: bool,
        groupby_columns: List[str],
        columns: List[str],
        aggfunc: str,
        force_output_geometrytype: GeometryType,
        clip_on_tiles: bool,
        input_layer: Optional[str],        
        output_layer: Optional[str],
        bbox: Tuple[float, float, float, float],
        verbose: bool) -> dict:

    ##### Init #####
    perfinfo = {}
    start_time = datetime.datetime.now()
    return_info = {"input_path": input_path,
                   "output_notonborder_path": output_notonborder_path,
                   "output_onborder_path": output_onborder_path,
                   "bbox": bbox,
                   "nb_rows_done": 0,
                   "total_time": 0,
                   "perfinfo": ""
                }

    # Read all records that are in the bbox
    retry_count = 0
    start_read = datetime.datetime.now()
    while True:
        try:
            input_gdf = geofile.read_file(
                    path=input_path, layer=input_layer, bbox=bbox, columns=columns)
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

    # Check result
    perfinfo['time_read'] = (datetime.datetime.now()-start_read).total_seconds()
    return_info['nb_rows_done'] = len(input_gdf)
    if return_info['nb_rows_done'] == 0:
        message = f"No input geometries found in {input_path}"
        logger.info(message)
        return_info['message'] = message
        return_info['total_time'] = (datetime.datetime.now()-start_time).total_seconds()
        return return_info

    # If the tiles don't need to be kept,
    # evade geometries from being processed in multiple tiles.    
    if clip_on_tiles is False:

        # Geometries can be on the border of tiles, so can intersect with 
        # multiple tiles. 
        # For a dissolve with a groupby this would result in duplicated rows.  
        # When no groupby_columns are specified but keep_tiles is False it 
        # is inefficient to treat these border-geometries multiple times.
        # So in these cases, retain only geometries where the first point of 
        # the geometry is in the bbox. Geometries that don't comply now will be 
        # treated in another tile.
        start_filter = datetime.datetime.now()
        representative_point_gs = input_gdf.geometry.representative_point()
        input_gdf['representative_point_x'] = representative_point_gs.x
        input_gdf['representative_point_y'] = representative_point_gs.y
        input_gdf = input_gdf.loc[
                (input_gdf['representative_point_x'] >= bbox[0]) &
                (input_gdf['representative_point_y'] >= bbox[1]) &
                (input_gdf['representative_point_x'] < bbox[2]) &
                (input_gdf['representative_point_y'] < bbox[3])].copy() 
        input_gdf.drop(['representative_point_x', 'representative_point_y'], axis=1, inplace=True)

        # Nb. rows that will be processed is reduced...
        return_info['nb_rows_done'] = len(input_gdf)
        perfinfo['time_filter_repr_point'] = (datetime.datetime.now()-start_filter).total_seconds()
            
    # Now the real processing
    # If no groupby_columns specified, perform unary_union
    if groupby_columns is None or len(groupby_columns) == 0:
        # unary union...
        start_unary_union = datetime.datetime.now()
        try:
            union_geom = input_gdf.geometry.unary_union
        except Exception as ex:
            message = f"Exception processing bbox {bbox}"
            logger.exception(message)
            raise Exception(message) from ex

        # TODO: also support other geometry types (points and lines)
        union_geom_cleaned = vector_util.collection_extract(
                union_geom, vector_util.to_primitivetype(force_output_geometrytype))
        diss_gdf = gpd.GeoDataFrame(geometry=[union_geom_cleaned], crs=input_gdf.crs)
        perfinfo['time_unary_union'] = (datetime.datetime.now()-start_unary_union).total_seconds()

        # For polygons, explode multi-geometries ...
        if(explodecollections is True 
           or force_output_geometrytype in [GeometryType.POLYGON, GeometryType.MULTIPOLYGON]):
            diss_gdf = diss_gdf.explode()
            # Reset the index, and drop the level_0 and lavel_1 multiindex
            diss_gdf.reset_index(drop=True, inplace=True)

        # If we want to keep the tiles, clip the result on the borders of the 
        # bbox not to have overlaps between the different tiles.
        if clip_on_tiles is True and bbox is not None:
            start_clip = datetime.datetime.now()
            polygon = sh_geom.Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])])
            bbox_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=input_gdf.crs)
            # keep_geom_type=True gives errors, so replace by own implementation
            diss_gdf = gpd.clip(diss_gdf, bbox_gdf)

            # Only keep geometries of the primitive type specified...
            
            # assert to evade pyLance warning 
            #assert isinstance(diss_gdf, gpd.GeoDataFrame)
            
            diss_gdf.geometry = diss_gdf.geometry.apply(
                    lambda geom: vector_util.collection_extract(
                            geom, vector_util.to_primitivetype(force_output_geometrytype)))
    
            perfinfo['time_clip'] = (datetime.datetime.now()-start_clip).total_seconds()
    else:
        # If groupby_columns specified, dissolve
        start_dissolve = datetime.datetime.now()
        diss_gdf = input_gdf.dissolve(by=groupby_columns, aggfunc=aggfunc, as_index=False)
        
        # assert to evade pyLance werning 
        assert isinstance(diss_gdf, gpd.GeoDataFrame)
        # Explode multi-geometries if asked...
        if explodecollections:
            diss_gdf = diss_gdf.explode()
            # Reset the index, and drop the level_0 and lavel_1 multiindex
            diss_gdf.reset_index(drop=True, inplace=True)
                
        perfinfo['time_dissolve'] = (datetime.datetime.now()-start_dissolve).total_seconds()

    # If there is no result, return
    if len(diss_gdf) == 0:
        message = f"Result is empty for {input_path}"
        return_info['message'] = message
        return_info['perfinfo'] = perfinfo
        return_info['total_time'] = (datetime.datetime.now()-start_time).total_seconds()
        return return_info

    # Save the result to destination file(s)
    start_to_file = datetime.datetime.now()

    # If the tiles don't need to be merged afterwards, we can just save the result as it is
    if str(output_notonborder_path) == str(output_onborder_path):
        # assert to evade pyLance warning 
        assert isinstance(diss_gdf, gpd.GeoDataFrame)
        geofile.to_file(diss_gdf, output_notonborder_path, layer=output_layer,
                force_output_geometrytype=force_output_geometrytype, append=True)
    else:
        # If not, save the polygons on the border seperately
        bbox_lines_gdf = vector_util.polygons_to_lines(
                gpd.GeoDataFrame(geometry=[sh_geom.box(bbox[0], bbox[1], bbox[2], bbox[3])], crs=input_gdf.crs))
        onborder_gdf = gpd.sjoin(diss_gdf, bbox_lines_gdf, op='intersects')
        onborder_gdf.drop('index_right', axis=1, inplace=True)
        if len(onborder_gdf) > 0:
            # assert to evade pyLance warning 
            assert isinstance(onborder_gdf, gpd.GeoDataFrame) 
            geofile.to_file(onborder_gdf, output_onborder_path, layer=output_layer,
                    force_output_geometrytype=force_output_geometrytype, append=True)
        
        notonborder_gdf = diss_gdf[~diss_gdf.index.isin(onborder_gdf.index)].dropna()
        if len(notonborder_gdf) > 0:
            # assert to evade pyLance warning 
            assert isinstance(notonborder_gdf, gpd.GeoDataFrame) 
            geofile.to_file(notonborder_gdf, output_notonborder_path, layer=output_layer,
                    force_output_geometrytype=force_output_geometrytype, append=True)
    perfinfo['time_to_file'] = (datetime.datetime.now()-start_to_file).total_seconds()

    # Finalise...
    message = f"dissolve ready in {datetime.datetime.now()-start_time} on {input_path}!"
    if verbose:
        logger.info(message)
    
    # Collect perfinfo
    total_perf_time = 0
    perfstring = ""
    for perfcode in perfinfo:
        total_perf_time += perfinfo[perfcode]
        perfstring += f"{perfcode}: {perfinfo[perfcode]:.2f}, "
    return_info['total_time'] = (datetime.datetime.now()-start_time).total_seconds()
    perfinfo['unaccounted'] = return_info['total_time'] - total_perf_time
    perfstring += f"unaccounted: {perfinfo['unaccounted']:.2f}"
    
    # Return
    return_info['perfinfo'] = perfinfo
    return_info['perfstring'] = perfstring
    return_info['message'] = message
    return return_info

if __name__ == '__main__':
    raise Exception("Not implemented!")
