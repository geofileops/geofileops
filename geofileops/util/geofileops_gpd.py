# -*- coding: utf-8 -*-
"""
Module containing the implementation of Geofile operations using GeoPandas.
"""

from concurrent import futures
import datetime
import logging
import logging.config
import multiprocessing
from pathlib import Path
import time
import shutil
from typing import Any, List, Optional, Tuple, Union

import geopandas as gpd
import shapely.geometry as sh_geom

from geofileops import geofile
from geofileops.util import geofileops_ogr
from geofileops.util import ogr_util
from . import general_util
from . import io_util
from . import vector_util

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

def buffer(
        input_path: Path,
        output_path: Path,
        distance: float,
        quadrantsegments: int = 5,
        input_layer: str = None,
        output_layer: str = None,
        columns: List[str] = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    # Init
    operation = 'buffer'
    operation_params = {
            'distance': distance,
            'quadrantsegments': quadrantsegments
        }
    
    # Go!
    return _apply_geooperation_to_layer(
            input_path=input_path,
            output_path=output_path,
            operation=operation,
            operation_params=operation_params,
            input_layer=input_layer,
            output_layer=output_layer,
            columns=columns,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def convexhull(
        input_path: Path,
        output_path: Path,
        input_layer: str = None,
        output_layer: str = None,
        columns: List[str] = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    # Init
    operation = 'convexhull'
    operation_params = {}
    
    # Go!
    return _apply_geooperation_to_layer(
            input_path=input_path,
            output_path=output_path,
            operation=operation,
            operation_params=operation_params,
            input_layer=input_layer,
            output_layer=output_layer,
            columns=columns,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def simplify(
        input_path: Path,
        output_path: Path,
        tolerance: float,
        input_layer: str = None,
        output_layer: str = None,
        columns: List[str] = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    # Init
    operation = 'simplify'
    operation_params = {
            'tolerance': tolerance
        }
    
    # Go!
    return _apply_geooperation_to_layer(
            input_path=input_path,
            output_path=output_path,
            operation=operation,
            operation_params=operation_params,
            input_layer=input_layer,
            output_layer=output_layer,
            columns=columns,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def _apply_geooperation_to_layer(
        input_path: Path,
        output_path: Path,
        operation: str,
        operation_params: dict,
        input_layer: str = None,
        columns: List[str] = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    """
    Applies a geo operation on a layer.

    The operation to apply can be the following:
      - buffer: apply a buffer. Operation parameters:
          - distance: distance to buffer
          - quadrantsegments: number of points used to represent 1/4 of a circle
      - convexhull: appy a convex hull.
      - simplify: simplify the geometry using Douglas-Peukert algorythm. 
          Operation parameters:
          - tolerance: maximum distance to simplify.  

    Args:
        input_path (Path): [description]
        output_path (Path): [description]
        operation (str): the geo operation to apply.
        operation_params (dict, optional): the parameters for the geo operation. 
            Defaults to None. 
        input_layer (str, optional): [description]. Defaults to None.
        output_layer (str, optional): [description]. Defaults to None.
        columns (List[str], optional): If not None, only output the columns 
            specified. Defaults to None.
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
    tempdir = io_util.create_tempdir(operation.replace(' ', '_'))
    logger.info(f"Start calculation to temp files in {tempdir}")

    try:
        ##### Calculate #####
        # Remark: calculating can be done in parallel, but only one process 
        # can write to the same output file at the time...

        # Calculate the best number of parallel processes and batches for 
        # the available resources
        layerinfo = geofile.getlayerinfo(input_path, input_layer)
        nb_rows_total = layerinfo.featurecount
        #force_output_geometrytype = layerinfo.geometrytypename
        force_output_geometrytype = 'MULTIPOLYGON'
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
                        verbose=verbose,
                        force=force)
                future_to_batch_id[future] = batch_id
                row_offset += row_limit
            
            # Loop till all parallel processes are ready, but process each one that is ready already
            for future in futures.as_completed(future_to_batch_id):
                try:
                    result = future.result()

                    if result is not None and verbose is True:
                        logger.debug(result)

                    # Start copy of the result to a common file
                    batch_id = future_to_batch_id[future]

                    # If the calculate gave results, copy to output
                    tmp_partial_output_path = batches[batch_id]['tmp_partial_output_path']
                    if tmp_partial_output_path.exists():
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
                general_util.report_progress(start_time, nb_done, nb_batches, operation)

        ##### Round up and clean up ##### 
        # Now create spatial index and move to output location
        geofile.create_spatial_index(path=output_tmp_path, layer=output_layer)
        geofile.move(output_tmp_path, output_path)

    finally:
        # Clean tmp dir
        shutil.rmtree(tempdir)
        logger.info(f"{operation} ready, took {datetime.datetime.now()-start_time}!")

def _apply_geooperation(
        input_path: Path,
        output_path: Path,
        operation: str,
        operation_params: dict,
        input_layer: str = None,
        output_layer: str = None,
        columns: List[str] = None,
        rows = None,
        verbose: bool = False,
        force: bool = False) -> Optional[str]:
    
    ##### Init #####
    if output_path.exists():
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path}")
            return None
        else:
            geofile.remove(output_path)

    ##### Now go! #####
    start_time = datetime.datetime.now()
    data_gdf = geofile.read_file(path=input_path, layer=input_layer, columns=columns, rows=rows)
    if len(data_gdf) == 0:
        logger.info(f"No input geometries found for rows: {rows} in layer: {input_layer} in input_path: {input_path}")
        return None

    if operation == 'buffer':
        data_gdf.geometry = data_gdf.geometry.buffer(
                distance=operation_params['distance'], 
                resolution=operation_params['quadrantsegments'])
        #data_gdf['geometry'] = [sh_geom.Polygon(sh_geom.mapping(x)['coordinates']) for x in data_gdf.geometry]
    elif operation == 'convexhull':
        data_gdf.geometry = data_gdf.geometry.convex_hull
    elif operation == 'simplify':
        data_gdf.geometry = data_gdf.geometry.simplify(
                tolerance=operation_params['tolerance'])
    else:
        raise Exception(f"Operation not supported: {operation}")     

    if len(data_gdf) > 0:
        geofile.to_file(gdf=data_gdf, path=output_path, layer=output_layer, index=False)

    message = f"Took {datetime.datetime.now()-start_time} for {len(data_gdf)} rows ({rows})!"
    #logger.info(message)

    return message

def dissolve(
        input_path: Path,  
        output_path: Path,
        groupby_columns: Optional[List[str]] = None,
        aggfunc: str = 'first',
        explodecollections: bool = False,
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

    Args:
        input_path (Path): path to the input file
        output_path (Path): path to the output file
        groupby_columns (List[str]): columns to group on
        aggfunc (str, optional): aggregation function to apply to columns not 
                grouped on. Defaults to None.
        explodecollections (bool, optional): after dissolving, evade having 
                multiparts in the output. Defaults to False.
        tiles_path (PathLike, optional): a file with the tiles to be used. 
                If not specified, a tiling scheme will be generated.
        nb_squarish_tiles (int, optional): if tiles_path is not specified,
                the number of tiles the output should consist of. Default is 1.
        clip_on_tiles (bool, optional): if True, the result will only be 
                dissolved on the tile level and not on the entire 
                dataset. Only available if no groupby_columns specified. 
                Defaults to False.
        input_layer (str, optional): input layername. If not specified, 
                there should be only one layer in the input file.
        output_layer (str, optional): output layername. If not specified, 
                then the filename is used as layer name.        
        nb_parallel (int, optional): number of parallel threads to use. If not
                specified, all available CPU's will be maximally used.
        verbose (bool, optional): output more detailed logging. Defaults to 
                False.
        force (bool, optional): overwrite result file if it exists already. 
                Defaults to False.
    """

    ##### Init #####
    result_info = {}
    start_time = datetime.datetime.now()
    operation = 'dissolve'
    if output_path.exists():
        if force is False:
            result_info['message'] = f"Stop {operation}: output exists already {output_path} and force is false"
            logger.info(result_info['message'])
            return result_info
        else:
            geofile.remove(output_path)

    # Check input parameters
    if aggfunc != 'first':
        raise Exception(f"aggfunc != 'first' is not supported")
    if(groupby_columns is None and explodecollections == False):
        raise Exception(f"The combination of groupby_columns is None AND explodecollections == False is not supported")
    if clip_on_tiles is True and tiles_path is None and nb_squarish_tiles == 1:
        raise Exception(f"For clip_on_tiles to be True, tiles_path or nb_squarish_tiles needs to be specified")
    
    # If a tiles_path is specified, read those tiles... 
    result_tiles_gdf = None
    if tiles_path is not None:
        result_tiles_gdf = geofile.read_file(tiles_path)
        if nb_parallel == -1:
            nb_cpu = multiprocessing.cpu_count()
            nb_parallel = int(1.25 * nb_cpu)
            logger.debug(f"Nb cpus found: {nb_cpu}, nb_parallel: {nb_parallel}")
    else:
        # Else, create a grid based on the number of tiles wanted as result
        layerinfo = geofile.getlayerinfo(input_path, input_layer)
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
            layerinfo = geofile.getlayerinfo(pass_input_path, input_layer)
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
            
            result = dissolve_pass(
                    input_path=pass_input_path,
                    output_notonborder_path=output_tmp_path,
                    output_onborder_path=output_tmp_onborder_path,
                    tiles_gdf=tiles_gdf,
                    groupby_columns=groupby_columns,
                    aggfunc=aggfunc,
                    explodecollections=explodecollections,
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
            
        # Now move tmp file to output location, but order the rows randomly
        # to evade having all complex geometries together...   
        # Add column to use for random ordering
        geofile.add_column(path=output_tmp_path, layer=output_layer, 
                name='temp_ordering_id', type='REAL', expression=f"RANDOM()", force_update=True)
        sqlite_stmt = f'CREATE INDEX idx_batch_id ON "{output_layer}"(temp_ordering_id)' 
        ogr_util.vector_info(path=output_tmp_path, sql_stmt=sqlite_stmt, sql_dialect='SQLITE', readonly=False)

        # Get columns to keep
        layerinfo = geofile.getlayerinfo(output_tmp_path, output_layer)
        columns_str = ''
        for column in layerinfo.columns:
            if column.lower() not in ('fid', 'temp_ordering_id'):
                columns_str += f',"{column}"'
        # Now write to final output file
        sql_stmt = f'''
                SELECT {{geometrycolumn}} 
                    {columns_str} 
                FROM "{output_layer}" 
                ORDER BY temp_ordering_id'''
        geofileops_ogr.select(output_tmp_path, output_path, sql_stmt)

    finally:
        # Clean tmp dir if it exists...
        if tempdir.exists():
            shutil.rmtree(tempdir)

    # Return result info
    result_info['message'] = f"Dissolve completely ready, took {datetime.datetime.now()-start_time}!"
    logger.info(result_info['message'])
    return result_info

def dissolve_pass(
        input_path: Path,  
        output_notonborder_path: Path,
        output_onborder_path: Path,
        tiles_gdf: gpd.GeoDataFrame,
        groupby_columns: Optional[List[str]] = None,
        aggfunc: str = 'first',
        explodecollections: bool = False,
        clip_on_tiles: bool = False,
        input_layer: str = None,        
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False) -> dict:

    # Start calculation in parallel
    start_time = datetime.datetime.now()
    result_info = {}
    start_time = datetime.datetime.now()
    layerinfo = geofile.getlayerinfo(input_path, input_layer)
    nb_rows_total = layerinfo.featurecount
    
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
                    groupby_columns=groupby_columns,
                    aggfunc=aggfunc,
                    explodecollections=explodecollections,
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
        groupby_columns: Optional[List[str]] = None,
        aggfunc: str = 'first',
        explodecollections: bool = False,
        clip_on_tiles: bool = False,
        input_layer: str = None,        
        output_layer: str = None,
        bbox: Tuple[float, float, float, float] = None,
        verbose: bool = False) -> dict:

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
            input_gdf = geofile.read_file(path=input_path, layer=input_layer, bbox=bbox)
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
    if groupby_columns is None:
        # unary union...
        start_unary_union = datetime.datetime.now()
        try:
            union_geom = input_gdf.geometry.unary_union
        except Exception as ex:
            message = f"Exception processing bbox {bbox}"
            logger.exception(message)
            raise Exception(message) from ex

        # TODO: also support other geometry types (points and lines) 
        union_polygons = vector_util.extract_polygons_from_list(union_geom)
        diss_gdf = gpd.GeoDataFrame(geometry=union_polygons, crs=input_gdf.crs)
        perfinfo['time_unary_union'] = (datetime.datetime.now()-start_unary_union).total_seconds()

        # If we want to keep the tiles, clip the result on the borders of the 
        # bbox not to have overlaps between the different tiles.
        if clip_on_tiles is True and bbox is not None:
            start_clip = datetime.datetime.now()
            polygon = sh_geom.Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])])
            bbox_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=input_gdf.crs)
            # keep_geom_type=True gives errors, so replace by own implementation
            diss_gdf = gpd.clip(diss_gdf, bbox_gdf)
            diss_gdf = vector_util.extract_polygons_from_gdf(diss_gdf)
    
            perfinfo['time_clip'] = (datetime.datetime.now()-start_clip).total_seconds()
    else:
        # If groupby_columns specified, dissolve
        start_dissolve = datetime.datetime.now()
        diss_gdf = input_gdf.dissolve(by=groupby_columns, aggfunc=aggfunc)
        diss_gdf.geometry = [sh_geom.MultiPolygon([feature]) 
                                if type(feature) == sh_geom.Polygon 
                                else feature for feature in diss_gdf.geometry]

        if explodecollections:
            diss_gdf = diss_gdf.explode() #.reset_index()
            # TODO: reset_index necessary???
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
        geofile.to_file(diss_gdf, output_notonborder_path, append=True)
    else:
        # If not, save the polygons on the border seperately
        bbox_lines_gdf = vector_util.polygons_to_lines(
                gpd.GeoDataFrame(geometry=[sh_geom.box(bbox[0], bbox[1], bbox[2], bbox[3])], crs=input_gdf.crs))
        onborder_gdf = gpd.sjoin(diss_gdf, bbox_lines_gdf, op='intersects')
        if len(onborder_gdf) > 0:                
            geofile.to_file(onborder_gdf, output_onborder_path, append=True)
        
        notonborder_gdf = diss_gdf[~diss_gdf.index.isin(onborder_gdf.index)].dropna()
        if len(notonborder_gdf) > 0:
            geofile.to_file(notonborder_gdf, output_notonborder_path, append=True)
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
