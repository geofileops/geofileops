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
from geofileops.util import general_util
from geofileops.util.general_util import ParallelizationConfig
from geofileops.util import geofileops_ogr
from geofileops.util import geometry_util
from geofileops.util.geometry_util import GeometryType, PrimitiveType, SimplifyAlgorithm 
from geofileops.util import geoseries_util
from geofileops.util import grid_util
from geofileops.util import io_util
from geofileops.util import ogr_util

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
        algorithm: SimplifyAlgorithm = SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER,
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
            
            # Loop till all parallel processes are ready, but process each one 
            # that is ready already
            # REMARK: writing to temp file and than appending the result here 
            # is 10 time faster than appending directly using geopandas to_file 
            # (in geopandas 0.8)!
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
                                create_spatial_index=False)
                        geofile.remove(tmp_partial_output_path)
                    else:
                        if verbose:
                            logger.info(f"Result file {tmp_partial_output_path} was empty")
                    
                except Exception as ex:
                    batch_id = future_to_batch_id[future]
                    #calculate_pool.shutdown()
                    logger.exception(f"Error executing {batches[batch_id]}")

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
        data_gdf.geometry = geoseries_util.simplify_ext(
                data_gdf.geometry,
                algorithm=operation_params['algorithm'], 
                tolerance=operation_params['tolerance'], 
                lookahead=operation_params['step'])
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
        # Use force_multitype, to evade warnings when some batches contain 
        # singletype and some contain multitype geometries  
        geofile.to_file(
                gdf=data_gdf, path=output_path, layer=output_layer, index=False,
                force_multitype=True)

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
        input_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        parallelization_config: ParallelizationConfig = None,
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
        raise Exception(f"The combination of groupby_columns is None AND explodecollections == False is not possible")
    if not input_path.exists():
        raise Exception(f"input_path does not exist: {input_path}")
    layerinfo = geofile.get_layerinfo(input_path, input_layer)
    if layerinfo.geometrytype.to_primitivetype in [PrimitiveType.POINT, PrimitiveType.LINESTRING]:
        if tiles_path is not None or nb_squarish_tiles > 1:
            raise Exception(f"Dissolve to tiles (tiles_path, nb_squarish_tiles) is not supported for {layerinfo.geometrytype}")
    if output_path.exists():
        if force is False:
            result_info['message'] = f"Stop {operation}: output exists already {output_path} and force is false"
            logger.info(result_info['message'])
            return result_info
        else:
            geofile.remove(output_path)
    
    # Prepare columns to retain
    if columns is None:
        # If no columns specified, keep all columns
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
        result_tiles_gdf = grid_util.create_grid2(layerinfo.total_bounds, nb_squarish_tiles, layerinfo.crs)
        if len(result_tiles_gdf) > 1:
            geofile.to_file(result_tiles_gdf, output_path.parent / f"{output_path.stem}_tiles.gpkg")

    ##### Now start dissolving... #####
    # Line and point layers are:
    #   * not so large (memory-wise)
    #   * aren't computationally heavy
    # Additionally line layers are a pain to handle correctly because of 
    # rounding issues at the borders of tiles... so just dissolve them using 
    # geopandas.   
    if layerinfo.geometrytype.to_primitivetype in [PrimitiveType.POINT, PrimitiveType.LINESTRING]:
        input_gdf = geofile.read_file(input_path, input_layer)

        # Geopandas < 0.9 didn't support dissolving without groupby
        if groupby_columns is None or len(groupby_columns) == 0:
            union_geom = input_gdf.geometry.unary_union

            # Only keep geometries of primitive type of input 
            union_geom_cleaned = geometry_util.collection_extract(
                    union_geom, layerinfo.geometrytype.to_primitivetype)
            diss_gdf = gpd.GeoDataFrame(geometry=[union_geom_cleaned], crs=input_gdf.crs)
        else:
            diss_gdf = input_gdf.dissolve(by=groupby_columns, aggfunc=aggfunc, as_index=False)

        # Explodecollections if needed
        if explodecollections is True:
            # assert to evade pyLance warning
            assert isinstance(diss_gdf, gpd.GeoDataFrame)
            diss_gdf = diss_gdf.explode()
            # Reset the index, and drop the level_0 and level_1 multiindex
            diss_gdf.reset_index(drop=True, inplace=True)
        
        # Now write to file
        # assert to evade pyLance warning
        assert isinstance(diss_gdf, gpd.GeoDataFrame)
        geofile.to_file(diss_gdf, output_path, output_layer)

    elif layerinfo.geometrytype.to_primitivetype is PrimitiveType.POLYGON:

        # The dissolve for polygons is done in several passes, and after the first 
        # pass, only the 'onborder' features are further dissolved, as the 
        # 'notonborder' features are already OK.  
        tempdir = io_util.create_tempdir(operation)
        try:
            pass_input_path = input_path
            if output_layer is None:
                output_layer = geofile.get_default_layer(output_path)
            output_tmp_path = tempdir / f"{output_path.stem}.gpkg"
            prev_nb_batches = None
            last_pass = False
            pass_id = 0
            logger.info(f"Start dissolve on file {input_path}")
            while True:
                
                # Get some info of the file that needs to be dissolved
                layerinfo = geofile.get_layerinfo(pass_input_path, input_layer)
                nb_rows_total = layerinfo.featurecount
                
                # Calculate the best number of parallel processes and batches for 
                # the available resources
                nb_parallel, nb_batches_recommended, _ = general_util.get_parallelization_params(
                        nb_rows_total=nb_rows_total,
                        nb_parallel=nb_parallel,
                        prev_nb_batches=prev_nb_batches,
                        parallelization_config=parallelization_config,
                        verbose=verbose)

                # If the ideal number of batches is close to the nb. result tiles asked,  
                # dissolve towards the asked result!
                # If not, a temporary result is created using smaller tiles 
                if nb_batches_recommended <= len(result_tiles_gdf)*1.1:
                    tiles_gdf = result_tiles_gdf
                    last_pass = True
                elif len(result_tiles_gdf) == 1:
                    # Create a grid based on the ideal number of batches, but make 
                    # sure the number is smaller than the maximum... 
                    nb_squarish_tiles_max = None
                    if prev_nb_batches is not None:
                        nb_squarish_tiles_max = max(prev_nb_batches-1, 1)
                    tiles_gdf = grid_util.create_grid2(
                            total_bounds=layerinfo.total_bounds, 
                            nb_squarish_tiles=nb_batches_recommended, 
                            nb_squarish_tiles_max=nb_squarish_tiles_max, 
                            crs=layerinfo.crs)
                else:
                    # If a grid is specified already, add extra columns/rows instead of 
                    # creating new one...
                    tiles_gdf = grid_util.split_tiles(
                        result_tiles_gdf, nb_batches_recommended)
                geofile.to_file(tiles_gdf, tempdir / f"{output_path.stem}_{pass_id}_tiles.gpkg")

                # If the number of tiles ends up as 1, it is the last pass anyway...
                if len(tiles_gdf) == 1:
                    last_pass = True

                # If we are not in the last pass, onborder parcels will need extra 
                # processing still in further passes, so are saved in a seperate 
                # file. The notonborder rows are final immediately 
                if last_pass is not True:
                    output_tmp_onborder_path = tempdir / f"{output_path.stem}_{pass_id}_onborder.gpkg"
                else:
                    output_tmp_onborder_path = output_tmp_path
                
                # Now go!
                logger.info(f"Start dissolve pass {pass_id} to {len(tiles_gdf)} tiles (nb_parallel: {nb_parallel})")       
                result = _dissolve_polygons_pass(
                        input_path=pass_input_path,
                        output_notonborder_path=output_tmp_path,
                        output_onborder_path=output_tmp_onborder_path,
                        explodecollections=explodecollections,
                        groupby_columns=groupby_columns,
                        columns=columns_to_retain,
                        aggfunc=aggfunc,
                        tiles_gdf=tiles_gdf,
                        input_layer=input_layer,        
                        output_layer=output_layer,
                        nb_parallel=nb_parallel,
                        verbose=verbose,
                        force=force)

                # Prepare the next pass...
                # The input path are the onborder rows...
                prev_nb_batches = len(tiles_gdf)
                pass_input_path = output_tmp_onborder_path
                pass_id += 1

                # If we are ready...
                if last_pass is True:
                    break

            ##### Calculation ready! Now finalise output! #####
            # If there is a result on border, append it to the rest
            if(str(output_tmp_onborder_path) != str(output_tmp_path) 
            and output_tmp_onborder_path.exists()):
                geofile.append_to(output_tmp_onborder_path, output_tmp_path, dst_layer=output_layer)

            # If there is a result...
            if output_tmp_path.exists():
                # Prepare columns to keep 
                columns_str = ''
                for column in columns_to_retain:
                    columns_str += f', "{column}"'
                
                # Now move tmp file to output location, but order the rows randomly
                # to evade having all complex geometries together...   
                geofile.add_column(path=output_tmp_path, layer=output_layer, 
                        name='temp_ordering_id', type='REAL', expression=f"RANDOM()", force_update=True)
                sqlite_stmt = f'CREATE INDEX idx_batch_id ON "{output_layer}"(temp_ordering_id)' 
                ogr_util.vector_info(path=output_tmp_path, sql_stmt=sqlite_stmt, sql_dialect='SQLITE', readonly=False)

                # Prepare SQL statement for final output file 
                # If explodecollections is False and there are groupby columns...
                if(explodecollections is False 
                        and groupby_columns is not None 
                        and len(groupby_columns) > 0):
                    # All tiles are already dissolved to groups, but now the resuts 
                    # from all tiles still need to be grouped/collected together.  
                    groupby_columns_with_prefix = [f'"{column}"' for column in groupby_columns]
                    groupby_columns_str = ", ".join(groupby_columns_with_prefix)
                    sql_stmt = f'''
                            SELECT ST_Collect({{geometrycolumn}}) AS {{geometrycolumn}} 
                                {columns_str} 
                            FROM "{output_layer}" 
                            GROUP BY {groupby_columns_str}
                            ORDER BY avg(temp_ordering_id)'''
                else:
                    # No group by columns, so only need to reorder output file
                    sql_stmt = f'''
                            SELECT {{geometrycolumn}} 
                                {columns_str} 
                            FROM "{output_layer}" 
                            ORDER BY temp_ordering_id'''
                # Go!
                geofileops_ogr.select(
                        output_tmp_path, output_path, sql_stmt, output_layer=output_layer,
                        explodecollections=explodecollections)

        finally:
            # Clean tmp dir if it exists...
            if tempdir.exists():
                shutil.rmtree(tempdir)
    else:
        raise NotImplementedError(f"Unsupported input geometrytype: {layerinfo.geometrytype}")

    # Return result info
    result_info['message'] = f"Dissolve completely ready, took {datetime.datetime.now()-start_time}!"
    logger.info(result_info['message'])
    return result_info

def _dissolve_polygons_pass(
        input_path: Path,  
        output_notonborder_path: Path,
        output_onborder_path: Path,
        explodecollections: bool,
        groupby_columns: List[str],
        columns: List[str],
        aggfunc: str,
        tiles_gdf: gpd.GeoDataFrame,
        input_layer: Optional[str],        
        output_layer: Optional[str],
        nb_parallel: int,
        verbose: bool,
        force: bool) -> dict:

    # Start calculation in parallel
    start_time = datetime.datetime.now()
    result_info = {}
    start_time = datetime.datetime.now()
    input_layerinfo = geofile.get_layerinfo(input_path, input_layer)
    nb_rows_total = input_layerinfo.featurecount
    with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

        batches = {}    
        future_to_batch_id = {}    
        nb_rows_done = 0
        for batch_id, tile in enumerate(tiles_gdf.itertuples()):
    
            batches[batch_id] = {}
            batches[batch_id]['layer'] = output_layer
            
            future = calculate_pool.submit(
                    _dissolve_polygons,
                    input_path=input_path,
                    output_notonborder_path=output_notonborder_path,
                    output_onborder_path=output_onborder_path,
                    explodecollections=explodecollections,
                    groupby_columns=groupby_columns,
                    columns=columns,
                    aggfunc=aggfunc,
                    input_geometrytype=input_layerinfo.geometrytype,
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

def _dissolve_polygons(
        input_path: Path,
        output_notonborder_path: Path,
        output_onborder_path: Path,
        explodecollections: bool,
        groupby_columns: List[str],
        columns: List[str],
        aggfunc: str,
        input_geometrytype: GeometryType,
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

    # Now the real processing
    # If no groupby_columns specified, perform unary_union
    # TODO: from geopandas 0.9, dissolve without groupby_columns is supported
    if groupby_columns is None or len(groupby_columns) == 0:
        # unary union...
        start_unary_union = datetime.datetime.now()
        try:
            union_geom = input_gdf.geometry.unary_union
        except Exception as ex:
            message = f"Exception processing bbox {bbox}"
            logger.exception(message)
            raise Exception(message) from ex

        # Only keep geometries of primitive type of input 
        union_geom_cleaned = geometry_util.collection_extract(
                union_geom, input_geometrytype.to_primitivetype)
        diss_gdf = gpd.GeoDataFrame(geometry=[union_geom_cleaned], crs=input_gdf.crs)
        perfinfo['time_unary_union'] = (datetime.datetime.now()-start_unary_union).total_seconds()
    else:
        # If groupby_columns specified, dissolve
        start_dissolve = datetime.datetime.now()
        diss_gdf = input_gdf.dissolve(by=groupby_columns, aggfunc=aggfunc, as_index=False)
        perfinfo['time_dissolve'] = (datetime.datetime.now()-start_dissolve).total_seconds()

    # If explodecollections is True and For polygons, explode multi-geometries.
    # If needed they will be 'collected' afterwards to multipolygons again.
    if(explodecollections is True 
        or input_geometrytype in [GeometryType.POLYGON, GeometryType.MULTIPOLYGON]):
        # assert to evade pyLance warning 
        assert isinstance(diss_gdf, gpd.GeoDataFrame)
        diss_gdf = diss_gdf.explode()
        # Reset the index, and drop the level_0 and level_1 multiindex
        diss_gdf.reset_index(drop=True, inplace=True)

    # Clip the result on the borders of the bbox not to have overlaps 
    # between the different tiles. 
    # If this is not applied, this results in some geometries not being merged 
    # or in duplicates.
    # REMARK: for (multi)linestrings, the endpoints created by the clip are not 
    # always the same due to rounding, so dissolving in a next pass doesn't 
    # always result in linestrings being re-connected... Because dissolving 
    # lines isn't so computationally heavy anyway, drop support here.    
    if bbox is not None:
        start_clip = datetime.datetime.now()
        polygon = sh_geom.Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])])
        bbox_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=input_gdf.crs)
        # keep_geom_type=True gave errors, but seems fixed since geopandas 0.8.2
        diss_gdf = gpd.clip(diss_gdf, bbox_gdf, keep_geom_type=True)

        """
        # Only keep geometries of the primitive type specified after clip...
        # assert to evade pyLance warning 
        assert isinstance(diss_gdf, gpd.GeoDataFrame)
        diss_gdf.geometry = geoseries_util.geometry_collection_extract(
                diss_gdf.geometry, input_geometrytype.to_primitivetype)
        """

        perfinfo['time_clip'] = (datetime.datetime.now()-start_clip).total_seconds()

    # Drop rows with None/empty geometries 
    diss_gdf = diss_gdf[~diss_gdf.geometry.isna()]
    diss_gdf = diss_gdf[~diss_gdf.geometry.is_empty]

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
        # Use force_multitype, to evade warnings when some batches contain 
        # singletype and some contain multitype geometries  
        geofile.to_file(
                diss_gdf, output_notonborder_path, layer=output_layer, 
                append=True, force_multitype=True)
    else:
        # If not, save the polygons on the border seperately
        bbox_lines_gdf = gpd.GeoDataFrame(
                geometry=geoseries_util.polygons_to_lines(
                        gpd.GeoSeries([sh_geom.box(bbox[0], bbox[1], bbox[2], bbox[3])])), 
                crs=input_gdf.crs)
        onborder_gdf = gpd.sjoin(diss_gdf, bbox_lines_gdf, op='intersects')
        onborder_gdf.drop('index_right', axis=1, inplace=True)
        if len(onborder_gdf) > 0:
            # assert to evade pyLance warning 
            assert isinstance(onborder_gdf, gpd.GeoDataFrame)
            # Use force_multitype, to evade warnings when some batches contain 
            # singletype and some contain multitype geometries  
            geofile.to_file(
                    onborder_gdf, output_onborder_path, layer=output_layer, 
                    append=True, force_multitype=True)
        
        notonborder_gdf = diss_gdf[~diss_gdf.index.isin(onborder_gdf.index)]
        if len(notonborder_gdf) > 0:
            # assert to evade pyLance warning 
            assert isinstance(notonborder_gdf, gpd.GeoDataFrame) 
            # Use force_multitype, to evade warnings when some batches contain 
            # singletype and some contain multitype geometries  
            geofile.to_file(
                    notonborder_gdf, output_notonborder_path, layer=output_layer, 
                    append=True, force_multitype=True)
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
