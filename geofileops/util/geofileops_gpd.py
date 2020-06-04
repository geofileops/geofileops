from concurrent import futures
import datetime
import logging
import logging.config
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple, Union

import geopandas as gpd
import psutil
import shapely.geometry as sh_geom

from geofileops import geofile
from . import ogr_util_direct

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

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

    operation = "buffer"
    if input_layer is None:
        input_layer = geofile.get_only_layer(input_path)

    ##### Init #####
    start_time = datetime.datetime.now()
    if os.path.exists(output_path):
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path}")
            return
        else:
            os.remove(output_path)

    if input_layer is None:
        input_layer = geofile.get_only_layer(input_path)
    if output_layer is None:
        output_layer = geofile.get_default_layer(output_path)

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
        layerinfo = geofile.getlayerinfo(input_path, input_layer)
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
                        partial_output_gdf = geofile.read_file(tmp_partial_output_path)
                        geofile.to_file(partial_output_gdf, tmp_output_path, mode='a')
                        """              
                        translate_description = f"Copy result {job_id} of {nb_todo} to {output_layer}"
                        translate_info = ogr_util_direct.VectorTranslateInfo(
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
        geofile.create_spatial_index(path=tmp_output_path, layer=output_layer)
        geofile.move(tmp_output_path, output_path)

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

    data_gdf = geofile.read_file(path=input_path, layer=input_layer, rows=rows)
    if len(data_gdf) == 0:
        logger.info(f"No input geometries found for rows: {rows} in layer: {input_layer} in input_path: {input_path}")
        return None

    data_gdf.geometry = data_gdf.geometry.buffer(distance=buffer, resolution=quadrantsegments)

    if len(data_gdf) > 0:
        geofile.to_file(gdf=data_gdf, path=output_path, layer=output_layer)

    message = f"Took {datetime.datetime.now()-start_time} for {len(data_gdf)} rows ({rows})!"
    logger.info(message)

    return message

def dissolve(
        input_path: Union[str, 'os.PathLike[Any]'],  
        output_path: Union[str, 'os.PathLike[Any]'],
        groupby_columns: Optional[List[str]],
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
    '''

    # Get the cardsheets we want the dissolve to be bound on to be able to parallelize
    if input_cardsheets_path is not None:
        input_cardsheets_path = str(input_cardsheets_path)
        cardsheets_gdf = geofile.read_file(input_cardsheets_path)
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
            output_layer = geofile.get_default_layer(output_path)
        
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
                        partial_output_gdf = geofile.read_file(tmp_partial_output_path)
                        geofile.to_file(partial_output_gdf, tmp_output_path, mode='a')
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
                geofile.create_spatial_index(path=tmp_output_path, layer=output_layer)
            else:
                logger.info("Now dissolve the elements on the borders as well to get final result")

                # First copy all elements that don't overlap with the borders of the tiles
                input_gdf = geofile.read_file(tmp_output_path)

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
                geofile.to_file(intersecting_gdf, tmp_output_path + '_inters.gpkg')

        else:
            # Now create spatial index and move to output location
            geofile.create_spatial_index(path=tmp_output_path, layer=output_layer)
            geofile.move(tmp_output_path, output_path)

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
            input_gdf = geofile.read_file(path=input_path, layer=input_layer, bbox=bbox)
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
        geofile.to_file(gdf=diss_gdf, path=output_path, layer=output_layer)
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
        geofile.copy(input_path, input_tmp_path)
        logger.debug("Copy ready")
    else:
        # Remark: this temp file doesn't need spatial index
        logger.info(f"Copy {input_path} to {input_tmp_path} using ogr2ogr")
        ogr_util_direct.vector_translate(
                input_path=input_path,
                output_path=input_tmp_path,
                create_spatial_index=False,
                output_layer=input_layer,
                verbose=verbose)
        logger.debug("Copy ready")

    # Load the cardsheets we want the unaryunion to be bound on
    cardsheets_gdf = geofile.read_file(input_cardsheets_path)

    try:
        # Start calculation in parallel
        logger.info(f"Start {operation} on file {input_tmp_path}")
        _, output_filename = os.path.split(output_path) 
        output_filename_noext, output_ext = os.path.splitext(output_filename)
        tmp_output_path = os.path.join(tempdir, output_filename)
        if output_layer is None:
            output_layer = geofile.get_default_layer(output_path)
        
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
                        partial_output_gdf = geofile.read_file(tmp_partial_output_path)
                        geofile.to_file(partial_output_gdf, tmp_output_path, mode='a')
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
        geofile.create_spatial_index(path=tmp_output_path, layer=output_layer)
        geofile.move(tmp_output_path, output_path)

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

    input_gdf = geofile.read_file(path=input_path, layer=input_layer, bbox=bbox)
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
        geofile.to_file(gdf=diss_gdf, path=output_path, layer=output_layer)
    logger.info(f"{operation} ready, took {datetime.datetime.now()-start_time}!")
            
def extract_polygons_from_list(
        in_geom: sh_geom.base.BaseGeometry) -> list:
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
