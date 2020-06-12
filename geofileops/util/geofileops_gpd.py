from concurrent import futures
import datetime
import logging
import logging.config
import math
import multiprocessing
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple, Union

import geopandas as gpd
import shapely.geometry as sh_geom

from . import general_util
from geofileops import geofile
from . import io_util
from . import ogr_util

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
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def convexhull(
        input_path: Path,
        output_path: Path,
        input_layer: str = None,
        output_layer: str = None,
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
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def simplify(
        input_path: Path,
        output_path: Path,
        tolerance: float,
        input_layer: str = None,
        output_layer: str = None,
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
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def _apply_geooperation_to_layer(
        input_path: Path,
        output_path: Path,
        operation: str,
        operation_params: dict,
        input_layer: str = None,
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
                Defaults to None 
        input_layer (str, optional): [description]. Defaults to None.
        output_layer (str, optional): [description]. Defaults to None.
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
        nb_parallel, nb_batches, batch_size = general_util.get_parallellisation_params(
                nb_rows_total=nb_rows_total,
                nb_parallel=nb_parallel,
                verbose=verbose)

        with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

            # Prepare output filename
            tmp_output_path = tempdir / output_path.name

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
                        logger.info(result)

                    # Start copy of the result to a common file
                    batch_id = future_to_batch_id[future]

                    # If the calculate gave results, copy to output
                    tmp_partial_output_path = batches[batch_id]['tmp_partial_output_path']
                    if tmp_partial_output_path.exists():

                        # TODO: append not yet supported in geopandas 0.7, but will be supported in next version
                        """
                        partial_output_gdf = geofile.read_file(tmp_partial_output_path)
                        geofile.to_file(partial_output_gdf, tmp_output_path, mode='a')
                        """              
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
                                force_py=True,
                                verbose=verbose)
                        ogr_util.vector_translate_by_info(info=translate_info)
                        geofile.remove(tmp_partial_output_path)
                    else:
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
        geofile.create_spatial_index(path=tmp_output_path, layer=output_layer)
        geofile.move(tmp_output_path, output_path)

    finally:
        # Clean tmp dir
        #shutil.rmtree(tempdir)
        logger.info(f"{operation} ready, took {datetime.datetime.now()-start_time}!")

def _apply_geooperation(
        input_path: Path,
        output_path: Path,
        operation: str,
        operation_params: dict,
        input_layer: str = None,
        output_layer: str = None,
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
    data_gdf = geofile.read_file(path=input_path, layer=input_layer, rows=rows)
    if len(data_gdf) == 0:
        logger.info(f"No input geometries found for rows: {rows} in layer: {input_layer} in input_path: {input_path}")
        return None

    if operation == 'buffer':
        data_gdf.geometry = data_gdf.geometry.buffer(
                distance=operation_params['distance'], 
                resolution=operation_params['quadrantsegments'])
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
    logger.info(message)

    return message

def dissolve(
        input_path: Path,  
        output_path: Path,
        groupby_columns: Optional[List[str]],
        aggfunc: str = None,
        explodecollections: bool = False,
        keep_cardsheets: bool = False,
        input_layer: str = None,        
        output_layer: str = None,
        input_cardsheets_path: Path = None,
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
    operation = 'dissolve'
    start_time = datetime.datetime.now()
    if output_path.exists():
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path} and force is false")
            return
        else:
            geofile.remove(output_path)
    if nb_parallel == -1:
        nb_cpu = multiprocessing.cpu_count()
        nb_parallel = int(1.25 * nb_cpu)
        logger.debug(f"Nb cpus found: {nb_cpu}, nb_parallel: {nb_parallel}")

    # Get input data to temp gpkg file
    # TODO: still necessary to copy locally?
    tempdir = io_util.create_tempdir(operation)

    input_tmp_path = input_path

    # Get the cardsheets we want the dissolve to be bound on to be able to parallelize
    if input_cardsheets_path is not None:
        cardsheets_gdf = geofile.read_file(input_cardsheets_path)
    else:
        # TODO: implement heuristic to choose a grid in a smart way
        cardsheets_gdf = None
        raise Exception("Not implemented!")

    try:
        # Start calculation in parallel
        logger.info(f"Start {operation} on file {input_tmp_path}")
        tmp_output_path = tempdir / output_path.name
        if output_layer is None:
            output_layer = geofile.get_default_layer(output_path)
        
        with futures.ProcessPoolExecutor(nb_parallel) as calculate_pool:

            batches = {}    
            future_to_batch_id = {}    
            nb_todo = len(cardsheets_gdf)
            nb_done = 0
            for batch_id, cardsheet in enumerate(cardsheets_gdf.itertuples()):
        
                batches[batch_id] = {}
                batches[batch_id]['layer'] = output_layer
                output_tmp_partial_path = tempdir / f"{output_path.stem}_{batch_id}{output_path.suffix}"
                batches[batch_id]['tmp_partial_output_path'] = output_tmp_partial_path
                
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
                future_to_batch_id[future] = batch_id
            
            # Loop till all parallel processes are ready, but process each one that is ready already
            for future in futures.as_completed(future_to_batch_id):
                try:
                    _ = future.result()

                    # Start copy of the result to a common file
                    batch_id = future_to_batch_id[future]

                    # If the calculate gave results, copy to output
                    tmp_partial_output_path = batches[batch_id]['tmp_partial_output_path']
                    if tmp_partial_output_path.exists():

                        # TODO: append not yet supported in geopandas 0.7, but will be supported in next version
                        """
                        partial_output_gdf = geofile.read_file(tmp_partial_output_path)
                        geofile.to_file(partial_output_gdf, tmp_output_path, mode='a')
                        """                  
                        translate_description = f"Copy result {batch_id} of {nb_todo} to {output_layer}"
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
                                force_py=True,
                                verbose=verbose)
                        ogr_util.vector_translate_by_info(info=translate_info)
                        geofile.remove(tmp_partial_output_path)

                except Exception as ex:
                    batch_id = future_to_batch_id[future]
                    #calculate_pool.shutdown()
                    logger.exception(f"Error executing {batches[batch_id]}: {ex}")

                # Log the progress and prediction speed
                nb_done += 1
                general_util.report_progress(start_time, nb_done, nb_todo, operation)

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
                geofile.to_file(intersecting_gdf, str(tmp_output_path) + '_inters.gpkg')

        else:
            # Now create spatial index and move to output location
            geofile.create_spatial_index(path=tmp_output_path, layer=output_layer)
            geofile.move(tmp_output_path, output_path)

    finally:
        # Clean tmp dir
        #shutil.rmtree(tempdir)
        logger.info(f"{operation} ready, took {datetime.datetime.now()-start_time}!")

def _dissolve(
        input_path: Path,
        output_path: Path,
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
    if output_path.exists():
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path}")
            return
        else:
            geofile.remove(output_path)

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
        input_path: Path,  
        output_path: Path,
        input_cardsheets_path: Path = None,
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
    operation = 'unaryunion_cardsheets'
    start_time = datetime.datetime.now()
    if output_path.exists():
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path} and force is false")
            return
        else:
            geofile.remove(output_path)
    if nb_parallel == -1:
        nb_cpu = multiprocessing.cpu_count()
        nb_parallel = int(1.25 * nb_cpu)
        logger.debug(f"Nb cpus found: {nb_cpu}, nb_parallel: {nb_parallel}")

    # Get input data to temp gpkg file
    tempdir = io_util.create_tempdir(operation)
    input_tmp_path = tempdir / "input_layers.gpkg"
    if(input_path.suffix.lower() == '.gpkg'):
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
                force_py=True,
                verbose=verbose)
        logger.debug("Copy ready")

    # Load the cardsheets we want the unaryunion to be bound on
    if input_cardsheets_path is not None:
        cardsheets_gdf = geofile.read_file(input_cardsheets_path)
    else:
        raise Exception("Not implemented")

    try:
        # Start calculation in parallel
        logger.info(f"Start {operation} on file {input_tmp_path}")
        tmp_output_path = tempdir / output_path.name
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

                output_tmp_partial_path = tempdir / f"{output_path.stem}_{job_id}{output_path.suffix}"
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
                    if tmp_partial_output_path.exists():

                        # TODO: append not yet supported in geopandas 0.7, but will be supported in next version
                        """
                        partial_output_gdf = geofile.read_file(tmp_partial_output_path)
                        geofile.to_file(partial_output_gdf, tmp_output_path, mode='a')
                        """
                        translate_description = f"Copy result {job_id} of {nb_todo} to {output_layer}"
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
                                force_py=True,
                                verbose=verbose)
                        ogr_util.vector_translate_by_info(info=translate_info)
                        geofile.remove(tmp_partial_output_path)

                except Exception as ex:
                    job_id = future_to_job_id[future]
                    #calculate_pool.shutdown()
                    logger.error(f"Error executing {jobs[job_id]}: {ex}")

                # Log the progress and prediction speed
                nb_done += 1
                general_util.report_progress(start_time, nb_done, nb_todo, operation)

        ##### Round up and clean up ##### 
        # Now create spatial index and move to output location
        geofile.create_spatial_index(path=tmp_output_path, layer=output_layer)
        geofile.move(tmp_output_path, output_path)

    finally:
        # Clean tmp dir
        #shutil.rmtree(tempdir)
        logger.info(f"{operation} ready, took {datetime.datetime.now()-start_time}!")

def _unaryunion(
        input_path: Path,
        output_path: Path,
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
    if output_path.exists():
        if force is False:
            logger.info(f"Stop {operation}: output exists already {output_path}")
            return
        else:
            geofile.remove(output_path)

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

if __name__ == '__main__':
    raise Exception("Not implemented!")
