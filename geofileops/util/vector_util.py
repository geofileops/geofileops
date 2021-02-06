# -*- coding: utf-8 -*-
"""
Module containing utilities regarding low level vector operations.
"""

import enum
import logging
import math
from typing import Any, Collection, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pyproj
import simplification.cutil as simpl
import shapely.geometry as sh_geom
import shapely.ops as sh_ops

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# Grid tile helpers
#-------------------------------------------------------------

def create_grid(
        total_bounds: Tuple[float, float, float, float],
        nb_columns: int,
        nb_rows: int,
        crs: Union[pyproj.CRS, str, None]) -> gpd.GeoDataFrame:

    xmin, ymin, xmax, ymax = total_bounds
    width = (xmax-xmin)/nb_columns
    height = (ymax-ymin)/nb_rows

    return create_grid3(total_bounds=total_bounds, width=width, height=height, crs=crs)

def create_grid3(
        total_bounds: Tuple[float, float, float, float],
        width: float,
        height: float,
        crs: Union[pyproj.CRS, str, None]) -> gpd.GeoDataFrame:

    xmin, ymin, xmax, ymax = total_bounds
    rows = int(math.ceil((ymax-ymin) / height))
    cols = int(math.ceil((xmax-xmin) / width))
     
    polygons = []
    cell_left = xmin
    cell_right = xmin + width
    for _ in range(cols):
        if cell_left > xmax:
            break
        cell_top = ymin + height
        cell_bottom = ymin
        for _ in range(rows):
            if cell_bottom > ymax:
                break
            polygons.append(sh_ops.Polygon([(cell_left, cell_top), (cell_right, cell_top), (cell_right, cell_bottom), (cell_left, cell_bottom)])) 
            cell_top += height
            cell_bottom += height
            
        cell_left += width
        cell_right += width
        
    return gpd.GeoDataFrame({'geometry': polygons}, crs=crs)

def create_grid2(
        total_bounds: Tuple[float, float, float, float], 
        nb_squarish_tiles: int,
        crs: Union[pyproj.CRS, str, None]) -> gpd.GeoDataFrame:
    """
    Creates a grid and tries to approximate the number of cells asked as
    good as possible with grid cells that as close to square as possible.

    Args:
        total_bounds (Tuple[float, float, float, float]): bounds of the grid to be created
        nb_squarish_cells (int): about the number of cells wanted

    Returns:
        gpd.GeoDataFrame: geodataframe with the grid
    """
    # If more cells asked, calculate optimal number
    xmin, ymin, xmax, ymax = total_bounds
    total_width = xmax-xmin
    total_height = ymax-ymin

    columns_vs_rows = total_width/total_height
    nb_rows = round(math.sqrt(nb_squarish_tiles/columns_vs_rows))

    # Evade having too many cells if few cells are asked...
    if nb_rows > nb_squarish_tiles:
        nb_rows = nb_squarish_tiles
    nb_columns = round(nb_squarish_tiles/nb_rows)
    
    # Now we know everything to create the grid
    return create_grid(
        total_bounds=total_bounds,
        nb_columns=nb_columns,
        nb_rows=nb_rows,
        crs=crs)

def split_tiles(
        input_tiles: gpd.GeoDataFrame,
        nb_tiles_wanted: int) -> gpd.GeoDataFrame:

    nb_tiles = len(input_tiles)
    if nb_tiles >= nb_tiles_wanted:
        return input_tiles
    
    nb_tiles_ratio_target = nb_tiles_wanted / nb_tiles

    # Loop over all tiles in the grid
    result_tiles = []
    for tile in input_tiles.itertuples():

        # For this tile, as long as the curr_nb_tiles_ratio_todo is not 1, keep splitting 
        curr_nb_tiles_ratio_todo = nb_tiles_ratio_target
        curr_tiles_being_split = [tile.geometry]
        while curr_nb_tiles_ratio_todo > 1:

            # Check in how many parts the tiles are split in this iteration
            divisor = 0 
            if round(curr_nb_tiles_ratio_todo) == 3:
                divisor = 3
            else:
                divisor = 2
            curr_nb_tiles_ratio_todo /= divisor

            # Split all current tiles
            tmp_tiles_after_split = []
            for tile_to_split in curr_tiles_being_split:
                xmin, ymin, xmax, ymax = tile_to_split.bounds
                width = abs(xmax-xmin)
                height = abs(ymax-ymin)

                # Split in 2 or 3...
                if divisor == 3:
                    if width > height:
                        split_line = sh_geom.LineString([
                                (xmin+width/3, ymin-1), (xmin+width/3, ymax+1),
                                (xmin+2*width/3, ymax+1), (xmin+2*width/3, ymin-1)])
                    else:
                        split_line = sh_geom.LineString([
                                (xmin-1, ymin+height/3), (xmax+1, ymin+height/3),
                                (xmax+1, ymin+2*height/3), (xmin-1, ymin+2*height/3)])
                else:
                    if width > height:
                        split_line = sh_geom.LineString([(xmin+width/2, ymin-1), (xmin+width/2, ymax+1)])
                    else:
                        split_line = sh_geom.LineString([(xmin-1, ymin+height/2), (xmax+1, ymin+height/2)])
                tmp_tiles_after_split.extend(sh_ops.split(tile_to_split, split_line))
            curr_tiles_being_split = tmp_tiles_after_split
        result_tiles.extend(curr_tiles_being_split)
    
    # We should be ready...
    return gpd.GeoDataFrame(geometry=result_tiles, crs=input_tiles.crs)

#-------------------------------------------------------------
# General helpers
#-------------------------------------------------------------

def extract_polygons_from_list(
        in_geom: sh_geom.base.BaseGeometry) -> List[Any]:
    """
    Extracts all polygons from the input geom and returns them as a list.
    """
    # Extract the polygons from the multipolygon, but store them as multipolygons anyway
    geoms = []
    if in_geom.geom_type == 'MultiPolygon':
        geoms = list(sh_geom.MultiPolygon(in_geom))
    elif in_geom.geom_type == 'Polygon':
        geoms.append(in_geom)
    elif in_geom.geom_type == 'GeometryCollection':
        for geom in sh_geom.GeometryCollection(in_geom):
            if geom.geom_type == 'MultiPolygon':
                geoms.append(list(geom))
            elif geom.geom_type == 'Polygon':
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
            collection_polys_gdf = gpd.GeoDataFrame(geometry=collection_polygons, crs=in_gdf.crs)

    # Only keep the polygons...
    ret_gdf = poly_gdf
    if len(multipoly_gdf) > 0:
        ret_gdf = ret_gdf.append(multipoly_gdf.explode(), ignore_index=True)
    if collection_polys_gdf is not None:
        ret_gdf = ret_gdf.append(collection_polys_gdf, ignore_index=True)
    
    return ret_gdf

def polygons_to_lines(input_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cardsheets_lines = []
    for cardsheet_poly in input_gdf.itertuples():
        cardsheet_boundary = cardsheet_poly.geometry.boundary
        if cardsheet_boundary.type == 'MultiLineString':
            for line in cardsheet_boundary:
                cardsheets_lines.append(line)
        else:
            cardsheets_lines.append(cardsheet_boundary)

    cardsheets_lines_gdf = gpd.GeoDataFrame(geometry=cardsheets_lines, crs=input_gdf.crs)

    return cardsheets_lines_gdf

def makevalid(geometry):
    
    # First check if the geom is None...
    if geometry is None:
        return None
    # If the geometry is valid, just return it
    if geometry.is_valid:
        return geometry

    # Else... try fixing it...
    geom_buf = geometry.buffer(0)
    if geom_buf.is_valid:
        return geom_buf
    else:
        logger.error(f"Error fixing geometry {geometry}")
        return geometry

def numberpoints(geometry: sh_geom.base.BaseGeometry) -> int:
    """
    Calculates the total number of points in a geometry.

    Args:
        geometry (sh_geom.base.BaseGeometry): the geometry to count the point of.

    Returns:
        int: the number of points in the geometry.
    """
    # If it is a multi-part, recursively call numberpoints for all parts. 
    if(isinstance(geometry, sh_geom.base.BaseMultipartGeometry)):
        nb_points = 0
        for geom in geometry:
            nb_points += numberpoints(geom)
        return nb_points
    elif(isinstance(geometry, sh_geom.polygon.Polygon)):
        # If it is a polygon, calculate number for exterior and interior rings. 
        nb_points = len(geometry.exterior.coords)
        for ring in geometry.interiors:
            nb_points += len(ring.coords)
        return nb_points
    else:
        # For other types, it is just the number of coordinates.
        return len(geometry.coords)

class SimplifyAlgorithm(enum.Enum):
    RAMER_DOUGLAS_PEUCKER = 'rdp'
    LANG = 'lang'
    VISVALINGAM_WHYATT = 'vw'

def simplify_ext(
        geometry: sh_geom.base.BaseGeometry,
        tolerance: float,
        algorithm: SimplifyAlgorithm = SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER,
        lookahead: int = 8,
        preserve_topology: bool = False,
        keep_points_on: sh_geom.base.BaseGeometry = None) -> Optional[sh_geom.base.BaseGeometry]:
    """
    Simplify the geometry, with extended options.

    Args:
        geometry (shapely geometry): the geometry to simplify
        tolerance (float): mandatory for the following algorithms:  
            * RAMER_DOUGLAS_PEUCKER: distance to use as tolerance  
            * LANG: distance to use as tolerance
            * VISVALINGAM_WHYATT: area to use as tolerance
        algorithm (SimplifyAlgorithm, optional): algorithm to use.
            Defaults to SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER.
        lookahead (int, optional): the number of points to consider for removing
            in a moving window. Used for LANG algorithm. Defaults to 8.
        preserve_topology (bool, optional): True to try to preserve topology. 
            Defaults to False.
        keep_points_on (BaseGeometry], optional): point of the geometry to 
            that intersect with these geometries are not removed. Defaults to None.

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]

    Returns:
        sh_geom.base.BaseGeometry: The simplified version of the geometry.
    """
    # Apply the simplification
    def simplify_polygon(polygon: sh_geom.Polygon) -> sh_geom.Polygon:
        # Simplify all rings
        exterior_simplified = simplify_coords(polygon.exterior.coords)
        interiors_simplified = []
        for interior in polygon.interiors:
            interior_simplified = simplify_coords(interior.coords)
            if interior_simplified is not None:
                interiors_simplified.append(interior_simplified)

        return sh_geom.Polygon(exterior_simplified, interiors_simplified)

    def simplify_coords(coords: list):

        # Determine the indexes of the coordinates to keep after simplification
        if algorithm is SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER:
            coords_simplify_idx = simpl.simplify_coords_idx(coords, tolerance)
        elif algorithm is SimplifyAlgorithm.VISVALINGAM_WHYATT:
            coords_simplify_idx = simpl.simplify_coords_vw_idx(coords, tolerance)
        elif algorithm is SimplifyAlgorithm.LANG:
            coords_simplify_idx = simplify_coords_lang_idx(coords, tolerance, lookahead=lookahead)    
        else:
            raise Exception(f"Unsupported algorithm: {algorithm}, supported: {SimplifyAlgorithm}")

        coords_gdf = gpd.GeoDataFrame(geometry=list(sh_geom.MultiPoint(coords)))
        coords_on_border_idx = []
        if keep_points_on is not None:
            coords_on_border_series = coords_gdf.intersects(keep_points_on)
            coords_on_border_idx = np.array(coords_on_border_series.index[coords_on_border_series]).tolist()

        # Extracts coordinates that need to be kept
        coords_to_keep = sorted(set(coords_simplify_idx + coords_on_border_idx))
        coords_simplified = np.array(coords)[coords_to_keep].tolist()

        # If simplified version has at least 3 points...
        if len(coords_simplified) >= 3:
            return coords_simplified
        else:
            if preserve_topology:
                return coords
            else:
                return None
                
    # Loop over the rings, and simplify them one by one...
    if geometry is None:
        return None
    elif isinstance(geometry, sh_geom.multipolygon.MultiPolygon):
        polygons_simplified = []
        for polygon in geometry:
            polygon_simplified = simplify_polygon(polygon)
            if polygon_simplified is not None:
                polygons_simplified.append(polygon_simplified)
        return_geom = sh_geom.MultiPolygon(polygons_simplified)
    elif isinstance(geometry, sh_geom.polygon.Polygon):
        return_geom = simplify_polygon(geometry)
    else:
        raise Exception(f"Unsupported geom_type: {geometry.geom_type}, {geometry}")

    return makevalid(return_geom)

def simplify_coords_lang(
        coords: Union[np.ndarray, list],
        tolerance: float,
        lookahead: int) -> Union[np.ndarray, list]:
    """
    Simplify a line using lang algorithm.

    Args:
        coords (Union[np.ndarray, list]): list of coordinates to be simplified.
        tolerance (float): distance tolerance to use.
        lookahead (int, optional): the number of points to consider for removing
            in a moving window. Defaults to 8.

    Returns:
        Return the coordinates kept after simplification. 
        If input coords is np.ndarray, returns np.ndarray, otherwise returns a list.
    """

    # Init variables 
    if isinstance(coords, np.ndarray):
        coords_arr = coords
    else:
        coords_arr = np.array(list(coords))

    # Determine the coordinates that need to be kept 
    coords_to_keep_idx = simplify_coords_lang(
            coords=coords_arr,
            tolerance=tolerance,
            lookahead=lookahead)
    coords_simplified_arr = coords_arr[coords_to_keep_idx]

    # If input was np.ndarray, return np.ndarray, otherwise list
    if isinstance(coords, np.ndarray):
        return coords_simplified_arr
    else:
        return coords_simplified_arr.tolist()

def simplify_coords_lang_idx(
        coords: Union[np.ndarray, list],
        tolerance: float,
        lookahead: int = 8) -> Union[np.ndarray, list]:
    """
    Simplify a line using lang algorithm and return the coordinate indexes to 
    be kept.

    Inspiration for the implementation came from:
        * https://github.com/giscan/Generalizer/blob/master/simplify.py
        * https://github.com/keszegrobert/polyline-simplification/blob/master/6.%20Lang.ipynb
        * https://web.archive.org/web/20171005193700/http://web.cs.sunyit.edu/~poissad/projects/Curve/about_algorithms/lang.php

    Args:
        coords (Union[np.ndarray, list]): list of coordinates to be simplified.
        tolerance (float): distance tolerance to use.
        lookahead (int, optional): the number of points to consider for removing
            in a moving window. Defaults to 8.

    Returns:
        Return the indexes of coordinates that need to be kept after 
        simplification. 
        If input coords is np.ndarray, returns np.ndarray, otherwise returns a list.  
    """
    
    def point_line_distance(x0, y0, x1, y1, x2, y2):
        return abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1)) / math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

    # Init variables 
    if isinstance(coords, np.ndarray):
        line_arr = coords
    else:
        line_arr = np.array(list(coords))

    # Prepare lookahead
    nb_points = len(line_arr)
    if lookahead == -1:
        window_size = nb_points
    else:
        window_size = min(lookahead, nb_points-1)

    #mask = np.ones(nb_points), dtype='bool')
    mask = np.zeros(nb_points, dtype='bool')
    mask[0] = True
    window_start = 0
    window_end = window_size
    
    # Apply simplification till the window_start arrives at the last point.
    ready = False
    while ready is False:
        # Check if all points between window_start and window_end are within  
        # tolerance distance to the line (window_start, window_end).
        all_points_in_tolerance = True
        for i in range(window_start+1, window_end):
            distance = point_line_distance(
                    line_arr[i, 0], line_arr[i, 1],
                    line_arr[window_start, 0], line_arr[window_start, 1],
                    line_arr[window_end, 0], line_arr[window_end, 1])
            if distance > tolerance:
                all_points_in_tolerance = False
                break

        # If not all the points are within the tolerance distance... 
        if not all_points_in_tolerance:
            # Move window_end to previous point, and try again
            window_end -= 1
        else:
            # All points are within the tolerance, so they can be masked
            mask[window_end] = True
            #mask[window_start+1:window_end-1] = False

            # Move the window forward
            window_start = window_end
            if window_start == nb_points - 1:
                ready = True
            window_end = window_start + window_size
            if window_end >= nb_points: 
                window_end = nb_points - 1
    
    # Prepare result: convert the mask to a list of indices of points to keep. 
    idx_to_keep_arr = mask.nonzero()[0]

    # If input was np.ndarray, return np.ndarray, otherwise list
    if isinstance(coords, np.ndarray):
        return idx_to_keep_arr
    else:
        return idx_to_keep_arr.tolist()

def remove_inner_rings(
        geometry,
        min_area_to_keep: float = None):
    
    # First check if the geom is None...
    if geometry is None:
        return None
    if geometry.type not in ('Polygon', 'Multipolgon'):
        raise Exception(f"remove_inner_rings is not possible with geometry.type: {geometry.type}, geometry: {geometry}")

    #if geometry.area > 91000 and geometry.area < 92000:
    #    logger.info("test")

    # If all inner rings need to be removed...
    if min_area_to_keep is None or min_area_to_keep == 0.0:
        # If there are no interior rings anyway, just return input
        if len(geometry.interiors) == 0:
            return geometry
        else:
            # Else create new polygon with only the exterior ring
            return sh_ops.Polygon(geometry.exterior)
    
    # If only small rings need to be removed... loop over them
    ring_coords_to_keep = []
    small_ring_found = False
    for ring in geometry.interiors:
        if abs(sh_ops.Polygon(ring).area) <= min_area_to_keep:
            small_ring_found = True
        else:
            ring_coords_to_keep.append(ring.coords)
    
    # If no small rings were found, just return input
    if small_ring_found == False:
        return geometry
    else:
        return sh_ops.Polygon(geometry.exterior.coords, 
                              ring_coords_to_keep)        
