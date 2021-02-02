# -*- coding: utf-8 -*-
"""
Module containing utilities regarding low level vector operations.
"""

import logging
import math
from typing import Any, List, Tuple, Union

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

def simplify_ext(
        geometry,
        algorythm: str,
        tolerance: float = None,
        preserve_topology: bool = False,
        keep_points_on: sh_geom.base.BaseGeometry = None):
    """
    Simplify the geometry, with extended options.

    Args:
        geometry (shapely geometry): the geometry to simplify
        algorythm (str): algorythm to use.
        tolerance (float): mandatory for the following algorythms:  
            * "ramer–douglas–peucker": distance to use as tolerance  
            * "visvalingam-whyatt": area to use as tolerance
        preserve_topology (bool, optional): True to try to preserve topology. 
            Defaults to False.
        keep_points_on (BaseGeometry], optional): point of the geometry to 
            that intersect with these geometries are not removed. Defaults to None.

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]

    Returns:
        [type]: [description]
    """

    # Check parameters
    algorythms = ['ramer-douglas-peucker', 'visvalingam-whyatt']
    if algorythm == 'ramer-douglas-peucker':
        if tolerance is None:
            raise Exception(f"Tolerance parameter needs to be specified for algorythm {algorythm}!")
    elif algorythm == 'visvalingam-whyatt':
        if tolerance is None:
            raise Exception(f"Tolerance parameter needs to be specified for algorythm {algorythm}!")
    else:
        raise Exception(f"Unsupported algorythm: {algorythm}, supported: {algorythms}")

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
        if algorythm == 'ramer-douglas-peucker':
            coords_simplify_idx = simpl.simplify_coords_idx(coords, tolerance)
        elif algorythm == 'visvalingam-whyatt':
            coords_simplify_idx = simpl.simplify_coords_vw_idx(coords, tolerance)
        else:
            raise Exception(f"Unsupported algorythm: {algorythm}, supported: {algorythms}")

        coords_gdf = gpd.GeoDataFrame(geometry=list(sh_geom.MultiPoint(coords)))
        coords_on_border_idx = []
        if keep_points_on is not None:
            coords_on_border_series = coords_gdf.intersects(keep_points_on)
            coords_on_border_idx = coords_on_border_series.index[coords_on_border_series].tolist()

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
    elif geometry.geom_type == 'MultiPolygon':
        polygons_simplified = []
        for polygon in geometry:
            polygon_simplified = simplify_polygon(polygon)
            if polygon_simplified is not None:
                polygons_simplified.append(polygon_simplified)
        return_geom = sh_geom.MultiPolygon(polygons_simplified)
    elif geometry.geom_type == 'Polygon':
        return_geom = simplify_polygon(geometry)
    else:
        raise Exception(f"Unsupported geom_type: {geometry.geom_type}, {geometry}")

    return makevalid(return_geom)

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

def get_nb_coords(geometry) -> int:
    # First check if the geom is None...
    if geometry is None:
        return 0
    
    # Get the number of points for all rings
    nb_coords = len(geometry.exterior.coords)
    for ring in geometry.interiors:
        nb_coords += len(ring.coords)
    
    return nb_coords
