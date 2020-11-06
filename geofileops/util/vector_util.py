# -*- coding: utf-8 -*-
"""
Module containing utilities regarding low level vector operations.
"""

import logging
import math
from typing import Tuple

import geopandas as gpd
import pyproj
import shapely.geometry as sh_geom
import shapely.ops as sh_ops

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def create_grid(
        total_bounds: Tuple[float, float, float, float],
        nb_columns: int,
        nb_rows: int,
        crs: pyproj.CRS) -> gpd.GeoDataFrame:

    xmin, ymin, xmax, ymax = total_bounds
    width = (xmax-xmin)/nb_columns
    height = (ymax-ymin)/nb_rows

    return create_grid3(total_bounds=total_bounds, width=width, height=height, crs=crs)

def create_grid3(
        total_bounds: Tuple[float, float, float, float],
        width: float,
        height: float,
        crs: pyproj.CRS) -> gpd.GeoDataFrame:

    xmin, ymin, xmax, ymax = total_bounds
    rows = int(math.ceil((ymax-ymin) / height))
    cols = int(math.ceil((xmax-xmin) / width))
     
    polygons = []
    cell_left = xmin
    cell_right = xmin + width
    for _ in range(cols+1):
        if cell_left > xmax:
            break
        cell_top = ymin + height
        cell_bottom = ymin
        for _ in range(rows+1):
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
        crs: pyproj.CRS) -> gpd.GeoDataFrame:
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
        while round(curr_nb_tiles_ratio_todo) > 1:

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

def extract_polygons_from_list(
        in_geom: sh_geom.base.BaseGeometry) -> list:
    """
    Extracts all polygons from the input geom and returns them as a list.
    """
    # Extract the polygons from the multipolygon, but store them as multipolygons anyway
    geoms = []
    if in_geom.geom_type == 'MultiPolygon':
        geoms = list(in_geom)
    elif in_geom.geom_type == 'Polygon':
        geoms.append(in_geom)
    elif in_geom.geom_type == 'GeometryCollection':
        for geom in in_geom:
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
