import logging
import math
from typing import Tuple

import geopandas as gpd
import pyproj
import shapely.geometry as sh_geom

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def create_grid2(
        total_bounds: Tuple[float, float, float, float], 
        nb_squarish_cells: int,
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
    nb_rows = round(math.sqrt(nb_squarish_cells/columns_vs_rows))

    # Evade having too many cells if few cells are asked...
    if nb_rows > nb_squarish_cells:
        nb_rows = nb_squarish_cells
    nb_columns = round(nb_squarish_cells/nb_rows)
    
    # Now we know everything to create the grid
    return create_grid(
        total_bounds=total_bounds,
        nb_columns=nb_columns,
        nb_rows=nb_rows,
        crs=crs)

def create_grid(
        total_bounds: Tuple[float, float, float, float],
        nb_columns: int,
        nb_rows: int,
        crs: pyproj.CRS) -> gpd.GeoDataFrame:

    xmin, ymin, xmax, ymax = total_bounds
    width = (xmax-xmin)/nb_columns
    height = (ymax-ymin)/nb_rows

    rows = int(math.ceil((ymax-ymin) / height))
    cols = int(math.ceil((xmax-xmin) / width))
    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymax
    YbottomOrigin = ymax- height
    polygons = []
    for _ in range(cols):
        Ytop = YtopOrigin
        Ybottom =YbottomOrigin
        for _ in range(rows):
            polygons.append(sh_geom.box(XleftOrigin, Ybottom, XrightOrigin, Ytop)) 
            Ytop = Ytop - height
            Ybottom = Ybottom - height
        XleftOrigin = XleftOrigin + width
        XrightOrigin = XrightOrigin + width     
    return gpd.GeoDataFrame({'geometry':polygons}, crs=crs)

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
